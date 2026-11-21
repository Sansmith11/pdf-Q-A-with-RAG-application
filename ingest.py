"""
ingest.py — PDF ingestion and Pinecone indexing pipeline.

Responsibilities
----------------
1. Parse PDFs with PyMuPDF, preserving per-page text + metadata.
2. Split text into overlapping chunks with RecursiveCharacterTextSplitter.
3. Deduplicate chunks by SHA-256 hash to avoid re-indexing unchanged content.
4. Embed chunks with the Gemini embedding model at configured dimensionality.
5. Upsert vectors into a Pinecone Serverless index under a user-supplied
   namespace so each session / document set remains isolated.

Usage (CLI)
-----------
    python ingest.py --files doc1.pdf doc2.pdf --namespace my-session

Usage (programmatic — called by app.py)
----------------------------------------
    from ingest import PDFIngestionPipeline
    pipeline = PDFIngestionPipeline()
    stats = pipeline.ingest_files(uploaded_files, namespace="session-abc")
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Sequence, Union

import fitz  # PyMuPDF
from google import genai
from google.genai import types
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IngestionStats:
    """Summary of a completed ingestion run."""
    files_processed: int = 0
    pages_extracted: int = 0
    chunks_created: int = 0
    chunks_upserted: int = 0
    chunks_skipped_duplicate: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"Files: {self.files_processed} | "
            f"Pages: {self.pages_extracted} | "
            f"Chunks: {self.chunks_created} | "
            f"Upserted: {self.chunks_upserted} | "
            f"Duplicates skipped: {self.chunks_skipped_duplicate} | "
            f"Time: {self.elapsed_seconds:.1f}s"
        )


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class PDFIngestionPipeline:
    """
    End-to-end pipeline: PDF bytes → Pinecone vectors.

    Parameters
    ----------
    batch_size:
        Number of Document chunks to embed and upsert in a single Pinecone
        call.  Lower values reduce memory pressure on large PDFs.
    """

    UPSERTED_HASHES_META_KEY = "chunk_hash"

    def __init__(self, batch_size: int = 100) -> None:
        self.batch_size = batch_size
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=cfg.gemini_embedding_model,
            google_api_key=cfg.google_api_key,
        )
        self._genai_client = genai.Client(api_key=cfg.google_api_key)
        self._pc = Pinecone(api_key=cfg.pinecone_api_key)
        self._ensure_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_files(
        self,
        files: Sequence[Union[str, Path, IO[bytes]]],
        namespace: str,
        clear_namespace_first: bool = False,
    ) -> IngestionStats:
        """
        Ingest a sequence of PDF files into the given Pinecone namespace.

        Parameters
        ----------
        files:
            Iterable of file-like objects, str paths, or Path objects.
        namespace:
            Pinecone namespace to upsert into.  Isolates documents
            per-session.
        clear_namespace_first:
            When True, delete all existing vectors in the namespace before
            upserting — useful for a "re-index" workflow.

        Returns
        -------
        IngestionStats
            Summary of the run including counts and elapsed time.
        """
        t0 = time.perf_counter()
        stats = IngestionStats()

        if clear_namespace_first:
            logger.info("Clearing namespace '%s' before ingestion…", namespace)
            self._delete_namespace(namespace)

        # Collect existing hashes to skip duplicates
        existing_hashes: set[str] = self._fetch_existing_hashes(namespace)

        all_docs: list[Document] = []

        for file_ref in files:
            try:
                docs = self._process_single_file(file_ref, stats)
                all_docs.extend(docs)
            except Exception as exc:  # noqa: BLE001
                name = getattr(file_ref, "name", str(file_ref))
                msg = f"Failed to process '{name}': {exc}"
                logger.error(msg)
                stats.errors.append(msg)

        stats.chunks_created = len(all_docs)

        # Deduplicate
        new_docs = self._filter_duplicates(all_docs, existing_hashes, stats)

        # Embed + upsert in batches
        self._upsert_batched(new_docs, namespace, stats)

        stats.elapsed_seconds = time.perf_counter() - t0
        logger.info("Ingestion complete: %s", stats)
        return stats

    def delete_namespace(self, namespace: str) -> None:
        """Remove all vectors under a namespace from the Pinecone index."""
        self._delete_namespace(namespace)

    # ------------------------------------------------------------------
    # Private helpers — PDF parsing
    # ------------------------------------------------------------------

    def _process_single_file(
        self,
        file_ref: Union[str, Path, IO[bytes]],
        stats: IngestionStats,
    ) -> list[Document]:
        """Return a list of chunked Documents for one PDF file."""
        pdf_bytes, source_name = self._read_file(file_ref)
        raw_docs = self._extract_pages(pdf_bytes, source_name, stats)
        chunks = self._splitter.split_documents(raw_docs)

        # Attach a deterministic hash per chunk for deduplication
        for chunk in chunks:
            chunk.metadata[self.UPSERTED_HASHES_META_KEY] = _sha256(
                chunk.page_content
            )

        logger.info(
            "  '%s' → %d pages → %d chunks",
            source_name,
            len(raw_docs),
            len(chunks),
        )
        stats.files_processed += 1
        return chunks

    @staticmethod
    def _read_file(
        file_ref: Union[str, Path, IO[bytes]]
    ) -> tuple[bytes, str]:
        """Return (raw_bytes, source_name) from various input types."""
        if isinstance(file_ref, (str, Path)):
            path = Path(file_ref)
            return path.read_bytes(), path.name
        # File-like object (e.g., Streamlit UploadedFile)
        name = getattr(file_ref, "name", "unknown.pdf")
        raw = file_ref.read() if hasattr(file_ref, "read") else bytes(file_ref)
        # Rewind if possible so callers can re-read the file later
        if hasattr(file_ref, "seek"):
            file_ref.seek(0)
        return raw, name

    @staticmethod
    def _extract_pages(
        pdf_bytes: bytes,
        source_name: str,
        stats: IngestionStats,
    ) -> list[Document]:
        """
        Use PyMuPDF to extract text from every page.

        Returns one Document per page with metadata:
            source: filename
            page:   1-based page number
        """
        docs: list[Document] = []
        with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as pdf:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                text = page.get_text("text").strip()
                if not text:
                    # Skip blank / image-only pages
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source_name,
                            "page": page_index + 1,  # 1-based
                        },
                    )
                )
        stats.pages_extracted += len(docs)
        return docs

    # ------------------------------------------------------------------
    # Private helpers — deduplication
    # ------------------------------------------------------------------

    def _fetch_existing_hashes(self, namespace: str) -> set[str]:
        """
        Retrieve chunk hashes already stored in the namespace by listing
        existing vector IDs.
        """
        try:
            index = self._pc.Index(cfg.pinecone_index_name)
            stats_resp = index.describe_index_stats()
            ns_stats = stats_resp.get("namespaces", {}).get(namespace, {})
            if ns_stats.get("vector_count", 0) == 0:
                return set()
            result = index.list(namespace=namespace, limit=10_000)
            existing: set[str] = set()
            for vec_id in result.get("vectors", []):
                parts = vec_id.split("_")
                if len(parts) >= 2:
                    existing.add(parts[-2])  # hash segment
            logger.info(
                "Found %d existing chunk hashes in namespace '%s'",
                len(existing),
                namespace,
            )
            return existing
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not fetch existing hashes (non-fatal): %s", exc
            )
            return set()

    @staticmethod
    def _filter_duplicates(
        docs: list[Document],
        existing_hashes: set[str],
        stats: IngestionStats,
    ) -> list[Document]:
        new_docs: list[Document] = []
        for doc in docs:
            h = doc.metadata.get("chunk_hash", "")
            if h and h in existing_hashes:
                stats.chunks_skipped_duplicate += 1
            else:
                new_docs.append(doc)
                if h:
                    existing_hashes.add(h)
        logger.info(
            "After deduplication: %d new / %d duplicate chunks",
            len(new_docs),
            stats.chunks_skipped_duplicate,
        )
        return new_docs

    # ------------------------------------------------------------------
    # Private helpers — Pinecone upsert
    # ------------------------------------------------------------------

    def _upsert_batched(
        self,
        docs: list[Document],
        namespace: str,
        stats: IngestionStats,
    ) -> None:
        """Embed with explicit output_dimensionality and upsert in batches."""
        if not docs:
            logger.info("No new chunks to upsert.")
            return

        index = self._pc.Index(cfg.pinecone_index_name)

        for batch_start in range(0, len(docs), self.batch_size):
            batch = docs[batch_start : batch_start + self.batch_size]
            try:
                texts = [d.page_content for d in batch]
                embed_resp = self._genai_client.models.embed_content(
                    model=cfg.gemini_embedding_model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        output_dimensionality=cfg.embedding_dimensions,
                    ),
                )
                vectors = [list(emb.values) for emb in embed_resp.embeddings]

                ids = [
                    f"{Path(d.metadata.get('source', 'doc')).stem}_"
                    f"{d.metadata.get('chunk_hash', _sha256(d.page_content))}_"
                    f"{batch_start + idx}"
                    for idx, d in enumerate(batch)
                ]
                metadatas = [
                    {**d.metadata, "text": d.page_content} for d in batch
                ]
                index.upsert(
                    vectors=[
                        {
                            "id": vec_id,
                            "values": vec,
                            "metadata": meta,
                        }
                        for vec_id, vec, meta in zip(ids, vectors, metadatas)
                    ],
                    namespace=namespace,
                )
                stats.chunks_upserted += len(batch)
                logger.info(
                    "  Upserted batch %d–%d (%d chunks, namespace='%s')",
                    batch_start,
                    batch_start + len(batch) - 1,
                    len(batch),
                    namespace,
                )
            except Exception as exc:  # noqa: BLE001
                msg = (
                    f"Upsert batch {batch_start}–{batch_start+len(batch)-1} "
                    f"failed: {exc}"
                )
                logger.error(msg)
                stats.errors.append(msg)
                raise

    # ------------------------------------------------------------------
    # Private helpers — Pinecone index management
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Create the Pinecone index if it does not already exist."""
        existing = [idx.name for idx in self._pc.list_indexes()]
        if cfg.pinecone_index_name in existing:
            logger.info("Pinecone index '%s' already exists.", cfg.pinecone_index_name)
            return

        logger.info(
            "Creating Pinecone index '%s' (dims=%d, metric=cosine)…",
            cfg.pinecone_index_name,
            cfg.embedding_dimensions,
        )
        self._pc.create_index(
            name=cfg.pinecone_index_name,
            dimension=cfg.embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=cfg.pinecone_cloud,
                region=cfg.pinecone_region,
            ),
        )
        # Wait until the index is ready
        for _ in range(30):
            desc = self._pc.describe_index(cfg.pinecone_index_name)
            if desc.status.get("ready", False):
                break
            time.sleep(2)
        logger.info("Pinecone index '%s' is ready.", cfg.pinecone_index_name)

    def _delete_namespace(self, namespace: str) -> None:
        try:
            index = self._pc.Index(cfg.pinecone_index_name)
            index.delete(delete_all=True, namespace=namespace)
            logger.info("Deleted all vectors in namespace '%s'.", namespace)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not delete namespace '%s': %s", namespace, exc)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest PDF files into the Pinecone RAG index."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        metavar="PDF",
        help="One or more PDF file paths to ingest.",
    )
    parser.add_argument(
        "--namespace",
        required=True,
        help="Pinecone namespace (e.g. a session ID or project name).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the namespace before ingesting (full re-index).",
    )
    args = parser.parse_args()

    pipeline = PDFIngestionPipeline()
    result = pipeline.ingest_files(
        files=args.files,
        namespace=args.namespace,
        clear_namespace_first=args.clear,
    )
    print(result)