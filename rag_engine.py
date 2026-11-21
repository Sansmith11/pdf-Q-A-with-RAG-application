"""
rag_engine.py — Retrieval-Augmented Generation engine.

Responsibilities
----------------
1. Load a PineconeVectorStore for a given namespace (creating the index if needed).
2. Reformulate follow-up questions into stand-alone queries using chat
   history (Condense-Question approach).
3. Retrieve the top-k most relevant chunks and filter by cosine similarity
   threshold.
4. Run the retrieved chunks through a strict, grounded Gemini prompt that:
   - answers ONLY from the retrieved context,
   - cites [Filename, Page X] inline for every claim,
   - explicitly states when it cannot answer.
5. Stream the response token-by-token for snappy UX.
6. Return full retrieval metadata (source chunks + scores) for the UI.

Usage
-----
    from rag_engine import RAGEngine, ChatMessage

    engine = RAGEngine(namespace="my-session")
    history = []
    result = engine.query("What is the main topic?", chat_history=history)
    print(result.answer)
    history.append(ChatMessage(role="user",    content="What is the main topic?"))
    history.append(ChatMessage(role="assistant", content=result.answer))
"""

from __future__ import annotations

import logging
import textwrap
import time
from dataclasses import dataclass
from typing import Generator, List

from google import genai
from google.genai import types
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """Single turn in the conversation history."""
    role: str   # "user" or "assistant"
    content: str


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with its similarity score."""
    content: str
    source: str
    page: int
    score: float

    @property
    def citation(self) -> str:
        return f"[{self.source}, Page {self.page}]"


@dataclass
class RAGResult:
    """Full result returned by RAGEngine.query()."""
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    reformulated_query: str
    retrieval_latency_s: float
    generation_latency_s: float
    total_latency_s: float
    namespace: str


class SizedGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    """Embeddings wrapper that enforces configured vector dimensions."""

    output_dimensionality: int = cfg.embedding_dimensions
    api_key_value: str = cfg.google_api_key
    embedding_model_name: str = cfg.gemini_embedding_model

    def embed_query(self, text: str, **kwargs) -> list[float]:
        client = genai.Client(api_key=self.api_key_value)
        resp = client.models.embed_content(
            model=self.embedding_model_name,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return list(resp.embeddings[0].values)

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        client = genai.Client(api_key=self.api_key_value)
        resp = client.models.embed_content(
            model=self.embedding_model_name,
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=self.output_dimensionality,
            ),
        )
        return [list(emb.values) for emb in resp.embeddings]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CONDENSE_SYSTEM_PROMPT = (
    "You are a query reformulation assistant.\n"
    "Given the chat history and the user's latest question, rewrite it into a single, standalone search query to find the answer in the documents.\n"
    "If the question is already clear or standalone, or if history is irrelevant, return the question as-is.\n"
    "NEVER ask clarifying questions. DO NOT output conversational replies. Output ONLY the standalone query text."
)

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _CONDENSE_SYSTEM_PROMPT),
        ("human", "Chat History:\n{history}\n\nLatest question: {question}\nStandalone query:"),
    ]
)

_QA_SYSTEM_PROMPT = (
    "You are a helpful and accurate document assistant.\n"
    "Your task is to answer the user's question using the context excerpts provided in the user prompt.\n\n"
    "INSTRUCTIONS:\n"
    "1. Answer the question directly, thoroughly, and factually using the provided context excerpts.\n"
    "2. Whenever stating a fact, cite the source in the format [Filename, Page X].\n"
    "3. If the context does not contain sufficient information to answer the question, respond with:\n"
    '   "I cannot find this information in the uploaded documents."\n'
    "4. Do not invent facts that are not present in the excerpts."
)

_QA_USER_TEMPLATE = (
    "Context from uploaded documents:\n"
    "{context}\n\n"
    "Question: {question}"
)

_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _QA_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", _QA_USER_TEMPLATE),
    ]
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Retrieval-Augmented Generation engine backed by Pinecone + Gemini.

    Parameters
    ----------
    namespace:
        The Pinecone namespace to retrieve from (matches the ingestion
        namespace so each session is isolated).
    top_k:
        Number of candidate chunks to retrieve.  Defaults to cfg.rag_top_k.
    score_threshold:
        Minimum cosine similarity score to include a chunk.  Chunks below
        this threshold are silently dropped.
    temperature:
        LLM generation temperature (0 = deterministic, 1 = creative).
    """

    def __init__(
        self,
        namespace: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.namespace = namespace
        self.top_k = top_k if top_k is not None else cfg.rag_top_k
        self.score_threshold = (
            score_threshold if score_threshold is not None else cfg.rag_score_threshold
        )
        self.temperature = temperature

        self._embeddings = SizedGoogleEmbeddings(
            model=cfg.gemini_embedding_model,
            google_api_key=cfg.google_api_key,
            output_dimensionality=cfg.embedding_dimensions,
            api_key_value=cfg.google_api_key,
            embedding_model_name=cfg.gemini_embedding_model,
        )
        self._llm = ChatGoogleGenerativeAI(
            model=cfg.gemini_generation_model,
            google_api_key=cfg.google_api_key,
            temperature=self.temperature,
            streaming=True,
        )
        self._condense_llm = ChatGoogleGenerativeAI(
            model=cfg.gemini_generation_model,
            google_api_key=cfg.google_api_key,
            temperature=0.0,
            streaming=False,
        )
        self._pc = Pinecone(api_key=cfg.pinecone_api_key)
        self._ensure_index()

        self._vector_store = PineconeVectorStore(
            index=self._pc.Index(cfg.pinecone_index_name),
            embedding=self._embeddings,
            namespace=self.namespace,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        chat_history: list[ChatMessage] | None = None,
    ) -> RAGResult:
        """Answer a question using RAG."""
        if chat_history is None:
            chat_history = []

        t_start = time.perf_counter()

        reformulated_query = self._reformulate_query(question, chat_history)
        logger.info("Reformulated query: %s", reformulated_query)

        t_retrieve = time.perf_counter()
        chunks = self._retrieve(reformulated_query)
        retrieval_latency = time.perf_counter() - t_retrieve
        logger.info(
            "Retrieved %d chunks above threshold %.2f (latency=%.2fs)",
            len(chunks),
            self.score_threshold,
            retrieval_latency,
        )

        context = self._build_context_string(chunks)

        _FALLBACK = "I cannot find this information in the uploaded documents."
        if not chunks:
            answer = _FALLBACK
            generation_latency = 0.0
        else:
            t_generate = time.perf_counter()
            answer = self._generate(question, context, chat_history)
            generation_latency = time.perf_counter() - t_generate
            if not answer.strip():
                answer = _FALLBACK

        total_latency = time.perf_counter() - t_start
        return RAGResult(
            answer=answer,
            retrieved_chunks=chunks,
            reformulated_query=reformulated_query,
            retrieval_latency_s=retrieval_latency,
            generation_latency_s=generation_latency,
            total_latency_s=total_latency,
            namespace=self.namespace,
        )

    def stream_query(
        self,
        question: str,
        chat_history: list[ChatMessage] | None = None,
    ) -> Generator[str, None, RAGResult]:
        """Stream the answer token-by-token and return RAGResult on completion."""
        if chat_history is None:
            chat_history = []

        t_start = time.perf_counter()

        reformulated_query = self._reformulate_query(question, chat_history)

        t_retrieve = time.perf_counter()
        chunks = self._retrieve(reformulated_query)
        retrieval_latency = time.perf_counter() - t_retrieve

        context = self._build_context_string(chunks)
        lc_history = _to_langchain_messages(chat_history)

        # If no relevant chunks were found, skip LLM and return the fallback message directly.
        _FALLBACK = "I cannot find this information in the uploaded documents."
        if not chunks:
            yield _FALLBACK
            generation_latency = 0.0
            full_answer = _FALLBACK
            return RAGResult(
                answer=full_answer,
                retrieved_chunks=chunks,
                reformulated_query=reformulated_query,
                retrieval_latency_s=retrieval_latency,
                generation_latency_s=generation_latency,
                total_latency_s=time.perf_counter() - t_start,
                namespace=self.namespace,
            )

        messages = _QA_PROMPT.format_messages(
            context=context,
            history=lc_history,
            question=question,
        )

        t_generate = time.perf_counter()
        full_answer = ""
        for chunk in self._llm.stream(messages):
            token = _extract_text(chunk.content)
            full_answer += token
            yield token

        # Ensure a non-empty answer; fall back if LLM returned nothing
        if not full_answer.strip():
            full_answer = _FALLBACK
            yield full_answer

        generation_latency = time.perf_counter() - t_generate

        return RAGResult(
            answer=full_answer,
            retrieved_chunks=chunks,
            reformulated_query=reformulated_query,
            retrieval_latency_s=retrieval_latency,
            generation_latency_s=generation_latency,
            total_latency_s=time.perf_counter() - t_start,
            namespace=self.namespace,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Create the Pinecone index if it does not already exist."""
        existing = [idx.name for idx in self._pc.list_indexes()]
        if cfg.pinecone_index_name in existing:
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
        for _ in range(30):
            desc = self._pc.describe_index(cfg.pinecone_index_name)
            if desc.status.get("ready", False):
                break
            time.sleep(2)

    def _reformulate_query(
        self,
        question: str,
        chat_history: list[ChatMessage],
    ) -> str:
        """Reformulate follow-up questions into standalone queries when history exists."""
        if not chat_history:
            return question

        # Format history as clean plain text
        hist_text = "\n".join(
            f"{msg.role}: {msg.content}" for msg in chat_history[-6:]
        )
        messages = _CONDENSE_PROMPT.format_messages(
            history=hist_text,
            question=question,
        )
        try:
            response = self._condense_llm.invoke(messages)
            reformulated = _extract_text(response.content).strip()
            # If reformulation returned something nonsensical or blank, fallback to original question
            if not reformulated or len(reformulated) > 300 or "please provide" in reformulated.lower():
                return question
            return reformulated
        except Exception as exc:
            logger.warning("Query reformulation failed, using original: %s", exc)
            return question

    def _retrieve(self, query: str) -> list[RetrievedChunk]:
        """Perform similarity-with-score search and filter by threshold."""
        raw: list[tuple[Document, float]] = (
            self._vector_store.similarity_search_with_score(
                query,
                k=self.top_k,
            )
        )

        chunks: list[RetrievedChunk] = []
        for doc, score in raw:
            if score < self.score_threshold:
                logger.debug(
                    "Dropping chunk (score=%.3f < threshold=%.2f): %s…",
                    score,
                    self.score_threshold,
                    doc.page_content[:60],
                )
                continue
            chunks.append(
                RetrievedChunk(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    page=int(doc.metadata.get("page", 0)),
                    score=score,
                )
            )

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks

    @staticmethod
    def _build_context_string(chunks: list[RetrievedChunk]) -> str:
        """Render retrieved chunks into a formatted context block for the LLM."""
        if not chunks:
            return "No relevant context found in the uploaded documents."

        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[Excerpt {i} | {chunk.citation} | Score: {chunk.score:.3f}]\n"
                f"{chunk.content}"
            )
        return "\n\n---\n\n".join(parts)

    def _generate(
        self,
        question: str,
        context: str,
        chat_history: list[ChatMessage],
    ) -> str:
        """Generate a full (non-streaming) answer."""
        lc_history = _to_langchain_messages(chat_history)
        messages = _QA_PROMPT.format_messages(
            context=context,
            history=lc_history,
            question=question,
        )
        response = self._llm.invoke(messages)
        return _extract_text(response.content)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(content: object) -> str:
    """
    Safely convert an LLM response `.content` value to a plain string.

    Newer versions of langchain-google-genai may return content as:
      - str                      → return as-is
      - list of str              → join them
      - list of dicts (blocks)   → extract 'text' keys and join
    This prevents the 'can only concatenate str (not list) to str' and
    "'list' object has no attribute 'strip'" errors.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Content block format: {"type": "text", "text": "..."}
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    # Fallback for any other unexpected type
    return str(content)


def _to_langchain_messages(
    chat_history: list[ChatMessage],
) -> list[HumanMessage | AIMessage]:
    """Convert ChatMessage list into LangChain message objects."""
    messages = []
    for msg in chat_history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))
    return messages