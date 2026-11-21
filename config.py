"""
config.py — Centralised configuration loader for the PDF Q&A RAG system.

All modules import from here so environment values are validated once and
type-annotated constants flow through the rest of the codebase.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap – load .env before anything else reads os.environ
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _require(key: str) -> str:
    """Return env var value or raise a clear RuntimeError."""
    value = os.getenv(key, "").strip()
    if not value:
        raise RuntimeError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in your credentials."
        )
    return value


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    # --- API Keys ---
    google_api_key: str = field(default_factory=lambda: _require("GOOGLE_API_KEY"))
    pinecone_api_key: str = field(default_factory=lambda: _require("PINECONE_API_KEY"))

    # --- Pinecone ---
    pinecone_index_name: str = field(
        default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "pdf-qa-rag")
    )

    # --- Gemini Models ---
    gemini_generation_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite")
    )
    gemini_embedding_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    )
    embedding_dimensions: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    )

    # --- RAG Retrieval ---
    rag_top_k: int = field(
        default_factory=lambda: int(os.getenv("RAG_TOP_K", "5"))
    )
    rag_score_threshold: float = field(
        default_factory=lambda: float(os.getenv("RAG_SCORE_THRESHOLD", "0.45"))
    )

    # --- Chunking ---
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    )

    # --- Pinecone Serverless spec ---
    pinecone_cloud: str = field(
        default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws")
    )
    pinecone_region: str = field(
        default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1")
    )


# Singleton — import `cfg` everywhere instead of instantiating repeatedly.
cfg = AppConfig()

logger.info(
    "Config loaded | index=%s | gen_model=%s | embed_model=%s | dims=%d",
    cfg.pinecone_index_name,
    cfg.gemini_generation_model,
    cfg.gemini_embedding_model,
    cfg.embedding_dimensions,
)