"""
app.py — Streamlit web application for the PDF Q&A RAG system.

Run with:
    streamlit run app.py

Features
--------
* Sidebar: upload multiple PDFs, configure top_k & temperature, clear index.
* Main area: chat interface with st.chat_message.
* Streaming response from the RAG engine token-by-token.
* Expandable "Retrieved Sources" panel per assistant message.
* Session state management for chat history and engine lifecycle.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Optional

import streamlit as st

from config import cfg
from ingest import IngestionStats, PDFIngestionPipeline
from rag_engine import ChatMessage, RAGEngine, RAGResult, RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF Q&A — RAG System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Global typography ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f0c29, #302b63, #24243e);
        color: #e8e8ff;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: #c9c0ff; }
    [data-testid="stSidebar"] label { color: #c9c0ff !important; }

    /* ---- Header banner ---- */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 14px;
        padding: 24px 32px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(102,126,234,0.3);
    }
    .hero-banner h1 { color: #fff; margin: 0; font-size: 2rem; font-weight: 700; }
    .hero-banner p  { color: rgba(255,255,255,0.85); margin: 6px 0 0; font-size: 1rem; }

    /* ---- Chat bubbles ---- */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 4px;
    }

    /* ---- Source expander ---- */
    .source-card {
        background: #1e1e2e;
        border: 1px solid #3d3d5c;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.85rem;
        color: #cdd6f4;
    }
    .source-card .src-header {
        font-weight: 600;
        color: #89b4fa;
        margin-bottom: 6px;
    }
    .score-badge {
        display: inline-block;
        background: #313244;
        color: #a6e3a1;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.78rem;
        margin-left: 8px;
    }

    /* ---- Stat chips ---- */
    .stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 6px; }
    .stat-chip {
        background: #2a2a3e;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        color: #a6adc8;
    }

    /* ---- Buttons ---- */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.15s;
    }
    div.stButton > button:hover { transform: translateY(-1px); }

    /* ---- File uploader ---- */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #6366f1 !important;
        border-radius: 10px !important;
        background: rgba(99,102,241,0.05) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session-state bootstrap
# ---------------------------------------------------------------------------

def _init_session() -> None:
    """Initialise all session-state keys on first load."""
    if "namespace" not in st.session_state:
        st.session_state.namespace = f"session-{uuid.uuid4().hex[:8]}"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[ChatMessage] = []
    if "messages" not in st.session_state:
        # Each item: {"role": str, "content": str, "result": RAGResult|None}
        st.session_state.messages: list[dict] = []
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files: set[str] = set()
    if "engine" not in st.session_state:
        st.session_state.engine: Optional[RAGEngine] = None
    if "ingest_stats" not in st.session_state:
        st.session_state.ingest_stats: Optional[IngestionStats] = None


_init_session()


# ---------------------------------------------------------------------------
# Sidebar — file upload & controls
# ---------------------------------------------------------------------------

def _build_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📄 PDF Q&A RAG System")
        st.markdown("*Powered by Gemini + Pinecone*")
        st.divider()

        # --- Upload ---
        st.markdown("### 📂 Upload Documents")
        uploaded = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to index and query.",
        )

        # --- Retrieval settings ---
        st.markdown("### ⚙️ Retrieval Settings")
        top_k = st.slider(
            "Top-K chunks",
            min_value=1,
            max_value=15,
            value=cfg.rag_top_k,
            step=1,
            help="Number of text chunks retrieved per query.",
        )
        score_threshold = st.slider(
            "Min. similarity score",
            min_value=0.0,
            max_value=1.0,
            value=cfg.rag_score_threshold,
            step=0.05,
            format="%.2f",
            help="Chunks scoring below this cosine similarity are ignored.",
        )
        temperature = st.slider(
            "LLM temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            format="%.2f",
            help="0 = deterministic, 1 = creative.",
        )

        # Rebuild engine when settings change
        engine_key = (
            st.session_state.namespace,
            top_k,
            score_threshold,
            temperature,
        )
        if (
            st.session_state.engine is None
            or getattr(st.session_state, "_engine_key", None) != engine_key
        ):
            st.session_state.engine = RAGEngine(
                namespace=st.session_state.namespace,
                top_k=top_k,
                score_threshold=score_threshold,
                temperature=temperature,
            )
            st.session_state._engine_key = engine_key

        st.divider()

        # --- Ingest button ---
        if uploaded:
            new_files = [
                f for f in uploaded
                if _file_fingerprint(f) not in st.session_state.ingested_files
            ]
            if new_files:
                btn_label = f"🚀 Index {len(new_files)} new PDF(s)"
            else:
                btn_label = "✅ All files already indexed"

            if st.button(btn_label, use_container_width=True, disabled=not new_files):
                _run_ingestion(new_files)

        # --- Namespace info ---
        st.markdown("### 🗂 Session Namespace")
        st.code(st.session_state.namespace, language=None)

        # --- Ingestion stats ---
        if st.session_state.ingest_stats:
            s = st.session_state.ingest_stats
            st.markdown("**Last Ingestion:**")
            st.markdown(
                f"- Pages: `{s.pages_extracted}` &nbsp;|&nbsp; "
                f"Chunks: `{s.chunks_created}`\n"
                f"- Upserted: `{s.chunks_upserted}` &nbsp;|&nbsp; "
                f"Duplicates: `{s.chunks_skipped_duplicate}`\n"
                f"- Time: `{s.elapsed_seconds:.1f}s`"
            )
            if s.errors:
                for err in s.errors:
                    st.error(err)

        # --- Clear session ---
        st.divider()
        st.markdown("### 🗑 Reset Session")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("Clear Index", use_container_width=True, type="primary"):
                _clear_index()


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

def _file_fingerprint(file) -> str:
    """SHA-256 of filename + size, used to track already-indexed files."""
    return hashlib.sha256(
        f"{file.name}:{file.size}".encode()
    ).hexdigest()[:12]


def _run_ingestion(files: list) -> None:
    with st.sidebar:
        with st.spinner("📥 Indexing PDFs… this may take a moment."):
            try:
                pipeline = PDFIngestionPipeline()
                stats = pipeline.ingest_files(
                    files=files,
                    namespace=st.session_state.namespace,
                    clear_namespace_first=False,
                )
                st.session_state.ingest_stats = stats
                for f in files:
                    st.session_state.ingested_files.add(_file_fingerprint(f))
                st.success(f"✅ Indexed {len(files)} file(s)!")
            except Exception as exc:
                st.error(f"❌ Ingestion failed: {exc}")
                logger.exception("Ingestion error")


def _clear_index() -> None:
    with st.sidebar:
        with st.spinner("Clearing Pinecone namespace…"):
            try:
                pipeline = PDFIngestionPipeline()
                pipeline.delete_namespace(st.session_state.namespace)
                st.session_state.ingested_files = set()
                st.session_state.ingest_stats = None
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.success("Index cleared.")
            except Exception as exc:
                st.error(f"Failed to clear index: {exc}")
    st.rerun()


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------

def _render_hero() -> None:
    st.markdown(
        """
        <div class="hero-banner">
          <h1>📄 PDF Question Answering System</h1>
          <p>Upload your documents in the sidebar, then ask anything about them.
             Answers are grounded exclusively in your uploaded PDFs with page-level citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sources_expander(result: RAGResult) -> None:
    """Show retrieved chunks and latency metrics in a collapsible section."""
    with st.expander(
        f"📚 Retrieved Sources & Confidence Scores ({len(result.retrieved_chunks)} chunks)",
        expanded=False,
    ):
        if not result.retrieved_chunks:
            st.info("No chunks met the similarity threshold for this query.")
            return

        # Latency summary
        st.markdown(
            f"""
            <div class="stat-row">
              <span class="stat-chip">🔍 Retrieval: {result.retrieval_latency_s:.2f}s</span>
              <span class="stat-chip">🤖 Generation: {result.generation_latency_s:.2f}s</span>
              <span class="stat-chip">⏱ Total: {result.total_latency_s:.2f}s</span>
              <span class="stat-chip">🔄 Reformulated: {result.reformulated_query[:60]}…</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        for i, chunk in enumerate(result.retrieved_chunks, start=1):
            score_pct = chunk.score * 100
            # Colour-code by score
            if score_pct >= 80:
                bar_colour = "#a6e3a1"  # green
            elif score_pct >= 60:
                bar_colour = "#f9e2af"  # yellow
            else:
                bar_colour = "#f38ba8"  # red

            st.markdown(
                f"""
                <div class="source-card">
                  <div class="src-header">
                    Excerpt {i} &nbsp;·&nbsp; {chunk.citation}
                    <span class="score-badge" style="color:{bar_colour}">
                      {score_pct:.1f}%
                    </span>
                  </div>
                  <div>{chunk.content[:500]}{"…" if len(chunk.content) > 500 else ""}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_chat_history() -> None:
    """Replay all stored messages so the conversation is visible on re-run."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("result"):
                _render_sources_expander(msg["result"])


def _handle_user_input() -> None:
    """Read from the chat input widget and trigger the RAG pipeline."""
    if not st.session_state.ingested_files:
        st.info(
            "👈 Upload at least one PDF in the sidebar and click **Index** to get started.",
            icon="ℹ️",
        )
        return

    if prompt := st.chat_input("Ask a question about your documents…"):
        # Append user message to history
        st.session_state.messages.append({"role": "user", "content": prompt, "result": None})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- RAG generation with streaming ---
        engine: RAGEngine = st.session_state.engine
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_text = ""
            final_result: Optional[RAGResult] = None

            try:
                stream_gen = engine.stream_query(
                    question=prompt,
                    chat_history=st.session_state.chat_history,
                )
                for token in stream_gen:
                    full_text += token
                    placeholder.markdown(full_text + "▌")

                # The generator's return value holds the RAGResult; extract it
                # by exhausting the generator one more time via StopIteration.
                try:
                    next(stream_gen)
                except StopIteration as e:
                    final_result = e.value

                # If the LLM returned nothing at all, show the fallback message
                if not full_text.strip():
                    full_text = "I cannot find this information in the uploaded documents."
                    if final_result:
                        final_result.answer = full_text

                placeholder.markdown(full_text)

            except Exception as exc:
                full_text = f"⚠️ Error generating response: {exc}"
                placeholder.markdown(full_text)
                logger.exception("RAG query error")

            if final_result:
                _render_sources_expander(final_result)

        # Save to session state
        st.session_state.messages.append(
            {"role": "assistant", "content": full_text, "result": final_result}
        )
        # Update LangChain-style chat history for next turn's query reformulation
        st.session_state.chat_history.append(
            ChatMessage(role="user", content=prompt)
        )
        st.session_state.chat_history.append(
            ChatMessage(role="assistant", content=full_text)
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _build_sidebar()
    _render_hero()
    _render_chat_history()
    _handle_user_input()


if __name__ == "__main__":
    main()
