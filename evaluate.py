"""
evaluate.py — Standalone evaluation script for the PDF Q&A RAG system.

What it measures
----------------
* Retrieval latency (s)
* Generation latency (s)
* Total round-trip latency (s)
* Number of chunks retrieved
* Source citation accuracy — did the answer cite at least one of the
  expected source files?
* Answer coverage — does the answer contain the expected keywords?

Usage
-----
1. Make sure the namespace already has indexed PDFs (run ingest.py first).
2. Edit EVAL_CASES below (or pass --cases-file <json-path>).
3. Run:
       python evaluate.py --namespace my-session

The script writes results to `evaluation_results.csv` and prints a
rich summary table to stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from tabulate import tabulate

from config import cfg
from rag_engine import ChatMessage, RAGEngine, RAGResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluation test cases
# ---------------------------------------------------------------------------
# Each case is a dict with:
#   question          : str   — the question to ask
#   expected_sources  : list  — list of filename substrings that SHOULD appear
#                                in citations (ANY match counts as correct)
#   expected_keywords : list  — keywords that should appear in the answer
#                                (case-insensitive, ANY match counts)
#   note              : str   — human-readable description (optional)
#
# Replace/extend these with real Q&A pairs from your indexed documents.
# ---------------------------------------------------------------------------

EVAL_CASES: list[dict] = [
    {
        "question": "What is the main topic discussed in the uploaded documents?",
        "expected_sources": [],          # accept any source
        "expected_keywords": [],         # accept any answer
        "note": "Open-ended summary test",
    },
    {
        "question": "Summarise the key findings or conclusions.",
        "expected_sources": [],
        "expected_keywords": ["finding", "conclusion", "result", "summary"],
        "note": "Keyword coverage test",
    },
    {
        "question": "What methodology or approach is described?",
        "expected_sources": [],
        "expected_keywords": ["method", "approach", "technique", "process"],
        "note": "Methodology keyword test",
    },
    {
        "question": "Are there any references or citations in the document?",
        "expected_sources": [],
        "expected_keywords": [],
        "note": "Reference detection test",
    },
    {
        "question": "xyzzy_does_not_exist_in_any_document_12345",
        "expected_sources": [],
        "expected_keywords": ["cannot find", "not find", "unable"],
        "note": "Hallucination guard — should return 'cannot find' response",
    },
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    case_index: int
    question: str
    note: str
    reformulated_query: str
    answer: str
    retrieved_chunks: int
    retrieval_latency_s: float
    generation_latency_s: float
    total_latency_s: float
    source_match: Optional[bool]   # None if no expected_sources specified
    keyword_match: Optional[bool]  # None if no expected_keywords specified
    cited_sources: str             # comma-separated source citations found
    error: str = ""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class PipelineEvaluator:
    """Runs evaluation cases against the RAG pipeline and collects metrics."""

    def __init__(self, namespace: str, top_k: int = cfg.rag_top_k) -> None:
        self.namespace = namespace
        self.engine = RAGEngine(
            namespace=namespace,
            top_k=top_k,
            score_threshold=cfg.rag_score_threshold,
            temperature=0.0,  # deterministic for reproducibility
        )

    def run(
        self,
        cases: list[dict],
        output_csv: str = "evaluation_results.csv",
    ) -> list[EvalResult]:
        results: list[EvalResult] = []

        for idx, case in enumerate(cases, start=1):
            question = case.get("question", "")
            expected_sources: list[str] = case.get("expected_sources", [])
            expected_keywords: list[str] = case.get("expected_keywords", [])
            note = case.get("note", "")

            logger.info("[Case %d/%d] %s", idx, len(cases), question[:80])
            print(f"\n{'─'*60}")
            print(f"Case {idx}: {note}")
            print(f"Q: {question}")

            result = self._evaluate_single(
                idx, question, note, expected_sources, expected_keywords
            )
            results.append(result)

            # Pretty-print per-case summary
            self._print_case_summary(result)

        # Write CSV
        self._write_csv(results, output_csv)

        # Print aggregate table
        self._print_aggregate(results)
        return results

    # ------------------------------------------------------------------

    def _evaluate_single(
        self,
        idx: int,
        question: str,
        note: str,
        expected_sources: list[str],
        expected_keywords: list[str],
    ) -> EvalResult:
        try:
            rag: RAGResult = self.engine.query(question, chat_history=[])
        except Exception as exc:  # noqa: BLE001
            logger.error("Error on case %d: %s", idx, exc)
            return EvalResult(
                case_index=idx,
                question=question,
                note=note,
                reformulated_query=question,
                answer="",
                retrieved_chunks=0,
                retrieval_latency_s=0.0,
                generation_latency_s=0.0,
                total_latency_s=0.0,
                source_match=False,
                keyword_match=False,
                cited_sources="",
                error=str(exc),
            )

        answer_lower = rag.answer.lower()
        cited = [c.citation for c in rag.retrieved_chunks]
        cited_str = "; ".join(cited)

        # Source match
        source_match: Optional[bool] = None
        if expected_sources:
            source_match = any(
                any(exp.lower() in c.source.lower() for c in rag.retrieved_chunks)
                for exp in expected_sources
            )

        # Keyword match
        keyword_match: Optional[bool] = None
        if expected_keywords:
            keyword_match = any(kw.lower() in answer_lower for kw in expected_keywords)

        return EvalResult(
            case_index=idx,
            question=question,
            note=note,
            reformulated_query=rag.reformulated_query,
            answer=rag.answer[:500],  # truncate for CSV readability
            retrieved_chunks=len(rag.retrieved_chunks),
            retrieval_latency_s=round(rag.retrieval_latency_s, 3),
            generation_latency_s=round(rag.generation_latency_s, 3),
            total_latency_s=round(rag.total_latency_s, 3),
            source_match=source_match,
            keyword_match=keyword_match,
            cited_sources=cited_str,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _print_case_summary(r: EvalResult) -> None:
        status_sym = {True: "✅", False: "❌", None: "—"}
        print(f"  Reformulated: {r.reformulated_query[:70]}")
        print(f"  Chunks retrieved: {r.retrieved_chunks}")
        print(
            f"  Latency → retrieval={r.retrieval_latency_s:.2f}s "
            f"generation={r.generation_latency_s:.2f}s "
            f"total={r.total_latency_s:.2f}s"
        )
        print(f"  Source match:  {status_sym[r.source_match]}")
        print(f"  Keyword match: {status_sym[r.keyword_match]}")
        if r.error:
            print(f"  ERROR: {r.error}")
        if r.cited_sources:
            print(f"  Citations: {r.cited_sources[:120]}")

    @staticmethod
    def _write_csv(results: list[EvalResult], path: str) -> None:
        df = pd.DataFrame([asdict(r) for r in results])
        df.to_csv(path, index=False)
        print(f"\n📊 Results saved to: {path}")

    @staticmethod
    def _print_aggregate(results: list[EvalResult]) -> None:
        total = len(results)
        avg_retrieve = sum(r.retrieval_latency_s for r in results) / total
        avg_generate = sum(r.generation_latency_s for r in results) / total
        avg_total = sum(r.total_latency_s for r in results) / total
        avg_chunks = sum(r.retrieved_chunks for r in results) / total

        scored_source = [r for r in results if r.source_match is not None]
        scored_keyword = [r for r in results if r.keyword_match is not None]

        source_acc = (
            sum(1 for r in scored_source if r.source_match) / len(scored_source)
            if scored_source
            else None
        )
        keyword_acc = (
            sum(1 for r in scored_keyword if r.keyword_match) / len(scored_keyword)
            if scored_keyword
            else None
        )

        summary = [
            ["Total cases", total],
            ["Avg retrieval latency (s)", f"{avg_retrieve:.3f}"],
            ["Avg generation latency (s)", f"{avg_generate:.3f}"],
            ["Avg total latency (s)", f"{avg_total:.3f}"],
            ["Avg chunks retrieved", f"{avg_chunks:.1f}"],
            [
                "Source citation accuracy",
                f"{source_acc*100:.1f}%" if source_acc is not None else "N/A",
            ],
            [
                "Keyword coverage accuracy",
                f"{keyword_acc*100:.1f}%" if keyword_acc is not None else "N/A",
            ],
        ]

        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(tabulate(summary, headers=["Metric", "Value"], tablefmt="rounded_outline"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_cases_from_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a top-level list of case dicts.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the PDF Q&A RAG pipeline."
    )
    parser.add_argument(
        "--namespace",
        required=True,
        help="Pinecone namespace that was used during ingestion.",
    )
    parser.add_argument(
        "--cases-file",
        default=None,
        metavar="JSON",
        help=(
            "Path to a JSON file with a list of evaluation case dicts. "
            "If omitted, the built-in EVAL_CASES are used."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=cfg.rag_top_k,
        help=f"Number of chunks to retrieve (default: {cfg.rag_top_k}).",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.csv",
        help="Output CSV file path (default: evaluation_results.csv).",
    )
    args = parser.parse_args()

    cases = EVAL_CASES
    if args.cases_file:
        cases = _load_cases_from_json(args.cases_file)
        logger.info("Loaded %d cases from '%s'", len(cases), args.cases_file)

    evaluator = PipelineEvaluator(namespace=args.namespace, top_k=args.top_k)
    evaluator.run(cases, output_csv=args.output)


if __name__ == "__main__":
    main()
