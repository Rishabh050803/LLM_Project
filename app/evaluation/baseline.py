"""Run a small, deterministic baseline evaluation locally."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from app.config import Settings
from app.rag.service import AcademicRAGService
from app.retrieval.schemas import QueryRequest

logger = logging.getLogger(__name__)


def _source_text(result: dict) -> str:
    return " ".join(
        str(result.get(field, ""))
        for field in ("filename", "source_url", "title", "text")
    ).lower()


def _has_abstained(answer: str) -> bool:
    lowered = answer.lower()
    markers = (
        "could not find",
        "not found",
        "not provide",
        "not available",
        "do not have",
        "does not contain",
        "does not include",
        "do not include",
        "does not specify",
        "do not cover",
        "cannot answer",
        "cannot provide",
        "no information",
    )
    return any(marker in lowered for marker in markers)


def run_baseline(settings: Settings, dataset_path: Path, output_path: Path,
                 top_k: int = 3) -> dict:
    cases = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    service = AcademicRAGService(settings)
    results: list[dict] = []
    for case in cases:
        started = time.monotonic()
        query = QueryRequest(question=case["question"], top_k=top_k)
        response = service.query(query)
        elapsed_ms = round((time.monotonic() - started) * 1000, 1)
        sources = [source.model_dump() for source in response.sources]
        source_blob = " ".join(_source_text(source) for source in sources)
        expected_terms = [term.lower() for term in case.get("expected_source_terms", [])]
        retrieval_hit = (
            not case["answerable"] and not expected_terms
        ) or any(term in source_blob for term in expected_terms)
        citation_numbers = {int(number) for number in re.findall(r"\[(\d+)\]", response.answer)}
        citation_valid = (
            not case["answerable"]
            or (bool(citation_numbers) and citation_numbers.issubset(
                set(range(1, len(sources) + 1))
            ))
        )
        abstained = _has_abstained(response.answer)
        results.append({
            **case,
            "answer": response.answer,
            "sources": sources,
            "latency_ms": elapsed_ms,
            "retrieval_hit": retrieval_hit,
            "citation_valid": citation_valid,
            "abstained": abstained,
            "abstention_correct": abstained if not case["answerable"] else not abstained,
        })
        logger.info("Baseline case complete: id=%s retrieval_hit=%s abstained=%s latency_ms=%.1f",
                    case["id"], retrieval_hit, abstained, elapsed_ms)

    answerable = [item for item in results if item["answerable"]]
    unanswerable = [item for item in results if not item["answerable"]]
    summary = {
        "dataset": str(dataset_path),
        "cases": len(results),
        "top_k": top_k,
        "retrieval_recall_at_k": round(
            sum(item["retrieval_hit"] for item in answerable) / len(answerable), 3
        ) if answerable else 0.0,
        "citation_format_valid_rate": round(
            sum(item["citation_valid"] for item in results) / len(results), 3
        ) if results else 0.0,
        "abstention_accuracy": round(
            sum(item["abstention_correct"] for item in unanswerable) / len(unanswerable), 3
        ) if unanswerable else 0.0,
        "mean_latency_ms": round(sum(item["latency_ms"] for item in results) / len(results), 1)
        if results else 0.0,
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
