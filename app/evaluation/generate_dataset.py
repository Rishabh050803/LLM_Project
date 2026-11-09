"""Generate a larger baseline dataset from accepted academic documents."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import httpx
from ollama import Client, ResponseError
from openai import OpenAI

from app.config import Settings

logger = logging.getLogger(__name__)

UNANSWERABLE_QUESTIONS = [
    "What is the daily hostel mess menu for this semester?",
    "What are the library opening and closing hours on public holidays?",
    "Which sports facilities are open after 10 PM?",
    "What is the current campus bus timetable?",
    "Which restaurants are located near the institute gate?",
    "What are the current medical-centre consultation hours?",
    "Who won the latest inter-college cricket tournament?",
    "What is the Wi-Fi password for the hostel?",
    "What is the weather forecast for Jaipur tomorrow?",
    "Which buses run from the railway station to campus?",
    "What is the current placement salary for Computer Science students?",
    "Which campus clubs are accepting new members this month?",
    "What is today's cafeteria lunch menu?",
    "How many parking spaces are currently available on campus?",
    "What are the current hostel room allotment results?",
    "Which faculty members are on campus today?",
    "What is the latest electricity outage schedule?",
    "Which student won the most recent athletics event?",
    "What is the current air quality near the institute?",
    "Which local shops deliver food to the hostel?",
]


def _clean_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()
    return json.loads(text)


def _prompt(documents: list[dict]) -> str:
    sections = []
    for document in documents:
        sections.append(
            f"DOCUMENT_FILENAME: {document['filename']}\n"
            f"TITLE: {document.get('metadata', {}).get('title', 'Unknown')}\n"
            f"SUMMARY: {document.get('metadata', {}).get('summary', '')}\n"
            f"TEXT_EXCERPT:\n{document['excerpt']}"
        )
    return (
        "Create exactly two distinct, answerable evaluation questions for EACH document below. "
        "Questions must be answerable only from the supplied document, concise, and useful for "
        "an academic institutional RAG system. Cover different facts where possible. Do not ask "
        "about information absent from the excerpt. Return only valid JSON in this shape: "
        "{\"queries\":[{\"question\":\"...\"}]}\n\n" + "\n\n---\n\n".join(sections)
    )


def _ollama_batch(client: Client, model: str, documents: list[dict]) -> list[dict]:
    response = client.chat(
        model=model,
        format="json",
        think=False,
        options={"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200},
        messages=[
            {"role": "system", "content": "You generate reliable evaluation questions from source text."},
            {"role": "user", "content": _prompt(documents)},
        ],
    )
    payload = _clean_json(response.message.content)
    queries = payload.get("queries", [])
    if len(queries) != len(documents) * 2:
        raise ValueError(f"Expected {len(documents) * 2} queries, received {len(queries)}")
    return queries


def _openai_batch(client: OpenAI, model: str, documents: list[dict]) -> list[dict]:
    response = client.responses.create(
        model=model,
        instructions=(
            "Return only valid JSON with a queries array. Create exactly two distinct, "
            "answerable questions per supplied document. Questions must be answerable only "
            "from the supplied text."
        ),
        input=_prompt(documents),
        store=False,
    )
    payload = _clean_json(response.output_text)
    queries = payload.get("queries", [])
    if len(queries) != len(documents) * 2:
        raise ValueError(f"Expected {len(documents) * 2} queries, received {len(queries)}")
    return queries


def generate_dataset(settings: Settings, output_path: Path, model: str = "gemma3:4b",
                     batch_documents: int = 4) -> dict:
    documents = []
    for path in sorted(settings.parsed_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("pages") and document.get("classification", {}).get("is_academic"):
            text = "\n".join(page.get("text", "") for page in document["pages"])
            documents.append({
                "filename": document["filename"],
                "metadata": document.get("metadata", {}),
                "excerpt": re.sub(r"\s+", " ", text)[:2800],
            })
    ollama = Client(host=settings.ollama_base_url, timeout=settings.ollama_request_timeout_seconds)
    openai = (
        OpenAI(api_key=settings.openai_api_key.get_secret_value(),
               timeout=settings.openai_request_timeout_seconds, max_retries=1)
        if settings.openai_api_key is not None else None
    )
    generated: list[dict] = []
    fallback_batches = 0
    for offset in range(0, len(documents), batch_documents):
        batch = documents[offset:offset + batch_documents]
        try:
            questions = _ollama_batch(ollama, model, batch)
            provider = "ollama"
        except (httpx.HTTPError, ResponseError, json.JSONDecodeError, ValueError) as error:
            logger.warning("Ollama dataset batch failed at offset=%d: %s", offset, error)
            if openai is None:
                raise
            questions = _openai_batch(openai, settings.openai_chat_model, batch)
            provider = "openai_fallback"
            fallback_batches += 1
        for document_index, document in enumerate(batch):
            for question_index in range(2):
                question = str(questions[document_index * 2 + question_index].get("question", "")).strip()
                if not question:
                    raise ValueError(f"Empty generated question for {document['filename']}")
                generated.append({
                    "id": f"generated_{len(generated) + 1:03d}",
                    "question": question,
                    "answerable": True,
                    "expected_source_terms": [Path(document["filename"]).stem],
                    "generation_provider": provider,
                })
        logger.info("Generated evaluation questions: %d/%d provider=%s",
                    len(generated), len(documents) * 2, provider)
    for index, question in enumerate(UNANSWERABLE_QUESTIONS, start=1):
        generated.append({
            "id": f"unanswerable_{index:03d}",
            "question": question,
            "answerable": False,
            "expected_source_terms": [],
            "generation_provider": "fixed",
        })
    if len(generated) < 100:
        raise ValueError(f"Only generated {len(generated)} evaluation questions")
    generated = generated[:100]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in generated) + "\n",
        encoding="utf-8",
    )
    return {
        "documents_used": len(documents),
        "queries_written": len(generated),
        "answerable_queries": sum(item["answerable"] for item in generated),
        "unanswerable_queries": sum(not item["answerable"] for item in generated),
        "ollama_model": model,
        "openai_fallback_batches": fallback_batches,
        "output": str(output_path),
    }
