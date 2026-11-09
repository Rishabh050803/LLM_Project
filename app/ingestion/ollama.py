"""Ollama-backed local classification and metadata extraction."""

import json
import logging
import time

import httpx
from ollama import Client, ResponseError
from pydantic import ValidationError

from app.config import Settings
from app.ingestion.schemas import AcademicAnalysis, AcademicClassification, AcademicMetadata

logger = logging.getLogger(__name__)


class OllamaAnalysisError(RuntimeError):
    """Raised after Ollama retries are exhausted; callers should keep work pending."""


class OllamaAcademicAnalyzer:
    """Uses local models only; this class never calls OpenAI."""

    def __init__(self, settings: Settings):
        self.client = Client(
            host=settings.ollama_base_url,
            timeout=settings.ollama_request_timeout_seconds,
        )
        self.classifier_model = settings.ollama_classifier_model
        self.metadata_model = settings.ollama_metadata_model
        self.request_timeout_seconds = settings.ollama_request_timeout_seconds
        self.retries = settings.ollama_retries
        self.combined_analysis = settings.ollama_combined_analysis

    @staticmethod
    def _text_excerpt(page_text: list[tuple[int, str]], limit: int = 6000) -> str:
        sections: list[str] = []
        total = 0
        for page_number, text in page_text:
            if not text:
                continue
            section = f"\n[Page {page_number}]\n{text}"
            sections.append(section)
            total += len(section)
            if total >= limit:
                break
        return "".join(sections)[:limit]

    def _json_chat(self, model: str, operation: str, system: str, user: str) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            started = time.monotonic()
            logger.info(
                "Ollama %s started: model=%s attempt=%d/%d timeout=%ss",
                operation, model, attempt, self.retries, self.request_timeout_seconds,
            )
            try:
                response = self.client.chat(
                    model=model,
                    format="json",
                    think=False,
                    options={"temperature": 0, "num_ctx": 4096, "num_predict": 512},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = response.message.content.strip()
                if content.startswith("```"):
                    content = content.removeprefix("```json").removeprefix("```")
                    content = content.removesuffix("```").strip()
                result = json.loads(content)
                logger.info(
                    "Ollama %s completed: model=%s duration=%.1fs",
                    operation, model, time.monotonic() - started,
                )
                return result
            except (httpx.HTTPError, ResponseError, json.JSONDecodeError, ValidationError) as error:
                last_error = error
                logger.warning(
                    "Ollama %s failed: model=%s duration=%.1fs error=%s",
                    operation, model, time.monotonic() - started, error,
                )
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 10))
        raise OllamaAnalysisError(
            f"Ollama {operation} failed after {self.retries} attempts: {last_error}"
        ) from last_error

    def analyze(self, page_text: list[tuple[int, str]]) -> AcademicAnalysis:
        """Classify and extract metadata in one local-model request."""
        result = self._json_chat(
            self.metadata_model,
            "combined academic classification and metadata extraction",
            (
                "Analyze a PDF for an academics-only institutional corpus. Return only JSON "
                "with nested classification and metadata objects. classification must contain "
                "is_academic (boolean), confidence (0 to 1), category, and reason. Academic "
                "includes courses, syllabi, curricula, examinations, academic calendars, "
                "admissions, results, regulations, research, and official academic notices. "
                "Exclude procurement, tenders, hostel, mess, sports, cultural, and unrelated "
                "administration. metadata must contain title, institution, department, program, "
                "document_type, academic_year, semester, audience, course_code, keywords "
                "(array of strings), and summary. Use 'Unknown' when absent; do not invent values. "
                "Keep reason and summary under 25 words and keywords to at most 8 items."
            ),
            self._text_excerpt(page_text),
        )
        return AcademicAnalysis.model_validate(result)

    def classify(self, page_text: list[tuple[int, str]]) -> AcademicClassification:
        excerpt = self._text_excerpt(page_text)
        result = self._json_chat(
            self.classifier_model,
            "academic classification",
            (
                "Classify a PDF for an academics-only institutional corpus. "
                "Academic includes courses, syllabi, curricula, examinations, "
                "academic calendars, admissions, results, regulations, research, "
                "and official academic notices. Exclude procurement, tenders, "
                "hostel, mess, sports, cultural, and unrelated administration. "
                "Return only JSON with is_academic, confidence, category, reason."
            ),
            excerpt,
        )
        return AcademicClassification.model_validate(result)

    def extract_metadata(self, page_text: list[tuple[int, str]]) -> AcademicMetadata:
        excerpt = self._text_excerpt(page_text)
        result = self._json_chat(
            self.metadata_model,
            "academic metadata extraction",
            (
                "Extract normalized metadata from an academic PDF. Return only JSON "
                "matching these fields: title, institution, department, program, "
                "document_type, academic_year, semester, audience, course_code, "
                "keywords (array of strings), summary. Use 'Unknown' when absent. "
                "Do not invent values."
            ),
            excerpt,
        )
        return AcademicMetadata.model_validate(result)

