"""OpenAI PDF-input fallback for scanned or image-only academic documents."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from openai import OpenAI, OpenAIError

from app.config import Settings
from app.ingestion.schemas import AcademicAnalysis

logger = logging.getLogger(__name__)


class OpenAIPdfAnalysisError(RuntimeError):
    """Raised when direct PDF analysis cannot be completed."""


class OpenAIPdfAnalyzer:
    """Analyze a PDF directly with a vision-capable OpenAI model."""

    def __init__(self, settings: Settings):
        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is not configured")
        self.client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_request_timeout_seconds,
            max_retries=0,
        )
        self.model = settings.openai_pdf_model
        self.detail = settings.openai_pdf_detail
        self.max_bytes = settings.openai_pdf_max_mb * 1024 * 1024
        self.retries = settings.openai_pdf_retries

    @staticmethod
    def _parse_json(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            content = content.removeprefix("```json").removeprefix("```")
            content = content.removesuffix("```").strip()
        return json.loads(content)

    def analyze_file(self, path: Path) -> AcademicAnalysis:
        if path.stat().st_size > self.max_bytes:
            raise OpenAIPdfAnalysisError(
                f"PDF exceeds OpenAI direct-input limit of {self.max_bytes // (1024 * 1024)} MB"
            )

        prompt = (
            "Analyze this academic PDF for an academics-only institutional corpus. Return only "
            "valid JSON with nested classification and metadata objects. classification must "
            "contain is_academic (boolean), confidence (0 to 1), category, and reason. Academic "
            "includes courses, syllabi, curricula, examinations, academic calendars, admissions, "
            "results, regulations, research, and official academic notices. Exclude procurement, "
            "tenders, hostel, mess, sports, cultural, and unrelated administration. metadata must "
            "contain title, institution, department, program, document_type, academic_year, "
            "semester, audience, course_code, keywords (array of strings), and summary. Use "
            "'Unknown' when absent; do not invent values. Keep reason and summary under 25 words."
        )
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            started = time.monotonic()
            file_id: str | None = None
            logger.info("OpenAI direct PDF analysis started: file=%s attempt=%d/%d model=%s",
                        path.name, attempt, self.retries, self.model)
            try:
                with path.open("rb") as file:
                    uploaded = self.client.files.create(file=file, purpose="user_data")
                file_id = uploaded.id
                response = self.client.responses.create(
                    model=self.model,
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": file_id, "detail": self.detail},
                            {"type": "input_text", "text": prompt},
                        ],
                    }],
                )
                result = AcademicAnalysis.model_validate(self._parse_json(response.output_text))
                logger.info("OpenAI direct PDF analysis completed: file=%s duration=%.1fs",
                            path.name, time.monotonic() - started)
                return result
            except (OSError, OpenAIError, json.JSONDecodeError, ValueError) as error:
                last_error = error
                logger.warning("OpenAI direct PDF analysis failed: file=%s duration=%.1fs error=%s",
                               path.name, time.monotonic() - started, error)
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 10))
            finally:
                if file_id:
                    try:
                        self.client.files.delete(file_id)
                    except OpenAIError:
                        logger.warning("Could not delete temporary OpenAI file: %s", file_id)
        raise OpenAIPdfAnalysisError(
            f"OpenAI direct PDF analysis failed after {self.retries} attempts: {last_error}"
        ) from last_error
