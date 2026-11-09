"""Orchestration for local academic PDF ingestion."""

import json
import logging
from pathlib import Path
from urllib.parse import urlparse

from app.config import Settings
from app.ingestion.manifest import Manifest
from app.ingestion.ollama import OllamaAcademicAnalyzer, OllamaAnalysisError
from app.ingestion.openai_pdf import OpenAIPdfAnalysisError, OpenAIPdfAnalyzer
from app.ingestion.pdf import extract_pages, sha256_file
from app.ingestion.schemas import IngestRecord

logger = logging.getLogger(__name__)


class AcademicIngestionService:
    """Ingests PDFs, classifies them locally, and persists page text/metadata."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.manifest = Manifest(settings.index_dir / "manifest.sqlite3")
        self.analyzer = OllamaAcademicAnalyzer(settings)
        self.openai_pdf_analyzer = (
            OpenAIPdfAnalyzer(settings) if settings.openai_api_key is not None else None
        )

    def _save_accepted(
        self,
        path: Path,
        source_url: str | None,
        content_hash: str,
        pages: list,
        classification,
        metadata,
        extraction_method: str,
    ) -> IngestRecord:
        parsed_path = self.settings.parsed_dir / f"{content_hash}.json"
        parsed_path.write_text(
            json.dumps({
                "content_hash": content_hash,
                "filename": path.name,
                "source_url": source_url,
                "extraction_method": extraction_method,
                "classification": classification.model_dump(),
                "metadata": metadata.model_dump(),
                "pages": [page.model_dump() for page in pages],
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        record = IngestRecord(
            content_hash=content_hash,
            filename=path.name,
            path=str(path),
            source_url=source_url,
            page_count=len(pages),
            status="accepted",
            classification=classification,
            metadata=metadata,
        )
        self.manifest.upsert(record)
        return record

    def _ingest_direct_pdf(
        self,
        path: Path,
        source_url: str | None,
        content_hash: str,
        pages: list,
    ) -> IngestRecord:
        """Use OpenAI vision PDF input when local extraction has no usable text."""
        try:
            analysis = self.openai_pdf_analyzer.analyze_file(path)
        except OpenAIPdfAnalysisError as error:
            record = IngestRecord(
                content_hash=content_hash,
                filename=path.name,
                path=str(path),
                source_url=source_url,
                page_count=len(pages),
                status="pending_openai",
                rejection_reason=str(error),
            )
            self.manifest.upsert(record)
            return record
        if self.settings.require_academic_classification and not analysis.classification.is_academic:
            record = IngestRecord(
                content_hash=content_hash,
                filename=path.name,
                path=str(path),
                source_url=source_url,
                page_count=len(pages),
                status="rejected",
                rejection_reason="not_academic",
                classification=analysis.classification,
            )
            self.manifest.upsert(record)
            return record
        return self._save_accepted(
            path, source_url, content_hash, pages, analysis.classification,
            analysis.metadata, "openai_direct_pdf",
        )

    def _source_allowed(self, source_url: str | None) -> bool:
        if not source_url or not self.settings.allowed_domains:
            return True
        hostname = (urlparse(source_url).hostname or "").lower().removeprefix("www.")
        return any(hostname == domain or hostname.endswith(f".{domain}")
                   for domain in self.settings.allowed_domains)

    def ingest_file(self, path: Path, source_url: str | None = None) -> IngestRecord:
        path = path.resolve()
        content_hash = sha256_file(path)
        base = {
            "content_hash": content_hash,
            "filename": path.name,
            "path": str(path),
            "source_url": source_url,
        }

        if path.suffix.lower() != ".pdf":
            record = IngestRecord(**base, status="rejected", rejection_reason="not_a_pdf")
            self.manifest.upsert(record)
            return record

        max_bytes = self.settings.max_pdf_size_mb * 1024 * 1024
        if path.stat().st_size > max_bytes:
            record = IngestRecord(**base, status="rejected", rejection_reason="file_too_large")
            self.manifest.upsert(record)
            return record

        if not self._source_allowed(source_url):
            record = IngestRecord(**base, status="rejected", rejection_reason="source_not_allowed")
            self.manifest.upsert(record)
            return record

        if self.manifest.has_hash(content_hash):
            return IngestRecord(**base, status="duplicate")

        try:
            pages = extract_pages(path)
        except Exception as error:
            logger.exception("PDF extraction failed: %s", path)
            if self.openai_pdf_analyzer is not None:
                return self._ingest_direct_pdf(path, source_url, content_hash, [])
            record = IngestRecord(
                **base,
                status="error",
                rejection_reason=f"pdf_extraction_error: {error}",
            )
            self.manifest.upsert(record)
            return record
        page_pairs = [(page.page_number, page.text) for page in pages]
        if not any(text for _, text in page_pairs):
            if self.openai_pdf_analyzer is not None:
                return self._ingest_direct_pdf(path, source_url, content_hash, pages)
            record = IngestRecord(
                **base, page_count=len(pages), status="rejected", rejection_reason="no_extractable_text"
            )
            self.manifest.upsert(record)
            return record

        try:
            if self.analyzer.combined_analysis:
                analysis = self.analyzer.analyze(page_pairs)
                classification = analysis.classification
                metadata = analysis.metadata
            else:
                classification = self.analyzer.classify(page_pairs)
                metadata = None
        except OllamaAnalysisError as error:
            record = IngestRecord(
                **base,
                page_count=len(pages),
                status="pending_ollama",
                rejection_reason=str(error),
            )
            self.manifest.upsert(record)
            return record

        if self.settings.require_academic_classification and not classification.is_academic:
            record = IngestRecord(
                **base,
                page_count=len(pages),
                status="rejected",
                rejection_reason="not_academic",
                classification=classification,
            )
            self.manifest.upsert(record)
            return record

        if metadata is None:
            try:
                metadata = self.analyzer.extract_metadata(page_pairs)
            except OllamaAnalysisError as error:
                record = IngestRecord(
                    **base,
                    page_count=len(pages),
                    status="pending_ollama",
                    rejection_reason=str(error),
                    classification=classification,
                )
                self.manifest.upsert(record)
                return record
        return self._save_accepted(
            path, source_url, content_hash, pages, classification, metadata, "local_text"
        )

    def ingest_directory(self, directory: Path) -> list[IngestRecord]:
        files = sorted(path for path in directory.rglob("*") if path.is_file())
        results: list[IngestRecord] = []
        for path in files:
            try:
                results.append(self.ingest_file(path))
            except Exception:
                logger.exception("Failed to ingest %s", path)
                results.append(IngestRecord(
                    content_hash="error",
                    filename=path.name,
                    path=str(path),
                    status="error",
                    rejection_reason="processing_error",
                ))
        return results

