"""Download discovered PDF URLs without invoking metadata extraction."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.config import Settings
from app.ingestion.crawler import AcademicCrawler

logger = logging.getLogger(__name__)


class PdfDownloader:
    """Download only the URLs in the discovery manifest."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.user_agent = "academic-rag/0.1 (+local academic research tool)"

    @staticmethod
    def _safe_filename(url: str) -> str:
        stem = Path(urlparse(url).path).stem or "document"
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:80]
        suffix = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        return f"{stem}_{suffix}.pdf"

    def _download_one(self, url: str) -> dict[str, object]:
        destination = self.settings.raw_pdf_dir / self._safe_filename(url)
        temporary = destination.with_suffix(".download")
        max_bytes = self.settings.max_pdf_size_mb * 1024 * 1024
        timeout = httpx.Timeout(self.settings.scrape_timeout_seconds)
        try:
            with (httpx.Client(headers={"User-Agent": self.user_agent},
                               follow_redirects=True, timeout=timeout) as client,
                  client.stream("GET", url) as response):
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    if not AcademicCrawler._is_pdf_url(url, content_type):
                        raise ValueError("response_is_not_pdf")
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > max_bytes:
                        raise ValueError("file_too_large")
                    written = 0
                    with temporary.open("wb") as file:
                        for chunk in response.iter_bytes(1024 * 64):
                            written += len(chunk)
                            if written > max_bytes:
                                raise ValueError("file_too_large")
                            file.write(chunk)
            temporary.replace(destination)
            logger.info("Downloaded PDF: %s -> %s", url, destination)
            return {"url": url, "path": str(destination), "status": "downloaded"}
        except (OSError, ValueError, httpx.HTTPError) as error:
            if temporary.exists():
                temporary.unlink()
            logger.warning("Download failed: %s: %s", url, error)
            return {"url": url, "path": str(destination), "status": "error", "error": str(error)}

    def download(self, urls: list[str], limit: int | None = None) -> dict[str, object]:
        target = limit or len(urls)
        unique_urls: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        for url in urls:
            key = AcademicCrawler._url_key(url)
            if key not in seen:
                seen.add(key)
                unique_urls.append(url)
        workers = max(1, self.settings.scrape_download_workers)
        logger.info("Download started: target=%d candidates=%d workers=%d",
                    target, len(unique_urls), workers)
        results: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            cursor = 0
            while cursor < len(unique_urls) and sum(
                    item["status"] == "downloaded" for item in results) < target:
                successful = sum(item["status"] == "downloaded" for item in results)
                remaining = max(1, target - successful)
                batch = unique_urls[cursor:cursor + min(workers, remaining)]
                cursor += len(batch)
                futures = [pool.submit(self._download_one, url) for url in batch]
                for future in as_completed(futures):
                    results.append(future.result())
        report = {
            "documents_requested": len(results),
            "documents_downloaded": sum(item["status"] == "downloaded" for item in results),
            "errors": sum(item["status"] == "error" for item in results),
            "documents": sorted(results, key=lambda item: str(item["url"])),
        }
        destination = self.settings.index_dir / "downloaded_pdfs.json"
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Download manifest saved: %s", destination)
        return report
