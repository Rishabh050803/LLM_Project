"""HTML-only PDF URL discovery.

This module deliberately does not download, parse, or classify PDFs. It only
visits HTML pages and follows their anchor links until the PDF URL limit is
reached or the link queue is exhausted.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from app.config import Settings

logger = logging.getLogger(__name__)


class AcademicCrawler:
    """Discover academic PDF URLs from HTML pages on one institutional host."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.user_agent = "academic-rag/0.1 (+local academic research tool)"

    @staticmethod
    def _normalize_url(url: str) -> str:
        url, _ = urldefrag(url)
        parsed = urlparse(url)
        return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/",
                           parsed.params, parsed.query, ""))

    @staticmethod
    def _url_key(url: str) -> tuple[str, str, str]:
        """Return a deduplication key for common institutional URL aliases."""
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower().removeprefix("www.")
        return hostname, parsed.path or "/", parsed.query

    @staticmethod
    def _is_pdf_url(url: str, content_type: str = "") -> bool:
        return (urlparse(url).path.lower().endswith(".pdf")
                or "application/pdf" in content_type.lower())

    def _same_domain(self, url: str, root_hostname: str) -> bool:
        hostname = (urlparse(url).hostname or "").lower().removeprefix("www.")
        if self.settings.scrape_allow_subdomains:
            return hostname == root_hostname or hostname.endswith(f".{root_hostname}")
        return hostname == root_hostname

    @staticmethod
    def _path_priority(url: str) -> int:
        path = urlparse(url).path.lower()
        if path.endswith(".pdf"):
            return 100
        if path.startswith(("/academics", "/dept_", "/students/academics",
                            "/students/admissions")):
            return 80
        if path.startswith("/research/"):
            return 50
        if path.startswith(("/quick_links/recruitment", "/admin", "/hostel", "/sports",
                            "/facilities", "/about_us", "/footer", "/alumni")):
            return -100
        return 0

    def _robots(self, client: httpx.Client, origin: str) -> RobotFileParser:
        parser = RobotFileParser()
        parser.set_url(urljoin(origin, "/robots.txt"))
        try:
            response = client.get(parser.url)
            parser.parse(response.text.splitlines() if response.status_code < 400 else [])
        except httpx.HTTPError:
            logger.warning("Could not read robots.txt for %s; continuing cautiously", origin)
            parser.parse([])
        return parser

    def discover(self, start_url: str, pdf_limit: int | None = None) -> dict[str, object]:
        """Visit HTML pages and return PDF URLs; PDF contents are never inspected."""
        start_url = self._normalize_url(start_url)
        parsed_start = urlparse(start_url)
        if parsed_start.scheme not in {"http", "https"} or not parsed_start.hostname:
            raise ValueError("start_url must be an absolute HTTP(S) URL")

        limit = pdf_limit or self.settings.scrape_pdf_limit
        root_hostname = parsed_start.hostname.lower().removeprefix("www.")
        origin = f"{parsed_start.scheme}://{parsed_start.netloc}"
        queue: deque[str] = deque([start_url])
        queued = {self._url_key(start_url)}
        visited: set[tuple[str, str, str]] = set()
        pdf_keys: set[tuple[str, str, str]] = set()
        pdf_urls: list[str] = []
        errors: list[dict[str, str]] = []
        pages_scanned = 0
        timeout = httpx.Timeout(self.settings.scrape_timeout_seconds)
        headers = {"User-Agent": self.user_agent, "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.1"}

        logger.info("Discovery started: url=%s pdf_limit=%d allow_subdomains=%s",
                    start_url, limit, self.settings.scrape_allow_subdomains)
        with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout) as client:
            robots = self._robots(client, origin)
            while queue and len(pdf_urls) < limit:
                current = queue.popleft()
                current_key = self._url_key(current)
                if current_key in visited or not self._same_domain(current, root_hostname):
                    continue
                visited.add(current_key)
                if not robots.can_fetch(self.user_agent, current):
                    logger.info("Skipped by robots.txt: %s", current)
                    continue

                # A PDF URL is a discovery result. Its bytes are not parsed here.
                if self._is_pdf_url(current):
                    if current_key not in pdf_keys:
                        pdf_keys.add(current_key)
                        pdf_urls.append(current)
                        logger.info("Discovered PDF %d/%d: %s", len(pdf_urls), limit, current)
                    continue

                pages_scanned += 1
                try:
                    response = client.get(current)
                    response.raise_for_status()
                    final_url = self._normalize_url(str(response.url))
                    final_key = self._url_key(final_url)
                    content_type = response.headers.get("content-type", "")
                    if self._is_pdf_url(current, content_type):
                        if final_key not in pdf_keys:
                            pdf_keys.add(final_key)
                            pdf_urls.append(final_url)
                            logger.info("Discovered PDF %d/%d: %s", len(pdf_urls), limit, final_url)
                    elif "text/html" in content_type.lower():
                        soup = BeautifulSoup(response.text, "html.parser")
                        links: set[str] = set()
                        for anchor in soup.find_all("a", href=True):
                            link = self._normalize_url(urljoin(final_url, anchor["href"]))
                            link_key = self._url_key(link)
                            if (link_key not in queued and self._same_domain(link, root_hostname)
                                    and urlparse(link).scheme in {"http", "https"}
                                    and self._path_priority(link) >= 0):
                                links.add(link)
                        for link in sorted(links, key=self._path_priority, reverse=True):
                            queued.add(self._url_key(link))
                            queue.append(link)
                except (httpx.HTTPError, ValueError) as error:
                    errors.append({"url": current, "error": str(error)})
                    logger.warning("Discovery skipped %s: %s", current, error)
                time.sleep(self.settings.scrape_delay_seconds)

        logger.info("Discovery completed: html_pages=%d pdfs=%d queue_remaining=%d",
                    pages_scanned, len(pdf_urls), len(queue))
        return {
            "start_url": start_url,
            "root_domain": root_hostname,
            "pdf_limit": limit,
            "pages_scanned": pages_scanned,
            "pdf_urls": pdf_urls,
            "errors": errors,
        }

    def save_discovery(self, result: dict[str, object]):
        destination = self.settings.index_dir / "discovered_pdfs.json"
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Discovery manifest saved: %s", destination)
        return destination

    def load_discovery(self) -> dict[str, object]:
        source = self.settings.index_dir / "discovered_pdfs.json"
        if not source.exists():
            raise FileNotFoundError(f"Discovery manifest not found: {source}; run crawl first")
        return json.loads(source.read_text(encoding="utf-8"))
