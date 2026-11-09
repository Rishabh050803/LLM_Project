"""Safe, page-preserving PDF extraction."""

import hashlib
import logging
from pathlib import Path

from pypdf import PdfReader

from app.ingestion.schemas import PageText

logger = logging.getLogger(__name__)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def extract_pages(path: Path) -> list[PageText]:
    logger.info("PDF extraction started: file=%s", path)
    reader = PdfReader(str(path), strict=False)
    pages: list[PageText] = []
    total_pages = len(reader.pages)
    for number, page in enumerate(reader.pages, start=1):
        logger.info("PDF extraction page %d/%d: file=%s", number, total_pages, path.name)
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            logger.exception("PDF extraction failed on page %d/%d: file=%s",
                             number, total_pages, path)
            raise
        pages.append(PageText(page_number=number, text=text))
    logger.info("PDF extraction completed: file=%s pages=%d text_pages=%d",
                path, total_pages, sum(bool(page.text) for page in pages))
    return pages

