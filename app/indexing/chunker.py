"""Page-aware text chunking for academic PDFs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A searchable chunk retaining document and page provenance."""

    chunk_id: str
    text: str
    page_start: int
    page_end: int
    chunk_index: int


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_pages(
    content_hash: str,
    pages: list[dict],
    *,
    chunk_size: int = 1600,
    overlap: int = 240,
) -> list[Chunk]:
    """Create deterministic, page-aware character chunks.

    Character boundaries keep this stage dependency-light; the embedding API
    enforces its own token limit and chunks are deliberately well below it.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[Chunk] = []
    for page in pages:
        page_number = int(page["page_number"])
        text = _normalise(str(page.get("text", "")))
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary + (2 if text[boundary:boundary + 2] == ". " else 1)
            piece = text[start:end].strip()
            if piece:
                index = len(chunks)
                raw_id = f"{content_hash}:{page_number}:{index}:{piece}"
                chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
                chunks.append(Chunk(chunk_id, piece, page_number, page_number, index))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks
