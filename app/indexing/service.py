"""Build the academic vector index from accepted parsed documents."""

from __future__ import annotations

import json
import logging

from app.config import Settings
from app.indexing.chunker import chunk_pages
from app.indexing.embeddings import EmbeddingService
from app.indexing.vector_store import AcademicVectorStore

logger = logging.getLogger(__name__)


class AcademicIndexingService:
    """Index accepted documents; direct-PDF records without text are skipped."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = EmbeddingService(settings)
        self.store = AcademicVectorStore(settings.index_dir / "vector_db")

    def build(self, limit: int | None = None, reset: bool = False) -> dict:
        if reset:
            self.store.client.delete_collection(self.store.collection.name)
            self.store = AcademicVectorStore(self.settings.index_dir / "vector_db")
        files = sorted(self.settings.parsed_dir.glob("*.json"))
        indexed_documents = skipped_documents = 0
        total_chunks = 0
        for path in files:
            if limit is not None and indexed_documents + skipped_documents >= limit:
                break
            document = json.loads(path.read_text(encoding="utf-8"))
            if not document.get("pages"):
                skipped_documents += 1
                logger.warning("Skipping accepted document without text pages: %s", path.name)
                continue
            chunks = chunk_pages(document["content_hash"], document["pages"])
            if not chunks:
                skipped_documents += 1
                continue
            embeddings = self.embeddings.embed(chunks)
            common = {
                "content_hash": document["content_hash"],
                "filename": document["filename"],
                "source_url": document.get("source_url") or "",
                "title": document.get("metadata", {}).get("title", "Unknown"),
                "document_type": document.get("metadata", {}).get("document_type", "unknown"),
                "category": document.get("classification", {}).get("category", "unknown"),
            }
            self.store.upsert(
                chunks,
                embeddings,
                [{"chunk_id": chunk.chunk_id, **common} for chunk in chunks],
            )
            indexed_documents += 1
            total_chunks += len(chunks)
            logger.info("Indexed document %d: file=%s chunks=%d",
                        indexed_documents, path.name, len(chunks))
        return {
            "indexed_documents": indexed_documents,
            "skipped_documents": skipped_documents,
            "chunks_upserted": total_chunks,
            "vector_count": self.store.count,
            "vector_db": str(self.settings.index_dir / "vector_db"),
            "embedding_model": self.embeddings.model,
        }
