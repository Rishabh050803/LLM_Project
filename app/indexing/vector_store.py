"""Persistent Chroma vector database for academic chunks."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.indexing.chunker import Chunk

logger = logging.getLogger(__name__)


class AcademicVectorStore:
    """Store chunks, embeddings, and provenance in a local persistent collection."""

    def __init__(self, directory: Path, collection_name: str = "academic_chunks"):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: dict[str, list[float]],
        document_metadata: list[dict],
    ) -> int:
        if not chunks:
            return 0
        metadata_by_id = {item["chunk_id"]: item for item in document_metadata}
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[embeddings[chunk.chunk_id] for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[{
                **metadata_by_id[chunk.chunk_id],
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chunk_index": chunk.chunk_index,
            } for chunk in chunks],
        )
        logger.info("Vector store upserted: chunks=%d total=%d", len(chunks), self.count)
        return len(chunks)
