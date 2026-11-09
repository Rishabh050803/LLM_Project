"""Batched OpenAI embedding generation with a small local cache."""

from __future__ import annotations

import json
import logging
import sqlite3

from openai import OpenAI

from app.config import Settings
from app.indexing.chunker import Chunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate and cache embeddings so reruns do not pay twice."""

    def __init__(self, settings: Settings):
        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required for embedding generation")
        self.settings = settings
        self.model = settings.openai_embedding_model
        self.client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_request_timeout_seconds,
            max_retries=2,
        )
        self.cache_path = settings.index_dir / "embedding_cache.sqlite3"
        with sqlite3.connect(self.cache_path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "chunk_id TEXT NOT NULL, model TEXT NOT NULL, vector_json TEXT NOT NULL, "
                "tokens INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (chunk_id, model))"
            )

    def _cached(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with sqlite3.connect(self.cache_path) as db:
            rows = db.execute(
                f"SELECT chunk_id, vector_json FROM embeddings WHERE model = ? "
                f"AND chunk_id IN ({placeholders})",
                [self.model, *ids],
            ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def embed(self, chunks: list[Chunk], batch_size: int = 64) -> dict[str, list[float]]:
        result = self._cached([chunk.chunk_id for chunk in chunks])
        missing = [chunk for chunk in chunks if chunk.chunk_id not in result]
        logger.info("Embedding preparation: chunks=%d cached=%d missing=%d model=%s",
                    len(chunks), len(result), len(missing), self.model)
        with sqlite3.connect(self.cache_path) as db:
            for offset in range(0, len(missing), batch_size):
                batch = missing[offset:offset + batch_size]
                response = self.client.embeddings.create(
                    model=self.model,
                    input=[chunk.text for chunk in batch],
                    encoding_format="float",
                )
                by_index = {item.index: item.embedding for item in response.data}
                for index, chunk in enumerate(batch):
                    vector = list(by_index[index])
                    result[chunk.chunk_id] = vector
                    db.execute(
                        "INSERT OR REPLACE INTO embeddings "
                        "(chunk_id, model, vector_json, tokens) VALUES (?, ?, ?, ?)",
                        (chunk.chunk_id, self.model, json.dumps(vector), response.usage.total_tokens),
                    )
                db.commit()
                logger.info("Embeddings generated: %d/%d batch_tokens=%d",
                            min(offset + len(batch), len(missing)), len(missing),
                            response.usage.total_tokens)
        return result
