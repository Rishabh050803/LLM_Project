"""OpenAI query embeddings and deterministic Chroma similarity retrieval."""

from __future__ import annotations

import logging

from openai import OpenAI

from app.config import Settings
from app.indexing.vector_store import AcademicVectorStore
from app.retrieval.schemas import RetrievalFilters, RetrievedChunk

logger = logging.getLogger(__name__)


class AcademicRetriever:
    """Retrieve by cosine similarity without an LLM reranking pass."""

    def __init__(self, settings: Settings):
        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required for retrieval")
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_request_timeout_seconds,
            max_retries=2,
        )
        self.store = AcademicVectorStore(settings.index_dir / "vector_db")

    @staticmethod
    def _where(filters: RetrievalFilters | None) -> dict | None:
        if filters is None:
            return None
        values = {key: value for key, value in filters.model_dump().items() if value}
        if not values:
            return None
        if len(values) == 1:
            key, value = next(iter(values.items()))
            return {key: {"$eq": value}}
        return {"$and": [{key: {"$eq": value}} for key, value in values.items()]}

    def retrieve(self, question: str, top_k: int = 8,
                 filters: RetrievalFilters | None = None) -> list[RetrievedChunk]:
        response = self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=question,
            encoding_format="float",
        )
        result = self.store.collection.query(
            query_embeddings=[response.data[0].embedding],
            n_results=min(max(top_k, 1), self.settings.retrieval_candidates),
            where=self._where(filters),
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            # Chroma returns cosine distance; higher score means closer match.
            chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                score=round(1.0 - float(distance), 6),
                filename=str(metadata.get("filename", "Unknown")),
                source_url=str(metadata.get("source_url", "")),
                page_start=int(metadata.get("page_start", 0)),
                page_end=int(metadata.get("page_end", 0)),
                title=str(metadata.get("title", "Unknown")),
                category=str(metadata.get("category", "unknown")),
            ))
        logger.info("Retrieved chunks: count=%d top_k=%d", len(chunks), top_k)
        return chunks[:top_k]
