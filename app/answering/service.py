"""Final answer generation with explicit source-grounding instructions."""

from __future__ import annotations

import logging

from openai import OpenAI

from app.config import Settings
from app.retrieval.schemas import GroundedAnswer, RetrievedChunk

logger = logging.getLogger(__name__)


class AcademicAnswerer:
    """Use one final OpenAI call; retrieval itself has no LLM reranker."""

    def __init__(self, settings: Settings):
        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required for answer generation")
        self.client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_request_timeout_seconds,
            max_retries=2,
        )
        self.model = settings.openai_chat_model

    @staticmethod
    def _context(chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            f"[{index}] {chunk.filename}, page {chunk.page_start}\n{chunk.text}"
            for index, chunk in enumerate(chunks, start=1)
        )

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> GroundedAnswer:
        if not chunks:
            return GroundedAnswer(
                answer="I could not find relevant academic information in the indexed documents.",
                sources=[],
                retrieval_count=0,
            )
        instructions = (
            "You answer questions about an institute's academic documents. Use only the supplied "
            "context. If the context does not support an answer, say so clearly. Do not invent "
            "dates, rules, fees, or requirements. Cite supporting sources inline as [1], [2], "
            "etc. Keep the answer concise and distinguish conflicting document versions."
        )
        prompt = f"Question: {question}\n\nAcademic context:\n{self._context(chunks)}"
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
            store=False,
        )
        answer = response.output_text.strip()
        logger.info("Grounded answer generated: model=%s sources=%d", self.model, len(chunks))
        return GroundedAnswer(answer=answer, sources=chunks, retrieval_count=len(chunks))
