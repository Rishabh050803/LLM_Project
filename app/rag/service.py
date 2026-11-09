"""End-to-end academic retrieval and answer generation."""

from app.answering.service import AcademicAnswerer
from app.config import Settings
from app.retrieval.schemas import GroundedAnswer, QueryRequest
from app.retrieval.service import AcademicRetriever


class AcademicRAGService:
    """Run retrieval followed by one grounded answer call, without reranking."""

    def __init__(self, settings: Settings):
        self.retriever = AcademicRetriever(settings)
        self.answerer = AcademicAnswerer(settings)

    def query(self, request: QueryRequest) -> GroundedAnswer:
        chunks = self.retriever.retrieve(request.question, request.top_k, request.filters)
        return self.answerer.answer(request.question, chunks)
