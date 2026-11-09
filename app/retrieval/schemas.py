"""Schemas for retrieval and grounded answers."""

from pydantic import BaseModel, Field


class RetrievalFilters(BaseModel):
    """Optional exact metadata filters applied inside Chroma."""

    category: str | None = None
    document_type: str | None = None
    department: str | None = None
    program: str | None = None
    academic_year: str | None = None


class RetrievedChunk(BaseModel):
    """One deterministically ranked chunk returned by vector similarity."""

    chunk_id: str
    text: str
    score: float
    filename: str
    source_url: str
    page_start: int
    page_end: int
    title: str = "Unknown"
    category: str = "unknown"


class QueryRequest(BaseModel):
    """User query and retrieval controls."""

    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)
    filters: RetrievalFilters | None = None


class GroundedAnswer(BaseModel):
    """Answer plus the retrieved evidence used to produce it."""

    answer: str
    sources: list[RetrievedChunk]
    retrieval_count: int
