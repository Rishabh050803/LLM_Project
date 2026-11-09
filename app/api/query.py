"""Query endpoint for the academic RAG pipeline."""

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.rag.service import AcademicRAGService
from app.retrieval.schemas import GroundedAnswer, QueryRequest

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=GroundedAnswer)
def query(request: QueryRequest) -> GroundedAnswer:
    """Retrieve academic evidence and answer with citations."""
    try:
        return AcademicRAGService(get_settings()).query(request)
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
