"""Health and configuration-safe diagnostics."""

from fastapi import APIRouter

from app.config import get_settings
from app.models.policy import ModelPolicy

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    settings = get_settings()
    policy = ModelPolicy.from_settings(settings)
    return {
        "status": "ok",
        "environment": settings.app_env,
        "academic_only": settings.academic_only,
        "openai_key_configured": settings.openai_api_key is not None,
        "ollama_base_url": settings.ollama_base_url,
        "models": {
            "metadata": policy.metadata_model,
            "query": policy.query_model,
            "classifier": policy.classifier_model,
            "answer": policy.answer_model,
            "embedding": policy.embedding_model,
        },
        "directories": {
            "raw_pdfs": str(settings.raw_pdf_dir),
            "parsed": str(settings.parsed_dir),
            "index": str(settings.index_dir),
        },
    }

