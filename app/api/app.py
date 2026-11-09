"""FastAPI application factory."""

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.query import router as query_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Academic RAG",
        version="0.1.0",
        description="Academics-only PDF retrieval foundation.",
    )
    app.include_router(health_router)
    app.include_router(query_router)
    return app


app = create_app()

