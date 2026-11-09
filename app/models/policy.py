"""Explicit model allocation policy for token and cost control."""

from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True)
class ModelPolicy:
    """Maps RAG stages to the cheapest suitable provider."""

    metadata_model: str
    query_model: str
    classifier_model: str
    answer_model: str
    embedding_model: str

    @classmethod
    def from_settings(cls, settings: Settings) -> "ModelPolicy":
        return cls(
            metadata_model=f"ollama/{settings.ollama_metadata_model}",
            query_model=f"ollama/{settings.ollama_query_model}",
            classifier_model=f"ollama/{settings.ollama_classifier_model}",
            answer_model=f"openai/{settings.openai_chat_model}",
            embedding_model=f"openai/{settings.openai_embedding_model}",
        )

