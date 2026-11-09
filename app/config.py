"""Typed application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        # Support running from this project while keeping the shared project
        # environment at /home/rkp/LLM_Project/.env available.
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    academic_only: bool = True
    academic_allowed_domains: str = ""
    require_academic_classification: bool = True
    max_pdf_size_mb: int = 100
    scrape_pdf_limit: int = 100
    scrape_process_limit: int = 10
    scrape_download_workers: int = 4
    scrape_processing_workers: int = 1
    scrape_allow_subdomains: bool = False
    scrape_delay_seconds: float = 0.25
    scrape_timeout_seconds: float = 20.0
    ollama_request_timeout_seconds: float = 120.0
    ollama_retries: int = 2
    ollama_combined_analysis: bool = True
    max_context_chunks: int = 8
    retrieval_candidates: int = 24
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_request_timeout_seconds: float = 60.0
    openai_pdf_model: str = "gpt-4o-mini"
    openai_pdf_detail: str = "low"
    openai_pdf_max_mb: int = 49
    openai_pdf_retries: int = 2
    ollama_base_url: str = "http://localhost:11434"
    ollama_metadata_model: str = "qwen3:4b"
    ollama_query_model: str = "qwen3:4b"
    ollama_classifier_model: str = "gemma3:4b"
    ollama_code_model: str = "qwen2.5-coder:7b"

    @property
    def raw_pdf_dir(self) -> Path:
        return self.data_dir / "raw_pdfs"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def allowed_domains(self) -> tuple[str, ...]:
        return tuple(
            domain.strip().lower().removeprefix("www.")
            for domain in self.academic_allowed_domains.split(",")
            if domain.strip()
        )

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.raw_pdf_dir, self.parsed_dir, self.index_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

