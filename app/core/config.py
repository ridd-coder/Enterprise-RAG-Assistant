"""
app/core/config.py

Centralised configuration management using Pydantic Settings.
All values are loaded from environment variables / .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_env: str = Field(default="development", description="Runtime environment")
    app_port: int = Field(default=8000)
    app_host: str = Field(default="0.0.0.0")
    log_level: str = Field(default="INFO")
    secret_key: str = Field(default="change-me-in-production")

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------
    openai_api_key: str = Field(..., description="OpenAI API key — required")
    openai_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    openai_max_tokens: int = Field(default=1024)
    openai_temperature: float = Field(default=0.0)

    # ------------------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------------------
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="enterprise_documents")
    qdrant_api_key: str = Field(default="")

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="rag_assistant")
    postgres_user: str = Field(default="rag_user")
    postgres_password: str = Field(default="")

    # ------------------------------------------------------------------
    # Document processing
    # ------------------------------------------------------------------
    max_file_size_mb: int = Field(default=50)
    allowed_extensions: List[str] = Field(default=["pdf"])
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    data_dir: str = Field(default="data/documents")

    # ------------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------------
    top_k: int = Field(default=5)
    min_relevance_score: float = Field(default=0.5)
    embedding_batch_size: int = Field(default=32)

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"]
    )

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, v: str | list) -> list:
        """Accept comma-separated string or list."""
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list) -> list:
        """Accept comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB limit to bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def data_path(self) -> Path:
        """Resolved path to document storage directory."""
        return Path(self.data_dir).resolve()

    @property
    def database_url(self) -> str:
        """Async PostgreSQL connection URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
