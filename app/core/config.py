"""
app/core/config.py

Centralised configuration management using Pydantic Settings.
All values are loaded from environment variables / .env file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(Exception):
    """Raised when required configuration (e.g. API keys) is missing or invalid."""


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
    # Gemini
    # ------------------------------------------------------------------
    gemini_api_key: str = Field(
        default="",
        description="Gemini API key — required for embedding and generation",
    )
    gemini_model: str = Field(default="gemini-3.8-flash")
    gemini_embedding_model: str = Field(default="gemini-embedding-2")
    gemini_max_tokens: int = Field(default=1024)
    gemini_temperature: float = Field(default=0.0)

    def require_gemini_key(self) -> str:
        """Validate and return Gemini API key, raising ConfigurationError if missing."""
        key = self.gemini_api_key.strip()
        if not key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your .env "
                "file or as an environment variable to enable embedding and generation."
            )
        return key

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
    allowed_extensions_str: str = Field(default="pdf", alias="allowed_extensions")
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
    cors_origins_str: str = Field(
        default="http://localhost:3000,http://localhost:5173", alias="cors_origins"
    )

    @property
    def allowed_extensions(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions_str.split(",")]

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins_str.split(",")]

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
