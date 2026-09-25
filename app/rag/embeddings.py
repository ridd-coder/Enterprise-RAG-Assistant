"""
app/rag/embeddings.py

Embedding service backed by Google Gemini's text-embedding API.

Features:
- Decoupled from the rest of the application (swap model via config)
- Batch embedding with configurable batch size
- Automatic retry with exponential back-off (tenacity)
- Clean error logging — API key is NEVER logged
"""

from google import genai
from google.genai import errors
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# Exceptions
# ======================================================================


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


# ======================================================================
# Service
# ======================================================================


class EmbeddingService:
    """
    Generates vector embeddings using Gemini's embedding API.

    This class is the single place in the application that calls Gemini
    for embeddings.
    """

    def __init__(self):
        settings = get_settings()
        api_key = settings.require_gemini_key()
        self._client = genai.Client(api_key=api_key)
        self._model = settings.gemini_embedding_model
        self._batch_size = settings.embedding_batch_size

        logger.info(
            "embedding_service_initialized",
            model=self._model,
            batch_size=self._batch_size,
        )

    @property
    def vector_size(self) -> int:
        """Return the dimensionality of the embedding model."""
        sizes = {
            "gemini-embedding-2": 3072,
        }
        return sizes.get(self._model, 3072)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts in batches.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (same order as input).

        Raises:
            EmbeddingError: On persistent API failure.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * self._batch_size
            end = start + self._batch_size
            batch = texts[start:end]

            logger.debug(
                "embedding_batch",
                batch=batch_idx + 1,
                total_batches=total_batches,
                batch_size=len(batch),
            )

            batch_embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        logger.info(
            "embeddings_generated",
            total_texts=len(texts),
            model=self._model,
        )

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.
        """
        result = self.embed_texts([text])
        return result[0]

    @retry(
        retry=retry_if_exception_type(errors.APIError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        """
        Call the Gemini embedding API with automatic retry.

        Note: API key is intentionally excluded from all log messages.
        """
        try:
            response = self._client.models.embed_content(
                model=self._model,
                contents=batch,
            )
            # Response.embeddings is a list of ContentEmbedding, each with .values
            return [item.values for item in response.embeddings]

        except errors.APIError as exc:
            logger.error("gemini_api_error", detail=str(exc))
            raise EmbeddingError(f"Gemini API error: {exc}") from exc
        except Exception as exc:
            logger.exception("embedding_unexpected_error", error=str(exc))
            raise EmbeddingError(f"Unexpected embedding error: {exc}") from exc
