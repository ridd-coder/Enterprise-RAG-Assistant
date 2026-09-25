"""
app/rag/embeddings.py

Embedding service backed by OpenAI's text-embedding API.

Features:
- Decoupled from the rest of the application (swap model via config)
- Batch embedding with configurable batch size
- Automatic retry with exponential back-off (tenacity)
- Clean error logging — API key is NEVER logged
"""

import time
from typing import List

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
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
    Generates vector embeddings using OpenAI's embedding API.

    This class is the single place in the application that calls OpenAI
    for embeddings.  Swap the model by changing OPENAI_EMBEDDING_MODEL.
    """

    def __init__(self):
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model
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
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return sizes.get(self._model, 1536)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
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

        all_embeddings: List[List[float]] = []
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

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.

        Slightly different from document embedding in OpenAI's API
        (query_embedding vs document_embedding) but the model is the same.
        """
        result = self.embed_texts([text])
        return result[0]

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        """
        Call the OpenAI embedding API with automatic retry.

        Note: API key is intentionally excluded from all log messages.
        """
        try:
            response = self._client.embeddings.create(
                input=batch,
                model=self._model,
            )
            # Sort by index to guarantee order matches input
            sorted_data = sorted(response.data, key=lambda e: e.index)
            return [item.embedding for item in sorted_data]

        except RateLimitError as exc:
            logger.warning("openai_rate_limited", detail=str(exc))
            raise
        except APIConnectionError as exc:
            logger.warning("openai_connection_error", detail=str(exc))
            raise
        except APIStatusError as exc:
            # Non-retryable API errors
            logger.error("openai_api_error", status=exc.status_code, detail=exc.message)
            raise EmbeddingError(
                f"OpenAI embedding API returned status {exc.status_code}"
            ) from exc
        except Exception as exc:
            logger.exception("embedding_unexpected_error", error=str(exc))
            raise EmbeddingError(f"Unexpected embedding error: {exc}") from exc
