"""
app/rag/generator.py

LLM answer generation service using OpenAI's chat completion API.

All LLM calls in the application go through this single class.
Do not scatter OpenAI calls elsewhere.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.prompt import build_messages
from app.rag.vector_store import SearchResult

logger = get_logger(__name__)


# ======================================================================
# Exceptions
# ======================================================================


class GenerationError(Exception):
    """Raised when LLM generation fails irrecoverably."""


# ======================================================================
# Result
# ======================================================================


@dataclass
class GenerationResult:
    """Output from the LLM generator."""

    answer: str
    model_used: str
    tokens_used: Optional[int]
    latency_ms: int


# ======================================================================
# Generator
# ======================================================================

# Sentinel string that indicates no usable context was retrieved
NO_CONTEXT_ANSWER = (
    "I could not find sufficient information in the uploaded documents "
    "to answer this question."
)


class LLMGenerator:
    """
    Generates answers to user questions using OpenAI's chat completion API.

    Enforces grounding: if no context is available, returns the
    NO_CONTEXT_ANSWER sentinel without calling the LLM.
    """

    def __init__(self):
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._max_tokens = settings.openai_max_tokens
        self._temperature = settings.openai_temperature

        logger.info(
            "llm_generator_initialized",
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

    def generate(
        self,
        question: str,
        chunks: List[SearchResult],
        conversation_history: Optional[List[dict]] = None,
    ) -> GenerationResult:
        """
        Generate an answer grounded in the retrieved chunks.

        If no chunks are provided (retrieval returned nothing above threshold),
        return the NO_CONTEXT_ANSWER without hitting the LLM.

        Args:
            question: User's natural language question.
            chunks: Retrieved document chunks from Qdrant.
            conversation_history: Previous turns for conversational context.

        Returns:
            GenerationResult with the answer and metadata.
        """
        start_ms = time.monotonic() * 1000

        # Hallucination safeguard: no context → no LLM call
        if not chunks:
            elapsed = int(time.monotonic() * 1000 - start_ms)
            logger.info(
                "generation_skipped_no_context",
                question_length=len(question),
                latency_ms=elapsed,
            )
            return GenerationResult(
                answer=NO_CONTEXT_ANSWER,
                model_used=self._model,
                tokens_used=0,
                latency_ms=elapsed,
            )

        messages = build_messages(
            question=question,
            chunks=chunks,
            conversation_history=conversation_history,
        )

        logger.info(
            "generation_started",
            model=self._model,
            chunks=len(chunks),
            messages=len(messages),
        )

        response = self._call_openai_with_retry(messages)

        answer = response.choices[0].message.content or NO_CONTEXT_ANSWER
        tokens_used = response.usage.total_tokens if response.usage else None
        elapsed = int(time.monotonic() * 1000 - start_ms)

        logger.info(
            "generation_completed",
            model=self._model,
            tokens=tokens_used,
            latency_ms=elapsed,
        )

        return GenerationResult(
            answer=answer,
            model_used=self._model,
            tokens_used=tokens_used,
            latency_ms=elapsed,
        )

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _call_openai_with_retry(self, messages: list):
        """Call the OpenAI chat completion API with retry on transient errors."""
        try:
            return self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except RateLimitError as exc:
            logger.warning("openai_rate_limited_generation")
            raise
        except APIConnectionError as exc:
            logger.warning("openai_connection_error_generation")
            raise
        except APIStatusError as exc:
            logger.error(
                "openai_api_error_generation",
                status=exc.status_code,
            )
            raise GenerationError(
                f"OpenAI API returned status {exc.status_code}: {exc.message}"
            ) from exc
