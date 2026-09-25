"""
app/rag/generator.py

LLM answer generation service using Google Gemini's chat completion API.

All LLM calls in the application go through this single class.
"""

import time
from dataclasses import dataclass

from google import genai
from google.genai import errors
from google.genai import types
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
    tokens_used: int | None
    latency_ms: int


# ======================================================================
# Generator
# ======================================================================

# Sentinel string that indicates no usable context was retrieved
NO_CONTEXT_ANSWER = (
    "I could not find sufficient information in the uploaded documents " "to answer this question."
)


class LLMGenerator:
    """
    Generates answers to user questions using Gemini API.
    """

    def __init__(self):
        settings = get_settings()
        api_key = settings.require_gemini_key()
        self._client = genai.Client(api_key=api_key)
        self._model = settings.gemini_model
        self._max_tokens = settings.gemini_max_tokens
        self._temperature = settings.gemini_temperature

        logger.info(
            "llm_generator_initialized",
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

    def generate(
        self,
        question: str,
        chunks: list[SearchResult],
        conversation_history: list[dict] | None = None,
    ) -> GenerationResult:
        """
        Generate an answer grounded in the retrieved chunks.
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

        response = self._call_gemini_with_retry(messages)

        answer = response.text or NO_CONTEXT_ANSWER
        
        tokens_used = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_used = response.usage_metadata.total_token_count

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
        retry=retry_if_exception_type(errors.APIError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _call_gemini_with_retry(self, messages: list):
        """Call the Gemini API with retry on transient errors."""
        try:
            # Map OpenAI format to Gemini format
            system_instruction = None
            gemini_contents = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                else:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
            
            config = types.GenerateContentConfig(
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
            )
            if system_instruction:
                config.system_instruction = system_instruction
                
            return self._client.models.generate_content(
                model=self._model,
                contents=gemini_contents,
                config=config,
            )
        except errors.APIError as exc:
            logger.error("gemini_api_error_generation", detail=str(exc))
            raise GenerationError(f"Gemini API error: {exc}") from exc
        except Exception as exc:
            logger.exception("gemini_unexpected_error", error=str(exc))
            raise GenerationError(f"Unexpected generation error: {exc}") from exc
