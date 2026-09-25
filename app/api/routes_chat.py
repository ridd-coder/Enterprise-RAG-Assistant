"""
app/api/routes_chat.py

FastAPI routes for the chat interface:

  POST /api/chat
  GET  /api/metrics
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.logging import get_logger
from app.models.schemas import ChatRequest, ChatResponse, MetricsResponse
from app.rag.generator import GenerationError
from app.rag.vector_store import VectorStoreError
from app.services.chat_service import ChatService

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["Chat"])


def get_chat_service() -> ChatService:
    """Dependency — returns the shared ChatService from app state."""
    from app.main import get_app_state

    state = get_app_state()
    service = state.get("chat_service")
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is unavailable. Check that Qdrant is running and OPENAI_API_KEY is set.",
        )
    return service


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question about uploaded documents",
)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """
    Ask a natural-language question against the document knowledge base.

    - Retrieves the most relevant document chunks from Qdrant.
    - Generates a grounded answer using the LLM.
    - Returns the answer, source citations, and latency metrics.
    - If no relevant context is found, returns an explicit "not found" message.
    """
    try:
        return service.ask(request)
    except VectorStoreError as exc:
        logger.error("chat_vector_store_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The vector database is currently unavailable.",
        ) from exc
    except GenerationError as exc:
        logger.error("chat_generation_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The LLM service returned an error. Please try again.",
        ) from exc
    except Exception as exc:
        logger.exception("chat_unexpected_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        ) from exc


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get application usage metrics",
)
async def get_metrics() -> MetricsResponse:
    """Return basic usage metrics for the dashboard."""
    import time

    from app.main import get_app_state

    state = get_app_state()
    uptime = time.monotonic() - state.get("start_time", time.monotonic())
    doc_service = state.get("document_service")
    total_docs = doc_service.get_document_count() if doc_service else 0

    return MetricsResponse(
        total_documents=total_docs,
        total_chunks=0,  # extend: query Qdrant for total count
        total_queries=state.get("total_queries", 0),
        avg_query_latency_ms=state.get("avg_latency_ms", 0.0),
        uptime_seconds=uptime,
    )
