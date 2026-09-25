"""
app/services/chat_service.py

Business logic layer for the chat/Q&A interface.
"""

from app.core.logging import get_logger
from app.models.schemas import ChatRequest, ChatResponse
from app.rag.pipeline import RAGPipeline

logger = get_logger(__name__)


class ChatService:
    """Handles question routing and response assembly."""

    def __init__(self, pipeline: RAGPipeline):
        self._pipeline = pipeline

    def ask(self, request: ChatRequest) -> ChatResponse:
        """
        Route a user question through the RAG pipeline and return the response.
        """
        logger.info(
            "chat_service_ask",
            conversation_id=request.conversation_id,
            question_len=len(request.question),
        )

        response = self._pipeline.query(
            question=request.question,
            conversation_id=request.conversation_id,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )

        return response
