"""
app/services/document_service.py

Business logic layer for document management.

Separates HTTP concerns (FastAPI routes) from domain logic.
Persists document metadata in PostgreSQL (or an in-memory store for dev).
"""

from datetime import datetime

from app.core.logging import get_logger
from app.models.schemas import (
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.rag.pipeline import RAGPipeline

logger = get_logger(__name__)

# Simple in-memory store (replace with PostgreSQL in production)
_DOCUMENT_REGISTRY: dict[str, DocumentInfo] = {}


class DocumentService:
    """Handles document upload, listing, and deletion."""

    def __init__(self, pipeline: RAGPipeline):
        self._pipeline = pipeline

    def upload_document(
        self, content: bytes, filename: str, file_size: int
    ) -> DocumentUploadResponse:
        """
        Validate, ingest, and register a document.

        Raises:
            DocumentLoadError: If the PDF is invalid/empty.
            HTTPException: For file size violations (raised in route).
        """
        logger.info("document_service_upload", filename=filename, size=file_size)

        # Ingest via RAG pipeline
        result = self._pipeline.ingest_document(
            content=content,
            filename=filename,
        )

        # Register metadata
        _DOCUMENT_REGISTRY[result.document_id] = DocumentInfo(
            document_id=result.document_id,
            filename=result.filename,
            page_count=result.page_count,
            chunk_count=result.chunk_count,
            uploaded_at=datetime.utcnow(),
            file_size_bytes=file_size,
        )

        return result

    def list_documents(self) -> DocumentListResponse:
        """Return all registered documents."""
        docs = list(_DOCUMENT_REGISTRY.values())
        return DocumentListResponse(documents=docs, total=len(docs))

    def delete_document(self, document_id: str) -> DocumentDeleteResponse:
        """Delete a document from the vector store and registry."""
        if document_id not in _DOCUMENT_REGISTRY:
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found.",
            )

        self._pipeline.delete_document_vectors(document_id)
        del _DOCUMENT_REGISTRY[document_id]

        logger.info("document_deleted", document_id=document_id)

        return DocumentDeleteResponse(
            document_id=document_id,
            message=f"Document '{document_id}' and all its vectors have been deleted.",
        )

    def get_document_count(self) -> int:
        return len(_DOCUMENT_REGISTRY)
