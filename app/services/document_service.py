"""
app/services/document_service.py

Business logic layer for document management.

Separates HTTP concerns (FastAPI routes) from domain logic.
Persists document metadata in PostgreSQL (or an in-memory store for dev).
"""

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.database import DocumentRecord
from app.models.schemas import (
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.rag.pipeline import RAGPipeline

logger = get_logger(__name__)


class DocumentService:
    """Handles document upload, listing, and deletion."""

    def __init__(self, pipeline: RAGPipeline):
        self._pipeline = pipeline

    async def upload_document(
        self, content: bytes, filename: str, file_size: int, db: AsyncSession
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

        # Register metadata in DB
        db_doc = DocumentRecord(
            id=result.document_id,
            filename=result.filename,
            original_filename=filename,
            page_count=result.page_count,
            chunk_count=result.chunk_count,
            file_size_bytes=file_size,
        )
        db.add(db_doc)
        await db.commit()

        return result

    async def list_documents(self, db: AsyncSession) -> DocumentListResponse:
        """Return all registered documents."""
        result = await db.execute(select(DocumentRecord))
        db_docs = result.scalars().all()
        
        docs = [
            DocumentInfo(
                document_id=doc.id,
                filename=doc.filename,
                page_count=doc.page_count,
                chunk_count=doc.chunk_count,
                uploaded_at=doc.uploaded_at,
                file_size_bytes=doc.file_size_bytes,
            )
            for doc in db_docs
        ]
        return DocumentListResponse(documents=docs, total=len(docs))

    async def delete_document(self, document_id: str, db: AsyncSession) -> DocumentDeleteResponse:
        """Delete a document from the vector store and registry."""
        # Query the document
        result = await db.execute(select(DocumentRecord).where(DocumentRecord.id == document_id))
        db_doc = result.scalar_one_or_none()

        if not db_doc:
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found.",
            )

        # Delete vectors in Qdrant
        self._pipeline.delete_document_vectors(document_id)
        
        # Delete from Postgres
        await db.delete(db_doc)
        await db.commit()

        logger.info("document_deleted", document_id=document_id)

        return DocumentDeleteResponse(
            document_id=document_id,
            message=f"Document '{document_id}' and all its vectors have been deleted.",
        )

    async def get_document_count(self, db: AsyncSession) -> int:
        result = await db.execute(select(func.count()).select_from(DocumentRecord))
        return result.scalar() or 0
