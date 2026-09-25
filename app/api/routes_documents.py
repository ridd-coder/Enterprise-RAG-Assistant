"""
app/api/routes_documents.py

FastAPI routes for document management:

  POST   /api/documents/upload
  GET    /api/documents
  DELETE /api/documents/{document_id}
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.logging import get_logger
from app.core.security import validate_file_size, validate_uploaded_file
from app.models.schemas import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.rag.document_loader import CorruptedDocumentError, DocumentLoadError, EmptyDocumentError
from app.services.document_service import DocumentService
from app.api.deps import get_db_session
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)
router = APIRouter(prefix="/api/documents", tags=["Documents"])


def get_document_service() -> DocumentService:
    """Dependency — returns the shared DocumentService from app state."""
    from app.main import get_app_state

    state = get_app_state()
    service = state.get("document_service")
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service is unavailable. Check that Qdrant is running and OPENAI_API_KEY is set.",
        )
    return service


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF document",
)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
    db: AsyncSession = Depends(get_db_session),
) -> DocumentUploadResponse:
    """
    Upload a PDF document for ingestion into the RAG knowledge base.

    - Validates file type (PDF only) and size (<= MAX_FILE_SIZE_MB).
    - Extracts text, chunks, embeds, and stores in Qdrant.
    - Returns document ID, page count, and chunk count.
    """
    # Security: validate extension BEFORE reading content
    validate_uploaded_file(file)

    content = await file.read()

    # Security: validate size after reading
    validate_file_size(content)

    logger.info(
        "upload_request",
        filename=file.filename,
        size_bytes=len(content),
    )

    try:
        result = await service.upload_document(
            content=content,
            filename=file.filename or "unknown.pdf",
            file_size=len(content),
            db=db,
        )
        return result

    except EmptyDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except CorruptedDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except DocumentLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("upload_unexpected_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion failed. Please try again.",
        ) from exc


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
)
async def list_documents(
    service: DocumentService = Depends(get_document_service),
    db: AsyncSession = Depends(get_db_session),
) -> DocumentListResponse:
    """Return metadata for all documents currently in the knowledge base."""
    return await service.list_documents(db)


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document and its vectors",
)
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
    db: AsyncSession = Depends(get_db_session),
) -> DocumentDeleteResponse:
    """
    Remove a document and all its associated vectors from Qdrant.
    """
    return await service.delete_document(document_id, db)
