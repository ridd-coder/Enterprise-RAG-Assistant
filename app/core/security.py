"""
app/core/security.py

Security utilities:
- File validation (type, size)
- CORS configuration
- Input sanitisation
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def validate_uploaded_file(file: UploadFile) -> None:
    """
    Validate that the uploaded file meets security requirements.

    Raises HTTPException on violation.
    """
    settings = get_settings()

    # --- Extension check ---
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File has no name.",
        )

    extension = Path(file.filename).suffix.lstrip(".").lower()
    if extension not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '.{extension}' is not supported. Allowed: {settings.allowed_extensions}",
        )


def validate_file_size(content: bytes) -> None:
    """
    Validate file content size against the configured limit.

    Raises HTTPException if exceeded.
    """
    settings = get_settings()
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size exceeds the {settings.max_file_size_mb} MB limit. "
                f"Actual size: {len(content) / (1024 * 1024):.2f} MB"
            ),
        )


def compute_document_id(content: bytes, filename: str) -> str:
    """
    Generate a deterministic SHA-256 document ID from content + filename.

    This prevents duplicate ingestion of the same document.
    """
    hasher = hashlib.sha256()
    hasher.update(filename.encode("utf-8"))
    hasher.update(content)
    return hasher.hexdigest()


def sanitise_filename(filename: str) -> str:
    """
    Remove path traversal sequences and unsafe characters from a filename.
    """
    # Strip directory components
    filename = os.path.basename(filename)
    # Replace anything that isn't alphanumeric, dash, underscore, or dot
    safe = "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_" for c in filename
    )
    return safe or "uploaded_file"
