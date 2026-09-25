"""
app/rag/document_loader.py

PDF text extraction using PyMuPDF (fitz).

Handles:
- Normal PDFs
- Empty PDFs
- Corrupted PDFs (raises a clean DocumentLoadError)
- Scanned PDFs (detected and warned about)
- Page-level text extraction with metadata
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from app.core.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# Exceptions
# ======================================================================


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or is invalid."""


class EmptyDocumentError(DocumentLoadError):
    """Raised when the PDF contains no extractable text."""


class CorruptedDocumentError(DocumentLoadError):
    """Raised when the PDF file is corrupted or unreadable."""


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class PageContent:
    """Text and metadata extracted from a single PDF page."""

    page_number: int  # 1-indexed
    text: str
    char_count: int
    is_scanned: bool = False  # True if text is very sparse (likely scanned)


@dataclass
class LoadedDocument:
    """Complete extracted content from a single PDF file."""

    document_id: str
    filename: str
    page_count: int
    pages: list[PageContent]
    total_chars: int
    has_scanned_pages: bool

    @property
    def full_text(self) -> str:
        """Concatenate all page texts."""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


# ======================================================================
# Loader
# ======================================================================


class PDFDocumentLoader:
    """
    Loads and extracts text from PDF documents using PyMuPDF.

    Preserves page boundaries for downstream citation support.
    """

    # Pages with fewer chars than this are considered likely scanned
    SCANNED_THRESHOLD_CHARS = 50

    def __init__(self, min_page_chars: int = 10):
        self.min_page_chars = min_page_chars

    def load_from_bytes(self, content: bytes, document_id: str, filename: str) -> LoadedDocument:
        """
        Extract text from raw PDF bytes.

        Args:
            content: Raw PDF file bytes.
            document_id: Deterministic hash ID (from security.compute_document_id).
            filename: Safe display filename.

        Returns:
            LoadedDocument with page-level text and metadata.

        Raises:
            CorruptedDocumentError: If the PDF cannot be parsed.
            EmptyDocumentError: If the PDF has no extractable text.
        """
        logger.info("loading_pdf", document_id=document_id, filename=filename)

        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except fitz.FileDataError as exc:
            raise CorruptedDocumentError(f"Cannot parse '{filename}': {exc}") from exc
        except Exception as exc:
            raise CorruptedDocumentError(f"Unexpected error opening '{filename}': {exc}") from exc

        if doc.is_encrypted:
            doc.close()
            raise DocumentLoadError(
                f"'{filename}' is password-protected. Encrypted PDFs are not supported."
            )

        pages: list[PageContent] = []
        has_scanned = False

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text("text")  # plain text extraction
                char_count = len(text.strip())
                is_scanned = char_count < self.SCANNED_THRESHOLD_CHARS

                if is_scanned:
                    has_scanned = True
                    logger.warning(
                        "possible_scanned_page",
                        document_id=document_id,
                        page=page_idx + 1,
                        chars=char_count,
                    )

                pages.append(
                    PageContent(
                        page_number=page_idx + 1,
                        text=text,
                        char_count=char_count,
                        is_scanned=is_scanned,
                    )
                )
        except Exception as exc:
            doc.close()
            raise CorruptedDocumentError(f"Cannot extract pages from '{filename}': {exc}") from exc
        finally:
            if not doc.is_closed:
                doc.close()

        total_chars = sum(p.char_count for p in pages)

        if total_chars < self.min_page_chars:
            raise EmptyDocumentError(
                f"'{filename}' appears to be empty or contains no extractable text. "
                "If it is a scanned document, OCR support is required."
            )

        loaded = LoadedDocument(
            document_id=document_id,
            filename=filename,
            page_count=len(pages),
            pages=pages,
            total_chars=total_chars,
            has_scanned_pages=has_scanned,
        )

        logger.info(
            "pdf_loaded",
            document_id=document_id,
            filename=filename,
            pages=len(pages),
            total_chars=total_chars,
            has_scanned=has_scanned,
        )

        return loaded

    def load_from_path(self, file_path: Path, document_id: str) -> LoadedDocument:
        """Load a PDF from the filesystem."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = file_path.read_bytes()
        return self.load_from_bytes(
            content=content,
            document_id=document_id,
            filename=file_path.name,
        )
