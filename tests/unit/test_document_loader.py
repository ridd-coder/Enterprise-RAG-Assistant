"""tests/unit/test_document_loader.py

Unit tests for the PDF document loader.
"""

import pytest

from app.rag.document_loader import (
    CorruptedDocumentError,
    PDFDocumentLoader,
)


class TestPDFDocumentLoader:
    def setup_method(self):
        self.loader = PDFDocumentLoader()

    def test_corrupted_pdf_raises(self):
        """Corrupted text bytes should raise CorruptedDocumentError."""
        with pytest.raises(CorruptedDocumentError):
            self.loader.load_from_bytes(
                content=b"this is not a pdf",
                document_id="bad123",
                filename="corrupt.pdf",
            )

    def test_empty_bytes_raises(self):
        """Empty bytes should raise CorruptedDocumentError."""
        with pytest.raises(CorruptedDocumentError):
            self.loader.load_from_bytes(
                content=b"",
                document_id="empty123",
                filename="empty.pdf",
            )

    def test_non_pdf_magic_bytes_raises(self):
        """Non-PDF bytes (PNG header) should raise CorruptedDocumentError."""
        with pytest.raises(CorruptedDocumentError):
            self.loader.load_from_bytes(
                content=b"\x89PNG\r\n\x1a\n",  # PNG magic bytes
                document_id="png123",
                filename="image.pdf",
            )
