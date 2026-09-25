"""tests/unit/test_chunker.py

Unit tests for the DocumentChunker.
"""

import pytest
from app.rag.chunker import DocumentChunk, DocumentChunker
from app.rag.document_loader import LoadedDocument, PageContent


def make_loaded_doc(pages_text: list[str], doc_id: str = "test123") -> LoadedDocument:
    pages = [
        PageContent(page_number=i + 1, text=t, char_count=len(t))
        for i, t in enumerate(pages_text)
    ]
    return LoadedDocument(
        document_id=doc_id,
        filename="test.pdf",
        page_count=len(pages),
        pages=pages,
        total_chars=sum(len(t) for t in pages_text),
        has_scanned_pages=False,
    )


class TestDocumentChunker:
    def test_basic_chunking(self):
        """A non-empty document should produce at least one chunk."""
        doc = make_loaded_doc(["This is a short test page."])
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1

    def test_chunk_metadata_correct(self):
        """Chunks must carry the correct document_id, filename, page_number."""
        doc = make_loaded_doc(["Page one content here."], doc_id="doc_abc")
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk_document(doc)

        assert chunks[0].document_id == "doc_abc"
        assert chunks[0].filename == "test.pdf"
        assert chunks[0].page_number == 1

    def test_page_number_preserved(self):
        """Each chunk must reference the page it came from."""
        doc = make_loaded_doc(["Page 1 text.", "Page 2 text."])
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk_document(doc)

        page_numbers = {c.page_number for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_empty_pages_skipped(self):
        """Empty pages should not produce chunks."""
        doc = make_loaded_doc(["", "   ", "Actual content here."])
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk_document(doc)

        # All chunks must come from page 3
        assert all(c.page_number == 3 for c in chunks)

    def test_chunk_id_is_unique(self):
        """All chunk IDs within a document must be unique."""
        long_text = "word " * 500
        doc = make_loaded_doc([long_text])
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_document(doc)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_to_payload_keys(self):
        """to_payload() must include all required metadata fields."""
        doc = make_loaded_doc(["Test content."])
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=0)
        chunk = chunker.chunk_document(doc)[0]
        payload = chunk.to_payload()

        required_keys = {
            "chunk_id", "document_id", "filename",
            "page_number", "chunk_index", "text", "char_count",
        }
        assert required_keys.issubset(set(payload.keys()))

    def test_large_document_chunked(self):
        """A document larger than chunk_size should produce multiple chunks."""
        long_text = "This is sentence number {}. ".format(1) * 200
        doc = make_loaded_doc([long_text])
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1
