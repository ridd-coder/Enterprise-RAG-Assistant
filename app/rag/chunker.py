"""
app/rag/chunker.py

Page-aware, configurable text chunking with metadata preservation.

Strategy:
1. Process each page individually to keep page numbers accurate.
2. Use LangChain's RecursiveCharacterTextSplitter within each page.
3. Attach document_id, filename, page_number, chunk_index to every chunk.
4. Avoid splitting in the middle of sentences where possible.
"""

from dataclasses import dataclass, field
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.document_loader import LoadedDocument, PageContent

logger = get_logger(__name__)


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class DocumentChunk:
    """
    A single text chunk with full provenance metadata.

    The chunk_id is globally unique and deterministic.
    """

    chunk_id: str           # "{document_id}_p{page}_c{index}"
    document_id: str
    filename: str
    page_number: int
    chunk_index: int        # index within the whole document
    page_chunk_index: int   # index within the page
    text: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)

    def to_payload(self) -> dict:
        """Serialize to Qdrant-compatible metadata payload."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "page_chunk_index": self.page_chunk_index,
            "text": self.text,
            "char_count": self.char_count,
        }


# ======================================================================
# Chunker
# ======================================================================


class DocumentChunker:
    """
    Splits a LoadedDocument into overlapping chunks while preserving page metadata.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            is_separator_regex=False,
        )

    def chunk_document(self, document: LoadedDocument) -> List[DocumentChunk]:
        """
        Split a LoadedDocument into a flat list of DocumentChunks.

        Each chunk knows its document, filename, and exact page number.
        """
        all_chunks: List[DocumentChunk] = []
        global_chunk_index = 0

        for page in document.pages:
            text = page.text.strip()
            if not text:
                logger.debug(
                    "skipping_empty_page",
                    document_id=document.document_id,
                    page=page.page_number,
                )
                continue

            page_texts = self._splitter.split_text(text)

            for local_idx, chunk_text in enumerate(page_texts):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                chunk_id = (
                    f"{document.document_id}"
                    f"_p{page.page_number}"
                    f"_c{local_idx}"
                )

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    filename=document.filename,
                    page_number=page.page_number,
                    chunk_index=global_chunk_index,
                    page_chunk_index=local_idx,
                    text=chunk_text,
                )
                all_chunks.append(chunk)
                global_chunk_index += 1

        logger.info(
            "document_chunked",
            document_id=document.document_id,
            filename=document.filename,
            total_chunks=len(all_chunks),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        return all_chunks
