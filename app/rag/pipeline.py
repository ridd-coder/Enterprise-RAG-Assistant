"""
app/rag/pipeline.py

The RAG pipeline orchestrator.

This module ties together:
  document loading → chunking → embedding → Qdrant insertion

and:

  user question → embedding → retrieval → LLM generation → structured response

It is the ONLY place that combines multiple RAG components.
"""

import time
import uuid

from app.core.logging import get_logger
from app.core.security import compute_document_id, sanitise_filename
from app.models.schemas import (
    ChatResponse,
    ConversationMessage,
    DocumentUploadResponse,
    RetrievedChunk,
    SourceReference,
)
from app.rag.chunker import DocumentChunker
from app.rag.document_loader import PDFDocumentLoader
from app.rag.embeddings import EmbeddingService
from app.rag.generator import LLMGenerator
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import QdrantVectorStore

logger = get_logger(__name__)


# ======================================================================
# In-memory conversation store (replace with Redis/DB for multi-user)
# ======================================================================

_CONVERSATION_STORE: dict[str, list[ConversationMessage]] = {}
_MAX_HISTORY_TURNS = 10  # Keep last 10 user+assistant pairs


# ======================================================================
# Pipeline
# ======================================================================


class RAGPipeline:
    """
    Orchestrates the complete RAG ingestion and query pipelines.

    Instantiated once at application startup and shared via dependency injection.
    """

    def __init__(self):
        # --- Document processing ---
        self._loader = PDFDocumentLoader()
        self._chunker = DocumentChunker()

        # --- AI components ---
        self._embedder = EmbeddingService()
        self._vector_store = QdrantVectorStore(vector_size=self._embedder.vector_size)
        self._retriever = RAGRetriever(
            embedding_service=self._embedder,
            vector_store=self._vector_store,
        )
        self._generator = LLMGenerator()

        # Initialise Qdrant collection on startup
        self._vector_store.initialize_collection()

        logger.info("rag_pipeline_initialized")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        content: bytes,
        filename: str,
    ) -> DocumentUploadResponse:
        """
        Full ingestion pipeline: PDF bytes → Qdrant.

        Steps:
        1. Compute deterministic document ID
        2. Load and extract PDF text
        3. Chunk the document
        4. Generate embeddings in batches
        5. Upsert into Qdrant
        6. Return metadata for persistence
        """
        start = time.monotonic()

        safe_filename = sanitise_filename(filename)
        document_id = compute_document_id(content, safe_filename)

        logger.info(
            "ingestion_started",
            document_id=document_id,
            filename=safe_filename,
        )

        # Step 1: Load PDF
        loaded_doc = self._loader.load_from_bytes(
            content=content,
            document_id=document_id,
            filename=safe_filename,
        )

        # Step 2: Chunk
        chunks = self._chunker.chunk_document(loaded_doc)

        if not chunks:
            logger.warning(
                "no_chunks_produced",
                document_id=document_id,
                filename=safe_filename,
            )

        # Step 3: Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_texts(texts)

        # Step 4: Upsert to Qdrant
        inserted = self._vector_store.upsert_chunks(chunks, embeddings)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "ingestion_completed",
            document_id=document_id,
            filename=safe_filename,
            chunks=inserted,
            latency_ms=elapsed_ms,
        )

        return DocumentUploadResponse(
            document_id=document_id,
            filename=safe_filename,
            page_count=loaded_doc.page_count,
            chunk_count=inserted,
            status="ingested",
            message=(
                f"Successfully ingested '{safe_filename}' "
                f"({loaded_doc.page_count} pages, {inserted} chunks)."
            ),
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        conversation_id: str | None = None,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> ChatResponse:
        """
        Full RAG query pipeline: question → answer with sources.

        Steps:
        1. Retrieve conversation history
        2. Embed question + retrieve chunks
        3. Check relevance threshold (hallucination guard)
        4. Generate answer
        5. Store turn in conversation history
        6. Return structured ChatResponse
        """
        start = time.monotonic()

        # Manage conversation ID
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        history = _CONVERSATION_STORE.get(conversation_id, [])
        history_dicts = [{"role": m.role, "content": m.content} for m in history]

        logger.info(
            "query_started",
            conversation_id=conversation_id,
            question_length=len(question),
        )

        # Retrieval
        retrieval_ctx = self._retriever.retrieve(
            query=question,
            top_k=top_k,
            document_ids=document_ids,
        )

        chunks = retrieval_ctx.chunks

        # Generation (with hallucination guard)
        gen_result = self._generator.generate(
            question=question,
            chunks=chunks,
            conversation_history=history_dicts,
        )

        # Build source references
        sources: list[SourceReference] = []
        retrieved_chunks: list[RetrievedChunk] = []

        seen_sources = set()
        for chunk in chunks:
            source_key = (chunk.document_id, chunk.page_number)
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append(
                    SourceReference(
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        relevance_score=round(chunk.score, 4),
                        text_snippet=chunk.text[:200],
                    )
                )
            retrieved_chunks.append(
                RetrievedChunk(
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    score=round(chunk.score, 4),
                    text=chunk.text,
                )
            )

        total_latency_ms = int((time.monotonic() - start) * 1000)

        # Persist conversation history
        history.append(ConversationMessage(role="user", content=question))
        history.append(ConversationMessage(role="assistant", content=gen_result.answer))
        # Trim to max history window
        max_messages = _MAX_HISTORY_TURNS * 2
        _CONVERSATION_STORE[conversation_id] = history[-max_messages:]

        logger.info(
            "query_completed",
            conversation_id=conversation_id,
            chunks_retrieved=len(chunks),
            latency_ms=total_latency_ms,
            tokens=gen_result.tokens_used,
        )

        return ChatResponse(
            answer=gen_result.answer,
            conversation_id=conversation_id,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
            chunks_retrieved=len(chunks),
            latency_ms=total_latency_ms,
            model_used=gen_result.model_used,
            tokens_used=gen_result.tokens_used,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def delete_document_vectors(self, document_id: str) -> None:
        """Remove all Qdrant vectors for a document."""
        self._vector_store.delete_document(document_id)

    def vector_store_health(self) -> dict:
        """Return health info from Qdrant."""
        return self._vector_store.health_check()
