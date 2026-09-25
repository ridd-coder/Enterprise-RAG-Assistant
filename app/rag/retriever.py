"""
app/rag/retriever.py

RAG retrieval layer.

Flow:
  User question
    → Embed question
    → Qdrant similarity search
    → Filter by minimum relevance threshold
    → Return ranked chunks + metadata

The retriever does NOT call the LLM — it is purely the "R" in RAG.
"""

from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import QdrantVectorStore, SearchResult

logger = get_logger(__name__)


@dataclass
class RetrievalContext:
    """
    The output of retrieval: ranked chunks ready to be passed to the LLM.
    """

    query: str
    chunks: list[SearchResult]
    total_retrieved: int
    passed_threshold: int
    min_score_used: float


class RAGRetriever:
    """
    Retrieves the most relevant document chunks for a user query.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
    ):
        self._embedder = embedding_service
        self._store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        document_ids: list[str] | None = None,
    ) -> RetrievalContext:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: Natural language question.
            top_k: Override for the number of results (defaults to config).
            min_score: Override for minimum relevance threshold (defaults to config).
            document_ids: Restrict search to specific documents.

        Returns:
            RetrievalContext containing the ranked chunks.
        """
        settings = get_settings()
        k = top_k or settings.top_k
        threshold = min_score if min_score is not None else settings.min_relevance_score

        logger.info(
            "retrieval_started",
            top_k=k,
            threshold=threshold,
            document_filter=document_ids,
        )

        # Step 1: Embed the query
        query_vector = self._embedder.embed_query(query)

        # Step 2: Search Qdrant (request more than k to account for threshold filtering)
        raw_results = self._store.search(
            query_vector=query_vector,
            top_k=k * 2,  # over-fetch then filter
            filter_document_ids=document_ids,
        )

        total_retrieved = len(raw_results)

        # Step 3: Apply relevance threshold
        filtered = [r for r in raw_results if r.score >= threshold]
        filtered = filtered[:k]  # take top-k after filtering

        logger.info(
            "retrieval_completed",
            total_retrieved=total_retrieved,
            passed_threshold=len(filtered),
            threshold=threshold,
        )

        return RetrievalContext(
            query=query,
            chunks=filtered,
            total_retrieved=total_retrieved,
            passed_threshold=len(filtered),
            min_score_used=threshold,
        )

    def has_sufficient_context(self, context: RetrievalContext) -> bool:
        """Return True if the context contains usable information."""
        return context.passed_threshold > 0
