"""tests/unit/test_retriever.py

Unit tests for RAG retriever and hallucination/threshold filtering.
"""

from unittest.mock import MagicMock

import pytest

from app.rag.retriever import RAGRetriever, RetrievalContext
from app.rag.vector_store import SearchResult


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    return embedder


@pytest.fixture
def mock_store():
    store = MagicMock()
    return store


def make_search_result(score: float, doc_id: str = "doc1", page: int = 1, text: str = "sample"):
    payload = {
        "document_id": doc_id,
        "filename": f"{doc_id}.pdf",
        "page_number": page,
        "chunk_index": 0,
        "text": text,
    }
    return SearchResult(payload=payload, score=score, point_id="pt1")


class TestRAGRetriever:
    def test_retrieve_filters_below_threshold(self, mock_embedder, mock_store):
        """Chunks with similarity below min_score must be excluded."""
        mock_store.search.return_value = [
            make_search_result(score=0.85, doc_id="doc1"),
            make_search_result(score=0.45, doc_id="doc2"),  # below 0.50 threshold
            make_search_result(score=0.60, doc_id="doc3"),
        ]

        retriever = RAGRetriever(mock_embedder, mock_store)
        ctx = retriever.retrieve("What is the policy?", top_k=5, min_score=0.50)

        assert isinstance(ctx, RetrievalContext)
        assert ctx.total_retrieved == 3
        assert ctx.passed_threshold == 2
        assert len(ctx.chunks) == 2
        assert all(c.score >= 0.50 for c in ctx.chunks)

    def test_retrieve_respects_top_k(self, mock_embedder, mock_store):
        """Retriever must limit results to top_k after threshold filtering."""
        mock_store.search.return_value = [
            make_search_result(score=0.95),
            make_search_result(score=0.90),
            make_search_result(score=0.85),
            make_search_result(score=0.80),
        ]

        retriever = RAGRetriever(mock_embedder, mock_store)
        ctx = retriever.retrieve("Query", top_k=2, min_score=0.50)

        assert len(ctx.chunks) == 2
        assert ctx.chunks[0].score == 0.95
        assert ctx.chunks[1].score == 0.90

    def test_has_sufficient_context(self, mock_embedder, mock_store):
        retriever = RAGRetriever(mock_embedder, mock_store)

        ctx_empty = RetrievalContext(
            query="q", chunks=[], total_retrieved=0, passed_threshold=0, min_score_used=0.5
        )
        assert retriever.has_sufficient_context(ctx_empty) is False

        ctx_with_data = RetrievalContext(
            query="q",
            chunks=[make_search_result(score=0.9)],
            total_retrieved=1,
            passed_threshold=1,
            min_score_used=0.5,
        )
        assert retriever.has_sufficient_context(ctx_with_data) is True
