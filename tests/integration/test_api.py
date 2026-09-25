"""tests/integration/test_api.py

Integration tests for FastAPI endpoints.

These tests use TestClient (no real Qdrant/OpenAI required)
by mocking the RAGPipeline and services.
"""

import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.routes_chat import get_chat_service
from app.api.routes_documents import get_document_service
from app.api.routes_health import get_rag_pipeline
from app.main import app, get_app_state
from app.models.schemas import (
    ChatResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    SourceReference,
)


@pytest.fixture
def mock_pipeline():
    """Mock the RAGPipeline so tests don't need Qdrant/OpenAI."""
    pipeline = MagicMock()
    pipeline.vector_store_health.return_value = {
        "status": "healthy",
        "collection": "enterprise_documents",
        "vectors_count": 42,
        "host": "localhost",
    }
    return pipeline


@pytest.fixture
def mock_doc_service(mock_pipeline):
    from app.services.document_service import DocumentService

    svc = MagicMock(spec=DocumentService)
    svc.upload_document.return_value = DocumentUploadResponse(
        document_id="abc123",
        filename="test.pdf",
        page_count=5,
        chunk_count=20,
        status="ingested",
        message="Ingested successfully.",
    )
    svc.list_documents.return_value = DocumentListResponse(documents=[], total=0)
    svc.get_document_count.return_value = 0
    return svc


@pytest.fixture
def mock_chat_service():
    from app.services.chat_service import ChatService

    svc = MagicMock(spec=ChatService)
    svc.ask.return_value = ChatResponse(
        answer="Employees receive 20 days annual leave.",
        conversation_id="conv-123",
        sources=[
            SourceReference(
                document_id="abc123",
                filename="leave_policy.pdf",
                page_number=4,
                chunk_index=0,
                relevance_score=0.91,
                text_snippet="Employees are entitled to 20 days...",
            )
        ],
        retrieved_chunks=[],
        chunks_retrieved=1,
        latency_ms=450,
        model_used="gpt-4o-mini",
        tokens_used=120,
    )
    return svc


@pytest.fixture
def client(mock_pipeline, mock_doc_service, mock_chat_service):
    """TestClient with mocked app state and dependency overrides."""
    state = get_app_state()
    state["pipeline"] = mock_pipeline
    state["document_service"] = mock_doc_service
    state["chat_service"] = mock_chat_service
    state["start_time"] = 0
    state["total_queries"] = 0
    state["avg_latency_ms"] = 0.0

    app.dependency_overrides[get_document_service] = lambda: mock_doc_service
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_pipeline

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_vector_db_health(self, client):
        resp = client.get("/api/health/vector-db")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["vectors_count"] == 42


class TestDocumentEndpoints:
    def test_list_documents_empty(self, client):
        resp = client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["documents"] == []

    def test_upload_invalid_extension(self, client):
        """Non-PDF files must be rejected."""
        fake_file = io.BytesIO(b"not a pdf")
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("malicious.exe", fake_file, "application/octet-stream")},
        )
        assert resp.status_code == 415

    def test_upload_pdf_success(self, client):
        """Valid upload returns 201 with document metadata."""
        # Minimal valid PDF bytes
        fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
        )
        # The mock returns success
        assert resp.status_code == 201
        data = resp.json()
        assert data["document_id"] == "abc123"
        assert data["chunk_count"] == 20


class TestChatEndpoints:
    def test_chat_returns_answer(self, client):
        resp = client.post(
            "/api/chat",
            json={"question": "What is the leave policy?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "latency_ms" in data
        assert len(data["sources"]) == 1

    def test_chat_empty_question_rejected(self, client):
        resp = client.post("/api/chat", json={"question": ""})
        assert resp.status_code == 422

    def test_chat_question_too_long_rejected(self, client):
        resp = client.post("/api/chat", json={"question": "x" * 2001})
        assert resp.status_code == 422
