"""
app/models/schemas.py

Pydantic v2 request/response schemas for all API endpoints.
These are the public contract — no internal implementation details here.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ======================================================================
# Shared / Base
# ======================================================================


class HealthStatus(BaseModel):
    status: str
    version: str = "1.0.0"
    environment: str


class VectorDBHealth(BaseModel):
    status: str
    collection: str
    vectors_count: int
    host: str


# ======================================================================
# Documents
# ======================================================================


class DocumentUploadResponse(BaseModel):
    """Returned after a successful document upload and ingestion."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    status: str = "ingested"
    message: str


class DocumentInfo(BaseModel):
    """Summary of a stored document."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    uploaded_at: datetime
    file_size_bytes: int


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class DocumentDeleteResponse(BaseModel):
    document_id: str
    message: str


# ======================================================================
# Chat
# ======================================================================


class ChatRequest(BaseModel):
    """Incoming chat question from the user."""

    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(default=None)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Restrict retrieval to specific document IDs",
    )

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be blank.")
        return v.strip()


class SourceReference(BaseModel):
    """A single source document chunk cited in the answer."""

    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    relevance_score: float
    text_snippet: str = Field(description="First 200 chars of the chunk for UI display")


class RetrievedChunk(BaseModel):
    """Full chunk detail returned for inspection / evaluation."""

    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    score: float
    text: str


class ChatResponse(BaseModel):
    """Full response to a chat question."""

    answer: str
    conversation_id: str
    sources: List[SourceReference]
    retrieved_chunks: List[RetrievedChunk]
    chunks_retrieved: int
    latency_ms: int
    model_used: str
    tokens_used: Optional[int] = None


# ======================================================================
# Conversation history (internal helpers)
# ======================================================================


class ConversationMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ======================================================================
# Evaluation
# ======================================================================


class EvaluationQuestion(BaseModel):
    question: str
    expected_answer: str
    expected_sources: List[str]


class EvaluationResult(BaseModel):
    question: str
    expected_answer: str
    actual_answer: str
    retrieval_hit: bool
    answer_correct: bool
    citation_correct: bool
    latency_ms: int
    retrieved_sources: List[str]
    expected_sources: List[str]


class EvaluationReport(BaseModel):
    total_questions: int
    retrieval_hit_rate: float
    answer_correctness: float
    citation_accuracy: float
    avg_latency_ms: float
    results: List[EvaluationResult]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ======================================================================
# Metrics
# ======================================================================


class MetricsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_queries: int
    avg_query_latency_ms: float
    uptime_seconds: float
