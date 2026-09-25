"""
app/models/database.py

SQLAlchemy async models for persisting document metadata and
conversation history in PostgreSQL.
"""

import uuid
from datetime import datetime

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.config import get_settings


# ======================================================================
# Base
# ======================================================================


class Base(AsyncAttrs, DeclarativeBase):
    pass


# ======================================================================
# Document metadata table
# ======================================================================


class DocumentRecord(Base):
    """Stores metadata about ingested documents."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    conversations: Mapped[list["ConversationRecord"]] = relationship(
        "ConversationRecord", back_populates="document", cascade="all, delete-orphan"
    )


# ======================================================================
# Conversation / history tables
# ======================================================================


class ConversationRecord(Base):
    """Stores a conversation session."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    document: Mapped["DocumentRecord | None"] = relationship(
        "DocumentRecord", back_populates="conversations"
    )
    messages: Mapped[list["MessageRecord"]] = relationship(
        "MessageRecord", back_populates="conversation", cascade="all, delete-orphan"
    )


class MessageRecord(Base):
    """Stores individual messages within a conversation."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    conversation: Mapped["ConversationRecord"] = relationship(
        "ConversationRecord", back_populates="messages"
    )


# ======================================================================
# Async engine + session factory
# ======================================================================


def create_engine():
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
    )


async def create_tables(engine) -> None:
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
