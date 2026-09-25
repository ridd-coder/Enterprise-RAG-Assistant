"""
app/api/deps.py

FastAPI dependencies, including database session injection.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.models.database import create_engine

# Global engine and session factory
engine = create_engine()
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to provide a database session per request."""
    async with AsyncSessionLocal() as session:
        yield session
