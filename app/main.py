"""
app/main.py

FastAPI application factory and startup lifecycle.

- Initialises all shared components (RAGPipeline, Services)
- Registers all routers
- Configures CORS
- Configures structured logging
- Provides app-wide state via get_app_state()
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

# Must configure logging before anything else logs
configure_logging()
logger = get_logger(__name__)

# Global application state
_APP_STATE: Dict[str, Any] = {}


def get_app_state() -> Dict[str, Any]:
    """Return the shared application state dictionary."""
    return _APP_STATE


# ======================================================================
# Lifespan — startup / shutdown
# ======================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.

    Startup:
    - Ensures data directory exists
    - Initialises RAG pipeline (connects to Qdrant, validates collection)
    - Initialises services

    Shutdown:
    - Logs graceful shutdown
    """
    settings = get_settings()

    logger.info("application_starting", env=settings.app_env)

    # Ensure document storage directory exists
    data_path = settings.data_path
    data_path.mkdir(parents=True, exist_ok=True)

    # Initialise the RAG pipeline (may raise if Qdrant is unreachable)
    from app.rag.pipeline import RAGPipeline
    from app.services.document_service import DocumentService
    from app.services.chat_service import ChatService

    try:
        pipeline = RAGPipeline()
    except Exception as exc:
        logger.error("pipeline_init_failed", error=str(exc))
        # Allow app to start even if Qdrant is temporarily down
        # Health endpoint will report unhealthy
        pipeline = None

    doc_service = DocumentService(pipeline) if pipeline else None
    chat_service = ChatService(pipeline) if pipeline else None

    _APP_STATE.update(
        {
            "pipeline": pipeline,
            "document_service": doc_service,
            "chat_service": chat_service,
            "start_time": time.monotonic(),
            "total_queries": 0,
            "avg_latency_ms": 0.0,
        }
    )

    logger.info("application_ready", env=settings.app_env, port=settings.app_port)

    yield  # Application runs here

    logger.info("application_shutting_down")


# ======================================================================
# Application factory
# ======================================================================


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Enterprise RAG Knowledge Assistant",
        description=(
            "A production-grade Retrieval-Augmented Generation system "
            "for enterprise document Q&A."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(chat_router)

    # Serve React frontend static files if built
    frontend_dist = Path("frontend/dist")
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


app = create_app()


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=not settings.is_production,
        log_level=settings.log_level.lower(),
    )
