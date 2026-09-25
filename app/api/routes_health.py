"""
app/api/routes_health.py

Health check endpoints:

  GET /api/health
  GET /api/health/vector-db
"""

from typing import Any

from fastapi import APIRouter, Depends

from app.core.config import get_settings
from app.models.schemas import HealthStatus, VectorDBHealth

router = APIRouter(prefix="/api/health", tags=["Health"])


def get_rag_pipeline() -> Any:
    """Dependency — returns the shared RAGPipeline from app state."""
    from app.main import get_app_state

    state = get_app_state()
    return state.get("pipeline")


@router.get(
    "",
    response_model=HealthStatus,
    summary="Application health check",
)
async def health() -> HealthStatus:
    """Returns API health status. Suitable for load-balancer probes."""
    settings = get_settings()
    return HealthStatus(
        status="healthy",
        version="1.0.0",
        environment=settings.app_env,
    )


@router.get(
    "/vector-db",
    response_model=VectorDBHealth,
    summary="Qdrant vector database health check",
)
async def vector_db_health(
    pipeline: Any = Depends(get_rag_pipeline),
) -> VectorDBHealth:
    """Checks connectivity to Qdrant and reports collection status."""
    settings = get_settings()

    if pipeline is None:
        return VectorDBHealth(
            status="unhealthy",
            collection=settings.qdrant_collection,
            vectors_count=0,
            host=settings.qdrant_host,
        )

    info = pipeline.vector_store_health()

    return VectorDBHealth(
        status=info.get("status", "unknown"),
        collection=info.get("collection", settings.qdrant_collection),
        vectors_count=info.get("vectors_count", 0),
        host=info.get("host", settings.qdrant_host),
    )
