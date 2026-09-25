"""
app/rag/vector_store.py

Qdrant vector store integration.

Implements:
- Collection creation / health checking
- Vector insertion with metadata payloads
- Similarity search with top-k and filtering
- Document deletion
- Duplicate chunk detection
"""

import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.chunker import DocumentChunk

logger = get_logger(__name__)


# ======================================================================
# Exceptions
# ======================================================================


class VectorStoreError(Exception):
    """Raised on Qdrant operations that cannot be recovered from."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when Qdrant is unreachable."""


# ======================================================================
# Search result
# ======================================================================


class SearchResult:
    """A single result returned from vector similarity search."""

    def __init__(self, payload: Dict[str, Any], score: float, point_id: str):
        self.payload = payload
        self.score = score
        self.point_id = point_id

    @property
    def document_id(self) -> str:
        return self.payload.get("document_id", "")

    @property
    def filename(self) -> str:
        return self.payload.get("filename", "")

    @property
    def page_number(self) -> int:
        return self.payload.get("page_number", 0)

    @property
    def chunk_index(self) -> int:
        return self.payload.get("chunk_index", 0)

    @property
    def text(self) -> str:
        return self.payload.get("text", "")


# ======================================================================
# Vector store
# ======================================================================


class QdrantVectorStore:
    """
    Manages all interactions with the Qdrant vector database.

    One instance is shared application-wide via dependency injection.
    """

    def __init__(self, vector_size: int):
        settings = get_settings()
        self._collection = settings.qdrant_collection
        self._vector_size = vector_size

        try:
            client_kwargs: Dict[str, Any] = {
                "host": settings.qdrant_host,
                "port": settings.qdrant_port,
            }
            if settings.qdrant_api_key:
                client_kwargs["api_key"] = settings.qdrant_api_key

            self._client = QdrantClient(**client_kwargs)
            logger.info(
                "qdrant_client_initialized",
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                collection=self._collection,
            )
        except Exception as exc:
            raise VectorStoreConnectionError(
                f"Cannot connect to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def initialize_collection(self) -> None:
        """
        Create the collection if it does not already exist.
        Safe to call on every startup.
        """
        existing = [c.name for c in self._client.get_collections().collections]

        if self._collection in existing:
            logger.info("qdrant_collection_exists", collection=self._collection)
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qdrant_models.VectorParams(
                size=self._vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        logger.info(
            "qdrant_collection_created",
            collection=self._collection,
            vector_size=self._vector_size,
        )

    def health_check(self) -> Dict[str, Any]:
        """Return collection info for the health endpoint."""
        try:
            info = self._client.get_collection(self._collection)
            return {
                "status": "healthy",
                "collection": self._collection,
                "vectors_count": info.vectors_count or 0,
                "host": self._client._client._host
                if hasattr(self._client._client, "_host")
                else "qdrant",
            }
        except Exception as exc:
            logger.error("qdrant_health_check_failed", error=str(exc))
            return {
                "status": "unhealthy",
                "collection": self._collection,
                "vectors_count": 0,
                "host": "unknown",
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Insert or update chunks in Qdrant.

        Uses the chunk_id as a deterministic point ID (via UUID v5)
        to avoid duplicates.

        Returns:
            Number of points upserted.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch."
            )

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Deterministic UUID from chunk_id string
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=chunk.to_payload(),
                )
            )

        self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )

        logger.info(
            "chunks_upserted",
            collection=self._collection,
            count=len(points),
        )

        return len(points)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_document_ids: Optional[List[str]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Perform cosine similarity search.

        Args:
            query_vector: Embedded query.
            top_k: Number of results to return.
            filter_document_ids: If set, restrict search to these documents.
            score_threshold: Minimum cosine similarity (0–1). None = no filter.

        Returns:
            List of SearchResult ordered by descending score.
        """
        query_filter = None
        if filter_document_ids:
            query_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchAny(any=filter_document_ids),
                    )
                ]
            )

        try:
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            logger.error("qdrant_search_failed", error=str(exc))
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        results = [
            SearchResult(
                payload=hit.payload or {},
                score=hit.score,
                point_id=str(hit.id),
            )
            for hit in hits
        ]

        logger.debug(
            "qdrant_search_completed",
            results=len(results),
            top_k=top_k,
            score_threshold=score_threshold,
        )

        return results

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_document(self, document_id: str) -> int:
        """
        Delete all points belonging to a document.

        Returns:
            Estimated number of points deleted.
        """
        try:
            result = self._client.delete(
                collection_name=self._collection,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="document_id",
                                match=qdrant_models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            logger.info("document_deleted_from_qdrant", document_id=document_id)
            return 1  # operation success
        except Exception as exc:
            logger.error(
                "qdrant_delete_failed", document_id=document_id, error=str(exc)
            )
            raise VectorStoreError(
                f"Failed to delete document {document_id}: {exc}"
            ) from exc

    def get_document_chunk_count(self, document_id: str) -> int:
        """Count the number of vectors stored for a document."""
        result = self._client.count(
            collection_name=self._collection,
            count_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchValue(value=document_id),
                    )
                ]
            ),
        )
        return result.count
