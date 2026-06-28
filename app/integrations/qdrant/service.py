from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


@dataclass(frozen=True)
class QdrantDocumentPayload:
    document_id: int
    filename: str
    chunk_index: int
    chunk_strategy: str
    total_chunks: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "chunk_strategy": self.chunk_strategy,
            "total_chunks": self.total_chunks,
            "text": self.text,
        }


class QdrantServiceError(Exception):
    """Raised when Qdrant operations fail."""


class QdrantService:
    """Store and search document embeddings in Qdrant."""

    def __init__(self, url: str, collection_name: str, vector_size: int = 3072) -> None:
        self._client = QdrantClient(url=url)
        self._collection_name = collection_name
        self._vector_size = vector_size

    def ensure_collection(self) -> None:
        try:
            collections = self._client.get_collections().collections
            existing_names = {collection.name for collection in collections}
            if self._collection_name in existing_names:
                return

            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self._vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive remote guard
            raise QdrantServiceError("Unable to ensure Qdrant collection.") from exc

    def store_embeddings(
        self,
        *,
        embeddings: list[list[float]],
        payloads: list[QdrantDocumentPayload],
        ids: list[int] | None = None,
    ) -> None:
        if len(embeddings) != len(payloads):
            raise QdrantServiceError("Embeddings and payloads must have the same length.")

        self.ensure_collection()

        point_ids = ids or list(range(1, len(embeddings) + 1))
        if len(point_ids) != len(embeddings):
            raise QdrantServiceError("Point IDs must match the number of embeddings.")

        points = [
            qdrant_models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload.to_dict(),
            )
            for point_id, embedding, payload in zip(point_ids, embeddings, payloads, strict=True)
        ]

        try:
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
        except Exception as exc:  # pragma: no cover - defensive remote guard
            raise QdrantServiceError("Unable to store embeddings in Qdrant.") from exc

    def similarity_search(
        self,
        *,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        self.ensure_collection()

        try:
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True,
            )
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload or {},
                }
                for result in results
            ]
        except Exception as exc:  # pragma: no cover - defensive remote guard
            raise QdrantServiceError("Unable to perform similarity search.") from exc
