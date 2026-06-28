from __future__ import annotations

import os

from app.integrations.qdrant import QdrantService
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService


def get_chunking_service() -> ChunkingService:
    return ChunkingService()


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def get_qdrant_service() -> QdrantService:
    return QdrantService(
        url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
        collection_name=os.getenv("QDRANT_COLLECTION", "documents"),
    )
