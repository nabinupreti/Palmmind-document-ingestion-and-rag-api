from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Sequence

from google import genai

logger = logging.getLogger(__name__)


class EmbeddingServiceError(Exception):
    """Raised when embeddings cannot be generated."""


class EmbeddingService:
    """Generate embeddings using Google Gemini Embeddings API.

    Public interface is kept the same as the previous service:
    - embed_text(text: str) -> list[float]
    - embed_texts(texts: list[str]) -> list[list[float]]
    """

    _model_name = "gemini-embedding-2"

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_client() -> genai.Client:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EmbeddingServiceError("GEMINI_API_KEY environment variable not set.")
        return genai.Client(api_key=api_key)

    @staticmethod
    def _extract_values(values: Sequence[float] | None) -> list[float]:
        if not values:
            raise EmbeddingServiceError("Embedding response did not include values.")
        return [float(value) for value in values]

    @staticmethod
    def embed_text(text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingServiceError("Text cannot be empty.")

        try:
            client = EmbeddingService._get_client()
            response = client.models.embed_content(
                model=EmbeddingService._model_name,
                contents=[text],
            )
            embeddings = getattr(response, "embeddings", None) or []
            if not embeddings:
                raise EmbeddingServiceError("Embedding response was empty.")
            values = getattr(embeddings[0], "values", None)
            return EmbeddingService._extract_values(values)
        except Exception as exc:  # pragma: no cover - defensive external API guard
            raise EmbeddingServiceError("Unable to generate embedding.") from exc

    @staticmethod
    def embed_texts(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        cleaned_texts = [text for text in texts if text.strip()]
        if not cleaned_texts:
            raise EmbeddingServiceError("Text list cannot be empty.")

        try:
            client = EmbeddingService._get_client()
            response = client.models.embed_content(
                model=EmbeddingService._model_name,
                contents=cleaned_texts,
            )
            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(cleaned_texts):
                raise EmbeddingServiceError("Embedding response size mismatch.")
            return [
                EmbeddingService._extract_values(getattr(item, "values", None))
                for item in embeddings
            ]
        except Exception as exc:
            logger.exception("Gemini embedding generation failed")
            raise EmbeddingServiceError("Unable to generate embedding.") from exc
