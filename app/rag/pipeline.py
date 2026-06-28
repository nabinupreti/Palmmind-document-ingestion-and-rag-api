from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from app.integrations.qdrant import QdrantService, QdrantServiceError
from app.services.embedding import EmbeddingService, EmbeddingServiceError


class RagPipelineError(Exception):
    """Raised when RAG pipeline execution fails."""


@dataclass(frozen=True)
class RagResult:
    answer: str
    contexts: list[str]


class RagPipelineService:
    """Custom Retrieval-Augmented Generation pipeline without RetrievalQAChain."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService | None = None,
        qdrant_service: QdrantService | None = None,
        collection_name: str | None = None,
        qdrant_url: str | None = None,
        model: str | None = None,
        top_k: int = 5,
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RagPipelineError("GEMINI_API_KEY environment variable not set.")
        self._client = genai.Client(api_key=api_key)

        self._embedding_service = embedding_service or EmbeddingService()
        resolved_collection = collection_name or os.getenv("QDRANT_COLLECTION", "documents")
        resolved_qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://qdrant:6333")
        self._qdrant_service = qdrant_service or QdrantService(
            url=resolved_qdrant_url,
            collection_name=resolved_collection,
        )
        self._model = model or os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
        self._top_k = top_k

    async def run(
        self,
        *,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> RagResult:
        if not query.strip():
            raise RagPipelineError("Query cannot be empty.")

        try:
            query_embedding = self._embedding_service.embed_text(query)
        except EmbeddingServiceError as exc:
            raise RagPipelineError("Failed to embed user query.") from exc

        try:
            search_results = self._qdrant_service.similarity_search(
                query_embedding=query_embedding,
                limit=self._top_k,
            )
        except QdrantServiceError as exc:
            raise RagPipelineError("Failed to retrieve context from Qdrant.") from exc

        contexts = self._extract_contexts(search_results)
        prompt = self._build_prompt(query=query, contexts=contexts)

        history_block = self._format_chat_history(chat_history)
        full_prompt = (
            "You are a helpful RAG assistant. "
            "Use only the provided context when possible and be concise.\n\n"
            f"Chat History:\n{history_block}\n\n"
            f"{prompt}"
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=full_prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            answer = response.text or ""
        except Exception as exc:  # pragma: no cover - defensive external API guard
            raise RagPipelineError("Failed to generate response from Gemini.") from exc

        return RagResult(answer=answer, contexts=contexts)

    @staticmethod
    def _extract_contexts(search_results: list[dict[str, Any]]) -> list[str]:
        contexts: list[str] = []
        for item in search_results:
            payload = item.get("payload") or {}
            text = payload.get("text")
            if isinstance(text, str) and text.strip():
                contexts.append(text.strip())

        return contexts

    @staticmethod
    def _build_prompt(*, query: str, contexts: list[str]) -> str:
        if contexts:
            context_block = "\n\n".join(
                f"[{index}] {context}"
                for index, context in enumerate(contexts, start=1)
            )
        else:
            context_block = "No relevant context retrieved from the knowledge base."

        return (
            "Use the context to answer the user question. "
            "If context is insufficient, say you could not find relevant information.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}"
        )

    @staticmethod
    def _format_chat_history(chat_history: list[dict[str, str]] | None) -> str:
        if not chat_history:
            return "No prior chat history."

        rows: list[str] = []
        for item in chat_history:
            role = item.get("role", "user")
            content = item.get("content", "")
            if content.strip():
                rows.append(f"{role}: {content.strip()}")

        return "\n".join(rows) if rows else "No prior chat history."
