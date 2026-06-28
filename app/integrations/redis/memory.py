from __future__ import annotations

import json
import os
from dataclasses import dataclass

from redis.asyncio import Redis


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def to_json(self) -> str:
        return json.dumps({"role": self.role, "content": self.content})

    @staticmethod
    def from_json(value: str) -> ChatMessage:
        payload = json.loads(value)
        return ChatMessage(role=str(payload["role"]), content=str(payload["content"]))


class RedisChatMemoryError(Exception):
    """Raised when chat memory operations fail."""


class RedisChatMemoryService:
    """Store and retrieve chat history by session_id."""

    def __init__(self, redis_url: str | None = None, max_messages: int = 20) -> None:
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379/0")
        self._redis = Redis.from_url(self._redis_url, decode_responses=True)
        self._max_messages = max_messages

    def _key(self, session_id: str) -> str:
        return f"chat:memory:{session_id}"

    async def get_messages(self, session_id: str) -> list[ChatMessage]:
        try:
            raw_messages = await self._redis.lrange(self._key(session_id), 0, -1)
            return [ChatMessage.from_json(message) for message in raw_messages]
        except Exception as exc:  # pragma: no cover - defensive external API guard
            raise RedisChatMemoryError("Unable to retrieve chat history.") from exc

    async def append_message(self, session_id: str, message: ChatMessage) -> None:
        try:
            key = self._key(session_id)
            await self._redis.rpush(key, message.to_json())
            await self._redis.ltrim(key, -self._max_messages, -1)
        except Exception as exc:  # pragma: no cover - defensive external API guard
            raise RedisChatMemoryError("Unable to store chat history.") from exc
