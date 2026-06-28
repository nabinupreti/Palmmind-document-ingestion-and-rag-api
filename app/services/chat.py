from __future__ import annotations

import os
from dataclasses import dataclass
from uuid import uuid4

from app.integrations.redis import ChatMessage, RedisChatMemoryError, RedisChatMemoryService
from app.rag import RagPipelineError, RagPipelineService
from app.services.interview_booking import (
    BookingExtractionResult,
    InterviewBookingService,
    InterviewBookingServiceError,
)


class ChatServiceError(Exception):
    """Raised when a chat completion cannot be generated."""


@dataclass(frozen=True)
class ChatResult:
    answer: str
    session_id: str | None = None
    booking: BookingExtractionResult | None = None
    saved_booking_id: int | None = None


class ChatService:
    """Generate chat responses using the Gemini API."""

    def __init__(
        self,
        model: str | None = None,
        memory_service: RedisChatMemoryService | None = None,
        rag_pipeline: RagPipelineService | None = None,
        interview_booking_service: InterviewBookingService | None = None,
    ) -> None:
        self._model = model or os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
        self._memory_service = memory_service or RedisChatMemoryService()
        self._rag_pipeline = rag_pipeline or RagPipelineService(
            model=self._model,
        )
        self._interview_booking_service = interview_booking_service or InterviewBookingService(
            model=self._model,
        )

    async def generate_answer(self, *, session_id: str | None, message: str) -> ChatResult:
        if not message.strip():
            raise ChatServiceError("Message cannot be empty.")

        resolved_session_id = session_id or str(uuid4())

        try:
            history = await self._memory_service.get_messages(resolved_session_id)
        except RedisChatMemoryError as exc:
            raise ChatServiceError("Unable to load chat history.") from exc

        chat_history = [
            {"role": chat_message.role, "content": chat_message.content}
            for chat_message in history
        ]

        try:
            rag_result = await self._rag_pipeline.run(
                query=message,
                chat_history=chat_history,
            )
            answer = rag_result.answer
        except RagPipelineError as exc:
            raise ChatServiceError("Unable to generate chat response.") from exc

        try:
            booking = await self._interview_booking_service.extract_booking_details(message)
            saved_booking_id = await self._interview_booking_service.save_booking_if_complete(booking)
        except InterviewBookingServiceError as exc:
            raise ChatServiceError("Unable to process interview booking.") from exc

        try:
            await self._memory_service.append_message(
                resolved_session_id,
                ChatMessage(role="user", content=message),
            )
            await self._memory_service.append_message(
                resolved_session_id,
                ChatMessage(role="assistant", content=answer),
            )
        except RedisChatMemoryError as exc:
            raise ChatServiceError("Unable to store chat history.") from exc

        return ChatResult(
            answer=answer,
            session_id=resolved_session_id,
            booking=booking,
            saved_booking_id=saved_booking_id,
        )
