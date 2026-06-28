from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.chat import BookingExtractionResponse, ChatRequest, ChatResponse
from app.services.chat import ChatService, ChatServiceError

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

_chat_service = ChatService()


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Invalid chat request."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unable to process chat."},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = await _chat_service.generate_answer(
            session_id=request.session_id,
            message=request.message,
        )
    except ChatServiceError as exc:
        logger.error("Chat service error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive endpoint guard
        logger.exception("Unexpected chat endpoint error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while processing chat request.",
        ) from exc

    booking = result.booking
    if booking is None:
        booking_response = BookingExtractionResponse(
            wants_booking=False,
            saved_booking_id=result.saved_booking_id,
        )
    else:
        booking_response = BookingExtractionResponse(
            wants_booking=booking.wants_booking,
            name=booking.name,
            email=booking.email,
            date=booking.date,
            time=booking.time,
            saved_booking_id=result.saved_booking_id,
        )

    response = ChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        booking=booking_response,
    )
    logger.info("Chat response generated successfully for session_id=%s", response.session_id)
    return response
