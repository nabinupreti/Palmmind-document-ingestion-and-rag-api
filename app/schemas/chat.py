from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BookingExtractionResponse(BaseModel):
    wants_booking: bool
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None
    saved_booking_id: int | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    answer: str
    session_id: str | None = None
    booking: BookingExtractionResponse
