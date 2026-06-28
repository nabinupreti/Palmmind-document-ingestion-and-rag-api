from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, time

from google import genai
from google.genai import types

from app.models.interview_booking import InterviewBooking


class InterviewBookingServiceError(Exception):
    """Raised when interview booking extraction or persistence fails."""


@dataclass(frozen=True)
class BookingExtractionResult:
    wants_booking: bool
    name: str | None
    email: str | None
    date: str | None
    time: str | None


@dataclass(frozen=True)
class BookingResult:
    extracted: BookingExtractionResult
    saved_booking_id: int | None


class InterviewBookingService:
    """Detect and save interview bookings from chat messages."""

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise InterviewBookingServiceError("GEMINI_API_KEY environment variable not set.")
        self._client = genai.Client(api_key=api_key)

        self._model = model or os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

    async def extract_booking_details(self, message: str) -> BookingExtractionResult:
        if not message.strip():
            return BookingExtractionResult(
                wants_booking=False,
                name=None,
                email=None,
                date=None,
                time=None,
            )

        prompt = (
            "Analyze the user message and detect if they want to book an interview. "
            "Return strict JSON with keys: wants_booking (boolean), name (string|null), "
            "email (string|null), date (YYYY-MM-DD|string|null), time (HH:MM|string|null). "
            "Return only JSON and no extra text.\n\n"
            f"User message: {message}"
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            content = response.text or "{}"
            payload = json.loads(content)
        except Exception as exc:  # pragma: no cover - defensive external API guard
            raise InterviewBookingServiceError("Unable to extract interview booking details.") from exc

        wants_booking = bool(payload.get("wants_booking"))

        return BookingExtractionResult(
            wants_booking=wants_booking,
            name=_normalize_string(payload.get("name")),
            email=_normalize_string(payload.get("email")),
            date=_normalize_string(payload.get("date")),
            time=_normalize_string(payload.get("time")),
        )

    async def save_booking_if_complete(self, extracted: BookingExtractionResult) -> int | None:
        if not extracted.wants_booking:
            return None

        if not extracted.name or not extracted.email or not extracted.date or not extracted.time:
            return None

        interview_date = _parse_date(extracted.date)
        interview_time = _parse_time(extracted.time)

        if interview_date is None or interview_time is None:
            return None

        from app.db.session import AsyncSessionLocal

        booking = InterviewBooking(
            name=extracted.name,
            email=extracted.email,
            interview_date=interview_date,
            interview_time=interview_time,
        )

        try:
            async with AsyncSessionLocal() as session:
                session.add(booking)
                await session.commit()
                await session.refresh(booking)
                return booking.id
        except Exception as exc:  # pragma: no cover - defensive database guard
            raise InterviewBookingServiceError("Unable to save interview booking.") from exc


def _normalize_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    return normalized if normalized else None


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_time(value: str) -> time | None:
    normalized = value.strip()
    if len(normalized) == 5:
        normalized = f"{normalized}:00"

    try:
        return time.fromisoformat(normalized)
    except ValueError:
        return None
