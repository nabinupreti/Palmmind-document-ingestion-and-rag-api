from __future__ import annotations

from datetime import date, datetime, time

from pydantic import BaseModel, ConfigDict, EmailStr


class InterviewBookingCreate(BaseModel):
    name: str
    email: EmailStr
    interview_date: date
    interview_time: time


class InterviewBookingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: EmailStr
    interview_date: date
    interview_time: time
    created_at: datetime
