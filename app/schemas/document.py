from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentUploadRequest(BaseModel):
    filename: str
    chunk_strategy: str


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    upload_date: datetime
    chunk_strategy: str
    total_chunks: int
