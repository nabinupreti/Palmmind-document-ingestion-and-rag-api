from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
import logging
from pathlib import Path
from typing import Annotated
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from pypdf import PdfReader

from app.api.v1.dependencies import get_async_session
from app.api.v1.dependencies import (
    get_chunking_service,
    get_embedding_service,
    get_qdrant_service,
)
from app.integrations.qdrant import QdrantDocumentPayload, QdrantService, QdrantServiceError
from app.models.document import Document
from app.schemas.document import DocumentUploadResponse
from app.services.chunking import ChunkingService, ChunkingError
from app.services.embedding import EmbeddingService, EmbeddingServiceError

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})
_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({"application/pdf", "text/plain"})
_ALLOWED_CHUNK_STRATEGIES: frozenset[str] = frozenset({"fixed", "recursive", "semantic"})


def _validate_upload(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A filename is required.",
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF and TXT files are supported.",
        )

    content_type = (file.content_type or "").lower()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid file content type. Use application/pdf or text/plain.",
        )

    return suffix


def _validate_chunk_strategy(chunk_strategy: str) -> str:
    normalized_strategy = chunk_strategy.strip().lower()
    if normalized_strategy not in _ALLOWED_CHUNK_STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Invalid chunk_strategy. Allowed values: "
                + ", ".join(sorted(_ALLOWED_CHUNK_STRATEGIES))
            ),
        )
    return normalized_strategy


def _extract_text(file_bytes: bytes, suffix: str) -> str:
    if suffix == ".txt":
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="TXT files must be UTF-8 encoded.",
            ) from exc

    try:
        reader = PdfReader(BytesIO(file_bytes))
        pages: Sequence[str] = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as exc:  # pragma: no cover - defensive file parsing guard
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to read PDF contents.",
        ) from exc


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid or empty file payload."},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"description": "Unsupported file type."},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Invalid chunk strategy."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error."},
    },
)
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    chunk_strategy: Annotated[str, Form(...)],
    db: Annotated[AsyncSession, Depends(get_async_session)],
    chunking_service: Annotated[ChunkingService, Depends(get_chunking_service)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    qdrant_service: Annotated[QdrantService, Depends(get_qdrant_service)],
) -> DocumentUploadResponse:
    document: Document | None = None
    try:
        normalized_strategy = _validate_chunk_strategy(chunk_strategy)
        suffix = _validate_upload(file)
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        extracted_text = _extract_text(file_bytes, suffix)
        chunk_size_map: dict[str, tuple[str, int, int]] = {
            "fixed": ("fixed", 1000, 200),
            "recursive": ("recursive", 1200, 0),
            "semantic": ("recursive", 1500, 0),
        }
        chunk_mode, chunk_size, overlap = chunk_size_map[normalized_strategy]

        try:
            chunks = chunking_service.chunk_text(
                extracted_text,
                strategy=chunk_mode,  # semantic is handled via recursive-style chunking
                chunk_size=chunk_size,
                overlap=overlap,
            )
        except ChunkingError as exc:
            logger.warning("Chunking failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        total_chunks = len(chunks)
        logger.info("Created %s chunks for filename=%s", total_chunks, file.filename)

        document = Document(
            filename=file.filename,
            upload_date=datetime.now(UTC),
            chunk_strategy=normalized_strategy,
            total_chunks=total_chunks,
        )

        db.add(document)
        await db.commit()
        await db.refresh(document)

        embeddings = embedding_service.embed_texts(chunks)
        if len(embeddings) != len(chunks):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Embedding generation returned an unexpected number of vectors.",
            )

        logger.info("Generated %s embeddings for document_id=%s", len(embeddings), document.id)

        payloads = [
            QdrantDocumentPayload(
                document_id=document.id,
                filename=document.filename,
                chunk_index=index,
                chunk_strategy=document.chunk_strategy,
                total_chunks=document.total_chunks,
                text=chunk,
            )
            for index, chunk in enumerate(chunks)
        ]

        try:
            qdrant_service.store_embeddings(
                embeddings=embeddings,
                payloads=payloads,
                ids=[document.id * 1_000_000 + index for index in range(len(chunks))],
            )
        except QdrantServiceError as exc:
            logger.exception("Failed to store vectors in Qdrant for document_id=%s", document.id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document metadata saved, but Qdrant ingestion failed.",
            ) from exc

        logger.info("Stored %s vectors in Qdrant for document_id=%s", len(embeddings), document.id)
    except HTTPException as exc:
        logger.warning("Document upload validation failed: %s", exc.detail)
        raise
    except EmbeddingServiceError as exc:
        logger.exception("Embedding generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to generate embeddings.",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive database guard
        await db.rollback()
        logger.exception("Unexpected error during document upload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to save document metadata.",
        ) from exc

    upload_date = document.upload_date if document is not None else None
    if upload_date is None:
        logger.error("Document upload succeeded but upload_date is missing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document timestamp was not generated.",
        )

    logger.info(
        "Document uploaded successfully: filename=%s, strategy=%s, total_chunks=%s",
        document.filename,
        document.chunk_strategy,
        document.total_chunks,
    )

    return DocumentUploadResponse(
        id=document.id,
        filename=document.filename,
        upload_date=upload_date,
        chunk_strategy=document.chunk_strategy,
        total_chunks=document.total_chunks,
    )
