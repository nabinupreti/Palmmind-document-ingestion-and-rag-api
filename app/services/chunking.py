from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


ChunkStrategy = Literal["fixed", "recursive"]


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 1000
    overlap: int = 200


class ChunkingError(ValueError):
    """Raised when chunking parameters are invalid."""


class ChunkingService:
    """Split text into chunks using fixed or recursive strategies."""

    _sentence_split_pattern = re.compile(r"(?<=[.!?])\s+")

    @staticmethod
    def chunk_text(
        text: str,
        strategy: ChunkStrategy = "fixed",
        *,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> list[str]:
        """Return a list of text chunks."""
        ChunkingService._validate_params(chunk_size=chunk_size, overlap=overlap)

        normalized_text = text.strip()
        if not normalized_text:
            return []

        if strategy == "fixed":
            return ChunkingService._fixed_chunks(
                normalized_text,
                chunk_size=chunk_size,
                overlap=overlap,
            )

        if strategy == "recursive":
            return ChunkingService._recursive_chunks(
                normalized_text,
                chunk_size=chunk_size,
            )

        raise ChunkingError(f"Unsupported chunking strategy: {strategy}")

    @staticmethod
    def _validate_params(*, chunk_size: int, overlap: int) -> None:
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be greater than 0.")
        if overlap < 0:
            raise ChunkingError("overlap cannot be negative.")
        if overlap >= chunk_size:
            raise ChunkingError("overlap must be smaller than chunk_size.")

    @staticmethod
    def _fixed_chunks(text: str, *, chunk_size: int, overlap: int) -> list[str]:
        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break
            start = max(end - overlap, start + 1)

        return chunks

    @staticmethod
    def _recursive_chunks(text: str, *, chunk_size: int) -> list[str]:
        chunks: list[str] = []
        buffer: list[str] = []
        buffer_length = 0

        def flush_buffer() -> None:
            nonlocal buffer, buffer_length
            if buffer:
                chunks.append("\n\n".join(buffer).strip())
                buffer = []
                buffer_length = 0

        for paragraph in ChunkingService._split_paragraphs(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(paragraph) <= chunk_size:
                if buffer_length and buffer_length + len(paragraph) + 2 > chunk_size:
                    flush_buffer()
                buffer.append(paragraph)
                buffer_length += len(paragraph) + (2 if buffer_length else 0)
                continue

            flush_buffer()
            for sentence in ChunkingService._split_sentences(paragraph):
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(sentence) <= chunk_size:
                    if buffer_length and buffer_length + len(sentence) + 1 > chunk_size:
                        flush_buffer()
                    buffer.append(sentence)
                    buffer_length += len(sentence) + (1 if buffer_length else 0)
                    continue

                flush_buffer()
                chunks.extend(ChunkingService._fixed_chunks(sentence, chunk_size=chunk_size, overlap=0))

        flush_buffer()
        return chunks

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        return [paragraph for paragraph in re.split(r"\n\s*\n+", text) if paragraph.strip()]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        sentences = ChunkingService._sentence_split_pattern.split(text.strip())
        return [sentence for sentence in sentences if sentence.strip()]
