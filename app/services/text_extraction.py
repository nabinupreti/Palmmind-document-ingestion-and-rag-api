from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


class TextExtractionError(Exception):
    """Raised when a document cannot be processed."""


class TextExtractionService:
    """Extract plain text from PDF and TXT files."""

    _pdf_suffix = ".pdf"
    _txt_suffix = ".txt"

    @staticmethod
    def extract_text(file_path: str | Path) -> str:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == TextExtractionService._pdf_suffix:
            return TextExtractionService._extract_pdf(path)

        if suffix == TextExtractionService._txt_suffix:
            return TextExtractionService._extract_txt(path)

        raise TextExtractionError("Unsupported file type. Only PDF and TXT files are supported.")

    @staticmethod
    def _extract_pdf(file_path: Path) -> str:
        try:
            reader = PdfReader(str(file_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:  # pragma: no cover - defensive parsing guard
            raise TextExtractionError("Unable to extract text from PDF.") from exc

    @staticmethod
    def _extract_txt(file_path: Path) -> str:
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise TextExtractionError("TXT files must be UTF-8 encoded.") from exc
        except OSError as exc:
            raise TextExtractionError("Unable to read TXT file.") from exc
