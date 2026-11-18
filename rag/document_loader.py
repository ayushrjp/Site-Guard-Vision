"""Safety-document loading, cleaning and chunking.

PDFs are read page by page with pypdf so citations can include page numbers.
Plain text and Markdown are read as a single untitled section.  Chunking is
paragraph-aware with character overlap, which keeps retrieval passages small
enough for a 3B local model while preserving surrounding context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import config

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INLINE_WS_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
MAX_CHUNKS_PER_DOCUMENT = 3000


class DocumentError(RuntimeError):
    pass


@dataclass(frozen=True)
class TextChunk:
    text: str
    page: int | None
    chunk_index: int


@dataclass
class LoadedDocument:
    filename: str
    chunk_count: int
    page_count: int | None
    text_chars: int
    chunks: list[TextChunk]


def clean_text(raw: str) -> str:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = _CONTROL_RE.sub(" ", text)
    lines = [_INLINE_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    *,
    size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split cleaned text into overlapping chunks on paragraph boundaries."""
    size = max(200, int(size or config.CHUNK_SIZE))
    overlap = max(0, min(int(overlap if overlap is not None else config.CHUNK_OVERLAP), size // 2))
    cleaned = clean_text(text)
    if not cleaned:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in paragraphs:
        if len(paragraph) > size:
            flush()
            start = 0
            while start < len(paragraph):
                piece = paragraph[start : start + size].strip()
                if piece:
                    chunks.append(piece)
                if start + size >= len(paragraph):
                    break
                start += size - overlap
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= size:
            current = candidate
            continue

        flush()
        if chunks and overlap:
            max_tail = max(0, size - len(paragraph) - 2)
            tail = chunks[-1][-min(overlap, max_tail):].strip() if max_tail else ""
            current = f"{tail}\n\n{paragraph}" if tail else paragraph
        else:
            current = paragraph

    flush()

    if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
        raise DocumentError(
            f"Document produced {len(chunks)} chunks, above the {MAX_CHUNKS_PER_DOCUMENT} limit. "
            "Split it into smaller policy documents."
        )
    return chunks


def _read_pdf(path: Path) -> list[tuple[int | None, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency installed in the app venv
        raise DocumentError("PDF support needs the pypdf package (pip install pypdf).") from exc

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001 - message matters, not the type
                raise DocumentError("The PDF is password protected and cannot be indexed.") from exc
        pages: list[tuple[int | None, str]] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001 - one bad page must not kill the document
                text = ""
            pages.append((number, text))
        return pages
    except DocumentError:
        raise
    except Exception as exc:  # noqa: BLE001 - pypdf raises several parser types
        raise DocumentError(f"Could not read this PDF: {exc}") from exc


def _read_text(path: Path) -> list[tuple[int | None, str]]:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return [(None, data.decode(encoding))]
        except UnicodeDecodeError:
            continue
    return [(None, data.decode("utf-8", errors="replace"))]


def load_document(path: str | Path) -> LoadedDocument:
    path = Path(path)
    if not path.is_file():
        raise DocumentError(f"Document not found: {path.name}")
    if path.suffix.lower() not in config.ALLOWED_DOCUMENT_EXTENSIONS:
        raise DocumentError("Only PDF, TXT and Markdown documents can be indexed.")

    sections = _read_pdf(path) if path.suffix.lower() == ".pdf" else _read_text(path)
    chunks: list[TextChunk] = []
    text_chars = 0
    for page_number, raw in sections:
        cleaned = clean_text(raw)
        text_chars += len(cleaned)
        for piece in chunk_text(cleaned):
            chunks.append(
                TextChunk(
                    text=piece,
                    page=page_number,
                    chunk_index=len(chunks),
                )
            )

    if not chunks:
        hint = " (scanned PDFs need OCR before indexing)" if path.suffix.lower() == ".pdf" else ""
        raise DocumentError(f"No readable text was found in {path.name}{hint}.")

    page_count = len(sections) if path.suffix.lower() == ".pdf" else None
    return LoadedDocument(
        filename=path.name,
        chunk_count=len(chunks),
        page_count=page_count,
        text_chars=text_chars,
        chunks=chunks,
    )
