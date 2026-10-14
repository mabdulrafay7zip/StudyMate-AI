"""Document ingestion for StudyMate AI.

Loads study material (PDF, .txt, .md) and splits it into overlapping
chunks that a retriever can search over.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    """One retrievable piece of a document."""

    index: int
    text: str
    source: str
    metadata: dict = field(default_factory=dict)

    def short(self, n: int = 220) -> str:
        text = " ".join(self.text.split())
        return text if len(text) <= n else text[:n].rstrip() + "…"


def extract_text_from_pdf(data: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def load_document(path: str | Path) -> str:
    """Load a .pdf, .txt or .md file from disk and return its text."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path.read_bytes())
    if suffix in (".txt", ".md", ".markdown"):
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {suffix} (use .pdf, .txt or .md)")


def load_document_bytes(name: str, data: bytes) -> str:
    """Load a document from raw bytes (used by the Streamlit uploader)."""
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(data)
    if suffix in (".txt", ".md", ".markdown"):
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {suffix} (use .pdf, .txt or .md)")


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    # end markdown heading lines with a full stop so the heading becomes its
    # own fragment, then drop the '#' markers (heading words stay as context)
    text = re.sub(r"^(#{1,6}\s+.+?)\s*$", r"\1.", text, flags=re.MULTILINE)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


_HEADING_RE = re.compile(r"^#{1,6}\s+\S")


def _split_sections(text: str) -> list[str]:
    """Split markdown text at heading lines; plain text stays one section."""
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if _HEADING_RE.match(line.strip()) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))
    return [s for s in sections if s.strip()]


def chunk_text(
    text: str,
    source: str = "document",
    chunk_size: int = 700,
    overlap: int = 150,
) -> list[Chunk]:
    """Split text into sentence-aligned chunks of ~chunk_size characters.

    Chunking never crosses a markdown heading, so each chunk stays on one
    topic. Within a section, consecutive chunks share roughly `overlap`
    characters worth of trailing sentences so that an idea split across a
    boundary is still retrievable from both sides.
    """
    chunks: list[Chunk] = []

    for section in _split_sections(text):
        sentences = split_sentences(section)
        current: list[str] = []
        current_len = 0

        def flush() -> None:
            nonlocal current, current_len
            if current:
                chunks.append(
                    Chunk(index=len(chunks), text=" ".join(current), source=source)
                )
            # keep the tail sentences as overlap for the next chunk
            tail: list[str] = []
            tail_len = 0
            for sent in reversed(current):
                if tail_len + len(sent) > overlap and tail:
                    break
                tail.insert(0, sent)
                tail_len += len(sent) + 1
            current = tail
            current_len = tail_len

        for sent in sentences:
            if current_len + len(sent) + 1 > chunk_size and current:
                flush()
            current.append(sent)
            current_len += len(sent) + 1
        if current:
            chunks.append(
                Chunk(index=len(chunks), text=" ".join(current), source=source)
            )

    # merge heading-only fragments (e.g. a document title) into a neighbour
    merged: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.text) < 60 and merged:
            prev = merged[-1]
            prev.text = prev.text + " " + chunk.text
        elif len(chunk.text) < 60 and len(chunks) > 1:
            continue  # leading fragment: folded into the next chunk below
        else:
            merged.append(chunk)
    if merged and chunks and len(chunks[0].text) < 60:
        merged[0].text = chunks[0].text + " " + merged[0].text
    for i, chunk in enumerate(merged):
        chunk.index = i
    return merged


def ingest(path: str | Path, **chunk_kwargs) -> list[Chunk]:
    """Full pipeline: file on disk -> list of chunks."""
    text = load_document(path)
    return chunk_text(text, source=Path(path).name, **chunk_kwargs)
