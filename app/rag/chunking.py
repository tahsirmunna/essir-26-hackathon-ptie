"""Turn extracted PDF content into the units you index.

Chunking is subsection-aware: `extract_pages` (see `ingest.py`) hands us `TextBlock`s
carrying each block's font size and boldness, which is enough to tell headings apart
from body text without any NLP. We group blocks into sections at each detected heading,
then split any section that is still bigger than `chunk_size` into overlapping pieces —
splitting on sentence boundaries so we never cut a fact in half. Every produced chunk
keeps the page it came from (so citations line up) and the heading it sits under.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from ..config import get_settings

_HEADING_MAX_CHARS = 120
_HEADING_SIZE_RATIO = 1.15
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TextBlock:
    """One layout block from the PDF, as produced by `ingest.extract_pages`."""

    text: str
    page: int       # 1-indexed
    font_size: float
    bold: bool


@dataclass
class Chunk:
    text: str
    page: int          # 1-indexed
    index: int         # position within the document
    section: str | None = None   # nearest preceding heading, if any


def _is_heading(block: TextBlock, body_size: float) -> bool:
    text = block.text.strip()
    if not text or "\n" in text or len(text) > _HEADING_MAX_CHARS:
        return False
    size_ratio = (block.font_size / body_size) if body_size else 1.0
    candidate = size_ratio >= _HEADING_SIZE_RATIO or (block.bold and size_ratio >= 1.0)
    if not candidate:
        return False
    # A bold/large sentence-length run of prose is not a heading; real headings rarely
    # trail off with sentence punctuation and a dozen more words.
    if text.endswith((".", ",", ";")) and len(text.split()) > 12:
        return False
    return True


def _sectionize(blocks: list[TextBlock]) -> list[tuple[str | None, list[TextBlock]]]:
    sizes = [b.font_size for b in blocks if b.font_size]
    body_size = statistics.median(sizes) if sizes else 0.0

    sections: list[tuple[str | None, list[TextBlock]]] = []
    title: str | None = None
    current: list[TextBlock] = []
    for block in blocks:
        if _is_heading(block, body_size):
            if current:
                sections.append((title, current))
            title = block.text.strip()
            current = []
        else:
            current.append(block)
    if current:
        sections.append((title, current))
    return sections


def _atoms(block: TextBlock, chunk_size: int) -> list[tuple[str, int]]:
    """A block, split into sentence-sized pieces if it alone exceeds chunk_size."""
    text = block.text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [(text, block.page)]
    sentences = _SENTENCE_SPLIT.split(text.replace("\n", " "))
    return [(s.strip(), block.page) for s in sentences if s.strip()]


def _pack(atoms: list[tuple[str, int]], chunk_size: int, chunk_overlap: int) -> list[tuple[str, int]]:
    """Greedily pack atoms into <= chunk_size pieces, carrying a character overlap."""
    pieces: list[tuple[str, int]] = []
    buf_parts: list[str] = []
    buf_len = 0
    buf_page: int | None = None

    for text, page in atoms:
        if buf_page is None:
            buf_page = page
        extra = len(text) + (2 if buf_parts else 0)
        if buf_parts and buf_len + extra > chunk_size:
            chunk_text = "\n\n".join(buf_parts)
            pieces.append((chunk_text, buf_page))
            tail = chunk_text[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
            buf_parts = [p for p in (tail, text) if p]
            buf_len = sum(len(p) for p in buf_parts) + 2 * (len(buf_parts) - 1)
            buf_page = page
        else:
            buf_parts.append(text)
            buf_len += extra

    if buf_parts:
        pieces.append(("\n\n".join(buf_parts), buf_page if buf_page is not None else atoms[-1][1]))
    return pieces


def _split_section(
    blocks: list[TextBlock], chunk_size: int, chunk_overlap: int
) -> list[tuple[str, int]]:
    atoms: list[tuple[str, int]] = []
    for block in blocks:
        atoms.extend(_atoms(block, chunk_size))
    if not atoms:
        return []
    return _pack(atoms, chunk_size, chunk_overlap)


def chunk_pages(blocks: list[TextBlock]) -> list[Chunk]:
    """Group blocks into subsections (by heading), then size-split each section.

    A section that fits within `chunk_size` becomes one chunk. A longer one is split on
    sentence boundaries into overlapping pieces so retrieval still gets precise, page-
    correct units instead of one giant blob per subsection.
    """
    settings = get_settings()
    chunks: list[Chunk] = []
    idx = 0
    for title, section_blocks in _sectionize(blocks):
        for text, page in _split_section(section_blocks, settings.chunk_size, settings.chunk_overlap):
            text = text.strip()
            if not text:
                continue
            chunks.append(Chunk(text=text, page=page, index=idx, section=title))
            idx += 1
    return chunks
