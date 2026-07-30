"""Load a PDF into the vector store.

    parse PDF (data/in) -> layout blocks -> [section-aware chunking] -> embeddings -> Qdrant

Extraction uses PyMuPDF (`fitz`) instead of pypdf: it reports every span's exact text,
font flags and line, which is what lets `extract_pages` recognise a heading like
"4.2.2 LLM-Based Reranking" — a numbered label in bold/italic characters, sometimes
immediately followed by the section's body text on the same PDF "line" (a run-in
heading) — and split it cleanly from the body text `chunking.py` groups into
sections. PyMuPDF also copes far better than pypdf with two-column layouts, ligatures
and hyphenation, so the text it returns matches the document closely enough that page
citations stay correct.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from qdrant_client import models

from ..config import get_settings
from ..models import IngestResponse
from ..vectorstore.qdrant_store import get_store
from .chunking import TextBlock, chunk_pages
from .embeddings import get_embedder

# A fixed namespace so re-ingesting the same document overwrites its points
# (idempotent ids) instead of duplicating them.
_NAMESPACE = uuid.UUID("6f0d9b1e-3b7a-4c2e-9a1d-000000000000")

# PDF "flags" bits (PyMuPDF span flags). See
# https://pymupdf.readthedocs.io/en/latest/textpage.html#span-dictionary
_ITALIC_FLAG = 1 << 1
_BOLD_FLAG = 1 << 4
_STYLED_FLAGS = _ITALIC_FLAG | _BOLD_FLAG

# A heading is a numbered label ("4", "4.1", "4.2.2", optionally trailing-dotted) in
# styled (bold/italic) characters. `_NUM_RE` matches just the label; `_FULL_RE` also
# requires a title to follow, so a bare page-footer number like "12" never qualifies.
_NUM_RE = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}\.?$")
_FULL_RE = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}\.?\s+\S.{0,118}$")
_HEADING_MAX_CHARS = 120


def _find_pdf(filename: str | None) -> Path:
    in_dir = Path(get_settings().in_dir)
    if filename:
        path = in_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"no such PDF: {path}")
        return path
    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no *.pdf found in {in_dir}/ — put your document there first")
    return pdfs[0]


def _flatten_spans(block: dict) -> list[tuple[str, bool, int]]:
    """(text, is_styled, source_line_index) for every non-empty span in a PDF block,
    in reading order.

    A big horizontal gap — e.g. between a heading's number and its title, which PyMuPDF
    often reports as two separate "lines" at the same visual row — carries no literal
    space character, so one is inserted at the start of every line but the first to
    rejoin it into normal, space-separated text.
    """
    out: list[tuple[str, bool, int]] = []
    for line_index, line in enumerate(block.get("lines", [])):
        for span_index, span in enumerate(line.get("spans", [])):
            text = span.get("text", "")
            if not text:
                continue
            styled = bool(int(span.get("flags", 0)) & _STYLED_FLAGS)
            if line_index > 0 and span_index == 0:
                text = " " + text
            out.append((text, styled, line_index))
    return out


def _split_heading(spans: list[tuple[str, bool, int]]) -> tuple[str | None, str, int]:
    """Split a block's spans into (heading, body_text, body_line_count).

    If the block opens with a styled (bold/italic) run whose text is a section number
    ("4.2.2"), that whole leading styled run is the heading — covering both a heading
    that fills its entire block ("4.1" then "Leveraging LLMs...", each its own PDF
    line) and a "run-in" heading immediately followed by body prose on the same PDF
    line ("4.1.1" then "Search Data Refinement." then, still styled, a period, then
    unstyled body text begins). Returns `(None, text, line_count)` unchanged when the
    block isn't a heading at all.
    """
    if not spans:
        return None, "", 1

    def _as_body(spans: list[tuple[str, bool, int]]) -> tuple[None, str, int]:
        text = "".join(t for t, _, _ in spans)
        line_count = spans[-1][2] + 1
        return None, text, line_count

    if not spans[0][1] or not _NUM_RE.match(spans[0][0].strip()):
        return _as_body(spans)

    i = 0
    while i < len(spans) and spans[i][1]:
        i += 1
    heading_text = "".join(t for t, _, _ in spans[:i]).strip()
    if len(heading_text) > _HEADING_MAX_CHARS or not _FULL_RE.match(heading_text):
        return _as_body(spans)

    body_spans = spans[i:]
    if not body_spans:
        return heading_text, "", 1
    body_text = "".join(t for t, _, _ in body_spans).strip()
    body_line_count = spans[-1][2] - body_spans[0][2] + 1
    return heading_text, body_text, body_line_count


def extract_pages(path: Path) -> tuple[list[TextBlock], int]:
    """Per-block text via PyMuPDF, splitting each block into a heading (if any) and
    its body text.

    Each PDF "block" (roughly a paragraph or heading, as PyMuPDF segments it) yields
    zero, one or two `TextBlock`s: a heading block when the text opens with a styled
    section number, a body block for whatever text remains, or just the one or the
    other when the block is purely a heading or purely body text. See
    `_split_heading` for exactly how a heading is recognised.
    """
    doc = fitz.open(str(path))
    try:
        blocks: list[TextBlock] = []
        for page_index in range(doc.page_count):
            page = doc[page_index]
            raw = page.get_text("dict")
            for block in raw.get("blocks", []):
                if block.get("type") != 0:  # skip images/drawings
                    continue
                spans = _flatten_spans(block)
                if not spans:
                    continue
                heading, body, body_lines = _split_heading(spans)
                if heading:
                    blocks.append(TextBlock(text=heading, page=page_index + 1, is_heading=True))
                if body.strip():
                    blocks.append(
                        TextBlock(
                            text=body.strip(),
                            page=page_index + 1,
                            line_count=max(1, body_lines),
                        )
                    )
        return blocks, doc.page_count
    finally:
        doc.close()


def ingest(filename: str | None = None, reset: bool = False) -> IngestResponse:
    settings = get_settings()
    embedder = get_embedder()
    store = get_store()

    path = _find_pdf(filename)
    blocks, page_count = extract_pages(path)
    chunks = chunk_pages(blocks)
    if not chunks:
        raise ValueError(f"{path.name} produced no text — is it a scanned/image PDF?")

    # Embed in batches. is_query=False marks these as documents ("passage:" for e5).
    vectors: list[list[float]] = []
    batch = 32
    for i in range(0, len(chunks), batch):
        texts = [c.text for c in chunks[i : i + batch]]
        vectors.extend(embedder.embed(texts, is_query=False))

    store.ensure_collection(dim=len(vectors[0]), reset=reset)

    points = [
        models.PointStruct(
            id=str(uuid.uuid5(_NAMESPACE, f"{path.name}:{c.index}")),
            vector=vec,
            payload={"text": c.text, "page": c.page, "section": c.section, "source": path.name},
        )
        for c, vec in zip(chunks, vectors)
    ]
    store.upsert(points)

    return IngestResponse(
        document=path.name,
        pages=page_count,
        chunks=len(chunks),
        collection=settings.qdrant_collection,
    )
