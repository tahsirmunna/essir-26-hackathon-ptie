"""Load a PDF into the vector store.

    parse PDF (data/in) -> layout blocks -> [chunk by subsection] -> embeddings -> Qdrant

Extraction uses PyMuPDF (`fitz`) instead of pypdf: it reports each text block's font size
and boldness, which is what `chunking.py` uses to tell headings apart from body text and
group content into subsections. PyMuPDF also copes far better than pypdf with two-column
layouts, ligatures and hyphenation, so the text it returns matches the document closely
enough that page citations stay correct.
"""

from __future__ import annotations

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

# PDF "flags" bit for bold text (PyMuPDF span flags, bit 4). See
# https://pymupdf.readthedocs.io/en/latest/textpage.html#span-dictionary
_BOLD_FLAG = 1 << 4


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


def extract_pages(path: Path) -> tuple[list[TextBlock], int]:
    """Per-block text via PyMuPDF, keeping font size and boldness for each block.

    Each PDF "block" (roughly a paragraph or heading line, as PyMuPDF segments it)
    becomes one `TextBlock`, tagged with the 1-indexed page it came from. Font size and
    boldness let `chunking.py` detect headings and chunk by subsection instead of by
    arbitrary character windows.
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
                lines_text: list[str] = []
                sizes: list[float] = []
                bold_flags: list[bool] = []
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = "".join(span.get("text", "") for span in spans)
                    if line_text.strip():
                        lines_text.append(line_text)
                    for span in spans:
                        sizes.append(float(span.get("size", 0.0)))
                        bold_flags.append(bool(int(span.get("flags", 0)) & _BOLD_FLAG))
                text = "\n".join(lines_text).strip()
                if not text:
                    continue
                avg_size = sum(sizes) / len(sizes) if sizes else 0.0
                blocks.append(
                    TextBlock(
                        text=text,
                        page=page_index + 1,
                        font_size=avg_size,
                        bold=any(bold_flags),
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
