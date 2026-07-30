"""Turn extracted PDF content into the units you index.

Chunking follows the document's own structure instead of an arbitrary character
window. `extract_pages` (see `ingest.py`) already splits each PDF block into a heading
(if any) and its body text — it finds section numbers like "4.1" or "4.2.2" from the
PDF's real bold/italic run of characters, not a guessed font-size ratio, so there is
nothing left to "detect" here: a block either IS a heading (`TextBlock.is_heading`) or
it is body text.

We group body blocks into sections at each heading, then turn each section into one
chunk per paragraph — except a table or figure (recognised by its "Table N."/"Fig. N"
caption), which stays together with the fragmented, multi-column rows that follow it
as a single chunk. Splitting a table by paragraph would scatter its cells across
unrelated chunks, which is exactly what looked "badly extracted" before. No size
limit, no overlap: a chunk is exactly one paragraph, or one whole table/figure, never
an arbitrary slice of either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CAPTION_RE = re.compile(r"^(table|fig(?:ure|\.)?)\s+\d", re.IGNORECASE)
_TABLE_LINE_RATIO = 40  # chars/line below this reads as fragmented table cells, not prose


@dataclass
class TextBlock:
    """One layout block from the PDF, as produced by `ingest.extract_pages`."""

    text: str
    page: int                  # 1-indexed
    is_heading: bool = False   # True if this whole block IS a section heading
    line_count: int = 1        # physical PDF lines this text was laid out on


@dataclass
class Chunk:
    text: str
    page: int          # 1-indexed
    index: int         # position within the document
    section: str | None = None   # nearest preceding heading, if any


def _sectionize(blocks: list[TextBlock]) -> list[tuple[str | None, list[TextBlock]]]:
    sections: list[tuple[str | None, list[TextBlock]]] = []
    title: str | None = None
    current: list[TextBlock] = []
    for block in blocks:
        if block.is_heading:
            if current:
                sections.append((title, current))
            title = block.text.strip()
            current = []
        else:
            current.append(block)
    if current:
        sections.append((title, current))
    return sections


def _looks_tabular(block: TextBlock) -> bool:
    """Many short physical lines for the amount of text means fragmented table
    cells (columns split at whitespace gaps), not a normally-wrapped paragraph."""
    return len(block.text) / max(1, block.line_count) < _TABLE_LINE_RATIO


def _group_blocks(blocks: list[TextBlock]) -> list[list[TextBlock]]:
    """One paragraph = one chunk, except a table/figure caption and whatever
    tabular-looking rows immediately follow it on the same page, which stay together.
    """
    groups: list[list[TextBlock]] = []
    table_buf: list[TextBlock] | None = None
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        same_page_as_table = table_buf is not None and block.page == table_buf[0].page
        if _CAPTION_RE.match(text):
            if table_buf:
                groups.append(table_buf)
            table_buf = [block]
        elif same_page_as_table and _looks_tabular(block):
            table_buf.append(block)
        else:
            if table_buf:
                groups.append(table_buf)
                table_buf = None
            groups.append([block])
    if table_buf:
        groups.append(table_buf)
    return groups


def chunk_pages(blocks: list[TextBlock]) -> list[Chunk]:
    """Group blocks into sections (by heading), then one chunk per paragraph — or
    per table/figure, kept whole (see `_group_blocks`)."""
    chunks: list[Chunk] = []
    idx = 0
    for title, section_blocks in _sectionize(blocks):
        for group in _group_blocks(section_blocks):
            text = "\n\n".join(b.text.strip() for b in group if b.text.strip())
            if not text:
                continue
            chunks.append(Chunk(text=text, page=group[0].page, index=idx, section=title))
            idx += 1
    return chunks
