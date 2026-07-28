# Level 1 implementation notes

What changed to resolve the five `TODO(level-1)` markers, and how the local LM Studio
setup fits together. Written so it can be pasted straight into the technical note.

## 1. PDF extraction — pypdf → PyMuPDF

**File:** [`app/rag/ingest.py`](app/rag/ingest.py) — `extract_pages()`

`pypdf` returns plain per-page strings with no layout information, so there was no way
to tell a heading from a paragraph, or to cope well with two-column layouts, ligatures
and hyphenation. Swapped it for **PyMuPDF** (`fitz`), which segments a page into layout
blocks and reports each block's font size and bold flag.

`extract_pages()` now returns `list[TextBlock]` (one entry per paragraph-like block,
each carrying its 1-indexed page, average font size, and whether it's bold) plus the
total page count. This is a prerequisite for subsection chunking below — the chunker
needs that font/bold signal to find headings.

`pypdf` was dropped from `pyproject.toml`/`uv.lock`; `pymupdf>=1.24` was added.

## 2. Chunking — one-vector-per-page → subsection-aware chunks

**File:** [`app/rag/chunking.py`](app/rag/chunking.py) — `chunk_pages()`

Previously each whole page became one vector: too coarse to retrieve precisely, and
too long for the embedding model on dense pages. The new pipeline:

1. **Detect headings.** A block is a heading if its font size is ≥1.15× the document's
   median body-text size, or it's bold, is a single line, under 120 characters, and
   doesn't trail off like a sentence (`_is_heading`).
2. **Group into sections.** Walk the blocks in document order; every heading starts a
   new section, and non-heading blocks accumulate under it (`_sectionize`). A document
   with no detectable headings degenerates to one big "section" — the fallback below
   still applies, so nothing breaks on a plain, unstructured PDF.
3. **Size-split oversized sections.** A subsection that fits under `CHUNK_SIZE` becomes
   one chunk as-is. A longer one is split on sentence boundaries (`_atoms`) and packed
   back together up to `CHUNK_SIZE` with a `CHUNK_OVERLAP`-character tail carried into
   the next piece (`_pack`), so a fact is never cut mid-sentence.

Every `Chunk` keeps `page` (for citations) and `section` (the heading it sits under,
stored in the Qdrant payload — not yet used for retrieval, but there for a Level-3 pass).

## 3. Precise citations — whole chunk → best supporting sentence

**File:** [`app/rag/pipeline.py`](app/rag/pipeline.py) — `_best_supporting_sentence()`,
`_sources_from()`

The baseline returned the entire retrieved chunk (truncated to 300 chars) as the
"quote". Now each chunk is split into sentences, and the sentence with the most
lexical word-overlap with the **generated answer** is picked as the quote (falling back
to overlapping with the question if the answer shares no vocabulary with the chunk —
e.g. a degraded "LLM unavailable" response). If no sentence overlaps at all, the full
chunk is kept rather than guessing — a safe, if broad, citation beats a confidently
wrong precise one.

This is a deterministic, no-extra-LLM-call heuristic (cheap, explainable, good for the
technical note/defence) rather than asking the model to point at its own quote.

## 4. Hybrid search — dense-only → dense + BM25 (RRF fusion)

**File:** [`app/vectorstore/qdrant_store.py`](app/vectorstore/qdrant_store.py) —
`VectorStore.search()`, `VectorStore._bm25()`

Embeddings blur exact terms — names, acronyms, numbers — that lexical matching catches
easily. Added a BM25 index (`rank-bm25`, pure Python, no extra service) built lazily
from the same payload text already stored in Qdrant, cached on the `VectorStore`
instance and invalidated on `upsert()`/`ensure_collection()`.

`search()` now over-fetches a larger dense candidate pool (`top_k * 4`), scores that
pool by BM25 against the query text, and re-ranks the pool with **Reciprocal Rank
Fusion** (`1/(60+rank)` summed across the two rankings). RRF was chosen over a weighted
score blend because it needs no normalisation between cosine similarity and BM25
scores, which live on unrelated scales.

`retrieve()` in [`app/rag/retrieve.py`](app/rag/retrieve.py) now passes the (rewritten)
query text through to `search()` so the fusion has something to score against.

Added `rank-bm25>=0.2` to `pyproject.toml`.

## 5. Batched embeddings — Ollama's `/api/embed`

**File:** [`app/llm/ollama.py`](app/llm/ollama.py) — `OllamaClient.embed()`

`embed()` tries the batched `/api/embed` endpoint first (one round trip for the whole
list of texts). If that fails — older Ollama builds don't have it — it falls back to
the original one-`/api/embeddings`-call-per-text loop, so nothing regresses on an older
server.

## Local models via LM Studio

`.env` (created from `.env.example`) now routes **both** the chat model and the
embedding model through LM Studio, so there's one local server and one place to swap
models:

```
LLM_PROVIDER=lmstudio
CHAT_MODEL=google/gemma-4-e2b          # must match the id LM Studio shows
LMSTUDIO_BASE_URL=http://localhost:1234

EMBEDDING_BACKEND=provider              # was sentence-transformers (local, no server)
EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
```

`EMBEDDING_BACKEND=provider` makes `app/rag/embeddings.py` reuse the LM Studio client's
`embed()` (`app/llm/lmstudio.py`) instead of downloading a local sentence-transformers
model — see `ProviderEmbedder` in `embeddings.py`.

**What still needs a one-time manual step in the LM Studio app** (installed via
`brew install --cask lm-studio` in this session — GUI-only steps can't be scripted):

1. Open LM Studio → **Discover**, download a chat model (e.g. a Gemma variant matching
   `CHAT_MODEL`) and an embedding model (e.g. `nomic-embed-text-v1.5`).
2. Load both models.
3. Open the **Developer** tab → **Start Server** (default port `1234`).
4. Confirm the exact model ids LM Studio reports match `CHAT_MODEL` /
   `EMBEDDING_MODEL` in `.env` — LM Studio's id is not always the download name.

## Dependency changes

`pyproject.toml` / `uv.lock` (regenerated with `uv lock`):

- Removed: `pypdf`
- Added: `pymupdf>=1.24`, `rank-bm25>=0.2`

## Not done (deliberately out of scope for Level 1)

- **Level 2** (`rag/memory.py`, `retrieve.py::rewrite_query`) and **Level 3**
  (multi-hop retrieval, payload filtering, query-scoped search) TODOs are untouched —
  they're separate levels with their own scope.
- `Diagnostics.tokens` (reporting real token usage) is an unlabelled `TODO`, not a
  Level-1 item, and was left as `None`.
