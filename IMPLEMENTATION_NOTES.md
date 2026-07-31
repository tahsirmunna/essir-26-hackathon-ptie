# Implementation notes

What changed to resolve the `TODO(level-N)` markers across all three levels. Written so
it can be pasted straight into the technical note.

## Level 1 — Retrieval (q1–q3)

- **Extraction** ([app/rag/ingest.py](app/rag/ingest.py)) — `pypdf` → **PyMuPDF**
  (`fitz`), which reports each span's font size/bold flag, so `extract_pages()` detects
  headings structurally (a bold/italic section number like "4.2.2") instead of guessing
  from font-size ratios. Copes far better with two-column layouts, ligatures, hyphenation.
- **Chunking** ([app/rag/chunking.py](app/rag/chunking.py)) — one-page-per-vector →
  section-aware, paragraph-level chunks. Blocks are grouped into sections by heading;
  each remaining paragraph becomes one chunk (no size limit/overlap needed since chunks
  are already paragraph-sized); a table/figure (recognised by its "Table N."/"Fig. N"
  caption) stays whole as one chunk instead of being split. Every `Chunk` keeps `page`
  and `section` for citations.
- **Precise citations** ([app/rag/pipeline.py](app/rag/pipeline.py) —
  `_best_supporting_sentence()`) — instead of returning the whole chunk (truncated to
  300 chars) as the quote, each chunk is split into sentences and the one with the most
  word-overlap with the *generated answer* (falling back to the question) is picked. A
  deterministic, no-extra-LLM-call heuristic.
- **Hybrid search** ([app/vectorstore/qdrant_store.py](app/vectorstore/qdrant_store.py)
  — `VectorStore.search()`) — dense Qdrant cosine search over-fetches a larger candidate
  pool, fused with an in-memory BM25 pass (`rank-bm25`) via Reciprocal Rank Fusion
  (`1/(60+rank)` summed across both rankings) — catches exact terms (names, numbers,
  acronyms) embeddings blur, with no extra service.
- **Batched embeddings** ([app/llm/ollama.py](app/llm/ollama.py)) — `embed()` tries
  Ollama's batched `/api/embed` endpoint first, falls back to one call per text on
  older servers.
- **Local models** — chat + embeddings routed through LM Studio/Ollama
  (`EMBEDDING_BACKEND=provider` reuses the chat provider's `embed()`, see
  `ProviderEmbedder` in [app/rag/embeddings.py](app/rag/embeddings.py)); an LLM-call
  logger ([app/llm/logging_client.py](app/llm/logging_client.py)) appends every
  `chat()` call to `data/logs/llm_calls.jsonl` for inspection.

Dependencies: removed `pypdf`; added `pymupdf>=1.24`, `rank-bm25>=0.2`.

## Level 2 — Conversational memory (q4–q6)

- **Memory** ([app/rag/memory.py](app/rag/memory.py)) — the process-local `dict` was
  replaced with **one running summary per conversation**, persisted to
  `data/memory/<conversation_id>.txt`. `append()` folds each new turn into the existing
  summary via an LLM call (degrades to a plain-text append if the call fails);
  `get_history()` returns it as a single `system` message; `reset()` deletes the file.
  `conversation_id` is `f"level-{req.level}"`, so every question at a level shares one
  summary — that's how a follow-up sees the prior turn.
- **Query rewriting** ([app/rag/retrieve.py](app/rag/retrieve.py) —
  `rewrite_query()`) — a follow-up like "why does that happen?" has no retrievable
  content on its own. `rewrite_query()` asks the chat model to resolve it into a
  standalone query against the flattened history before it's embedded. No-ops (no LLM
  call) when there's no history, and falls back to the original question if the LLM
  call fails — a no-op is a safer failure mode than a broken query.

## Level 3 — Whole-document reasoning (q7–q9)

Single embed + single search fails three ways Level 1 never hits: **multi-hop**
questions whose combined embedding ranks no single passage highly, **table + text**
questions where the two passages share no vocabulary, and **whole-document synthesis**
("summarise every section"), which isn't a ranking problem — it needs one passage
*per section*, not the top-k closest to the phrasing.

- **Multi-query decomposition** ([app/rag/retrieve.py](app/rag/retrieve.py) —
  `decompose_query()`) — an LLM call breaks the question into 2–4 focused sub-queries
  (or repeats it unchanged if one search suffices). Each sub-query runs through the
  unchanged Level-1 hybrid search independently; results are merged and deduped
  (`dedupe_contexts()`, keeps the highest score per `(page, text prefix)`). Fixes
  multi-hop and table+text questions.
- **Section coverage backstop** (`_section_coverage()` / `_wants_full_coverage()` in
  `retrieve.py`, plus `VectorStore.list_sections()` / `points_for_section()` in
  [app/vectorstore/qdrant_store.py](app/vectorstore/qdrant_store.py)) — when the
  question matches a synthesis phrasing ("each section", "whole document", ...), pulls
  one **real** chunk per section via an exact structural filter on the `section`
  payload field (not a ranked search, not an LLM-written summary) — the closest thing
  to a second, coarser index this scaffold has, without a second Qdrant collection or
  quoting anything that isn't verbatim in the PDF.
- **One bounded retrieve-reason-retrieve hop**
  ([app/rag/pipeline.py](app/rag/pipeline.py) — `_find_gap_query()`) — after the first
  retrieval pass, one LLM call checks whether a specific piece is still missing; if so,
  exactly one more targeted `retrieve()` call runs before the final answer is
  generated. Deliberately one hop, not an open-ended loop, to keep latency/cost bounded.
- **Wiring** — `retrieve()` gained a `level: int = 1` parameter (default preserves
  Level 1/2 behaviour exactly); `pipeline.answer()` passes `level=req.level` through
  and only runs the gap-check hop at level 3. Context cap: `min(len(merged), 12)`
  normally, raised to 20 when the section backstop fires.

**Verified end to end** (Ollama / `gemma4:e4b`, ingested `data/in/paper.pdf`):
- *"Compare the roles of the retriever, reranker and reader..."* → decomposed
  per-role, cited 12 passages across 8 pages.
- *"Summarise the contribution of each section..."* → triggered the coverage
  backstop, cited 20 passages spanning the whole document.

`data/logs/llm_calls.jsonl` shows the expected call sequence per Level-3 question:
rewrite → decompose → gap-check → final answer.

## Deliberately out of scope

- **Level 3 table-aware retrieval** beyond Level 1's whole-chunk tables — decomposition
  treats a table chunk like any other; no row/column-aware querying.
- **Multi-hop beyond one gap-check hop** — a question needing three sequential,
  dependent hops (C's query depends on what B's retrieval returns) isn't handled.
- **Multi-worker safety** for the memory summary file — a single writer is assumed;
  concurrent workers appending to the same `conversation_id` would race.
- **Caching** on the rewrite/decompose/gap-check LLM calls — every Level-2 follow-up
  costs one extra round trip, every Level-3 question costs two to three. Fine at
  hackathon volumes; would want caching or short-circuits at real scale.
- **`Diagnostics.tokens`** (real token usage) — left as `None`, not implemented for any
  provider.
