# Level 3 implementation notes

What changed to resolve the `TODO(level-3)` marker — whole-document reasoning. Written
so it can be pasted straight into the technical note.

## The problem

A single embed + single search finds the one passage closest to the question as a
whole. That fails three ways a Level-1 question never hits:

- **Multi-hop** — "compare the retriever, reranker and reader" needs three passages
  from three different sections; the mixed query's embedding sits somewhere between
  all of them and often ranks none of them highly.
- **Table + text** — a value in a table and the sentence elsewhere that explains it
  rarely share vocabulary, so one search surfaces one or the other, not both.
- **Whole-document synthesis** — "summarise the contribution of each section" isn't a
  ranking problem at all. There is no passage that is "closest" to the query; the
  question needs one passage *per section*, and a top-k search will always favour
  whichever sections happen to embed closest to the phrasing.

## 1. Multi-query decomposition — one search → fan-out + fuse

**File:** [`app/rag/retrieve.py`](app/rag/retrieve.py) — `decompose_query()`, `retrieve()`

`decompose_query()` asks the chat model to break the (rewritten) question into 2–4
focused sub-queries — one per hop, or one per fact the question needs — and to just
repeat the question unchanged if it's already answerable from a single search. Each
sub-query goes through the existing Level-1 hybrid search (`_search()`, unchanged
dense+BM25 RRF) independently, and the results are merged with `dedupe_contexts()`
(keeps the highest score per `(page, text prefix)` pair, so a chunk retrieved by two
sub-queries isn't duplicated in the prompt).

This is the fix for multi-hop and table+text questions: each sub-query gets its own
ranked search, so a passage that would be drowned out in one combined query has its
own shot at the top-k.

## 2. Section coverage — a "second index" for synthesis questions

**File:** [`app/rag/retrieve.py`](app/rag/retrieve.py) — `_section_coverage()`,
`_wants_full_coverage()`
**File:** [`app/vectorstore/qdrant_store.py`](app/vectorstore/qdrant_store.py) —
`VectorStore.list_sections()`, `VectorStore.points_for_section()`

No amount of query rewriting turns "summarise every section" into a ranking problem —
it needs a chunk from every section, full stop. `_wants_full_coverage()` matches the
question against a short list of synthesis phrasings ("each section", "whole
document", "summarise the document", ...). When it fires, `_section_coverage()` pulls
one real chunk per section by an **exact structural filter** on the `section` payload
field already stored at ingest time (`list_sections()` enumerates the distinct section
names once, cached like the existing BM25 index; `points_for_section()` does a
`scroll()` with a `FieldCondition` filter, not a ranked search) — the closest thing to
a second, coarser index this scaffold has, without standing up a second collection.

These structural chunks are still the model's real paragraph text, not an LLM-written
summary — the citation grounding requirement (`evidence_quote` must appear verbatim on
the cited page) holds for them exactly as it does for a normally-retrieved chunk. An
earlier design considered embedding one LLM-generated summary per section as a second
Qdrant collection; it was dropped because a summary sentence quoted back as `sources`
would not appear verbatim in the PDF — ungrounded citations score worse than not
attempting synthesis at all.

## 3. One bounded retrieve-reason-retrieve hop

**File:** [`app/rag/pipeline.py`](app/rag/pipeline.py) — `_find_gap_query()`

After the first retrieval pass, `_find_gap_query()` shows the model what was retrieved
and asks a yes/no-shaped question: is this enough, or is there a specific missing
piece? If it names one, `answer()` runs exactly one more `retrieve(..., level=1)` call
for that gap query and merges it in via `dedupe_contexts()` before generating the
final answer.

This is deliberately **one hop, not a loop** — it catches the common case where
decomposition missed a fact, without turning every Level-3 answer into an open-ended
agent loop with unbounded latency and cost. If the LLM call fails or returns `ENOUGH`,
the pipeline proceeds with what it already has — same fail-safe pattern as
`rewrite_query()` in Level 2.

## Wiring

`retrieve()` gained a `level: int = 1` parameter (default preserves Level 1/2
behaviour exactly — same single search as before). `pipeline.answer()` now passes
`level=req.level` through, and only runs the gap-check hop when `req.level == 3`.

## Context budget

Level 1/2 still return `top_k` contexts, unchanged. Level 3 without full-coverage caps
at `min(len(merged), 12)` (typically 2–4 sub-queries × top_k, deduped); with
full-coverage triggered, the cap rises to 20 so the section backfill isn't cut off by
a small `top_k`. Both are prompt-size trade-offs, not accuracy limits — raise them if
the document has many more than ~20 sections.

## Verified end to end

Ran against the ingested document (`data/in/paper.pdf`) with Ollama / `gemma4:e4b`:

- *"Compare the roles of the retriever, reranker and reader in a modern Information
  Retrieval pipeline"* — decomposed into per-role sub-queries; the answer cited 12
  distinct passages across pages 2, 4, 5, 12, 19, 25, 33 and 49, each role backed by
  its own section.
- *"Summarise the contribution of each section of the document"* — triggered the
  full-coverage path; 20 sources spanning pages 2–53, one section-by-section, cited
  quote from each.

`data/logs/llm_calls.jsonl` shows the expected call sequence per question: rewrite →
decompose → (search, no LLM call) → gap-check → final answer.

## Not done (deliberately out of scope)

- **Table-aware retrieval beyond Level 1's whole-chunk tables.** Tables are already
  kept intact as one chunk (Level 1); Level 3 does not add table-specific parsing
  (e.g. row/column-aware querying) — decomposition treats a table chunk like any other.
- **Multi-hop beyond one gap-check hop.** A question needing three sequential hops
  (A → B → C, where C's query depends on what B's retrieval returns) is not handled;
  the gap-check only fires once.
- **Caching decomposition/gap-check results.** Every Level-3 question costs 2–3 extra
  LLM round trips before the final answer (decompose, gap-check, and the query
  rewrite it shares with Level 2). Fine at hackathon volumes; would want caching or
  skipping short-circuits at real scale.
