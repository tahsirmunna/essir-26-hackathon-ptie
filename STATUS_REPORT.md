# Status report — ESSIR '26 hackathon RAG pipeline

_Generated 2026-07-31 from the current state of `main` (HEAD `5fe77b2`)._

## Level 1 — Retrieval (q1–q3) — Implemented

All five `TODO(level-1)` markers resolved (commits `930031d`, `4fdb3d2`, `bf7ff9a`, `5fe77b2`):

- **Extraction** ([app/rag/ingest.py](app/rag/ingest.py)) — swapped `pypdf` for **PyMuPDF**
  (`fitz`), which segments a page into layout blocks and reports font size/bold per span.
  `extract_pages()` returns `list[TextBlock]`, splitting each block into a heading (if it opens
  with a bold/italic section number like "4.2.2") and its body text, so headings are detected
  structurally instead of guessed from a font-size ratio.
- **Chunking** ([app/rag/chunking.py](app/rag/chunking.py)) — one-page-per-vector replaced with
  section-aware, paragraph-level chunks. Blocks are grouped into sections by heading, each
  remaining paragraph becomes one chunk (no char-limit/overlap needed), and tables/figures are
  kept intact as one chunk rather than split by paragraph. Every `Chunk` keeps `page` and
  `section` for citations.
- **Precise citations** ([app/rag/pipeline.py](app/rag/pipeline.py):39-62) —
  `_best_supporting_sentence()` splits the retrieved chunk into sentences and picks the one with
  the most word overlap with the generated answer (falling back to the question) as the evidence
  quote, instead of returning the whole chunk truncated to 300 chars.
- **Hybrid search** ([app/vectorstore/qdrant_store.py](app/vectorstore/qdrant_store.py):95-134) —
  dense Qdrant cosine search over-fetches a larger candidate pool, fused with an in-memory BM25
  pass (`rank-bm25`) via Reciprocal Rank Fusion (`1/(60+rank)` summed across both rankings).
- **Batched embeddings** ([app/llm/ollama.py](app/llm/ollama.py)) — `embed()` tries Ollama's
  batched `/api/embed` endpoint first, falling back to one call per text on older servers.
- **Extras**: local models routed through LM Studio/Ollama; an LLM-call logger
  ([app/llm/logging_client.py](app/llm/logging_client.py)) appends every `chat()` call
  (prompt + reply) to `data/logs/llm_calls.jsonl`, added in the latest commit.

Full write-up: [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md).

## Level 2 — Conversational memory (q4–q6) — Implemented

Both `TODO(level-2)` markers resolved (commit `bbc818d`):

- **Memory** ([app/rag/memory.py](app/rag/memory.py)) — the process-local `dict` was replaced
  with one running summary per conversation, persisted to `data/memory/<conversation_id>.txt`.
  `append()` folds each new turn into the existing summary via an LLM call (degrading to a plain
  text append if the call fails); `get_history()` returns it as a single `system` message;
  `reset()` deletes the file. `conversation_id` is `f"level-{req.level}"`, so every question at a
  level shares one summary.
- **Query rewriting** ([app/rag/retrieve.py](app/rag/retrieve.py):34-53) — `rewrite_query()`
  resolves a follow-up ("why does that happen?") into a standalone query against the flattened
  history before it's embedded. Returns the question unchanged if there's no history, and falls
  back to the original question if the LLM call fails — a no-op is a safer failure mode than a
  broken query.

Verified in practice: `data/memory/level-1.txt` and `level-2.txt` contain real running summaries
from prior test runs, and `data/logs/llm_calls.jsonl` shows the corresponding LLM traffic.

Full write-up: [IMPLEMENTATION_NOTES_LEVEL2.md](IMPLEMENTATION_NOTES_LEVEL2.md).

## Level 3 — Whole-document reasoning (q7–q9) — Implemented

The `TODO(level-3)` marker in `app/rag/retrieve.py` is resolved:

- **Multi-query decomposition** ([app/rag/retrieve.py](app/rag/retrieve.py) —
  `decompose_query()`) — an LLM call breaks the question into 2–4 focused sub-queries when it
  needs more than one hop; each goes through the existing Level-1 hybrid search independently,
  merged and deduped (`dedupe_contexts()`). Handles multi-hop and table+text questions.
- **Section coverage backstop** (`_section_coverage()` in `retrieve.py`, plus
  `VectorStore.list_sections()`/`points_for_section()` in
  [app/vectorstore/qdrant_store.py](app/vectorstore/qdrant_store.py)) — for whole-document
  synthesis questions ("summarise every section"), pulls one real chunk per section by an exact
  structural filter rather than a ranked search, so citations stay grounded (never an
  LLM-written summary quoted as if it were the document).
- **One bounded retrieve-reason-retrieve hop** ([app/rag/pipeline.py](app/rag/pipeline.py) —
  `_find_gap_query()`) — after the first retrieval pass, one LLM call checks whether anything
  critical is missing and, if so, triggers exactly one more targeted retrieval before the final
  answer is generated.

Verified end to end against the ingested document with Ollama/`gemma4:e4b`: a multi-hop
"compare retriever/reranker/reader" question correctly cited 12 passages across 8 pages; a
"summarise the contribution of each section" question triggered the coverage backstop and cited
20 passages spanning the whole document. Full write-up:
[IMPLEMENTATION_NOTES_LEVEL3.md](IMPLEMENTATION_NOTES_LEVEL3.md).

## Submission deliverables — Not filled in

| Item | State |
|---|---|
| `submission/team.json` | Empty — `team_name`, `members`, `repo_url` all blank |
| `submission/level-1/q1–q3.json` | Empty `{}` |
| `submission/level-2/q4–q6.json` | Empty `{}` |
| `submission/level-3/q7–q9.json` | Empty `{}` |
| `data/out/` | Empty except `.gitkeep` — no answer has been saved by a full `/query` run recently |
| `TECHNICAL_NOTE.md` | Does not exist (template at [templates/technical-note.md](templates/technical-note.md)) |

A candidate set of nine questions exists in the **untracked** `data/in/question.txt`
(3 per level, sensible level split — see below) but is not in the `submission/` JSON format and
isn't committed:

```
Level 1: What is the purpose of the reranker? / What formats can rewritten queries take? /
         What are search agents?
Level 2: How is that different from the retriever? / Which of those works particularly well
         with sparse retrievers? / How do they differ from conventional IR systems?
Level 3: Compare the roles of the retriever, reranker and reader in a modern IR pipeline /
         How do prompting, SFT and RL differ for query rewriting? / How do LLMs contribute
         differently to conversational vs. ad-hoc search?
```

## Outstanding work, in priority order

1. **Run all nine questions** through `POST /query`, copy the best result from `data/out/` into
   the matching `submission/level-N/qX.json`. (A draft set of nine questions already exists in
   the untracked `data/in/question.txt`.)
2. **Fill `submission/team.json`.**
3. **Write `TECHNICAL_NOTE.md`** — the content already exists across
   `IMPLEMENTATION_NOTES.md` / `IMPLEMENTATION_NOTES_LEVEL2.md` / `IMPLEMENTATION_NOTES_LEVEL3.md`
   and just needs adapting to the template.
4. **Validate** with [ai_skill/VALIDATE_SUBMISSION.md](ai_skill/VALIDATE_SUBMISSION.md) before
   pushing, and commit `data/in/*.pdf` alongside the code.
