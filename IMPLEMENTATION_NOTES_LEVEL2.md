# Level 2 implementation notes

What changed to resolve the two `TODO(level-2)` markers — conversational memory and
query rewriting. Written so it can be pasted straight into the technical note.

## 1. Conversation memory — process-local dict → persisted running summary

**File:** [`app/rag/memory.py`](app/rag/memory.py)

The baseline kept history in a process-local `_STORE: dict[str, list[Message]]`: fine
to demonstrate the idea, but it forgets everything on restart and can't be shared
across more than one worker. Rather than add infrastructure (Redis, Postgres) to fix
that, each conversation now gets **one running summary**, persisted as a plain text
file under `memory_dir` (`data/memory/<conversation_id>.txt`, new setting in
[`app/config.py`](app/config.py)):

- `append(conversation_id, user, assistant)` reads the existing summary (if any),
  sends it plus the new turn to the chat model with a prompt that asks it to fold the
  turn into an updated summary — a few sentences, keeping any fact or entity a
  follow-up might refer to ("the dataset", "that table on page 12") — and overwrites
  the file with the result. If the LLM call fails, it degrades to a plain text append
  of the raw turn rather than losing it.
- `get_history(conversation_id)` reads that file and returns it as a single `system`
  message (`"Summary of the conversation so far:\n..."`), so callers still see the
  `list[Message]` shape they expect — `pipeline.py::_build_messages` extends the
  prompt with it unchanged, and `retrieve.py::rewrite_query` (below) consumes it the
  same way.
- `reset(conversation_id)` deletes the file.

`conversation_id` is already `f"level-{req.level}"` (assigned in
[`app/rag/pipeline.py`](app/rag/pipeline.py)), so every question at a given level
shares one summary — that's how a Level-2 follow-up sees the prior turn.

## 2. Query rewriting — no-op → LLM-rewritten standalone query

**File:** [`app/rag/retrieve.py`](app/rag/retrieve.py) — `rewrite_query()`

A follow-up like *"and the test split?"* has no retrievable content on its own —
embedding it directly returns noise. `rewrite_query(question, history)`:

1. Returns `question` unchanged immediately if there's no history (genuinely
   standalone questions, or the first turn of a conversation) — no LLM call, no
   added latency.
2. Otherwise flattens `history` into `"role: content"` lines (in practice, the one
   summary message from `memory.get_history`) and asks the chat model to rewrite the
   question into a single self-contained query, resolving pronouns/implicit
   references against that context. The prompt explicitly tells the model to repeat
   the question unchanged if it already stands on its own, so it doesn't over-rewrite
   simple follow-ups.
3. Falls back to the original `question` if the LLM call raises `LLMError` or returns
   an empty string — a no-op is a safer failure mode than retrieving on a broken
   query.

`retrieve()` already called `rewrite_query()` and passed its result into both the
embedder and the BM25 fusion pass (`query_text=query` in
`store.search()`) — that wiring was untouched, only the function body changed.

## Example

```
Turn 1 — q: "How large is the training split of the CIFAR-10 experiments?"
         summary after: "Discussed CIFAR-10 experiments; training split size given."

Turn 2 (level-2) — q: "And the test split?"
  history:  [{"role": "system", "content": "Summary of the conversation so far:
             Discussed CIFAR-10 experiments; training split size given."}]
  rewritten query: "How large is the test split of the CIFAR-10 experiments?"
```

## Not done (deliberately out of scope for Level 2)

- **Multi-worker safety.** The summary file assumes a single writer; two workers
  appending to the same `conversation_id` concurrently would race on the same file.
  Acceptable for this scaffold's single-process run, called out for anyone scaling it
  up.
- **Level 3** (multi-hop retrieval, payload filtering, query-scoped search) is
  untouched — separate level, separate scope.
- No caching/dedup on the rewrite call — every follow-up question costs one extra LLM
  round trip before retrieval even starts. Fine at hackathon query volumes; would want
  measuring (and possibly skipping when the summary is short/stale) at real scale.
