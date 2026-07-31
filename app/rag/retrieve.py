"""Find the chunks most relevant to a question."""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.base import LLMError, Message
from ..llm.factory import get_client
from ..vectorstore.qdrant_store import get_store
from .embeddings import get_embedder


@dataclass
class Context:
    text: str
    page: int
    score: float
    section: str | None = None


def dedupe_contexts(contexts: list[Context]) -> list[Context]:
    """Merge context lists, keeping the highest score per (page, text) pair.

    Used both internally (fusing hits from several sub-queries) and by the pipeline's
    level-3 gap-check, which retrieves one more batch of context after the first pass.
    """
    best: dict[tuple[int, str], Context] = {}
    for c in contexts:
        key = (c.page, c.text[:80])
        if key not in best or c.score > best[key].score:
            best[key] = c
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


_REWRITE_PROMPT = (
    "You rewrite follow-up questions into standalone search queries for a document "
    "retrieval system.\n\n"
    "Conversation so far:\n{context}\n\n"
    "Follow-up question: {question}\n\n"
    "If the follow-up question already stands on its own, repeat it unchanged. "
    "Otherwise, rewrite it into a single self-contained question that resolves any "
    "pronouns or implicit references (e.g. 'it', 'that table', 'the dataset') using "
    "the conversation above. Output only the rewritten question — no explanation, "
    "no quotes."
)


def rewrite_query(question: str, history: list[Message]) -> str:
    """Resolve a follow-up into a standalone search query.

    "And the test split?" has no retrievable content on its own, so embedding it
    returns noise. If there's history, ask the chat model to rewrite the question
    into something self-contained before it gets embedded. Genuinely standalone
    questions pass through unchanged, and so does anything if the LLM is unavailable
    — a no-op is a safer fallback than a broken query.
    """
    if not history:
        return question

    context = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    prompt = _REWRITE_PROMPT.format(context=context, question=question)
    try:
        rewritten = get_client().chat([{"role": "user", "content": prompt}]).strip()
    except LLMError:
        return question

    return rewritten or question


_DECOMPOSE_PROMPT = (
    "You break a question about a single document into focused search queries for a "
    "retrieval system over that document.\n\n"
    "Question: {question}\n\n"
    "If the question is answerable from one self-contained search, output just that one "
    "query, unchanged. Otherwise — a multi-hop question that chains facts from different "
    "parts of the document, or one that needs a value from a table combined with a "
    "statement elsewhere — break it into 2 to 4 sub-queries that together cover every "
    "fact it needs, one per line. Output only the queries, one per line, no numbering, "
    "no explanation."
)

_MAX_SUBQUERIES = 4

# Whole-document synthesis ("summarise every section") needs a chunk from every section,
# not just the ones a query happens to rank highest — no amount of query rewriting fixes
# that, since it isn't a retrieval-ranking problem.
_FULL_COVERAGE_HINTS = (
    "each section", "every section", "each chapter", "every chapter",
    "whole document", "entire document", "throughout the document",
    "summarise the document", "summarize the document",
    "across the document", "contribution of each",
)


def decompose_query(question: str) -> list[str]:
    """Break a whole-document question into the sub-queries it actually needs.

    A single embed+search finds the passage closest to the question as a whole, but a
    multi-hop question ("how does X, introduced on p.4, relate to Y in the results
    table on p.20?") needs two different passages that individually rank far apart.
    Decomposing into separate, focused sub-queries and searching each one lets both
    surface, instead of one drowning out the other in a single ranked list.
    """
    try:
        raw = get_client().chat([{"role": "user", "content": _DECOMPOSE_PROMPT.format(question=question)}])
    except LLMError:
        return [question]

    queries = [q.strip(" \t-•").lstrip("0123456789.) ") for q in raw.splitlines()]
    queries = [q for q in queries if q]
    return queries[:_MAX_SUBQUERIES] or [question]


def _wants_full_coverage(question: str) -> bool:
    q = question.lower()
    return any(hint in q for hint in _FULL_COVERAGE_HINTS)


def _search(query_text: str, top_k: int) -> list[Context]:
    embedder = get_embedder()
    store = get_store()
    vector = embedder.embed([query_text], is_query=True)[0]
    # query_text enables the BM25 fusion pass in VectorStore.search (hybrid retrieval).
    hits = store.search(vector, top_k, query_text=query_text)
    return [
        Context(
            text=str(h.payload.get("text", "")),
            page=int(h.payload.get("page", 0)),
            score=float(h.score),
            section=h.payload.get("section") or None,
        )
        for h in hits
    ]


def _section_coverage(question: str) -> list[Context]:
    """One real chunk per document section, fetched by exact structural match rather
    than a ranked search — the "second index" for questions no ranking can serve."""
    if not _wants_full_coverage(question):
        return []
    store = get_store()
    out: list[Context] = []
    for name in store.list_sections():
        for payload in store.points_for_section(name, limit=1):
            out.append(
                Context(
                    text=str(payload.get("text", "")),
                    page=int(payload.get("page", 0)),
                    score=0.0,
                    section=name,
                )
            )
    return out


def retrieve(
    question: str, top_k: int, history: list[Message] | None = None, level: int = 1
) -> list[Context]:
    query = rewrite_query(question, history or [])

    if level != 3:
        return _search(query, top_k)

    # Level 3: fan out over decomposed sub-queries (multi-hop / table+text questions),
    # then backfill one chunk per section for whole-document synthesis questions —
    # neither is served by a single embed + single search.
    queries = decompose_query(query)
    contexts = dedupe_contexts([c for q in queries for c in _search(q, top_k)])
    contexts = dedupe_contexts(contexts + _section_coverage(question))

    limit = 20 if _wants_full_coverage(question) else min(max(len(contexts), top_k), 12)
    return contexts[:limit]
