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


def retrieve(question: str, top_k: int, history: list[Message] | None = None) -> list[Context]:
    embedder = get_embedder()
    store = get_store()

    query = rewrite_query(question, history or [])

    # TODO(level-3): one query + one search is not enough for whole-document
    #   questions ("summarise every chapter", "combine the table on p.40 with the
    #   reference on p.90"). Consider multi-query fan-out, iterative/agentic retrieval
    #   (retrieve -> reason -> retrieve again), or a second index (e.g. a graph or a
    #   per-section summary index) alongside this one.
    vector = embedder.embed([query], is_query=True)[0]
    # query_text enables the BM25 fusion pass in VectorStore.search (hybrid retrieval).
    hits = store.search(vector, top_k, query_text=query)

    return [
        Context(
            text=str(h.payload.get("text", "")),
            page=int(h.payload.get("page", 0)),
            score=float(h.score),
            section=h.payload.get("section") or None,
        )
        for h in hits
    ]
