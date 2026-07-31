"""Answer a question end to end.

    history + retrieved context  ->  prompt  ->  LLM  ->  grounded answer

This is what `POST /query` calls. You send only a question and its level; the system
assigns the id, threads the conversation (level-2 follow-ups share memory), produces the
answer, and writes it to `data/out/` as a JSON file you can later copy into `submission/`.
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ..config import get_settings
from ..llm.base import LLMError, Message
from ..llm.factory import get_client
from ..models import Diagnostics, QueryRequest, QueryResponse, Source
from . import memory
from .retrieve import Context, dedupe_contexts, retrieve

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "were", "that", "this", "it", "as", "by", "with", "be", "been", "has", "have", "had",
    "at", "from", "which", "their", "its", "these", "those", "not", "but", "can", "will",
    "would", "could", "should", "we", "they", "you", "do", "does", "did",
}


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) > 2 and w.lower() not in _STOPWORDS}


def _best_supporting_sentence(chunk_text: str, answer_text: str, question: str) -> str:
    """The sentence within a retrieved chunk that best supports the answer.

    A whole chunk is not a precise citation. We split the chunk into sentences and pick
    the one with the most word overlap with the generated answer (falling back to the
    question if the answer shares no vocabulary with the chunk, e.g. a degraded/LLM-
    unavailable response). If nothing overlaps at all, we keep the full chunk rather than
    guess — a safe citation beats a confidently wrong one.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(chunk_text.replace("\n", " ")) if s.strip()]
    if not sentences:
        return chunk_text.strip()

    target = _tokenize(answer_text) or _tokenize(question)
    if not target:
        return sentences[0]

    best_sentence, best_score = sentences[0], -1
    for sentence in sentences:
        score = len(target & _tokenize(sentence))
        if score > best_score:
            best_sentence, best_score = sentence, score

    return best_sentence if best_score > 0 else chunk_text.strip()


_GAP_CHECK_PROMPT = (
    "You check whether retrieved context is enough to answer a whole-document question, "
    "or whether one more targeted search would fill a real gap — this kind of question "
    "often needs evidence from more than one part of the document.\n\n"
    "Context retrieved so far:\n{context}\n\n"
    "Question: {question}\n\n"
    "If this context already covers everything the question needs, reply with exactly: "
    "ENOUGH\n"
    "Otherwise, reply with exactly one focused search query for the single most "
    "important missing piece — nothing else, no explanation."
)


def _find_gap_query(question: str, contexts: list[Context]) -> str | None:
    """One bounded retrieve-reason-retrieve hop for level 3: ask the model whether the
    first retrieval pass left a gap, and if so, what to search for next.

    Deliberately a single hop, not an open-ended loop — enough to catch the common case
    (decomposition missed a fact) without turning every level-3 answer into an
    unbounded, slow agent loop.
    """
    if not contexts:
        return None
    block = "\n\n".join(f"[page {c.page}] {c.text[:400]}" for c in contexts)
    prompt = _GAP_CHECK_PROMPT.format(context=block, question=question)
    try:
        reply = get_client().chat([{"role": "user", "content": prompt}]).strip()
    except LLMError:
        return None
    if not reply or reply.upper().startswith("ENOUGH"):
        return None
    return reply


SYSTEM_PROMPT = (
    "You answer questions about a single document using only the context provided below. "
    "Each passage is tagged with its location, e.g. \"[page 4, section \\\"Related Work\\\"]\" "
    "or \"[page 4]\" when it falls outside any heading. "
    "Every claim you make must end with an inline citation copied verbatim from one of "
    "those tags — page alone if that is all the passage gives you, page and section if "
    "both are present. Cite every distinct passage a claim relies on; never merge two "
    "citations into one, never cite a page or section that is not shown above, and never "
    "invent one. "
    "If the context does not contain the answer, say so plainly and cite nothing — a "
    "missing citation is better than a wrong one. "
    "Be specific and concise. Answer plainly without any special formatting, lists, or bullet points. "
)


def _build_messages(question: str, contexts: list[Context], history: list[Message]) -> list[Message]:
    def _tag(c: Context) -> str:
        return f'[page {c.page}, section "{c.section}"]' if c.section else f"[page {c.page}]"

    context_block = "\n\n".join(f"{_tag(c)} {c.text}" for c in contexts) or "(no context retrieved)"
    messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Prior turns give the model the conversation so far (Level 2). Retrieval still
    # needs the rewritten query — history in the prompt is necessary but not sufficient.
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Context from the document:\n{context_block}\n\nQuestion: {question}",
        }
    )
    return messages


def _sources_from(contexts: list[Context], answer_text: str, question: str) -> list[Source]:
    out: list[Source] = []
    for c in contexts:
        quote = _best_supporting_sentence(c.text, answer_text, question).replace("\n", " ").strip()
        if len(quote) > 300:
            quote = quote[:300].rsplit(" ", 1)[0] + "…"
        out.append(Source(page=c.page, quote=quote, score=round(c.score, 4)))
    return out


def _save(response: QueryResponse, when: datetime) -> None:
    """Write the answer to data/out/q_<id>_level_<level>_<datetime>.json."""
    out_dir = Path(get_settings().out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = when.strftime("%Y%m%d-%H%M%S")
    name = f"q_{response.question_id}_level_{response.level}_{stamp}.json"
    (out_dir / name).write_text(response.model_dump_json(indent=2), encoding="utf-8")


def answer(req: QueryRequest) -> QueryResponse:
    settings = get_settings()
    client = get_client()
    top_k = req.top_k or settings.top_k

    # The system owns the ids. Level-N questions share one conversation, so level-2
    # follow-ups automatically see the earlier turns of the same level.
    question_id = "q" + uuid.uuid4().hex[:6]
    conversation_id = f"level-{req.level}"

    history = memory.get_history(conversation_id)
    now = datetime.now(UTC)
    started = time.perf_counter()

    contexts = retrieve(req.question, top_k, history, level=req.level)
    if req.level == 3:
        gap_query = _find_gap_query(req.question, contexts)
        if gap_query:
            contexts = dedupe_contexts(contexts + retrieve(gap_query, top_k, level=1))
    messages = _build_messages(req.question, contexts, history)

    try:
        answer_text = client.chat(messages)
    except LLMError as e:
        # Degrade rather than 500: return the retrieved evidence so the pipeline is
        # still usable (and debuggable) without a running LLM.
        answer_text = (
            f"[LLM unavailable: {e}] Retrieved context is attached as sources; "
            "no generated answer."
        )

    memory.append(conversation_id, req.question, answer_text)
    latency_ms = int((time.perf_counter() - started) * 1000)

    response = QueryResponse(
        question_id=question_id,
        level=req.level,
        question=req.question,
        answer=answer_text,
        conversation_id=conversation_id,
        sources=_sources_from(contexts, answer_text, req.question),
        diagnostics=Diagnostics(
            provider=settings.llm_provider,
            chat_model=settings.chat_model,
            embedding_model=settings.embedding_model,
            retrieved_chunks=len(contexts),
            tokens=None,          # TODO: report real token usage if your provider returns it
            latency_ms=latency_ms,
            timestamp=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    )
    _save(response, now)
    return response
