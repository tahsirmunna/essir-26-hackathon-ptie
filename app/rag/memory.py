"""Conversation memory — where Level 2 lives.

A Level-2 question is a follow-up: *"and how large is its test split?"* only makes
sense given the earlier turn it refers to. To answer it you need the history of the
conversation it belongs to.

Rather than keep raw transcripts (which grow without bound and vanish on restart),
each conversation gets one running summary, updated by the chat model after every
turn and persisted to a small text file under `memory_dir`. This survives restarts
without adding infrastructure (no Redis/Postgres) — the tradeoff is that it still
assumes a single writer: concurrent workers touching the same conversation_id would
clobber each other's file.
"""

from __future__ import annotations

from pathlib import Path

from ..config import get_settings
from ..llm.base import LLMError, Message
from ..llm.factory import get_client

_SUMMARY_PROMPT = (
    "You maintain a running summary of a conversation between a user and an assistant "
    "that answers questions about a document. Update the summary so it includes the new "
    "turn below. Keep it to a few sentences, preserve every fact or entity a follow-up "
    "question might refer to (e.g. 'the dataset', 'that table on page 12'), and drop "
    "pleasantries.\n\n"
    "Existing summary:\n{summary}\n\n"
    "New turn:\nUser: {user}\nAssistant: {assistant}\n\n"
    "Updated summary:"
)


def _path(conversation_id: str) -> Path:
    memory_dir = Path(get_settings().memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir / f"{conversation_id}.txt"


def _read(conversation_id: str) -> str:
    path = _path(conversation_id)
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def get_history(conversation_id: str | None) -> list[Message]:
    if not conversation_id:
        return []
    summary = _read(conversation_id)
    if not summary:
        return []
    return [{"role": "system", "content": f"Summary of the conversation so far:\n{summary}"}]


def append(conversation_id: str | None, user: str, assistant: str) -> None:
    if not conversation_id:
        return
    summary = _read(conversation_id)
    prompt = _SUMMARY_PROMPT.format(summary=summary or "(none yet)", user=user, assistant=assistant)
    try:
        updated = get_client().chat([{"role": "user", "content": prompt}]).strip()
    except LLMError:
        # Degrade to a plain append rather than losing the turn if the LLM is down.
        updated = f"{summary}\nUser: {user}\nAssistant: {assistant}".strip()
    _path(conversation_id).write_text(updated, encoding="utf-8")


def reset(conversation_id: str) -> None:
    _path(conversation_id).unlink(missing_ok=True)


# TODO(level-2): history alone is not enough. The retrieval step still embeds the
#   raw follow-up ("and the test split?"), which has no searchable content. The high-
#   leverage fix is to REWRITE the question into a standalone query using this history
#   BEFORE retrieving — see rag/retrieve.py::rewrite_query.
