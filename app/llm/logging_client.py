"""Wraps a chat client to log every call to a file — see exactly what was sent to
the LLM (system prompt, history, retrieved context) and what it replied.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .base import ChatModel, Message


class LoggingChatModel:
    """Forwards chat()/embed() to `inner`, appending one JSON line per chat() call."""

    def __init__(self, inner: ChatModel, log_path: str, provider: str, model: str):
        self._inner = inner
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._provider = provider
        self._model = model

    def chat(self, messages: list[Message]) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "provider": self._provider,
            "model": self._model,
            "messages": messages,
        }
        try:
            reply = self._inner.chat(messages)
        except Exception as e:
            entry["error"] = str(e)
            raise
        else:
            entry["reply"] = reply
            return reply
        finally:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
