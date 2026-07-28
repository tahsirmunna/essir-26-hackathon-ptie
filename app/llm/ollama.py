"""Ollama provider — https://ollama.com

Talks to a local Ollama server over its native REST API. Pull models first:

    ollama pull llama3.1
    ollama pull nomic-embed-text
"""

from __future__ import annotations

import httpx

from .base import LLMError, Message


class OllamaClient:
    def __init__(self, base_url: str, chat_model: str, embedding_model: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout = timeout

    def chat(self, messages: list[Message]) -> str:
        try:
            resp = httpx.post(
                f"{self.base_url}/api/chat",
                json={"model": self.chat_model, "messages": messages, "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            raise LLMError(f"ollama chat failed: {e}") from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Newer Ollama serves a batched /api/embed endpoint (one round trip for the whole
        # list). Try it first; fall back to the legacy one-prompt-per-call /api/embeddings
        # for older servers that don't have it yet.
        try:
            resp = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embedding_model, "input": texts},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            embeddings = resp.json().get("embeddings")
            if embeddings is not None:
                return embeddings
        except Exception:
            pass  # older Ollama without /api/embed — fall back below

        out: list[list[float]] = []
        try:
            for text in texts:
                resp = httpx.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": text},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                out.append(resp.json()["embedding"])
            return out
        except Exception as e:
            raise LLMError(f"ollama embed failed: {e}") from e
