"""Pick a provider from configuration.

`get_client()` returns one object exposing both `chat()` and `embed()`. It is
cached, so the whole app shares a single client.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from .litellm_client import LiteLLMClient
from .lmstudio import LMStudioClient
from .logging_client import LoggingChatModel
from .ollama import OllamaClient


@lru_cache
def get_client():
    s = get_settings()
    provider = s.llm_provider.lower()

    if provider == "ollama":
        client = OllamaClient(s.ollama_base_url, s.chat_model, s.embedding_model, s.request_timeout)
    elif provider == "lmstudio":
        client = LMStudioClient(s.lmstudio_base_url, s.chat_model, s.embedding_model, s.request_timeout)
    elif provider == "litellm":
        client = LiteLLMClient(
            s.chat_model, s.embedding_model, s.litellm_api_base, s.litellm_api_key, s.request_timeout
        )
    else:
        # TODO: add your own provider here (vLLM, TGI, a company gateway) by implementing
        # the ChatModel / EmbeddingModel protocols in a new module and branching on its name.
        raise ValueError(
            f"unknown LLM_PROVIDER: {s.llm_provider!r} (expected ollama, lmstudio or litellm)"
        )

    if s.llm_log_path:
        return LoggingChatModel(client, s.llm_log_path, provider, s.chat_model)
    return client
