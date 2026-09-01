"""The one place a provider name is turned into a chat backend.

Adding a provider is a new module plus one entry in ``_BACKENDS``; removing
one is the reverse. Nothing else in the codebase branches on the provider.
"""

from __future__ import annotations

from typing import Callable, Dict

from aorta.chat.config import settings
from aorta.chat.inference.providers.base import ChatBackend
from aorta.chat.inference.providers.local_vllm import LocalVLLMBackend
from aorta.chat.inference.providers.remote_litellm import RemoteLiteLLMBackend
from aorta.chat.inference.providers.remote_openai import RemoteOpenAIBackend

_BACKENDS: Dict[str, Callable[[], ChatBackend]] = {
    LocalVLLMBackend.name: LocalVLLMBackend,
    RemoteOpenAIBackend.name: RemoteOpenAIBackend,
    RemoteLiteLLMBackend.name: RemoteLiteLLMBackend,
}

_instances: Dict[str, ChatBackend] = {}


def available_providers() -> tuple[str, ...]:
    """Return the provider names ``get_backend()`` accepts, sorted."""
    return tuple(sorted(_BACKENDS))


def get_backend(name: str | None = None) -> ChatBackend:
    """Return the chat backend for *name*, defaulting to ``settings.llm_provider``."""
    provider = (name if name is not None else settings.llm_provider).strip().lower()
    if provider not in _BACKENDS:
        raise ValueError(
            f"unknown LLM provider: {provider!r} "
            f"(expected one of {', '.join(available_providers())})"
        )
    backend = _instances.get(provider)
    if backend is None:
        backend = _BACKENDS[provider]()
        _instances[provider] = backend
    return backend


def reset_backend_cache() -> None:
    """Drop the cached backend instances. Intended for tests."""
    _instances.clear()
