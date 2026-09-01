"""The single entry point every graph node uses to obtain a chat model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aorta.chat.inference.providers.factory import get_backend

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


def get_chat_llm(
    temperature: float = 0.1,
    streaming: bool = True,
) -> BaseChatModel:
    """Return a chat model from the backend selected by ``LLM_PROVIDER``."""
    return get_backend().get_chat_model(temperature=temperature, streaming=streaming)
