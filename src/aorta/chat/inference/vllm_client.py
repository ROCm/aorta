"""Compatibility shim for the pre-provider LLM entry points.

``get_chat_llm`` now lives in :mod:`src.inference.chat` and dispatches on
``LLM_PROVIDER``; the raw vLLM helpers moved to
:mod:`src.inference.providers.local_vllm`. Both are re-exported here so
existing imports keep working unchanged.
"""

from __future__ import annotations

from aorta.chat.inference.chat import get_chat_llm
from aorta.chat.inference.providers.local_vllm import get_async_openai_client, stream_chat

__all__ = ["get_async_openai_client", "get_chat_llm", "stream_chat"]
