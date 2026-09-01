"""LOCAL flow: a self-hosted vLLM server speaking the OpenAI wire protocol.

Everything vLLM-specific lives here. Removing the local flow means deleting
this file and its one entry in ``factory.py``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import httpx
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from aorta.chat.config import settings

logger = logging.getLogger(__name__)


class LocalVLLMBackend:
    """Chat backend for the OpenAI-compatible endpoint vLLM exposes."""

    name = "vllm"

    def get_chat_model(
        self,
        *,
        temperature: float = 0.1,
        streaming: bool = True,
    ) -> ChatOpenAI:
        """Return a LangChain ChatOpenAI instance pointed at the vLLM server."""
        return ChatOpenAI(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            model=settings.vllm_model,
            temperature=temperature,
            streaming=streaming,
        )

    async def preflight(self, timeout: int = 300, interval: int = 5) -> None:
        """Block until the vLLM server is reachable or *timeout* seconds elapse."""
        url = settings.vllm_base_url.rstrip("/").replace("/v1", "") + "/health"
        logger.info("Waiting for vLLM at %s ...", url)
        deadline = asyncio.get_event_loop().time() + timeout
        async with httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(url, timeout=5)
                    if resp.status_code == 200:
                        logger.info("vLLM is ready.")
                        return
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass
                logger.info("vLLM not ready yet, retrying in %ds ...", interval)
                await asyncio.sleep(interval)
        logger.warning(
            "vLLM did not become ready within %ds -- starting anyway.", timeout
        )

    def describe(self) -> str:
        return f"local vLLM -- {settings.vllm_model} at {settings.vllm_base_url}"


def get_async_openai_client() -> AsyncOpenAI:
    """Return a raw AsyncOpenAI client for custom streaming logic."""
    return AsyncOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
    )


async def stream_chat(
    messages: list[dict],
    temperature: float = 0.1,
) -> AsyncIterator[str]:
    """Low-level streaming helper that yields text deltas."""
    client = get_async_openai_client()
    response = await client.chat.completions.create(
        model=settings.vllm_model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    async for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
