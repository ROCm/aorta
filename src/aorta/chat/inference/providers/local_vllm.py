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
from aorta.chat.inference.unreachable import BackendUnreachableError

logger = logging.getLogger(__name__)

#: ``preflight``'s budget. Minutes, because a large model on a cold page cache
#: legitimately takes that long to start serving and the interactive path would
#: rather wait than refuse.
PREFLIGHT_TIMEOUT = 300
PREFLIGHT_INTERVAL = 5

#: ``probe``'s budget, which is a diagnostic's budget rather than a session's:
#: it runs while its operator waits, and "nothing is listening" and "listening
#: but not yet serving" lead to the same next step.
PROBE_TIMEOUT = 5.0
PROBE_INTERVAL = 1.0

#: Ceiling on one ``/health`` attempt. Capped by whatever budget is left, so a
#: host that drops packets rather than refusing them cannot make a five-second
#: probe spend five seconds inside a single connect and then start another.
_REQUEST_TIMEOUT = 5.0


class LocalVLLMBackend:
    """Chat backend for the OpenAI-compatible endpoint vLLM exposes."""

    name = "vllm"

    def get_chat_model(
        self,
        *,
        temperature: float = 0.1,
        streaming: bool = True,
    ) -> ChatOpenAI:
        """Return a LangChain ChatOpenAI instance pointed at the vLLM server.

        ``llm_max_tokens`` / ``llm_timeout`` / ``llm_max_retries`` are applied
        here as they are by the remote backends: config documents them as
        LLM-wide, so a vLLM user setting them and getting no cap at all is the
        setting silently not meaning what it says. No ``LLMCallCounter``
        though -- that exists to count billable remote calls, and this flow has
        none.
        """
        return ChatOpenAI(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            model=settings.vllm_model,
            temperature=temperature,
            streaming=streaming,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )

    def health_url(self) -> str:
        """The endpoint ``preflight`` and ``probe`` poll.

        Only a *trailing* ``/v1`` is dropped. ``replace`` removed every
        occurrence, so a base URL carrying it in the authority or an earlier
        path segment -- ``http://v1.example/v1`` -- was probed at the malformed
        ``http:/.example/health`` and reported unreachable while serving fine.
        """
        base = settings.vllm_base_url.rstrip("/").removesuffix("/v1")
        return base + "/health"

    async def _await_health(self, timeout: float, interval: float) -> bool:
        """Poll ``/health`` until it answers 200. Returns whether it ever did.

        Reports rather than decides, so that the two callers can differ: the
        session tolerates a server that never answered, a diagnostic must not.
        """
        url = self.health_url()
        logger.info("Waiting for vLLM at %s ...", url)
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        async with httpx.AsyncClient() as client:
            while (remaining := deadline - loop.time()) > 0:
                try:
                    resp = await client.get(url, timeout=min(_REQUEST_TIMEOUT, remaining))
                    if resp.status_code == 200:
                        logger.info("vLLM is ready.")
                        return True
                # Every transport-level failure means "not answering yet", and
                # the catch has to be the base class to say so: ConnectTimeout
                # is a TimeoutException rather than a ConnectError, so naming
                # subclasses let a host that drops packets escape both callers --
                # preflight would raise where it promises not to, and probe would
                # surface a bare ConnectTimeout instead of the hint.
                except httpx.TransportError:
                    pass
                logger.info("vLLM not ready yet, retrying in %gs ...", interval)
                await asyncio.sleep(min(interval, max(0.0, deadline - loop.time())))
        return False

    async def preflight(
        self, timeout: int = PREFLIGHT_TIMEOUT, interval: int = PREFLIGHT_INTERVAL
    ) -> None:
        """Wait for the vLLM server, and start anyway if it never answers.

        Permissive on purpose, and this is the one backend where that is the
        right call: a server still loading weights answers a minute later, and
        refusing here would make the REPL unusable during a warm-up the user can
        see progressing. The cost of that tolerance is that this method cannot
        also serve as a diagnostic -- see :meth:`probe`.
        """
        if not await self._await_health(timeout, interval):
            logger.warning("vLLM did not become ready within %ds -- starting anyway.", timeout)

    async def probe(self, timeout: float | None = None) -> None:
        """Raise unless the server answers ``/health`` within *timeout*."""
        budget = PROBE_TIMEOUT if timeout is None else timeout
        if not await self._await_health(budget, min(PROBE_INTERVAL, budget)):
            raise BackendUnreachableError(
                f"nothing answered {self.health_url()} within {budget:g}s.\n"
                f"{self.unreachable_hint()}"
            )

    def unreachable_hint(self) -> str:
        return (
            f"llm_provider=vllm expects a self-hosted vLLM server serving the "
            f"OpenAI wire protocol at {settings.vllm_base_url}.\n"
            "Check that the server is running and listening on that address, or "
            "point aorta at the one that is:\n"
            "  export AORTA_CHAT_VLLM_BASE_URL=http://<host>:<port>/v1\n"
            "or set vllm_base_url in the profile: aorta chat config init "
            "--profile local-vllm\n"
            "To use a hosted provider instead: aorta chat ask --llm-provider openai \"...\""
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
