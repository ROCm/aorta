"""REMOTE flow: any API-key provider that speaks the OpenAI wire protocol.

Covers OpenAI itself plus OpenRouter, Together, Groq, Fireworks and friends
-- point ``REMOTE_LLM_BASE_URL`` at them. Also covers enterprise gateways that
authenticate with a custom header instead of a bearer token, via
``REMOTE_LLM_AUTH_HEADER`` (see ``aorta/chat/remote_auth.py``). No new dependency:
``langchain-openai`` is already required for the local flow. Removing this flow
means deleting this file and its one entry in ``factory.py``.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from aorta.chat.config import settings
from aorta.chat.inference.callcount import LLMCallCounter
from aorta.chat.remote_auth import build_auth, describe_auth

logger = logging.getLogger(__name__)

MISSING_API_KEY_MESSAGE = (
    "remote_llm_api_key is not set, and llm_provider=openai requires it.\n"
    "Set it in the profile with 'aorta chat config init', or in the "
    "environment:\n"
    "  export AORTA_CHAT_REMOTE_LLM_API_KEY=sk-...\n"
    "For an OpenAI-compatible provider other than OpenAI, also set "
    "remote_llm_base_url (for example https://openrouter.ai/api/v1)."
)


class RemoteOpenAIBackend:
    """Chat backend for a hosted, metered OpenAI-compatible endpoint."""

    name = "openai"

    def get_chat_model(
        self,
        *,
        temperature: float = 0.1,
        streaming: bool = True,
    ) -> ChatOpenAI:
        client_key, headers = build_auth(
            api_key=_require_api_key(),
            auth_header=settings.remote_llm_auth_header,
            extra_headers=settings.remote_llm_extra_headers,
        )
        # An empty base URL means "the provider's own default endpoint";
        # ChatOpenAI wants it omitted, not passed as "".
        return ChatOpenAI(
            base_url=settings.remote_llm_base_url.strip() or None,
            api_key=client_key,
            model=settings.remote_llm_model,
            temperature=temperature,
            streaming=streaming,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
            default_headers=headers,
            callbacks=[LLMCallCounter()],
        )

    async def preflight(self) -> None:
        """Validate configuration. Makes no network call, so it costs nothing."""
        _require_api_key()
        logger.info("Using %s", self.describe())

    async def probe(self, timeout: float | None = None) -> None:
        """Same as :meth:`preflight`, so *timeout* is unused.

        A reachability probe here would mean a billable completion against a
        metered endpoint, which a diagnostic command has no business spending.
        Configuration is what this backend can be wrong about, and preflight
        already raises on it.
        """
        await self.preflight()

    def unreachable_hint(self) -> str:
        return (
            f"could not reach {self._endpoint()}.\n"
            "Check that the endpoint is correct and reachable from this host:\n"
            "  export AORTA_CHAT_REMOTE_LLM_BASE_URL=https://...\n"
            "or set remote_llm_base_url in the profile: aorta chat config init"
        )

    def _endpoint(self) -> str:
        return settings.remote_llm_base_url.strip() or "the provider default endpoint"

    def describe(self) -> str:
        endpoint = self._endpoint()
        auth = describe_auth(
            auth_header=settings.remote_llm_auth_header,
            extra_headers=settings.remote_llm_extra_headers,
        )
        return (
            f"remote OpenAI-compatible -- {settings.remote_llm_model} "
            f"at {endpoint} (auth: {auth})"
        )


def _require_api_key() -> str:
    """Return the configured key, raising if it is missing or blank."""
    api_key = settings.remote_llm_api_key.strip()
    if not api_key:
        raise ValueError(MISSING_API_KEY_MESSAGE)
    return api_key
