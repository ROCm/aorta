"""REMOTE flow: LiteLLM, for providers with a native (non-OpenAI) protocol.

Anthropic, Gemini, Bedrock and the rest of LiteLLM's routing table. Keys are
normally read by LiteLLM itself from its own standard environment variables
(``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, ...), matching how the aorta agent's
LiteLLM proposer behaves; a gateway that wants the key in a named header is
handled through ``REMOTE_LLM_AUTH_HEADER`` instead. Removing this flow means
deleting this file and its one entry in ``factory.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aorta.chat.config import settings
from aorta.chat.inference.callcount import LLMCallCounter
from aorta.chat.remote_auth import PLACEHOLDER_API_KEY, build_auth, describe_auth

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

LITELLM_IMPORT_MESSAGE = (
    "LLM_PROVIDER=litellm needs both litellm and langchain-litellm. "
    "Install them with either:\n"
    "  pip install litellm langchain-litellm\n"
    '  pip install -e ".[remote]"   # from the repo root (editable + extra)\n'
    "Keys normally come from LiteLLM's own environment variables "
    "(ANTHROPIC_API_KEY, GEMINI_API_KEY, ...). Set REMOTE_LLM_AUTH_HEADER "
    "instead when a gateway wants the key in a named header."
)


class RemoteLiteLLMBackend:
    """Chat backend that routes through LiteLLM to any provider it supports."""

    name = "litellm"

    def get_chat_model(
        self,
        *,
        temperature: float = 0.1,
        streaming: bool = True,
    ) -> BaseChatModel:
        chat_litellm = _load_chat_litellm()
        kwargs: dict[str, Any] = {
            "model": settings.remote_llm_model,
            "api_base": settings.remote_llm_base_url.strip() or None,
            "temperature": temperature,
            "streaming": streaming,
            "max_tokens": settings.llm_max_tokens,
            "request_timeout": settings.llm_timeout,
            "max_retries": settings.llm_max_retries,
            "callbacks": [LLMCallCounter()],
        }

        # A gateway in front of the provider wants the key in its own header,
        # exactly as on the openai backend. Without an auth header configured,
        # key handling is left to LiteLLM's own environment variables.
        client_key, headers = build_auth(
            api_key=settings.remote_llm_api_key or PLACEHOLDER_API_KEY,
            auth_header=settings.remote_llm_auth_header,
            extra_headers=settings.remote_llm_extra_headers,
        )
        if headers:
            kwargs["model_kwargs"] = {"extra_headers": headers}
            if settings.remote_llm_auth_header.strip():
                kwargs["api_key"] = client_key

        return chat_litellm(**kwargs)

    async def preflight(self) -> None:
        """Surface a missing litellm install now rather than mid-query."""
        _load_chat_litellm()
        logger.info("Using %s", self.describe())

    def describe(self) -> str:
        endpoint = settings.remote_llm_base_url.strip() or "LiteLLM's own routing"
        auth = (
            describe_auth(
                auth_header=settings.remote_llm_auth_header,
                extra_headers=settings.remote_llm_extra_headers,
            )
            if settings.remote_llm_auth_header.strip()
            or settings.remote_llm_extra_headers
            else "LiteLLM environment variables"
        )
        return (
            f"remote LiteLLM -- {settings.remote_llm_model} "
            f"via {endpoint} (auth: {auth})"
        )


def _load_chat_litellm() -> Any:
    """Import ChatLiteLLM lazily so the extra is only needed when selected."""
    try:
        import litellm
        from langchain_litellm import ChatLiteLLM
    except ImportError as exc:
        raise ImportError(LITELLM_IMPORT_MESSAGE) from exc

    # Graph nodes ask for temperature 0.0 or 0.1 to keep routing and criticism
    # deterministic. Some models accept only temperature=1 -- current Claude
    # Opus builds among them -- and LiteLLM raises UnsupportedParamsError rather
    # than negotiating, which would fail every call. Dropping the parameter is
    # LiteLLM's documented way to write provider-portable code, and losing
    # determinism on such a model is better than not reaching it at all.
    litellm.drop_params = True

    # LiteLLM's debug logger prints the outbound request, headers included, so
    # `--verbose` would put REMOTE_LLM_API_KEY in plaintext into the terminal and
    # any captured log. Everything else in this repo is careful never to emit a
    # key -- describe_auth() reports header names only -- and that guarantee is
    # worthless if a dependency prints it instead. WARNING keeps genuine errors.
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    return ChatLiteLLM
