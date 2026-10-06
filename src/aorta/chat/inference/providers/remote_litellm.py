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
from aorta.chat.inference.providers.base import REMOTE_NATIVE_REQUIREMENT
from aorta.chat.remote_auth import PLACEHOLDER_API_KEY, build_auth, describe_auth

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


def _require_auth_config() -> tuple[str, str]:
    """Return the stripped ``(api_key, auth_header)``, or raise if unusable.

    Checked in ``preflight`` as well as at build time: a custom auth header with
    no key behind it is a configuration error the operator can fix, and this
    backend's whole job is to reach a gateway.
    """
    api_key = settings.remote_llm_api_key.strip()
    auth_header = settings.remote_llm_auth_header.strip()
    if auth_header and not api_key:
        # The placeholder exists to fill the *unused* bearer slot. Sending it as
        # the gateway's own credential authenticates nothing, and would fail at
        # the first real query after preflight called the backend healthy.
        raise ValueError(
            f"remote_llm_auth_header is set to {auth_header!r}, but "
            "remote_llm_api_key is empty, so there is no secret to put in that "
            "header.\n"
            "Set AORTA_CHAT_REMOTE_LLM_API_KEY, or put remote_llm_api_key in "
            "the chat profile ('aorta chat config init')."
        )
    return api_key, auth_header


def _require_model() -> str:
    """Return the stripped model name, or raise if it names no Azure deployment.

    ``azure/`` alone is what the ``azure-openai`` profile writes when its prompt
    is accepted unchanged, and LiteLLM would build a deployment path with an
    empty name from it.
    """
    model = settings.remote_llm_model.strip()
    if model.partition("/")[0] == "azure" and not model.partition("/")[2].strip():
        raise ValueError(
            f"remote_llm_model is {model!r}, which names no Azure deployment.\n"
            "Put the deployment name after the prefix, for example "
            'remote_llm_model = "azure/gpt-4o-mini", or set '
            "AORTA_CHAT_REMOTE_LLM_MODEL."
        )
    return model


def _provider_prefix(model: str) -> str | None:
    """The leading component of *model* when LiteLLM routes on it, else None.

    Read from LiteLLM's own registry, not ``litellm.get_llm_provider``: that
    prints a banner to stdout for a name it does not recognise, which would
    land in ``--json`` output.
    """
    import litellm

    prefix, separator, _ = model.partition("/")
    if not separator:
        return None
    known = {getattr(p, "value", p) for p in getattr(litellm, "provider_list", ())}
    return prefix if prefix in known else None


def route_warning() -> str | None:
    """Why the configured model will not reach the API behind the base URL, or None.

    Only a base URL makes the model name decide the request path. Without a
    provider prefix LiteLLM can only treat that URL as OpenAI-compatible and
    post to ``<base>/chat/completions`` -- a 404 from a gateway that serves
    Azure deployments only (#556). Legitimate for a gateway that does speak
    the OpenAI protocol, hence a warning and not a refusal.
    """
    base = settings.remote_llm_base_url.strip().rstrip("/")
    model = settings.remote_llm_model.strip()
    if not base or not model:
        return None
    prefix = _provider_prefix(model)
    if prefix is None:
        return (
            f"remote_llm_model {model!r} has no LiteLLM provider prefix, so LiteLLM "
            f"treats {base} as an OpenAI-compatible endpoint and posts to\n"
            f"{base}/chat/completions. A gateway that serves Azure OpenAI deployments "
            "answers that with 404 'Resource not found'.\n"
            'If yours does, set remote_llm_model = "azure/<deployment>" and '
            "remote_llm_api_version, or run\n"
            "'aorta chat config init --profile azure-openai --force'.\n"
            "For an OpenAI-compatible gateway, prefix the model with openai/ to "
            'silence this, or use llm_provider = "openai".'
        )
    if prefix == "azure" and base.endswith("/openai"):
        return (
            "remote_llm_base_url ends in /openai, and LiteLLM appends "
            "/openai/deployments/<deployment>/... itself, so requests go to\n"
            f"{base}/openai/deployments/... . Drop the trailing /openai from the base URL."
        )
    return None


LITELLM_IMPORT_MESSAGE = (
    "llm_provider=litellm needs both litellm and langchain-litellm. "
    "Install them with either:\n"
    "  pip install litellm langchain-litellm\n"
    "  pip install 'amd-aorta[chat-all]'\n"
    "Keys normally come from LiteLLM's own environment variables "
    "(ANTHROPIC_API_KEY, GEMINI_API_KEY, ...). Set remote_llm_auth_header "
    "instead when a gateway wants the key in a named header."
)


class RemoteLiteLLMBackend:
    """Chat backend that routes through LiteLLM to any provider it supports."""

    name = "litellm"

    native_requirement = REMOTE_NATIVE_REQUIREMENT

    @property
    def model_name(self) -> str:
        return settings.remote_llm_model

    def get_chat_model(
        self,
        *,
        temperature: float = 0.1,
        streaming: bool = True,
    ) -> BaseChatModel:
        chat_litellm = _load_chat_litellm()
        kwargs: dict[str, Any] = {
            "model": _require_model(),
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
        # key handling is left to LiteLLM's own environment variables -- unless
        # a key was configured, which is then what the caller meant by setting
        # it.
        api_key, auth_header = _require_auth_config()
        client_key, headers = build_auth(
            api_key=api_key or PLACEHOLDER_API_KEY,
            auth_header=settings.remote_llm_auth_header,
            extra_headers=settings.remote_llm_extra_headers,
        )
        model_kwargs: dict[str, Any] = {}
        if headers:
            model_kwargs["extra_headers"] = headers
        # ChatLiteLLM has no api_version field; model_kwargs reach
        # litellm.completion unchanged, which is where Azure reads it.
        api_version = settings.remote_llm_api_version.strip()
        if api_version:
            model_kwargs["api_version"] = api_version
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        # Not gated on ``headers``: with extra headers but no auth header, and
        # with no headers at all, ``build_auth`` hands back the configured key
        # itself, and dropping it left LiteLLM to find a credential in the
        # environment that the profile had already been given.
        if api_key or auth_header:
            kwargs["api_key"] = client_key

        return chat_litellm(**kwargs)

    async def preflight(self) -> None:
        """Surface a missing litellm install, unusable auth or model, before a query."""
        _load_chat_litellm()
        _require_auth_config()
        _require_model()
        logger.info("Using %s", self.describe())
        warning = self.route_warning()
        if warning:
            logger.warning("%s", warning)

    def route_warning(self) -> str | None:
        """See :func:`route_warning`; read by ``aorta chat doctor`` when present."""
        return route_warning()

    async def probe(self, timeout: float | None = None) -> None:
        """Same as :meth:`preflight`, so *timeout* is unused.

        LiteLLM routes to a metered provider, so reaching for the network here
        would bill the operator for running a diagnostic.
        """
        await self.preflight()

    def unreachable_hint(self) -> str:
        endpoint = settings.remote_llm_base_url.strip()
        target = endpoint or f"the provider LiteLLM routes {settings.remote_llm_model} to"
        return (
            f"could not reach {target}.\n"
            "Check that the route is correct and reachable from this host:\n"
            "  export AORTA_CHAT_REMOTE_LLM_BASE_URL=https://...\n"
            "or set remote_llm_base_url in the profile: aorta chat config init"
        )

    def describe(self) -> str:
        endpoint = settings.remote_llm_base_url.strip() or "LiteLLM's own routing"
        auth = (
            describe_auth(
                auth_header=settings.remote_llm_auth_header,
                extra_headers=settings.remote_llm_extra_headers,
            )
            if settings.remote_llm_auth_header.strip() or settings.remote_llm_extra_headers
            else "LiteLLM environment variables"
        )
        return f"remote LiteLLM -- {settings.remote_llm_model} via {endpoint} (auth: {auth})"


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
    # `--verbose` would put remote_llm_api_key in plaintext into the terminal and
    # any captured log. Everything else in this repo is careful never to emit a
    # key -- describe_auth() reports header names only -- and that guarantee is
    # worthless if a dependency prints it instead. WARNING keeps genuine errors.
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    return ChatLiteLLM
