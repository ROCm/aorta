from __future__ import annotations

import logging
import os

try:
    import dspy
except ImportError as exc:  # pragma: no cover - only without the extra
    raise ImportError(
        "The Cluster Intelligence Agents need DSPy, which comes with the [cia] "
        "extra: pip install 'amd-aorta[cia]'. (The extra installs on every "
        "Python this package supports; if it appeared to install and left "
        "nothing behind, say so -- that is a packaging bug, not a missing step.)"
    ) from exc

#: Where the CA bundle is named, for callers who want to look.
_CA_ENV_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"

DEFAULT_MAX_TOKENS = 4096
"""Enough for a reasoning model to think and still answer.

A reasoning model bills its hidden reasoning against the same budget as the
reply, and spends it freely. At the 1024 this used to be, Autopsy's reasoning
consumed the whole budget and the response was cut off before a single output
field was emitted. The router received empty text, failed to parse it, and
scored every bundle ``tooling_gap`` at confidence 0.0 -- a wrong verdict, with
nothing in it to say the reasoning step had been truncated on the way.
"""

_configured: bool = False


def _use_certifi_bundle() -> None:
    """Point TLS verification at certifi, for this process, when asked to.

    LiteLLM fetches a remote price map and can fail on a corporate TLS
    interception proxy whose CA the certifi bundle knows and the system store
    does not. Certifi fixes that site and breaks the opposite one, where the
    corporate CA is in the system store and not in certifi.

    So three things. It runs from here rather than at import, because importing
    a module should not change TLS verification for everything else in the
    process -- including unrelated aorta code and the chat provider layer,
    which never asked. It defers to SSL_CERT_FILE or REQUESTS_CA_BUNDLE if
    either is already set, because that is somebody having decided. And
    CIA_SSL_USE_CERTIFI=0 turns it off for the site it would otherwise break.
    """
    if os.environ.get("CIA_SSL_USE_CERTIFI", "1") == "0":
        return
    if any(os.environ.get(var) for var in _CA_ENV_VARS):
        return
    try:
        import certifi
    except ImportError:
        log.debug("certifi is not installed; leaving TLS verification alone")
        return
    for var in _CA_ENV_VARS:
        os.environ[var] = certifi.where()


class ProviderNotConfigured(RuntimeError):
    """No endpoint was configured, and guessing at one is worse than saying so."""


def chat_provider(*, configured_only: bool = True) -> tuple[str, str, str] | None:
    """(base_url, api_key, model) from the chat configuration, or None.

    The rest of this package reaches a model through ``get_chat_llm()``,
    selected by ``llm_provider`` and configured in ``~/.config/aorta/chat.toml``
    or ``AORTA_CHAT_*`` (docs/chat/providers.md, docs/chat/configuration.md).
    The agents read those same settings, so that configuring chat configures
    them: a user who has run ``aorta chat config init --profile openai`` should
    not then have a Watch and an Autopsy quietly talking somewhere else.

    Only the settings are read, not the provider layer. ``aorta.chat.config``
    imports pydantic and stdlib and nothing from the chat extras, so this keeps
    the agents runnable on a base install -- and when that module is absent,
    because those extras are not installed, this returns None rather than
    failing the import.

    With *configured_only*, None also means "nothing here was actually set".
    Every field has a default, so answering with one would silently outrank a
    deployment that had configured the agents some other way.
    """
    try:
        from aorta.chat.config import settings
    except Exception:  # pragma: no cover - depends on what is installed
        log.debug("aorta.chat.config is not importable; not reading chat settings")
        return None

    if getattr(settings, "llm_provider", "") == "vllm":
        fields = ("vllm_base_url", "vllm_api_key", "vllm_model")
    else:
        # An empty remote base URL means "the provider's own endpoint", which
        # is a decision rather than a gap.
        fields = ("remote_llm_base_url", "remote_llm_api_key", "remote_llm_model")

    if configured_only and not ({*fields, "llm_provider"} & settings.model_fields_set):
        return None
    return tuple(getattr(settings, f) for f in fields)  # type: ignore[return-value]


def _legacy_env() -> tuple[str, str, str] | None:
    """The LITELLM_* variables, which predate reading the chat settings."""
    base = os.environ.get("LITELLM_API_BASE")
    if not base:
        return None
    log.warning(
        "Reading LITELLM_API_BASE/KEY/MODEL. These predate the agents taking "
        "their configuration from the same place as the rest of aorta chat; "
        "set AORTA_CHAT_* or ~/.config/aorta/chat.toml and one setting will do "
        "for both."
    )
    return (
        base,
        os.environ.get("LITELLM_API_KEY", ""),
        os.environ.get("LITELLM_MODEL", DEFAULT_MODEL),
    )


def build_lm(
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dspy.LM:
    """Build an LM for one module to bind to its own program.

    A module whose settings matter builds its own here rather than racing the
    others to configure a shared one; see ``ensure_configured`` for what the
    race costs.

    Where the endpoint comes from, in order: an explicit argument, then the
    chat settings, then the LITELLM_* variables that predate them. An explicit
    argument wins over all of it -- the reverse, which is what this did, let a
    module name the model it needed, receive a different one, and have no way
    to find out.

    There is no default of this package's own. It used to fall back to
    http://localhost:4000 with the key "dummy", so a user whose chat was
    configured against a real provider had a Watch and an Autopsy silently
    addressing a proxy that was not running, and read the resulting quiet as
    nothing being wrong.
    """
    # Configured chat first, then a deployment still on the old variables, then
    # chat's own defaults -- so an unconfigured agent fails exactly the way an
    # unconfigured chat does, rather than in a second way at a second address.
    resolved = chat_provider() or _legacy_env() or chat_provider(configured_only=False)
    if resolved is None and not api_base:
        raise ProviderNotConfigured(
            "The agents reach a model through the same configuration as the "
            "rest of aorta chat, and none is available. Run `aorta chat config "
            "init`, or set AORTA_CHAT_VLLM_BASE_URL."
        )
    settings_base, settings_key, settings_model = resolved or ("", "", "")

    _use_certifi_bundle()
    return dspy.LM(
        model=f"openai/{model or settings_model or DEFAULT_MODEL}",
        api_base=(api_base or settings_base) or None,
        api_key=api_key or settings_key or "EMPTY",
        max_tokens=max_tokens,
        cache=False,
    )


def configure_dspy(
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> None:
    """Set the process-wide default LM.

    Defaults come from the environment so callers need no hard-coded secrets:
      LITELLM_API_BASE  — proxy URL  (default: http://localhost:4000)
      LITELLM_API_KEY   — master key (default: "dummy")
      LITELLM_MODEL     — model name (default: claude-haiku-4-5)
    """
    global _configured

    dspy.configure(lm=build_lm(model, api_base, api_key, max_tokens))
    _configured = True


def ensure_configured(**kwargs) -> None:
    """Set the process-wide default LM, if nothing has set one yet.

    The first caller wins, so every later caller's arguments are discarded.
    That is harmless for a caller passing none, and a silent downgrade for a
    caller passing some: Autopsy asked here for a larger model and a larger
    budget, Watch had already configured the default from its own poll loop,
    and Autopsy reasoned at Watch's settings with neither of them able to tell.

    So the arguments are no longer accepted quietly. A module whose settings
    matter should build its own LM with :func:`build_lm` and bind it to its own
    program, which no other module can then take away.
    """
    global _configured

    if _configured:
        if kwargs:
            log.warning(
                "DSPy already has a default LM, so %s had no effect here. Build "
                "an LM with build_lm() and bind it to your own module instead.",
                ", ".join(sorted(kwargs)),
            )
        return
    configure_dspy(**kwargs)
