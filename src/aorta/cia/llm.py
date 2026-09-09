from __future__ import annotations

import logging
import os

import certifi

import dspy

# LiteLLM fetches a remote price map on import and fails on corporate TLS.
# Point it at the certifi bundle so imports are clean.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

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

    An explicit argument wins over the environment. The reverse -- which is
    what this did -- let a module name the model it needed, receive a different
    one, and have no way to find out.
    """
    return dspy.LM(
        model=f"openai/{model or os.environ.get('LITELLM_MODEL', DEFAULT_MODEL)}",
        api_base=api_base or os.environ.get("LITELLM_API_BASE", "http://localhost:4000"),
        api_key=api_key or os.environ.get("LITELLM_API_KEY", "dummy"),
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
