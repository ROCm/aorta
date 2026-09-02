from __future__ import annotations

import os
import ssl
import certifi

import dspy

# LiteLLM fetches a remote price map on import and fails on corporate TLS.
# Point it at the certifi bundle so imports are clean.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

_configured: bool = False


def configure_dspy(
    model: str = "claude-haiku-4-5",
    api_base: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 1024,
) -> None:
    """Configure DSPy to use the LiteLLM proxy.

    Defaults pull from env vars so callers need no hard-coded secrets:
      LITELLM_API_BASE  — proxy URL  (default: http://localhost:4000)
      LITELLM_API_KEY   — master key (default: "dummy")
      LITELLM_MODEL     — model name (default: claude-haiku-4-5)
    """
    global _configured

    # Default: proxy runs locally on whatever node the agent is on (chi2878).
    # Set LITELLM_API_BASE to override (e.g. point at a remote proxy).
    resolved_base = api_base or os.environ.get("LITELLM_API_BASE", "http://localhost:4000")
    resolved_key = api_key or os.environ.get("LITELLM_API_KEY", "dummy")
    resolved_model = os.environ.get("LITELLM_MODEL", model)

    lm = dspy.LM(
        model=f"openai/{resolved_model}",
        api_base=resolved_base,
        api_key=resolved_key,
        max_tokens=max_tokens,
        cache=False,
    )
    dspy.configure(lm=lm)
    _configured = True


def ensure_configured(**kwargs) -> None:
    """Call configure_dspy with kwargs only if not already configured."""
    if not _configured:
        configure_dspy(**kwargs)
