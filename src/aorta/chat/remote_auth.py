"""Header assembly shared by the two REMOTE flows (chat and embeddings).

Most hosted endpoints authenticate with a bearer token, which the OpenAI client
sends for us when given ``api_key``. Enterprise gateways frequently do not: an
Azure API Management front end wants the key in ``Ocp-Apim-Subscription-Key``,
Azure OpenAI wants ``api-key``, Anthropic's own API wants ``x-api-key``. All of
those are still a single static header, so one mechanism covers them.

Auth styles that are *not* a static header -- OAuth token exchange, AWS SigV4 --
are out of scope here on purpose: they need refresh or signing logic, which is
what ``llm_provider=litellm`` already provides.

This module belongs to the remote flows only. Removing them means deleting it
along with ``providers/remote_*.py`` and ``rag/embeddings/remote_api.py``.
"""

from __future__ import annotations

#: Sent as ``api_key`` when the real secret travels in a custom header instead.
#: The OpenAI client rejects an empty key even when it is never used, which is
#: why gateway examples in the wild all pass some throwaway string here.
PLACEHOLDER_API_KEY = "unused"


def build_auth(
    *,
    api_key: str,
    auth_header: str = "",
    extra_headers: dict[str, str] | None = None,
) -> tuple[str, dict[str, str] | None]:
    """Return the ``api_key`` and ``default_headers`` to hand the client.

    With no ``auth_header`` the key goes out as a bearer token, unchanged from
    the default OpenAI behaviour. With one, the key moves into that header and
    ``api_key`` becomes a placeholder, because a gateway that reads a custom
    header has no use for the Authorization one.
    """
    headers = {k: v for k, v in (extra_headers or {}).items() if k and v}
    header_name = auth_header.strip()
    if not header_name:
        return api_key, headers or None
    headers[header_name] = api_key
    return PLACEHOLDER_API_KEY, headers


def describe_auth(
    *,
    auth_header: str = "",
    extra_headers: dict[str, str] | None = None,
) -> str:
    """Summarise the auth style using header *names* only.

    Never includes a header value: this string is logged at startup and shown
    in the Chainlit welcome message.
    """
    header_name = auth_header.strip()
    style = f"{header_name} header" if header_name else "bearer token"
    extras = sorted(k for k in (extra_headers or {}) if k and k != header_name)
    if extras:
        return f"{style}, plus {', '.join(extras)}"
    return style
