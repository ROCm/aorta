"""Safe, compact rendering for environment-variable bundles in CLI output."""

from __future__ import annotations

import re
from collections.abc import Mapping

_SAFE_VALUE_RE = re.compile(r"[A-Za-z0-9_@%+=:,./-]+")


def format_env_bundle(env: Mapping[str, str], *, empty: str = "(none)") -> str:
    """Render sorted ``KEY=VALUE`` tokens without emitting unsafe characters.

    Simple values stay unquoted for readable listings. Values containing
    whitespace, terminal control characters, or characters outside the
    conservative token-safe set use ``repr`` so each pair remains one visible
    token and cannot alter the surrounding terminal output.
    """

    pairs = []
    for key, value in sorted(env.items()):
        rendered_value = value if _SAFE_VALUE_RE.fullmatch(value) else repr(value)
        pairs.append(f"{key}={rendered_value}")
    return " ".join(pairs) or empty
