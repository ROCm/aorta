"""XDG-aware locations for per-user aorta state. Stdlib only.

Lives in core rather than under ``aorta.chat`` for two reasons:

* ``aorta.bundle.writer`` must know the chat config path in order to refuse
  bundling it (Decision 9b writes an API key there), and nothing in ``aorta.*``
  may import ``aorta.chat``.
* ``aorta.chat.config`` needs the same values, and a second copy would drift.

Every default is user-owned. Nothing here resolves inside ``site-packages``:
an installed wheel is read-only on a shared node, and a tool that writes into
its own install directory cannot be pip-upgraded cleanly.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Basename of the chat profile file, held separately because
#: :mod:`aorta.bundle.writer` matches on it.
CHAT_CONFIG_FILENAME = "chat.toml"


def _xdg_dir(env_var: str, fallback: str) -> Path:
    """Resolve an XDG base directory, ignoring a non-absolute override.

    The XDG basedir spec says a relative value "should be considered invalid";
    honouring one would anchor the cache to whatever directory the user
    happened to be standing in, so a bad value falls back to the default
    rather than producing a surprising path.
    """
    raw = os.environ.get(env_var, "").strip()
    if raw and Path(raw).is_absolute():
        return Path(raw)
    return Path.home() / fallback


def config_home() -> Path:
    """``$XDG_CONFIG_HOME`` or ``~/.config``."""
    return _xdg_dir("XDG_CONFIG_HOME", ".config")


def cache_home() -> Path:
    """``$XDG_CACHE_HOME`` or ``~/.cache``."""
    return _xdg_dir("XDG_CACHE_HOME", ".cache")


def chat_config_path() -> Path:
    """The ``aorta chat`` profile file: ``$XDG_CONFIG_HOME/aorta/chat.toml``."""
    return config_home() / "aorta" / CHAT_CONFIG_FILENAME


def chat_cache_dir() -> Path:
    """Where chat keeps its regenerable artifacts (index, repo map, history)."""
    return cache_home() / "aorta" / "chat"


__all__ = [
    "CHAT_CONFIG_FILENAME",
    "cache_home",
    "chat_cache_dir",
    "chat_config_path",
    "config_home",
]
