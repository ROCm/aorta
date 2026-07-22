"""Shared environment-variable name/value rules (dependency-free leaf module).

An env var declared in a recipe, a registry environment, a sidecar file, a
mitigation bundle, or a direct ``--extra-env`` flag must satisfy the SAME two
rules before it can be applied to ``os.environ`` or forwarded into a container:

1. The NAME must be a POSIX env-var name (``[A-Za-z_][A-Za-z0-9_]*``).
2. The VALUE must not contain a NUL byte -- a valid Python ``str`` character
   that cannot be stored in an OS environment variable (``os.environ.update``
   and ``execve`` both reject it).

These rules were previously duplicated (and drifted) across the dispatcher,
the Docker helper, the recipe parser, and the registry loaders -- and two
declaration boundaries enforced neither, so a bad entry could pass recipe
loading / ``--dry-run`` and only fail per-cell at run time (a "green command,
zero work" outcome). This module is the single source of truth for the two
predicates. It deliberately has NO aorta imports so every layer -- registry,
triage, run -- can use it without an import cycle, and each caller raises its
own public exception type (``RegistryError`` / ``RecipeSchemaError`` /
``ValueError``) rather than a shared one.

The predicates never accept or echo the value's content, so callers can build
value-redacting error messages (env values may be secrets).
"""

from __future__ import annotations

import re

# POSIX env-var name shape. The single definition; every layer imports this
# rather than re-compiling its own copy.
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_valid_env_name(key: str) -> bool:
    """True if ``key`` is a valid POSIX env-var name (``[A-Za-z_][A-Za-z0-9_]*``)."""
    return ENV_KEY_RE.fullmatch(key) is not None


def value_has_nul(value: str) -> bool:
    """True if ``value`` contains a NUL byte (cannot live in an env var)."""
    return "\x00" in value


__all__ = ["ENV_KEY_RE", "is_valid_env_name", "value_has_nul"]
