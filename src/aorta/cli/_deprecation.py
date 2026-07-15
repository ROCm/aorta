"""Shared deprecation-notice helper for legacy CLI command aliases.

``aorta triage`` and ``aorta probe`` were merged into the single
``aorta sweep`` front door (issue #248). The old commands keep working --
they delegate to the very same ``execute_*`` functions ``aorta sweep``
calls -- but each invocation prints a one-line notice naming the exact
replacement so users migrate without guessing.

The notice goes to **stderr** (``err=True``) so it never contaminates
stdout / artifact-path output that scripts may parse. The wording is
deliberately free of substrings that existing CLI tests assert on
(e.g. "double-dash", "Missing option", "Invalid value", "looks like").
"""

from __future__ import annotations

import click


def emit_deprecation(old_command: str, new_command: str) -> None:
    """Print a stderr deprecation notice mapping ``old`` -> ``new``.

    Args:
        old_command: The legacy invocation, e.g. ``"aorta triage run"``.
        new_command: The supported replacement, e.g. ``"aorta sweep run"``.
    """
    click.echo(
        f"warning: '{old_command}' is deprecated and will be removed in a "
        f"future release; use '{new_command}' instead.",
        err=True,
    )


__all__ = ["emit_deprecation"]
