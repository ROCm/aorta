"""Pure helpers for the ``aorta probe`` CLI.

Kept out of :mod:`aorta.cli.probe` so the Click handler stays a thin shim
(see FR 1.15 -- handler body is bounded at 60 lines so it can't silently
grow business logic). Every function here is pure: no FS, no env mutation,
no subprocess.
"""

from __future__ import annotations

from typing import Literal

VALID_ENV_PASSTHROUGH_MODES: tuple[Literal["inherit", "file"], ...] = ("inherit", "file")


class ProbeUsageError(ValueError):
    """User-input error that the CLI should surface as ``ClickException``.

    Kept as a plain ``ValueError`` subclass (not a ``click.ClickException``)
    so non-CLI callers -- tests, the recipe-builder, future programmatic
    consumers -- don't need to depend on Click to catch it. The Click
    handler bridges this into a ``ClickException`` at the CLI boundary.
    """


def parse_env_passthrough_mode(value: str) -> Literal["inherit", "file"]:
    """Validate the ``--env-passthrough-mode`` value.

    Both modes share the same in-process env-var application (the
    dispatcher sets per-cell mitigation + diagnostic env vars on
    ``os.environ`` before the workload's ``run()``); ``file`` mode
    additionally drops a ``probe.env`` file in the trial dir and exports
    ``AORTA_ENV_FILE`` to point at it. See ``docs/probe-188/usage.md``
    §"Env-passthrough modes" for the F6 rationale.
    """
    if value not in VALID_ENV_PASSTHROUGH_MODES:
        raise ProbeUsageError(
            f"--env-passthrough-mode: must be one of "
            f"{list(VALID_ENV_PASSTHROUGH_MODES)}, got {value!r}"
        )
    return value  # type: ignore[return-value]


def validate_trailing_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    """Reject an empty trailing-argv list.

    ``aorta probe -- <argv>`` is the only legal channel for the user
    command; ``aorta probe`` without a trailing ``--`` (or with ``--``
    followed by nothing) is a usage error. The "no parsing" invariant
    means we don't otherwise inspect ``argv`` -- it's forwarded
    byte-for-byte to :class:`SubprocessWorkload`.
    """
    if not argv:
        raise ProbeUsageError(
            "no trailing argv supplied. Usage: aorta probe [options] -- <command> [args...]"
        )
    return argv


def format_dry_run_line(
    cell_name: str,
    env: dict[str, str],
    argv: tuple[str, ...],
) -> str:
    """Render one cell's dry-run line.

    Stable key order (sorted) so snapshot-style tests are deterministic
    across runs. Kept in the helper module so the dry-run formatter can
    be unit-tested without invoking the CLI.
    """
    env_part = " ".join(f"{k}={v}" for k, v in sorted(env.items())) or "(no env)"
    return f"  {cell_name}: env=[{env_part}] argv={list(argv)}"


__all__ = [
    "VALID_ENV_PASSTHROUGH_MODES",
    "ProbeUsageError",
    "format_dry_run_line",
    "parse_env_passthrough_mode",
    "validate_trailing_argv",
]
