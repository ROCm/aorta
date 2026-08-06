"""Shell-free subprocess execution with explicit health states."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    launch_error: str | None = None


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode(errors="replace")


def run_argv(
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    env: Mapping[str, str] | None = None,
) -> ProcessResult:
    """Execute exact argv without a shell and normalize timeout/launch errors."""

    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ValueError("argv must contain non-empty strings")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    command = tuple(argv)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=None if env is None else dict(env),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ProcessResult(
            argv=command,
            returncode=None,
            stdout=_text(exc.stdout),
            stderr=_text(exc.stderr),
            timed_out=True,
        )
    except OSError as exc:
        return ProcessResult(
            argv=command,
            returncode=None,
            stdout="",
            stderr="",
            launch_error=f"{type(exc).__name__}: {exc}",
        )
    return ProcessResult(
        argv=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
