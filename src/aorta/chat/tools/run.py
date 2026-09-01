"""Sandboxed terminal command execution tool."""

from __future__ import annotations

import asyncio
import shlex

from langchain_core.tools import tool

from aorta.chat.config import settings

_BLOCKED_PATTERNS = [
    "rm -rf",
    "rm -r /",
    "curl ",
    "wget ",
    "nc ",
    "ncat ",
    "ssh ",
    "scp ",
    "> /dev/",
    "mkfs",
    "dd if=",
    ":(){ :|:& };:",
]


def _validate_command(command: str) -> str | None:
    """Return an error message if the command is disallowed, else None."""
    lower = command.lower().strip()

    for pattern in _BLOCKED_PATTERNS:
        if pattern in lower:
            return f"Blocked: command contains disallowed pattern '{pattern}'"

    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return f"Invalid command syntax: {exc}"

    if not parts:
        return "Empty command."

    executable = parts[0].split("/")[-1]
    if executable not in settings.allowed_commands:
        allowed = ", ".join(settings.allowed_commands)
        return (
            f"Command '{executable}' is not in the allowlist. "
            f"Allowed: {allowed}"
        )

    return None


@tool
def run_terminal_command(command: str) -> str:
    """Execute a terminal command inside the AORTA codebase directory.

    The command must use one of the allowed executables and must not
    contain destructive or network-access patterns.

    Args:
        command: Shell command to run (e.g. 'pytest tests/', 'python -m mymod').

    Returns:
        Combined stdout + stderr output, truncated to 4000 chars.
    """
    error = _validate_command(command)
    if error:
        return f"DENIED: {error}"

    cwd = settings.aorta_root
    if not cwd.exists():
        return f"Error: AORTA_PATH '{cwd}' does not exist."

    try:
        result = asyncio.get_event_loop().run_until_complete(
            _run(command, str(cwd))
        )
    except RuntimeError:
        import subprocess

        result = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=settings.command_timeout,
        )
        output = result.stdout + result.stderr
        exit_code = result.returncode
        return _format_output(output, exit_code)

    return result


async def _run(command: str, cwd: str) -> str:
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=settings.command_timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        return f"Error: command timed out after {settings.command_timeout}s."

    output = (stdout or b"").decode(errors="replace") + (stderr or b"").decode(
        errors="replace"
    )
    return _format_output(output, proc.returncode or 0)


def _format_output(output: str, exit_code: int) -> str:
    max_chars = 4000
    if len(output) > max_chars:
        output = output[:max_chars] + "\n... (truncated)"
    return f"Exit code: {exit_code}\n{output}"
