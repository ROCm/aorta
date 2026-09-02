"""Allowlisted terminal command execution tool.

Not a sandbox, deliberately named as such: containment here is an *executable*
allowlist plus a refusal of shell control syntax, and the working directory is
not a boundary. See :func:`run_terminal_command` for what that does and does
not buy, and #440 for the OS-level isolation this would need to earn the other
word. Registered only when ``enable_shell_tool`` is set (Decision 17a).
"""

from __future__ import annotations

import shlex
import subprocess

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


#: Shell syntax that can introduce a second command behind the first. The
#: allowlist is an *executable* check, so refusing these is what makes it mean
#: anything: ``ls ; cat /etc/passwd`` satisfies a test that reads only ``ls``,
#: and the command is handed to a shell afterwards. ``|`` is deliberately not
#: here -- the shipped allowlist includes ``grep``, ``head``, ``wc`` and
#: ``tail``, which are pipeline tools, so every stage is validated instead.
_SHELL_CONTROL_SYNTAX = (";", "&", "`", "$(", ">", "<", "\n", "\r")


def _validate_command(command: str) -> str | None:
    """Return an error message if the command is disallowed, else None."""
    lower = command.lower().strip()

    for pattern in _BLOCKED_PATTERNS:
        if pattern in lower:
            return f"Blocked: command contains disallowed pattern '{pattern}'"

    for token in _SHELL_CONTROL_SYNTAX:
        if token in command:
            return (
                f"Blocked: command contains shell control syntax '{token}'. "
                "Run a single command, optionally as a pipeline."
            )

    # Each stage, not only the first: a pipeline runs one process per stage, and
    # checking the head of it would let any unlisted executable follow a listed.
    for stage in command.split("|"):
        try:
            parts = shlex.split(stage)
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
    """Execute a terminal command, starting in the AORTA codebase directory.

    Every executable used must be on the allowlist, and the command must not
    contain destructive or network-access patterns. Run ONE command, optionally
    as a pipeline (``a | b``); chaining with ';' or '&&', command substitution
    with backticks or $(...), and redirection with '>' or '<' are refused.

    This is not a filesystem sandbox. The working directory sets where relative
    paths resolve, not where the command may reach, and allowed interpreters
    can read and write anywhere the user can. Prefer read_file, list_files and
    grep_code, which are contained to the codebase, whenever they will do.

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
        return f"Error: aorta_path '{cwd}' does not exist."

    # Synchronous by design. This tool is called from inside a running event
    # loop, where ``get_event_loop().run_until_complete()`` can only raise --
    # abandoning its coroutine unawaited -- and fall through to precisely this
    # call, so the async path was never the one that ran. Its timeout was also
    # the only one handled: ``TimeoutExpired`` escaped the fallback entirely.
    try:
        result = subprocess.run(
            command,
            shell=True,  # noqa: S602 - _validate_command refuses control syntax
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=settings.command_timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {settings.command_timeout}s."
    except OSError as exc:
        return f"Error: could not run the command: {exc}"

    return _format_output(result.stdout + result.stderr, result.returncode)


def _format_output(output: str, exit_code: int) -> str:
    max_chars = 4000
    if len(output) > max_chars:
        output = output[:max_chars] + "\n... (truncated)"
    return f"Exit code: {exit_code}\n{output}"
