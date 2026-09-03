"""The text the web UI greets a session with, built from live settings.

Separate from ``app.py`` so it can be tested without the ``chat-ui`` extra:
``app.py`` imports ``chainlit`` at module scope, which a ``chat-cli`` install
does not have, and these strings are the part that carries a security claim.
Keeping them here means the claim is checked on every install rather than only
where Chainlit happens to be present.

The claims are derived rather than written down, because the version that was
written down drifted: the welcome message advertised command execution as
sandboxed and always available while ``enable_shell_tool`` defaulted to false
and ``run_terminal_command`` stated it was not a sandbox.
"""

from __future__ import annotations

from aorta.chat.config import settings


def capabilities() -> str:
    """Markdown bullets naming what this session can actually do.

    The shell tool is off unless ``enable_shell_tool`` is set (Decision 17a),
    and when on it is an executable allowlist, not a container: an allowed
    interpreter reaches anything the account running the server reaches. #440
    tracks real isolation.
    """
    lines = [
        "- read files, list directories, and search the codebase -- these are "
        "confined to the configured AORTA and run-artifact roots",
    ]
    if settings.enable_shell_tool:
        allowed = "`, `".join(settings.allowed_commands)
        lines.append(
            f"- run commands from an allowlist (`{allowed}`), starting in the "
            "AORTA directory. **This is not a sandbox**: an allowed "
            "interpreter can read and write anything the account running this "
            "server can."
        )
    else:
        lines.append(
            "- command execution is **disabled** on this server "
            "(`enable_shell_tool = false`, the default)"
        )
    return "\n".join(lines)


def redaction_status() -> str:
    """Disclose the egress-redaction boundary before the user types.

    Decision 16's scope is filesystem paths and IP addresses; #420 tracks
    everything else. The per-request notice reports what a given send had
    removed, but that necessarily arrives after the send, and someone deciding
    what is safe to paste needs the boundary beforehand.

    The claim names *LLM requests* rather than outbound traffic, because
    ``redact_for_send`` is applied to the message list and nothing else. With
    ``embedding_provider = "remote"`` the retrieval query and the indexed
    corpus text reach a second API without passing through it, so that case
    gets said out loud here as well as in ``docs/chat/redaction.md``.
    """
    if not settings.redact:
        return (
            "**Outbound redaction is off** (`redact = false`). Prompts and "
            "retrieved file contents are sent to the model as they are."
        )
    lines = [
        "LLM requests have filesystem paths and IP addresses rewritten "
        "before they leave this machine. Nothing else is removed -- notably "
        "**not** credentials, hostnames or user names -- so treat anything you "
        "paste as leaving the machine."
    ]
    if settings.embedding_provider == "remote":
        lines.append(
            "Retrieval on this server uses a **remote embedding API** "
            '(`embedding_provider = "remote"`), and that path is not redacted: '
            "your question and the indexed text reach it as they are."
        )
    return "\n\n".join(lines)


def welcome_message(backend_description: str) -> str:
    """The full greeting, including the backend line."""
    return (
        "Welcome to the **AORTA Codebase Assistant**.\n\n"
        "I can help you understand, navigate, and work with the AORTA "
        "codebase. In this session I can:\n\n"
        f"{capabilities()}\n\n"
        f"{redaction_status()}\n\n"
        f"_LLM backend: {backend_description}_\n\n"
        "_Type your question below to get started._"
    )


__all__ = ["capabilities", "redaction_status", "welcome_message"]
