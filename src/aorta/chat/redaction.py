"""Egress redaction for outbound LLM traffic (Decision 16).

On by default, because of what Decision 11b put into retrieval. The run-artifact
collection indexes the user's own ``matrix.json`` / ``env.json``, so a retrieved
chunk can carry a customer's filesystem layout and host addresses -- and the
whole point of the remote providers is that the prompt leaves the machine. Both
halves of that are individually reasonable and together they are an
exfiltration path, so the gate closes by default and says so.

No new scrubbers are written here. The path and IP rewriters are
:func:`aorta.probe.redaction.scrub_text`, which ``aorta bundle`` already uses to
make a diagnostic bundle shareable; this module is the adapter that applies them
to a message list and accounts for what it changed. A second implementation
would be a second thing to keep correct, and the regex-DoS windowing and the
per-document placeholder numbering in that module are not worth re-deriving.

Scope is deliberately paths and addresses, not credentials. Chat never puts a
key in a message body -- keys travel as request headers, which this never sees
-- so a "secret scrubber" here would imply a guarantee it cannot make.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import IO, Any

from aorta.chat.config import settings


def _notice_stream() -> IO[str]:
    """Where the notice goes: stderr, resolved at call time.

    stderr and not stdout because ``--json`` and ``--plain`` are consumed by
    scripts, and a notice on stdout would corrupt the same session's JSONL.

    ``sys.__stderr__`` in preference to ``sys.stderr``: quiet mode -- which is
    the default, not a flag -- points ``sys.stderr`` at ``os.devnull`` to
    swallow the embedding model's LOAD REPORT table, and on the CLI front doors
    this notice is the one thing Decision 16 requires the user actually sees.
    Falls back when the interpreter was started without a real stderr
    (``pythonw``, some embedded hosts), where ``sys.__stderr__`` is ``None``.

    Under ``aorta chat ui`` this stream is the *server's* console, which the
    person typing cannot see, so it is the log copy rather than the disclosure.
    :class:`NoticeState.pending` is what that front door renders.
    """
    return sys.__stderr__ if sys.__stderr__ is not None else sys.stderr


@dataclass(frozen=True)
class RedactionSummary:
    """What one redaction pass changed.

    Counts are rewrite *occurrences*, not distinct values, matching what
    :func:`aorta.probe.redaction.scrub_text` reports.
    """

    paths: int = 0
    ipv4: int = 0
    ipv6: int = 0

    @property
    def total(self) -> int:
        return self.paths + self.ipv4 + self.ipv6

    def __bool__(self) -> bool:
        return self.total > 0

    def describe(self) -> str:
        """Name what was removed, in the order a reader cares about."""
        parts: list[str] = []
        if self.paths:
            parts.append(f"{self.paths} filesystem path{'s' if self.paths != 1 else ''}")
        if self.ipv4:
            parts.append(f"{self.ipv4} IPv4 address{'es' if self.ipv4 != 1 else ''}")
        if self.ipv6:
            parts.append(f"{self.ipv6} IPv6 address{'es' if self.ipv6 != 1 else ''}")
        if not parts:
            return "nothing"
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _scrub(text: str) -> tuple[str, RedactionSummary]:
    """Apply the bundle scrubbers to one string.

    Imported per call rather than at module scope: ``aorta.probe.redaction``
    reaches ``aorta.triage.recipe`` for its YAML seam, and nothing should pay
    that to import ``aorta.chat``.
    """
    from aorta.probe.redaction import scrub_text

    scrubbed, paths, ipv4, ipv6 = scrub_text(
        text,
        scrub_paths=True,
        scrub_ip_addresses=True,
    )
    return scrubbed, RedactionSummary(paths=paths, ipv4=ipv4, ipv6=ipv6)


def redact_text(text: str) -> tuple[str, RedactionSummary]:
    """Redact one string, or return it untouched when the gate is off."""
    if not settings.redact or not text:
        return text, RedactionSummary()
    return _scrub(text)


def redact_messages(messages: list[Any]) -> tuple[list[Any], RedactionSummary]:
    """Redact the text content of every message about to be sent.

    Returns a new list; the caller's messages are left alone because the graph
    keeps them as conversation state and a user must still see their own paths
    echoed back. Only the copy that crosses the wire is rewritten.

    Non-string content (a tool-call block, or the list form a multimodal
    message uses) is passed through unchanged rather than coerced to text --
    rewriting a structure we do not model would corrupt the request, and
    ``str()`` on it would silently destroy the tool protocol.
    """
    if not settings.redact:
        return messages, RedactionSummary()

    out: list[Any] = []
    paths = ipv4 = ipv6 = 0
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content:
            out.append(message)
            continue
        scrubbed, counts = _scrub(content)
        paths += counts.paths
        ipv4 += counts.ipv4
        ipv6 += counts.ipv6
        # A no-op scrub must not clone the message: identity matters to
        # LangChain's tool-call bookkeeping, which pairs a ToolMessage with the
        # AIMessage that requested it.
        out.append(message.copy(update={"content": scrubbed}) if scrubbed != content else message)
    return out, RedactionSummary(paths=paths, ipv4=ipv4, ipv6=ipv6)


# ── first-send notice ─────────────────────────────────────────────────────


@dataclass
class NoticeState:
    """One session's record of the first-redaction disclosure.

    A mutable object rather than a bare flag in a :class:`ContextVar`, because
    the flag has to survive across the several asyncio tasks one session spans.
    A task copies the context at creation and its own writes do not propagate
    back, so a ``ContextVar[bool]`` flipped inside a graph node would be
    forgotten by the next message and the "once per session" notice would fire
    on every turn. The var carries a handle to shared state; the state is what
    changes.
    """

    emitted: bool = False
    #: The notice text, held for a front door that shows it itself rather than
    #: reading the process's stderr. ``None`` once drained (or never set).
    pending: str | None = field(default=None)


#: Fallback for the single-session front doors. ``aorta chat`` and
#: ``aorta chat ask`` are one session per process, so process-wide state is
#: session state there and no caller has to bind anything.
_process_notice = NoticeState()

_notice_state: ContextVar[NoticeState] = ContextVar(
    "aorta_chat_notice_state", default=_process_notice
)


def current_notice_state() -> NoticeState:
    """The :class:`NoticeState` in force for this call."""
    return _notice_state.get()


@contextmanager
def use_notice_state(state: NoticeState) -> Iterator[NoticeState]:
    """Bind *state* for the duration of the block.

    For a front door that multiplexes sessions over one process: the Chainlit
    server holds a state per browser session and binds it around each turn, so
    one user's redaction cannot consume another user's disclosure.
    """
    token = _notice_state.set(state)
    try:
        yield state
    finally:
        _notice_state.reset(token)


def take_pending_notice(state: NoticeState | None = None) -> str | None:
    """Remove and return the notice *state* has not displayed yet.

    Draining is what makes it show once: the caller renders whatever it gets
    back, and a second call in the same session gets ``None``.
    """
    state = state if state is not None else current_notice_state()
    pending, state.pending = state.pending, None
    return pending


def reset_session_notice(state: NoticeState | None = None) -> None:
    """Forget that the notice was shown. For tests, and for a new session."""
    state = state if state is not None else current_notice_state()
    state.emitted = False
    state.pending = None


def notice_line(summary: RedactionSummary) -> str:
    """The one line the user gets, naming the removal and the way out.

    Both halves are obligatory per Decision 16: a silent gate trains people to
    distrust the tool when an answer looks wrong, and naming what was removed
    without naming the opt-out leaves them stuck.
    """
    return (
        f"aorta chat: redacted {summary.describe()} from the outbound request. "
        "Disable with --no-redact, or 'redact = false' in "
        "~/.config/aorta/chat.toml."
    )


def emit_notice_once(summary: RedactionSummary, stream: IO[str] | None = None) -> bool:
    """Print the notice on the first send of this session that redacted anything.

    Returns whether it printed. Keyed on a non-empty summary rather than on the
    first send outright: a session whose prompts held no paths would otherwise
    be told about a redaction that never happened.

    The line always goes to the stream, which is the server's log under
    Chainlit, and is also parked on the session's :class:`NoticeState` for a
    front door that renders it to the user itself.
    """
    state = current_notice_state()
    if state.emitted or not summary:
        return False
    state.emitted = True
    line = notice_line(summary)
    state.pending = line
    target = stream if stream is not None else _notice_stream()
    print(line, file=target, flush=True)
    return True


def redact_for_send(messages: list[Any]) -> list[Any]:
    """Redact *messages* for one outbound call and emit the session notice.

    The single call every graph node makes on its way to the model.
    """
    redacted, summary = redact_messages(messages)
    emit_notice_once(summary)
    return redacted


__all__ = [
    "NoticeState",
    "RedactionSummary",
    "current_notice_state",
    "emit_notice_once",
    "notice_line",
    "redact_for_send",
    "redact_messages",
    "redact_text",
    "reset_session_notice",
    "take_pending_notice",
    "use_notice_state",
]
