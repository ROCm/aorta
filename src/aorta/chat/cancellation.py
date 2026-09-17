"""The cancellation token for one tool call, carried to the thread running it.

A tool runs on a worker thread, and cancelling the coroutine that awaited it
does not reach the thread: Python cannot interrupt a running thread, so the
work continues -- for a diagnostic tool, until the triage timeout, with a Slurm
allocation held for as long as that takes. The turn was over and the cluster
did not know.

A context variable rather than an argument because the path between the two is
not ours to widen: the coroutine calls a LangChain ``BaseTool``, which calls
the function underneath it, which calls the triage seam. The executor already
copies the context across for the per-conversation cache, so this rides the
same way.

Stdlib only, and deliberately not :mod:`aorta.cia.cancellation`: chat has to
import this without the cia extra installed.
"""

from __future__ import annotations

import contextvars
import threading

#: The event the current tool call watches, or None outside one.
_CANCEL: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "aorta_chat_cancel", default=None
)


def bind_cancel_token(token: threading.Event) -> None:
    """Make *token* the cancellation signal for work in this context.

    Called inside the worker's copied context, so it binds there and not in
    the caller's -- two tool calls run at once and must not share a token.
    """
    _CANCEL.set(token)


def current_cancel_token() -> threading.Event | None:
    """The token for the tool call on this thread, if it was given one."""
    return _CANCEL.get()


def cancelled() -> bool:
    """Whether the caller has given up on this tool call."""
    token = _CANCEL.get()
    return token is not None and token.is_set()
