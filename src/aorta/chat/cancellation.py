"""Per-tool cancellation propagated from the graph into synchronous workers.

Cancelling an asyncio Future does not interrupt the callable already running in
its executor. Diagnostic tools then continued through their internal timeout
and held a Slurm allocation after the browser session had disconnected.

A context variable crosses the ``BaseTool.ainvoke`` executor boundary without
adding an argument to every LangChain tool schema. Each tool call binds its own
event, so cancelling one call cannot stop another.
"""

from __future__ import annotations

import contextvars
import threading

_CANCEL: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "aorta_chat_cancel", default=None
)


def bind_cancel_token(token: threading.Event) -> contextvars.Token:
    """Bind *token* in this execution context and return its reset handle."""
    return _CANCEL.set(token)


def reset_cancel_token(bound: contextvars.Token) -> None:
    """Restore the context that preceded :func:`bind_cancel_token`."""
    _CANCEL.reset(bound)


def current_cancel_token() -> threading.Event | None:
    """The cancellation event for the current tool call, if it has one."""
    return _CANCEL.get()
