"""Per-session, bounded reuse of results that cost a cluster job to produce.

Triaging a kernel is a multi-minute job on a GPU node, and the same kernel
arrives more than once in a conversation: the model re-checks its own answer
and the critic can re-run a turn. Reusing the verdict for an identical paste is
what keeps that to one job per kernel rather than one per attempt.

Where the reuse has to stop is at the edge of the conversation. A module-level
dict in the Chainlit server is one dict for every browser session it serves, so
a second user pasting the same kernel was handed the first user's verdict and
told a job had been reused -- someone else's job, over their code. The same
dict never evicted, so it accumulated every kernel anyone had ever submitted
for as long as the process lived.

The shape here is the one :mod:`aorta.chat.redaction` already uses for its
per-session notice: a mutable holder, a :class:`ContextVar` carrying a handle
to it, and a context manager the front door binds around a turn. A var holding
the cache itself would not do, because a task copies the context at creation
and its writes do not propagate back -- entries stored while answering would be
forgotten by the next message.

This module deliberately imports nothing from the diagnostic stack: the
Chainlit app binds a cache per session and must stay importable without the
``[cia]`` extra installed.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Hashable, Iterator

#: Entries kept per cache, per session, before the least recently used is
#: dropped. A conversation that revisits more than this many distinct kernels
#: is re-running the cheapest part of the work, which is the right thing to
#: lose: the alternative is holding every paste for the process's lifetime.
MAX_ENTRIES = 32


class BoundedCache:
    """A small LRU over pasted source, holding rendered tool output.

    Keys are the paste itself, so entries are as large as what the user sent
    and a cap on their number is what keeps the process's memory bounded.
    """

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self._max = max_entries
        self._entries: OrderedDict[Hashable, str] = OrderedDict()

    def get(self, key: Hashable) -> str | None:
        """The stored output for *key*, marking it as most recently used."""
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return self._entries[key]

    def put(self, key: Hashable, value: str) -> None:
        """Store *value*, evicting the least recently used entry if full."""
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._entries


class ToolCache:
    """One conversation's memory of tool results already computed.

    The two are bounded separately because they are not worth the same. A
    triage verdict cost minutes on a GPU node; an assembly analysis cost about
    a second. Sharing one bound would let a run of pasted fragments evict the
    expensive answers.
    """

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self.triage = BoundedCache(max_entries)
        self.asm = BoundedCache(max_entries)

    def clear(self) -> None:
        self.triage.clear()
        self.asm.clear()


#: Fallback for the single-session front doors. ``aorta chat`` and
#: ``aorta chat ask`` are one conversation per process, so process-wide state
#: is session state there and no caller has to bind anything.
_process_cache = ToolCache()

_tool_cache: ContextVar[ToolCache] = ContextVar("aorta_chat_tool_cache", default=_process_cache)


def current_tool_cache() -> ToolCache:
    """The :class:`ToolCache` in force for this call."""
    return _tool_cache.get()


@contextmanager
def use_tool_cache(cache: ToolCache) -> Iterator[ToolCache]:
    """Bind *cache* for the duration of the block.

    For a front door that multiplexes conversations over one process: the
    Chainlit server holds a cache per browser session and binds it around each
    turn, so one user's paste cannot be answered from another user's run.
    """
    token = _tool_cache.set(cache)
    try:
        yield cache
    finally:
        _tool_cache.reset(token)


def reset_tool_cache(cache: ToolCache | None = None) -> None:
    """Empty the cache in force, or *cache* if given.

    Mainly for tests, which otherwise inherit whatever the process-wide
    fallback accumulated in an earlier one.
    """
    (cache if cache is not None else current_tool_cache()).clear()


__all__ = [
    "MAX_ENTRIES",
    "BoundedCache",
    "ToolCache",
    "current_tool_cache",
    "reset_tool_cache",
    "use_tool_cache",
]
