"""Per-query LLM call counting, for remote-provider cost visibility.

One question fans out into many model calls: router, plan, an act loop of up to
``max_act_rounds_search`` rounds followed by a synthesis call, and a critic
pass after every act pass.

``max_retry_iterations`` counts act *passes*, not hand-backs: the critic
increments ``iteration`` and returns to ``act`` only while it is below the cap,
so the shipped 3 buys three act passes and two hand-backs. The hand-backs are
where the cost is, because each one re-runs the *whole* act loop rather than
adding a round to it -- so the ceiling goes as
``max_retry_iterations * max_act_rounds_search`` and not as their sum. On the
shipped caps a search query the critic rejects every pass costs 32 calls,
measured, and 34 if it is also the query that escalates to the native tool
protocol (the two extra text rounds that buy the diagnosis). A turn the critic
accepts first time is 14. See ``docs/chat/providers.md`` for the breakdown.

Against a metered endpoint that is real money, so the remote backends attach
:class:`LLMCallCounter` to every chat model they build and ``invoke_agent``
logs the total for the query at INFO.

The local vLLM backend deliberately does not attach it: those calls cost
nothing, and its chat-model construction stays identical to the pre-provider
code.

The tally is a process-wide total read as a before/after delta, so
overlapping Chainlit sessions inflate each other's numbers rather than
dropping calls. It is a spend indicator, not an accounting record.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_total_calls = 0


class LLMCallCounter(BaseCallbackHandler):
    """Counts chat-model invocations made through the remote backends."""

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: Any,
        **kwargs: Any,
    ) -> None:
        global _total_calls
        with _lock:
            _total_calls += 1


@contextmanager
def count_llm_calls(label: str = "query") -> Iterator[None]:
    """Log how many remote LLM calls the wrapped block made."""
    before = _total_calls
    try:
        yield
    finally:
        made = _total_calls - before
        if made:
            logger.info("Remote LLM calls for this %s: %d", label, made)
