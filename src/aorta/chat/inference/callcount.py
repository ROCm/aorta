"""Per-query LLM call counting, for remote-provider cost visibility.

One question fans out into many model calls: router, plan, up to
``max_act_rounds_search`` act rounds, answer, and up to
``max_retry_iterations`` critic retries -- roughly fifteen in the worst case.
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
