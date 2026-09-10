"""Recognising "the endpoint did not answer", so it can be reported as such.

A connection failure is an operational condition rather than a bug: nothing is
listening, or the base URL points elsewhere. It arrives, though, as an exception
raised deep inside ``httpcore`` and re-wrapped by ``httpx``, the openai SDK and
``langchain_openai`` in turn -- so rendering it as a traceback costs the reader
a screen and a half of frames and never names the address that was tried.

The advice itself belongs to whichever backend was configured, so each one
supplies its own ``unreachable_hint()``; this module only answers "is this that
kind of failure".
"""

from __future__ import annotations

import httpx
from openai import APIConnectionError


class BackendUnreachableError(RuntimeError):
    """A backend's readiness probe found nothing answering.

    Carries operator-facing advice as its message: both callers -- ``aorta chat
    doctor`` and the CLI's failure path -- print it verbatim.
    """


#: Exception types that mean "the configured endpoint was not reachable".
#: ``openai.APIConnectionError`` covers the LangChain and LiteLLM paths as well:
#: ``langchain_openai.chat_models.base.OpenAIConnectionError`` and LiteLLM's own
#: ``APIConnectionError`` both subclass it.
_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    BackendUnreachableError,
    APIConnectionError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
)


def is_connection_failure(exc: BaseException) -> bool:
    """Whether *exc*, or anything it chains to, is a failure to reach the endpoint.

    The chain has to be walked rather than the top frame inspected: LangGraph
    wraps a node's exception, and the openai SDK sets ``__cause__`` to the httpx
    error underneath it.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _CONNECTION_ERRORS):
            return True
        current = current.__cause__ or current.__context__
    return False


__all__ = ["BackendUnreachableError", "is_connection_failure"]
