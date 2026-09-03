"""The structural interface every chat backend implements.

Deliberately tiny, in the style of the ``LLMProposer`` protocol in the aorta
agent: one module per backend, and a single factory that turns a provider
name into an implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


@runtime_checkable
class ChatBackend(Protocol):
    """A source of chat models, plus its own readiness check."""

    name: str

    def get_chat_model(
        self,
        *,
        temperature: float = 0.1,
        streaming: bool = True,
    ) -> BaseChatModel:
        """Build a LangChain chat model for a graph node to invoke."""
        ...

    async def preflight(self) -> None:
        """Check the backend is usable. Runs once per session, not per call.

        Allowed to be permissive -- the local backend proceeds against a server
        that is still loading weights rather than refuse a session the user can
        see is warming up. Callers that must be told the truth use
        :meth:`probe`.
        """
        ...

    async def probe(self, timeout: float | None = None) -> None:
        """Raise unless the backend is usable *now*.

        The raising counterpart of :meth:`preflight`, and the reason the two are
        separate methods: ``aorta chat doctor`` exists to report what is broken,
        so a permissive readiness check is the wrong primitive for it. Backends
        that reach the network honour *timeout*; the ones that only validate
        configuration ignore it.
        """
        ...

    def unreachable_hint(self) -> str:
        """What to check when this backend's endpoint does not answer.

        Rendered instead of a traceback when a query fails to connect, so it
        names the address that was tried, the setting that holds it, and the way
        to change it.
        """
        ...

    def describe(self) -> str:
        """One-line human-readable summary of the live configuration."""
        ...
