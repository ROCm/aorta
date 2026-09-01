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
        """Check the backend is usable. Runs once per session, not per call."""
        ...

    def describe(self) -> str:
        """One-line human-readable summary of the live configuration."""
        ...
