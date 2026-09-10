"""The structural interface every chat backend implements.

Deliberately tiny, in the style of the ``LLMProposer`` protocol in the aorta
agent: one module per backend, and a single factory that turns a provider
name into an implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


#: The :attr:`ChatBackend.native_requirement` both remote backends share.
#:
#: Protocol-neutral on purpose, and shared rather than written out twice. The
#: LiteLLM flow exists precisely for providers with a native, non-OpenAI
#: protocol -- Anthropic, Gemini, Bedrock -- so calling every remote endpoint
#: an "OpenAI-compatible gateway" described those users' setup wrongly while
#: telling them what it needs.
REMOTE_NATIVE_REQUIREMENT = (
    "It needs an endpoint that accepts the 'tools' parameter, which a remote\n"
    "provider's tool-calling API normally does."
)


@runtime_checkable
class ChatBackend(Protocol):
    """A source of chat models, plus its own readiness check."""

    name: str

    #: What ``native`` tool mode costs on this backend, beyond the setting
    #: itself. Advice text, in the same spirit as :meth:`unreachable_hint`:
    #: whether an endpoint accepts the ``tools`` parameter is a fact about the
    #: backend, so the sentence that says so lives with it.
    #:
    #: Here rather than in ``aorta chat doctor`` because doctor used to keep
    #: its own set of provider names to decide this, and a backend added to
    #: the factory was silently treated as needing nothing and having no
    #: model. The factory is the only registry (see :mod:`.factory`); anything
    #: that varies per provider is reached through this interface.
    native_requirement: str

    @property
    def model_name(self) -> str:
        """The model this backend is configured to send to, read live.

        A property rather than an attribute because the settings behind it are
        read at call time -- a test or a job script that changes the model
        after the backend is constructed gets the new value.
        """
        ...

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
