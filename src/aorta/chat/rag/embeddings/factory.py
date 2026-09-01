"""Selects the embedding provider from settings.embedding_provider.

Local and remote live in sibling modules that never import each other: a flow
is removed by deleting its module and its entry in _PROVIDERS below.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from aorta.chat.config import settings
from aorta.chat.rag.embeddings.base import EmbeddingProvider
from aorta.chat.rag.embeddings.fastembed_bge import FastembedBgeProvider
from aorta.chat.rag.embeddings.remote_api import RemoteApiProvider

_PROVIDERS = {
    "local": FastembedBgeProvider,
    "remote": RemoteApiProvider,
}

#: Accepted spellings of "local". Decision 19a left one local flow rather than
#: two, so the name now describes where inference happens, not which runtime
#: does it -- but the runtime is what the discussion called it, and a profile
#: saying ``onnx`` should not fail to start.
_ALIASES = {
    "onnx": "local",
    "fastembed": "local",
}


def get_provider(name: str | None = None) -> EmbeddingProvider:
    """Return the embedding provider named by settings.embedding_provider.

    Args:
        name: Override the configured provider; mainly for tests.

    Raises:
        ValueError: If the provider name is not registered.
    """
    key = (name or settings.embedding_provider).strip().lower()
    key = _ALIASES.get(key, key)
    provider_cls = _PROVIDERS.get(key)
    if provider_cls is None:
        raise ValueError(
            f"unknown embedding provider: {key!r} "
            f"(expected one of {', '.join(sorted(_PROVIDERS))})"
        )
    return provider_cls()


def get_embeddings() -> Embeddings:
    """Build the configured embedding model."""
    return get_provider().get_embeddings()


def collection_name() -> str:
    """Collection for the configured provider.

    Every name encodes both the flow and the model. Dimension equality is not a
    safe basis for sharing one -- see :class:`~aorta.chat.rag.embeddings.base`.
    """
    return get_provider().collection_name()
