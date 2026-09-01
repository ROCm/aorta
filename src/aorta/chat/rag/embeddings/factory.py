"""Selects the embedding provider from settings.embedding_provider.

Local and remote live in sibling modules that never import each other: a flow
is removed by deleting its module and its entry in _PROVIDERS below.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from aorta.chat.config import settings
from aorta.chat.rag.embeddings.base import EmbeddingProvider
from aorta.chat.rag.embeddings.local_bge import LocalBgeProvider
from aorta.chat.rag.embeddings.remote_api import RemoteApiProvider

_PROVIDERS = {
    "local": LocalBgeProvider,
    "remote": RemoteApiProvider,
}


def get_provider(name: str | None = None) -> EmbeddingProvider:
    """Return the embedding provider named by settings.embedding_provider.

    Args:
        name: Override the configured provider; mainly for tests.

    Raises:
        ValueError: If the provider name is not registered.
    """
    key = (name or settings.embedding_provider).strip().lower()
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
    """Chroma collection for the configured provider.

    Local returns "aorta" -- the pre-split name -- so existing on-disk indexes
    are read unchanged. Remote returns a per-model name, because the two
    providers emit different vector dimensions and cannot share a collection.
    """
    return get_provider().collection_name()
