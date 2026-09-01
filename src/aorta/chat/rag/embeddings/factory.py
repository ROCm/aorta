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
    # Phase 4 (Decision 19a): "onnx" -> FastembedBgeProvider, in a new sibling
    # module fastembed_bge.py. Same BGE-small model and therefore the same
    # vectors, on onnxruntime instead of torch, which removes the CUDA-wheels
    # hazard the "local" provider carries. It becomes the default and the
    # "local" entry above is deleted with it.
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
    """Collection for the configured provider.

    Local returns the bare "aorta"; remote returns a per-model name, because
    the two emit different vector dimensions and cannot share a collection.
    The names predate the sqlite-vec swap, which invalidated every on-disk
    index -- they are kept for continuity of the config surface, not because
    an older index can still be read.
    """
    return get_provider().collection_name()
