"""Retriever that loads the persisted ChromaDB collection."""

from __future__ import annotations

from pathlib import Path

from langchain_core.vectorstores import VectorStoreRetriever

from aorta.chat.config import settings
from aorta.chat.rag.embeddings.factory import get_provider
from aorta.chat.rag.sqlite_compat import ensure_modern_sqlite

# ``langchain_chroma`` imports ``chromadb`` at module import time, where the
# deprecated ``langchain_community.vectorstores.Chroma`` imported it lazily
# inside its constructor. The sqlite swap therefore has to happen before the
# import below, not merely before the first Chroma() call.
ensure_modern_sqlite()

from langchain_chroma import Chroma  # noqa: E402

_retriever_cache: VectorStoreRetriever | None = None
_vectorstore_cache: Chroma | None = None


def reset_caches() -> None:
    """Drop the cached vectorstore and retriever.

    Both caches are tied to one embedding provider, so they must be cleared
    together when the configured provider changes within a process.
    """
    global _retriever_cache, _vectorstore_cache
    _retriever_cache = None
    _vectorstore_cache = None


def _get_vectorstore() -> Chroma:
    """Return the cached ChromaDB vectorstore, initialising it on first call."""
    global _vectorstore_cache
    if _vectorstore_cache is not None:
        return _vectorstore_cache

    chroma_dir = Path(settings.chroma_path)
    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at {chroma_dir}. "
            "Run `python scripts/index_aorta.py` first."
        )

    # One provider decides both halves: embeddings and the collection they
    # were written to. Mixing them would query the wrong vector dimension.
    provider = get_provider()
    _vectorstore_cache = Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=provider.get_embeddings(),
        collection_name=provider.collection_name(),
    )
    return _vectorstore_cache


def get_retriever(k: int | None = None) -> VectorStoreRetriever:
    """Return a retriever over the persisted AORTA ChromaDB collection.

    Uses MMR (Maximal Marginal Relevance) to balance relevance and diversity.
    Results are cached after the first call.
    """
    global _retriever_cache
    if _retriever_cache is not None:
        return _retriever_cache

    k = k or settings.retriever_k
    fetch_k = settings.retriever_fetch_k
    _retriever_cache = _get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.7},
    )
    return _retriever_cache


def search_docs(query: str, k: int) -> list:
    """Search the ChromaDB collection for exactly k results using MMR.

    Unlike the cached retriever (which has a fixed k), this function queries
    the vectorstore directly so callers can request any k at call time.
    """
    fetch_k = max(settings.retriever_fetch_k, k * 2)
    return _get_vectorstore().max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=0.7
    )
