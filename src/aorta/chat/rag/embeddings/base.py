"""Protocol shared by the local and remote embedding providers."""

from __future__ import annotations

from typing import Protocol

from langchain_core.embeddings import Embeddings


class EmbeddingProvider(Protocol):
    """One way of turning text into vectors, plus the collection it owns.

    Providers emit different vector dimensions, so each one owns a distinct
    Chroma collection name; they can then coexist in the same CHROMA_PATH.
    """

    name: str

    def get_embeddings(self) -> Embeddings:
        """Build the embedding model. May raise ValueError on bad config."""
        ...

    def collection_name(self) -> str:
        """Chroma collection this provider indexes into and reads from."""
        ...

    def describe(self) -> str:
        """One-line human-readable summary for logs and the welcome message."""
        ...
