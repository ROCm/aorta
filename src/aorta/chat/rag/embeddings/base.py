"""Protocol shared by the local and remote embedding providers."""

from __future__ import annotations

from typing import Protocol

from langchain_core.embeddings import Embeddings


class EmbeddingProvider(Protocol):
    """One way of turning text into vectors, plus the collection it owns.

    Providers emit different vector dimensions, so each one owns a distinct
    collection name; they can then coexist in the same index file. Under
    sqlite-vec that separation is mandatory rather than tidy -- a vec0 table
    fixes its dimension in the CREATE statement, so two providers physically
    cannot share one.
    """

    name: str

    def get_embeddings(self) -> Embeddings:
        """Build the embedding model. May raise ValueError on bad config."""
        ...

    def collection_name(self) -> str:
        """Collection this provider indexes into and reads from.

        Must be a bare identifier: the store interpolates it into table names.
        """
        ...

    def describe(self) -> str:
        """One-line human-readable summary for logs and the welcome message."""
        ...
