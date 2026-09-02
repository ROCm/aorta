"""Protocol shared by the local and remote embedding providers."""

from __future__ import annotations

import re
from typing import Protocol

from langchain_core.embeddings import Embeddings

#: Collection names become part of a sqlite table name, which cannot be a bound
#: parameter, so they have to be bare identifiers; ``SqliteVecStore`` rejects
#: anything else. sqlite imposes no length limit -- 63 was Chroma's and is kept
#: as our own cap, so a long model id cannot produce an unreadable table name.
MAX_COLLECTION_NAME = 63


class EmbeddingProvider(Protocol):
    """One way of turning text into vectors, plus the collection it owns.

    Two providers must never share a collection. Where their dimensions differ
    the separation is enforced for us -- a vec0 table fixes its dimension in the
    CREATE statement, so they physically cannot share one -- but dimension
    equality is not enough to make sharing *safe*: BGE-small on onnxruntime and
    BGE-small on torch are both 384-dimension, but the ONNX weights are
    quantised, so the vectors differ and a cross-read degrades retrieval without
    raising. Every ``collection_name()`` therefore encodes the model, not just
    the flow.
    """

    name: str

    def get_embeddings(self) -> Embeddings:
        """Build the embedding model. May raise ValueError on bad config."""
        ...

    def collection_name(self) -> str:
        """Collection this provider indexes into and reads from."""
        ...

    def model_id(self) -> str:
        """The model whose vectors this provider actually produces.

        Each provider reads a different setting, so a caller that reaches for
        ``settings.embedding_model`` directly labels remote vectors with the
        local model's name -- which the corpus digest, the manifest and the
        load-time compatibility check all did. Ask the selected provider
        instead, and the three agree by construction.
        """
        ...

    def describe(self) -> str:
        """One-line human-readable summary for logs and the welcome message."""
        ...


def model_slug(model: str) -> str:
    """Reduce a model id to the alphanumerics and underscores a table name allows."""
    slug = re.sub(r"[^a-z0-9]+", "_", model.strip().lower()).strip("_")
    return slug or "model"


def build_collection_name(prefix: str, model: str) -> str:
    """``prefix`` + a slug of ``model``, capped at :data:`MAX_COLLECTION_NAME`."""
    return (prefix + model_slug(model))[:MAX_COLLECTION_NAME].rstrip("_")
