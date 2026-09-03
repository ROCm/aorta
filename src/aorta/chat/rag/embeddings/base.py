"""Protocol shared by the local and remote embedding providers."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

from langchain_core.embeddings import Embeddings

#: Collection names become part of a sqlite table name, which cannot be a bound
#: parameter, so they have to be bare identifiers; ``SqliteVecStore`` rejects
#: anything else. sqlite imposes no length limit -- 63 was Chroma's and is kept
#: as our own cap, so a long model id cannot produce an unreadable table name.
MAX_COLLECTION_NAME = 63

#: Hex characters of the identity digest every collection name carries. The
#: slug alone is not injective -- it discards punctuation, so ``foo/bar`` and
#: ``foo-bar`` collide, and it is truncated, so ids differing after the cap
#: collide too. Eight hex characters is 32 bits over a set of identities that
#: is realistically single-digit per install; the slug remains the readable
#: part, and this is what makes the whole name unique.
DIGEST_CHARS = 8


class EmbeddingProvider(Protocol):
    """One way of turning text into vectors, plus the collection it owns.

    Two providers must never share a collection. Where their dimensions differ
    the separation is enforced for us -- a vec0 table fixes its dimension in the
    CREATE statement, so they physically cannot share one -- but dimension
    equality is not enough to make sharing *safe*: BGE-small on onnxruntime and
    BGE-small on torch are both 384-dimension, but the ONNX weights are
    quantised, so the vectors differ and a cross-read degrades retrieval without
    raising. Every ``collection_name()`` therefore encodes the flow, the model
    and -- via a digest of :meth:`vector_identity` -- everything else that
    decides which vectors come out, since the readable slug alone is neither
    injective nor unbounded.
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

    def vector_identity(self) -> str:
        """Everything that has to match for two vectors to be comparable.

        A superset of :meth:`model_id`: for a remote provider the model name is
        endpoint-local, so ``text-embedding-3-small`` at two different
        OpenAI-compatible gateways is two vector spaces that share a name.
        Recorded in the manifest and digested into the collection name, which
        is what makes an endpoint switch a refusal instead of plausible
        nonsense.
        """
        ...

    def describe(self) -> str:
        """One-line human-readable summary for logs and the welcome message."""
        ...


def model_slug(model: str) -> str:
    """Reduce a model id to the alphanumerics and underscores a table name allows."""
    slug = re.sub(r"[^a-z0-9]+", "_", model.strip().lower()).strip("_")
    return slug or "model"


def identity_digest(identity: str) -> str:
    """Stable short digest of a full vector identity."""
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def build_collection_name(prefix: str, model: str, *, identity: str | None = None) -> str:
    """``prefix`` + a readable slug of ``model`` + a digest of ``identity``.

    ``identity`` defaults to ``model`` and is whatever fully determines the
    vector space -- for a remote provider that includes the endpoint, because
    an arbitrary OpenAI-compatible model name only means something relative to
    the API serving it.

    The digest is appended rather than mixed in, and the slug is truncated to
    make room for it, so the name stays within :data:`MAX_COLLECTION_NAME`
    while two different identities can no longer land on one collection. That
    matters most for the run-artifact collection, which is keyed by name and
    carries no manifest to catch a mismatch after the fact.
    """
    suffix = "_" + identity_digest(identity if identity is not None else model)
    room = MAX_COLLECTION_NAME - len(prefix) - len(suffix)
    return prefix + model_slug(model)[:room].rstrip("_") + suffix
