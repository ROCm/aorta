"""Local BGE-small embeddings on onnxruntime. No torch, no network at query time.

Replaces the ``sentence-transformers`` path deleted in Phase 4 (Decision 19a).
That path was the only reason the chat extras could deposit CUDA wheels on an
AMD ROCm node: ``sentence-transformers`` hard-requires torch, and on a host
without torch already present pip resolves the default PyPI build, which is the
CUDA one -- roughly 2.7 GB of ``nvidia-*`` for a 384-dimension text embedder.
``fastembed`` serves the same ``BAAI/bge-small-en-v1.5`` on onnxruntime instead,
so the retrieval quality is BGE's rather than MiniLM's and the hazard is gone by
construction rather than by warning.

One thing the swap does *not* preserve is bit-identical vectors. fastembed
sources this model from ``qdrant/bge-small-en-v1.5-onnx-q``, which is quantised
(67 MB against the 130 MB fp32 weights). Same architecture, same 384 dimensions,
near-identical rankings -- but not the same numbers, which is precisely why
:func:`FastembedBgeProvider.collection_name` encodes the provider *and* the
model. A dimension check alone would let a torch-built index load here and
answer plausibly from vectors it cannot actually compare against.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings

from aorta.chat.config import settings
from aorta.chat.rag.embeddings.base import build_collection_name

if TYPE_CHECKING:  # fastembed is imported lazily; see _text_embedding().
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

#: Prefix keeping this provider's collections apart from every other one. The
#: model is appended, so switching ``AORTA_CHAT_EMBEDDING_MODEL`` switches
#: collection rather than silently reusing vectors from a different model.
LOCAL_COLLECTION_PREFIX = "aorta_fastembed_"

#: The model this provider is built and tested around, and the one the published
#: index is built with. Also ``fastembed``'s own default, verified against
#: ``TextEmbedding.__init__`` rather than taken from its README.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

#: BGE is trained asymmetrically: queries carry an instruction prefix, passages
#: do not. Carried over verbatim from ``langchain_community``'s
#: ``HuggingFaceBgeEmbeddings`` default, which is what built every index to
#: date. fastembed's ``query_embed()`` applies no prefix of its own, so dropping
#: this would change what gets embedded on the query side only -- and nothing
#: would fail, retrieval would just quietly pair queries with passages
#: differently than it was tuned to. ``rag/eval.py`` is where its value is
#: measurable rather than assumed.
BGE_QUERY_INSTRUCTION = "Represent this question for searching relevant passages: "

#: How an air-gapped user gets the weights onto the node. Printed by
#: ``aorta chat doctor`` when the model is absent and HuggingFace is
#: unreachable, and attached to a download failure so the procedure arrives at
#: the moment the command fails rather than only in the docs (Decision 21b's
#: mitigation).
PRE_SEED_PROCEDURE = (
    "The {model} weights are not in the local cache and HuggingFace is not "
    "reachable, so they cannot be downloaded.\n"
    "\n"
    "To pre-seed the cache from a machine that does have egress:\n"
    "  1. On the connected machine, with the same aorta version installed:\n"
    "       export HF_HOME=/tmp/aorta-model-cache\n"
    "       aorta chat doctor            # downloads nothing\n"
    "       python -c 'from fastembed import TextEmbedding; "
    'TextEmbedding("{model}", cache_dir="/tmp/aorta-model-cache")\'\n'
    "  2. Copy that directory to this machine (it is ~90 MB):\n"
    "       rsync -a /tmp/aorta-model-cache/ <this-host>:{cache}/\n"
    "  3. On this machine, point at it and forbid network lookups:\n"
    "       export HF_HOME={cache}\n"
    "       export HF_HUB_OFFLINE=1\n"
    "\n"
    "{cache} is where this install looks. Setting HF_HOME moves it; without "
    "HF_HOME it is\nAORTA_CHAT_MODEL_CACHE_PATH, defaulting under the XDG cache.\n"
    "\n"
    "The prebuilt index is a separate artifact and a separate problem: fetch it\n"
    "elsewhere and side-load it with 'aorta chat index fetch --from <file>'."
)


class ModelUnavailableError(RuntimeError):
    """The embedding weights are neither cached nor downloadable.

    Carries :data:`PRE_SEED_PROCEDURE` so the caller does not have to know that
    a ``huggingface_hub`` connection error means "pre-seed a cache".
    """


def model_cache_dir() -> Path:
    """Where this install keeps the ONNX weights.

    An explicit directory, not fastembed's default. Verified rather than
    assumed: fastembed ignores ``HF_HOME`` and caches under
    ``/tmp/fastembed_cache``, which is wiped on reboot -- so every reboot would
    cost a 90 MB re-download -- and is world-writable on a shared node.

    ``HF_HOME`` still wins when set, because a user who set it did so
    deliberately and because pre-seeding through it is the HuggingFace idiom
    :data:`PRE_SEED_PROCEDURE` documents. fastembed lays out
    ``models--org--name`` directly under whatever directory it is given, so this
    coexists with ``huggingface_hub``'s own ``$HF_HOME/hub``.
    """
    raw = os.environ.get("HF_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(settings.model_cache_path).expanduser()


def _model_dir_slug(model: str) -> str:
    """The cache directory name for a repo id: ``org/name`` -> ``models--org--name``."""
    return "models--" + model.replace("/", "--")


def model_is_cached(model: str | None = None) -> bool:
    """Whether the ONNX weights are already on disk.

    Deliberately a filesystem check rather than a ``TextEmbedding(...)``
    construction: ``doctor`` must answer this without a 90 MB download as a side
    effect. fastembed downloads from a *source* repo rather than from the model
    id, so both names are accepted -- a cache seeded through either counts. The
    ``hub`` subdirectory is checked too, so a cache seeded by plain
    ``huggingface_hub`` into ``$HF_HOME/hub`` is recognised.
    """
    model = model or settings.embedding_model
    root = model_cache_dir()
    candidates = {_model_dir_slug(model), _model_dir_slug(_source_repo(model))}
    for base in (root, root / "hub"):
        for candidate in candidates:
            directory = base / candidate
            if directory.is_dir() and any(directory.rglob("*.onnx")):
                return True
    return False


def _source_repo(model: str) -> str:
    """The HuggingFace repo fastembed actually downloads ``model`` from.

    fastembed re-hosts ONNX conversions under its own org, so
    ``BAAI/bge-small-en-v1.5`` is fetched from ``qdrant/bge-small-en-v1.5-onnx-q``
    and that is the name the cache directory carries. Falls back to the model id
    when fastembed is absent or does not know the model, which only makes the
    cache probe conservative.
    """
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return model
    for description in TextEmbedding._list_supported_models():
        if description.model == model:
            source = getattr(description.sources, "hf", None)
            return source or model
    return model


def _text_embedding(model: str, cache_dir: Path | None) -> TextEmbedding:
    """Construct fastembed's model, converting a download failure into advice.

    This is the constructor that reaches the network on first use, so it is the
    one place where "no weights and no egress" can be named. Any exception
    raised while the weights are missing is re-raised as
    :class:`ModelUnavailableError`; a failure with the weights already present
    is a real bug and passes through untouched.
    """
    from fastembed import TextEmbedding

    try:
        return TextEmbedding(model_name=model, cache_dir=str(cache_dir) if cache_dir else None)
    except Exception as exc:
        if model_is_cached(model):
            raise
        raise ModelUnavailableError(
            PRE_SEED_PROCEDURE.format(model=model, cache=model_cache_dir())
            + f"\n\nUnderlying error: {type(exc).__name__}: {exc}"
        ) from exc


class FastembedBgeEmbeddings(Embeddings):
    """LangChain ``Embeddings`` over ``fastembed.TextEmbedding``.

    Written here rather than taken from ``langchain_community`` for two reasons:
    its wrapper applies no BGE query instruction, and it would put the whole of
    ``langchain_community`` on the retrieval critical path for thirty lines of
    adapter. Both halves of what ``HuggingFaceBgeEmbeddings`` used to do -- the
    query prefix and the newline collapse -- are reinstated below, so vectors
    stay comparable with how every index to date was built.

    The model is loaded on first embed, not in ``__init__``: constructing a
    provider must stay cheap enough for ``aorta chat doctor`` and the collection
    -name lookup, neither of which embeds anything.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Path | None = None) -> None:
        self.model_name = model_name
        # Resolved on first embed, not here, so a test or a job script that sets
        # HF_HOME after constructing the provider is still honoured.
        self.cache_dir = cache_dir
        self._model: TextEmbedding | None = None

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            self._model = _text_embedding(self.model_name, self.cache_dir or model_cache_dir())
        return self._model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # fastembed yields numpy arrays and normalises them itself (verified in
        # OnnxTextEmbedding._post_process_onnx_output), which is what the vec0
        # L2 ranking in SqliteVecStore assumes. tolist() keeps the store's
        # struct.pack path free of a numpy dependency.
        return [vector.tolist() for vector in self._get_model().embed(texts)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([text.replace("\n", " ") for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([BGE_QUERY_INSTRUCTION + text.replace("\n", " ")])[0]


class FastembedBgeProvider:
    """``EMBEDDING_PROVIDER=local`` -- BGE-small on onnxruntime, no torch."""

    name = "local"

    def get_embeddings(self) -> FastembedBgeEmbeddings:
        try:
            import fastembed  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "EMBEDDING_PROVIDER=local needs fastembed, which is part of the "
                "chat-cli extra.\n"
                "  pip install 'amd-aorta[chat-cli]'\n"
                "Or set AORTA_CHAT_EMBEDDING_PROVIDER=remote to use an "
                "embeddings API with no local model at all."
            ) from exc
        return FastembedBgeEmbeddings(model_name=settings.embedding_model)

    def collection_name(self) -> str:
        """Encodes the provider and the model, so no other index can load here.

        Sharing on dimension alone is the trap this closes: the deleted torch
        provider emitted 384-dimension vectors from the same model family, so an
        index it built would pass ``SqliteVecStore``'s dimension check and then
        answer from vectors that were never comparable.
        """
        return build_collection_name(LOCAL_COLLECTION_PREFIX, settings.embedding_model)

    def describe(self) -> str:
        return f"local BGE embeddings ({settings.embedding_model} on onnxruntime)"


def describe_model_state(model: str | None = None) -> dict[str, Any]:
    """Cache state for ``aorta chat doctor``, without downloading anything."""
    model = model or settings.embedding_model
    return {
        "model": model,
        "source_repo": _source_repo(model),
        "cache_dir": str(model_cache_dir()),
        "cached": model_is_cached(model),
        "offline": os.environ.get("HF_HUB_OFFLINE", "").strip() not in ("", "0"),
    }
