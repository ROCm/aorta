"""Local BGE embeddings (sentence-transformers on CPU). No network at query time.

TRANSITIONAL. Phase 4 (Decision 19a) deletes this module in favour of
``fastembed_bge.py``, which runs the same BGE-small model on onnxruntime and so
produces the same vectors without torch. Until then the path stays, but its
dependency is NOT in the ``chat-cli`` extra: ``sentence-transformers``
hard-requires torch, and on a host without torch already present pip resolves
the default PyPI build, which is CUDA -- roughly 2.7 GB of ``nvidia-*`` wheels
landing on an AMD ROCm node. Opting into that has to be deliberate, hence
``chat-embeddings-torch``.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from aorta.chat.config import settings

#: What to tell a user who selected this provider without its extra.
#: ``langchain_huggingface`` itself imports fine without torch -- it is
#: ``HuggingFaceEmbeddings.__init__`` that reaches for sentence_transformers --
#: so the failure surfaces at construction, which is where this is raised.
_INSTALL_HINT = (
    "EMBEDDING_PROVIDER=local needs sentence-transformers, which is not part "
    "of the chat-cli extra because it pulls torch (and, on a host without "
    "torch, the CUDA build).\n"
    "  pip install 'amd-aorta[chat-embeddings-torch]'\n"
    "Or set AORTA_CHAT_EMBEDDING_PROVIDER=remote to use an embeddings API "
    "with no local model at all."
)

#: The historical collection name. Kept verbatim so indexes built before the
#: provider split keep working without a re-index.
LOCAL_COLLECTION_NAME = "aorta"

#: What ``HuggingFaceBgeEmbeddings.query_instruction`` defaulted to. BGE models
#: are trained with an asymmetric prompt: queries carry this prefix, passages do
#: not. Copied verbatim from langchain-community so vectors stay identical.
BGE_QUERY_INSTRUCTION = "Represent this question for searching relevant passages: "


class BgeEmbeddings(HuggingFaceEmbeddings):
    """``HuggingFaceEmbeddings`` plus the two things the BGE class did for us.

    ``langchain_community.embeddings.HuggingFaceBgeEmbeddings`` is deprecated,
    but its replacement is not a like-for-like swap: it prepends no query
    instruction and collapses no newlines. Losing either changes what actually
    gets embedded, and nothing would fail -- documents already indexed stay
    readable, queries just stop pairing with them the way they were indexed, so
    retrieval quietly gets worse. Both behaviours are reinstated here.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([text.replace("\n", " ") for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(BGE_QUERY_INSTRUCTION + text.replace("\n", " "))


class LocalBgeProvider:
    """EMBEDDING_PROVIDER=local -- unchanged behaviour, now behind an extra."""

    name = "local"

    def get_embeddings(self) -> BgeEmbeddings:
        try:
            return BgeEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except ImportError as exc:
            # langchain's own message says "pip install sentence-transformers",
            # which is true but hides both why it is not already there and the
            # remote provider that needs no local model at all.
            raise ValueError(_INSTALL_HINT) from exc

    def collection_name(self) -> str:
        return LOCAL_COLLECTION_NAME

    def describe(self) -> str:
        return f"local BGE embeddings ({settings.embedding_model}, cpu)"
