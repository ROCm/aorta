"""Local BGE embeddings (sentence-transformers on CPU). No network at query time."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from aorta.chat.config import settings

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
    gets embedded, and nothing would fail -- documents already in Chroma stay
    readable, queries just stop pairing with them the way they were indexed, so
    retrieval quietly gets worse. Both behaviours are reinstated here.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([text.replace("\n", " ") for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(BGE_QUERY_INSTRUCTION + text.replace("\n", " "))


class LocalBgeProvider:
    """EMBEDDING_PROVIDER=local -- the default, unchanged behaviour."""

    name = "local"

    def get_embeddings(self) -> BgeEmbeddings:
        return BgeEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def collection_name(self) -> str:
        return LOCAL_COLLECTION_NAME

    def describe(self) -> str:
        return f"local BGE embeddings ({settings.embedding_model}, cpu)"
