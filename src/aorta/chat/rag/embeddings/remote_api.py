"""Remote embeddings over an OpenAI-compatible API. No local model is loaded."""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from aorta.chat.config import settings
from aorta.chat.rag.embeddings.base import build_collection_name
from aorta.chat.remote_auth import build_auth, describe_auth

#: Prefix keeping remote collections apart from the local provider's.
REMOTE_COLLECTION_PREFIX = "aorta_remote_"


class RemoteApiProvider:
    """``embedding_provider=remote`` -- OpenAI, or any OpenAI-compatible endpoint."""

    name = "remote"

    def get_embeddings(self) -> OpenAIEmbeddings:
        api_key = settings.remote_embedding_api_key.strip()
        if not api_key:
            raise ValueError(
                "remote_embedding_api_key is not set, but embedding_provider is "
                "'remote'. Set AORTA_CHAT_REMOTE_EMBEDDING_API_KEY, or put "
                "remote_embedding_api_key in the chat profile "
                "('aorta chat config init'), or set "
                "AORTA_CHAT_EMBEDDING_PROVIDER=local to use the on-disk BGE model."
            )
        client_key, headers = build_auth(
            api_key=api_key,
            auth_header=settings.remote_embedding_auth_header,
            extra_headers=settings.remote_embedding_extra_headers,
        )
        # An empty base URL means "the provider's own default endpoint";
        # OpenAIEmbeddings wants it omitted, not passed as "".
        base_url = settings.remote_embedding_base_url.strip() or None
        return OpenAIEmbeddings(
            model=settings.remote_embedding_model,
            api_key=client_key,
            base_url=base_url,
            default_headers=headers,
        )

    def collection_name(self) -> str:
        return build_collection_name(
            REMOTE_COLLECTION_PREFIX,
            settings.remote_embedding_model,
            identity=self.vector_identity(),
        )

    def model_id(self) -> str:
        return settings.remote_embedding_model

    def vector_identity(self) -> str:
        """Endpoint and model together, because the model name alone is ambiguous.

        ``text-embedding-3-small`` is whatever the configured OpenAI-compatible
        API decides it is. Keying only on the name let a switch from endpoint A
        to endpoint B keep A's stored vectors and query them with B's, which
        passes every dimension and model check and returns plausible nonsense.
        The empty base URL is the provider's own default, and is a distinct
        endpoint from any explicit one.
        """
        endpoint = settings.remote_embedding_base_url.strip().rstrip("/") or "<provider-default>"
        return f"{endpoint}\n{settings.remote_embedding_model}"

    def describe(self) -> str:
        endpoint = settings.remote_embedding_base_url.strip() or "provider default"
        auth = describe_auth(
            auth_header=settings.remote_embedding_auth_header,
            extra_headers=settings.remote_embedding_extra_headers,
        )
        return (
            f"remote embeddings ({settings.remote_embedding_model} "
            f"via {endpoint}, auth: {auth})"
        )
