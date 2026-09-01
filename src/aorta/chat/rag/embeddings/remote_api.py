"""Remote embeddings over an OpenAI-compatible API. No local model is loaded."""

from __future__ import annotations

import re

from langchain_openai import OpenAIEmbeddings

from aorta.chat.config import settings
from aorta.chat.remote_auth import build_auth, describe_auth

#: Prefix keeping remote collections out of the local provider's "aorta".
REMOTE_COLLECTION_PREFIX = "aorta_remote_"

#: Cap on the generated name. sqlite imposes no such limit -- this is ours,
#: kept so a long model id cannot produce an unreadable table name.
_MAX_COLLECTION_NAME = 63


def _model_slug(model: str) -> str:
    """Reduce a model id to the bare identifier the store requires.

    The collection lands in a table name, which cannot be a bound parameter,
    so SqliteVecStore rejects anything outside ``[A-Za-z0-9_]``.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", model.strip().lower()).strip("_")
    return slug or "model"


class RemoteApiProvider:
    """EMBEDDING_PROVIDER=remote -- OpenAI, or any OpenAI-compatible endpoint."""

    name = "remote"

    def get_embeddings(self) -> OpenAIEmbeddings:
        api_key = settings.remote_embedding_api_key.strip()
        if not api_key:
            raise ValueError(
                "REMOTE_EMBEDDING_API_KEY is not set, but EMBEDDING_PROVIDER="
                "'remote'. Set REMOTE_EMBEDDING_API_KEY in your .env, or set "
                "EMBEDDING_PROVIDER=local to use the on-disk BGE model."
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
        name = REMOTE_COLLECTION_PREFIX + _model_slug(settings.remote_embedding_model)
        return name[:_MAX_COLLECTION_NAME].rstrip("_")

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
