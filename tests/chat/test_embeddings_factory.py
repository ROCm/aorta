"""Provider selection and collection naming for the embeddings layer.

Nothing here loads a model. The local BGE provider is only ever asked for its
name and collection -- never for ``get_embeddings()``, whose first embed would
download 65 MB of ONNX weights -- and the remote provider is built with a
placeholder key under the ``no_network`` guard.

Provider-specific behaviour lives in ``test_fastembed_provider.py``; this file
covers selection and the collection-naming contract between providers.
"""

from __future__ import annotations

import pytest

from aorta.chat.config import settings
from aorta.chat.rag.embeddings.base import model_slug
from aorta.chat.rag.embeddings.factory import collection_name, get_embeddings, get_provider
from aorta.chat.rag.embeddings.fastembed_bge import (
    LOCAL_COLLECTION_PREFIX,
    FastembedBgeProvider,
)
from aorta.chat.rag.embeddings.remote_api import REMOTE_COLLECTION_PREFIX, RemoteApiProvider

DEFAULT_REMOTE_MODEL = "text-embedding-3-small"


@pytest.fixture()
def remote_embedding_settings(monkeypatch):
    """Select the remote provider with a placeholder key that is never used."""
    monkeypatch.setattr(settings, "embedding_provider", "remote")
    monkeypatch.setattr(settings, "remote_embedding_model", DEFAULT_REMOTE_MODEL)
    monkeypatch.setattr(settings, "remote_embedding_api_key", "sk-placeholder-not-real")
    monkeypatch.setattr(settings, "remote_embedding_base_url", "")


class TestProviderSelection:
    def test_each_name_resolves_to_its_own_provider(self):
        assert isinstance(get_provider("local"), FastembedBgeProvider)
        assert isinstance(get_provider("remote"), RemoteApiProvider)

    def test_provider_name_matches_the_setting_string(self):
        assert get_provider("local").name == "local"
        assert get_provider("remote").name == "remote"

    def test_default_follows_the_embedding_provider_setting(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "local")
        assert isinstance(get_provider(), FastembedBgeProvider)
        monkeypatch.setattr(settings, "embedding_provider", "remote")
        assert isinstance(get_provider(), RemoteApiProvider)

    def test_provider_string_is_case_and_whitespace_tolerant(self):
        assert isinstance(get_provider("  LOCAL "), FastembedBgeProvider)
        assert isinstance(get_provider("Remote"), RemoteApiProvider)

    def test_unknown_provider_names_the_valid_choices(self):
        with pytest.raises(ValueError) as exc:
            get_provider("sbert")
        assert str(exc.value) == (
            "unknown embedding provider: 'sbert' (expected one of local, remote)"
        )

    def test_unknown_provider_in_settings_also_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "sbert")
        with pytest.raises(ValueError, match="unknown embedding provider: 'sbert'"):
            get_provider()

    @pytest.mark.parametrize("alias", ["onnx", "fastembed"])
    def test_runtime_named_aliases_resolve_to_local(self, alias: str):
        """Decision 19a left one local flow, but the discussion named the runtime.

        A profile saying ``onnx`` should start rather than fail on a name that
        was correct while there were two local providers.
        """
        assert isinstance(get_provider(alias), FastembedBgeProvider)


class TestCollectionNames:
    def test_local_collection_name_encodes_the_model(self, monkeypatch):
        """Not the bare ``aorta`` the deleted torch provider used.

        Both emit 384 dimensions from the same model family, so sharing a name
        would let one's index satisfy the other's only other check.
        """
        monkeypatch.setattr(settings, "embedding_provider", "local")
        monkeypatch.setattr(settings, "embedding_model", "BAAI/bge-small-en-v1.5")
        assert collection_name() == "aorta_fastembed_baai_bge_small_en_v1_5"
        assert collection_name().startswith(LOCAL_COLLECTION_PREFIX)

    def test_remote_collection_name_is_per_model(self, remote_embedding_settings):
        assert collection_name() == "aorta_remote_text_embedding_3_small"

    def test_the_two_providers_never_share_a_collection(self, monkeypatch):
        """The vector dimensions differ, so the collections must too."""
        monkeypatch.setattr(settings, "remote_embedding_model", DEFAULT_REMOTE_MODEL)
        local = get_provider("local").collection_name()
        remote = get_provider("remote").collection_name()
        assert local != remote
        assert remote.startswith(REMOTE_COLLECTION_PREFIX)
        assert not local.startswith(REMOTE_COLLECTION_PREFIX)

    def test_switching_model_switches_collection(self, monkeypatch):
        monkeypatch.setattr(settings, "remote_embedding_model", DEFAULT_REMOTE_MODEL)
        small = get_provider("remote").collection_name()
        monkeypatch.setattr(settings, "remote_embedding_model", "text-embedding-3-large")
        assert get_provider("remote").collection_name() != small

    def test_model_slug_keeps_only_what_a_table_name_accepts(self):
        assert model_slug("text-embedding-3-small") == "text_embedding_3_small"
        assert model_slug("Voyage/Voyage-3") == "voyage_voyage_3"
        assert model_slug("  spaced  model  ") == "spaced_model"

    def test_model_slug_never_returns_an_empty_name(self):
        """An empty collection name is not an identifier, so fall back to a literal."""
        assert model_slug("") == "model"
        assert model_slug("---") == "model"

    def test_long_model_names_stay_within_the_name_limit(self, monkeypatch):
        monkeypatch.setattr(settings, "remote_embedding_model", "x" * 49 + "-" + "y" * 20)
        name = get_provider("remote").collection_name()
        assert len(name) <= 63
        assert not name.endswith("_")


class TestRemoteApiProvider:
    def test_missing_key_is_reported_when_the_model_is_built(self, monkeypatch):
        """Selection stays cheap; only building embeddings needs the key."""
        monkeypatch.setattr(settings, "embedding_provider", "remote")
        monkeypatch.setattr(settings, "remote_embedding_api_key", "")
        provider = get_provider()
        assert isinstance(provider, RemoteApiProvider)
        with pytest.raises(ValueError) as exc:
            provider.get_embeddings()
        assert "REMOTE_EMBEDDING_API_KEY" in str(exc.value)
        assert "EMBEDDING_PROVIDER=local" in str(exc.value)

    def test_whitespace_only_key_counts_as_missing(self, monkeypatch):
        monkeypatch.setattr(settings, "remote_embedding_api_key", "   ")
        with pytest.raises(ValueError, match="REMOTE_EMBEDDING_API_KEY"):
            RemoteApiProvider().get_embeddings()

    def test_factory_builds_the_remote_model_without_a_network_call(
        self, remote_embedding_settings, no_network
    ):
        embeddings = get_embeddings()
        assert embeddings.model == DEFAULT_REMOTE_MODEL
        assert embeddings.openai_api_base is None

    def test_base_url_is_passed_through_when_set(
        self, monkeypatch, remote_embedding_settings, no_network
    ):
        monkeypatch.setattr(settings, "remote_embedding_base_url", "https://example.invalid/v1")
        assert get_embeddings().openai_api_base == "https://example.invalid/v1"

    def test_describe_names_the_model_and_endpoint(self, remote_embedding_settings):
        assert DEFAULT_REMOTE_MODEL in RemoteApiProvider().describe()
        assert "provider default" in RemoteApiProvider().describe()


class TestRetrieverCaches:
    def test_reset_caches_clears_both_halves(self, monkeypatch):
        """Vectorstore and retriever are tied to one provider, so both must go."""
        from aorta.chat.rag import retriever

        monkeypatch.setattr(retriever, "_vectorstore_cache", object())
        monkeypatch.setattr(retriever, "_retriever_cache", object())
        retriever.reset_caches()
        assert retriever._vectorstore_cache is None
        assert retriever._retriever_cache is None
