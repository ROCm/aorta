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
        assert collection_name().startswith(LOCAL_COLLECTION_PREFIX)
        assert "baai_bge_small_en_v1_5" in collection_name()

    def test_remote_collection_name_is_per_model(self, remote_embedding_settings):
        assert collection_name().startswith(REMOTE_COLLECTION_PREFIX)
        assert "text_embedding_3_small" in collection_name()

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


class TestCollectionNamesAreUnique:
    """The slug alone is not injective, and the run collection has no manifest.

    Source retrieval refuses a mismatch via its sidecar, but the run-artifact
    collection is keyed by name and carries no model record, so two models that
    collide on a name and share a dimension silently query each other's
    vectors. The digest in the name is what stops that.
    """

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            # Punctuation is discarded, so these slug identically.
            ("foo/bar", "foo-bar"),
            ("foo.bar", "foo_bar"),
            ("Voyage/Voyage-3", "voyage-voyage-3"),
            # Differ only after the 63-character cap.
            ("m" * 70 + "-alpha", "m" * 70 + "-beta"),
        ],
    )
    def test_models_that_slug_alike_do_not_share_a_collection(self, monkeypatch, left, right):
        monkeypatch.setattr(settings, "remote_embedding_model", left)
        left_name = get_provider("remote").collection_name()
        monkeypatch.setattr(settings, "remote_embedding_model", right)

        assert get_provider("remote").collection_name() != left_name

    def test_a_colliding_name_still_fits_and_is_an_identifier(self, monkeypatch):
        from aorta.chat.rag.retriever import _SAFE_COLLECTION

        monkeypatch.setattr(settings, "remote_embedding_model", "z" * 200)
        name = get_provider("remote").collection_name()

        assert len(name) <= 63
        assert _SAFE_COLLECTION.match(name)

    def test_the_name_is_stable_across_calls(self, monkeypatch):
        """A digest that moved would orphan the collection it named yesterday."""
        monkeypatch.setattr(settings, "remote_embedding_model", "text-embedding-3-small")

        assert get_provider("remote").collection_name() == (
            get_provider("remote").collection_name()
        )


class TestTheRemoteEndpointIsPartOfTheIdentity:
    """A model name is endpoint-local for an arbitrary OpenAI-compatible API.

    Keeping ``text-embedding-3-small`` while moving from gateway A to gateway B
    kept A's stored vectors and queried them with B's. Same name, same
    dimensions, so the model check and the dimension check both passed and
    retrieval returned plausible nonsense.
    """

    @pytest.fixture(autouse=True)
    def _remote(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "remote")
        monkeypatch.setattr(settings, "remote_embedding_model", "text-embedding-3-small")
        monkeypatch.setattr(settings, "remote_embedding_api_key", "sk-test")

    def test_switching_endpoint_switches_collection(self, monkeypatch):
        monkeypatch.setattr(settings, "remote_embedding_base_url", "https://a.example/v1")
        at_a = get_provider("remote").collection_name()
        monkeypatch.setattr(settings, "remote_embedding_base_url", "https://b.example/v1")

        assert get_provider("remote").collection_name() != at_a

    def test_the_provider_default_is_its_own_endpoint(self, monkeypatch):
        """An empty base URL is a real endpoint, not "any endpoint"."""
        monkeypatch.setattr(settings, "remote_embedding_base_url", "")
        default = get_provider("remote").collection_name()
        monkeypatch.setattr(settings, "remote_embedding_base_url", "https://a.example/v1")

        assert get_provider("remote").collection_name() != default

    def test_a_trailing_slash_is_the_same_endpoint(self, monkeypatch):
        """Otherwise a cosmetic profile edit orphans the whole index."""
        monkeypatch.setattr(settings, "remote_embedding_base_url", "https://a.example/v1")
        plain = get_provider("remote").collection_name()
        monkeypatch.setattr(settings, "remote_embedding_base_url", "https://a.example/v1/")

        assert get_provider("remote").collection_name() == plain

    def test_the_identity_names_both_halves(self, monkeypatch):
        monkeypatch.setattr(settings, "remote_embedding_base_url", "https://a.example/v1")
        identity = get_provider("remote").vector_identity()

        assert "https://a.example/v1" in identity
        assert "text-embedding-3-small" in identity

    def test_the_local_identity_is_just_the_model(self, monkeypatch):
        """Weights come from a fixed repo, so there is no endpoint to record."""
        monkeypatch.setattr(settings, "embedding_model", "BAAI/bge-small-en-v1.5")

        assert get_provider("local").vector_identity() == "BAAI/bge-small-en-v1.5"


class TestRemoteApiProvider:
    def test_missing_key_is_reported_when_the_model_is_built(self, monkeypatch):
        """Selection stays cheap; only building embeddings needs the key."""
        monkeypatch.setattr(settings, "embedding_provider", "remote")
        monkeypatch.setattr(settings, "remote_embedding_api_key", "")
        provider = get_provider()
        assert isinstance(provider, RemoteApiProvider)
        with pytest.raises(ValueError) as exc:
            provider.get_embeddings()
        # The namespaced spellings, not the bare ones: pydantic-settings reads
        # only AORTA_CHAT_*, so advice naming a bare name does nothing.
        assert "AORTA_CHAT_REMOTE_EMBEDDING_API_KEY" in str(exc.value)
        assert "AORTA_CHAT_EMBEDDING_PROVIDER=local" in str(exc.value)

    def test_whitespace_only_key_counts_as_missing(self, monkeypatch):
        monkeypatch.setattr(settings, "remote_embedding_api_key", "   ")
        with pytest.raises(ValueError, match="AORTA_CHAT_REMOTE_EMBEDDING_API_KEY"):
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


class TestEffectiveModelId:
    """Each provider names the model whose vectors it actually produces.

    Reaching for ``settings.embedding_model`` directly labelled remote vectors
    with the local model's name, in the corpus digest, the manifest and the
    load-time compatibility check alike -- so the check compared that wrong
    label against the same wrong setting and agreed with itself. Asking the
    selected provider is what makes the three agree by construction.
    """

    def test_the_local_provider_reports_the_local_setting(self, monkeypatch):
        from aorta.chat.rag.embeddings.fastembed_bge import FastembedBgeProvider

        monkeypatch.setattr(settings, "embedding_model", "BAAI/bge-small-en-v1.5")
        assert FastembedBgeProvider().model_id() == "BAAI/bge-small-en-v1.5"

    def test_the_remote_provider_reports_the_remote_setting(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_model", "BAAI/bge-small-en-v1.5")
        monkeypatch.setattr(settings, "remote_embedding_model", "text-embedding-3-small")
        assert RemoteApiProvider().model_id() == "text-embedding-3-small"

    def test_the_two_providers_disagree_which_is_the_whole_point(self, monkeypatch):
        from aorta.chat.rag.embeddings.fastembed_bge import FastembedBgeProvider

        monkeypatch.setattr(settings, "embedding_model", "BAAI/bge-small-en-v1.5")
        monkeypatch.setattr(settings, "remote_embedding_model", "text-embedding-3-small")
        assert FastembedBgeProvider().model_id() != RemoteApiProvider().model_id()


class TestRetrieverCaches:
    def test_reset_caches_clears_both_halves(self, monkeypatch):
        """Vectorstore and retriever are tied to one provider, so both must go."""
        from aorta.chat.rag import retriever

        monkeypatch.setattr(retriever, "_vectorstore_cache", object())
        monkeypatch.setattr(retriever, "_retriever_cache", object())
        retriever.reset_caches()
        assert retriever._vectorstore_cache is None
        assert retriever._retriever_cache is None
