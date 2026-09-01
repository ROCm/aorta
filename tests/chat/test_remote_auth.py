"""Custom-header authentication for the two REMOTE flows.

Covers the header assembly itself, the ``.env`` parsing that feeds it, and the
two client constructions that consume it. Nothing here contacts a provider: the
clients are built and inspected, never invoked, and the gateway test runs under
``no_network`` so a stray connection would fail loudly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aorta.chat.config import Settings, settings
from aorta.chat.inference.providers.remote_openai import RemoteOpenAIBackend
from aorta.chat.rag.embeddings.remote_api import RemoteApiProvider
from aorta.chat.remote_auth import PLACEHOLDER_API_KEY, build_auth, describe_auth

#: The shape AMD's internal gateway uses: an Azure API Management front end.
APIM_HEADER = "Ocp-Apim-Subscription-Key"


class TestBuildAuth:
    def test_without_an_auth_header_the_key_stays_a_bearer_token(self):
        key, headers = build_auth(api_key="sk-real")
        assert key == "sk-real"
        assert headers is None

    def test_an_auth_header_moves_the_key_out_of_the_bearer_slot(self):
        key, headers = build_auth(api_key="sk-real", auth_header=APIM_HEADER)
        assert key == PLACEHOLDER_API_KEY
        assert headers == {APIM_HEADER: "sk-real"}

    def test_extra_headers_travel_alongside_the_key(self):
        key, headers = build_auth(
            api_key="sk-real",
            auth_header=APIM_HEADER,
            extra_headers={"user": "alice"},
        )
        assert key == PLACEHOLDER_API_KEY
        assert headers == {APIM_HEADER: "sk-real", "user": "alice"}

    def test_extra_headers_work_without_an_auth_header(self):
        key, headers = build_auth(api_key="sk-real", extra_headers={"user": "alice"})
        assert key == "sk-real"
        assert headers == {"user": "alice"}

    def test_a_whitespace_only_header_name_means_bearer(self):
        key, headers = build_auth(api_key="sk-real", auth_header="   ")
        assert key == "sk-real"
        assert headers is None

    def test_blank_extras_are_dropped_rather_than_sent_empty(self):
        _key, headers = build_auth(
            api_key="sk-real",
            extra_headers={"user": "", "": "orphan", "x-tenant": "amd"},
        )
        assert headers == {"x-tenant": "amd"}

    def test_the_caller_s_extra_headers_dict_is_not_mutated(self):
        extras = {"user": "alice"}
        build_auth(api_key="sk-real", auth_header=APIM_HEADER, extra_headers=extras)
        assert extras == {"user": "alice"}


class TestDescribeAuth:
    def test_bearer_is_the_default_description(self):
        assert describe_auth() == "bearer token"

    def test_a_custom_header_is_named(self):
        assert describe_auth(auth_header=APIM_HEADER) == f"{APIM_HEADER} header"

    def test_extra_header_names_are_listed(self):
        described = describe_auth(
            auth_header=APIM_HEADER, extra_headers={"user": "alice"}
        )
        assert described == f"{APIM_HEADER} header, plus user"

    def test_the_secret_never_appears_in_the_description(self):
        described = describe_auth(
            auth_header=APIM_HEADER,
            extra_headers={APIM_HEADER: "sk-super-secret", "user": "alice"},
        )
        assert "sk-super-secret" not in described


def _load_settings() -> Settings:
    """Build ``Settings`` from the process environment only.

    ``_env_file=None`` is load-bearing: without it these tests read whatever
    ``.env`` the developer happens to have, so a machine configured for a real
    gateway would fail on the default-value assertions.
    """
    return Settings(_env_file=None)


class TestExtraHeaderParsing:
    """``REMOTE_*_EXTRA_HEADERS`` comes off the environment as a string."""

    def test_comma_separated_pairs(self, monkeypatch):
        monkeypatch.setenv("REMOTE_LLM_EXTRA_HEADERS", "user=alice,x-tenant=amd")
        assert _load_settings().remote_llm_extra_headers == {
            "user": "alice",
            "x-tenant": "amd",
        }

    def test_json_object(self, monkeypatch):
        monkeypatch.setenv("REMOTE_LLM_EXTRA_HEADERS", '{"user": "alice"}')
        assert _load_settings().remote_llm_extra_headers == {"user": "alice"}

    def test_json_is_the_escape_hatch_for_a_value_with_a_comma(self, monkeypatch):
        monkeypatch.setenv("REMOTE_LLM_EXTRA_HEADERS", '{"x-tags": "a,b"}')
        assert _load_settings().remote_llm_extra_headers == {"x-tags": "a,b"}

    def test_only_the_first_equals_splits_so_padded_values_survive(self, monkeypatch):
        monkeypatch.setenv("REMOTE_LLM_EXTRA_HEADERS", "x-token=abc==")
        assert _load_settings().remote_llm_extra_headers == {"x-token": "abc=="}

    def test_surrounding_whitespace_is_ignored(self, monkeypatch):
        monkeypatch.setenv("REMOTE_LLM_EXTRA_HEADERS", " user = alice , ")
        assert _load_settings().remote_llm_extra_headers == {"user": "alice"}

    def test_empty_means_no_headers(self, monkeypatch):
        monkeypatch.setenv("REMOTE_LLM_EXTRA_HEADERS", "")
        assert _load_settings().remote_llm_extra_headers == {}

    def test_default_is_no_headers(self, monkeypatch):
        monkeypatch.delenv("REMOTE_LLM_EXTRA_HEADERS", raising=False)
        assert _load_settings().remote_llm_extra_headers == {}

    def test_a_pair_without_equals_is_rejected(self, monkeypatch):
        monkeypatch.setenv("REMOTE_LLM_EXTRA_HEADERS", "user")
        with pytest.raises(ValidationError, match="missing '='"):
            _load_settings()

    def test_malformed_json_is_rejected_rather_than_comma_split(self, monkeypatch):
        monkeypatch.setenv("REMOTE_LLM_EXTRA_HEADERS", '{"user": alice}')
        with pytest.raises(ValidationError, match="not valid JSON"):
            _load_settings()

    def test_the_embedding_field_parses_the_same_way(self, monkeypatch):
        monkeypatch.setenv("REMOTE_EMBEDDING_EXTRA_HEADERS", "user=alice")
        assert _load_settings().remote_embedding_extra_headers == {"user": "alice"}

    def test_the_two_fields_are_independent(self, monkeypatch):
        monkeypatch.setenv("REMOTE_LLM_EXTRA_HEADERS", "user=alice")
        monkeypatch.delenv("REMOTE_EMBEDDING_EXTRA_HEADERS", raising=False)
        assert _load_settings().remote_embedding_extra_headers == {}


@pytest.fixture()
def amd_gateway(monkeypatch):
    """Configure both remote flows the way AMD's internal gateway needs."""
    monkeypatch.setattr(settings, "remote_llm_model", "GPT-oss-20B")
    monkeypatch.setattr(settings, "remote_llm_api_key", "sk-super-secret")
    monkeypatch.setattr(
        settings, "remote_llm_base_url", "https://gateway.example.com/v1"
    )
    monkeypatch.setattr(settings, "remote_llm_auth_header", APIM_HEADER)
    monkeypatch.setattr(settings, "remote_llm_extra_headers", {"user": "alice"})
    monkeypatch.setattr(settings, "llm_max_tokens", None)
    monkeypatch.setattr(settings, "llm_timeout", 120.0)
    monkeypatch.setattr(settings, "llm_max_retries", 2)
    monkeypatch.setattr(settings, "remote_embedding_api_key", "sk-super-secret")
    monkeypatch.setattr(
        settings, "remote_embedding_base_url", "https://gateway.example.com/v1"
    )
    monkeypatch.setattr(settings, "remote_embedding_auth_header", APIM_HEADER)
    monkeypatch.setattr(settings, "remote_embedding_extra_headers", {"user": "alice"})


class TestChatBackendWiring:
    def test_the_gateway_headers_reach_the_client(self, amd_gateway, no_network):
        llm = RemoteOpenAIBackend().get_chat_model(streaming=False)
        assert llm.default_headers == {
            APIM_HEADER: "sk-super-secret",
            "user": "alice",
        }

    def test_the_bearer_slot_holds_only_the_placeholder(self, amd_gateway, no_network):
        llm = RemoteOpenAIBackend().get_chat_model(streaming=False)
        assert llm.openai_api_key.get_secret_value() == PLACEHOLDER_API_KEY

    def test_a_plain_bearer_provider_sends_no_extra_headers(
        self, monkeypatch, no_network
    ):
        monkeypatch.setattr(settings, "remote_llm_api_key", "sk-real")
        monkeypatch.setattr(settings, "remote_llm_auth_header", "")
        monkeypatch.setattr(settings, "remote_llm_extra_headers", {})
        llm = RemoteOpenAIBackend().get_chat_model(streaming=False)
        assert llm.default_headers is None
        assert llm.openai_api_key.get_secret_value() == "sk-real"

    def test_describe_names_the_header_without_leaking_the_key(self, amd_gateway):
        described = RemoteOpenAIBackend().describe()
        assert APIM_HEADER in described
        assert "sk-super-secret" not in described

    @pytest.mark.asyncio
    async def test_preflight_accepts_the_gateway_config_without_a_call(
        self, amd_gateway, no_network
    ):
        await RemoteOpenAIBackend().preflight()

    @pytest.mark.asyncio
    async def test_preflight_logs_the_header_name_not_the_key(
        self, amd_gateway, no_network, caplog
    ):
        with caplog.at_level("INFO"):
            await RemoteOpenAIBackend().preflight()
        logged = caplog.text
        assert APIM_HEADER in logged
        assert "sk-super-secret" not in logged


class TestEmbeddingProviderWiring:
    def test_the_gateway_headers_reach_the_client(self, amd_gateway, no_network):
        embeddings = RemoteApiProvider().get_embeddings()
        assert embeddings.default_headers == {
            APIM_HEADER: "sk-super-secret",
            "user": "alice",
        }

    def test_the_bearer_slot_holds_only_the_placeholder(self, amd_gateway, no_network):
        embeddings = RemoteApiProvider().get_embeddings()
        assert embeddings.openai_api_key.get_secret_value() == PLACEHOLDER_API_KEY

    def test_describe_names_the_header_without_leaking_the_key(self, amd_gateway):
        described = RemoteApiProvider().describe()
        assert APIM_HEADER in described
        assert "sk-super-secret" not in described

    def test_the_collection_name_ignores_the_auth_style(self, amd_gateway):
        """Auth is transport; it must not move where vectors are stored."""
        assert RemoteApiProvider().collection_name() == (
            "aorta_remote_text_embedding_3_small"
        )
