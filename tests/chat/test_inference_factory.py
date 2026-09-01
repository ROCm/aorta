"""Provider selection for the chat layer: factory, backends, and the facade.

No test here contacts a provider. The remote backends are configured with
placeholder keys, model construction is inert, and the two preflight tests run
under the ``no_network`` fixture so any connection attempt raises instead of
leaving the suite dependent on a live endpoint.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from unittest.mock import patch

import pytest

from aorta.chat.config import settings
from aorta.chat.inference import chat, vllm_client
from aorta.chat.inference.callcount import LLMCallCounter, count_llm_calls
from aorta.chat.inference.providers import local_vllm
from aorta.chat.inference.providers.base import ChatBackend
from aorta.chat.inference.providers.factory import (
    available_providers,
    get_backend,
    reset_backend_cache,
)
from aorta.chat.inference.providers.local_vllm import LocalVLLMBackend
from aorta.chat.inference.providers.remote_litellm import (
    LITELLM_IMPORT_MESSAGE,
    RemoteLiteLLMBackend,
    _load_chat_litellm,
)
from aorta.chat.inference.providers.remote_openai import (
    MISSING_API_KEY_MESSAGE,
    RemoteOpenAIBackend,
)
from tests.conftest import NetworkUsed

_LITELLM_INSTALLED = importlib.util.find_spec("litellm") is not None


@pytest.fixture(autouse=True)
def _clean_backend_cache():
    """The factory caches instances process-wide; do not leak them across tests."""
    reset_backend_cache()
    yield
    reset_backend_cache()


@pytest.fixture()
def remote_llm_settings(monkeypatch):
    """Configure the REMOTE block with a placeholder key that is never used."""
    monkeypatch.setattr(settings, "remote_llm_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "remote_llm_api_key", "sk-placeholder-not-real")
    monkeypatch.setattr(settings, "remote_llm_base_url", "")
    monkeypatch.setattr(settings, "remote_llm_auth_header", "")
    monkeypatch.setattr(settings, "remote_llm_extra_headers", {})
    monkeypatch.setattr(settings, "llm_max_tokens", None)
    monkeypatch.setattr(settings, "llm_timeout", 120.0)
    monkeypatch.setattr(settings, "llm_max_retries", 2)


class TestProviderSelection:
    def test_available_providers_are_the_three_documented_ones(self):
        assert available_providers() == ("litellm", "openai", "vllm")

    def test_each_name_resolves_to_its_own_backend(self):
        assert isinstance(get_backend("vllm"), LocalVLLMBackend)
        assert isinstance(get_backend("openai"), RemoteOpenAIBackend)
        assert isinstance(get_backend("litellm"), RemoteLiteLLMBackend)

    def test_backend_name_matches_the_provider_string(self):
        for provider in available_providers():
            assert get_backend(provider).name == provider

    def test_default_follows_the_llm_provider_setting(self, monkeypatch):
        for provider, expected in (
            ("vllm", LocalVLLMBackend),
            ("openai", RemoteOpenAIBackend),
            ("litellm", RemoteLiteLLMBackend),
        ):
            monkeypatch.setattr(settings, "llm_provider", provider)
            reset_backend_cache()
            assert isinstance(get_backend(), expected)

    def test_provider_string_is_case_and_whitespace_tolerant(self):
        """An LLM_PROVIDER copied out of a doc keeps working."""
        assert isinstance(get_backend("  VLLM  "), LocalVLLMBackend)
        assert isinstance(get_backend("OpenAI"), RemoteOpenAIBackend)

    def test_unknown_provider_names_the_valid_choices(self):
        with pytest.raises(ValueError) as exc:
            get_backend("gemini")
        assert str(exc.value) == (
            "unknown LLM provider: 'gemini' "
            "(expected one of litellm, openai, vllm)"
        )

    def test_unknown_provider_in_settings_also_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "gemini")
        with pytest.raises(ValueError, match="unknown LLM provider: 'gemini'"):
            get_backend()

    def test_backends_are_cached_until_reset(self):
        first = get_backend("vllm")
        assert get_backend("vllm") is first
        reset_backend_cache()
        assert get_backend("vllm") is not first

    def test_every_backend_satisfies_the_protocol(self):
        for provider in available_providers():
            assert isinstance(get_backend(provider), ChatBackend)


class TestChatFacade:
    def test_get_chat_llm_builds_the_local_model_by_default(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "vllm")
        llm = chat.get_chat_llm(temperature=0.5, streaming=False)
        assert llm.model_name == settings.vllm_model
        assert str(llm.openai_api_base) == settings.vllm_base_url
        assert llm.temperature == 0.5
        assert llm.streaming is False

    def test_get_chat_llm_follows_the_provider_setting(
        self, monkeypatch, remote_llm_settings
    ):
        monkeypatch.setattr(settings, "llm_provider", "openai")
        assert chat.get_chat_llm().model_name == "gpt-4o-mini"

    def test_vllm_client_still_exports_the_pre_provider_names(self):
        """The shim keeps old imports working after the facade refactor."""
        assert vllm_client.get_chat_llm is chat.get_chat_llm
        assert (
            vllm_client.get_async_openai_client is local_vllm.get_async_openai_client
        )
        assert vllm_client.stream_chat is local_vllm.stream_chat

    def test_nodes_llm_entry_point_forwards_to_the_facade(self):
        """``src.graph.nodes._get_llm`` is the mock point the graph tests patch."""
        from aorta.chat.graph import nodes

        assert nodes.get_chat_llm is chat.get_chat_llm
        sentinel = object()
        with patch.object(nodes, "get_chat_llm", return_value=sentinel) as fake:
            assert nodes._get_llm(temperature=0.0, streaming=False) is sentinel
        fake.assert_called_once_with(temperature=0.0, streaming=False)


class TestRemoteOpenAIBackend:
    def test_missing_key_raises_before_a_model_is_built(self, monkeypatch):
        monkeypatch.setattr(settings, "remote_llm_api_key", "")
        with pytest.raises(ValueError) as exc:
            RemoteOpenAIBackend().get_chat_model()
        assert str(exc.value) == MISSING_API_KEY_MESSAGE
        assert "REMOTE_LLM_API_KEY" in str(exc.value)

    @pytest.mark.asyncio
    async def test_missing_key_fails_preflight(self, monkeypatch):
        monkeypatch.setattr(settings, "remote_llm_api_key", "")
        with pytest.raises(ValueError) as exc:
            await RemoteOpenAIBackend().preflight()
        assert str(exc.value) == MISSING_API_KEY_MESSAGE

    def test_whitespace_only_key_counts_as_missing(self, monkeypatch):
        """Matches the embeddings provider: a blank key fails here, not at the
        provider with an opaque 401."""
        monkeypatch.setattr(settings, "remote_llm_api_key", "   ")
        with pytest.raises(ValueError, match="REMOTE_LLM_API_KEY"):
            RemoteOpenAIBackend().get_chat_model()

    def test_blank_base_url_is_omitted_rather_than_passed_through(
        self, monkeypatch, remote_llm_settings, no_network
    ):
        monkeypatch.setattr(settings, "remote_llm_base_url", "   ")
        assert RemoteOpenAIBackend().get_chat_model().openai_api_base is None

    @pytest.mark.asyncio
    async def test_preflight_makes_no_network_call(
        self, remote_llm_settings, no_network
    ):
        """Validating config must cost nothing against a metered endpoint."""
        await RemoteOpenAIBackend().preflight()

    def test_chat_model_carries_the_call_counter(self, remote_llm_settings):
        llm = RemoteOpenAIBackend().get_chat_model()
        assert any(isinstance(cb, LLMCallCounter) for cb in llm.callbacks)

    def test_describe_names_the_endpoint(self, monkeypatch, remote_llm_settings):
        assert "provider default endpoint" in RemoteOpenAIBackend().describe()
        monkeypatch.setattr(
            settings, "remote_llm_base_url", "https://openrouter.ai/api/v1"
        )
        assert "https://openrouter.ai/api/v1" in RemoteOpenAIBackend().describe()


class TestLocalVLLMBackend:
    @pytest.mark.asyncio
    async def test_preflight_does_reach_for_the_network(self, no_network):
        """Counterpart to the remote no-network test: the guard really bites."""
        with pytest.raises(NetworkUsed):
            await LocalVLLMBackend().preflight(timeout=1, interval=1)

    def test_chat_model_is_unchanged_by_the_provider_split(self):
        """The local flow attaches no counter: those calls are not metered."""
        llm = LocalVLLMBackend().get_chat_model()
        assert not llm.callbacks
        assert llm.model_name == settings.vllm_model


class TestRemoteLiteLLMBackend:
    def test_missing_litellm_raises_an_actionable_import_error(self):
        """A ``None`` entry in sys.modules makes ``import litellm`` fail cleanly.

        Faking the absence this way keeps the test meaningful whether or not
        litellm is installed in the environment running it.
        """
        with patch.dict(sys.modules, {"litellm": None}):
            with pytest.raises(ImportError) as exc:
                _load_chat_litellm()
        assert str(exc.value) == LITELLM_IMPORT_MESSAGE
        assert "pip install litellm" in str(exc.value)
        assert isinstance(exc.value.__cause__, ImportError)

    @pytest.mark.asyncio
    async def test_preflight_surfaces_the_missing_extra(self):
        with patch.dict(sys.modules, {"litellm": None}):
            with pytest.raises(ImportError) as exc:
                await RemoteLiteLLMBackend().preflight()
        assert str(exc.value) == LITELLM_IMPORT_MESSAGE

    def test_get_chat_model_surfaces_the_missing_extra(self):
        with patch.dict(sys.modules, {"litellm": None}):
            with pytest.raises(ImportError) as exc:
                RemoteLiteLLMBackend().get_chat_model()
        assert str(exc.value) == LITELLM_IMPORT_MESSAGE

    @pytest.mark.skipif(
        not _LITELLM_INSTALLED, reason="the litellm extra is not installed"
    )
    def test_chat_litellm_is_importable_when_the_extra_is_present(self):
        """Guards the langchain-community<0.4 pin.

        0.4 deleted ``ChatLiteLLM``, which would make ``_load_chat_litellm``
        raise LITELLM_IMPORT_MESSAGE even with litellm correctly installed --
        an import error blaming the wrong package.
        """
        assert _load_chat_litellm().__name__ == "ChatLiteLLM"

    def test_describe_names_the_model(self, remote_llm_settings):
        assert "gpt-4o-mini" in RemoteLiteLLMBackend().describe()


class TestCallCounting:
    def test_counter_logs_the_calls_made_inside_the_block(self, caplog):
        counter = LLMCallCounter()
        with caplog.at_level(logging.INFO, logger="aorta.chat.inference.callcount"):
            with count_llm_calls("query"):
                counter.on_chat_model_start({}, [])
                counter.on_chat_model_start({}, [])
        assert "Remote LLM calls for this query: 2" in caplog.text

    def test_nothing_is_logged_when_no_call_is_made(self, caplog):
        with caplog.at_level(logging.INFO, logger="aorta.chat.inference.callcount"):
            with count_llm_calls("query"):
                pass
        assert "Remote LLM calls" not in caplog.text
