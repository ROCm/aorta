"""The LiteLLM backend behind a gateway, and the secret it must not print.

All three behaviours here come from getting Claude working through AMD's
Anthropic gateway path, and each was a hard failure or a leak:

* The gateway authenticates with `Ocp-Apim-Subscription-Key`, but this backend
  passed no headers at all, so a gateway was unreachable via LiteLLM.
* Current Claude Opus builds accept only `temperature=1`, and LiteLLM raises
  `UnsupportedParamsError` rather than negotiating. Graph nodes ask for 0.0 and
  0.1, so every call failed until `drop_params` was set.
* LiteLLM's debug logger prints outbound request headers. Under `--verbose` that
  put the subscription key in plaintext into the terminal.
"""

from __future__ import annotations

import importlib.util
import logging

import pytest

from aorta.chat.config import settings
from aorta.chat.inference.providers.remote_litellm import (
    RemoteLiteLLMBackend,
    _load_chat_litellm,
)
from aorta.chat.remote_auth import PLACEHOLDER_API_KEY

_LITELLM_INSTALLED = importlib.util.find_spec("litellm") is not None
_needs_litellm = pytest.mark.skipif(
    not _LITELLM_INSTALLED, reason="requires the 'remote' extra"
)

APIM_HEADER = "Ocp-Apim-Subscription-Key"
SECRET = "sk-not-a-real-subscription-key"


@pytest.fixture()
def gateway(monkeypatch):
    """AMD's Anthropic path: native Anthropic protocol behind APIM."""
    monkeypatch.setattr(settings, "remote_llm_model", "anthropic/Claude-Opus-4.7")
    monkeypatch.setattr(
        settings, "remote_llm_base_url", "https://gateway.example.com/anthropic"
    )
    monkeypatch.setattr(settings, "remote_llm_api_key", SECRET)
    monkeypatch.setattr(settings, "remote_llm_auth_header", APIM_HEADER)
    monkeypatch.setattr(settings, "remote_llm_extra_headers", {})
    monkeypatch.setattr(settings, "llm_max_tokens", None)
    monkeypatch.setattr(settings, "llm_timeout", 120.0)
    monkeypatch.setattr(settings, "llm_max_retries", 2)


@pytest.fixture()
def plain_provider(monkeypatch):
    """A provider reached directly, with LiteLLM reading its own env vars."""
    monkeypatch.setattr(settings, "remote_llm_model", "claude-opus-5")
    monkeypatch.setattr(settings, "remote_llm_base_url", "")
    monkeypatch.setattr(settings, "remote_llm_api_key", "")
    monkeypatch.setattr(settings, "remote_llm_auth_header", "")
    monkeypatch.setattr(settings, "remote_llm_extra_headers", {})
    monkeypatch.setattr(settings, "llm_max_tokens", None)
    monkeypatch.setattr(settings, "llm_timeout", 120.0)
    monkeypatch.setattr(settings, "llm_max_retries", 2)


@_needs_litellm
class TestGatewayHeaders:
    def test_the_key_travels_in_the_gateway_header(self, gateway, no_network):
        llm = RemoteLiteLLMBackend().get_chat_model(streaming=False)
        assert llm.model_kwargs["extra_headers"] == {APIM_HEADER: SECRET}

    def test_the_bearer_slot_holds_only_the_placeholder(self, gateway, no_network):
        llm = RemoteLiteLLMBackend().get_chat_model(streaming=False)
        assert llm.api_key == PLACEHOLDER_API_KEY

    def test_the_base_url_reaches_api_base(self, gateway, no_network):
        llm = RemoteLiteLLMBackend().get_chat_model(streaming=False)
        assert llm.api_base == "https://gateway.example.com/anthropic"

    def test_extra_headers_are_merged(self, gateway, monkeypatch, no_network):
        monkeypatch.setattr(settings, "remote_llm_extra_headers", {"user": "alice"})
        llm = RemoteLiteLLMBackend().get_chat_model(streaming=False)
        assert llm.model_kwargs["extra_headers"] == {
            APIM_HEADER: SECRET,
            "user": "alice",
        }


@_needs_litellm
class TestWithoutAGateway:
    def test_no_headers_are_sent_and_litellm_keeps_its_own_auth(
        self, plain_provider, no_network
    ):
        """Unchanged behaviour: keys come from ANTHROPIC_API_KEY and friends."""
        llm = RemoteLiteLLMBackend().get_chat_model(streaming=False)
        assert "extra_headers" not in (llm.model_kwargs or {})

    def test_an_empty_base_url_is_omitted_not_passed_blank(
        self, plain_provider, no_network
    ):
        llm = RemoteLiteLLMBackend().get_chat_model(streaming=False)
        assert llm.api_base is None


@_needs_litellm
class TestUnsupportedParams:
    def test_drop_params_is_enabled(self):
        """Claude Opus accepts only temperature=1; nodes ask for 0.0 and 0.1."""
        import litellm

        _load_chat_litellm()
        assert litellm.drop_params is True


@_needs_litellm
class TestTheKeyIsNotLogged:
    def test_litellm_debug_logging_is_suppressed(self):
        """Its debug records include outbound headers, key and all."""
        import litellm

        logging.getLogger("LiteLLM").setLevel(logging.DEBUG)
        _load_chat_litellm()
        assert litellm.suppress_debug_info is True
        assert logging.getLogger("LiteLLM").level >= logging.WARNING

    def test_describe_names_the_header_without_the_value(self, gateway):
        described = RemoteLiteLLMBackend().describe()
        assert APIM_HEADER in described
        assert SECRET not in described

    def test_the_cli_pins_the_litellm_logger_under_verbose(self):
        """--verbose sets root DEBUG; LiteLLM must be exempted by name."""
        import inspect

        from aorta.cli import chat as cli

        source = inspect.getsource(cli.main)
        assert 'getLogger("LiteLLM")' in source
