"""LiteLLM must not phone home while the tests run.

It downloads its model-price table from GitHub the first time it is asked
about a model it does not recognise, and every model name in this suite is
made up. The download happens once per process, so under xdist it lands in
whichever test gets there first on each worker. When that was a test holding
the no_network fixture it failed, while the identical call in the next class
passed on another worker -- five failures in CI that reproduce nowhere else,
because a developer's machine has usually done the fetch at import time during
collection and serves every later lookup from memory.

The conftest sets LITELLM_LOCAL_MODEL_COST_MAP so the bundled copy is used
instead. These tests hold that fixed from the outside: they assert the
behaviour, not the environment variable, so they still mean something if
LiteLLM changes how it decides.
"""

from __future__ import annotations

import os
import socket

import pytest

pytest.importorskip("litellm", reason="the gateway tests need the chat-all extra")

import litellm


@pytest.fixture()
def connection_attempts(monkeypatch):
    """Record every outbound connect instead of making it."""
    attempts: list[object] = []

    def _record(self, address, *args, **kwargs):
        attempts.append(address)
        raise OSError("refused by test")

    monkeypatch.setattr(socket.socket, "connect", _record)
    return attempts


class TestTheCostMapIsLocal:
    def test_the_environment_asks_for_the_bundled_copy(self):
        """Set in conftest, before litellm is imported anywhere."""
        assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "True"

    def test_an_unknown_model_does_not_reach_for_the_network(
        self, connection_attempts
    ):
        """The case that broke CI: a model name that is not in the table."""
        with pytest.raises(Exception):
            litellm.get_model_info("a-model-that-does-not-exist")

        assert connection_attempts == []

    def test_a_known_model_does_not_either(self, connection_attempts):
        litellm.get_model_info("gpt-4o")

        assert connection_attempts == []


class TestTheBackendStaysOffline:
    def test_building_a_chat_model_makes_no_connection(
        self, connection_attempts, monkeypatch
    ):
        """Constructing a backend inspects config; it should not call out."""
        from aorta.chat.config import get_settings
        from aorta.chat.inference.providers.remote_litellm import RemoteLiteLLMBackend

        settings = get_settings()
        monkeypatch.setattr(settings, "remote_llm_model", "claude-example", raising=False)
        monkeypatch.setattr(settings, "remote_llm_base_url", "", raising=False)
        monkeypatch.setattr(settings, "remote_llm_api_key", "", raising=False)

        RemoteLiteLLMBackend().get_chat_model(streaming=False)

        assert connection_attempts == []
