"""The chat settings still reach the agents on Python 3.10.

``aorta.chat.config`` reads ``chat.toml`` with stdlib ``tomllib``, which is
3.11, and this package supports 3.10. There the module cannot be imported at
all — so a 3.10 user who had exported ``AORTA_CHAT_VLLM_BASE_URL`` correctly got
``chat_provider() -> None`` and then ``ProviderNotConfigured``, with the only
trace a ``DEBUG`` line nobody sees.

That message is worse than unhelpful. It is wrong, and it sends the reader to
configure something they had already configured.

The environment is honoured directly when the settings module is missing —
same variables, same precedence, one thing lost: the profile file, which is
said out loud rather than left to be discovered.
"""

from __future__ import annotations

import builtins
import logging

import pytest

from aorta.cia import llm as llm_mod

CHAT_ENV = [
    "AORTA_CHAT_LLM_PROVIDER",
    "AORTA_CHAT_VLLM_BASE_URL",
    "AORTA_CHAT_VLLM_API_KEY",
    "AORTA_CHAT_VLLM_MODEL",
    "AORTA_CHAT_REMOTE_LLM_BASE_URL",
    "AORTA_CHAT_REMOTE_LLM_API_KEY",
    "AORTA_CHAT_REMOTE_LLM_MODEL",
]


@pytest.fixture
def as_310(monkeypatch):
    """An interpreter where `import tomllib` fails, as 3.10's does."""
    for var in CHAT_ENV + ["LITELLM_API_BASE", "LITELLM_API_KEY", "LITELLM_MODEL"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm_mod, "_warned_no_settings", False)

    real_import = builtins.__import__

    def no_tomllib(name, *args, **kwargs):
        if name == "tomllib" or name == "aorta.chat.config":
            raise ImportError("No module named 'tomllib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tomllib)


class TestTheEnvironmentIsStillRead:
    def test_a_configured_vllm_endpoint_survives(self, as_310, monkeypatch):
        monkeypatch.setenv("AORTA_CHAT_VLLM_BASE_URL", "http://configured:4000/v1")
        monkeypatch.setenv("AORTA_CHAT_VLLM_MODEL", "qwen3-35b")
        assert llm_mod.chat_provider() == ("http://configured:4000/v1", "", "qwen3-35b")

    def test_a_remote_provider_is_read_from_its_own_fields(self, as_310, monkeypatch):
        monkeypatch.setenv("AORTA_CHAT_LLM_PROVIDER", "openai")
        monkeypatch.setenv("AORTA_CHAT_REMOTE_LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("AORTA_CHAT_REMOTE_LLM_API_KEY", "sk-remote")
        assert llm_mod.chat_provider() == ("", "sk-remote", "gpt-4o-mini")

    def test_nothing_set_still_means_nothing(self, as_310):
        """An empty environment is not a configuration."""
        assert llm_mod.chat_provider() is None

    def test_build_lm_reaches_the_configured_endpoint(self, as_310, monkeypatch):
        monkeypatch.setenv("AORTA_CHAT_VLLM_BASE_URL", "http://configured:4000/v1")
        built: dict = {}
        monkeypatch.setattr(llm_mod, "RedactingLM", lambda **kw: built.update(kw) or object())

        llm_mod.build_lm()

        assert built["api_base"] == "http://configured:4000/v1"


class TestItSaysWhatItLost:
    def test_the_warning_names_the_cause(self, as_310, monkeypatch, caplog):
        monkeypatch.setenv("AORTA_CHAT_VLLM_BASE_URL", "http://configured:4000/v1")
        with caplog.at_level(logging.WARNING, logger="aorta.cia.llm"):
            llm_mod.chat_provider()

        assert caplog.records, "falling back must not be silent"
        message = caplog.records[0].getMessage()
        assert "tomllib" in message
        assert "AORTA_CHAT_" in message

    def test_it_says_the_profile_file_is_what_is_missing(self, as_310, caplog):
        with caplog.at_level(logging.WARNING, logger="aorta.cia.llm"):
            llm_mod.chat_provider()
        assert "profile file" in caplog.records[0].getMessage()

    def test_it_warns_once_not_per_poll(self, as_310, caplog):
        """Watch calls this on every round; a per-call warning is noise."""
        with caplog.at_level(logging.WARNING, logger="aorta.cia.llm"):
            for _ in range(5):
                llm_mod.chat_provider()
        assert len(caplog.records) == 1

    def test_the_unconfigured_error_mentions_the_limitation(self, as_310):
        with pytest.raises(llm_mod.ProviderNotConfigured, match="3.10"):
            llm_mod.build_lm()


def test_the_import_failure_is_caught_narrowly():
    """A broad except here would swallow a real bug in the settings module."""
    import inspect

    source = inspect.getsource(llm_mod.chat_provider)
    assert "except ImportError" in source
    assert "except Exception" not in source
