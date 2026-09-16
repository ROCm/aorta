"""The provider chosen for chat is the provider the agents use.

``chat_provider`` handed back a base URL, a key and a model, and ``build_lm``
rebuilt all of it as ``openai/<model>``. That is right for vLLM and for a
proxy -- both speak the OpenAI protocol whatever sits behind them -- and wrong
for a litellm profile addressing a vendor directly, because those model names
carry their own vendor.

So ``aorta chat config init --profile anthropic``, which sets
``llm_provider = "litellm"``, gave Watch and Autopsy a model called
``openai/anthropic/claude-3-5-sonnet``: a Claude model routed through the
OpenAI backend. The failure is at the far end of a network call, which is a
long way from the configuration that caused it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dspy", reason="the agents need the [cia] extra")

from aorta.cia.llm import _qualified_model


class TestAVendorQualifiedModelIsLeftAlone:
    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-3-5-sonnet",
            "gemini/gemini-1.5-pro",
            "openai/gpt-4o",
            "bedrock/anthropic.claude-v2",
        ],
    )
    def test_it_is_passed_through(self, model):
        assert _qualified_model(model, "litellm", False) == model

    def test_it_is_not_prefixed_twice(self):
        """The bug, in one line."""
        out = _qualified_model("anthropic/claude-3-5-sonnet", "litellm", False)

        assert not out.startswith("openai/anthropic/")


class TestAnOpenAIShapedEndpointStillGetsThePrefix:
    def test_vllm_speaks_the_openai_protocol(self):
        assert _qualified_model("qwen3-35b", "vllm", True) == "openai/qwen3-35b"

    def test_a_proxy_does_too(self):
        """What this deployment runs: a LiteLLM proxy addressed as OpenAI."""
        assert _qualified_model("claude-haiku-4-5", "vllm", True) == (
            "openai/claude-haiku-4-5"
        )

    def test_the_openai_provider_does(self):
        assert _qualified_model("gpt-4o-mini", "openai", False) == "openai/gpt-4o-mini"

    def test_litellm_with_an_endpoint_does(self):
        """An endpoint means something OpenAI-shaped is answering."""
        assert _qualified_model("claude-haiku-4-5", "litellm", True) == (
            "openai/claude-haiku-4-5"
        )


class TestLitellmWithoutAnEndpointRoutesItself:
    def test_an_unqualified_name_is_left_to_litellm(self):
        """litellm has its own rules for a bare name; ours would override them."""
        assert _qualified_model("claude-3-5-sonnet", "litellm", False) == (
            "claude-3-5-sonnet"
        )


class TestTheProviderReachesTheDecision:
    def test_chat_provider_reports_it(self, monkeypatch):
        import aorta.cia.llm as llm_mod

        monkeypatch.setenv("AORTA_CHAT_LLM_PROVIDER", "litellm")
        monkeypatch.setenv("AORTA_CHAT_REMOTE_LLM_MODEL", "anthropic/claude-3-5-sonnet")
        from aorta.chat.config import reset_settings

        reset_settings()
        try:
            resolved = llm_mod.chat_provider()
            assert resolved is not None
            assert resolved[-1] == "litellm", resolved
        finally:
            reset_settings()

    def test_the_legacy_variables_name_a_proxy(self, monkeypatch):
        """LITELLM_API_BASE is an endpoint, whatever it routes to."""
        import aorta.cia.llm as llm_mod

        monkeypatch.setenv("LITELLM_API_BASE", "http://proxy:4000/v1")
        resolved = llm_mod._legacy_env()

        assert resolved is not None and resolved[-1] == "openai"


class TestTheLmIsBuiltWithIt:
    def test_a_litellm_profile_is_not_rebuilt_as_openai(self, monkeypatch):
        """End to end through build_lm, not just the mapping helper."""
        import aorta.cia.llm as llm_mod

        monkeypatch.setattr(
            llm_mod,
            "chat_provider",
            lambda **_k: ("", "sk-x", "anthropic/claude-3-5-sonnet", "litellm"),
        )
        monkeypatch.setattr(llm_mod, "_use_certifi_bundle", lambda: None)

        lm = llm_mod.build_lm()

        assert lm.model == "anthropic/claude-3-5-sonnet", lm.model

    def test_a_proxy_profile_still_gets_openai(self, monkeypatch):
        import aorta.cia.llm as llm_mod

        monkeypatch.setattr(
            llm_mod,
            "chat_provider",
            lambda **_k: ("http://proxy:4000/v1", "sk-x", "claude-haiku-4-5", "vllm"),
        )
        monkeypatch.setattr(llm_mod, "_use_certifi_bundle", lambda: None)

        lm = llm_mod.build_lm()

        assert lm.model == "openai/claude-haiku-4-5", lm.model
