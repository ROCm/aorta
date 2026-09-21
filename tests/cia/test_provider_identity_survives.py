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

from aorta.cia.llm import _litellm_provider_prefix, _qualified_model


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

    def test_the_default_namespaced_vllm_model_does_too(self):
        """A repository namespace is not a provider prefix."""
        model = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

        assert _qualified_model(model, "vllm", True) == f"openai/{model}"

    def test_the_actual_chat_default_stays_routable(self):
        """Cross-contract check against the default that exposed the bug."""
        from aorta.chat.config import Settings

        model = Settings.model_fields["vllm_model"].default

        assert _qualified_model(model, "vllm", True) == f"openai/{model}"

    def test_the_configured_route_wins_over_a_vendor_shaped_name(self):
        """A vLLM endpoint does not become Anthropic because the name says so."""
        model = "anthropic/locally-served-claude-compatible-model"

        assert _qualified_model(model, "vllm", True) == f"openai/{model}"

    def test_a_proxy_does_too(self):
        """What this deployment runs: a LiteLLM proxy addressed as OpenAI."""
        assert _qualified_model("claude-haiku-4-5", "vllm", True) == ("openai/claude-haiku-4-5")

    def test_the_openai_provider_does(self):
        assert _qualified_model("gpt-4o-mini", "openai", False) == "openai/gpt-4o-mini"

    def test_litellm_with_an_endpoint_does(self):
        """An endpoint means something OpenAI-shaped is answering."""
        assert _qualified_model("claude-haiku-4-5", "litellm", True) == ("openai/claude-haiku-4-5")

    def test_an_existing_openai_prefix_is_not_doubled(self):
        assert _qualified_model("openai/gpt-4o", "vllm", True) == "openai/gpt-4o"


class TestASlashIsNotEnoughToNameAProvider:
    def test_a_hugging_face_organization_is_not_a_provider(self):
        assert _litellm_provider_prefix("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct") is None

    @pytest.mark.parametrize(
        "model",
        [
            "openai/gpt-4o",
            "anthropic/claude-3-5-sonnet",
            "gemini/gemini-1.5-pro",
            "bedrock/anthropic.claude-v2",
            "deepseek/deepseek-chat",
        ],
    )
    def test_real_litellm_prefixes_are_recognized(self, model):
        assert _litellm_provider_prefix(model) == model.split("/", 1)[0]

    def test_a_bare_model_has_no_prefix(self):
        assert _litellm_provider_prefix("qwen3-35b") is None


class TestLitellmWithoutAnEndpointRoutesItself:
    def test_an_unqualified_name_is_left_to_litellm(self):
        """litellm has its own rules for a bare name; ours would override them."""
        assert _qualified_model("claude-3-5-sonnet", "litellm", False) == ("claude-3-5-sonnet")


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
        lm = llm_mod.build_lm()

        assert lm.model == "anthropic/claude-3-5-sonnet", lm.model

    def test_a_proxy_profile_still_gets_openai(self, monkeypatch):
        import aorta.cia.llm as llm_mod

        monkeypatch.setattr(
            llm_mod,
            "chat_provider",
            lambda **_k: ("http://proxy:4000/v1", "sk-x", "claude-haiku-4-5", "vllm"),
        )
        lm = llm_mod.build_lm()

        assert lm.model == "openai/claude-haiku-4-5", lm.model

    def test_the_default_vllm_model_reaches_dspy_through_openai(self, monkeypatch):
        """End to end: the model with a slash is still sent to api_base."""
        import aorta.cia.llm as llm_mod
        from aorta.chat.config import Settings

        model = Settings.model_fields["vllm_model"].default
        monkeypatch.setattr(
            llm_mod,
            "chat_provider",
            lambda **_k: ("http://vllm:8000/v1", "EMPTY", model, "vllm"),
        )
        lm = llm_mod.build_lm()

        assert lm.model == f"openai/{model}", lm.model
        assert lm.kwargs["api_base"] == "http://vllm:8000/v1"
