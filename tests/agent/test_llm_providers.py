"""Phase 5b: the agent proposer on the shared chat provider layer (Decision 7a).

Two things are being protected here, and they pull in opposite directions.

The new behaviour: ``vllm`` / ``openai`` / ``litellm`` resolve through the chat
provider factory, so an endpoint, gateway header or auth scheme is configured
once and both front doors read it.

The shipped contract: ``--llm-backend`` values that worked before must keep
working, ``fake`` must stay the default, and ``fake`` must stay *fully offline*
-- it is what the whole test suite and ``--dry-run`` depend on.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
from types import SimpleNamespace

import pytest

from aorta.agent.llm import (
    CHAT_PROVIDER_BACKENDS,
    ChatProviderProposer,
    FakeLLMProposer,
    LiteLLMProposer,
    make_proposer,
)

CANDIDATES = ["none", "tf32_off", "hsa_xnack"]
TRIED: list[str] = []

#: Most of this file stubs the chat model out and so runs anywhere. Two groups
#: cannot: they assert what happens *because* the chat extra is present, and
#: reaching them without it raises out of ``aorta.chat.config`` (pydantic on
#: 3.11+, stdlib ``tomllib`` on 3.10) rather than testing anything.
#:
#: The predicate is the one production uses -- ``llm._chat_layer_available``
#: asks exactly this of exactly this package -- so the tests are gated on the
#: condition they describe rather than on a proxy for it. chat-tests.yml runs
#: this file on the legs that do install the extra, so the gated cases are
#: covered in CI rather than skipped everywhere.
_CHAT_AVAILABLE = importlib.util.find_spec("langchain_core") is not None


def _propose(proposer, candidates=None, tried=None):
    return proposer.propose(
        symptom="loss went to nan",
        cell_summaries=[{"cell_name": "none-none", "verdict": "fail"}],
        candidates=CANDIDATES if candidates is None else candidates,
        tried=TRIED if tried is None else tried,
    )


class FakeChatModel:
    """Stands in for the LangChain chat model the provider layer returns."""

    def __init__(self, content):
        self.content = content
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.content)


@pytest.fixture()
def chat_model(monkeypatch):
    """Install a fake chat model behind ChatProviderProposer._chat_model."""

    def _install(content):
        model = FakeChatModel(content)
        monkeypatch.setattr(ChatProviderProposer, "_chat_model", lambda self: model)
        return model

    return _install


# ── the factory ───────────────────────────────────────────────────────────


class TestMakeProposer:
    def test_fake_is_the_default_and_is_the_fake(self):
        assert isinstance(make_proposer("fake"), FakeLLMProposer)

    @pytest.mark.parametrize("backend", ["vllm", "openai"])
    def test_the_new_backends_use_the_shared_provider_layer(self, backend):
        proposer = make_proposer(backend)
        assert isinstance(proposer, ChatProviderProposer)
        assert proposer._provider == backend

    @pytest.mark.skipif(
        not _CHAT_AVAILABLE,
        reason="chat extra absent, so the fallback below is the correct path here",
    )
    def test_litellm_prefers_the_shared_layer_when_chat_is_installed(self):
        """Decision 7a: configured once, not twice."""
        assert isinstance(make_proposer("litellm"), ChatProviderProposer)

    def test_litellm_falls_back_to_the_direct_path_without_the_chat_extra(self, monkeypatch):
        """It has worked on an [agent]-only install since before aorta.chat existed.

        Breaking that to satisfy a new decision would be a regression for a
        shipped flag, so the fallback is deliberate rather than defensive.
        """
        monkeypatch.setattr("aorta.agent.llm._chat_layer_available", lambda: False)
        proposer = make_proposer("litellm")
        assert isinstance(proposer, LiteLLMProposer)
        assert proposer._model == "gpt-4o-mini"

    def test_the_fallback_still_honours_an_explicit_model(self, monkeypatch):
        monkeypatch.setattr("aorta.agent.llm._chat_layer_available", lambda: False)
        assert make_proposer("litellm", model="claude-sonnet-4-5")._model == "claude-sonnet-4-5"

    def test_no_model_means_no_opinion_on_the_shared_path(self):
        """The chat profile decides, which is the point of a single provider layer."""
        assert make_proposer("openai")._model is None

    def test_an_unknown_backend_names_the_valid_ones(self):
        with pytest.raises(ValueError) as exc:
            make_proposer("gpt5")
        message = str(exc.value)
        assert "gpt5" in message
        for name in ("fake", *CHAT_PROVIDER_BACKENDS):
            assert name in message


class TestOfflineDefault:
    def test_the_fake_proposer_imports_nothing(self, monkeypatch):
        """``fake`` must work on a base install with no network and no extras."""

        def _explode(name, *args, **kwargs):
            raise AssertionError(f"the fake path imported {name!r}")

        proposer = make_proposer("fake")
        monkeypatch.setattr(builtins, "__import__", _explode)
        step = _propose(proposer)
        assert step.next_mitigations == ["tf32_off"]

    def test_an_exhausted_search_needs_no_backend_at_all(self):
        """Checked before the import, so no extra and no tokens are needed."""
        proposer = ChatProviderProposer("openai")
        step = _propose(proposer, candidates=["none", "tf32_off"], tried=["tf32_off"])
        assert step.stop is True
        assert step.stop_reason == "exhausted_candidates"


# ── the shared-layer proposer ──────────────────────────────────────────────


class TestChatProviderProposer:
    def test_a_well_formed_reply_becomes_an_agent_step(self, chat_model):
        chat_model(
            json.dumps(
                {
                    "category": "illegal_mem",
                    "hypothesis": "tf32 rounding",
                    "next_mitigations": ["tf32_off"],
                    "confidence": 0.7,
                    "stop": False,
                }
            )
        )
        step = _propose(ChatProviderProposer("openai"))
        assert step.category == "illegal_mem"
        assert step.next_mitigations == ["tf32_off"]
        assert step.confidence == 0.7
        assert step.stop is False

    def test_the_prompt_carries_a_system_and_a_user_turn(self, chat_model):
        model = chat_model('{"next_mitigations": ["tf32_off"]}')
        _propose(ChatProviderProposer("vllm"))
        roles = [role for role, _ in model.calls[0]]
        assert roles == ["system", "human"]
        system = model.calls[0][0][1]
        assert "ONLY registered mitigation" in system
        assert "illegal_mem" in system

    def test_a_fenced_json_block_still_parses(self, chat_model):
        """There is no response_format on this path, so fences are expected."""
        chat_model('```json\n{"next_mitigations": ["tf32_off"], "confidence": 0.4}\n```')
        step = _propose(ChatProviderProposer("openai"))
        assert step.next_mitigations == ["tf32_off"]
        assert step.confidence == 0.4

    def test_a_bare_fence_without_a_language_also_parses(self, chat_model):
        chat_model('```\n{"next_mitigations": ["hsa_xnack"]}\n```')
        assert _propose(ChatProviderProposer("openai")).next_mitigations == ["hsa_xnack"]

    @pytest.mark.parametrize("reply", ["", "   ", None])
    def test_an_empty_reply_stops_safely(self, chat_model, reply):
        chat_model(reply)
        step = _propose(ChatProviderProposer("openai"))
        assert step.stop is True
        assert step.stop_reason == "agent_requested"
        assert step.next_mitigations == []

    @pytest.mark.parametrize("reply", ["not json at all", "[1, 2, 3]", '{"broken": '])
    def test_an_unparseable_reply_stops_safely_rather_than_raising(self, chat_model, reply):
        """The loop must still write a report; a traceback here loses the run."""
        chat_model(reply)
        step = _propose(ChatProviderProposer("openai"))
        assert step.stop is True
        assert "unparseable" in step.hypothesis or "Empty" in step.hypothesis

    def test_a_mitigation_outside_the_candidate_list_is_dropped(self, chat_model):
        """The model does not get to widen its own allowlist."""
        chat_model(json.dumps({"next_mitigations": ["tf32_off", "rm -rf /", "unregistered"]}))
        assert _propose(ChatProviderProposer("openai")).next_mitigations == ["tf32_off"]

    def test_an_already_tried_mitigation_is_dropped(self, chat_model):
        chat_model(json.dumps({"next_mitigations": ["tf32_off", "hsa_xnack"]}))
        step = _propose(ChatProviderProposer("openai"), tried=["tf32_off"])
        assert step.next_mitigations == ["hsa_xnack"]

    def test_a_stop_without_a_reason_is_attributed_to_the_agent(self, chat_model):
        chat_model(json.dumps({"stop": True, "next_mitigations": []}))
        step = _propose(ChatProviderProposer("openai"))
        assert step.stop is True
        assert step.stop_reason == "agent_requested"

    def test_a_string_false_does_not_stop_the_loop(self, chat_model):
        """``bool("false")`` is True, so only a real JSON boolean may stop it."""
        chat_model(json.dumps({"stop": "false", "next_mitigations": ["tf32_off"]}))
        assert _propose(ChatProviderProposer("openai")).stop is False


class TestMissingExtra:
    def test_a_missing_chat_dependency_names_the_extra(self, monkeypatch):
        real_import = builtins.__import__

        def _fake(name, *args, **kwargs):
            if name.startswith("aorta.chat"):
                raise ModuleNotFoundError("No module named 'langchain_core'", name="langchain_core")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake)
        with pytest.raises(ImportError) as exc:
            _propose(ChatProviderProposer("openai"))
        message = str(exc.value)
        assert "amd-aorta[chat-cli]" in message
        assert "langchain_core" in message
        assert "configured once" in message

    def test_a_broken_chat_submodule_surfaces_instead_of_advising_an_install(self, monkeypatch):
        """Telling someone to install what they already have buries a real bug.

        Same external-versus-internal distinction as ``cli/chat.py::_load``.
        """
        real_import = builtins.__import__

        def _fake(name, *args, **kwargs):
            if name.startswith("aorta.chat"):
                raise ModuleNotFoundError(
                    "No module named 'aorta.chat.inference'", name="aorta.chat.inference"
                )
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake)
        with pytest.raises(ModuleNotFoundError) as exc:
            _propose(ChatProviderProposer("openai"))
        assert exc.value.name == "aorta.chat.inference"


@pytest.mark.skipif(not _CHAT_AVAILABLE, reason="amd-aorta[chat-cli] not installed")
class TestRealProviderResolution:
    """No mock of ``_chat_model`` here -- this is the wiring itself.

    Every other test in this file stubs the chat model out, which would let a
    broken seam pass: the point of Decision 7a is that the agent reads the
    *chat* configuration, and only an unmocked resolution proves it does.

    Because nothing is mocked, this is the one group that needs the extra for
    real. chat-tests.yml runs this file so that it does.
    """

    @pytest.fixture(autouse=True)
    def _isolated(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        monkeypatch.setenv("AORTA_CHAT_REMOTE_LLM_API_KEY", "sk-test-not-real")
        from aorta.chat.config import reset_settings
        from aorta.chat.inference.providers.factory import reset_backend_cache

        reset_settings()
        reset_backend_cache()
        yield
        reset_settings()
        reset_backend_cache()

    def test_a_model_override_lands_on_the_local_field_for_vllm(self):
        from aorta.chat.config import get_settings

        make_proposer("vllm", model="some/local-model")._chat_model()
        settings = get_settings()
        assert settings.llm_provider == "vllm"
        assert settings.vllm_model == "some/local-model"

    def test_a_model_override_lands_on_the_remote_field_for_openai(self):
        """Which field a model name belongs to depends on the resolved provider."""
        from aorta.chat.config import get_settings

        make_proposer("openai", model="gpt-4.1-mini")._chat_model()
        settings = get_settings()
        assert settings.llm_provider == "openai"
        assert settings.remote_llm_model == "gpt-4.1-mini"
        assert settings.vllm_model != "gpt-4.1-mini"

    def test_the_layer_returns_a_usable_chat_model(self):
        model = make_proposer("openai", model="gpt-4o-mini")._chat_model()
        assert hasattr(model, "invoke")

    def test_the_chat_layers_own_config_errors_reach_the_agent(self, monkeypatch):
        """A missing key must be reported by the provider layer, not guessed at here."""
        monkeypatch.delenv("AORTA_CHAT_REMOTE_LLM_API_KEY")
        from aorta.chat.config import reset_settings

        reset_settings()
        with pytest.raises(ValueError, match="API_KEY"):
            make_proposer("openai")._chat_model()


class TestCliSurface:
    def test_the_llm_backend_choice_matches_the_registry(self):
        """The Click choice is hard-coded so `aorta --help` stays cheap.

        This is the test that comment promises: it fails if the two drift.
        """
        from aorta.cli.agent_mitigate import mitigate

        option = next(p for p in mitigate.params if p.name == "llm_backend")
        assert set(option.type.choices) == {"fake", *CHAT_PROVIDER_BACKENDS}

    def test_fake_remains_the_click_default(self):
        from aorta.cli.agent_mitigate import mitigate

        option = next(p for p in mitigate.params if p.name == "llm_backend")
        assert option.default == "fake"

    def test_llm_model_defaults_to_no_opinion(self):
        """A hardcoded gpt-4o-mini would be the wrong model for --llm-backend=vllm."""
        from aorta.cli.agent_mitigate import mitigate

        option = next(p for p in mitigate.params if p.name == "llm_model")
        assert option.default is None
