"""Prompt profiles: the default request is untouched, ``rl-episode`` is frozen.

Three things are protected, and each has a failure that would be silent:

* **``default`` sends what it always sent.** Adding a profile must not move a
  byte of the shipped prompt or add a field to the shipped request; a general
  model's behaviour is measured against exactly that request.
* **``rl-episode`` sends what the checkpoint was trained on.** The text and the
  user-message layout are pinned against the trainer's own rendering. A drift
  here does not fail anything at runtime -- the replies still parse -- it just
  quietly takes the trained gain away.
* **A profile is never guessed.** An unknown name fails, and so does a profile
  the chosen backend would ignore.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import aorta.agent.loop as loop_mod
import aorta.cli.agent_mitigate as mitigate_cli
from aorta.agent.llm import (
    PROBE_CATEGORIES,
    AgentStep,
    ChatProviderProposer,
    FakeLLMProposer,
    LiteLLMProposer,
    _build_prompt,
    _profile_prompt,
    _step_from_content,
    make_proposer,
)
from aorta.agent.loop import AgentConfig, AgentLoopResult, run_agent_loop
from aorta.agent.policy import AgentPolicy
from aorta.agent.prompt_profiles import (
    DEFAULT_PROMPT_PROFILE,
    PROMPT_PROFILES,
    RL_EPISODE_PROMPT_PROFILE,
    RL_EPISODE_SYSTEM,
    RL_EPISODE_SYSTEM_SHA256,
    build_rl_episode_prompt,
    get_prompt_profile,
)
from aorta.agent.state import AgentState
from aorta.cli.agent_mitigate import mitigate

#: The digest of the system prompt the probe policy was post-trained on,
#: written out here rather than read from the module, so that editing the
#: prompt *and* its digest together still fails.
TRAINED_SYSTEM_SHA256 = "72f18e5541a8a6bbc7ef3653a6d9b7243ba41422610f7aa1766fc6e7f83421c6"

#: One loop state, chosen to exercise every rule of the trained layout: a warn
#: detector to union, detectors out of order, a ``None`` capture and exit
#: code, and names already tried.
SYMPTOM = "loss went to nan"
SUMMARIES = [
    {
        "cell_name": "none-none",
        "verdict": "fail",
        "failure_detectors_fired": ["tier4:nan_signature", "tier1:exit_nonzero"],
        "warn_detectors_fired": ["tier3:slow_step"],
        "capture": {"step_time_ms": [12.5]},
        "exit_code": 1,
    },
    {
        "cell_name": "tf32_off-none",
        "verdict": "fail",
        "failure_detectors_fired": ["tier1:exit_nonzero"],
        "warn_detectors_fired": [],
        "capture": {},
        "exit_code": 1,
    },
    {
        "cell_name": "xnack-none",
        "verdict": "error",
        "failure_detectors_fired": [],
        "warn_detectors_fired": [],
        "capture": None,
        "exit_code": None,
    },
]
TRIED = ["tf32_off", "xnack"]
REMAINING = ["gpu_max_hw_queues_2", "hip_launch_blocking", "pytorch_no_cuda_memory_caching"]
CANDIDATES = ["none", *TRIED, *REMAINING]

#: ``SUMMARIES`` / ``REMAINING`` as the training environment's own
#: ``user_message`` renders them -- produced by the trainer's code, not by the
#: function under test.
TRAINED_USER = (
    '{"candidates": ["gpu_max_hw_queues_2", "hip_launch_blocking", '
    '"pytorch_no_cuda_memory_caching"], "cell_summaries": [{"capture": '
    '{"step_time_ms": [12.5]}, "cell_name": "none-none", "exit_code": 1, '
    '"failure_detectors_fired": ["tier1:exit_nonzero", "tier3:slow_step", '
    '"tier4:nan_signature"], "verdict": "fail"}, {"capture": {}, "cell_name": '
    '"tf32_off-none", "exit_code": 1, "failure_detectors_fired": '
    '["tier1:exit_nonzero"], "verdict": "fail"}, {"capture": {}, "cell_name": '
    '"xnack-none", "exit_code": null, "failure_detectors_fired": [], "verdict": '
    '"error"}]}'
)
TRAINED_USER_SHA256 = "b8edb7e047ddb841c32d103be72a1d98cf5bcb886b2394786acbf46896466de8"

#: A reply in the shape the policy was trained to produce: seven keys, two of
#: which (``verdict``, ``detectors``) the loop does not read.
TRAINED_REPLY = json.dumps(
    {
        "verdict": "fail",
        "detectors": ["tier1:exit_nonzero", "tier4:nan_signature"],
        "category": "unknown",
        "hypothesis": "NaN from a stale workspace buffer.",
        "next_mitigations": ["pytorch_no_cuda_memory_caching", "tf32_off", "not_registered"],
        "confidence": 0.8,
        "stop": False,
    }
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _propose(proposer, *, summaries=None, candidates=None, tried=None):
    return proposer.propose(
        symptom=SYMPTOM,
        cell_summaries=copy.deepcopy(SUMMARIES if summaries is None else summaries),
        candidates=list(CANDIDATES if candidates is None else candidates),
        tried=list(TRIED if tried is None else tried),
    )


class RecordingChatModel:
    """The LangChain chat model, reduced to what the proposer calls."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[tuple[list, dict]] = []

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(content=self.content)


@pytest.fixture()
def chat_model(monkeypatch):
    model = RecordingChatModel(TRAINED_REPLY)
    monkeypatch.setattr(ChatProviderProposer, "_chat_model", lambda self: model)
    return model


@pytest.fixture()
def litellm_calls(monkeypatch):
    """A stand-in ``litellm`` module recording every ``completion`` call."""
    calls: list[dict] = []

    def completion(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content=TRAINED_REPLY)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    return calls


# ── default: unchanged ─────────────────────────────────────────────────────


class TestDefaultIsUnchanged:
    def test_the_default_profile_is_the_default_everywhere(self):
        assert DEFAULT_PROMPT_PROFILE == "default"
        assert AgentConfig(
            output_dir=Path("."), ticket=None, subprocess_argv=()
        ).prompt_profile == ("default")
        assert ChatProviderProposer("vllm")._profile.name == "default"
        assert LiteLLMProposer()._profile.name == "default"

    def test_default_messages_are_the_shipped_prompt(self):
        profile = get_prompt_profile("default")
        got = _profile_prompt(profile, SYMPTOM, copy.deepcopy(SUMMARIES), REMAINING, TRIED)
        assert got == _build_prompt(SYMPTOM, copy.deepcopy(SUMMARIES), REMAINING, TRIED)

    def test_the_chat_request_carries_nothing_new(self, chat_model):
        """One positional argument and no keywords: the call as it was shipped."""
        _propose(ChatProviderProposer("vllm"))
        ((messages, kwargs),) = chat_model.calls
        assert kwargs == {}
        system, user = _build_prompt(SYMPTOM, copy.deepcopy(SUMMARIES), REMAINING, TRIED)
        assert messages == [("system", system), ("human", user)]

    def test_the_litellm_request_carries_nothing_new(self, litellm_calls):
        _propose(LiteLLMProposer(model="some-model"))
        (call,) = litellm_calls
        system, user = _build_prompt(SYMPTOM, copy.deepcopy(SUMMARIES), REMAINING, TRIED)
        assert call == {
            "model": "some-model",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }

    def test_a_default_run_logs_no_profile(self, tmp_path, monkeypatch):
        """Logs written before profiles existed and after must read the same."""
        _run_loop(tmp_path, monkeypatch, backend="fake", profile="default")
        (session,) = _log_events(tmp_path / "out", "session_start")
        assert "prompt_profile" not in session


# ── rl-episode: frozen as trained ──────────────────────────────────────────


class TestRlEpisodeIsFrozen:
    def test_the_system_prompt_is_the_one_trained_on(self):
        assert _sha256(RL_EPISODE_SYSTEM) == TRAINED_SYSTEM_SHA256
        assert RL_EPISODE_SYSTEM_SHA256 == TRAINED_SYSTEM_SHA256
        assert len(RL_EPISODE_SYSTEM.encode("utf-8")) == 930

    def test_the_user_message_is_laid_out_as_trained(self):
        system, user = build_rl_episode_prompt(copy.deepcopy(SUMMARIES), list(REMAINING))
        assert system == RL_EPISODE_SYSTEM
        assert user == TRAINED_USER
        assert _sha256(user) == TRAINED_USER_SHA256

    def test_the_symptom_and_tried_list_are_not_sent(self):
        """The policy never saw either; what was tried is on screen as cells."""
        _, user = _profile_prompt(
            get_prompt_profile("rl-episode"), SYMPTOM, copy.deepcopy(SUMMARIES), REMAINING, TRIED
        )
        doc = json.loads(user)
        assert set(doc) == {"cell_summaries", "candidates"}
        assert SYMPTOM not in user
        assert doc["candidates"] == REMAINING

    def test_every_category_it_offers_is_one_the_loop_accepts(self):
        """Written out as trained, so check it still agrees with the loop.

        A category the prompt offers and ``validate_step`` refuses would end
        the search on a policy violation the first time the model used it.
        """
        match = re.search(r"`category` must be one of (\[[^\]]*\])", RL_EPISODE_SYSTEM)
        assert match is not None
        offered = ast.literal_eval(match.group(1))
        assert offered == sorted(PROBE_CATEGORIES)
        policy = AgentPolicy()
        for category in offered:
            step = AgentStep(category, "", [], 0.5, False)
            assert policy.validate_step(step).category == category

    def test_a_trained_reply_parses_with_the_shared_parser(self):
        """The two extra keys are ignored and the menu filter still applies."""
        step = _step_from_content(TRAINED_REPLY, REMAINING)
        assert step.next_mitigations == ["pytorch_no_cuda_memory_caching"]
        assert step.category == "unknown"
        assert step.stop is False
        assert AgentPolicy().validate_step(step).next_mitigations == [
            "pytorch_no_cuda_memory_caching"
        ]


class TestRlEpisodeRequest:
    def test_the_chat_request_sends_the_trained_prompt_with_thinking_off(self, chat_model):
        step = _propose(ChatProviderProposer("vllm", prompt_profile="rl-episode"))
        ((messages, kwargs),) = chat_model.calls
        assert messages == [("system", RL_EPISODE_SYSTEM), ("human", TRAINED_USER)]
        assert kwargs == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        assert step.next_mitigations == ["pytorch_no_cuda_memory_caching"]

    def test_the_litellm_request_drops_json_mode(self, litellm_calls):
        """Training decoded without a grammar; JSON mode is not what it learned."""
        _propose(LiteLLMProposer(model="m", prompt_profile="rl-episode"))
        (call,) = litellm_calls
        assert "response_format" not in call
        assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
        assert call["messages"] == [
            {"role": "system", "content": RL_EPISODE_SYSTEM},
            {"role": "user", "content": TRAINED_USER},
        ]

    def test_extra_body_is_a_fresh_dict(self):
        profile = get_prompt_profile("rl-episode")
        first = profile.extra_body()
        first["chat_template_kwargs"]["enable_thinking"] = True
        assert profile.extra_body() == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_an_exhausted_menu_still_spends_no_tokens(self, chat_model):
        step = _propose(
            ChatProviderProposer("vllm", prompt_profile="rl-episode"),
            candidates=["none", "tf32_off"],
            tried=["tf32_off"],
        )
        assert step.stop_reason == "exhausted_candidates"
        assert chat_model.calls == []


# ── fail closed ────────────────────────────────────────────────────────────


class TestUnknownProfile:
    @pytest.mark.parametrize("name", ["rl_episode", "RL-EPISODE", "", "training"])
    def test_an_unknown_name_is_refused_and_the_valid_ones_named(self, name):
        with pytest.raises(ValueError) as exc:
            get_prompt_profile(name)
        message = str(exc.value)
        assert repr(name) in message
        for valid in PROMPT_PROFILES:
            assert valid in message

    @pytest.mark.parametrize("backend", ["vllm", "openai", "litellm", "fake"])
    def test_every_backend_refuses_an_unknown_profile(self, backend):
        with pytest.raises(ValueError, match="unknown agent prompt profile"):
            make_proposer(backend, prompt_profile="rl_episode")

    def test_the_proposers_refuse_it_directly_too(self):
        with pytest.raises(ValueError, match="unknown agent prompt profile"):
            ChatProviderProposer("vllm", prompt_profile="nope")
        with pytest.raises(ValueError, match="unknown agent prompt profile"):
            LiteLLMProposer(prompt_profile="nope")

    def test_the_fake_backend_refuses_a_profile_it_would_ignore(self):
        with pytest.raises(ValueError, match="needs a real model"):
            make_proposer("fake", prompt_profile="rl-episode")
        assert isinstance(make_proposer("fake", prompt_profile="default"), FakeLLMProposer)

    def test_the_cli_refuses_an_unknown_profile(self, monkeypatch, tmp_path):
        loop = MagicMock()
        monkeypatch.setattr(mitigate_cli, "run_agent_loop", loop)
        result = CliRunner().invoke(
            mitigate, ["--output", str(tmp_path), "--prompt-profile", "rl_episode", "--", "true"]
        )
        assert result.exit_code == 2
        assert "rl_episode" in result.output
        loop.assert_not_called()


# ── the profile flows from the CLI to the request ──────────────────────────


class TestCliToRequest:
    def test_the_click_choice_matches_the_registry(self):
        """Hard-coded so `aorta --help` stays cheap; this fails if they drift."""
        option = next(p for p in mitigate.params if p.name == "prompt_profile")
        assert set(option.type.choices) == set(PROMPT_PROFILES)
        assert option.default == DEFAULT_PROMPT_PROFILE

    @pytest.mark.parametrize("backend", ["vllm", "openai", "litellm"])
    @pytest.mark.parametrize("chat_layer", [True, False])
    def test_the_factory_hands_the_profile_to_every_real_proposer(
        self, monkeypatch, backend, chat_layer
    ):
        """Including the direct LiteLLM path an ``[agent]``-only install takes."""
        monkeypatch.setattr("aorta.agent.llm._chat_layer_available", lambda: chat_layer)
        proposer = make_proposer(backend, prompt_profile=RL_EPISODE_PROMPT_PROFILE)
        assert proposer._profile.name == RL_EPISODE_PROMPT_PROFILE

    @pytest.mark.parametrize(
        "argv, expected", [([], "default"), (["--prompt-profile", "rl-episode"], "rl-episode")]
    )
    def test_the_flag_reaches_the_config(self, monkeypatch, tmp_path, argv, expected):
        loop = MagicMock(
            return_value=AgentLoopResult(
                run_dir=tmp_path,
                state=AgentState(ticket="T1"),
                report_path=None,
                outcome="converged",
                recommended_action="done",
            )
        )
        monkeypatch.setattr(mitigate_cli, "run_agent_loop", loop)
        result = CliRunner().invoke(
            mitigate, ["--output", str(tmp_path), "--llm-backend", "vllm", *argv, "--", "true"]
        )
        assert result.exit_code == 0, result.output
        (config,), _ = loop.call_args
        assert config.prompt_profile == expected

    def test_the_cli_sends_the_trained_prompt_end_to_end(self, monkeypatch, tmp_path, chat_model):
        """Real CLI, real loop, real factory and proposer; only I/O is stubbed."""
        _stub_cells(monkeypatch, tmp_path)
        result = CliRunner().invoke(
            mitigate,
            [
                "--output",
                str(tmp_path / "out"),
                "--ticket",
                "T1",
                "--llm-backend",
                "vllm",
                "--prompt-profile",
                "rl-episode",
                "--",
                "true",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "converged" in result.output
        ((messages, kwargs),) = chat_model.calls
        assert messages[0] == ("system", RL_EPISODE_SYSTEM)
        doc = json.loads(messages[1][1])
        assert [row["cell_name"] for row in doc["cell_summaries"]] == ["none-none"]
        assert "tf32_off" in doc["candidates"] and "none" not in doc["candidates"]
        assert kwargs == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        (session,) = _log_events(tmp_path / "out", "session_start")
        assert session["prompt_profile"] == "rl-episode"


# ── helpers ────────────────────────────────────────────────────────────────


def _stub_cells(monkeypatch, tmp_path):
    """Baseline fails; the cell for the proposal's first name passes."""
    monkeypatch.setattr(loop_mod, "run_recipe", MagicMock(return_value=tmp_path / "out" / "T1"))
    baseline = {
        "cell_name": "none-none",
        "verdict": "fail",
        "failure_detectors_fired": ["tier1:exit_nonzero", "tier4:nan_signature"],
        "warn_detectors_fired": [],
        "capture": {},
        "exit_code": 1,
    }
    fixed = {
        **baseline,
        "cell_name": "pytorch_no_cuda_memory_caching-none",
        "verdict": "pass",
        "failure_detectors_fired": [],
        "exit_code": 0,
    }
    seq = iter([[baseline], [baseline, fixed]])
    monkeypatch.setattr(
        loop_mod, "_read_cell_summaries", lambda run_dir: next(seq, [baseline, fixed])
    )


def _run_loop(tmp_path, monkeypatch, *, backend, profile):
    _stub_cells(monkeypatch, tmp_path)
    config = AgentConfig(
        output_dir=tmp_path / "out",
        ticket="T1",
        subprocess_argv=("true",),
        policy=AgentPolicy(max_iterations=3),
        llm_backend=backend,
        prompt_profile=profile,
    )
    return run_agent_loop(config)


def _log_events(output_dir: Path, event: str, *, ticket: str = "T1") -> list[dict]:
    path = output_dir / ticket / "agent_log.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [row for row in rows if row["type"] == event]
