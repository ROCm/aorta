"""aorta#510: the JSON-mode request bounds how many names one reply may carry.

Under ``response_format={"type": "json_object"}`` a grammar-constrained server
let about one reply in nine name nearly the whole candidate menu. The mitigation
axis is cumulative and the iteration budget charges once per proposal cycle, so
such a reply consumed the whole axis in one charged iteration and the search
ended there. The request now carries a schema with ``maxItems`` instead, and
nothing else about the budget or the other request paths moves.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import aorta.agent.llm as llm_mod
import aorta.agent.loop as loop_mod
from aorta.agent.llm import ChatProviderProposer, LiteLLMProposer, _build_prompt
from aorta.agent.loop import AgentConfig, run_agent_loop
from aorta.agent.policy import AgentPolicy
from aorta.registry import load_mitigations

#: Written out rather than read from the module, so that loosening the cap and
#: the test together still fails.
CAP = 5

SUMMARIES = [
    {
        "cell_name": "none-none",
        "verdict": "fail",
        "failure_detectors_fired": ["tier4:nan_signature"],
        "warn_detectors_fired": [],
        "capture": {},
        "exit_code": 1,
    }
]

#: Twelve registered names, so a shotgun is wider than the cap and the cap
#: still takes more than two steps to walk the menu.
MENU = sorted(n for n in load_mitigations() if n != "none")[:12]


def _reply(names: list[str]) -> str:
    return json.dumps(
        {
            "category": "unknown",
            "hypothesis": "try everything",
            "next_mitigations": names,
            "confidence": 0.5,
            "stop": False,
        }
    )


def _schema_cap(response_format: dict | None) -> int | None:
    """What a grammar-constrained server would bound ``next_mitigations`` to."""
    if not response_format or response_format.get("type") != "json_schema":
        return None
    schema = response_format["json_schema"]["schema"]
    return schema["properties"]["next_mitigations"].get("maxItems")


@pytest.fixture()
def grammar_server(monkeypatch):
    """A stand-in ``litellm`` whose model names every candidate it is offered.

    It enforces the request's schema the way constrained decoding does: the
    array closes at ``maxItems``. Under ``json_object`` there is no bound, which
    is the tail aorta#510 measured.
    """
    calls: list[dict] = []

    def completion(**kwargs):
        calls.append(kwargs)
        offered = json.loads(kwargs["messages"][1]["content"])["candidates"]
        cap = _schema_cap(kwargs.get("response_format"))
        names = offered if cap is None else offered[:cap]
        message = SimpleNamespace(content=_reply(names))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    return calls


def _propose(proposer, candidates=None):
    return proposer.propose(
        symptom="loss went to nan",
        cell_summaries=copy.deepcopy(SUMMARIES),
        candidates=list(MENU if candidates is None else candidates),
        tried=[],
    )


class TestTheRequest:
    def test_the_json_mode_request_caps_next_mitigations(self, grammar_server):
        _propose(LiteLLMProposer(model="m"))
        (call,) = grammar_server
        assert _schema_cap(call["response_format"]) == CAP
        assert llm_mod.MAX_PROPOSED_MITIGATIONS == CAP

    def test_the_schema_asks_for_the_keys_the_prompt_asks_for_in_order(self, grammar_server):
        """A strict schema emits keys in schema order and admits no others."""
        _propose(LiteLLMProposer(model="m"))
        (call,) = grammar_server
        json_schema = call["response_format"]["json_schema"]
        schema = json_schema["schema"]
        system, _ = _build_prompt(None, [], [], [])
        match = re.search(r"keys: (.*?)\. ", system)
        assert match is not None
        asked = [re.sub(r" \(.*\)$", "", key) for key in match.group(1).split(", ")]
        assert list(schema["properties"]) == asked
        assert schema["required"] == asked
        assert schema["additionalProperties"] is False
        assert json_schema["strict"] is True

    def test_only_the_width_is_constrained(self, grammar_server):
        """An unregistered name or category must still reach the filter and the policy."""
        _propose(LiteLLMProposer(model="m"))
        (call,) = grammar_server
        properties = call["response_format"]["json_schema"]["schema"]["properties"]
        assert properties["next_mitigations"] == {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": CAP,
        }
        assert properties["category"] == {"type": "string"}

    def test_each_request_gets_its_own_schema(self, grammar_server):
        proposer = LiteLLMProposer(model="m")
        _propose(proposer)
        grammar_server[0]["response_format"]["json_schema"]["schema"]["properties"][
            "next_mitigations"
        ]["maxItems"] = 99
        _propose(proposer)
        assert _schema_cap(grammar_server[1]["response_format"]) == CAP

    def test_rl_episode_still_sends_no_response_format(self, grammar_server):
        _propose(LiteLLMProposer(model="m", prompt_profile="rl-episode"))
        (call,) = grammar_server
        assert "response_format" not in call

    def test_the_chat_path_still_sends_no_response_format(self, monkeypatch):
        model = MagicMock()
        model.invoke.return_value = SimpleNamespace(content=_reply(MENU[:2]))
        monkeypatch.setattr(ChatProviderProposer, "_chat_model", lambda self: model)
        _propose(ChatProviderProposer("vllm"))
        ((_, kwargs),) = [(c.args, c.kwargs) for c in model.invoke.call_args_list]
        assert kwargs == {}


class TestTheReply:
    def test_a_server_that_ignores_max_items_is_read_as_before(self, monkeypatch):
        """The bound is the server's; the agent does not truncate behind it."""
        message = SimpleNamespace(content=_reply(MENU))
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        monkeypatch.setitem(
            sys.modules, "litellm", SimpleNamespace(completion=lambda **kw: response)
        )
        step = _propose(LiteLLMProposer(model="m"))
        assert step.next_mitigations == MENU
        assert step.unresolved_mitigations == []


class TestTheSearch:
    def test_one_reply_no_longer_takes_the_whole_axis(self, tmp_path, monkeypatch, grammar_server):
        """Same budget semantics; the search now has more than one decision in it."""
        monkeypatch.setattr(llm_mod, "_chat_layer_available", lambda: False)
        monkeypatch.setattr(
            loop_mod, "run_recipe", MagicMock(return_value=tmp_path / "out" / "T1")
        )
        monkeypatch.setattr(
            loop_mod, "_read_cell_summaries", lambda run_dir: copy.deepcopy(SUMMARIES)
        )
        result = run_agent_loop(
            AgentConfig(
                output_dir=tmp_path / "out",
                ticket="T1",
                subprocess_argv=("true",),
                policy=AgentPolicy(max_iterations=8),
                llm_backend="litellm",
                mitigations_allowlist=tuple(MENU),
            )
        )

        per_iteration: list[int] = []
        added = 0
        for row in _log(tmp_path / "out" / "T1"):
            if row["type"] == "mitigation_tried":
                added += 1
            elif row["type"] == "iteration_complete":
                per_iteration.append(added)
                added = 0

        assert per_iteration == [5, 5, 2]
        assert result.state.iterations_completed == 3
        assert result.state.tried_mitigations == MENU
        assert result.outcome == "exhausted_candidates"


def _log(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "agent_log.jsonl").read_text().splitlines()]
