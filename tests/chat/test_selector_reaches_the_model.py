"""The selector's ranking reaches the model, not just the screen.

``candidate_tools`` was written by ``selector_node``, declared in state,
asserted in tests and displayed in the UI — and read by no node. So the ranking
this PR is named for cost a model call per action turn and changed nothing about
which tool ran: the act paths still received the whole tool list and chose
unaided.

It is advice, and says so in the prompt. Every registered tool stays bound, so a
bad ranking widens the choice rather than narrowing it — which is what makes it
safe to act on one model's opinion of another's options.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import aorta.chat.graph.nodes as nodes
from aorta.chat.graph.nodes import _act_messages, _recommendation


def _state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage(content="here is a kernel that gives wrong results")],
        "retrieved_context": "",
        "plan": "",
        "critic_feedback": "",
        "candidate_tools": ["triage_kernel_source", "search_code"],
        "selection_rationale": "the user pasted a HIP kernel with a shared-memory race",
    }
    return {**base, **overrides}


def _framing(state: dict) -> str:
    return "\n".join(str(m.content) for m in _act_messages(state))


class TestItReachesTheActPaths:
    def test_the_ranking_is_in_the_framing_both_protocols_share(self):
        assert "triage_kernel_source" in _framing(_state())

    def test_the_order_is_preserved(self):
        """Checked on the ranking line: the base prompt names tools too."""
        line = next(
            l for l in _recommendation(_state()).splitlines() if "Most likely tools" in l
        )
        assert line.index("triage_kernel_source") < line.index("search_code")

    def test_the_reason_travels_with_it(self):
        """A ranking without its reason is a number the model cannot weigh."""
        assert "shared-memory race" in _framing(_state())

    def test_it_is_offered_as_advice(self):
        assert "not a restriction" in _framing(_state())


class TestItReachesThePlan:
    def test_the_plan_prompt_carries_it(self, monkeypatch):
        sent: dict = {}

        class FakeLM:
            async def ainvoke(self, messages):
                sent["text"] = "\n".join(str(m.content) for m in messages)
                return AIMessage(content="a plan")

        monkeypatch.setattr(nodes, "_get_llm", lambda **kw: FakeLM())
        monkeypatch.setattr(nodes, "load_repo_map", lambda: "(repo map)")

        asyncio.run(nodes.plan_node(_state()))

        assert "triage_kernel_source" in sent["text"]
        assert "not a restriction" in sent["text"]


class TestABadOrAbsentRankingCostsNothing:
    def test_an_empty_ranking_adds_nothing(self):
        """The selector declining leaves the model where it started."""
        assert _recommendation(_state(candidate_tools=[])) == ""

    def test_a_state_with_no_selector_output_adds_nothing(self):
        assert _recommendation({}) == ""

    def test_the_framing_is_unchanged_without_a_ranking(self):
        with_none = _act_messages(_state(candidate_tools=[], selection_rationale=""))
        assert not any("Most likely tools" in str(m.content) for m in with_none)

    def test_a_ranking_never_removes_a_tool_from_the_prompt(self):
        """The full catalogue is still described whatever the selector said."""
        narrow = _framing(_state(candidate_tools=["search_code"], selection_rationale="r"))
        assert "triage_kernel_source" in nodes.TOOL_DESCRIPTIONS
        assert "search_code" in narrow

    def test_empty_strings_in_the_ranking_are_dropped(self):
        assert _recommendation(_state(candidate_tools=["", ""])) == ""


def test_the_rationale_is_optional():
    """A ranking with no reason is still worth passing on."""
    text = _recommendation(_state(selection_rationale=""))
    assert "triage_kernel_source" in text
    assert "Why:" not in text
