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

    def test_a_ranking_never_removes_a_tool_from_the_prompt(self, cluster_jobs_enabled):
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


# ── both protocols, not just the one that was checked ─────────────────────


class TestEitherToolProtocol:
    """The first fix reached ``_act_native`` and ``plan_node`` only.

    ``_act_text`` built its own message list, so it never saw the ranking --
    and it is the protocol ``llm_tool_mode`` defaults to, which is to say the
    one that runs unless something says otherwise. The ranking reached the
    screen and not the model in exactly the configuration nobody had changed.
    """

    @staticmethod
    def _sent(mode: str, state: dict) -> list[str]:
        """Everything the model is handed in *mode*, as text."""
        from unittest.mock import MagicMock, patch

        captured: list[str] = []

        def fake_llm(*args, **kwargs):
            llm = MagicMock()

            async def ainvoke(messages, *a, **k):
                captured.extend(str(getattr(m, "content", m)) for m in messages)
                return AIMessage(content="FINAL: done")

            llm.ainvoke = ainvoke
            llm.bind_tools = lambda *a, **k: llm
            return llm

        with patch.object(nodes, "_get_llm", fake_llm), patch.object(
            nodes.settings, "llm_tool_mode", mode
        ):
            try:
                asyncio.run(nodes.act_node(dict(state)))
            except Exception:  # noqa: BLE001 - the loop's own exits are not the subject
                pass
        return captured

    @pytest.mark.parametrize("mode", ["native", "text"])
    def test_the_ranking_is_sent(self, mode):
        sent = self._sent(mode, _state())

        assert any("triage_kernel_source" in m for m in sent)
        assert any("Most likely tools" in m for m in sent)

    @pytest.mark.parametrize("mode", ["native", "text"])
    def test_the_reason_goes_with_it(self, mode):
        sent = self._sent(mode, _state())

        assert any("shared-memory race" in m for m in sent)

    @pytest.mark.parametrize("mode", ["native", "text"])
    def test_it_is_still_only_advice(self, mode):
        sent = self._sent(mode, _state())

        assert any("not a restriction" in m for m in sent)

    @pytest.mark.parametrize("mode", ["native", "text"])
    def test_nothing_is_said_when_the_selector_declined(self, mode):
        sent = self._sent(mode, _state(candidate_tools=[], selection_rationale=""))

        assert not any("Most likely tools" in m for m in sent)

    def test_the_text_protocol_still_describes_the_tools(self):
        """It has no tool-calling API, so the prompt is the only place they appear."""
        sent = self._sent("text", _state())

        assert any(nodes.TOOL_DESCRIPTIONS[:60] in m for m in sent)

    def test_and_describes_them_before_ranking_them(self):
        """A ranking of things the reader has not been shown yet reads oddly.

        Measured inside the text rather than across the list: the system
        messages are merged into one before the request goes out.
        """
        whole = "\n".join(self._sent("text", _state()))

        assert whole.index(nodes.TOOL_DESCRIPTIONS[:60]) < whole.index("Most likely tools")

    def test_the_two_protocols_share_one_framing(self):
        """They drifted because there were two copies; this is the fix holding."""
        import inspect

        source = inspect.getsource(nodes._act_text)

        assert "_act_messages(state)" in source
