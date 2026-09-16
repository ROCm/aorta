"""The critic has to see the code it is asked to check claims against.

It was given the tool results and the generated answer, and nothing else. So
"the loss diverges because lr=5.0" was unverifiable by construction: the
learning rate was in the user's paste, which the critic never received. It
rejected the claim as an invented value -- correctly, for what it could see --
and the answer fell back to "root cause unlocalized" on a turn where the cause
was four lines above the optimiser.

Telling the critic that pasted code counts as evidence did nothing while the
paste was not in front of it. This is the half that makes that rule mean
something.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aorta.chat.graph.nodes import _last_human_message


class TestItFindsWhatTheUserSaid:
    def test_the_question_is_returned(self):
        state = {"messages": [HumanMessage(content="why does my loss go nan?")]}

        assert "nan" in _last_human_message(state)

    def test_the_paste_comes_with_it(self):
        """The point: the code has to travel, not just the sentence."""
        state = {"messages": [HumanMessage(content="why?\n\nopt = SGD(lr=5.0)")]}

        assert "lr=5.0" in _last_human_message(state)

    def test_the_latest_turn_wins(self):
        state = {
            "messages": [
                HumanMessage(content="first question"),
                AIMessage(content="an answer"),
                HumanMessage(content="second question"),
            ]
        }

        assert _last_human_message(state) == "second question"

    def test_an_assistant_turn_is_not_mistaken_for_the_user(self):
        state = {"messages": [AIMessage(content="I said this, not them")]}

        assert "not available" in _last_human_message(state)

    def test_no_messages_is_not_a_crash(self):
        assert "not available" in _last_human_message({})


class TestALargePasteDoesNotSwampIt:
    def test_it_is_truncated(self):
        """A disassembly can be megabytes; the critic checks claims, not listings."""
        state = {"messages": [HumanMessage(content="x" * 50_000)]}

        assert len(_last_human_message(state)) < 5_000

    def test_the_truncation_is_announced(self):
        state = {"messages": [HumanMessage(content="x" * 50_000)]}

        assert "truncated" in _last_human_message(state)

    def test_the_head_survives(self):
        """Where a learning rate or a launch config actually sits."""
        state = {"messages": [HumanMessage(content="lr=5.0\n" + "x" * 50_000)]}

        assert "lr=5.0" in _last_human_message(state)


class TestTheCriticIsGivenIt:
    @staticmethod
    def _source() -> str:
        from pathlib import Path

        import aorta.chat.graph.nodes as nodes

        return Path(nodes.__file__).read_text(encoding="utf-8")

    def test_the_prompt_carries_the_user_turn(self):
        assert "WHAT THE USER ASKED" in self._source()

    def test_it_is_filled_from_the_state(self):
        assert "asked = _last_human_message(state)" in self._source()

    def test_the_tool_results_are_still_there(self):
        """Adding the paste must not displace what it was checking before."""
        source = self._source()
        start = source.index("WHAT THE USER ASKED")

        assert "TOOL RESULTS" in source[start : start + 400]
        assert "GENERATED RESPONSE" in source[start : start + 400]
