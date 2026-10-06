"""The selector has to see the code it is deciding about.

Two halves of one question were reading different amounts of the conversation.
The requirement check looked back over the recent human turns, so a kernel
pasted a turn or two ago kept the source tools eligible. The model was sent the
newest message and nothing else.

Between those, a follow-up like "32, go ahead" landed in the gap: the source
tools survived the requirement check, and then the selector was asked to rank
them against four words with no code in them. It ranks on "would the evidence
this tool returns answer this problem?", and the problem it could see was a
block size. The kernel was one turn out of reach.

So the window is one definition now, used by both. These tests hold that from
the outside: what the check counts as recent is what the model is shown.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("langchain_core", reason="the graph needs the chat-cli extra")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from aorta.chat.graph.nodes import (
    _SOURCE_LOOKBACK,
    _conversation_has_source,
    _recent_human_turns,
    _selector_view,
    selector_node,
)

_KERNEL = "```\n__global__ void k(float* o) { __shared__ float s[64]; o[0] += s[1]; }\n```"


def _sent_to_the_model(messages: list) -> str:
    """Run the selector over *messages* and return what the model was asked."""
    seen: dict = {}

    llm = MagicMock()

    async def capture(sent, *args, **kwargs):
        seen["human"] = "\n".join(
            str(m.content) for m in sent if isinstance(m, HumanMessage)
        )
        return AIMessage(content=json.dumps({"tools": [], "why": ""}))

    llm.ainvoke = AsyncMock(side_effect=capture)

    async def run():
        with patch("aorta.chat.graph.nodes._get_llm", return_value=llm):
            await selector_node({"messages": messages})

    import asyncio

    asyncio.run(run())
    return seen.get("human", "")


class TestTheFollowUpTurn:
    """A kernel, a clarifying question, and a four-word reply."""

    @staticmethod
    def _conversation() -> list:
        return [
            HumanMessage(content=_KERNEL),
            AIMessage(content="What block size should I launch it with?"),
            HumanMessage(content="32, go ahead"),
        ]

    def test_the_requirement_check_finds_the_paste(self):
        """The half that already worked, pinned so the two stay together."""
        assert _conversation_has_source(self._conversation()) is True

    def test_and_now_the_model_is_shown_it_too(self):
        asked = _sent_to_the_model(self._conversation())

        assert "__global__" in asked

    def test_the_current_message_is_still_there(self):
        asked = _sent_to_the_model(self._conversation())

        assert "32, go ahead" in asked

    def test_the_current_message_comes_last(self):
        """It is the ask; the rest is context for it."""
        asked = _sent_to_the_model(self._conversation())

        assert asked.index("__global__") < asked.index("32, go ahead")


class TestTheWindowIsTheSameOne:
    def test_both_halves_read_the_same_turns(self):
        convo = [HumanMessage(content=f"turn {i}") for i in range(_SOURCE_LOOKBACK + 4)]

        assert len(_recent_human_turns(convo)) == _SOURCE_LOOKBACK

    def test_a_paste_older_than_the_window_is_not_shown(self):
        """The bound is the point: an ancient kernel stops deciding this turn."""
        convo = [HumanMessage(content=_KERNEL)]
        convo += [HumanMessage(content=f"later {i}") for i in range(_SOURCE_LOOKBACK)]

        assert _conversation_has_source(convo) is False
        assert "__global__" not in _selector_view(_recent_human_turns(convo))

    def test_a_paste_at_the_edge_of_the_window_still_is(self):
        convo = [HumanMessage(content=_KERNEL)]
        convo += [HumanMessage(content=f"later {i}") for i in range(_SOURCE_LOOKBACK - 1)]

        assert _conversation_has_source(convo) is True
        assert "__global__" in _selector_view(_recent_human_turns(convo))


class TestOnlyTheUsersTurns:
    def test_the_assistant_is_not_quoted_back_to_the_selector(self):
        """Its own suggestions are not evidence about what the user has."""
        convo = [
            HumanMessage(content="my loss goes to NaN"),
            AIMessage(content=f"Something like {_KERNEL} would reproduce it"),
            HumanMessage(content="go ahead"),
        ]

        assert "__global__" not in _sent_to_the_model(convo)

    def test_tool_output_is_not_either(self):
        convo = [
            HumanMessage(content="my loss goes to NaN"),
            ToolMessage(content=_KERNEL, tool_call_id="1"),
            HumanMessage(content="go ahead"),
        ]

        assert "__global__" not in _sent_to_the_model(convo)

    def test_the_system_prompt_is_not_mistaken_for_a_turn(self):
        convo = [
            SystemMessage(content="you are a helpful assistant"),
            HumanMessage(content="my loss goes to NaN"),
        ]

        assert _recent_human_turns(convo) == ["my loss goes to NaN"]


class TestTheOrdinarySingleTurnIsUntouched:
    """One message in, that exact message out -- the common case and the one
    the prompt was written against."""

    def test_it_is_passed_through_verbatim(self):
        assert _selector_view(["just this one thing"]) == "just this one thing"

    def test_no_framing_is_added_around_it(self):
        asked = _sent_to_the_model([HumanMessage(content="my loss goes to NaN")])

        assert asked == "my loss goes to NaN"

    def test_an_empty_conversation_does_not_raise(self):
        assert _selector_view([]) == ""
