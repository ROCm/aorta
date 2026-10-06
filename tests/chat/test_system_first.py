"""The system message reaches the provider at the front of the conversation.

AMD's on-prem gateway rejects a conversation whose system message is anywhere
else -- ``System message must be at the beginning``, HTTP 400 -- and the act
nodes add a second system message ahead of the history to force a behaviour for
one turn. That combination takes down every on-prem model, so it is pinned here
rather than rediscovered per model.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from aorta.chat.graph.nodes import _send, _system_first


def _kinds(messages):
    return [type(m).__name__ for m in messages]


def test_single_leading_system_message_is_untouched():
    messages = [SystemMessage(content="rules"), HumanMessage(content="hi")]
    assert _system_first(messages) is messages


def test_conversation_without_a_system_message_is_untouched():
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    assert _system_first(messages) is messages


def test_two_system_messages_collapse_to_one_at_the_front():
    out = _system_first(
        [
            SystemMessage(content="standing rules"),
            SystemMessage(content="force a search this turn"),
            HumanMessage(content="hi"),
        ]
    )

    assert _kinds(out) == ["SystemMessage", "HumanMessage"]
    assert out[0].content == "standing rules\n\nforce a search this turn"


def test_system_message_behind_the_history_moves_to_the_front():
    """The shape the gateway actually rejected."""
    out = _system_first(
        [
            HumanMessage(content="analyse this kernel"),
            SystemMessage(content="you must call a tool"),
        ]
    )

    assert _kinds(out) == ["SystemMessage", "HumanMessage"]
    assert out[0].content == "you must call a tool"


def test_merge_keeps_the_order_the_instructions_were_added_in():
    out = _system_first(
        [
            SystemMessage(content="first"),
            HumanMessage(content="hi"),
            SystemMessage(content="second"),
            SystemMessage(content="third"),
        ]
    )

    assert out[0].content == "first\n\nsecond\n\nthird"


def test_non_system_messages_keep_their_order_and_count():
    history = [
        HumanMessage(content="one"),
        AIMessage(content="two"),
        ToolMessage(content="three", tool_call_id="t1"),
        HumanMessage(content="four"),
    ]
    out = _system_first([*history, SystemMessage(content="rules")])

    assert [m.content for m in out[1:]] == ["one", "two", "three", "four"]
    assert _kinds(out[1:]) == _kinds(history)


def test_empty_system_bodies_do_not_leave_blank_separators():
    out = _system_first(
        [
            SystemMessage(content="rules"),
            SystemMessage(content="   "),
            HumanMessage(content="hi"),
        ]
    )

    assert out[0].content == "rules"


@pytest.mark.asyncio
async def test_send_normalises_before_the_model_sees_the_messages():
    """The guarantee has to hold at the choke point, not just in the helper."""
    seen: list[list] = []

    class Recorder:
        async def ainvoke(self, messages):
            seen.append(list(messages))
            return AIMessage(content="ok")

    await _send(
        Recorder(),
        [HumanMessage(content="analyse this"), SystemMessage(content="rules")],
    )

    assert _kinds(seen[0]) == ["SystemMessage", "HumanMessage"]
