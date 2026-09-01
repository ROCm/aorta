"""Tool results reaching the critic, and conversations ending in a user turn.

Two bugs found running an action query against Claude behind an APIM gateway:

* The critic looked for tool results in ``state["messages"]``, but act appends
  them to its own working list and returns only its final answer, and
  ``add_messages`` merges only what a node returns. So the critic always saw
  "(no tool results gathered)" and rejected any answer citing a file -- in both
  protocols, since the repo's first commit -- which then cost a retry.
* That retry left the rejected answer as the last message, and Anthropic reads a
  trailing assistant message as a prefill to continue. The gateway rejected it
  outright: "This model does not support assistant message prefill. The
  conversation must end with a user message." HTTP 400, whole request dead.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from aorta.chat.graph.nodes import (
    _RETRY_NUDGE,
    _ensure_ends_with_user,
    act_node,
    critic_node,
)


class TestEndsWithUser:
    def test_a_trailing_assistant_message_gets_a_user_turn(self):
        messages = [HumanMessage(content="q"), AIMessage(content="rejected answer")]
        _ensure_ends_with_user(messages)
        assert isinstance(messages[-1], HumanMessage)
        assert messages[-1].content == _RETRY_NUDGE

    def test_a_trailing_user_message_is_left_alone(self):
        messages = [AIMessage(content="a"), HumanMessage(content="q")]
        _ensure_ends_with_user(messages)
        assert len(messages) == 2

    def test_a_trailing_system_message_is_left_alone(self):
        """LiteLLM lifts system messages out, so they do not end the turn."""
        messages = [HumanMessage(content="q"), SystemMessage(content="s")]
        _ensure_ends_with_user(messages)
        assert len(messages) == 2

    def test_an_empty_list_is_safe(self):
        messages: list = []
        _ensure_ends_with_user(messages)
        assert messages == []


def _retry_state():
    """State as the graph presents it after the critic rejected an answer."""
    return {
        "messages": [
            HumanMessage(content="give me a command to profile with rocprofv3"),
            AIMessage(content="an answer the critic rejected"),
        ],
        "retrieved_context": "### docs/x.md\n```\ntext\n```",
        "route": "action",
        "plan": "1. read the docs",
        "command_output": "an answer the critic rejected",
        "critic_feedback": "no tool results were gathered",
        "iteration": 1,
        "tool_trace": None,
    }


class TestActNeverSendsATrailingAssistantMessage:
    @pytest.mark.asyncio
    async def test_native_retry_appends_a_user_turn(self, monkeypatch):
        from aorta.chat.graph import nodes

        monkeypatch.setattr(nodes.settings, "llm_tool_mode", "native")
        monkeypatch.setattr(nodes.settings, "max_act_rounds", 5)
        monkeypatch.setattr(nodes.settings, "max_act_rounds_search", 8)

        plain = MagicMock()
        plain.ainvoke = AsyncMock(return_value=AIMessage(content="revised"))
        bound = MagicMock()
        bound.ainvoke = AsyncMock(return_value=AIMessage(content="revised"))
        plain.bind_tools = MagicMock(return_value=bound)

        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            await act_node(_retry_state())

        sent = bound.ainvoke.call_args[0][0]
        assert not isinstance(sent[-1], AIMessage)

    @pytest.mark.asyncio
    async def test_text_retry_appends_a_user_turn(self, monkeypatch):
        from aorta.chat.graph import nodes

        monkeypatch.setattr(nodes.settings, "llm_tool_mode", "text")
        monkeypatch.setattr(nodes.settings, "max_act_rounds", 5)
        monkeypatch.setattr(nodes.settings, "max_act_rounds_search", 8)

        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content="revised"))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            await act_node(_retry_state())

        sent = fake.ainvoke.call_args[0][0]
        assert not isinstance(sent[-1], AIMessage)


class TestActReportsWhatItsToolsReturned:
    @pytest.mark.asyncio
    async def test_native_returns_a_trace(self, monkeypatch):
        from aorta.chat.graph import nodes

        monkeypatch.setattr(nodes.settings, "llm_tool_mode", "native")
        monkeypatch.setattr(nodes.settings, "max_act_rounds", 5)
        monkeypatch.setattr(nodes.settings, "max_act_rounds_search", 8)

        call = {
            "name": "read_file",
            "args": {"file_path": "docs/profiling.md"},
            "id": "c1",
            "type": "tool_call",
        }
        plain = MagicMock()
        plain.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
        bound = MagicMock()
        bound.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content="", tool_calls=[call]),
                AIMessage(content="Use `aorta sweep run --collect rocprof`."),
            ]
        )
        plain.bind_tools = MagicMock(return_value=bound)

        state = _retry_state()
        state["messages"] = [HumanMessage(content="how do I profile?")]
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="file contents here"),
        ):
            result = await act_node(state)

        assert result["tool_trace"]
        assert "file contents here" in result["tool_trace"][0]
        assert "read_file" in result["tool_trace"][0]

    @pytest.mark.asyncio
    async def test_text_returns_a_trace(self, monkeypatch):
        from aorta.chat.graph import nodes

        monkeypatch.setattr(nodes.settings, "llm_tool_mode", "text")
        monkeypatch.setattr(nodes.settings, "max_act_rounds", 5)
        monkeypatch.setattr(nodes.settings, "max_act_rounds_search", 8)

        fake = MagicMock()
        fake.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content='ACTION: read_file(file_path="docs/profiling.md")'),
                AIMessage(content="Use `aorta sweep run --collect rocprof`."),
            ]
        )
        state = _retry_state()
        state["messages"] = [HumanMessage(content="how do I profile?")]
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=fake),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="file contents here"),
        ):
            result = await act_node(state)

        assert result["tool_trace"]
        assert "file contents here" in result["tool_trace"][0]


class TestCriticSeesTheTrace:
    @pytest.mark.asyncio
    async def test_a_grounded_answer_is_accepted(self, monkeypatch):
        """With the trace visible the critic can validate rather than reject."""
        from aorta.chat.graph import nodes

        monkeypatch.setattr(nodes.settings, "max_retry_iterations", 3)
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content="VALID"))
        state = {
            "messages": [HumanMessage(content="how do I profile?")],
            "command_output": (
                "Use `aorta sweep run --collect rocprof` (docs/profiling.md:11)."
            ),
            "tool_trace": [
                "TOOL RESULT from read_file:\naorta sweep run --collect rocprof"
            ],
            "iteration": 0,
            "critic_feedback": None,
            "route": "action",
            "plan": None,
            "retrieved_context": "",
        }
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await critic_node(state)
        assert result["critic_feedback"] is None

    @pytest.mark.asyncio
    async def test_the_trace_is_what_the_critic_is_shown(self, monkeypatch):
        from aorta.chat.graph import nodes

        monkeypatch.setattr(nodes.settings, "max_retry_iterations", 3)
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content="VALID"))
        state = {
            "messages": [HumanMessage(content="q")],
            "command_output": "an answer",
            "tool_trace": ["TOOL RESULT from grep_code:\nunmistakable-marker"],
            "iteration": 0,
            "critic_feedback": None,
            "route": "action",
            "plan": None,
            "retrieved_context": "",
        }
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            await critic_node(state)
        prompt = fake.ainvoke.call_args[0][0][-1].content
        assert "unmistakable-marker" in prompt
        assert "no tool results gathered" not in prompt

    @pytest.mark.asyncio
    async def test_a_nonzero_exit_code_in_the_trace_is_still_caught(
        self, monkeypatch
    ):
        """Failure detection reads the trace too, not only the message scan."""
        from aorta.chat.graph import nodes

        monkeypatch.setattr(nodes.settings, "max_retry_iterations", 3)
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content="the command failed"))
        state = {
            "messages": [HumanMessage(content="run the tests")],
            "command_output": "I ran pytest",
            "tool_trace": [
                "TOOL RESULT from run_terminal_command:\nExit code: 1\nboom"
            ],
            "iteration": 0,
            "critic_feedback": None,
            "route": "action",
            "plan": None,
            "retrieved_context": "",
        }
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await critic_node(state)
        assert result["critic_feedback"] == "the command failed"

    @pytest.mark.asyncio
    async def test_an_absent_trace_does_not_crash(self, monkeypatch):
        from aorta.chat.graph import nodes

        monkeypatch.setattr(nodes.settings, "max_retry_iterations", 3)
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content="VALID"))
        state = {
            "messages": [HumanMessage(content="q")],
            "command_output": "an answer",
            "tool_trace": None,
            "iteration": 0,
            "critic_feedback": None,
            "route": "action",
            "plan": None,
            "retrieved_context": "",
        }
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await critic_node(state)
        assert result["critic_feedback"] is None
