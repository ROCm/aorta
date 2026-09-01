"""Integration tests for act_node, critic_node, and retrieve_node with mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tests.chat.conftest import make_fake_llm


class TestRetrieveNode:
    @pytest.mark.asyncio
    async def test_retrieves_context(self, fake_retriever):
        with patch("aorta.chat.graph.nodes.get_retriever", return_value=fake_retriever):
            from aorta.chat.graph.nodes import retrieve_node

            state = {"messages": [HumanMessage(content="How do I run scenarios?")]}
            result = await retrieve_node(state)

            assert "retrieved_context" in result
            assert "run_scenario" in result["retrieved_context"]

    @pytest.mark.asyncio
    async def test_no_human_message(self):
        from aorta.chat.graph.nodes import retrieve_node

        state = {"messages": [AIMessage(content="Hi")]}
        result = await retrieve_node(state)
        assert result["retrieved_context"] == ""

    @pytest.mark.asyncio
    async def test_index_not_built(self):
        def _raise(*a, **kw):
            raise FileNotFoundError("no index")

        with patch("aorta.chat.graph.nodes.get_retriever", side_effect=_raise):
            from aorta.chat.graph.nodes import retrieve_node

            state = {"messages": [HumanMessage(content="test query")]}
            result = await retrieve_node(state)
            assert "not built yet" in result["retrieved_context"]

    @pytest.mark.asyncio
    async def test_empty_docs(self):
        mock_ret = MagicMock()
        mock_ret.invoke.return_value = []
        with patch("aorta.chat.graph.nodes.get_retriever", return_value=mock_ret):
            from aorta.chat.graph.nodes import retrieve_node

            state = {"messages": [HumanMessage(content="obscure query")]}
            result = await retrieve_node(state)
            assert "No relevant code" in result["retrieved_context"]


class TestActNode:
    @pytest.mark.asyncio
    async def test_direct_answer_no_action(self):
        """LLM answers directly without calling any tools."""
        fake = make_fake_llm(["The AORTA project uses Python 3.10+."])
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=fake),
            patch("aorta.chat.graph.nodes.load_repo_map", return_value="(map)"),
        ):
            from aorta.chat.graph.nodes import act_node

            state = {
                "messages": [HumanMessage(content="What Python version?")],
                "retrieved_context": "Python 3.10+",
            }
            result = await act_node(state)

            assert "messages" in result
            ai_msg = result["messages"][0]
            assert isinstance(ai_msg, AIMessage)
            assert "Python" in ai_msg.content

    @pytest.mark.asyncio
    async def test_single_tool_call_then_answer(self):
        """LLM calls a tool, gets result, then answers."""
        fake = make_fake_llm([
            'Let me check. ACTION: list_files(path=".")',
            "The root directory has: src/, config.yaml, README.md",
        ])
        mock_tool_result = "src/\nconfig.yaml\nREADME.md"

        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=fake),
            patch("aorta.chat.graph.nodes.load_repo_map", return_value="(map)"),
            patch("aorta.chat.graph.nodes._execute_tool", return_value=mock_tool_result),
        ):
            from aorta.chat.graph.nodes import act_node

            state = {
                "messages": [HumanMessage(content="What files are in the root?")],
                "retrieved_context": "",
            }
            result = await act_node(state)

            ai_msg = result["messages"][0]
            assert isinstance(ai_msg, AIMessage)
            assert "src/" in ai_msg.content or "README" in ai_msg.content

    @pytest.mark.asyncio
    async def test_max_rounds_exceeded(self):
        """LLM keeps calling tools and hits the 5-round limit."""
        responses = [
            'ACTION: list_files(path=".")',
            'ACTION: read_file(file_path="src/main.py")',
            'ACTION: search_code(query="entry")',
            'ACTION: list_files(path="src")',
            'ACTION: read_file(file_path="config.yaml")',
            "Final answer after exhausting rounds.",
        ]
        fake = make_fake_llm(responses)

        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=fake),
            patch("aorta.chat.graph.nodes.load_repo_map", return_value="(map)"),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="tool output"),
        ):
            from aorta.chat.graph.nodes import act_node

            state = {
                "messages": [HumanMessage(content="Complex question")],
                "retrieved_context": "",
            }
            result = await act_node(state)

            ai_msg = result["messages"][0]
            assert isinstance(ai_msg, AIMessage)
            assert fake.ainvoke.call_count == 6

    @pytest.mark.asyncio
    async def test_critic_feedback_injected(self):
        """When critic_feedback is set, it's included in the system messages."""
        fake = make_fake_llm(["Corrected answer based on feedback."])

        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=fake),
            patch("aorta.chat.graph.nodes.load_repo_map", return_value="(map)"),
        ):
            from aorta.chat.graph.nodes import act_node

            state = {
                "messages": [HumanMessage(content="Run scenario X")],
                "retrieved_context": "",
                "critic_feedback": "Previous command was wrong: script not found.",
            }
            await act_node(state)

            call_args = fake.ainvoke.call_args[0][0]
            system_texts = [m.content for m in call_args if hasattr(m, "content")]
            assert any("PREVIOUS COMMAND FAILED" in t for t in system_texts)


class TestCriticNode:
    @pytest.mark.asyncio
    async def test_no_command_output(self):
        """Critic returns no feedback when there's no command output."""
        with patch("aorta.chat.graph.nodes.settings") as mock_s:
            mock_s.max_retry_iterations = 3
            from aorta.chat.graph.nodes import critic_node

            state = {
                "messages": [HumanMessage(content="hi")],
                "command_output": "",
                "iteration": 0,
            }
            result = await critic_node(state)
            assert result["critic_feedback"] is None
            assert result["iteration"] == 1

    @pytest.mark.asyncio
    async def test_exceeds_max_iterations(self):
        """Critic gives up when iteration exceeds max."""
        with patch("aorta.chat.graph.nodes.settings") as mock_s:
            mock_s.max_retry_iterations = 2
            from aorta.chat.graph.nodes import critic_node

            state = {
                "messages": [],
                "command_output": "some output",
                "iteration": 3,
            }
            result = await critic_node(state)
            assert result["critic_feedback"] is None

    @pytest.mark.asyncio
    async def test_detects_nonzero_exit_code(self):
        """Critic detects command failure from exit code in tool results."""
        failure_msg = HumanMessage(
            content="TOOL RESULT from run_terminal_command:\n"
            "Exit code: 1\nError: file not found"
        )
        fake = make_fake_llm(["Root cause: missing file. Fix: create it."])

        with (
            patch("aorta.chat.graph.nodes.settings") as mock_s,
            patch("aorta.chat.graph.nodes._get_llm", return_value=fake),
        ):
            mock_s.max_retry_iterations = 3
            from aorta.chat.graph.nodes import critic_node

            state = {
                "messages": [failure_msg],
                "command_output": "python run.py",
                "iteration": 0,
            }
            result = await critic_node(state)
            assert result["critic_feedback"] is not None
            assert "missing file" in result["critic_feedback"]

    @pytest.mark.asyncio
    async def test_valid_response_passes(self):
        """Critic returns no feedback when response is VALID."""
        tool_msg = HumanMessage(
            content="TOOL RESULT from list_files:\nsrc/\nconfig.yaml"
        )
        fake = make_fake_llm(["VALID"])

        with (
            patch("aorta.chat.graph.nodes.settings") as mock_s,
            patch("aorta.chat.graph.nodes._get_llm", return_value=fake),
        ):
            mock_s.max_retry_iterations = 3
            from aorta.chat.graph.nodes import critic_node

            state = {
                "messages": [tool_msg],
                "command_output": "The root has src/ and config.yaml",
                "iteration": 0,
            }
            result = await critic_node(state)
            assert result["critic_feedback"] is None

    @pytest.mark.asyncio
    async def test_invalid_response_triggers_feedback(self):
        """Critic rejects hallucinated commands."""
        tool_msg = HumanMessage(
            content="TOOL RESULT from list_files:\nsrc/\nconfig.yaml"
        )
        fake = make_fake_llm([
            "The response references run_experiment.sh which was not found by any tool."
        ])

        with (
            patch("aorta.chat.graph.nodes.settings") as mock_s,
            patch("aorta.chat.graph.nodes._get_llm", return_value=fake),
        ):
            mock_s.max_retry_iterations = 3
            from aorta.chat.graph.nodes import critic_node

            state = {
                "messages": [tool_msg],
                "command_output": "Run: bash run_experiment.sh --gpu 0",
                "iteration": 0,
            }
            result = await critic_node(state)
            assert result["critic_feedback"] is not None
            assert "not found" in result["critic_feedback"]
