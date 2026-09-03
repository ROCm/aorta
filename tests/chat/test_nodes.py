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


class TestRunArtifactsReachTheToolFreeBranch:
    """"Why did this sweep fail?" is specific, so the router calls it a question.

    That branch has no tools, and retrieval only ever queried the source
    collection -- so the PR's headline use case was answerable only from source
    code, by the route most likely to ask it.
    """

    @staticmethod
    def _run_docs(*contents):
        from langchain_core.documents import Document

        return [
            Document(page_content=text, metadata={"source": "run_nan/matrix.json",
                                                  "artifact_kind": "matrix"})
            for text in contents
        ]

    @pytest.mark.asyncio
    async def test_run_documents_are_added_to_the_context(self, fake_retriever):
        from aorta.chat.graph import nodes

        with (
            patch.object(nodes, "get_retriever", return_value=fake_retriever),
            patch(
                "aorta.chat.rag.runs.search_run_docs",
                return_value=self._run_docs("failure_rate: 0.750 nan detected in loss"),
            ),
        ):
            state = {"messages": [HumanMessage(content="why did this sweep fail?")]}
            result = await nodes.retrieve_node(state)

        assert "nan detected in loss" in result["retrieved_context"]
        assert "run_nan/matrix.json" in result["retrieved_context"]

    @pytest.mark.asyncio
    async def test_run_context_answers_even_when_no_source_matched(self):
        """Otherwise a run-only question still reports "no relevant code"."""
        from aorta.chat.graph import nodes

        empty = MagicMock()
        empty.invoke.return_value = []
        with (
            patch.object(nodes, "get_retriever", return_value=empty),
            patch(
                "aorta.chat.rag.runs.search_run_docs",
                return_value=self._run_docs("failure_rate: 0.750"),
            ),
        ):
            state = {"messages": [HumanMessage(content="why did this sweep fail?")]}
            result = await nodes.retrieve_node(state)

        assert "failure_rate: 0.750" in result["retrieved_context"]
        assert "No relevant code" not in result["retrieved_context"]

    @pytest.mark.asyncio
    async def test_a_missing_run_collection_is_not_an_error(self, fake_retriever):
        """Most installs have never run ``aorta chat index runs``."""
        from aorta.chat.graph import nodes
        from aorta.chat.rag.runs import RunCollectionMissingError

        with (
            patch.object(nodes, "get_retriever", return_value=fake_retriever),
            patch(
                "aorta.chat.rag.runs.search_run_docs",
                side_effect=RunCollectionMissingError("no collection"),
            ),
        ):
            state = {"messages": [HumanMessage(content="how do I run scenarios?")]}
            result = await nodes.retrieve_node(state)

        assert "run_scenario" in result["retrieved_context"]

    @pytest.mark.asyncio
    async def test_a_broken_run_store_never_takes_the_answer_with_it(self, fake_retriever):
        """Supplementary context. The query is still answerable from source."""
        from aorta.chat.graph import nodes

        with (
            patch.object(nodes, "get_retriever", return_value=fake_retriever),
            patch(
                "aorta.chat.rag.runs.search_run_docs",
                side_effect=RuntimeError("remote embedding endpoint went away"),
            ),
        ):
            state = {"messages": [HumanMessage(content="how do I run scenarios?")]}
            result = await nodes.retrieve_node(state)

        assert "run_scenario" in result["retrieved_context"]


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
