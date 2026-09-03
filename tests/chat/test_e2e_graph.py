"""End-to-end graph tests: full router -> plan -> retrieve -> act -> critic flow.

Every node in the action path asks for its own chat model, so each test scripts
one fake LLM per node in visit order. ``make_llm_sequence`` reports a shortfall
as a readable AssertionError; a bare iterator surfaces it as ``RuntimeError:
coroutine raised StopIteration`` from deep inside asyncio instead.

Both ``settings`` bindings are patched: ``src.graph.nodes.settings`` drives the
act-round and critic-iteration limits, while ``route_after_critic`` reads the
separate binding in ``src.graph.graph``, so patching only the first leaves the
retry ceiling untouched.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tests.chat.conftest import make_fake_llm, make_llm_sequence


def _patch_settings(
    mock_nodes_settings, mock_graph_settings, max_retries: int = 3
) -> None:
    """Pin every settings value the action path reads to a concrete value.

    A MagicMock attribute left unset reaches real logic as a Mock: unpinned,
    ``llm_tool_mode`` matches neither protocol name and act_node rejects it.
    """
    for mock_s in (mock_nodes_settings, mock_graph_settings):
        mock_s.max_retry_iterations = max_retries
        mock_s.max_act_rounds = 5
        mock_s.max_act_rounds_search = 8
        mock_s.llm_tool_mode = "text"


class TestEndToEndGraph:
    @pytest.mark.asyncio
    async def test_full_flow_direct_answer(self, fake_retriever):
        """Full graph: router -> plan -> retrieve -> act answers -> critic passes."""
        router_llm = make_fake_llm(["action"])
        plan_llm = make_fake_llm(["1. Read src/main.py to find the entry point."])
        act_llm = make_fake_llm(["The AORTA main entry point is src/main.py."])
        critic_llm = make_fake_llm(["VALID"])

        with (
            patch("aorta.chat.graph.nodes.get_retriever", return_value=fake_retriever),
            patch(
                "aorta.chat.graph.nodes._get_llm",
                side_effect=make_llm_sequence(
                    router_llm, plan_llm, act_llm, critic_llm
                ),
            ),
            patch("aorta.chat.graph.nodes.load_repo_map", return_value="(test map)"),
            patch("aorta.chat.graph.nodes.settings") as mock_nodes_s,
            patch("aorta.chat.graph.graph.settings") as mock_graph_s,
        ):
            _patch_settings(mock_nodes_s, mock_graph_s)

            from aorta.chat.graph.graph import build_graph

            graph = build_graph()
            # Phrased to miss _SEARCH_KEYWORDS, so act may answer in one round
            # instead of being re-prompted to use tools first.
            result = await graph.ainvoke({
                "messages": [HumanMessage(content="How is AORTA started?")],
                "retrieved_context": None,
                "command_output": None,
                "critic_feedback": None,
                "iteration": 0,
            })

            assert "messages" in result
            final_msgs = result["messages"]
            ai_messages = [m for m in final_msgs if isinstance(m, AIMessage)]
            assert len(ai_messages) >= 1
            assert "main" in ai_messages[-1].content.lower()
            assert router_llm.ainvoke.call_count == 1
            assert plan_llm.ainvoke.call_count == 1
            assert critic_llm.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_full_flow_with_tool_call(self, fake_retriever):
        """Full graph: act calls a tool, then answers, and the critic passes."""
        router_llm = make_fake_llm(["action"])
        plan_llm = make_fake_llm(['1. list_files(path=".")'])
        act_llm = make_fake_llm([
            'ACTION: list_files(path=".")',
            "The root has src/, README.md, config.yaml.",
        ])
        critic_llm = make_fake_llm(["VALID"])

        with (
            patch("aorta.chat.graph.nodes.get_retriever", return_value=fake_retriever),
            patch(
                "aorta.chat.graph.nodes._get_llm",
                side_effect=make_llm_sequence(
                    router_llm, plan_llm, act_llm, critic_llm
                ),
            ),
            patch("aorta.chat.graph.nodes.load_repo_map", return_value="(test map)"),
            patch(
                "aorta.chat.graph.nodes._execute_tool",
                return_value="src/\nREADME.md\nconfig.yaml",
            ),
            patch("aorta.chat.graph.nodes.settings") as mock_nodes_s,
            patch("aorta.chat.graph.graph.settings") as mock_graph_s,
        ):
            _patch_settings(mock_nodes_s, mock_graph_s)

            from aorta.chat.graph.graph import build_graph

            graph = build_graph()
            result = await graph.ainvoke({
                "messages": [HumanMessage(content="What files are in the repo?")],
                "retrieved_context": None,
                "command_output": None,
                "critic_feedback": None,
                "iteration": 0,
            })

            assert "messages" in result
            final_msgs = result["messages"]
            ai_messages = [m for m in final_msgs if isinstance(m, AIMessage)]
            assert len(ai_messages) >= 1
            assert act_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_critic_rejection_triggers_retry(self, fake_retriever):
        """Full graph: critic rejects the answer, act runs a second time."""
        router_llm = make_fake_llm(["action"])
        plan_llm = make_fake_llm(["1. Find the scenario runner."])
        act_llm_first = make_fake_llm(["Run: bash experiment.sh --gpu 0"])
        critic_llm_first = make_fake_llm([
            "experiment.sh was not found in any tool result."
        ])
        act_llm_second = make_fake_llm(["Run: python src/run_scenario.py --gpu 0"])
        critic_llm_second = make_fake_llm(["VALID"])

        with (
            patch("aorta.chat.graph.nodes.get_retriever", return_value=fake_retriever),
            patch(
                "aorta.chat.graph.nodes._get_llm",
                side_effect=make_llm_sequence(
                    router_llm,
                    plan_llm,
                    act_llm_first,
                    critic_llm_first,
                    act_llm_second,
                    critic_llm_second,
                ),
            ),
            patch("aorta.chat.graph.nodes.load_repo_map", return_value="(test map)"),
            patch("aorta.chat.graph.nodes.settings") as mock_nodes_s,
            patch("aorta.chat.graph.graph.settings") as mock_graph_s,
        ):
            _patch_settings(mock_nodes_s, mock_graph_s)

            from aorta.chat.graph.graph import build_graph

            graph = build_graph()
            result = await graph.ainvoke({
                "messages": [HumanMessage(content="Run scenario X")],
                "retrieved_context": None,
                "command_output": None,
                "critic_feedback": None,
                "iteration": 0,
            })

            assert result["iteration"] >= 2
            assert act_llm_second.ainvoke.call_count == 1
            assert critic_llm_second.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_ceiling_ends_the_loop(self, fake_retriever):
        """MAX_RETRY_ITERATIONS=1 stops the loop even while the critic objects.

        The ceiling is read by ``route_after_critic`` through the ``settings``
        binding in ``src.graph.graph``, so a test that patches only
        ``src.graph.nodes.settings`` never controls it.
        """
        router_llm = make_fake_llm(["action"])
        plan_llm = make_fake_llm(["1. Find the scenario runner."])
        act_llm = make_fake_llm(["Run: bash experiment.sh --gpu 0"])
        critic_llm = make_fake_llm(["experiment.sh was never found."])

        with (
            patch("aorta.chat.graph.nodes.get_retriever", return_value=fake_retriever),
            patch(
                "aorta.chat.graph.nodes._get_llm",
                side_effect=make_llm_sequence(
                    router_llm, plan_llm, act_llm, critic_llm
                ),
            ),
            patch("aorta.chat.graph.nodes.load_repo_map", return_value="(test map)"),
            patch("aorta.chat.graph.nodes.settings") as mock_nodes_s,
            patch("aorta.chat.graph.graph.settings") as mock_graph_s,
        ):
            _patch_settings(mock_nodes_s, mock_graph_s, max_retries=1)

            from aorta.chat.graph.graph import build_graph

            graph = build_graph()
            result = await graph.ainvoke({
                "messages": [HumanMessage(content="Run scenario X")],
                "retrieved_context": None,
                "command_output": None,
                "critic_feedback": None,
                "iteration": 0,
            })

            assert result["iteration"] == 1
            assert act_llm.ainvoke.call_count == 1
