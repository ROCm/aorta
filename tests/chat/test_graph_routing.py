"""Tests for route_after_critic() and graph structure."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.graph import END

from aorta.chat.graph.graph import build_graph, route_after_critic
from aorta.chat.graph.nodes import finalize_node


class TestRouteAfterCritic:
    @patch("aorta.chat.graph.graph.settings")
    def test_returns_act_when_feedback_and_under_max(self, mock_settings):
        mock_settings.max_retry_iterations = 3
        state = {"critic_feedback": "Command not grounded", "iteration": 1}
        assert route_after_critic(state) == "act"

    @patch("aorta.chat.graph.graph.settings")
    def test_returns_end_when_no_feedback(self, mock_settings):
        mock_settings.max_retry_iterations = 3
        state = {"critic_feedback": None, "iteration": 1}
        assert route_after_critic(state) == END

    @patch("aorta.chat.graph.graph.settings")
    def test_returns_end_when_feedback_empty_string(self, mock_settings):
        mock_settings.max_retry_iterations = 3
        state = {"critic_feedback": "", "iteration": 1}
        assert route_after_critic(state) == END

    @patch("aorta.chat.graph.graph.settings")
    def test_finalizes_when_iteration_exceeds_max(self, mock_settings):
        """Exhaustion with the critic still objecting is not a plain END.

        This used to route to END, which left the rejected AIMessage last in
        state for ``invoke_agent`` to return as an ordinary answer. The verdict
        then reached nobody, so the one case the critic exists to catch was the
        one case it could not report.
        """
        mock_settings.max_retry_iterations = 3
        state = {"critic_feedback": "some issue", "iteration": 5}
        assert route_after_critic(state) == "finalize"

    @patch("aorta.chat.graph.graph.settings")
    def test_finalizes_when_iteration_equals_max(self, mock_settings):
        mock_settings.max_retry_iterations = 3
        state = {"critic_feedback": "some issue", "iteration": 3}
        assert route_after_critic(state) == "finalize"

    @patch("aorta.chat.graph.graph.settings")
    def test_an_accepted_answer_at_the_ceiling_still_ends(self, mock_settings):
        """Only an outstanding objection finalizes; a clean pass is just done."""
        mock_settings.max_retry_iterations = 3
        state = {"critic_feedback": None, "iteration": 3}
        assert route_after_critic(state) == END

    @patch("aorta.chat.graph.graph.settings")
    def test_returns_act_at_boundary(self, mock_settings):
        mock_settings.max_retry_iterations = 3
        state = {"critic_feedback": "try again", "iteration": 2}
        assert route_after_critic(state) == "act"

    @patch("aorta.chat.graph.graph.settings")
    def test_missing_iteration_defaults_to_zero(self, mock_settings):
        mock_settings.max_retry_iterations = 3
        state = {"critic_feedback": "retry"}
        assert route_after_critic(state) == "act"


class TestGraphStructure:
    def test_graph_compiles(self):
        """build_graph() should return a compiled graph without errors."""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "retrieve" in node_names
        assert "act" in node_names
        assert "critic" in node_names
        assert "finalize" in node_names


class TestFinalizeNode:
    """What the user is told when the retry budget ran out on a rejection."""

    async def test_the_unresolved_criticism_is_reported(self):
        state = {
            "messages": [],
            "command_output": "Run `aorta frobnicate --all` to fix it.",
            "critic_feedback": "no tool result mentions a frobnicate command",
            "iteration": 3,
        }
        result = await finalize_node(state)
        reply = result["messages"][0].content
        assert "could not verify" in reply
        assert "no tool result mentions a frobnicate command" in reply

    async def test_the_rejected_answer_is_kept_but_marked(self):
        """Kept because it is often partly right and the user waited for it."""
        state = {
            "messages": [],
            "command_output": "The router node lives in graph/nodes.py.",
            "critic_feedback": "ungrounded",
            "iteration": 3,
        }
        reply = (await finalize_node(state))["messages"][0].content
        assert "The router node lives in graph/nodes.py." in reply
        assert reply.index("could not verify") < reply.index("The router node")

    async def test_it_reports_the_budget_it_spent(self):
        """Named as attempts, which is what ``iteration`` counts.

        The first critic pass is the initial validation, not a retry, so
        calling the count "retries" would overstate the budget by one.
        """
        state = {
            "messages": [],
            "command_output": "an answer",
            "critic_feedback": "nope",
            "iteration": 3,
        }
        reply = (await finalize_node(state))["messages"][0].content
        assert "all 3 attempts" in reply
        assert "retries" not in reply

    async def test_it_costs_no_llm_call(self):
        """Exhausting the budget must not itself be able to fail or bill."""
        state = {
            "messages": [],
            "command_output": "an answer",
            "critic_feedback": "nope",
            "iteration": 3,
        }
        with patch("aorta.chat.graph.nodes._get_llm") as get_llm:
            await finalize_node(state)
        get_llm.assert_not_called()

    async def test_it_falls_back_to_the_last_ai_message(self):
        state = {
            "messages": [AIMessage(content="a prior answer")],
            "command_output": None,
            "critic_feedback": "nope",
            "iteration": 3,
        }
        reply = (await finalize_node(state))["messages"][0].content
        assert "a prior answer" in reply

    async def test_it_says_so_when_there_was_no_answer_at_all(self):
        state = {
            "messages": [],
            "command_output": "",
            "critic_feedback": "nope",
            "iteration": 3,
        }
        reply = (await finalize_node(state))["messages"][0].content
        assert "no answer was produced" in reply


class TestExhaustionReachesTheUser:
    """The end-to-end property the finalize node exists for.

    ``invoke_agent`` reads its reply from the last AIMessage in state, so a
    node that only logged the rejection would not fix anything.
    """

    async def test_the_reply_extracted_from_state_carries_the_warning(self):
        from aorta.chat.session import extract_reply

        state = {
            "messages": [AIMessage(content="Run `aorta frobnicate`.")],
            "command_output": "Run `aorta frobnicate`.",
            "critic_feedback": "no such command in the tool output",
            "iteration": 3,
        }
        result = await finalize_node(state)
        # add_messages appends, so the finalize message is last.
        reply = extract_reply([*state["messages"], *result["messages"]])
        assert "could not verify" in reply
        assert "no such command in the tool output" in reply
