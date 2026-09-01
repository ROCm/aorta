"""Tests for route_after_critic() and graph structure."""

from __future__ import annotations

from unittest.mock import patch

from langgraph.graph import END

from aorta.chat.graph.graph import build_graph, route_after_critic


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
    def test_returns_end_when_iteration_exceeds_max(self, mock_settings):
        mock_settings.max_retry_iterations = 3
        state = {"critic_feedback": "some issue", "iteration": 5}
        assert route_after_critic(state) == END

    @patch("aorta.chat.graph.graph.settings")
    def test_returns_end_when_iteration_equals_max(self, mock_settings):
        mock_settings.max_retry_iterations = 3
        state = {"critic_feedback": "some issue", "iteration": 3}
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
