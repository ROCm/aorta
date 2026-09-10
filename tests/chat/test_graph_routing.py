"""Tests for router_node(), route_after_critic() and graph structure."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from aorta.chat.graph.graph import build_graph, route_after_critic
from aorta.chat.graph.nodes import _parse_route, finalize_node, router_node


async def _route_reply(reply: str) -> str:
    """The branch ``router_node`` lands on when the model answers *reply*.

    Drives the node rather than reimplementing its decision, so a test that
    claims a reply moved is reporting where it actually goes.
    """
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=reply))
    with patch("aorta.chat.graph.nodes._get_llm", return_value=llm):
        result = await router_node(
            {"messages": [HumanMessage(content="where are the tokenspeed docs?")]}
        )
    return result["route"]


class TestParseRoute:
    """The classification itself, without a model in the way."""

    def test_a_bare_route_name_parses(self):
        assert _parse_route("question") == "question"
        assert _parse_route("action") == "action"

    def test_a_name_inside_a_sentence_still_parses(self):
        assert _parse_route("this is an action") == "action"

    def test_an_empty_reply_names_nothing(self):
        assert _parse_route("") is None

    def test_a_reply_naming_neither_names_nothing(self):
        assert _parse_route("let me look at the code") is None

    def test_a_reply_naming_both_is_ambiguous_rather_than_a_winner(self):
        """Substring matching has to be symmetric to be honest.

        ``"question" in text`` alone called this one ``question`` while
        ``"action" in text`` alone would have called it ``action``; neither is a
        classification the model made.
        """
        assert _parse_route("action, not question") is None


class TestRouterNodeFallback:
    """Which branch a reply that classified nothing lands on.

    There are two such replies and they are not the same failure, which is the
    distinction this class exists to pin.

    An **empty** reply used to go to ``action``, by omission: ``"question" if
    "question" in reply else "action"``. That is the branch a model returning
    empty content cannot drive, and empty content is exactly what such a model
    returns -- so a router that failed to answer routed into the one branch
    guaranteed to fail.

    A **content-bearing** reply we could not parse is the opposite case: the
    model did classify, and the substring match failed to recognise how it said
    so. Those still go to ``action``. Sending them to ``question`` fixed nothing
    and removed tool access from a model that was asking for tools.
    """

    @pytest.mark.parametrize("reply", ["question", "action"])
    async def test_a_real_classification_is_honoured(self, reply):
        assert await _route_reply(reply) == reply

    @pytest.mark.parametrize("reply", ["", "   ", "\n\t "])
    async def test_an_empty_reply_takes_the_answerable_branch(self, reply):
        """Nothing was classified, so prefer the branch that can still answer.

        ``question`` degrades to a retrieval-only answer; ``action`` to none.
        Whitespace counts as empty because the reply is stripped first -- the
        model said nothing either way.
        """
        assert await _route_reply(reply) == "question"

    @pytest.mark.parametrize(
        "reply",
        [
            "let me check",
            "action or question",  # named both, so it parsed to neither
            "act",
            "use the shell to fix it",
            "tool",
        ],
    )
    async def test_a_content_bearing_reply_keeps_its_tools(self, reply):
        """The regression this narrowing removes.

        Each of these is a model that answered the router; we just could not
        read the answer. ``action`` is the only branch with tool access, and an
        unreadable reply is not evidence that none were wanted.
        """
        assert await _route_reply(reply) == "action"

    @pytest.mark.parametrize("reply", ["", "let me check"])
    async def test_the_fallback_is_reported_rather_than_silent(self, reply, caplog):
        """Both fallbacks are logged, not just the empty one."""
        with caplog.at_level("WARNING"):
            await _route_reply(reply)
        assert "does not name exactly one route" in caplog.text


class TestOnlyUnclassifiedRepliesChanged:
    """The bound on this change, and the reason it is not a reweighting.

    Gap 3c also proposed biasing the router away from ``action``. The reported
    symptom does not support that: the reply was a real ``'action'``, the prompt
    asked for ``action`` on a "find"-shaped question, and the query died in the
    act loop rather than in the classifier. Reweighting would move queries the
    model *did* classify, which is exactly what
    https://github.com/ROCm/aorta/issues/433 warns regresses in mirror image.

    So the change is confined to replies that are **empty**. This class is the
    proof, and it is deliberately stated end to end: for every reply naming
    exactly one route -- which is every reply the model is asked for, and every
    reply either issue's repro produced -- the new rule agrees with the old one,
    and for every reply that named none but *said something*, the branch is
    unchanged too.

    The moved set was briefly wider than this. Routing every unparsed reply to
    ``question`` also moved content-bearing ones, which took tools away from a
    model that had asked for them; narrowing the fallback to empty replies gave
    those back. The parametrisations below name the boundary rather than
    asserting a single destination for both populations, so widening it again
    would fail here rather than pass quietly.
    """

    @staticmethod
    def _old_rule(route_text: str) -> str:
        """Verbatim from before: ``"question" if "question" in reply else "action"``."""
        return "question" if "question" in route_text else "action"

    @pytest.mark.parametrize(
        "reply",
        [
            "question",
            "action",
            "Question",
            "this is an action",
            "question.",
            "the answer is: action",
            # #433's repro shape: a run question the model calls a question.
            "question -- it asks about a run on this machine",
        ],
    )
    def test_a_real_classification_is_decided_exactly_as_before(self, reply):
        lowered = reply.strip().lower()
        assert _parse_route(lowered) == self._old_rule(lowered)

    @pytest.mark.parametrize("reply", ["", "   "])
    async def test_only_an_empty_reply_moved(self, reply):
        """The changed case: the old rule sent it to the branch it could not drive."""
        lowered = reply.strip().lower()
        assert _parse_route(lowered) is None, "it classified nothing"
        assert self._old_rule(lowered) == "action"
        assert await _route_reply(reply) == "question"

    @pytest.mark.parametrize("reply", ["let me check", "unsure", "act"])
    async def test_a_reply_that_said_something_did_not_move(self, reply):
        """The bound on the moved set, and what the narrowing restored.

        These classify nothing either -- ``_parse_route`` returns ``None`` for
        all three -- so the *parser* treats them exactly like an empty reply.
        The router does not, and that is the point: they still land where they
        always did.
        """
        lowered = reply.strip().lower()
        assert _parse_route(lowered) is None, "it classified nothing"
        assert self._old_rule(lowered) == "action"
        assert await _route_reply(reply) == "action"


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
