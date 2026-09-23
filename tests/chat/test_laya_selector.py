"""The selector's Laya tier: one noul per tool, one forward pass, still advisory.

The advisory property is the single most important behaviour in this change and
it is the one a port is most likely to lose, because a calibrated probability
looks authoritative in a way a JSON reply from an LLM does not. It is stated in
``graph.py``: what ``candidate_tools`` writes steers which tool the model
reaches for while leaving every registered tool bound and described, because a
shortlist that *removed* a tool would dead-end the turn that needed it -- with
no way for the model to recover, since it would not know the tool existed.

So the tests below assert the property from both ends: that a Laya failure
produces the same empty list an LLM failure produces, and that an empty list
means the act node binds the whole registry and is shown no narrowing message.

Everything here runs against ``FakeLayaPredictor`` with pinned answers. It has
no weights and its unpinned answers are a hash; nothing it returns is a
measurement of anything, which is exactly why pinning is how a test says what
it means.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aorta.chat.graph.nodes import TOOL_REGISTRY, _recommendation, selector_node
from aorta.chat.laya.questions import tool_question
from aorta.chat.tools.capabilities import MAX_CANDIDATES, describe_tools
from aorta.laya.predictor import NoulAnswer

_PROSE = "My training loss goes to NaN a few steps in. No crash, it just stops being a number."


def _described() -> dict[str, str]:
    return describe_tools(dict(TOOL_REGISTRY))


class ScriptedPredictor:
    """Answers each tool's noul with a pinned probability, and counts its passes.

    A recorder rather than ``FakeLayaPredictor`` for the tests that are about
    the *shape* of the call: the claim that N tools cost one forward pass is
    only checkable by something that can say how many it was asked for.
    """

    def __init__(self, probabilities: dict[str, float], *, default: float = 0.01):
        self._probabilities = probabilities
        self._default = default
        self.calls: list[tuple[list[str], list]] = []

    def model_id(self) -> str:
        return "scripted"

    def _for(self, question) -> float:
        for name, probability in self._probabilities.items():
            if tool_question(name, _described().get(name, "")).question == question.question:
                return probability
        return self._default

    def ask(self, states, questions):
        self.calls.append((list(states), list(questions)))
        return [[NoulAnswer(probability=self._for(q)) for q in questions] for _ in states]


@pytest.fixture()
def laya_on(monkeypatch):
    """Turn the tier on and install a predictor. See ``test_laya_router.py``."""
    from aorta.chat.config import settings

    monkeypatch.setattr(settings, "laya_enabled", True)

    def _install(predictor):
        monkeypatch.setattr("aorta.chat.laya.chat_predictor", lambda: predictor)
        return predictor

    return _install


async def _select(message: str = _PROSE) -> dict:
    return await selector_node({"messages": [HumanMessage(content=message)]})


# ── ranking ───────────────────────────────────────────────────────────────


class TestTheRanking:
    async def test_the_tools_come_back_best_first(self, laya_on):
        laya_on(ScriptedPredictor({"search_code": 0.7, "grep_code": 0.9, "read_file": 0.8}))
        out = await _select()
        assert out["candidate_tools"] == ["grep_code", "read_file", "search_code"]

    async def test_the_shortlist_is_capped(self, laya_on):
        laya_on(ScriptedPredictor(dict.fromkeys(_described(), 0.9)))
        out = await _select()
        assert len(out["candidate_tools"]) == MAX_CANDIDATES

    async def test_a_tool_below_the_threshold_is_not_recommended(self, laya_on, monkeypatch):
        """"If nothing fits, return fewer tools, or none" -- kept.

        A pure ranking always produces the top three, so without this the node
        would recommend a tool whose own probability says it is wrong.
        """
        from aorta.chat.config import settings

        monkeypatch.setattr(settings, "laya_selector_threshold", 0.5)
        laya_on(ScriptedPredictor({"grep_code": 0.9, "search_code": 0.49}))
        out = await _select()
        assert out["candidate_tools"] == ["grep_code"]

    async def test_nothing_above_the_threshold_recommends_nothing(self, laya_on):
        laya_on(ScriptedPredictor({}, default=0.01))
        out = await _select()
        assert out["candidate_tools"] == []
        assert "no tool reached" in out["selection_rationale"]

    async def test_the_rationale_names_the_model_and_the_probabilities(self, laya_on):
        """Decision 22 asks for the identity inline, not beside.

        It is also the only rationale this node can produce that carries none
        of the user's own text -- ``decision_log.py`` digests the LLM's version
        in summary mode precisely because that one quotes the question.
        """
        laya_on(ScriptedPredictor({"grep_code": 0.93}))
        rationale = (await _select())["selection_rationale"]
        assert "scripted" in rationale
        assert "p(grep_code)=0.93" in rationale
        assert _PROSE not in rationale


class TestTheBatchingShape:
    async def test_every_tool_is_one_question_in_one_forward_pass(self, laya_on):
        """The cost claim, asserted rather than described.

        M questions about one state share that state's encoding, so asking one
        question per registered tool is a single pass. Writing this as one
        state per tool would be N passes for the same answer, which is what
        keeps reranking thirty retrieved chunks off the interactive path.
        """
        predictor = laya_on(ScriptedPredictor({"grep_code": 0.9}))
        await _select()
        assert len(predictor.calls) == 1
        states, questions = predictor.calls[0]
        assert len(states) == 1
        assert len(questions) == len(_described())

    async def test_no_llm_is_consulted_when_laya_answers(self, laya_on):
        laya_on(ScriptedPredictor({"grep_code": 0.9}))
        with patch("aorta.chat.graph.nodes._get_llm") as get_llm:
            await _select()
        get_llm.assert_not_called()


# ── the advisory property ─────────────────────────────────────────────────


class TestItStaysAdvisory:
    """The invariant in ``graph.py``: a selector failure means every tool."""

    async def test_a_failing_predictor_falls_back_to_the_llm(self, laya_on):
        class Broken:
            def model_id(self):
                return "broken"

            def ask(self, states, questions):
                raise RuntimeError("the artifact and the manifest disagree")

        laya_on(Broken())
        reply = json.dumps({"tools": ["search_code"], "why": "the index has it"})
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content=reply))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=llm):
            out = await _select()
        assert out["candidate_tools"] == ["search_code"]

    async def test_both_tiers_failing_widens_to_every_tool(self, laya_on):
        """The two failures converge on one empty list, which is the whole design."""

        class Broken:
            def model_id(self):
                return "broken"

            def ask(self, states, questions):
                raise RuntimeError("no")

        laya_on(Broken())
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=ConnectionError("connection refused"))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=llm):
            out = await _select()
        assert out["candidate_tools"] == [], (
            "a failed ranking must widen to every tool, not narrow to a guess"
        )

    async def test_a_predictor_answering_the_wrong_shape_falls_back(self, laya_on):
        """A noul answered with a distribution is a contract break, not a ranking."""
        from aorta.laya.predictor import ChoiceAnswer

        class WrongShape:
            def model_id(self):
                return "wrong-shape"

            def ask(self, states, questions):
                answer = ChoiceAnswer(probabilities=(("a", 0.5), ("b", 0.5)))
                return [[answer for _ in questions] for _ in states]

        laya_on(WrongShape())
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="not json"))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=llm):
            out = await _select()
        assert out["candidate_tools"] == []

    async def test_a_predictor_returning_a_non_iterable_falls_back(self, laya_on):
        """The contract check must not be the thing that breaks the contract.

        ``ask`` promises one list of answers per state. A predictor returning
        a bare integer per state raised ``TypeError: 'int' object is not
        iterable`` straight out of ``selector_node``, because the
        ``isinstance`` sweep written to catch a broken predictor sat outside
        the guard that handles one.
        """

        class NotIterable:
            def model_id(self):
                return "not-iterable"

            def ask(self, states, questions):
                return [7 for _ in states]

        laya_on(NotIterable())
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="not json"))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=llm):
            out = await _select()
        assert out["candidate_tools"] == []

    async def test_a_predictor_answering_fewer_nouls_than_asked_falls_back(
        self, laya_on, caplog
    ):
        """Silent under-ranking, which is worse than a loud failure.

        ``zip`` truncates to the shorter of the two rather than raising, and
        the pairing stays aligned -- so a predictor answering one of nine
        nouls produced a confident single-tool shortlist with nothing anywhere
        saying the other eight were never scored. That reads as a decision.
        """
        import logging

        class Partial:
            def model_id(self):
                return "partial"

            def ask(self, states, questions):
                return [[NoulAnswer(probability=0.99)] for _ in states]

        laya_on(Partial())
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="not json"))
        with caplog.at_level(logging.WARNING, logger="aorta.chat.graph.nodes"):
            with patch("aorta.chat.graph.nodes._get_llm", return_value=llm):
                out = await _select()
        assert out["candidate_tools"] == []
        assert f"answered 1 of {len(_described())} tool nouls" in caplog.text

    async def test_a_predictor_that_cannot_name_itself_still_ranks(self, laya_on):
        """As in the router: the ranking is in hand, only the label is missing."""

        class Nameless(ScriptedPredictor):
            def model_id(self):
                raise RuntimeError("the manifest went away")

        laya_on(Nameless({"grep_code": 0.97}))
        out = await _select()
        assert out["candidate_tools"] == ["grep_code"]
        assert "unidentified predictor" in out["selection_rationale"]

    def test_an_empty_shortlist_shows_the_model_no_narrowing_message(self):
        """Half the invariant: nothing is said, so nothing is withdrawn."""
        assert _recommendation({"candidate_tools": [], "selection_rationale": ""}) == ""

    def test_a_shortlist_that_did_arrive_still_says_it_is_not_a_restriction(self):
        """The other half: even a confident ranking may not read as a gate."""
        text = _recommendation({"candidate_tools": ["grep_code"], "selection_rationale": ""})
        assert "not a restriction" in text
        assert "still the right call if the request needs it" in text

    async def test_the_act_node_binds_every_tool_whatever_the_selector_said(self, laya_on):
        """The end of the invariant, where it would actually bite.

        ``_act_native`` binds ``TOOL_REGISTRY`` and not the shortlist. Asserted
        against a Laya ranking that named exactly one tool, because the failure
        this guards is someone later deciding that a calibrated shortlist is
        good enough to bind instead -- at which point a turn needing the tool
        Laya ranked fourth has no way to reach it.
        """
        from aorta.chat.graph import nodes

        laya_on(ScriptedPredictor({"grep_code": 0.99}))
        out = await _select()
        assert out["candidate_tools"] == ["grep_code"]

        bound: dict = {}
        plain = MagicMock()

        def _bind_tools(tools):
            bound["tools"] = list(tools)
            return plain

        plain.bind_tools = _bind_tools
        plain.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
        with patch.object(nodes, "_get_llm", return_value=plain):
            await nodes._act_native(
                {
                    "messages": [HumanMessage(content=_PROSE)],
                    "candidate_tools": out["candidate_tools"],
                    "selection_rationale": out["selection_rationale"],
                    "retrieved_context": "",
                    "plan": "",
                    "iteration": 0,
                }
            )
        assert {tool.name for tool in bound["tools"]} == set(TOOL_REGISTRY)


# ── the one narrowing that is not a judgement ─────────────────────────────


class TestTheStructuralFilterStillApplies:
    async def test_a_tool_needing_a_paste_is_dropped_when_nothing_was_pasted(
        self, laya_on, monkeypatch
    ):
        """``enforce_requirements`` is unchanged and runs on both tiers.

        Declared against a base tool here rather than using the real
        source-reading tools, which need the ``[cia]`` extra: the rule under
        test is the filter, not which tools happen to declare it.
        """
        from aorta.chat.tools import capabilities

        monkeypatch.setattr(
            capabilities, "_REQUIRES", {"read_file": frozenset({"pasted_source"})}
        )
        laya_on(ScriptedPredictor({"read_file": 0.99, "grep_code": 0.8}))
        out = await _select(_PROSE)
        assert out["candidate_tools"] == ["grep_code"]

    async def test_dropping_is_explained_rather_than_silent(self, laya_on, monkeypatch):
        from aorta.chat.tools import capabilities

        monkeypatch.setattr(
            capabilities, "_REQUIRES", {"read_file": frozenset({"pasted_source"})}
        )
        laya_on(ScriptedPredictor({"read_file": 0.99}))
        rationale = (await _select(_PROSE))["selection_rationale"]
        assert "read_file" in rationale
        assert "nothing was pasted in this conversation" in rationale

    async def test_the_same_tool_survives_when_source_is_present(self, laya_on, monkeypatch):
        from aorta.chat.tools import capabilities

        monkeypatch.setattr(
            capabilities, "_REQUIRES", {"read_file": frozenset({"pasted_source"})}
        )
        laya_on(ScriptedPredictor({"read_file": 0.99}))
        pasted = "```\n__global__ void k(float* o) { __shared__ float s[64]; }\n```"
        out = await _select(pasted)
        assert out["candidate_tools"] == ["read_file"]
