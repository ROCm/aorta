"""The router's Laya tier: a threshold where a sentence used to be.

``ROUTER_PROMPT`` ends with "If in doubt, classify as action", which is a
calibration instruction written in English. The Laya tier replaces it with
:data:`~aorta.chat.laya.questions.DEFAULT_ROUTER_THRESHOLD`, which is a number
that can be re-derived against a corpus. Neither the number nor the model has
been measured -- the flag is off by default and these tests drive a fake
predictor with pinned answers, which is the only honest way to exercise a
threshold with no weights on the machine.

The two fallback constants are not tested here because the Laya path has no use
for them: there is no reply to fail to parse. Their existing tests still cover
the LLM path, which is unchanged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aorta.chat.graph.nodes import router_node
from aorta.chat.laya.questions import ROUTER_QUESTION
from aorta.laya.predictor import FakeLayaPredictor, NoulAnswer

_ASK = "why does my kernel produce NaN on the second iteration?"


@pytest.fixture()
def laya_on(monkeypatch):
    """Turn the tier on and hand the nodes whichever predictor a test wants.

    The factory is patched on ``aorta.chat.laya`` rather than on the node,
    because the node imports it inside the function body -- which is what keeps
    onnxruntime off the chat import graph, and which means the attribute is
    resolved at call time.
    """
    from aorta.chat.config import settings

    monkeypatch.setattr(settings, "laya_enabled", True)

    def _install(predictor):
        monkeypatch.setattr("aorta.chat.laya.chat_predictor", lambda: predictor)
        return predictor

    return _install


def _pinned(probability: float) -> FakeLayaPredictor:
    return FakeLayaPredictor(
        pinned={ROUTER_QUESTION.question: NoulAnswer(probability=probability)}
    )


async def _route(message: str = _ASK) -> dict:
    return await router_node({"messages": [HumanMessage(content=message)]})


# ── the tier answers ──────────────────────────────────────────────────────


class TestTheThresholdDecides:
    async def test_a_confident_yes_is_the_action_branch(self, laya_on):
        laya_on(_pinned(0.95))
        assert (await _route())["route"] == "action"

    async def test_a_confident_no_is_the_question_branch(self, laya_on):
        laya_on(_pinned(0.02))
        assert (await _route())["route"] == "question"

    async def test_at_the_threshold_counts_as_reaching_it(self, laya_on, monkeypatch):
        """``NoulAnswer.at`` is ``>=``, matching ``should_alert`` on the poll side.

        Pinned exactly at the boundary, because "at the threshold" meaning
        different things in two places is the kind of difference nobody finds
        by reading.
        """
        from aorta.chat.config import settings

        monkeypatch.setattr(settings, "laya_router_threshold", 0.4)
        laya_on(_pinned(0.4))
        assert (await _route())["route"] == "action"

    async def test_the_threshold_is_what_moves_the_answer(self, laya_on, monkeypatch):
        """The whole claim: the decision is a number, not a prompt."""
        from aorta.chat.config import settings

        laya_on(_pinned(0.3))
        monkeypatch.setattr(settings, "laya_router_threshold", 0.2)
        assert (await _route())["route"] == "action"
        monkeypatch.setattr(settings, "laya_router_threshold", 0.6)
        assert (await _route())["route"] == "question"

    async def test_the_default_threshold_keeps_an_uncertain_turn_on_the_action_branch(
        self, laya_on
    ):
        """"If in doubt, classify as action", ported.

        An action misrouted to the question branch reaches a node with no tool
        access and dead-ends the turn; the reverse costs a plan call and still
        answers.
        """
        laya_on(_pinned(0.45))
        assert (await _route())["route"] == "action"

    async def test_no_llm_is_consulted_when_laya_answers(self, laya_on):
        """The point of the tier: two remote round-trips become two local passes."""
        laya_on(_pinned(0.9))
        with patch("aorta.chat.graph.nodes._get_llm") as get_llm:
            assert (await _route())["route"] == "action"
        get_llm.assert_not_called()


# ── falling back ──────────────────────────────────────────────────────────


class TestItFallsBackToTheLLM:
    async def test_the_tier_is_off_by_default(self):
        """Nothing changes for anyone who has not opted in."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="question"))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=llm):
            assert (await _route())["route"] == "question"
        llm.ainvoke.assert_awaited_once()

    async def test_a_predictor_that_cannot_load_falls_back(self, monkeypatch):
        """No artifact staged is an ordinary state, not an error."""
        from aorta.chat.config import settings
        from aorta.chat.laya.artifact import ArtifactUnavailableError

        monkeypatch.setattr(settings, "laya_enabled", True)

        def _no_artifact():
            raise ArtifactUnavailableError("nothing staged")

        monkeypatch.setattr("aorta.chat.laya.chat_predictor", _no_artifact)
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="action"))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=llm):
            assert (await _route())["route"] == "action"

    async def test_a_predictor_that_raises_mid_question_falls_back(self, laya_on):
        """A staged artifact that turns out to be unreadable must not cost the turn."""

        class Broken:
            def model_id(self):
                return "broken"

            def ask(self, states, questions):
                raise RuntimeError("the graph and the manifest disagree")

        laya_on(Broken())
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="question"))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=llm):
            assert (await _route())["route"] == "question"

    async def test_a_predictor_answering_the_wrong_shape_falls_back(self, laya_on):
        """A noul answered with a distribution is a contract break, not a verdict.

        ``ask_noul`` raises a ``TypeError`` for this, and the node treats it
        like any other unavailability rather than letting it reach the user.
        """
        from aorta.laya.predictor import ChoiceAnswer

        laya_on(
            FakeLayaPredictor(
                pinned={
                    ROUTER_QUESTION.question: ChoiceAnswer(
                        probabilities=(("question", 0.4), ("action", 0.6))
                    )
                }
            )
        )
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="question"))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=llm):
            assert (await _route())["route"] == "question"


class TestWhatItSaysAboutItself:
    async def test_the_log_names_the_model_and_the_probability(self, laya_on, caplog):
        """Decision 22: a verdict that cannot name what produced it is not evidence."""
        import logging

        laya_on(_pinned(0.91))
        with caplog.at_level(logging.INFO, logger="aorta.chat.graph.nodes"):
            await _route()
        assert "fake" in caplog.text
        assert "p(action)=0.91" in caplog.text

    async def test_a_predictor_that_cannot_name_itself_still_routes(self, laya_on, caplog):
        """The dead turn the fallback was written to prevent, arriving late.

        ``model_id`` was called outside the ``try``, while building this very
        log line -- so a predictor whose ``ask`` succeeded and whose
        ``model_id`` raised took the turn down after the answer was already in
        hand. Degraded rather than fallen back: the verdict is good, and
        paying for a remote round-trip because a label would not format is the
        wrong trade.
        """
        import logging

        class Nameless(FakeLayaPredictor):
            def model_id(self):
                raise RuntimeError("the manifest went away")

        laya_on(Nameless(pinned={ROUTER_QUESTION.question: NoulAnswer(probability=0.95)}))
        with caplog.at_level(logging.INFO, logger="aorta.chat.graph.nodes"):
            assert (await _route())["route"] == "action"
        # Said, not swallowed. A report must never attribute a number to
        # weights that did not produce it, and "unidentified" is the only
        # honest thing to attribute it to here.
        assert "unidentified predictor" in caplog.text
        assert "RuntimeError" in caplog.text
