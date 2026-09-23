"""One definition of each question, and no high-cardinality choice anywhere.

Two properties, both of which would otherwise be eroded by somebody acting
reasonably.

**Train/serve skew by duplicated question text.** Track B found the corpus
builder under ``src/aorta/laya/corpus/`` had independently written its own
phrasing of the questions the inference path asks. Nothing failed: fine-tuning
on one wording and asking another just answers slightly worse, for a reason
nobody would go looking for. A future chat corpus builder is the obvious place
for that to happen again, so the guard below fails if either question's text
exists as a string literal in more than one module.

**The calibration cliff at eleven options.** Laya 0.3.5 clamps fitted
temperatures into [0.5, 5.0] because the shipped ``choice:11+`` bucket is
0.1006 -- below 1.0 sharpens logits rather than softening them, so that bucket
multiplies them roughly tenfold and publishes a 0.24 top probability as 0.99.
The plan's stated reason for N independent nouls in the selector is the
``head_max_len`` budget split; avoiding that bucket is the second reason, and
the brief asks for it to be verified rather than assumed. It is verified here
against the live registry, so it stays true as tools are added.

Pure AST and stdlib, so this costs nothing and needs no model.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aorta.chat.laya.artifact import calibration_key
from aorta.chat.laya.questions import (
    DEFAULT_ROUTER_THRESHOLD,
    DEFAULT_SELECTOR_THRESHOLD,
    ROUTER_QUESTION,
    SELECTOR_QUESTION_TEMPLATE,
    tool_question,
)
from aorta.laya.predictor import Noul

_SRC = Path(__file__).resolve().parents[2] / "src" / "aorta"
_QUESTIONS_MODULE = _SRC / "chat" / "laya" / "questions.py"

#: Where Laya's own high-cardinality calibration gets bad, per the model card
#: and the ``[laya]`` floor's reasoning in ``pyproject.toml``.
_CHOICE_CLIFF = 11


def _string_constants(path: Path) -> set[str]:
    """Every string literal in *path*, whitespace-normalised.

    Via the AST rather than a text search, because the interesting case is a
    question written across several source lines: Python folds adjacent string
    literals into one constant at parse time, so a copy that happens to wrap
    differently is still found. Normalising whitespace covers a copy that was
    re-flowed by a formatter.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):  # pragma: no cover - nothing in-tree hits this
        return set()
    return {
        " ".join(node.value.split())
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _modules_containing(text: str) -> list[Path]:
    wanted = " ".join(text.split())
    return [
        path
        for path in sorted(_SRC.rglob("*.py"))
        if "__pycache__" not in path.parts and wanted in _string_constants(path)
    ]


class TestEachQuestionIsWrittenOnce:
    @pytest.mark.parametrize(
        "text",
        [
            ROUTER_QUESTION.question,
            ROUTER_QUESTION.when_true,
            ROUTER_QUESTION.when_false,
            SELECTOR_QUESTION_TEMPLATE,
        ],
        ids=["router-question", "router-true", "router-false", "selector-template"],
    )
    def test_it_appears_in_exactly_one_module(self, text: str):
        found = _modules_containing(text)
        assert found == [_QUESTIONS_MODULE], (
            "this question's text is written in more than one place: "
            f"{[str(p) for p in found]}. Import it from "
            "aorta.chat.laya.questions instead. Asking one wording and "
            "fine-tuning on another does not fail, it answers worse."
        )


class TestNothingHereIsAWideChoice:
    """The selector is per-tool nouls, and this is what keeps it that way."""

    def test_the_router_asks_a_noul(self):
        assert isinstance(ROUTER_QUESTION, Noul)
        assert calibration_key(ROUTER_QUESTION) == "noul:2"

    def test_every_tool_question_is_a_noul(self):
        assert calibration_key(tool_question("search_code", "searches code")) == "noul:2"

    def test_every_question_the_live_registry_produces_is_a_noul(self):
        """Asserted against the registry, so it stays true as tools are added."""
        from aorta.chat.graph.nodes import TOOL_REGISTRY
        from aorta.chat.tools.capabilities import describe_tools

        described = describe_tools(dict(TOOL_REGISTRY))
        assert described, "no tool carries a description, so the selector has nothing to ask"
        questions = [tool_question(name, text) for name, text in described.items()]
        assert {calibration_key(question) for question in questions} == {"noul:2"}

    def test_a_single_choice_would_cross_the_cliff_on_a_full_install(self):
        """The counterfactual, and the reason it is stated conditionally.

        One ``choice`` over the registered tools is the obvious way to write a
        selector. Measured here rather than asserted in prose, because the
        answer depends on the install: ``[chat-cli]`` alone does not reach
        eleven, and adding the ``[cia]`` diagnostic tools does. A defect that
        is real for some users and not others is the worst kind to reason
        about from a docstring, so the numbers are checked.
        """
        from aorta.chat.graph.nodes import TOOL_REGISTRY
        from aorta.chat.plugins import _DIAGNOSTIC_TOOL_NAMES
        from aorta.chat.tools.capabilities import describe_tools
        from aorta.laya.predictor import Choice

        described = describe_tools(dict(TOOL_REGISTRY))
        full_surface = sorted(set(described) | set(_DIAGNOSTIC_TOOL_NAMES))
        assert len(full_surface) >= _CHOICE_CLIFF, (
            f"a full install now offers {len(full_surface)} tools, below the "
            "eleven-option cliff. The per-tool-noul shape is still right for the "
            "head_max_len reason, but the calibration argument in "
            "aorta/chat/laya/questions.py no longer applies and should be "
            "reworded rather than left overstated."
        )
        as_one_choice = Choice(question="which tool?", options=tuple(full_surface))
        assert calibration_key(as_one_choice) == f"choice:{len(full_surface)}"

    def test_the_nouls_are_the_same_bucket_on_every_install(self):
        """Which is the property the choice does not have."""
        for count in (9, 14, 40):
            names = [f"tool_{index}" for index in range(count)]
            questions = [tool_question(name, "does something") for name in names]
            assert {calibration_key(question) for question in questions} == {"noul:2"}


class TestTheThresholds:
    """Neither is a calibration figure. Both have to be readable as policy."""

    def test_the_router_prefers_action_when_in_doubt(self):
        """``ROUTER_PROMPT``'s closing sentence, expressed as a number.

        An action sent to the question branch reaches a node with no tool
        access and dead-ends the turn; a question sent to the action branch
        costs a plan call and still answers. The threshold has to sit below the
        midpoint or that asymmetry is lost in the port.
        """
        assert DEFAULT_ROUTER_THRESHOLD < 0.5

    def test_the_selector_can_still_recommend_nothing(self):
        """"If nothing fits, return fewer tools, or none" -- the prompt's own rule.

        A pure ranking always produces the top three, so a question no tool
        suits would come back with three recommendations whose best entry the
        model thought was wrong.
        """
        assert 0.0 < DEFAULT_SELECTOR_THRESHOLD <= 1.0
