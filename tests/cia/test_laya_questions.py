"""One definition of each question the CIA tracks ask Laya.

The CIA counterpart of ``tests/chat/test_laya_questions.py``, and it exists
because the property it guards has now been broken four times in one effort --
the proposer, Watch, chat, and the log finder -- each found by hand, each after
the code was written. Three of those were found and fixed with nothing left
behind to stop the fourth.

**What goes wrong.** A Laya probability is calibrated by a temperature fitted per
question type and option count against a corpus of question/answer pairs. A
corpus labelled against one wording and an inference path asking another is a
fit made against a question nobody asks, and the number that comes back is
calibrated against nothing while still looking like a probability. Nothing
raises. The gate just answers slightly worse, for a reason nobody would go
looking for.

**The settled direction** is the one the proposer and chat took: the question
belongs to the module that owns the decision, and the corpus builder imports it.
The seam in ``aorta.laya.predictor`` deliberately holds none of them -- it serves
three tracks, and a string registry is what it would become if each of them put
its phrasing there.

**Watch used to be the exception and no longer is.** Its two copies were pinned
to each other by an equality test rather than reduced to one, because
``aorta.laya.corpus`` must stay importable without the ``[cia]`` extra and
``aorta.cia.watch.watcher`` imports dspy at module scope. The corpus half is now
gone: it calls ``watcher.healthy_question()`` and ``signal_question()`` lazily,
inside the function that needs them, which keeps the extra optional without
keeping a second copy. So every CIA decision holds the same write-once property
and this file needs no special case for any of them -- see
:data:`_WATCH_QUESTION_HOME`.

Pure AST and stdlib, so this costs nothing and needs no model.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import pytest

from aorta.agent.llm import (
    LAYA_CATEGORY_QUESTION,
    LAYA_MITIGATION_QUESTION,
    LAYA_STOP_QUESTION,
    LAYA_STOP_WHEN_FALSE,
    LAYA_STOP_WHEN_TRUE,
)
from aorta.cia.watch.log_finder import (
    LAYA_USEFUL_QUESTION_TEMPLATE,
    LAYA_USEFUL_WHEN_FALSE,
    LAYA_USEFUL_WHEN_TRUE,
)
from aorta.cia.watch.watcher import healthy_question, signal_question
from aorta.laya.corpus.proposer import (
    LAYA_CANDIDATE_QUESTION_TEMPLATE,
    LAYA_CANDIDATE_WHEN_FALSE,
    LAYA_CANDIDATE_WHEN_TRUE,
)

_SRC = Path(__file__).resolve().parents[2] / "src" / "aorta"

_PROPOSER = _SRC / "agent" / "llm.py"
_LOG_FINDER = _SRC / "cia" / "watch" / "log_finder.py"
_WATCHER = _SRC / "cia" / "watch" / "watcher.py"
_CORPUS_PROPOSER = _SRC / "laya" / "corpus" / "proposer.py"
_CORPUS_WATCH = _SRC / "laya" / "corpus" / "watch.py"
_SEAM = _SRC / "laya" / "predictor.py"

#: Watch's question text used to live in ``_WATCHER`` *and* ``_CORPUS_WATCH``, as a
#: pinned pair with an equality test holding the two halves together. The corpus
#: half is gone: it now calls ``watcher.healthy_question()`` and
#: ``signal_question()`` lazily, so Watch has the same write-once property as every
#: other CIA decision and joins the class below.
#:
#: The pin was the best available guarantee while the copy existed and it could
#: never cover the case that actually spread this defect -- a *third* copy, which
#: passes an equality test by agreeing with both. Removing the copy is what removes
#: the class of bug rather than detecting it.
_WATCH_QUESTION_HOME = _WATCHER


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


@lru_cache(maxsize=1)
def _constants_by_module() -> tuple[tuple[Path, frozenset[str]], ...]:
    """Every module's literals, parsed once.

    Cached because the sweep below asks about every question it finds and the
    parametrised cases ask about twenty more; without this each lookup reparses
    the whole package and the file takes half a minute to say nothing happened.
    """
    return tuple(
        (path, frozenset(_string_constants(path)))
        for path in sorted(_SRC.rglob("*.py"))
        if "__pycache__" not in path.parts
    )


def _source_files() -> list[Path]:
    return [path for path, _constants in _constants_by_module()]


def _modules_containing(text: str) -> list[Path]:
    wanted = " ".join(text.split())
    return [path for path, constants in _constants_by_module() if wanted in constants]


class TestEachQuestionIsWrittenOnce:
    """The strong property, and the one every CIA decision now has."""

    @pytest.mark.parametrize(
        ("text", "home"),
        [
            (LAYA_MITIGATION_QUESTION, _PROPOSER),
            (LAYA_CATEGORY_QUESTION, _PROPOSER),
            (LAYA_STOP_QUESTION, _PROPOSER),
            (LAYA_STOP_WHEN_TRUE, _PROPOSER),
            (LAYA_STOP_WHEN_FALSE, _PROPOSER),
            # The per-candidate noul lives with the builder rather than with
            # the proposer, because the builder is its first writer -- Track B
            # has not implemented the shape yet, and writing the question into
            # the proposer before anything was labelled against it is the
            # sequencing that has Phase 1 scoring a question nothing was trained
            # on. ``aorta/laya/corpus/proposer.py`` records the follow-up.
            (LAYA_CANDIDATE_QUESTION_TEMPLATE, _CORPUS_PROPOSER),
            (LAYA_CANDIDATE_WHEN_TRUE, _CORPUS_PROPOSER),
            (LAYA_CANDIDATE_WHEN_FALSE, _CORPUS_PROPOSER),
            (LAYA_USEFUL_QUESTION_TEMPLATE, _LOG_FINDER),
            (LAYA_USEFUL_WHEN_TRUE, _LOG_FINDER),
            (LAYA_USEFUL_WHEN_FALSE, _LOG_FINDER),
            # Watch, now that the corpus half is gone. Taken off the assemblers
            # rather than off the five constants, because the assemblers are what a
            # caller imports and the constants exist so this sweep has one spelling
            # to find.
            (healthy_question().question, _WATCH_QUESTION_HOME),
            (healthy_question().when_true, _WATCH_QUESTION_HOME),
            (healthy_question().when_false, _WATCH_QUESTION_HOME),
            (signal_question().question, _WATCH_QUESTION_HOME),
        ],
        ids=[
            "proposer-mitigation",
            "proposer-category",
            "proposer-stop",
            "proposer-stop-true",
            "proposer-stop-false",
            "proposer-candidate-template",
            "proposer-candidate-true",
            "proposer-candidate-false",
            "log-finder-template",
            "log-finder-true",
            "log-finder-false",
            "watch-healthy-question",
            "watch-healthy-true",
            "watch-healthy-false",
            "watch-signal-question",
        ],
    )
    def test_it_appears_in_exactly_one_module(self, text: str, home: Path):
        found = _modules_containing(text)
        assert found == [home], (
            "this question's text is written in more than one place: "
            f"{[str(path) for path in found]}. Import it from "
            f"{home.relative_to(_SRC.parent.parent)} instead. Asking one wording "
            "and fitting a temperature on another does not fail, it answers worse."
        )


class TestWatchHasOneHomeNow:
    """What removing the corpus copy bought, asserted rather than assumed.

    The glosses get their own case because they are the part most likely to come
    back: a caller wanting "the signal question" is tempted to rebuild it from
    ``WATCH_SIGNALS`` and a fresh list of descriptions, and a rebuilt criteria
    mapping is how an option order or a missing gloss creeps in. The criteria keys
    *are* the answer space, and ``ChoiceAnswer`` re-reads a distribution in the
    order offered.
    """

    @pytest.mark.parametrize(
        "text",
        [gloss for _option, gloss in signal_question().criteria],
        ids=[f"signal-gloss-{option}" for option, _gloss in signal_question().criteria],
    )
    def test_each_gloss_is_written_once(self, text: str):
        assert _modules_containing(text) == [_WATCH_QUESTION_HOME]

    def test_the_corpus_builder_holds_no_copy(self):
        """The removal itself. Every string, including the glosses."""
        every = [
            healthy_question().question,
            healthy_question().when_true,
            healthy_question().when_false,
            signal_question().question,
            *[gloss for _option, gloss in signal_question().criteria],
        ]
        offenders = [text for text in every if _CORPUS_WATCH in _modules_containing(text)]
        assert not offenders, (
            f"{_CORPUS_WATCH.name} has written Watch's question text again: "
            f"{offenders}. It calls watcher.healthy_question() and signal_question() "
            "instead, which is what gives Watch the write-once property the class "
            "above tests for every other decision."
        )

    def test_the_builder_labels_against_the_assembled_questions(self):
        """The end state of the drift guard: the two are now the same object.

        A tautology, and the right kind. It fails the moment the corpus goes back
        to building its own.
        """
        from aorta.laya.corpus.watch import watch_questions

        assert watch_questions() == (healthy_question(), signal_question())


class TestNoNewQuestionIsWrittenTwice:
    """The sweep, for a question nobody thought to add to the lists above.

    Every named question in this file was added to it by hand, which is the same
    process that missed the defect three times. This finds a literal
    ``Noul(question=...)`` or ``Choice(question=...)`` whose text is also written
    somewhere else, whether or not anyone remembered to name it here.

    Literal keywords only. A question built from a constant, an f-string or a
    ``.format`` call is invisible to this, which is a limitation and not a hole:
    those are the shapes that already have one definition to be built from.
    """

    @staticmethod
    def _inline_question_text(path: Path) -> set[str]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):  # pragma: no cover
            return set()
        found: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else ""
            if name not in ("Noul", "Choice"):
                continue
            for keyword in node.keywords:
                if keyword.arg not in ("question", "when_true", "when_false"):
                    continue
                if isinstance(keyword.value, ast.Constant) and isinstance(
                    keyword.value.value, str
                ):
                    found.add(" ".join(keyword.value.value.split()))
        return found

    def test_every_inline_question_has_one_home(self):
        homes: dict[str, list[Path]] = {}
        for path in _source_files():
            for text in self._inline_question_text(path):
                homes.setdefault(text, []).extend(_modules_containing(text))

        # No exemption. This carried one for Watch's sanctioned pair; the pair is
        # gone, so every question in the tree now has to have exactly one home and
        # the sweep can say so without a special case.
        duplicated = {
            text: sorted(set(paths))
            for text, paths in homes.items()
            if len(set(paths)) > 1
        }
        assert not duplicated, (
            "these question texts are written in more than one module: "
            f"{ {text: [str(p) for p in paths] for text, paths in duplicated.items()} }. "
            "Define each in the module that owns the decision and import it."
        )


class TestTheSeamHoldsNoQuestionText:
    """``aorta.laya.predictor`` serves three tracks and must not collect phrasings.

    The chat side keeps its questions in ``aorta/chat/laya/questions.py`` and the
    CIA side keeps each beside its decision, and the thing both arrangements have
    in common is that neither puts them in the seam. Moving one there looks like
    de-duplication and is how the seam becomes a registry that every track has to
    edit.
    """

    @pytest.mark.parametrize(
        "text",
        [
            LAYA_MITIGATION_QUESTION,
            LAYA_STOP_QUESTION,
            LAYA_CANDIDATE_QUESTION_TEMPLATE,
            LAYA_USEFUL_QUESTION_TEMPLATE,
            healthy_question().question,
            signal_question().question,
        ],
        ids=[
            "proposer-mitigation",
            "proposer-stop",
            "proposer-candidate",
            "log-finder",
            "watch-healthy",
            "watch-signal",
        ],
    )
    def test_the_seam_does_not_hold_it(self, text: str):
        assert _SEAM not in _modules_containing(text)
