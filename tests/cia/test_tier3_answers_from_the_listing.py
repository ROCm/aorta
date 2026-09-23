"""Tier 3 of the log finder may only name a file the listing actually held.

``_within_job`` stopped a path that escaped the job. It did not stop a path
that stayed inside the job and was never in the directory listing the model was
shown, because the three checks between an answer and being watched were
containment, ``is_file()`` and the exclude list -- and a real file the model
never saw passes all three. The answer space is now derived from the listing
itself, so there is nothing to invent from.

That property belongs to the tier, not to whichever engine ranks it, which is
why it is asserted here against both the DSPy tier and the Laya tier. The Laya
tier ships disabled; if the containment had landed inside its flag, the shipped
default would still have been the hazardous one.

Two things this file deliberately does not do. It never requires weights: the
Laya tier is driven through ``FakeLayaPredictor``, whose answers are a hash of
the question and are stable, arbitrary, and not predictions. And it asserts no
accuracy, because nothing in this repository has measured this decision -- which
is also why ``laya.enabled`` is false in the shipped config.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("dspy", reason="the log finder needs the [cia] extra")

from aorta.cia.watch.log_finder import (
    DEFAULT_LAYA_BACKEND,
    DEFAULT_LAYA_MIN_PROBABILITY,
    LogFinder,
    _dir_listing,
    useful_question,
)
from aorta.laya.predictor import (
    ChoiceAnswer,
    FakeLayaPredictor,
    LayaUnavailable,
    NoulAnswer,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = ROOT / "src" / "aorta" / "cia" / "watch" / "watch_config.yaml"
SOURCE = ROOT / "src" / "aorta" / "cia" / "watch" / "log_finder.py"

#: The six files every job directory below holds. Six rather than three,
#: because tier 2 returns directly when it finds three or fewer known logs and
#: tier 3 is never reached.
_LISTED = tuple(f"part{n}.log" for n in range(6))


@pytest.fixture()
def job(tmp_path: Path) -> Path:
    """A job directory ambiguous enough to reach tier 3, plus one file it hides.

    ``notes.dat`` is the interesting one. It is a real file, it is genuinely
    inside the job, and it is not on the exclude list -- so every check that
    used to guard tier 3 accepts it. It is absent from the listing for the
    ordinary reason a file is: ``_dir_listing`` walks ``-maxdepth 4`` and this
    sits one level deeper. It is also absent from the extension scan, so a test
    that finds it in the result knows tier 3 put it there.
    """
    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True)
    for name in _LISTED:
        (job_dir / name).write_text("step=1 loss=0.5\n" * 20, encoding="utf-8")
    deep = job_dir / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "notes.dat").write_text("private\n" * 40, encoding="utf-8")
    return job_dir


def _answering(monkeypatch, finder: LogFinder, files: list[str]) -> None:
    """Make the DSPy tier's model return *files*."""

    class Pred:
        relevant_files = files

    monkeypatch.setattr(finder, "_discovery", lambda **_kwargs: Pred())


def _refusing(monkeypatch, finder: LogFinder) -> None:
    """Fail loudly if the DSPy tier is consulted at all."""

    def _no(**_kwargs):
        raise AssertionError("the LLM tier ran while the Laya tier was enabled")

    monkeypatch.setattr(finder, "_discovery", _no)


def _pinned(**probabilities: float) -> FakeLayaPredictor:
    """A predictor with a fixed answer for each named file.

    Keyed through ``useful_question`` rather than through a question string
    written out here, so a reworded question fails the assertions in this file
    rather than silently falling back to the fake's hashed answers -- which
    would still be numbers, and would still rank.
    """
    return FakeLayaPredictor(
        pinned={
            useful_question(label.replace("_", ".")).question: NoulAnswer(probability=value)
            for label, value in probabilities.items()
        }
    )


def _laya(enabled: bool = True, **extra) -> dict:
    return {"extensions": [".log"], "laya": {"enabled": enabled, **extra}}


class TestTheAnswerSpaceIsTheListing:
    """The security property, asserted against both engines."""

    def test_the_old_barriers_would_have_let_the_hidden_file_through(self, job):
        """Establishes that this test is about the new check and not the old one."""
        finder = LogFinder(config={})
        resolved = finder._within_job("a/b/c/d/notes.dat", job)

        assert resolved is not None, "containment accepts it"
        assert resolved.is_file(), "is_file() accepts it"
        assert not finder._excluded(resolved), "the exclude list accepts it"
        assert "notes.dat" not in _dir_listing(job), "and it is not in the listing"

    def test_the_llm_tier_cannot_name_a_file_the_listing_omitted(self, job, monkeypatch):
        finder = LogFinder(config={"extensions": [".log"]})
        _answering(monkeypatch, finder, ["a/b/c/d/notes.dat"])

        found = finder.find(job)

        assert all(p.name != "notes.dat" for p in found), found

    def test_the_llm_tier_still_gets_a_file_the_listing_held(self, job, monkeypatch):
        """Binding the answer space must not mean discovery stops working."""
        finder = LogFinder(config={"extensions": [".log"]})
        _answering(monkeypatch, finder, ["part3.log"])

        assert [p.name for p in finder.find(job)] == ["part3.log"]

    def test_the_llm_tier_does_not_return_one_file_twice(self, job, monkeypatch):
        """A repeated answer used to become a repeated watched file and cursor."""
        finder = LogFinder(config={"extensions": [".log"]})
        _answering(monkeypatch, finder, ["part3.log", str(job / "part3.log")])

        assert [p.name for p in finder.find(job)] == ["part3.log"]

    def test_the_laya_tier_is_only_ever_offered_the_listing(self, job, monkeypatch):
        """It cannot name the hidden file because it is never asked about it."""
        predictor = _Recording()
        finder = LogFinder(config=_laya(), predictor=predictor)
        _refusing(monkeypatch, finder)

        finder.find(job)

        asked = predictor.calls[0].questions
        assert len(asked) == len(_LISTED)
        assert all(any(name in question for name in _LISTED) for question in asked)
        assert not any("notes.dat" in question for question in asked)


class _Recording(FakeLayaPredictor):
    """``FakeLayaPredictor`` that remembers what it was asked.

    Subclassed rather than written from scratch for the reason the predictor
    tests subclass it: the answers stay the reference implementation's, so a
    change to the contract fails here too.
    """

    class Call:
        def __init__(self, states, questions):
            self.states = list(states)
            self.questions = [q.question for q in questions]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls: list[_Recording.Call] = []

    def ask(self, states, questions):
        self.calls.append(_Recording.Call(states, questions))
        return super().ask(states, questions)


class TestTheShapeOfTheCall:
    def test_one_directory_costs_one_forward_pass(self, job, monkeypatch):
        """One state, N questions.

        The contract is that questions about one state share its encoding and
        only a second *state* costs a second pass. A per-file state would make
        a sixty-entry directory sixty passes inside the poll loop.
        """
        predictor = _Recording()
        finder = LogFinder(config=_laya(), predictor=predictor)
        _refusing(monkeypatch, finder)

        finder.find(job)

        assert len(predictor.calls) == 1
        assert predictor.calls[0].states == [_dir_listing(job)]

    def test_every_entry_gets_its_own_question(self, job, monkeypatch):
        """Two files sharing a question text would share one answer."""
        predictor = _Recording()
        finder = LogFinder(config=_laya(), predictor=predictor)
        _refusing(monkeypatch, finder)

        finder.find(job)

        assert len(set(predictor.calls[0].questions)) == len(_LISTED)


class TestRanking:
    def test_files_come_back_most_probable_first(self, job, monkeypatch):
        finder = LogFinder(
            config=_laya(),
            predictor=_pinned(part0_log=0.6, part1_log=0.9, part2_log=0.7,
                              part3_log=0.1, part4_log=0.2, part5_log=0.3),
        )
        _refusing(monkeypatch, finder)

        assert [p.name for p in finder.find(job)] == [
            "part1.log", "part2.log", "part0.log",
        ]

    def test_max_files_still_caps_the_answer(self, job, monkeypatch):
        finder = LogFinder(
            config={**_laya(), "max_files": 2},
            predictor=_pinned(part0_log=0.6, part1_log=0.9, part2_log=0.7,
                              part3_log=0.8, part4_log=0.99, part5_log=0.55),
        )
        _refusing(monkeypatch, finder)

        assert [p.name for p in finder.find(job)] == ["part4.log", "part1.log"]

    def test_a_file_below_the_threshold_is_not_watched(self, job, monkeypatch):
        finder = LogFinder(
            config=_laya(min_probability=0.8),
            predictor=_pinned(part0_log=0.6, part1_log=0.9, part2_log=0.7,
                              part3_log=0.1, part4_log=0.2, part5_log=0.85),
        )
        _refusing(monkeypatch, finder)

        assert [p.name for p in finder.find(job)] == ["part1.log", "part5.log"]

    def test_at_the_threshold_counts_as_reaching_it(self, job, monkeypatch):
        """``NoulAnswer.at`` uses ``>=``, matching ``should_alert``."""
        finder = LogFinder(
            config=_laya(min_probability=0.7),
            predictor=_pinned(part0_log=0.7, part1_log=0.1, part2_log=0.1,
                              part3_log=0.1, part4_log=0.1, part5_log=0.1),
        )
        _refusing(monkeypatch, finder)

        assert [p.name for p in finder.find(job)] == ["part0.log"]


class TestTheFallbacksSurvive:
    """``_scan_by_extension`` is still the last thing standing."""

    def test_a_directory_the_classifier_rejects_falls_back_to_the_scan(
        self, job, monkeypatch
    ):
        finder = LogFinder(
            config=_laya(),
            predictor=_pinned(part0_log=0.1, part1_log=0.1, part2_log=0.1,
                              part3_log=0.1, part4_log=0.1, part5_log=0.1),
        )
        _refusing(monkeypatch, finder)

        assert finder.find(job) == finder._scan_by_extension(job)[:8]

    def test_an_unavailable_checkpoint_falls_back_and_says_so(
        self, job, monkeypatch, capsys
    ):
        """Silence here would read as "discovery got worse", not "stage the weights"."""
        finder = LogFinder(config=_laya(), predictor=_Unavailable())
        _refusing(monkeypatch, finder)

        found = finder.find(job)

        assert found == finder._scan_by_extension(job)[:8]
        assert "LayaUnavailable" in capsys.readouterr().out

    def test_it_is_not_retried_for_every_later_job(self, job, monkeypatch):
        """One finder serves the whole poll loop; a missing checkpoint stays missing."""
        predictor = _Unavailable()
        finder = LogFinder(config=_laya(), predictor=predictor)
        _refusing(monkeypatch, finder)

        finder.find(job)
        finder.find(job)

        assert predictor.attempts == 1

    def test_a_predictor_answering_the_wrong_shape_is_not_read_as_a_score(
        self, job, monkeypatch, capsys
    ):
        """A choice where a noul was asked has no p(yes) to threshold."""
        finder = LogFinder(config=_laya(), predictor=_AnsweringChoices())
        _refusing(monkeypatch, finder)

        assert finder.find(job) == finder._scan_by_extension(job)[:8]
        assert "answered a noul" in capsys.readouterr().out

    def test_an_unknown_backend_falls_back_rather_than_raising(
        self, job, monkeypatch, capsys
    ):
        """Discovery failing costs the extension scan's answer, never the poll loop."""
        finder = LogFinder(config=_laya(backend="not-a-checkpoint"))
        _refusing(monkeypatch, finder)

        assert finder.find(job) == finder._scan_by_extension(job)[:8]
        assert "not-a-checkpoint" in capsys.readouterr().out

    def test_the_fake_backend_is_refused_by_name(self, job, monkeypatch, capsys):
        """A hash choosing which files Watch tails, and which roots its tools get.

        Refused rather than resolved, as ``watch.laya`` refuses it. A test that
        wants the fake passes it through ``predictor=``, which is what every
        test above does.
        """
        finder = LogFinder(config=_laya(backend="fake"))
        _refusing(monkeypatch, finder)

        assert finder.find(job) == finder._scan_by_extension(job)[:8]
        assert "ranks by hash" in capsys.readouterr().out

    def test_an_unreadable_listing_spends_nothing(self, tmp_path, monkeypatch):
        """No listing means no candidates, so neither engine is consulted."""
        predictor = _Recording()
        finder = LogFinder(config=_laya(), predictor=predictor)
        _refusing(monkeypatch, finder)

        assert finder.find(tmp_path / "gone") == []
        assert predictor.calls == []


class _Unavailable(FakeLayaPredictor):
    """A staged-weights failure, counted so a retry shows up as one."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def ask(self, states, questions):
        self.attempts += 1
        raise LayaUnavailable("no checkpoint on this node")


class _AnsweringChoices(FakeLayaPredictor):
    def ask(self, states, questions):
        return [[ChoiceAnswer(probabilities=(("yes", 1.0),)) for _ in questions] for _ in states]


class TestItShipsOff:
    @pytest.fixture(scope="class")
    def config(self) -> dict:
        return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["log_finder"]["laya"]

    def test_the_flag_is_false_in_the_shipped_config(self, config):
        """Phase 1 has not measured this tier against the extension scan."""
        assert config["enabled"] is False

    def test_the_default_config_still_reaches_the_llm_tier(self, job, monkeypatch):
        predictor = _Recording()
        finder = LogFinder(config={"extensions": [".log"]}, predictor=predictor)
        _answering(monkeypatch, finder, ["part3.log"])

        assert [p.name for p in finder.find(job)] == ["part3.log"]
        assert predictor.calls == []

    def test_every_key_under_laya_has_a_reader(self, config):
        """The rule ``tests/cia/test_watch_config.py`` applies to the top level."""
        source = SOURCE.read_text(encoding="utf-8")
        unread = [key for key in config if f'"{key}"' not in source]

        assert not unread, f"nothing reads these: {unread}"

    def test_the_defaults_in_code_match_the_file(self, config):
        """A fallback that disagrees with the file is a third behaviour."""
        assert DEFAULT_LAYA_BACKEND == config["backend"]
        assert DEFAULT_LAYA_MIN_PROBABILITY == config["min_probability"]


def test_reaching_the_log_finder_does_not_import_the_predictor():
    """The loader behind ``aorta.laya.predictor`` brings torch (Decision 22).

    A subprocess rather than a ``sys.modules`` check, because by the time this
    file runs the module is imported at the top of it. ``aorta.cia.watch`` is on
    the import path of every Watch poll and, through the chat tools, of
    ``aorta.cli``.
    """
    probe = (
        "import aorta.cia.watch.log_finder, sys; "
        "print(','.join(m for m in sys.modules if m.startswith('aorta.laya')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"imported eagerly: {result.stdout.strip()}"
