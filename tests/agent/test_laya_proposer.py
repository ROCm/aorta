"""Track B: the probe agent's proposer on a local calibrated encoder.

Nothing here loads weights. ``FakeLayaPredictor`` answers from a hash of its
inputs, which is useless as a prediction and exactly right as a test double: it
is deterministic across processes, it needs no extra installed, and ``pinned=``
is how a test says what it actually wants to exercise. Anything that asserts on
an *answer* pins it; only the tests that assert on what was *asked* leave the
fake to its hash.

Two properties get the most attention, because they are the two a later change
could break without anything looking wrong.

The **answer space is the candidate list**, so this proposer structurally cannot
name a mitigation that was not offered. ``_step_from_content`` filters the LLM
backends' replies for exactly that reason; here there is nothing to filter.

The **state is the LLM backends' prompt, verbatim**, because that is what
``aorta/laya/corpus/proposer.py`` recorded when it built the training corpus.
Serialising it a second way would be train/serve skew hidden inside a helper,
and it would also stop the Phase 1 comparison between the two backends being a
comparison.
"""

from __future__ import annotations

import builtins
import warnings

import pytest

from aorta.agent.llm import (
    DEFAULT_LAYA_CHECKPOINT,
    DEFAULT_LAYA_STOP_THRESHOLD,
    EVIDENCE_ONLY_CATEGORIES,
    LAYA_CATEGORY_QUESTION,
    LAYA_MITIGATION_QUESTION,
    LAYA_STOP_QUESTION,
    LAYA_STOP_WHEN_FALSE,
    LAYA_STOP_WHEN_TRUE,
    PROBE_CATEGORIES,
    AgentStep,
    LayaProposer,
    _build_prompt,
    make_proposer,
)
from aorta.laya.predictor import (
    Choice,
    ChoiceAnswer,
    FakeLayaPredictor,
    LayaUnavailableError,
    Noul,
    NoulAnswer,
)

CANDIDATES = ["none", "tf32_off", "hsa_xnack", "hip_sync"]
SUMMARIES = [
    {
        "cell_name": "none-none",
        "verdict": "fail",
        "failure_detectors_fired": ["tier1:exit_code"],
    }
]


def _propose(proposer, candidates=None, tried=None, summaries=None, symptom="loss went to nan"):
    return proposer.propose(
        symptom=symptom,
        cell_summaries=SUMMARIES if summaries is None else summaries,
        candidates=CANDIDATES if candidates is None else candidates,
        tried=[] if tried is None else tried,
    )


class Recording(FakeLayaPredictor):
    """A fake that remembers what it was asked, and in how many calls."""

    def __init__(self, *, pinned=None, model_id="laya-typed-decisions"):
        super().__init__(pinned=pinned)
        self._model_id = model_id
        self.calls: list[tuple[list[str], list]] = []

    def model_id(self):
        return self._model_id

    def ask(self, states, questions):
        self.calls.append((list(states), list(questions)))
        return super().ask(states, questions)


def _pinned(*, stop, mitigation=None, category=None, model_id="laya-typed-decisions"):
    """A predictor whose answers are chosen rather than hashed.

    ``stop`` has no default on purpose: every test that reads a step has to say
    which side of the threshold it meant to be on, or it is a test whose verdict
    depends on a hash of its own fixture text.
    """
    answers = {LAYA_STOP_QUESTION: NoulAnswer(probability=stop)}
    if mitigation is not None:
        answers[LAYA_MITIGATION_QUESTION] = ChoiceAnswer(probabilities=mitigation)
    if category is not None:
        answers[LAYA_CATEGORY_QUESTION] = ChoiceAnswer(probabilities=category)
    return Recording(pinned=answers, model_id=model_id)


# ── the factory ───────────────────────────────────────────────────────────


class TestMakeProposer:
    def test_laya_resolves_to_the_laya_proposer(self):
        assert isinstance(make_proposer("laya"), LayaProposer)

    def test_it_defaults_to_the_fine_tuned_checkpoint(self):
        """The base checkpoint scores below a majority baseline on the card's own data."""
        assert make_proposer("laya")._checkpoint == DEFAULT_LAYA_CHECKPOINT
        assert DEFAULT_LAYA_CHECKPOINT == "laya-typed-decisions"

    def test_llm_model_names_a_checkpoint_on_this_backend(self):
        """Not a model on someone's endpoint: laya is an encoder, not a provider."""
        proposer = make_proposer("laya", model="/srv/weights/laya-rocm-finetune")
        assert proposer._checkpoint == "/srv/weights/laya-rocm-finetune"

    def test_building_the_proposer_resolves_no_weights(self):
        """The loop builds a proposer before it knows whether it needs one."""
        assert make_proposer("laya", model="not-a-real-checkpoint")._predictor is None


# ── the import boundary ────────────────────────────────────────────────────


class TestTheImportBoundary:
    """``aorta.agent.llm`` is import-gated, and this backend is why that matters.

    ``tests/cli/test_chat_boundaries.py`` asserts the whole module's import graph
    in a fresh interpreter, which is the only honest measurement of it. These two
    assert the halves that live in this class rather than in the module scope.
    """

    def test_an_exhausted_search_needs_neither_weights_nor_the_extra(self):
        proposer = LayaProposer(checkpoint="not-a-real-checkpoint")
        step = _propose(proposer, candidates=["none", "tf32_off"], tried=["tf32_off"])
        assert step.stop is True
        assert step.stop_reason == "exhausted_candidates"
        assert proposer._predictor is None

    def test_constructing_one_imports_nothing(self, monkeypatch):
        """Selecting a backend must not be the thing that pays for it."""

        def _explode(name, *args, **kwargs):
            raise AssertionError(f"building the proposer imported {name!r}")

        monkeypatch.setattr(builtins, "__import__", _explode)
        assert LayaProposer(checkpoint="laya")._checkpoint == "laya"


# ── the questions, and where they came from ────────────────────────────────


class TestTheQuestions:
    def test_one_state_and_three_questions_in_one_forward_pass(self):
        """The efficiency claim the whole track rests on, asserted rather than assumed.

        M questions about one state is one pass; N states is N passes. Asking
        through ``ask_noul`` / ``ask_choice`` would have been three.
        """
        predictor = Recording()
        _propose(LayaProposer(predictor=predictor))
        assert len(predictor.calls) == 1
        states, questions = predictor.calls[0]
        assert len(states) == 1
        assert len(questions) == 3

    def test_the_mitigation_choice_offers_exactly_what_is_left_to_try(self):
        """No baseline, nothing already tried, and nothing invented."""
        predictor = Recording()
        _propose(LayaProposer(predictor=predictor), tried=["tf32_off"])
        mitigation = next(
            q for q in predictor.calls[0][1] if q.question == LAYA_MITIGATION_QUESTION
        )
        assert mitigation.options == ("hsa_xnack", "hip_sync")

    def test_the_category_choice_offers_the_derived_probe_set(self):
        """Not AUTOPSY_CATEGORIES: a category the agent cannot reach teaches it to guess one."""
        predictor = Recording()
        _propose(LayaProposer(predictor=predictor))
        category = next(
            q for q in predictor.calls[0][1] if q.question == LAYA_CATEGORY_QUESTION
        )
        assert set(category.options) == PROBE_CATEGORIES
        assert not set(category.options) & EVIDENCE_ONLY_CATEGORIES

    def test_the_stop_question_is_a_glossed_noul(self):
        """Whether a search should stop is ambiguous enough to need both sides spelled out."""
        predictor = Recording()
        _propose(LayaProposer(predictor=predictor))
        stop = next(q for q in predictor.calls[0][1] if q.question == LAYA_STOP_QUESTION)
        assert isinstance(stop, Noul)
        assert stop.when_true == LAYA_STOP_WHEN_TRUE
        assert stop.when_false == LAYA_STOP_WHEN_FALSE

    def test_the_state_is_the_llm_backends_prompt_verbatim(self):
        """What the encoder reads and what the LLM reads differ in model, not in input."""
        predictor = Recording()
        _propose(LayaProposer(predictor=predictor), tried=["tf32_off"])
        _system, expected = _build_prompt(
            "loss went to nan", SUMMARIES, ["hsa_xnack", "hip_sync"], ["tf32_off"]
        )
        assert predictor.calls[0][0] == [expected]

    def test_the_questions_match_the_ones_the_corpus_labels(self):
        """Training on one question and asking another does not fail, it just answers worse.

        ``aorta/laya/corpus/proposer.py`` now imports all three from here, so
        the two copies cannot drift: there is only one. What is left to check is
        that the builder is still reaching for these and has not grown a local
        rephrasing beside them.
        """
        from aorta.laya.corpus import proposer as corpus

        assert corpus._STOP.question == LAYA_STOP_QUESTION
        assert corpus._STOP.when_true == LAYA_STOP_WHEN_TRUE
        assert corpus._STOP.when_false == LAYA_STOP_WHEN_FALSE
        assert corpus.LAYA_MITIGATION_QUESTION is LAYA_MITIGATION_QUESTION
        assert corpus.LAYA_CATEGORY_QUESTION is LAYA_CATEGORY_QUESTION


# ── the step it builds ─────────────────────────────────────────────────────


class TestTheProposedStep:
    def test_a_proposal_names_the_most_probable_candidate(self):
        predictor = _pinned(
            stop=0.1, mitigation=(("tf32_off", 0.2), ("hsa_xnack", 0.7), ("hip_sync", 0.1))
        )
        step = _propose(LayaProposer(predictor=predictor))
        assert step.next_mitigations == ["hsa_xnack"]
        assert step.stop is False

    def test_confidence_is_the_probability_of_the_mitigation_proposed(self):
        """Off the calibrated head, not a float a prompt asked a model to invent."""
        predictor = _pinned(
            stop=0.1, mitigation=(("tf32_off", 0.25), ("hsa_xnack", 0.6), ("hip_sync", 0.15))
        )
        assert _propose(LayaProposer(predictor=predictor)).confidence == pytest.approx(0.6)

    def test_the_category_comes_from_the_category_head(self):
        predictor = _pinned(stop=0.1, category=(("illegal_mem", 0.8), ("rccl_hang", 0.2)))
        assert _propose(LayaProposer(predictor=predictor)).category == "illegal_mem"

    def test_one_mitigation_is_proposed_at_a_time(self):
        """The loop adds one axis value per iteration; a list would widen the sweep."""
        step = _propose(LayaProposer(predictor=_pinned(stop=0.0)))
        assert len(step.next_mitigations) == 1

    @pytest.mark.parametrize("symptom", ["a", "b", "c", "d", "e"])
    def test_a_mitigation_outside_the_candidate_list_cannot_come_back(self, symptom):
        """Structural rather than filtered: the option set *is* the answer space.

        Driven across several states, whose hashed answers land on different
        options, so that this says something the option tuple alone does not.
        """
        step = _propose(
            LayaProposer(predictor=_pinned(stop=0.0)), tried=["tf32_off"], symptom=symptom
        )
        assert step.next_mitigations
        assert set(step.next_mitigations) <= {"hsa_xnack", "hip_sync"}

    def test_the_hypothesis_follows_the_fake_proposers_template(self):
        """Laya never emits text, so there is nothing to ask it for."""
        predictor = _pinned(
            stop=0.1, mitigation=(("tf32_off", 0.9), ("hsa_xnack", 0.05), ("hip_sync", 0.05))
        )
        step = _propose(LayaProposer(predictor=predictor))
        assert step.hypothesis.startswith(
            "Try mitigation 'tf32_off' based on detectors ['tier1:exit_code']."
        )
        assert "Symptom: loss went to nan" in step.hypothesis

    def test_the_number_of_candidates_travels_with_the_probability(self):
        """A probability off an N-way choice is not interpretable without N.

        The argmax of a 21-way answer is a strong signal at 0.24 and a uniform
        one at 0.048, and ``AgentStep.confidence`` is one float compared
        downstream against thresholds picked for producers with other widths.
        """
        predictor = _pinned(
            stop=0.1, mitigation=(("tf32_off", 0.5), ("hsa_xnack", 0.3), ("hip_sync", 0.2))
        )
        step = _propose(LayaProposer(predictor=predictor))
        assert "of 3 candidates" in step.hypothesis

    def test_the_checkpoint_that_answered_is_recorded_in_the_step(self):
        """Decision 22: a report read on its own still has to say which model said this.

        ``hypothesis`` is the only ``AgentStep`` field that both
        ``agent_log.jsonl`` and ``agent_report.md`` carry through unchanged, so
        it is where the identity goes.
        """
        predictor = _pinned(stop=0.1, model_id="laya-rocm-finetune@cpu")
        assert "laya-rocm-finetune@cpu" in _propose(LayaProposer(predictor=predictor)).hypothesis


# ── what the step discloses about its own calibration ──────────────────────


def _clamping(*buckets, stop=0.0, model_id="laya-typed-decisions"):
    """A predictor whose checkpoint reports *buckets* as clamped."""
    from aorta.laya.predictor import Calibration, ClampedBucket

    predictor = _pinned(stop=stop, model_id=model_id)
    predictor.calibration = lambda: Calibration(  # type: ignore[method-assign]
        model_id=model_id,
        clamped=tuple(
            ClampedBucket(bucket=b, shipped=0.1006, applied=0.5) for b in buckets
        ),
        applied=tuple((b, 0.5) for b in buckets),
    )
    return predictor


class TestTheDisclosure:
    """Two halves, known at two different times, and the step carries both.

    Which bucket a question falls in is a property of the *question*:
    ``bucket_for`` is pure, so it is on every step whether or not a checkpoint
    ever loaded. Whether that bucket was trustworthy is a property of the
    *checkpoint*: ``Calibration.caveat`` reads the checkpoint's own shipped and
    applied temperature tables, which is why it is not re-derived here.
    """

    def test_the_bucket_is_named_offline_even_when_nothing_is_known(self):
        """``FakeLayaPredictor`` reports unknown, and the width half still lands."""
        wide = ["none"] + [f"mitigation_{n}" for n in range(21)]
        step = _propose(LayaProposer(predictor=_pinned(stop=0.0)), candidates=wide)
        assert "of 21 candidates (choice:11+)" in step.hypothesis

    def test_the_real_registry_lands_in_the_widest_bucket(self):
        """Not a corner case: this is what ``aorta agent mitigate`` does by default."""
        from aorta.registry.mitigations import load_mitigations

        candidates = [name for name in load_mitigations() if name != "none"]
        step = _propose(
            LayaProposer(predictor=_pinned(stop=0.0)), candidates=["none", *candidates]
        )
        assert f"of {len(candidates)} candidates (choice:11+)" in step.hypothesis

    def test_a_clamped_bucket_is_disclosed_with_both_temperatures(self):
        step = _propose(
            LayaProposer(predictor=_clamping("choice:11+")),
            candidates=["none"] + [f"mitigation_{n}" for n in range(21)],
        )
        assert "NOT CALIBRATED" in step.hypothesis
        assert "0.1006" in step.hypothesis and "0.5" in step.hypothesis

    def test_a_clean_bucket_says_nothing(self):
        """Crying wolf on every line is how a real warning stops being read."""
        step = _propose(
            LayaProposer(predictor=_clamping("choice:3-5")),
            candidates=["none"] + [f"mitigation_{n}" for n in range(21)],
        )
        assert "NOT CALIBRATED" not in step.hypothesis
        assert "choice:11+" in step.hypothesis

    def test_the_broken_bucket_is_the_checkpoints_to_name_not_ours(self):
        """The bug the option-count arithmetic this replaced would have shipped.

        An earlier version hardcoded "eleven or more options is the broken
        bucket". That is true of the published checkpoint and false for exactly
        the artifact Phase 1 exists to produce: a fine-tune with a sane
        ``choice:11+`` and a broken ``choice:3-5``. On such a checkpoint the old
        code stayed silent on the bucket that was actually broken and kept
        warning about one that was fine -- both errors at once, and in the
        direction that flatters the model.
        """
        narrow = _propose(
            LayaProposer(predictor=_clamping("choice:3-5")),
            candidates=["none", "tf32_off", "hsa_xnack", "hip_sync"],
        )
        assert "NOT CALIBRATED" in narrow.hypothesis
        assert "choice:3-5" in narrow.hypothesis

    def test_a_clamped_noul_is_disclosed_on_the_step_that_thresholded_it(self):
        """Which the width arithmetic could not express at all.

        It read a candidate count, so it had nothing to say about a stop, and the
        comment beside the threshold asserted a noul was safe by construction. It
        is not: whether the noul bucket was clamped is a fact about a checkpoint.
        """
        step = _propose(LayaProposer(predictor=_clamping("noul:2", stop=0.99)))
        assert step.stop is True
        assert "NOT CALIBRATED" in step.hypothesis
        assert "noul:2" in step.hypothesis

    def test_an_unknown_calibration_reads_as_unknown_rather_than_as_clean(self):
        """A fake has no temperature table, and silence would read as a clean bill.

        ``is_clamped`` is three-valued for the same reason: a pre-0.3.5 library
        reports unknown because there the sharpening was applied and nobody was
        told.
        """
        step = _propose(LayaProposer(predictor=_pinned(stop=0.0)))
        assert "CALIBRATION UNKNOWN" in step.hypothesis

    def test_the_calibration_is_read_after_the_answers_not_before(self):
        """``calibration()`` does not load, so asking early would report unknown forever.

        A predictor that has not been asked a question yet has no tables to diff.
        Reading it before ``ask_one`` would disclose an unknown calibration on
        every step of a run whose calibration was perfectly knowable.
        """
        order: list[str] = []

        class Ordered(Recording):
            def ask(self, states, questions):
                order.append("ask")
                return super().ask(states, questions)

            def calibration(self):
                order.append("calibration")
                return super().calibration()

        _propose(
            LayaProposer(predictor=Ordered(pinned={LAYA_STOP_QUESTION: NoulAnswer(0.0)}))
        )
        assert order == ["ask", "calibration"]

    def test_the_category_choice_has_its_own_bucket(self):
        """``PROBE_CATEGORIES`` is derived, so it can widen with nobody editing this file.

        ``AUTOPSY_CATEGORIES`` has eleven members and this set is that minus the
        three evidence-only ones, so eight -- ``choice:6-10``, a different bucket
        from the candidate choice's and independently clampable. Three more
        probe-reachable categories would move it without a line of Track B
        changing, which is why the bucket is asked for rather than assumed.
        """
        from aorta.laya.predictor import Choice, bucket_for

        category_q = Choice(
            question=LAYA_CATEGORY_QUESTION, options=tuple(sorted(PROBE_CATEGORIES))
        )
        assert bucket_for(category_q) == "choice:6-10"

    def test_a_warning_raised_while_answering_is_not_swallowed(self):
        """0.3.5 names the clamped bucket in a ``RuntimeWarning``, and it has to get out.

        Nothing in ``aorta`` calls ``logging.captureWarnings`` or installs a
        filter on this path, and ``propose`` must not become the first thing to.
        This is the safeguard; the caveat above is what covers the run where it
        is silenced from outside.
        """

        class WarnsOnLoad(Recording):
            def ask(self, states, questions):
                warnings.warn(
                    "clamped fitted temperature for 'choice:11+'",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return super().ask(states, questions)

        noisy = WarnsOnLoad(pinned={LAYA_STOP_QUESTION: NoulAnswer(probability=0.0)})
        with pytest.warns(RuntimeWarning, match="choice:11"):
            _propose(LayaProposer(predictor=noisy))


# ── stopping ───────────────────────────────────────────────────────────────


class TestStopping:
    def test_it_stops_once_the_noul_reaches_the_threshold(self):
        step = _propose(LayaProposer(predictor=_pinned(stop=0.9), stop_threshold=0.75))
        assert step.stop is True
        assert step.next_mitigations == []

    def test_it_keeps_going_below_the_threshold(self):
        step = _propose(LayaProposer(predictor=_pinned(stop=0.74), stop_threshold=0.75))
        assert step.stop is False
        assert step.next_mitigations

    def test_at_the_threshold_counts_as_reaching_it(self):
        """``>=``, the same reading ``should_alert`` uses on the other side of the tree."""
        step = _propose(LayaProposer(predictor=_pinned(stop=0.75), stop_threshold=0.75))
        assert step.stop is True

    def test_the_default_threshold_is_above_the_midpoint(self):
        """A false stop ends an investigation; a false continue costs one probe cell.

        ``AgentPolicy`` bounds how many cells there can be, so the asymmetry is
        paid for in cells rather than in a truncated search.
        """
        assert DEFAULT_LAYA_STOP_THRESHOLD > 0.5

    def test_a_stop_says_why_rather_than_leaving_the_loop_to_read_the_prose(self):
        """``_resolve_stop_outcome`` infers a reason from the hypothesis when given none."""
        step = _propose(LayaProposer(predictor=_pinned(stop=0.99)))
        assert step.stop_reason == "agent_requested"
        assert "No remaining" not in step.hypothesis

    def test_a_stop_reports_the_probability_and_the_threshold_it_stopped_on(self):
        step = _propose(LayaProposer(predictor=_pinned(stop=0.92), stop_threshold=0.75))
        assert step.confidence == pytest.approx(0.92)
        assert "p(stop)=0.92" in step.hypothesis
        assert "0.75" in step.hypothesis

    def test_an_exhausted_candidate_list_stops_before_the_model_is_consulted(self):
        """The existing guard, kept: ``_exhausted_step`` costs no forward pass."""
        predictor = Recording()
        step = _propose(
            LayaProposer(predictor=predictor),
            candidates=["none", "tf32_off"],
            tried=["tf32_off"],
        )
        assert predictor.calls == []
        assert step == AgentStep(
            category="unknown",
            hypothesis="No remaining registered mitigations to try.",
            next_mitigations=[],
            confidence=0.9,
            stop=True,
            stop_reason="exhausted_candidates",
        )


# ── failure modes ──────────────────────────────────────────────────────────


class TestFailures:
    def test_a_missing_checkpoint_is_loud_rather_than_a_silent_stop(self):
        """The opposite of how the LLM backends treat an unparseable reply, deliberately.

        There, a bad response is expected and has to still produce a report. Here
        it means the weights are not on this machine, and a stop would read as
        "the agent decided the search was over". ``run_agent_loop`` catches it
        into an ``error`` outcome with the audit trail intact.
        """
        with pytest.raises(LayaUnavailableError, match="unknown Laya checkpoint"):
            _propose(LayaProposer(checkpoint="laya-multilingual"))

    def test_a_predictor_answering_the_wrong_shape_is_not_read_as_a_probability(self):
        """``ask_noul`` / ``ask_choice`` make this check; one forward pass has to make it too."""

        class Wrong(FakeLayaPredictor):
            def ask(self, states, questions):
                return [[NoulAnswer(probability=0.5)] * len(questions) for _ in states]

        with pytest.raises(TypeError, match="answered the proposer's three questions"):
            _propose(LayaProposer(predictor=Wrong()))

    def test_the_weights_are_resolved_once_across_iterations(self, monkeypatch):
        """The loop calls ``propose`` once per iteration, up to ``--max-iterations``.

        A checkpoint rebuilt per call would cost seconds an iteration, which is
        most of the cost this track exists to remove.
        """
        import aorta.laya.predictor as seam

        built: list[str] = []

        def _build(backend="fake", *, device=None, checkpoint=None):
            built.append(checkpoint or backend)
            return Recording()

        monkeypatch.setattr(seam, "make_predictor", _build)
        proposer = LayaProposer(checkpoint="laya-typed-decisions")
        _propose(proposer)
        _propose(proposer)
        assert built == ["laya-typed-decisions"]


class TestTheProtocol:
    def test_it_is_an_llm_proposer(self):
        """No call-site change: the loop only ever sees ``LLMProposer.propose``."""
        from aorta.agent.llm import LLMProposer

        proposer: LLMProposer = LayaProposer(predictor=_pinned(stop=0.0))
        assert isinstance(_propose(proposer), AgentStep)

    def test_it_asks_in_the_seams_own_question_types(self):
        """Built here, answered there: a second question type would be one too many."""
        predictor = Recording()
        _propose(LayaProposer(predictor=predictor))
        assert all(isinstance(q, (Choice, Noul)) for q in predictor.calls[0][1])
