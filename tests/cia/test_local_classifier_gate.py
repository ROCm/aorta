"""The go/no-go: the baselines, the refusals, and the kill criterion.

The decision this arithmetic feeds is whether three integration tracks get built,
so the two things worth pinning hardest are the comparison itself and the
refusals around it. A gate that reports a number it could not honestly measure is
worse than a gate that reports nothing: the second stops the plan, and the first
lets it proceed on a fabrication.

The latency and census paths are asserted by their refusals rather than by their
figures, because a figure from anything available here would describe a hash
function. Running them for real is a command in the final report, not a test.
"""

from __future__ import annotations

import pytest

from aorta.local_classifier.corpus.schema import LabelledExample
from aorta.local_classifier.eval import score
from aorta.local_classifier.gate import (
    CONTEXT_WINDOWS,
    LATENCY_BATCHES,
    LOWER_IS_BETTER,
    PRIMARY_METRIC,
    NotMeasurableError,
    TokenCensus,
    assert_measurable,
    available_baselines,
    gate_verdict,
    measure_latency,
    prior_distributions,
    recorded_distributions,
    run_gate,
    token_census,
)
from aorta.local_classifier.predictor import Choice, FakeDecisionPredictor, Noul, NoulAnswer

_SIGNAL = Choice(question="which signal?", options=("nan", "hang", "oom"))
_CLEAN = Noul(question="is this log healthy?")


def _candidate(
    mitigation: str, *, label: str, group: str, **baselines: str
) -> LabelledExample:
    """One per-candidate noul: the shape whose decision is a ranking, not a row."""
    return LabelledExample(
        decision="proposer_candidate",
        state=f"cells for {group}",
        question=Noul(question=f"Would applying {mitigation!r} help?"),
        label=label,
        join_key=f"{group}:{mitigation}",
        group=group,
        baselines=tuple(sorted(baselines.items())),
    )


def _signal(label: str, *, key: str, **baselines: str) -> LabelledExample:
    return LabelledExample(
        decision="watch_signal",
        state=f"log {key}",
        question=_SIGNAL,
        label=label,
        join_key=key,
        baselines=tuple(sorted(baselines.items())),
    )


class TestTheRefusalToFabricate:
    def test_the_fake_predictor_cannot_be_measured(self):
        """The single guard this whole package's trustworthiness rests on.

        A hash of the inputs produces exactly what a plausible result file needs:
        stable across runs, varied across examples, in range. Nothing downstream
        could tell, so the refusal has to be here.
        """
        with pytest.raises(NotMeasurableError, match="hash"):
            assert_measurable(FakeDecisionPredictor())

    def test_the_refusal_says_what_to_run_instead(self):
        with pytest.raises(NotMeasurableError, match="laya-typed-decisions"):
            assert_measurable(FakeDecisionPredictor())

    def test_a_named_checkpoint_is_measurable(self):
        class Real(FakeDecisionPredictor):
            def model_id(self) -> str:
                return "laya-typed-decisions"

        assert_measurable(Real())

    def test_the_whole_gate_refuses_the_fake(self):
        with pytest.raises(NotMeasurableError):
            run_gate([_signal("nan", key="a")], FakeDecisionPredictor())

    def test_latency_refuses_the_fake(self):
        with pytest.raises(NotMeasurableError):
            measure_latency(FakeDecisionPredictor(), ["a log"], [_CLEAN])

    def test_a_census_refuses_a_predictor_that_cannot_count_tokens(self):
        """No characters-per-token rule of thumb, because it is calibrated on prose.

        The fraction of deltas over the context window decides whether chunking is
        required, which decides whether the single-forward-pass claim survives. A
        guess there would answer the load-bearing question.
        """
        with pytest.raises(NotMeasurableError, match="characters per token"):
            token_census([_signal("nan", key="a")], FakeDecisionPredictor())


class TestMajorityBaseline:
    def test_it_is_the_fit_sets_label_frequencies(self):
        """A prior rather than a one-hot, so the floor gets a fair shot at calibration.

        A classifier that always answers the common slug and says 60% while being
        right 60% of the time is perfectly calibrated. Scoring it as though it had
        claimed certainty would let the encoder win on ECE by construction.
        """
        fit = [_signal("nan", key="a"), _signal("nan", key="b"), _signal("oom", key="c")]
        report = [_signal("nan", key="d")]
        distribution = dict(prior_distributions(fit, report)[0])
        assert distribution["nan"] == pytest.approx(2 / 3)
        assert distribution["oom"] == pytest.approx(1 / 3)
        assert distribution["hang"] == 0.0

    def test_it_predicts_the_most_common_label(self):
        fit = [_signal("oom", key=f"f{index}") for index in range(5)]
        report = [_signal("oom", key="r")]
        result = score(report, prior_distributions(fit, report))
        assert result.accuracy == 1.0

    def test_a_decision_with_no_fit_rows_gets_uniform_rather_than_an_invented_prior(self):
        report = [_signal("nan", key="r")]
        distribution = dict(prior_distributions([], report)[0])
        assert all(value == pytest.approx(1 / 3) for value in distribution.values())

    def test_the_prior_is_per_decision(self):
        """Watch's signal choice and its healthy noul have unrelated label spaces."""
        fit = [
            _signal("oom", key="a"),
            LabelledExample(
                decision="watch_healthy",
                state="log",
                question=_CLEAN,
                label="true",
                join_key="b",
            ),
        ]
        report = [_signal("oom", key="r")]
        assert dict(prior_distributions(fit, report)[0])["oom"] == pytest.approx(1.0)


class TestRecordedBaselines:
    def test_a_recorded_answer_becomes_a_near_one_hot(self):
        examples = [_signal("nan", key="a", dspy_signal="nan")]
        baseline = recorded_distributions("dspy_signal", examples)
        assert baseline.coverage == 1.0
        assert dict(baseline.distributions[0])["nan"] > 0.9

    def test_a_recorded_confidence_is_used_when_the_corpus_has_one(self):
        """The float ``should_alert`` compares against 0.70 today, scored as a probability.

        How badly calibrated it turns out to be is half the argument for replacing
        it, so it has to be scored as the probability it is claimed to be.
        """
        examples = [_signal("nan", key="a", dspy_signal="nan", dspy_confidence="0.6")]
        baseline = recorded_distributions("dspy_signal", examples)
        assert dict(baseline.distributions[0])["nan"] == pytest.approx(0.6)

    def test_the_remaining_mass_is_spread_over_the_other_options(self):
        examples = [_signal("nan", key="a", dspy_signal="nan", dspy_confidence="0.6")]
        distribution = dict(recorded_distributions("dspy_signal", examples).distributions[0])
        assert sum(distribution.values()) == pytest.approx(1.0)
        assert distribution["hang"] == pytest.approx(0.2)

    def test_an_answer_outside_the_option_set_is_an_abstention_not_an_error(self):
        """Scoring the regex as wrong on logs it never read would report the wrong fact.

        ``sanitizer_assessment()`` answers a machine-readable sanitizer summary and
        nothing else, so its coverage is the useful figure and its accuracy on the
        rest is meaningless.
        """
        examples = [_signal("nan", key="a", regex_signal="WATCH_CLEAN")]
        baseline = recorded_distributions("regex_signal", examples)
        assert baseline.coverage == 0.0
        assert all(
            value == pytest.approx(1 / 3) for _, value in baseline.distributions[0]
        )

    def test_a_missing_answer_is_an_abstention(self):
        baseline = recorded_distributions("dspy_signal", [_signal("nan", key="a")])
        assert baseline.coverage == 0.0

    def test_coverage_is_the_fraction_actually_answered(self):
        examples = [
            _signal("nan", key="a", regex_signal="nan"),
            _signal("oom", key="b"),
        ]
        assert recorded_distributions("regex_signal", examples).coverage == 0.5

    def test_a_certain_wrong_answer_is_clamped_away_from_one(self):
        """An unbounded NLL from one mistake would dominate every aggregate it appears in."""
        examples = [_signal("oom", key="a", dspy_signal="nan", dspy_confidence="1.0")]
        distribution = dict(recorded_distributions("dspy_signal", examples).distributions[0])
        assert distribution["nan"] <= 0.99
        assert distribution["oom"] > 0.0

    def test_a_malformed_confidence_does_not_crash_the_baseline(self):
        examples = [_signal("nan", key="a", dspy_signal="nan", dspy_confidence="n/a")]
        assert recorded_distributions("dspy_signal", examples).coverage == 1.0

    def test_only_the_baselines_the_corpus_carries_are_offered(self):
        examples = [_signal("nan", key="a", dspy_signal="nan")]
        assert available_baselines(examples) == ["dspy_signal"]

    def test_a_corpus_with_no_recorded_answers_offers_none(self):
        assert available_baselines([_signal("nan", key="a")]) == []


class TestTheVerdict:
    def _result(self, *rows: tuple[str, str]):
        examples = [_signal(label, key=f"k{index}") for index, (label, _) in enumerate(rows)]
        distributions = [
            tuple(
                (option, 0.8 if option == predicted else 0.1)
                for option in _SIGNAL.options
            )
            for _, predicted in rows
        ]
        return score(examples, distributions)

    def test_beating_the_strongest_baseline_clears_the_decision(self):
        candidate = self._result(("nan", "nan"), ("oom", "oom"))
        floor = self._result(("nan", "nan"), ("oom", "nan"))
        verdict = gate_verdict(candidate, {"majority": floor})
        assert verdict.passed
        assert verdict.cleared == ["watch_signal"]

    def test_a_tie_is_not_a_win(self):
        """The baselines cost no weights, no dependency and no CPU.

        Matching them means paying all three for nothing, so the comparison is
        strict.
        """
        candidate = self._result(("nan", "nan"), ("oom", "nan"))
        floor = self._result(("nan", "nan"), ("oom", "nan"))
        assert not gate_verdict(candidate, {"majority": floor}).passed

    def test_the_strongest_baseline_is_the_one_that_has_to_be_beaten(self):
        """Beating the majority floor while losing to the existing assessment is a fail.

        That third baseline is the real bar: replacing a ReAct call that already
        works is not justified by clearing a floor it also clears.
        """
        candidate = self._result(("nan", "nan"), ("oom", "nan"))
        weak = self._result(("nan", "hang"), ("oom", "hang"))
        strong = self._result(("nan", "nan"), ("oom", "oom"))
        verdict = gate_verdict(candidate, {"majority": weak, "dspy_signal": strong})
        assert not verdict.passed
        assert "dspy_signal" in verdict.decisions[0].reason

    def test_a_decision_with_no_baseline_cannot_clear(self):
        """A figure with no floor under it is not a gate."""
        verdict = gate_verdict(self._result(("nan", "nan")), {})
        assert not verdict.passed
        assert "nothing to have beaten" in verdict.decisions[0].reason

    def test_watch_signal_is_judged_on_macro_accuracy(self):
        """Four Autopsy categories collapse onto one slug, so plain accuracy misleads."""
        assert PRIMARY_METRIC["watch_signal"] == "macro_accuracy"

    def test_the_remaining_decisions_are_judged_on_plain_accuracy(self):
        """What is left once the ranking and thresholded shapes are accounted for:
        a genuine argmax over a balanced-enough label set."""
        assert PRIMARY_METRIC["proposer_mitigation"] == "accuracy"
        assert PRIMARY_METRIC["proposer_category"] == "accuracy"

    def test_the_two_ranking_decisions_are_judged_on_group_top1(self):
        """Both ask N yes/no questions where the decision is "which of these N".

        Accuracy over the rows scores a model that answers false to everything at
        0.95 while it chooses nothing, which is the opposite of what is measured.
        ``log_finder_useful`` was accuracy in the first version of the gate and
        that was a mistake in the same direction.
        """
        assert PRIMARY_METRIC["proposer_candidate"] == "group_top1"
        assert PRIMARY_METRIC["log_finder_useful"] == "group_top1"

    def test_the_two_thresholded_decisions_are_judged_on_safe_coverage(self):
        """Neither takes an argmax, so neither is scored as though it did.

        Watch's gate compares p(healthy) against ``clean_threshold`` and the probe
        agent stops when p(stop) reaches its own. ``watch_healthy`` was accuracy in
        the first two versions of this table, which was wrong in the same direction
        as ``log_finder_useful`` and worse: its answer moves with how the corpus
        was built, so one metric crowned opposite do-nothing models.
        """
        assert PRIMARY_METRIC["watch_healthy"] == "safe_coverage"
        assert PRIMARY_METRIC["proposer_stop"] == "safe_coverage"

    def test_the_argmaxed_decisions_keep_a_ranking_or_accuracy_metric(self):
        """Autopsy's category is a choice over eleven options, not a threshold.

        The number ``probe.py`` thresholds at 0.85 is its *confidence*, which no
        builder emits a decision for yet.
        """
        assert PRIMARY_METRIC["proposer_category"] == "accuracy"
        assert PRIMARY_METRIC["watch_signal"] == "macro_accuracy"

    def test_every_thresholded_primary_metric_has_an_operating_point(self):
        """``safe_coverage`` silently reads zero for a decision with no threshold,
        which would fail a model for a gap in this table rather than for its
        answers."""
        from aorta.local_classifier.eval import DEFAULT_THRESHOLDS

        gated = {
            decision
            for decision, metric in PRIMARY_METRIC.items()
            if metric in ("safe_coverage", "false_positive_rate", "true_positive_rate")
        }
        assert gated <= set(DEFAULT_THRESHOLDS)

    def test_every_decision_the_builders_emit_has_a_primary_metric(self):
        """A decision with no entry silently falls back to accuracy, which for a
        ranking shape is the wrong answer rather than a conservative one."""
        from aorta.local_classifier.corpus.schema import DECISIONS

        assert not sorted(set(DECISIONS) - set(PRIMARY_METRIC))

    def test_a_ranking_decision_can_be_lost_on_top1_while_won_on_accuracy(self):
        """Why the metric choice is load-bearing and not a preference.

        One candidate in five is the right one. The first model answers false to
        everything and is 80% accurate over the rows while ranking the wrong
        candidate first; the second finds it and is 20% accurate. The gate has to
        prefer the second.
        """
        rows = [
            _candidate(f"m{index}", label="true" if index == 4 else "false", group="step0")
            for index in range(5)
        ]
        never = [(("false", 0.9), ("true", 0.1))] * 5
        finds = [
            (("false", 0.1), ("true", 0.9)) if index == 4 else (("false", 0.4), ("true", 0.6))
            for index in range(5)
        ]
        refuser = score(rows, never)
        finder = score(rows, finds)
        assert refuser.accuracy == 0.8
        assert finder.accuracy == 0.2
        assert refuser.group_top1 == 0.0
        assert finder.group_top1 == 1.0
        assert gate_verdict(finder, {"majority": refuser}).cleared == ["proposer_candidate"]

    def test_macro_accuracy_is_what_decides_a_collapsed_corpus(self):
        """Always answering the common slug beats a real model on accuracy and loses on macro.

        Four of these five labels are ``oom``. A classifier that only emits ``oom``
        scores 0.8 accuracy against the other's 0.6 -- and 0.5 macro against its
        0.75, which is the figure that says which of them can tell the two slugs
        apart. This is why ``PRIMARY_METRIC`` reads Watch's signal on macro.
        """
        always_oom = self._result(
            ("oom", "oom"), ("oom", "oom"), ("oom", "oom"), ("oom", "oom"), ("nan", "oom")
        )
        discriminating = self._result(
            ("oom", "oom"), ("oom", "oom"), ("oom", "nan"), ("oom", "nan"), ("nan", "nan")
        )
        assert always_oom.accuracy == 0.8
        assert discriminating.accuracy == 0.6
        assert always_oom.macro_accuracy == 0.5
        assert discriminating.macro_accuracy == pytest.approx(0.75)
        assert gate_verdict(discriminating, {"majority": always_oom}).passed

    def test_one_track_may_clear_while_another_fails(self):
        """The plan is explicit: re-sequence around whichever cleared, do not stop."""
        examples = [
            _signal("nan", key="a"),
            LabelledExample(
                decision="watch_healthy",
                state="log b",
                question=_CLEAN,
                label="true",
                join_key="b",
            ),
        ]
        good = tuple((option, 0.8 if option == "nan" else 0.1) for option in _SIGNAL.options)
        candidate = score(examples, [good, (("false", 0.9), ("true", 0.1))])
        floor = score(examples, [good, (("false", 0.1), ("true", 0.9))])
        verdict = gate_verdict(candidate, {"majority": floor})
        assert verdict.cleared == []
        opposite = gate_verdict(floor, {"majority": candidate})
        assert opposite.cleared == ["watch_healthy"]

    def test_a_total_failure_says_to_write_up_the_negative_result(self):
        candidate = self._result(("nan", "oom"))
        floor = self._result(("nan", "nan"))
        assert "negative result" in gate_verdict(candidate, {"majority": floor}).summary()

    def test_the_verdict_serialises_with_its_reasons(self):
        candidate = self._result(("nan", "nan"))
        floor = self._result(("nan", "oom"))
        payload = gate_verdict(candidate, {"majority": floor}).to_dict()
        assert payload["passed"] is True
        assert payload["decisions"][0]["baselines"] == {"majority": 0.0}


class TestMetricDirection:
    """A rate of mistakes is beaten by being smaller, and the gate has to know.

    Every metric was compared as higher-is-better until a false-positive rate
    arrived. That was a latent inversion rather than a simplification: naming
    ``brier`` or ``ece`` in ``PRIMARY_METRIC`` would have silently reversed the
    verdict, and both are figures somebody might reasonably gate on.
    """

    def _healthy(self, label: str, *, key: str) -> LabelledExample:
        return LabelledExample(
            decision="watch_healthy",
            state=f"delta {key}",
            question=_CLEAN,
            label=label,
            join_key=key,
        )

    def _result(self, *rows: tuple[str, float]):
        examples = [self._healthy(label, key=f"j{i}") for i, (label, _) in enumerate(rows)]
        return score(examples, [(("false", 1.0 - p), ("true", p)) for _, p in rows])

    def test_the_rates_are_listed_as_lower_is_better(self):
        assert "false_positive_rate" in LOWER_IS_BETTER

    def test_brier_and_ece_are_too(self):
        """Listed rather than inferred from a name, because guessing is how the next
        one gets it wrong."""
        assert {"brier", "ece"} <= LOWER_IS_BETTER

    def test_safe_coverage_is_not(self):
        """It is a coverage, so more of it is better, which is why it is what the
        thresholded decisions are gated on."""
        assert "safe_coverage" not in LOWER_IS_BETTER

    def test_a_lower_rate_beats_a_higher_one(self, monkeypatch):
        monkeypatch.setitem(PRIMARY_METRIC, "watch_healthy", "false_positive_rate")
        safer = self._result(("false", 0.10), ("false", 0.10))
        leakier = self._result(("false", 0.99), ("false", 0.10))
        verdict = gate_verdict(safer, {"majority": leakier})
        assert verdict.passed
        assert verdict.decisions[0].candidate == 0.0

    def test_a_higher_rate_loses(self, monkeypatch):
        monkeypatch.setitem(PRIMARY_METRIC, "watch_healthy", "false_positive_rate")
        safer = self._result(("false", 0.10), ("false", 0.10))
        leakier = self._result(("false", 0.99), ("false", 0.10))
        assert not gate_verdict(leakier, {"majority": safer}).passed

    def test_the_strongest_baseline_is_the_lowest_one(self, monkeypatch):
        """Otherwise a candidate clears by beating the worst baseline offered."""
        monkeypatch.setitem(PRIMARY_METRIC, "watch_healthy", "false_positive_rate")
        candidate = self._result(("false", 0.10), ("false", 0.99))
        strong = self._result(("false", 0.10), ("false", 0.10))
        weak = self._result(("false", 0.99), ("false", 0.99))
        verdict = gate_verdict(candidate, {"tight": strong, "loose": weak})
        assert not verdict.passed
        assert "tight" in verdict.decisions[0].reason

    def test_a_tie_is_not_a_win_in_either_direction(self, monkeypatch):
        monkeypatch.setitem(PRIMARY_METRIC, "watch_healthy", "false_positive_rate")
        same = self._result(("false", 0.10), ("false", 0.99))
        assert not gate_verdict(same, {"majority": same}).passed


class TestTokenCensus:
    def test_it_counts_each_distinct_state_once(self):
        """A listing of forty files is one state, not forty.

        Counting per example would overstate the fraction over the context window
        by exactly the imbalance of the log-finder corpus.
        """

        class Counting(FakeDecisionPredictor):
            calls = 0

            def token_count(self, text: str) -> int:
                Counting.calls += 1
                return len(text.split())

        examples = [
            LabelledExample(
                decision="log_finder_useful",
                state="one shared listing",
                question=Noul(question=f"is {index} useful?"),
                label="false",
                join_key=f"listing:{index}",
            )
            for index in range(5)
        ]
        census = token_census(examples, Counting(), model_id="stub")
        assert Counting.calls == 1
        assert len(census.counts) == 1

    def test_the_fraction_over_a_window_is_what_decides_chunking(self):
        census = TokenCensus(model_id="stub", counts=[100, 600, 2000, 300])
        assert census.fraction_over(512) == 0.5
        assert census.fraction_over(1024) == 0.25

    def test_both_context_windows_are_reported(self):
        """512 rules out one checkpoint and 1024 rules out both without chunking."""
        assert CONTEXT_WINDOWS == (512, 1024)
        payload = TokenCensus(model_id="stub", counts=[10]).to_dict()
        assert sorted(payload["fraction_over"]) == ["1024", "512"]

    def test_an_empty_census_reports_zero_rather_than_dividing_by_zero(self):
        assert TokenCensus(model_id="stub").fraction_over(512) == 0.0
        assert TokenCensus(model_id="stub").median == 0.0


class TestLatencyShape:
    def test_thirty_is_timed_because_it_decides_the_reranking_backlog(self):
        assert LATENCY_BATCHES == (1, 5, 30)

    def test_it_times_real_calls_against_a_named_model(self):
        """No figure is asserted. What is asserted is that calls were made per repeat."""

        class Real(FakeDecisionPredictor):
            calls = 0

            def model_id(self) -> str:
                return "stub-checkpoint"

            def ask(self, states, questions):
                Real.calls += 1
                return super().ask(states, questions)

        samples = measure_latency(
            Real(), ["a", "b", "c"], [_CLEAN], batches=(1, 2), repeats=3
        )
        assert [sample.batch for sample in samples] == [1, 2]
        # One warm-up plus three repeats at each of two batch sizes.
        assert Real.calls == 1 + 3 + 3
        assert all(len(sample.seconds) == 3 for sample in samples)

    def test_a_batch_larger_than_the_corpus_is_skipped_not_padded(self):
        """Recycling states would let a cache report thirty as cheaper than five."""

        class Real(FakeDecisionPredictor):
            def model_id(self) -> str:
                return "stub-checkpoint"

        samples = measure_latency(Real(), ["only one"], [_CLEAN], batches=(1, 5, 30))
        assert [sample.batch for sample in samples] == [1]

    def test_per_state_cost_is_reported_beside_the_total(self):
        """N states are N forward passes, so the per-state figure is not expected to fall."""

        class Real(FakeDecisionPredictor):
            def model_id(self) -> str:
                return "stub-checkpoint"

        sample = measure_latency(Real(), ["a", "b"], [_CLEAN], batches=(2,), repeats=1)[0]
        assert sample.per_state_ms == pytest.approx(sample.median_ms / 2)

    def test_no_states_is_refused(self):
        class Real(FakeDecisionPredictor):
            def model_id(self) -> str:
                return "stub-checkpoint"

        with pytest.raises(NotMeasurableError, match="no states"):
            measure_latency(Real(), [], [_CLEAN])


class TestTheWholeGate:
    """Driven by a stub that claims a checkpoint name, so the wiring is covered.

    Its answers are still a hash, so nothing here asserts a score -- only that the
    baselines the corpus supports are all scored, on the same rows as the
    candidate.
    """

    class _Stub(FakeDecisionPredictor):
        def model_id(self) -> str:
            return "stub-checkpoint"

        def token_count(self, text: str) -> int:
            return len(text.split())

    def test_every_baseline_the_corpus_supports_is_scored(self):
        examples = [
            _signal("nan", key=f"j{index}", dspy_signal="nan", dspy_confidence="0.7")
            for index in range(20)
        ]
        result = run_gate(examples, self._Stub(), holdout=None, with_latency=False)
        assert sorted(result.baselines) == ["dspy_signal", "majority"]
        assert result.coverage["dspy_signal"] == 1.0

    def test_the_baselines_are_scored_on_the_same_rows_as_the_candidate(self):
        """What the deterministic split buys: both sides compute it and cannot disagree."""
        examples = [_signal("nan", key=f"j{index}", dspy_signal="nan") for index in range(60)]
        result = run_gate(examples, self._Stub(), holdout=0.2, with_latency=False)
        assert result.candidate.count == result.baselines["majority"].count
        assert result.candidate.count < len(examples)

    def test_the_checkpoints_own_calibration_is_recorded_beside_our_refit(self):
        """Decision 22 asks for the fit, and there are two of them.

        The one baked into the checkpoint and the one fitted on our corpus. A
        result naming only ours would attribute a clamped bucket's uncalibrated
        answers to a fit that had nothing to do with them.
        """
        examples = [_signal("nan", key=f"j{index}") for index in range(10)]
        result = run_gate(examples, self._Stub(), holdout=None, with_latency=False)
        assert result.calibration is not None
        payload = result.to_dict()
        assert "checkpoint_calibration" in payload
        assert "temperatures" in payload["candidate"]

    def test_a_clamped_bucket_reaches_the_printed_summary(self):
        """The one fact that changes how every probability under it should be read.

        Leaving it in the JSON only would put it where the person running the gate
        is not looking.
        """

        class Clamped(self._Stub):
            def calibration(self):
                from aorta.local_classifier.predictor import Calibration, ClampedBucket

                return Calibration(
                    model_id="stub-checkpoint",
                    clamped=(
                        ClampedBucket(bucket="choice:11+", shipped=0.1006, applied=0.5),
                    ),
                )

        examples = [_signal("nan", key=f"j{index}") for index in range(10)]
        summary = run_gate(
            examples, Clamped(), holdout=None, with_latency=False
        ).summary()
        assert "clamped" in summary
        assert "choice:11+" in summary

    def test_an_unknown_calibration_is_said_out_loud_too(self):
        examples = [_signal("nan", key=f"j{index}") for index in range(10)]
        summary = run_gate(
            examples, self._Stub(), holdout=None, with_latency=False
        ).summary()
        assert "calibration: unknown" in summary

    def test_the_census_answers_the_chunking_question(self):
        examples = [_signal("nan", key=f"j{index}") for index in range(10)]
        result = run_gate(examples, self._Stub(), holdout=None, with_latency=False)
        assert result.census is not None
        assert result.census.model_id == "stub-checkpoint"

    def test_skipping_the_census_leaves_the_question_unanswered_and_visible(self):
        examples = [_signal("nan", key=f"j{index}") for index in range(10)]
        result = run_gate(
            examples, self._Stub(), holdout=None, with_latency=False, with_census=False
        )
        assert result.census is None
        assert result.to_dict()["token_census"] is None

    def test_the_summary_names_the_model_and_the_verdict(self):
        examples = [_signal("nan", key=f"j{index}", dspy_signal="nan") for index in range(10)]
        summary = run_gate(
            examples, self._Stub(), holdout=None, with_latency=False
        ).summary()
        assert "stub-checkpoint" in summary
        assert "verdict:" in summary

    def test_a_pinned_perfect_model_clears_a_corpus_the_baselines_get_wrong(self):
        """End to end through the gate's own arithmetic, with the model pinned.

        The pin is what makes this a test of the comparison rather than of a hash:
        the predictor is made to answer correctly, the recorded baseline is made
        to answer wrongly, and the verdict has to follow.

        The labels are deliberately balanced. With every row labelled the same
        way, the majority prior is also perfect, and a perfect candidate ties with
        it -- which the gate correctly refuses to call a win.
        """
        broken = Noul(question="is this log broken?")
        examples = [
            LabelledExample(
                decision="watch_healthy",
                state=f"log {index}",
                question=_CLEAN if index % 2 else broken,
                label="true" if index % 2 else "false",
                join_key=f"j{index}",
                # Wrong on every row, at high confidence: the shape of an
                # uncalibrated self-report.
                baselines=(
                    ("dspy_signal", "false" if index % 2 else "true"),
                    ("dspy_confidence", "0.9"),
                ),
            )
            for index in range(20)
        ]

        class Pinned(self._Stub):
            def __init__(self) -> None:
                super().__init__(
                    pinned={
                        _CLEAN.question: NoulAnswer(probability=0.95),
                        broken.question: NoulAnswer(probability=0.05),
                    }
                )

        result = run_gate(examples, Pinned(), holdout=None, with_latency=False)
        assert result.candidate.accuracy == 1.0
        assert result.baselines["dspy_signal"].accuracy == 0.0
        assert result.baselines["majority"].accuracy == 0.5
        assert result.verdict.cleared == ["watch_healthy"]
