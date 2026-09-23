"""The Phase 1 scoring, pinned without a model.

The gate these numbers feed decides whether three integration tracks get built,
so the arithmetic has to be trustworthy on a machine with no weights on it. That
is the same split ``tests/chat/test_retrieval_eval.py`` makes: the end-to-end run
needs a loaded model and is deliberately not a test, and what *is* tested is the
pure function that turns answers into scores.

Every metric here is checked against a value worked out by hand rather than
against the implementation's own output, because a test that only pins current
behaviour would happily pin a sign error.
"""

from __future__ import annotations

import math

import pytest

from aorta.laya.corpus.schema import CorpusError, LabelledExample
from aorta.laya.eval import (
    DEFAULT_THRESHOLDS,
    FALSE_POSITIVE_BUDGET,
    EvalError,
    apply_temperature,
    calibration_key,
    distribution_of,
    evaluate,
    fit_temperature,
    fit_temperatures,
    negative_log_likelihood,
    recalibrate,
    score,
    split_examples,
)
from aorta.laya.predictor import (
    Choice,
    ChoiceAnswer,
    FakeLayaPredictor,
    Noul,
    NoulAnswer,
)

_HEALTHY = Noul(question="is it healthy?")
_SIGNAL = Choice(question="which signal?", options=("nan", "hang", "oom"))


def _noul(label: str, *, key: str = "job-1", **baselines: str) -> LabelledExample:
    return LabelledExample(
        decision="watch_healthy",
        state=f"log for {key}",
        question=_HEALTHY,
        label=label,
        join_key=key,
        baselines=tuple(sorted(baselines.items())),
    )


def _choice(label: str, *, key: str = "job-1", **baselines: str) -> LabelledExample:
    return LabelledExample(
        decision="watch_signal",
        state=f"log for {key}",
        question=_SIGNAL,
        label=label,
        join_key=key,
        baselines=tuple(sorted(baselines.items())),
    )


class TestDistributions:
    def test_a_noul_becomes_a_two_option_distribution(self):
        """p(yes) on 'true' and the remainder on 'false', in the model's own order."""
        got = distribution_of(_HEALTHY, NoulAnswer(probability=0.8))
        assert [name for name, _ in got] == ["false", "true"]
        assert dict(got)["true"] == pytest.approx(0.8)
        assert dict(got)["false"] == pytest.approx(0.2)

    def test_a_choice_keeps_the_order_it_was_offered(self):
        answer = ChoiceAnswer(probabilities=(("nan", 0.5), ("hang", 0.3), ("oom", 0.2)))
        assert [name for name, _ in distribution_of(_SIGNAL, answer)] == ["nan", "hang", "oom"]

    def test_an_answer_over_different_options_is_refused(self):
        """A score against the wrong option set is not a score of that question."""
        answer = ChoiceAnswer(probabilities=(("nan", 0.9), ("throttle", 0.1)))
        with pytest.raises(EvalError, match="do not match"):
            distribution_of(_SIGNAL, answer)

    def test_a_noul_answer_to_a_choice_is_refused(self):
        with pytest.raises(EvalError, match="noul answer"):
            distribution_of(_SIGNAL, NoulAnswer(probability=0.5))


class TestAccuracy:
    def test_the_argmax_decides_correctness(self):
        result = score([_choice("nan")], [(("nan", 0.4), ("hang", 0.35), ("oom", 0.25))])
        assert result.accuracy == 1.0
        assert result.rows[0].predicted == "nan"
        assert result.rows[0].confidence == pytest.approx(0.4)

    def test_a_confident_wrong_answer_scores_zero(self):
        result = score([_choice("oom")], [(("nan", 0.9), ("hang", 0.05), ("oom", 0.05))])
        assert result.accuracy == 0.0
        assert result.rows[0].label_probability == pytest.approx(0.05)

    def test_accuracy_averages_over_the_set(self):
        result = score(
            [_choice("nan", key="a"), _choice("oom", key="b")],
            [(("nan", 0.9), ("hang", 0.05), ("oom", 0.05)),
             (("nan", 0.9), ("hang", 0.05), ("oom", 0.05))],
        )
        assert result.accuracy == 0.5

    def test_an_empty_set_scores_zero_rather_than_dividing_by_zero(self):
        assert score([], []).accuracy == 0.0

    def test_mismatched_lengths_are_a_programming_error(self):
        with pytest.raises(EvalError, match="1 examples but 2 distributions"):
            score([_choice("nan")], [(("nan", 1.0),), (("nan", 1.0),)])

    def test_a_distribution_is_renormalised_rather_than_trusted(self):
        """Probabilities that do not sum to one still have to rank and score."""
        result = score([_choice("nan")], [(("nan", 2.0), ("hang", 1.0), ("oom", 1.0))])
        assert result.rows[0].confidence == pytest.approx(0.5)


class TestMacroAccuracy:
    def test_it_punishes_a_model_that_only_ever_answers_the_common_label(self):
        """The figure Watch's signal choice is read on, and why.

        Three quarters of these labels are 'oom', so always answering 'oom'
        scores 0.75 on plain accuracy and 0.5 on macro -- which is what a
        two-class problem answered one way actually deserves.
        """
        always_oom = (("nan", 0.1), ("hang", 0.1), ("oom", 0.8))
        examples = [
            _choice("oom", key="a"),
            _choice("oom", key="b"),
            _choice("oom", key="c"),
            _choice("nan", key="d"),
        ]
        result = score(examples, [always_oom] * 4)
        assert result.accuracy == 0.75
        assert result.macro_accuracy == 0.5

    def test_a_class_absent_from_the_labels_is_not_counted(self):
        """Nothing is labelled 'hang' here, so it has no recall to average in."""
        result = score([_choice("nan")], [(("nan", 0.8), ("hang", 0.1), ("oom", 0.1))])
        assert result.macro_accuracy == 1.0

    def test_it_equals_accuracy_when_the_classes_are_balanced(self):
        examples = [_choice("nan", key="a"), _choice("oom", key="b")]
        distributions = [
            (("nan", 0.8), ("hang", 0.1), ("oom", 0.1)),
            (("nan", 0.8), ("hang", 0.1), ("oom", 0.1)),
        ]
        result = score(examples, distributions)
        assert result.accuracy == result.macro_accuracy == 0.5


def _candidate(
    mitigation: str, *, label: str, group: str, decision: str = "proposer_candidate"
) -> LabelledExample:
    """One per-candidate noul, carrying the group that makes it part of a ranking."""
    return LabelledExample(
        decision=decision,
        state=f"cells for {group}",
        question=Noul(question=f"Would applying {mitigation!r} help?"),
        label=label,
        join_key=f"{group}:{mitigation}",
        group=group,
    )


def _ranked(*probabilities: float) -> list[tuple[tuple[str, float], ...]]:
    return [(("false", 1.0 - p), ("true", p)) for p in probabilities]


class TestGroupTop1:
    """The score for "which one of these N", asked as N separate yes/no questions.

    Two corpora are that shape -- the probe agent's per-candidate nouls and the log
    finder's per-file nouls -- and both are imbalanced by construction, so accuracy
    over the rows rewards a model for choosing nothing. What the caller does is
    rank the group and take the front of it, and this is that.
    """

    def _group(self, winner: int, count: int = 5, group: str = "step0"):
        return [
            _candidate(f"m{index}", label="true" if index == winner else "false", group=group)
            for index in range(count)
        ]

    def test_ranking_the_right_candidate_first_scores_one(self):
        rows = self._group(winner=2)
        result = score(rows, _ranked(0.1, 0.2, 0.9, 0.3, 0.4))
        assert result.group_top1 == 1.0

    def test_ranking_the_wrong_candidate_first_scores_zero(self):
        rows = self._group(winner=2)
        result = score(rows, _ranked(0.95, 0.2, 0.9, 0.3, 0.4))
        assert result.group_top1 == 0.0

    def test_a_model_that_answers_false_to_everything_scores_zero_not_eighty_percent(self):
        """The whole reason this metric exists.

        One candidate in five is right, so refusing every one of them is 80%
        accurate over the rows while proposing nothing at all.
        """
        rows = self._group(winner=4)
        result = score(rows, _ranked(0.1, 0.1, 0.1, 0.1, 0.1))
        assert result.accuracy == 0.8
        assert result.group_top1 == 0.0

    def test_it_averages_over_groups_not_over_rows(self):
        """Two steps, one found and one missed, is 0.5 however many candidates each had."""
        first = self._group(winner=0, count=2, group="step0")
        second = self._group(winner=0, count=8, group="step1")
        result = score(
            first + second,
            _ranked(0.9, 0.1) + _ranked(0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        )
        assert result.group_top1 == 0.5

    def test_a_group_with_nothing_to_find_is_not_counted(self):
        """A step whose winner had already been tried has no right answer in it.

        Scoring it zero would punish a model for a gap in the corpus.
        """
        rows = [
            _candidate("a", label="false", group="step0"),
            _candidate("b", label="false", group="step0"),
        ]
        assert score(rows, _ranked(0.9, 0.1)).group_top1 == 0.0

    def test_rows_with_no_group_are_not_ranked(self):
        """A choice-shaped row stands alone, and a corpus written before groups existed."""
        result = score([_choice("nan")], [(("nan", 0.9), ("hang", 0.05), ("oom", 0.05))])
        assert result.group_top1 == 0.0

    def test_two_decisions_sharing_a_group_are_ranked_separately(self):
        """The stop noul and the candidate nouls share a step, and must not compete.

        Every row from one step carries the same group so the split cannot divide
        them; ranking a stop question against a candidate question would be
        meaningless, so the grouping is by (decision, group).
        """
        rows = [
            _candidate("a", label="true", group="step0"),
            _candidate("b", label="false", group="step0"),
            _candidate("stop", label="false", group="step0", decision="proposer_stop"),
        ]
        # The stop row is the most confident 'true' in the group. If decisions were
        # pooled it would outrank the winning candidate and score this zero.
        result = score(rows, _ranked(0.8, 0.1, 0.99))
        assert result.group_top1 == 1.0

    def test_a_choice_shaped_label_never_contributes(self):
        """Only true/false rows are ranked, so a mitigation-name label is skipped."""
        rows = [
            LabelledExample(
                decision="proposer_mitigation",
                state="s",
                question=_SIGNAL,
                label="nan",
                join_key="step0",
                group="step0",
            )
        ]
        assert score(rows, [(("nan", 0.9), ("hang", 0.05), ("oom", 0.05))]).group_top1 == 0.0

    def test_an_empty_set_scores_zero(self):
        assert score([], []).group_top1 == 0.0

    def test_the_summary_names_it_when_there_is_a_ranking_to_report(self):
        rows = self._group(winner=0, count=3)
        assert "top-1" in score(rows, _ranked(0.9, 0.1, 0.1)).summary()

    def test_the_summary_leaves_it_out_when_there_is_not(self):
        """A corpus of standalone rows should not carry a 0.000 that means "not asked"."""
        standalone = score([_choice("nan")], [(("nan", 0.9), ("hang", 0.05), ("oom", 0.05))])
        assert "top-1" not in standalone.summary()


class TestThresholdedDecisions:
    """The metrics for a decision nobody takes an argmax of.

    Watch's gate compares ``p(healthy)`` against ``clean_threshold``; the probe
    agent stops when ``p(stop)`` reaches its own. Accuracy measures a decision
    neither of them makes.
    """

    def _healthy(self, label: str, *, key: str) -> LabelledExample:
        return LabelledExample(
            decision="watch_healthy",
            state=f"delta {key}",
            question=_HEALTHY,
            label=label,
            join_key=key,
        )

    def _at(self, *rows: tuple[str, float]):
        examples = [self._healthy(label, key=f"j{i}") for i, (label, _) in enumerate(rows)]
        distributions = [(("false", 1.0 - p), ("true", p)) for _, p in rows]
        return score(examples, distributions)

    def test_a_false_clean_is_one_scored_at_or_above_the_threshold(self):
        """Two unhealthy deltas, one of which the gate would have skipped."""
        result = self._at(("false", 0.95), ("false", 0.10))
        assert result.thresholds["watch_healthy"] == 0.90
        assert result.false_positive_rate == 0.5

    def test_exactly_at_the_threshold_counts_as_acting(self):
        """``>=``, the same reading ``NoulAnswer.at`` and ``should_alert`` use."""
        assert self._at(("false", 0.90)).false_positive_rate == 1.0

    def test_just_below_the_threshold_does_not(self):
        assert self._at(("false", 0.8999)).false_positive_rate == 0.0

    def test_healthy_rows_do_not_enter_the_false_positive_rate(self):
        """It is conditioned on the label, which is what makes it base-rate free."""
        result = self._at(("true", 0.99), ("true", 0.99), ("false", 0.10))
        assert result.false_positive_rate == 0.0

    def test_coverage_is_the_share_of_healthy_deltas_actually_gated(self):
        """What the threshold buys. Without it a false-positive rate is unreadable."""
        result = self._at(("true", 0.95), ("true", 0.20), ("false", 0.10))
        assert result.true_positive_rate == 0.5

    def test_a_model_that_never_gates_is_perfectly_safe_and_worthless(self):
        """Which is why the verdict is not allowed to read the rate on its own."""
        result = self._at(("true", 0.1), ("true", 0.1), ("false", 0.1))
        assert result.false_positive_rate == 0.0
        assert result.true_positive_rate == 0.0
        assert result.safe_coverage == 0.0

    def test_a_model_that_always_gates_loses_the_budget_and_scores_zero(self):
        result = self._at(("true", 0.99), ("true", 0.99), ("false", 0.99))
        assert result.true_positive_rate == 1.0
        assert result.false_positive_rate == 1.0
        assert result.safe_coverage == 0.0

    def test_safe_coverage_is_the_coverage_earned_inside_the_budget(self):
        result = self._at(("true", 0.95), ("true", 0.95), ("false", 0.10))
        assert result.safe_coverage == 1.0

    def test_the_budget_is_a_cliff_rather_than_a_penalty(self):
        """Past the budget the decision is not worth shipping, so coverage stops buying.

        A model must not be able to trade missed alerts against LLM calls saved to
        clear a gate.
        """
        generous = self._at(("true", 0.99), ("true", 0.99), ("true", 0.99), ("false", 0.99))
        assert generous.true_positive_rate == 1.0
        assert generous.safe_coverage == 0.0

    def test_a_decision_with_no_threshold_is_excluded_rather_than_defaulted(self):
        """An argmaxed decision has no operating point, and inventing 0.5 is worse
        than reporting nothing."""
        result = score([_choice("nan")], [(("nan", 0.99), ("hang", 0.005), ("oom", 0.005))])
        assert "watch_signal" not in result.thresholds
        assert result.false_positive_rate == 0.0
        assert result.safe_coverage == 0.0

    def test_the_operating_point_is_overridable_for_a_sweep(self):
        """"What would a lower threshold have cost" is the question a sweep asks."""
        examples = [self._healthy("false", key="j0")]
        distributions = [(("false", 0.3), ("true", 0.7))]
        assert score(examples, distributions).false_positive_rate == 0.0
        loosened = score(examples, distributions, thresholds={"watch_healthy": 0.6})
        assert loosened.false_positive_rate == 1.0

    def test_a_sub_result_keeps_the_operating_point(self):
        """The gate reads per-decision sub-results, so losing it there reads as zero."""
        result = self._at(("false", 0.95))
        assert result.by_decision()["watch_healthy"].false_positive_rate == 1.0

    def test_the_summary_prints_the_rate_and_the_coverage_together(self):
        text = self._at(("true", 0.95), ("false", 0.95)).summary()
        assert "FPR@0.90" in text
        assert "coverage" in text

    def test_the_summary_omits_them_for_an_argmaxed_decision(self):
        assert "FPR@" not in score(
            [_choice("nan")], [(("nan", 0.9), ("hang", 0.05), ("oom", 0.05))]
        ).summary()


class TestWhyAccuracyIsWrongForAGate:
    """The defect, demonstrated on one model and two corpora.

    This is the whole argument for the metric above, and it is worth a test rather
    than a paragraph because the two corpora are not hypothetical: the first is
    what the Watch builder produces today, and the second is what it produces once
    clean deltas are archived at full length.
    """

    def _rows(self, labels: list[str], probability_for):
        examples = [
            LabelledExample(
                decision="watch_healthy",
                state=f"delta {index}",
                question=_HEALTHY,
                label=label,
                join_key=f"j{index}",
            )
            for index, label in enumerate(labels)
        ]
        distributions = [
            (("false", 1.0 - probability_for(label)), ("true", probability_for(label)))
            for label in labels
        ]
        return score(examples, distributions)

    #: A model that answers "not healthy" to everything: the gate that never gates.
    _NEVER = staticmethod(lambda _label: 0.05)
    #: A model that answers "healthy" to everything: the gate that skips failures.
    _ALWAYS = staticmethod(lambda _label: 0.99)

    def test_on_todays_corpus_accuracy_rewards_never_gating(self):
        """Mostly unhealthy, because healthy deltas are only 500-character excerpts
        and most get skipped for being shorter than the minimum."""
        labels = ["false"] * 8 + ["true"] * 2
        assert self._rows(labels, self._NEVER).accuracy == 0.8
        assert self._rows(labels, self._ALWAYS).accuracy == 0.2

    def test_on_the_archived_corpus_accuracy_rewards_gating_everything(self):
        """A healthy poll every minute and a failure once: the majority flips."""
        labels = ["true"] * 8 + ["false"] * 2
        assert self._rows(labels, self._ALWAYS).accuracy == 0.8
        assert self._rows(labels, self._NEVER).accuracy == 0.2

    def test_the_same_metric_crowns_opposite_do_nothing_models(self):
        """One sentence, two corpora, opposite winners, neither model changed."""
        mostly_unhealthy = ["false"] * 8 + ["true"] * 2
        mostly_healthy = ["true"] * 8 + ["false"] * 2
        assert (
            self._rows(mostly_unhealthy, self._NEVER).accuracy
            > self._rows(mostly_unhealthy, self._ALWAYS).accuracy
        )
        assert (
            self._rows(mostly_healthy, self._ALWAYS).accuracy
            > self._rows(mostly_healthy, self._NEVER).accuracy
        )

    def test_safe_coverage_refuses_both_on_either_corpus(self):
        """The metric the gate reads does not move with how the corpus was built."""
        for labels in (["false"] * 8 + ["true"] * 2, ["true"] * 8 + ["false"] * 2):
            assert self._rows(labels, self._NEVER).safe_coverage == 0.0
            assert self._rows(labels, self._ALWAYS).safe_coverage == 0.0

    def test_a_model_that_separates_the_classes_is_the_one_that_clears(self):
        labels = ["true"] * 8 + ["false"] * 2
        separating = self._rows(labels, lambda label: 0.97 if label == "true" else 0.02)
        assert separating.safe_coverage == 1.0
        assert separating.false_positive_rate == 0.0


class TestTheShippedOperatingPoints:
    """The copied thresholds, pinned against the numbers actually in force.

    ``aorta.laya.eval`` hard-codes them so it stays importable with no extras, on
    the same terms as ``aorta/cli/chat.py``'s provider list. The copy is only safe
    while something fails when it drifts.
    """

    def test_the_watch_threshold_matches_the_one_the_tier_gates_on(self):
        from aorta.cia.watch.watcher import DEFAULT_CLEAN_THRESHOLD

        assert DEFAULT_THRESHOLDS["watch_healthy"] == DEFAULT_CLEAN_THRESHOLD

    def test_the_stop_threshold_matches_the_one_the_proposer_stops_on(self):
        from aorta.agent.llm import DEFAULT_LAYA_STOP_THRESHOLD

        assert DEFAULT_THRESHOLDS["proposer_stop"] == DEFAULT_LAYA_STOP_THRESHOLD

    def test_every_thresholded_decision_has_a_budget(self):
        """``safe_coverage`` needs both, and silently reads zero with only one."""
        assert set(DEFAULT_THRESHOLDS) == set(FALSE_POSITIVE_BUDGET)

    def test_the_budgets_are_the_strict_reading_until_somebody_owns_them(self):
        """Policy, not measurement. Loosening one is a decision with an owner.

        Pinned so that a non-zero budget cannot arrive without this test being
        edited, which is the moment to ask whose call it was.
        """
        assert set(FALSE_POSITIVE_BUDGET.values()) == {0.0}


class TestBrier:
    def test_a_perfect_confident_answer_scores_zero(self):
        result = score([_choice("nan")], [(("nan", 1.0), ("hang", 0.0), ("oom", 0.0))])
        assert result.brier == pytest.approx(0.0, abs=1e-9)

    def test_it_is_the_summed_squared_error_over_the_whole_vector(self):
        """0.5^2 + 0.3^2 + 0.2^2 with the target on 'nan' is 0.25+0.09+0.04 = 0.38."""
        result = score([_choice("nan")], [(("nan", 0.5), ("hang", 0.3), ("oom", 0.2))])
        assert result.brier == pytest.approx(0.25 + 0.09 + 0.04)

    def test_a_noul_is_scored_on_both_sides(self):
        """Documented in distribution_of: a noul's Brier is twice the two-class one.

        p(true)=0.8 with the label 'true' gives 0.2^2 on each side, so 0.08 rather
        than the 0.04 a two-class Brier would report. Pinned because it is the
        reason Brier is compared within a question shape and never across.
        """
        result = score([_noul("true")], [(("false", 0.2), ("true", 0.8))])
        assert result.brier == pytest.approx(0.08)

    def test_a_confident_wrong_answer_is_punished_more_than_an_unsure_one(self):
        confident = score([_choice("oom")], [(("nan", 0.98), ("hang", 0.01), ("oom", 0.01))])
        unsure = score([_choice("oom")], [(("nan", 0.4), ("hang", 0.3), ("oom", 0.3))])
        assert confident.brier > unsure.brier


class TestECE:
    def test_perfect_calibration_scores_zero(self):
        """Ten answers at 0.8 confidence, eight of them right."""
        distribution = (("false", 0.2), ("true", 0.8))
        examples = [_noul("true" if index < 8 else "false", key=f"j{index}") for index in range(10)]
        result = score(examples, [distribution] * 10)
        assert result.accuracy == 0.8
        assert result.ece == pytest.approx(0.0, abs=1e-9)

    def test_overconfidence_is_the_gap_between_confidence_and_correctness(self):
        """Ten answers at 0.9, five right: the gap is 0.4 and every row is one bin."""
        distribution = (("false", 0.1), ("true", 0.9))
        examples = [_noul("true" if index < 5 else "false", key=f"j{index}") for index in range(10)]
        assert score(examples, [distribution] * 10).ece == pytest.approx(0.4)

    def test_bins_are_weighted_by_how_many_answers_fell_in_them(self):
        """Ten calibrated answers and one badly wrong one must not average 50/50.

        Ten rows sit in the 0.8 bin and are right eight times, so that bin
        contributes nothing. One row sits in the top bin at 1.0 and is wrong,
        contributing (1/11) * 1.0 = 0.0909. An unweighted mean over the two bins
        would report 0.5, which would condemn a model for one row in eleven.
        """
        calibrated = (("false", 0.2), ("true", 0.8))
        certain = (("false", 0.0), ("true", 1.0))
        examples = [_noul("true" if index < 8 else "false", key=f"j{index}") for index in range(10)]
        examples.append(_noul("false", key="j10"))
        result = score(examples, [calibrated] * 10 + [certain])
        assert result.ece == pytest.approx(1.0 / 11.0, abs=1e-6)

    def test_an_empty_set_scores_zero(self):
        assert score([], []).ece == 0.0

    def test_the_bin_count_is_adjustable(self):
        distribution = (("false", 0.1), ("true", 0.9))
        examples = [_noul("true" if index < 5 else "false", key=f"j{index}") for index in range(10)]
        result = score(examples, [distribution] * 10)
        assert result.ece_at(1) == pytest.approx(0.4)
        assert result.ece_at(50) == pytest.approx(0.4)


class TestTemperature:
    def test_temperature_one_changes_nothing(self):
        distribution = (("nan", 0.5), ("hang", 0.3), ("oom", 0.2))
        scaled = apply_temperature(distribution, 1.0)
        for (name, before), (also, after) in zip(distribution, scaled):
            assert name == also
            assert after == pytest.approx(before)

    def test_a_high_temperature_flattens_toward_uniform(self):
        flat = apply_temperature((("a", 0.9), ("b", 0.1)), 1000.0)
        assert dict(flat)["a"] == pytest.approx(0.5, abs=0.01)

    def test_a_low_temperature_sharpens_toward_the_argmax(self):
        sharp = apply_temperature((("a", 0.6), ("b", 0.4)), 0.01)
        assert dict(sharp)["a"] == pytest.approx(1.0, abs=1e-6)

    def test_it_preserves_the_ranking(self):
        """Temperature scaling cannot change which answer is given, only its probability.

        This is why a refit moves ECE and Brier and leaves accuracy alone, and
        why reporting a pre-refit ECE would fail a model for something one scalar
        fixes.
        """
        for temperature in (0.1, 0.5, 2.0, 10.0):
            scaled = apply_temperature((("a", 0.5), ("b", 0.3), ("c", 0.2)), temperature)
            assert [name for name, _ in sorted(scaled, key=lambda p: -p[1])] == ["a", "b", "c"]

    def test_a_saturated_answer_does_not_become_infinite(self):
        """A hard 1.0 has an infinite logit, which would make the whole fit degenerate."""
        scaled = apply_temperature((("a", 1.0), ("b", 0.0)), 5.0)
        assert all(math.isfinite(value) for _, value in scaled)
        assert sum(value for _, value in scaled) == pytest.approx(1.0)

    def test_a_non_positive_temperature_is_refused(self):
        with pytest.raises(EvalError, match="must be positive"):
            apply_temperature((("a", 0.5), ("b", 0.5)), 0.0)


class TestFittingTemperature:
    def test_a_calibrated_set_fits_near_one(self):
        distribution = (("false", 0.2), ("true", 0.8))
        labels = ["true"] * 8 + ["false"] * 2
        assert fit_temperature([distribution] * 10, labels) == pytest.approx(1.0, abs=0.15)

    def test_an_overconfident_set_fits_above_one(self):
        """Answers at 0.99 that are right half the time need flattening."""
        distribution = (("false", 0.01), ("true", 0.99))
        labels = ["true"] * 5 + ["false"] * 5
        assert fit_temperature([distribution] * 10, labels) > 1.5

    def test_an_underconfident_set_fits_below_one(self):
        """Answers at 0.55 that are always right need sharpening."""
        distribution = (("false", 0.45), ("true", 0.55))
        assert fit_temperature([distribution] * 10, ["true"] * 10) < 1.0

    def test_the_fit_lowers_the_objective_it_minimises(self):
        distribution = (("false", 0.01), ("true", 0.99))
        labels = ["true"] * 5 + ["false"] * 5
        fitted = fit_temperature([distribution] * 10, labels)
        assert negative_log_likelihood([distribution] * 10, labels, fitted) < (
            negative_log_likelihood([distribution] * 10, labels, 1.0)
        )

    def test_an_empty_set_fits_the_identity(self):
        assert fit_temperature([], []) == 1.0

    def test_it_is_deterministic(self):
        """Two eval runs over one corpus must report the same ECE."""
        distribution = (("false", 0.1), ("true", 0.9))
        labels = ["true"] * 7 + ["false"] * 3
        first = fit_temperature([distribution] * 10, labels)
        assert first == fit_temperature([distribution] * 10, labels)

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(EvalError, match="1 distributions but 2 labels"):
            negative_log_likelihood([(("a", 1.0),)], ["a", "a"])


class TestPerBucketTemperature:
    def test_the_bucket_is_the_question_type_and_option_count(self):
        assert calibration_key(_noul("true")) == "noul:2"
        assert calibration_key(_choice("nan")) == "choice:3"

    def test_one_temperature_is_fitted_per_bucket(self):
        examples = [_noul("true", key="a"), _choice("nan", key="b")]
        distributions = [
            (("false", 0.1), ("true", 0.9)),
            (("nan", 0.8), ("hang", 0.1), ("oom", 0.1)),
        ]
        fitted = fit_temperatures(examples, distributions)
        assert sorted(fitted) == ["choice:3", "noul:2"]

    def test_a_bucket_with_no_fit_is_left_alone(self):
        """An unseen question shape must pass through rather than be guessed at."""
        examples = [_choice("nan")]
        distributions = [(("nan", 0.8), ("hang", 0.1), ("oom", 0.1))]
        got = recalibrate(examples, distributions, {"noul:2": 4.0})
        assert dict(got[0])["nan"] == pytest.approx(0.8)

    def test_a_refit_moves_ece_without_moving_accuracy(self):
        """The whole reason ECE is reported after a refit and not before."""
        overconfident = (("false", 0.01), ("true", 0.99))
        examples = [_noul("true" if index < 5 else "false", key=f"j{index}") for index in range(10)]
        distributions = [overconfident] * 10
        before = score(examples, distributions)
        temperatures = fit_temperatures(examples, distributions)
        after = score(examples, recalibrate(examples, distributions, temperatures))
        assert after.accuracy == before.accuracy
        assert after.ece < before.ece


class TestSplitting:
    def test_rows_sharing_a_join_key_land_on_the_same_side(self):
        """The leak this split exists to prevent.

        Every file in one directory listing shares a state and a job. A shuffled
        split would calibrate on some of a job's rows and report on the rest,
        which shows up as unusually good ECE and nothing else.
        """
        examples = [
            LabelledExample(
                decision="log_finder_useful",
                state="one listing",
                question=Noul(question="useful?"),
                label="true",
                join_key="job-7",
            )
            for _ in range(20)
        ]
        fit, report = split_examples(examples, holdout=0.5)
        assert not (fit and report)

    def test_it_is_stable_as_the_corpus_grows(self):
        first = [_noul("true", key=f"job-{index}") for index in range(40)]
        second = first + [_noul("false", key=f"job-{index}") for index in range(40, 60)]
        held_first = {e.join_key for e in split_examples(first)[1]}
        held_second = {e.join_key for e in split_examples(second)[1]}
        assert held_first <= held_second

    def test_it_splits_roughly_in_the_requested_proportion(self):
        examples = [_noul("true", key=f"job-{index}") for index in range(400)]
        _fit, report = split_examples(examples, holdout=0.25)
        assert 0.15 < len(report) / 400 < 0.35

    def test_an_out_of_range_holdout_is_refused(self):
        with pytest.raises(EvalError, match="between 0 and 1"):
            split_examples([_noul("true")], holdout=1.0)

    def test_the_group_outranks_the_join_key_as_the_split_boundary(self):
        """Twenty-one candidate nouls for one step must not be divided.

        Their join keys all differ -- one per candidate -- so splitting on those
        would calibrate on some of a step's candidates and report on the rest.
        """
        rows = [
            _candidate(f"m{index}", label="false", group="step0") for index in range(30)
        ]
        fit, report = split_examples(rows, holdout=0.5)
        assert not (fit and report)

    def test_both_question_shapes_for_one_step_land_together(self):
        """What makes a head-to-head a comparison rather than two half-corpora.

        The choice-shaped row and the noul-shaped rows for one step share a group,
        so the two shapes are scored over the same steps.
        """
        shared = "run:step0"
        rows: list[LabelledExample] = [
            LabelledExample(
                decision="proposer_mitigation",
                state="s",
                question=Choice(question="which?", options=("a", "b")),
                label="a",
                join_key=shared,
                group=shared,
            ),
            _candidate("a", label="true", group=shared),
            _candidate("b", label="false", group=shared),
        ]
        fit, report = split_examples(rows, holdout=0.5)
        assert not (fit and report)


class TestEvaluate:
    """The harness, driven by the fake so its plumbing is covered without weights.

    None of these assert a score. The fake answers from a hash, so a score here
    would describe a hash function; what is asserted is the arrangement -- that
    the holdout is honoured, that the model is named, and that an in-sample fit
    says so.
    """

    def test_it_reports_only_the_held_out_rows(self):
        examples = [_noul("true", key=f"job-{index}") for index in range(50)]
        result = evaluate(examples, FakeLayaPredictor(), holdout=0.2)
        assert 0 < result.count < len(examples)
        assert not result.calibrated_in_sample

    def test_it_names_the_model_that_answered(self):
        examples = [_noul("true", key=f"job-{index}") for index in range(50)]
        assert evaluate(examples, FakeLayaPredictor()).model_id == "fake"

    def test_an_in_sample_fit_is_flagged_and_explained(self):
        examples = [_noul("true", key=f"job-{index}") for index in range(4)]
        result = evaluate(examples, FakeLayaPredictor(), holdout=None)
        assert result.calibrated_in_sample
        assert result.count == 4
        assert any("optimistic" in note for note in result.notes)

    def test_a_corpus_too_small_to_split_says_so(self):
        with pytest.raises(EvalError, match="too small to split"):
            evaluate([_noul("true", key="only-one")], FakeLayaPredictor(), holdout=0.2)

    def test_an_empty_corpus_is_refused(self):
        with pytest.raises(EvalError, match="no examples"):
            evaluate([], FakeLayaPredictor())

    def test_questions_about_one_state_cost_one_call(self):
        """The property that makes a per-file listing affordable.

        Ten files in one directory are ten questions about one state, and the
        harness has to ask them together or it measures a cost the integration
        would never pay.
        """
        calls: list[int] = []

        class Counting(FakeLayaPredictor):
            def ask(self, states, questions):
                calls.append(len(states))
                return super().ask(states, questions)

        examples = [
            LabelledExample(
                decision="log_finder_useful",
                state="one listing",
                question=Noul(question=f"is {index} useful?"),
                label="false",
                join_key=f"listing:{index}",
            )
            for index in range(10)
        ]
        evaluate(examples, Counting(), holdout=None)
        assert calls == [1]

    def test_pinned_answers_reach_the_score(self):
        """How a later track's test controls the model without owning a checkpoint."""
        examples = [_noul("true", key=f"job-{index}") for index in range(4)]
        predictor = FakeLayaPredictor(
            pinned={_HEALTHY.question: NoulAnswer(probability=0.97)}
        )
        result = evaluate(examples, predictor, holdout=None)
        assert result.accuracy == 1.0


class TestLabelIntegrity:
    def test_a_label_outside_the_option_set_is_refused_at_construction(self):
        """Caught here, because at scoring time it reads as the model being wrong."""
        with pytest.raises(CorpusError, match="not one of the offered options"):
            LabelledExample(
                decision="watch_signal",
                state="log",
                question=_SIGNAL,
                label="WATCH_CLEAN",
                join_key="job-1",
            )

    def test_a_noul_takes_only_true_or_false(self):
        with pytest.raises(CorpusError, match="not one of the offered options"):
            LabelledExample(
                decision="watch_healthy",
                state="log",
                question=_HEALTHY,
                label="yes",
                join_key="job-1",
            )
