"""Phase 1's measurement: is a calibrated encoder good enough to gate on?

Deliberately not a pytest suite. It needs a corpus of real cluster runs and a
loaded checkpoint, neither of which exists in the mock-only CI gate, and
pretending otherwise would produce a test that is skipped everywhere and trusted
anyway. That is the same split ``aorta/chat/rag/eval.py`` makes, and it is made
the same way here: everything that turns answers into numbers is a pure function
with no model in it, so the metrics are pinned by
``tests/cia/test_laya_scoring.py``, and :func:`evaluate` -- the part that needs
weights -- is reachable only from the command line.

    aorta laya corpus watch --output watch.jsonl
    aorta laya eval --corpus watch.jsonl --backend laya-typed-decisions

**Three metrics, and one of them only means anything after a refit.** Accuracy
says how often the argmax is right, which is what a gate's control flow turns on.
Brier says whether the probability beside it is worth anything, which is the
whole reason for preferring a calibrated encoder over a self-reported confidence
float. ECE says whether that probability is *on the right scale* -- and a raw
encoder's is not: the card's own mean ECE is 0.466 before refitting a temperature
per question type and option count, and 0.081 after. Reporting a pre-refit ECE
would fail a model for a miscalibration that one scalar per question shape fixes.

**Why this does not import Laya's own metric helpers.** ``laya.common`` ships
``ece_score`` and ``confidence_from_probs``, and importing either would put torch
and numpy on the import path of the one module in this package that has to keep
working with no model installed -- which is also the module whose numbers nobody
should have to take on trust. The implementations below are stdlib and about
thirty lines, and they are the thing the unit tests pin.

**A number from a fake predictor is not a measurement.** :class:`FakeLayaPredictor`
answers from a hash. It exists so the harness and the scoring can be exercised,
and every result carries the ``model_id`` that produced it so a fake's output
cannot be mistaken for a run; the CLI refuses to write a results file for one.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from aorta.laya.corpus.schema import CorpusError, LabelledExample
from aorta.laya.predictor import Answer, ChoiceAnswer, LayaPredictor, Noul, NoulAnswer, Question

logger = logging.getLogger(__name__)

#: A probability per option, in the order the question offered them.
Distribution = tuple[tuple[str, float], ...]

#: Bin count for ECE. Ten equal-width bins over [0, 1] is the convention the
#: calibration literature reports against, so a figure here is comparable with a
#: published one. Laya's own helper uses fifteen; the count is exposed on
#: :meth:`EvalResult.ece_at` rather than argued about, because the ranking
#: between two models is not usually sensitive to it and the absolute value is.
DEFAULT_ECE_BINS = 10

#: Fraction of a corpus held out of the temperature fit. The card's own
#: typed-decisions fine-tune used an 80/20 split, and matching it keeps the two
#: sets of numbers on the same footing.
DEFAULT_HOLDOUT = 0.2

#: Probabilities are clamped to this before being read as logits. A model that
#: returns a hard 0.0 or 1.0 has an infinite logit, and temperature scaling of an
#: infinity is an infinity however large the temperature -- so one saturated
#: answer would otherwise make the whole fit degenerate.
_EPSILON = 1e-9

#: The probability each thresholded decision is actually gated on in the tree.
#:
#: **Not every decision is an argmax, and for the two below the argmax is not what
#: anybody runs.** Watch's Laya tier skips the LLM when ``p(healthy)`` reaches
#: ``watch.laya.clean_threshold``; the probe agent stops when ``p(stop)`` reaches
#: ``DEFAULT_LAYA_STOP_THRESHOLD``. Scoring those on accuracy measures a decision
#: neither of them makes, and measures it in a way that moves with how the corpus
#: was built rather than with the model -- see :attr:`EvalResult.false_positive_rate`.
#:
#: Hard-coded here rather than imported, for the reason ``aorta/cli/chat.py``
#: hard-codes its provider list and ``tests/cia/test_watch_threshold.py`` pins the
#: shipped threshold against the YAML: the real numbers live in
#: ``aorta.cia.watch.watcher`` and ``aorta.agent.llm``, and importing either would
#: make this module -- the one that has to stay trustworthy on a machine with no
#: model and no extras on it -- need the ``[cia]`` extra to compute a mean.
#: ``tests/cia/test_laya_scoring.py`` fails if a number here and the shipped one
#: drift apart, which is the only thing that makes the copy safe.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "watch_healthy": 0.90,
    "proposer_stop": 0.75,
}

#: How many of the expensive mistakes a decision may make and still be worth
#: shipping, as a fraction of the examples labelled ``false``.
#:
#: **Zero for both, and that is a policy choice rather than a measurement.** It is
#: the same kind of number as ``DEFAULT_LAYA_STOP_THRESHOLD`` and carries the same
#: warning: nothing here derived it from data, and it must be re-derived by
#: whoever owns the alerting SLA. Zero is the strictest possible reading and it is
#: chosen because neither error budget has an owner yet, and inventing a tolerance
#: -- five percent of missed alerts sounds modest and nobody agreed to it -- would
#: be worse than choosing the conservative end and saying so.
#:
#: For Watch the strict reading is close to forced: a false clean does not delay an
#: alert, it loses it, because a healthy verdict commits the log cursor and a NaN
#: is usually printed once. For the proposer a false stop ends an investigation
#: and writes a report nobody went on to test, which is recoverable by re-running
#: and so genuinely admits a non-zero budget once someone sets one.
#:
#: The small-sample caveat is real and is not a reason to loosen it: zero false
#: cleans over a dozen examples is weak evidence of safety. It is evidence of
#: *not having seen* the failure, which is the honest thing for a gate to require
#: before it is switched on.
FALSE_POSITIVE_BUDGET: dict[str, float] = {
    "watch_healthy": 0.0,
    "proposer_stop": 0.0,
}

#: The label a thresholded decision's action is authorised by, and its opposite.
#: A noul's distribution is over these two names; see :func:`distribution_of`.
_TRUE, _FALSE = "true", "false"


class EvalError(RuntimeError):
    """The corpus or the answers cannot be scored as given."""


# ── turning answers into distributions ─────────────────────────────────────


def distribution_of(question: Question, answer: Answer) -> Distribution:
    """One answer as a probability per offered option.

    A noul becomes a two-option distribution over ``false`` and ``true``. That
    is what makes one scorer enough for both question types, and it has one
    consequence worth stating: the Brier score of a noul computed this way is
    twice the conventional two-class figure, because the error is counted once
    on each side. Brier is therefore compared within a question shape, which is
    also how the temperature is fitted.
    """
    if isinstance(answer, NoulAnswer):
        if not isinstance(question, Noul):
            raise EvalError(f"a noul answer was given for a {type(question).__name__} question")
        probability = _clamped(answer.probability)
        return (("false", 1.0 - probability), ("true", probability))
    if not isinstance(answer, ChoiceAnswer):
        raise EvalError(f"cannot score a {type(answer).__name__}")
    offered = list(question.options)
    got = [name for name, _ in answer.probabilities]
    if got != offered:
        raise EvalError(
            f"answer options {got} do not match the question's {offered}; an answer "
            "scored against a different option set is not an answer to that question"
        )
    return tuple((name, _clamped(value)) for name, value in answer.probabilities)


def _clamped(value: float) -> float:
    return min(1.0 - _EPSILON, max(_EPSILON, float(value)))


def _normalised(distribution: Distribution) -> Distribution:
    total = sum(probability for _, probability in distribution)
    if total <= 0.0:
        # Every option at zero is not a distribution. Uniform is the only
        # answer that does not silently favour whichever option came first.
        count = len(distribution) or 1
        return tuple((name, 1.0 / count) for name, _ in distribution)
    return tuple((name, probability / total) for name, probability in distribution)


# ── temperature scaling ────────────────────────────────────────────────────


def calibration_key(example: LabelledExample) -> str:
    """The bucket a temperature is fitted for: question type and option count.

    Per the card, which fits one temperature per (question type, option count).
    Keyed on the exact count rather than the library's coarser buckets, because
    our option counts are small and known -- two for every noul, six for Watch's
    signal, and as many as the candidate list for the proposer -- so there is
    nothing to be gained by grouping a six-way with a ten-way.

    The refit is not cosmetic for the proposer. Its candidate list is 21
    registered mitigations, which falls in the library's ``choice:11+`` bucket,
    and that is the bucket whose shipped temperature Laya 0.3.5 had to clamp
    because the fitted value sharpens rather than softens. A temperature fitted
    here on our own corpus supersedes the shipped one for every figure this module
    reports -- but it cannot undo a sharpening the library already applied before
    handing the probabilities back, which is why the ``[laya]`` extra floors at
    the release that clamps rather than at the one that does not.
    """
    kind = "noul" if isinstance(example.question, Noul) else "choice"
    return f"{kind}:{len(example.question.options)}"


def apply_temperature(distribution: Distribution, temperature: float) -> Distribution:
    """*distribution* re-softmaxed at *temperature*. Pure.

    Temperature 1.0 returns the distribution unchanged (up to renormalisation),
    above 1.0 flattens it, below 1.0 sharpens it. Working in log space rather
    than on the probabilities directly is what makes this the same operation the
    model would have applied to its own logits, so a temperature fitted here
    means the same thing as one baked into a checkpoint.
    """
    if temperature <= 0.0:
        raise EvalError(f"temperature must be positive, got {temperature}")
    scaled = [math.log(_clamped(p)) / temperature for _, p in distribution]
    if not scaled:
        return distribution
    highest = max(scaled)
    weights = [math.exp(value - highest) for value in scaled]
    total = sum(weights)
    return tuple(
        (name, weight / total)
        for (name, _), weight in zip(distribution, weights, strict=True)
    )


def negative_log_likelihood(
    distributions: Sequence[Distribution],
    labels: Sequence[str],
    temperature: float = 1.0,
) -> float:
    """Mean NLL of the labels under the distributions at *temperature*. Pure.

    The objective the temperature is fitted against, and a strictly proper
    scoring rule, which is the property that stops the fit having a degenerate
    optimum: no temperature can improve it by making the answers less honest.
    """
    if len(distributions) != len(labels):
        raise EvalError(f"{len(distributions)} distributions but {len(labels)} labels")
    if not distributions:
        return 0.0
    total = 0.0
    for distribution, label in zip(distributions, labels, strict=True):
        scaled = dict(apply_temperature(distribution, temperature))
        total -= math.log(_clamped(scaled.get(label, 0.0)))
    return total / len(distributions)


def fit_temperature(
    distributions: Sequence[Distribution],
    labels: Sequence[str],
    *,
    low: float = 0.05,
    high: float = 20.0,
) -> float:
    """The temperature minimising NLL on this data. Pure and deterministic.

    A coarse log-spaced sweep to bracket the minimum, then a ternary search
    inside the bracket. NLL is convex in the inverse temperature, so the
    bracketing sweep cannot land on a local minimum that is not the global one,
    and the refinement is only buying decimal places.

    Deterministic matters more than fast: two eval runs over one corpus have to
    produce the same temperature, or the ECE they report differs for a reason
    that has nothing to do with the model.

    Returns 1.0 for an empty set, which is the identity: refusing to fit
    anything is the right answer when there is nothing to fit on.
    """
    if not distributions:
        return 1.0
    steps = 40
    ratio = (high / low) ** (1.0 / (steps - 1))
    grid = [low * ratio**index for index in range(steps)]
    scores = [negative_log_likelihood(distributions, labels, value) for value in grid]
    best = min(range(steps), key=lambda index: scores[index])
    left = grid[max(0, best - 1)]
    right = grid[min(steps - 1, best + 1)]
    for _ in range(40):
        if right - left < 1e-4:
            break
        first = left + (right - left) / 3.0
        second = right - (right - left) / 3.0
        if negative_log_likelihood(distributions, labels, first) <= negative_log_likelihood(
            distributions, labels, second
        ):
            right = second
        else:
            left = first
    return (left + right) / 2.0


def fit_temperatures(
    examples: Sequence[LabelledExample], distributions: Sequence[Distribution]
) -> dict[str, float]:
    """One temperature per :func:`calibration_key`. Pure."""
    if len(examples) != len(distributions):
        raise EvalError(f"{len(examples)} examples but {len(distributions)} distributions")
    grouped: dict[str, tuple[list[Distribution], list[str]]] = defaultdict(lambda: ([], []))
    for example, distribution in zip(examples, distributions, strict=True):
        bucket = grouped[calibration_key(example)]
        bucket[0].append(distribution)
        bucket[1].append(example.label)
    return {
        key: fit_temperature(group, labels) for key, (group, labels) in sorted(grouped.items())
    }


def recalibrate(
    examples: Sequence[LabelledExample],
    distributions: Sequence[Distribution],
    temperatures: dict[str, float],
) -> list[Distribution]:
    """Apply the per-bucket temperatures. A bucket with no fit is left alone. Pure."""
    return [
        apply_temperature(distribution, temperatures.get(calibration_key(example), 1.0))
        for example, distribution in zip(examples, distributions, strict=True)
    ]


# ── scoring ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoredRow:
    """One example, and what was answered for it."""

    example: LabelledExample
    distribution: Distribution

    @property
    def predicted(self) -> str:
        return max(self.distribution, key=lambda pair: pair[1])[0]

    @property
    def confidence(self) -> float:
        """p(:attr:`predicted`) -- what a threshold compares and ECE bins on."""
        return max((probability for _, probability in self.distribution), default=0.0)

    @property
    def label_probability(self) -> float:
        """p(the right answer). What NLL and Brier are built out of."""
        return dict(self.distribution).get(self.example.label, 0.0)

    @property
    def correct(self) -> bool:
        return self.predicted == self.example.label


@dataclass
class EvalResult:
    """Scores over a corpus, plus every row.

    The rows are not optional colour, for the reason the retrieval eval gives:
    an aggregate that moved 0.03 says nothing about which decision moved, and
    which decision moved is the only actionable part of a comparison. Here it
    also decides the outcome -- Phase 1 may pass for one track and fail for
    another, and re-sequencing around whichever track cleared needs the split.
    """

    model_id: str = ""
    rows: list[ScoredRow] = field(default_factory=list)
    temperatures: dict[str, float] = field(default_factory=dict)
    #: The probability each thresholded decision is gated on, by decision slug.
    #:
    #: Populated by :func:`score` from :data:`DEFAULT_THRESHOLDS`, and overridable
    #: so a sweep can ask what a different operating point would have cost. A
    #: decision absent from here is one that is genuinely argmaxed, and its rows
    #: are excluded from the threshold-conditioned metrics rather than scored
    #: against an invented operating point.
    thresholds: dict[str, float] = field(default_factory=dict)
    #: True when the temperatures were fitted on the same rows being scored.
    #: An in-sample ECE flatters the model, so this travels with the number
    #: rather than being remembered by whoever ran it.
    calibrated_in_sample: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def accuracy(self) -> float:
        """Fraction of rows whose argmax is the label."""
        if not self.rows:
            return 0.0
        return sum(1 for row in self.rows if row.correct) / len(self.rows)

    @property
    def macro_accuracy(self) -> float:
        """Mean per-class recall over the classes that appear as labels.

        The figure to read on Watch's signal choice, and the reason is in the
        corpus: four Autopsy categories collapse onto ``WATCH_UNKNOWN_ERROR``, so
        plain accuracy over that label distribution is mostly a majority-class
        score. Averaging recall per class makes a model that only ever emits the
        common slug score like one.

        Classes absent from the labels are not counted. A class nothing is
        labelled with has no recall, and scoring it zero would punish a model for
        a gap in the corpus.
        """
        if not self.rows:
            return 0.0
        hits: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        for row in self.rows:
            totals[row.example.label] += 1
            if row.correct:
                hits[row.example.label] += 1
        return sum(hits[label] / total for label, total in totals.items()) / len(totals)

    def _thresholded(self, label: str) -> list[tuple[ScoredRow, float]]:
        """Rows labelled *label* whose decision has an operating threshold."""
        return [
            (row, self.thresholds[row.example.decision])
            for row in self.rows
            if row.example.decision in self.thresholds and row.example.label == label
        ]

    @property
    def false_positive_rate(self) -> float:
        """Of the rows that should not have been acted on, the fraction that were.

        The number that actually decides a thresholded decision, and for Watch's
        gate it is the false-*clean* rate: of the deltas an Autopsy went on to
        diagnose as a real failure, the fraction this model scored at or above
        ``clean_threshold`` and would therefore have skipped the LLM call on. For
        the proposer it is the false-*stop* rate.

        **Accuracy is not merely imprecise here, it is unstable in a way that
        rewards opposite do-nothing strategies.** The healthy class of the Watch
        corpus comes only from ``watchdog_ok`` excerpts today, most of which are
        skipped for being too short, so the corpus is mostly unhealthy and
        "always answer unhealthy" wins on accuracy -- a gate that never gates.
        Once clean deltas are archived at full length, a healthy poll happens every
        minute and a failure happens once, the majority flips, and "always answer
        healthy" wins -- a gate that skips every LLM call including the failures.
        The same metric, the same model, opposite winners, decided by which corpus
        somebody built.

        It is also the wrong shape. The gate never takes an argmax; it compares
        p(healthy) against a threshold, and the two errors cost wildly different
        amounts. Declining to gate costs one LLM call. A false clean costs an alert
        outright rather than delaying one, because a healthy verdict commits the
        log cursor and a NaN is usually printed once.

        Lower is better, which is why ``aorta.laya.gate.LOWER_IS_BETTER`` exists.
        Read it beside :attr:`true_positive_rate`: on its own a model that never
        acts scores a perfect 0.0 and saves nothing, which is what
        :attr:`safe_coverage` exists to stop the verdict rewarding.

        Returns 0.0 when no row qualifies. That reads as "nothing false was
        cleared", which is true of an empty set and is why a decision may only be
        gated on this metric if it has a threshold -- asserted in
        ``tests/cia/test_laya_gate.py`` rather than left to a sentinel.
        """
        negatives = self._thresholded(_FALSE)
        if not negatives:
            return 0.0
        acted = sum(
            1
            for row, threshold in negatives
            if dict(row.distribution).get(_TRUE, 0.0) >= threshold
        )
        return acted / len(negatives)

    @property
    def true_positive_rate(self) -> float:
        """Of the rows that should have been acted on, the fraction that were.

        What the threshold buys, and the figure that makes
        :attr:`false_positive_rate` readable. For Watch it is the share of healthy
        deltas the gate actually skips the LLM on, which is the entire saving; a
        gate with no false cleans and no coverage is a gate nobody should install.
        """
        positives = self._thresholded(_TRUE)
        if not positives:
            return 0.0
        acted = sum(
            1
            for row, threshold in positives
            if dict(row.distribution).get(_TRUE, 0.0) >= threshold
        )
        return acted / len(positives)

    @property
    def safe_coverage(self) -> float:
        """:attr:`true_positive_rate`, but only if the false-positive budget held.

        The number a thresholded decision is gated on, because neither half is a
        criterion by itself. A model that never acts has a perfect
        false-positive rate and saves nothing; a model that always acts has
        perfect coverage and loses alerts. The engineering question is how much
        can be skipped while staying inside the budget, and that is one number
        that is higher-better, so the verdict cannot be cleared by doing nothing.

        Zero when the budget is exceeded, which is deliberately a cliff rather
        than a penalty. The budget is the point at which the decision stops being
        worth shipping at all, so a model past it should not be able to trade
        coverage against missed alerts to clear a gate.

        A decision with no entry in :data:`FALSE_POSITIVE_BUDGET` is not one this
        metric describes, and scores 0.0.

        On a result mixing several decisions the strictest budget applies, because
        the two rates above are aggregates and splitting them per decision here
        would duplicate what :meth:`by_decision` already does. The gate always
        reads a per-decision sub-result, where the aggregate *is* the decision's.
        """
        budgets = [
            FALSE_POSITIVE_BUDGET[decision]
            for decision in {row.example.decision for row in self.rows}
            if decision in FALSE_POSITIVE_BUDGET
        ]
        if not budgets:
            return 0.0
        if self.false_positive_rate > min(budgets):
            return 0.0
        return self.true_positive_rate

    @property
    def group_top1(self) -> float:
        """Fraction of grouped decisions whose top-ranked row is the right one.

        The score for a decision that is really "which one of these N", asked as N
        separate yes/no questions. Two corpora are shaped that way: the probe
        agent's per-candidate nouls, and the log finder's per-file nouls over a
        directory listing.

        Accuracy is the wrong number for both, and badly so rather than subtly. One
        candidate in twenty-one is labelled true, so a model that answers false to
        everything scores 0.95 while choosing nothing at all -- and the thing it is
        being asked to do is choose. Ranking the group by p(true) and checking
        whether the top one is the labelled one is the decision the caller actually
        makes: the proposer proposes the best candidate, and the log finder takes
        ``max_files`` off the front of a ranked listing.

        Grouped by (decision, group), never by group alone. A step contributes its
        choice-shaped row, its candidate nouls, its stop noul and its category row
        under one group -- deliberately, so the split cannot divide them -- and
        ranking a stop noul against a candidate noul would be meaningless.

        Only rows whose label is ``true`` or ``false`` are ranked, and only groups
        holding at least one ``true`` count. A group with nothing to find cannot be
        got right, and scoring it zero would punish a model for a gap in the corpus.
        Returns 0.0 when no group qualifies, which is the same "nothing to report"
        every other metric here returns for an empty set.
        """
        groups: dict[tuple[str, str], list[ScoredRow]] = defaultdict(list)
        for row in self.rows:
            if row.example.group and row.example.label in ("true", "false"):
                groups[(row.example.decision, row.example.group)].append(row)
        scored = 0
        hits = 0
        for rows in groups.values():
            if not any(row.example.label == "true" for row in rows):
                continue
            scored += 1
            # Ties resolve to the first row, which for a baseline that cannot
            # discriminate means the corpus's own order decides. That is honest --
            # a uniform ranker has picked nothing -- and it averages to 1/N across
            # groups because the right answer's position varies between them.
            best = max(rows, key=lambda row: dict(row.distribution).get("true", 0.0))
            if best.example.label == "true":
                hits += 1
        return hits / scored if scored else 0.0

    @property
    def brier(self) -> float:
        """Mean squared error of the whole probability vector.

        Summed over options, so a noul scores twice the conventional two-class
        figure -- see :func:`distribution_of`. Read it within a decision, which
        is what :meth:`by_decision` is for.
        """
        if not self.rows:
            return 0.0
        total = 0.0
        for row in self.rows:
            for option, probability in row.distribution:
                target = 1.0 if option == row.example.label else 0.0
                total += (probability - target) ** 2
        return total / len(self.rows)

    @property
    def ece(self) -> float:
        return self.ece_at(DEFAULT_ECE_BINS)

    def ece_at(self, bins: int = DEFAULT_ECE_BINS) -> float:
        """Expected calibration error over *bins* equal-width confidence bins.

        The gap between how confident the answers were and how often they were
        right, weighted by how many answers fell in each bin. Zero means a
        0.7-confidence answer is right 70% of the time, which is the property a
        threshold needs and the one a self-reported LLM confidence does not have.
        """
        if not self.rows or bins < 1:
            return 0.0
        buckets: dict[int, list[ScoredRow]] = defaultdict(list)
        for row in self.rows:
            index = min(bins - 1, max(0, int(row.confidence * bins)))
            buckets[index].append(row)
        error = 0.0
        for rows in buckets.values():
            mean_confidence = sum(row.confidence for row in rows) / len(rows)
            mean_correct = sum(1 for row in rows if row.correct) / len(rows)
            error += (len(rows) / len(self.rows)) * abs(mean_confidence - mean_correct)
        return error

    def by_decision(self) -> dict[str, EvalResult]:
        """One sub-result per decision slug, carrying this result's provenance."""
        grouped: dict[str, list[ScoredRow]] = defaultdict(list)
        for row in self.rows:
            grouped[row.example.decision].append(row)
        return {
            decision: EvalResult(
                model_id=self.model_id,
                rows=rows,
                temperatures=dict(self.temperatures),
                # Carried, or a sub-result would have no operating point and every
                # threshold-conditioned metric would silently read zero -- which
                # is what the gate reads.
                thresholds=dict(self.thresholds),
                calibrated_in_sample=self.calibrated_in_sample,
            )
            for decision, rows in sorted(grouped.items())
        }

    def _gated_decisions(self) -> list[str]:
        return sorted(
            {row.example.decision for row in self.rows} & set(self.thresholds)
        )

    def summary(self) -> str:
        suffix = " (temperature fitted in-sample)" if self.calibrated_in_sample else ""
        ranking = f", top-1 {self.group_top1:.3f}" if self.group_top1 else ""
        # Printed only where a threshold applies, and always as the pair. A
        # false-positive rate on its own reads as a safety figure, and a model
        # that never acts scores a perfect one.
        gated = self._gated_decisions()
        operating = ""
        if gated:
            at = "/".join(f"{self.thresholds[decision]:.2f}" for decision in gated)
            operating = (
                f", FPR@{at} {self.false_positive_rate:.3f}"
                f", coverage {self.true_positive_rate:.3f}"
            )
        return (
            f"{self.count} answer(s) from {self.model_id or 'an unnamed model'}: "
            f"accuracy {self.accuracy:.3f}, macro {self.macro_accuracy:.3f}"
            f"{ranking}{operating}, Brier {self.brier:.4f}, ECE {self.ece:.4f}{suffix}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "count": self.count,
            "accuracy": round(self.accuracy, 4),
            "macro_accuracy": round(self.macro_accuracy, 4),
            "group_top1": round(self.group_top1, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "true_positive_rate": round(self.true_positive_rate, 4),
            "safe_coverage": round(self.safe_coverage, 4),
            "thresholds": {
                decision: self.thresholds[decision] for decision in self._gated_decisions()
            },
            "brier": round(self.brier, 4),
            "ece": round(self.ece, 4),
            "calibrated_in_sample": self.calibrated_in_sample,
            "temperatures": {
                key: round(value, 4) for key, value in sorted(self.temperatures.items())
            },
            "notes": list(self.notes),
            "by_decision": {
                decision: {
                    "count": result.count,
                    "accuracy": round(result.accuracy, 4),
                    "macro_accuracy": round(result.macro_accuracy, 4),
                    "group_top1": round(result.group_top1, 4),
                    "false_positive_rate": round(result.false_positive_rate, 4),
                    "true_positive_rate": round(result.true_positive_rate, 4),
                    "safe_coverage": round(result.safe_coverage, 4),
                    "brier": round(result.brier, 4),
                    "ece": round(result.ece, 4),
                    "labels": dict(
                        sorted(Counter(row.example.label for row in result.rows).items())
                    ),
                }
                for decision, result in self.by_decision().items()
            },
        }


def score(
    examples: Sequence[LabelledExample],
    distributions: Sequence[Distribution],
    *,
    model_id: str = "",
    thresholds: dict[str, float] | None = None,
) -> EvalResult:
    """Score pre-computed distributions. Pure, so the metrics are unit-testable.

    *thresholds* defaults to :data:`DEFAULT_THRESHOLDS`, so a caller that does
    nothing still scores the thresholded decisions at the operating point the tree
    actually ships. Pass a narrowed or widened table to ask what a different one
    would have cost.
    """
    if len(examples) != len(distributions):
        raise EvalError(
            f"{len(examples)} examples but {len(distributions)} distributions"
        )
    rows = [
        ScoredRow(example=example, distribution=_normalised(distribution))
        for example, distribution in zip(examples, distributions, strict=True)
    ]
    return EvalResult(
        model_id=model_id,
        rows=rows,
        thresholds=dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds),
    )


# ── splitting ──────────────────────────────────────────────────────────────


def split_examples(
    examples: Sequence[LabelledExample], *, holdout: float = DEFAULT_HOLDOUT
) -> tuple[list[LabelledExample], list[LabelledExample]]:
    """Deterministically split into (fit, report) sets.

    Partitioned on a hash of the example's ``group``, falling back to its
    ``join_key``, rather than shuffled with a seeded RNG -- and that is not a
    stylistic preference. Several examples share one key: every file in one
    directory listing, both questions about one log delta, all twenty-one
    candidate nouls for one proposer step. A random split would put some of a
    job's rows in the fit set and the rest in the report set, so the model reports
    on a state it was calibrated against. That leak shows up as unusually good ECE
    and nothing else.

    ``group`` first, because it is the coarser of the two and the one that spans a
    whole decision. It is also what keeps a head-to-head honest: the proposer's
    choice-shaped row and its noul-shaped rows for one step carry the same group,
    so the two shapes are compared over the same steps rather than over two
    different halves of the corpus.

    Also stable as the corpus grows: adding jobs does not move the existing rows
    between sides, so two evals a week apart remain comparable.
    """
    if not 0.0 < holdout < 1.0:
        raise EvalError(f"holdout must be between 0 and 1, got {holdout}")
    threshold = int(holdout * (1 << 32))
    fit: list[LabelledExample] = []
    report: list[LabelledExample] = []
    for example in examples:
        digest = hashlib.blake2b(
            (example.group or example.join_key or example.state).encode("utf-8"),
            digest_size=4,
        ).digest()
        (report if int.from_bytes(digest, "big") < threshold else fit).append(example)
    return fit, report


# ── the harness ────────────────────────────────────────────────────────────


def answer_all(
    predictor: LayaPredictor, examples: Sequence[LabelledExample]
) -> list[Distribution]:
    """Ask *predictor* every example's question. One forward pass per example.

    Grouped by state so that several questions about one state -- which is what a
    log-finder listing produces, one per file -- cost the single forward pass
    they are meant to. A naive loop over examples would ask N times for N files
    in one directory and measure a cost the integration would not pay.
    """
    order: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        order[example.state].append(index)

    distributions: list[Distribution | None] = [None] * len(examples)
    for state, indices in order.items():
        questions = [examples[index].question for index in indices]
        answers = predictor.ask([state], questions)[0]
        if len(answers) != len(questions):
            raise EvalError(
                f"{type(predictor).__name__} returned {len(answers)} answers for "
                f"{len(questions)} questions"
            )
        for index, answer in zip(indices, answers, strict=True):
            distributions[index] = distribution_of(examples[index].question, answer)
    missing = [index for index, value in enumerate(distributions) if value is None]
    if missing:
        raise EvalError(f"no answer produced for example(s) {missing[:5]}")
    return [value for value in distributions if value is not None]


def evaluate(
    examples: Sequence[LabelledExample],
    predictor: LayaPredictor,
    *,
    holdout: float | None = DEFAULT_HOLDOUT,
) -> EvalResult:
    """Run *predictor* over *examples* and score it. Needs a model.

    With *holdout*, the temperatures are fitted on one side of the split and the
    scores reported on the other, which is the only arrangement in which the ECE
    means what its name says. Passing ``None`` fits and reports on everything and
    marks the result ``calibrated_in_sample``, which is useful for a corpus too
    small to split and misleading if that is not said out loud.
    """
    if not examples:
        raise EvalError("no examples to evaluate")

    raw = answer_all(predictor, examples)
    model_id = predictor.model_id()

    if holdout is None:
        temperatures = fit_temperatures(examples, raw)
        result = score(examples, recalibrate(examples, raw, temperatures), model_id=model_id)
        result.temperatures = temperatures
        result.calibrated_in_sample = True
        result.notes.append(
            "Temperatures were fitted on every row that is scored here, so the ECE is "
            "optimistic. Pass a holdout once the corpus is large enough to split."
        )
        return result

    fit_set, report_set = split_examples(examples, holdout=holdout)
    if not report_set or not fit_set:
        raise EvalError(
            f"a {holdout:.0%} holdout over {len(examples)} example(s) left one side empty; "
            "the corpus is too small to split, so pass holdout=None and read the ECE as "
            "in-sample"
        )
    index_of = {id(example): position for position, example in enumerate(examples)}
    fit_distributions = [raw[index_of[id(example)]] for example in fit_set]
    temperatures = fit_temperatures(fit_set, fit_distributions)
    report_distributions = [raw[index_of[id(example)]] for example in report_set]
    result = score(
        report_set,
        recalibrate(report_set, report_distributions, temperatures),
        model_id=model_id,
    )
    result.temperatures = temperatures
    result.notes.append(
        f"Temperatures fitted on {len(fit_set)} held-in example(s); scores are over the "
        f"{len(report_set)} held out."
    )
    return result


def load_corpus(path: str) -> list[LabelledExample]:
    """Read a corpus, re-raising its failures as this module's.

    One exception type for a caller that is about to print a message and stop.
    """
    from aorta.laya.corpus.schema import read_corpus

    try:
        return read_corpus(path)
    except CorpusError as exc:
        raise EvalError(str(exc)) from exc


__all__ = [
    "DEFAULT_ECE_BINS",
    "DEFAULT_HOLDOUT",
    "DEFAULT_THRESHOLDS",
    "FALSE_POSITIVE_BUDGET",
    "Distribution",
    "EvalError",
    "EvalResult",
    "ScoredRow",
    "answer_all",
    "apply_temperature",
    "calibration_key",
    "distribution_of",
    "evaluate",
    "fit_temperature",
    "fit_temperatures",
    "load_corpus",
    "negative_log_likelihood",
    "recalibrate",
    "score",
    "split_examples",
]
