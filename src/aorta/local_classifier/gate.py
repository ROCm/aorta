"""The Phase 1 go/no-go: three baselines, a token census, and CPU latency.

An accuracy figure on its own decides nothing. What decides whether any of the
three integration tracks is worth building is whether a calibrated encoder beats
what is already in the tree, and there are three things already in the tree:

* **Majority class.** The floor. Watch's label distribution collapses four
  Autopsy categories onto one slug, so a model that only ever emits the common
  answer can score well; this is the number that exposes it.
* **The existing regex.** ``sanitizer_assessment()`` answers a machine-readable
  sanitizer summary with confidence 1.0 and no model at all. Where it applies it
  is unbeatable, so the useful figure is its *coverage*: a local-classifier tier is only
  worth adding for the deltas it does not answer.
* **The current DSPy assessment's own agreement with the eventual Autopsy
  category.** This is the real bar, and the one the kill criterion is written
  against. Beating a majority baseline proves nothing about replacing a ReAct
  call that already works.

Two measurements sit beside them because either can invalidate the plan on its
own. If most Watch deltas exceed the context window, chunking is required and the
single-forward-pass latency claim erodes. And the card's 33 ms is a T4 figure: on
CPU, which is where this runs, the same card says 193-464 ms, and that has to be
confirmed on the hardware rather than quoted.

**Nothing here will produce a number without the thing that number is about.** A
census needs the model's own tokenizer and raises without it, latency is timed
with ``perf_counter`` around real calls, and every result records the
``model_id`` that produced it. :func:`assert_measurable` exists purely to stop a
fake predictor's output being filed as a result.
"""

from __future__ import annotations

import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from aorta.local_classifier.corpus.schema import LabelledExample
from aorta.local_classifier.eval import (
    DEFAULT_HOLDOUT,
    Distribution,
    EvalResult,
    evaluate,
    score,
    split_examples,
)
from aorta.local_classifier.predictor import Calibration, DecisionPredictor, Question

#: Which metric decides each decision, because they do not all decide the same
#: way and the wrong one makes a gate meaningless rather than merely imprecise.
#:
#: ``watch_signal`` is macro accuracy: its labels are dominated by the slug four
#: Autopsy categories collapse onto, so plain accuracy there rewards a model for
#: learning the corpus's imbalance.
#:
#: ``proposer_candidate`` and ``log_finder_useful`` are ``group_top1``. Both ask
#: N yes/no questions where the decision is "which of these N", and both are
#: imbalanced by construction -- one candidate in twenty-one is labelled true, and
#: a job directory holds far more files than logs. Accuracy over the rows scores a
#: model that answers false to everything at 0.95 while it chooses nothing, which
#: is the opposite of the thing being measured. ``log_finder_useful`` was accuracy
#: in the first version of this file and that was a mistake in the same direction;
#: the builder warned about it in prose where a metric was what was needed.
#:
#: ``watch_healthy`` and ``proposer_stop`` are ``safe_coverage``, which is the
#: third shape: both are *thresholded* rather than argmaxed, so the decision being
#: scored is "did p reach the operating point", not "which label won". Accuracy
#: was wrong for them in the same direction again and worse, because its answer
#: moves with how the corpus was built -- today the Watch corpus is mostly
#: unhealthy and "always answer unhealthy" wins, and once clean deltas are archived
#: the majority flips and "always answer healthy" wins. One metric, two opposite
#: do-nothing winners. ``safe_coverage`` is how much of the saving a model
#: captures while staying inside its false-positive budget, which neither degenerate
#: strategy clears. See :attr:`aorta.local_classifier.eval.EvalResult.false_positive_rate`.
#:
#: Everything else is plain accuracy, which is what the control flow turns on.
#: ``proposer_category`` and ``watch_signal`` are genuinely argmaxed -- Autopsy's
#: category is a choice over eleven options, and the number ``probe.py`` thresholds
#: at 0.85 is its *confidence*, which no builder emits a decision for yet. When
#: Phase 4 gives it one, it belongs in this table as a thresholded decision too.
PRIMARY_METRIC: dict[str, str] = {
    "watch_signal": "macro_accuracy",
    "watch_healthy": "safe_coverage",
    "proposer_mitigation": "accuracy",
    "proposer_candidate": "group_top1",
    "proposer_stop": "safe_coverage",
    "proposer_category": "accuracy",
    "log_finder_useful": "group_top1",
}

#: Metrics a smaller number is better on.
#:
#: The gate compared every metric as higher-is-better until a false-positive rate
#: arrived, which was a latent bug rather than a simplification: naming ``brier``
#: or ``ece`` in the table above would have inverted the verdict silently, and both
#: are figures somebody might reasonably want to gate on. Listed rather than
#: inferred, because guessing a direction from a metric's name is how the next one
#: gets it wrong.
LOWER_IS_BETTER: frozenset[str] = frozenset({"false_positive_rate", "brier", "ece"})

#: Baselines read out of a corpus's recorded answers, and what each one is.
RECORDED_BASELINES: dict[str, str] = {
    "dspy_signal": "the ReAct assessment's own slug, as it answered on the day",
    "dspy_category": "the proposer's own category",
    "dspy_mitigation": "the mitigation the proposer proposed first",
    "dspy_stop": "whether the proposer chose to stop",
    "regex_signal": "sanitizer_assessment(), where the log was machine-readable",
}

#: The two context windows the checkpoints offer. The English encoder is 512 and
#: typed-decisions is 1024, so a delta over the smaller rules out one checkpoint
#: and a delta over the larger rules out both without chunking.
CONTEXT_WINDOWS: tuple[int, ...] = (512, 1024)

#: Batch sizes worth timing. One is what Watch's poll loop and the proposer pay.
#: Five is a small log-finder listing. Thirty is the deferred reranking idea, and
#: it is here because that is the number that decides whether it ever leaves the
#: backlog -- thirty chunks are thirty *states*, so thirty forward passes.
LATENCY_BATCHES: tuple[int, ...] = (1, 5, 30)


class NotMeasurableError(RuntimeError):
    """Asked for a measurement the available predictor cannot honestly produce."""


class Tokenizing(Protocol):
    """A predictor that can count tokens the way its own model counts them."""

    def token_count(self, text: str) -> int: ...


def assert_measurable(predictor: DecisionPredictor) -> None:
    """Refuse to treat a fake predictor's output as a result.

    The one place this package is opinionated about honesty rather than merely
    careful about it. ``FakeDecisionPredictor`` answers from a hash of its inputs,
    which is exactly what a plausible-looking result file needs in order to be
    indistinguishable from a real one: stable across runs, varied across
    examples, in range. Nothing downstream could tell, so the refusal is here.
    """
    if predictor.model_id() == "fake":
        raise NotMeasurableError(
            "the fake predictor answers from a hash of its inputs, so its accuracy, "
            "Brier, ECE and latency describe a hash function and not a model. Pass "
            "--backend laya-typed-decisions (which needs the [local-classifier] extra and the "
            "weights) to measure anything."
        )


# ── baselines ──────────────────────────────────────────────────────────────


def prior_distributions(
    fit: Sequence[LabelledExample], report: Sequence[LabelledExample]
) -> list[Distribution]:
    """The majority-class baseline, as a calibrated prior. Pure.

    The label frequencies of the *fit* set, per decision, applied to the report
    set -- not a one-hot on the most common label. A prior is the honest form of
    this baseline: a classifier that always answers the common slug and says
    "60%" while being right 60% of the time is perfectly calibrated, and a
    baseline scored on Brier and ECE has to be given that chance, or the encoder
    beats it on calibration by construction rather than on merit.

    Frequencies come from the fit side, so the floor is not measured with
    knowledge of the rows it is scored on.
    """
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for example in fit:
        counts[example.decision][example.label] += 1
    distributions: list[Distribution] = []
    for example in report:
        seen = counts.get(example.decision) or Counter()
        options = example.question.options
        total = sum(seen.get(option, 0) for option in options)
        if total:
            distributions.append(
                tuple((option, seen.get(option, 0) / total) for option in options)
            )
        else:
            # A decision with no fit-set rows has no prior, and uniform is the
            # only answer that does not invent one.
            width = len(options) or 1
            distributions.append(tuple((option, 1.0 / width) for option in options))
    return distributions


@dataclass(frozen=True)
class RecordedBaseline:
    """A baseline read out of the corpus, and how much of it it could answer."""

    name: str
    distributions: tuple[Distribution, ...]
    #: Rows the baseline actually answered. The rest were given a uniform
    #: distribution, which scores like an abstention rather than like an error.
    covered: int
    total: int

    @property
    def coverage(self) -> float:
        return self.covered / self.total if self.total else 0.0


def recorded_distributions(name: str, examples: Sequence[LabelledExample]) -> RecordedBaseline:
    """Turn a recorded answer into a distribution per example. Pure.

    A recorded answer is a hard label with no probability, so it becomes a
    near-one-hot -- unless the corpus also recorded the confidence the old system
    reported, in which case that is used. For the DSPy assessment it did, and
    using it is the point: ``should_alert`` compares that float against 0.70
    today, so how badly calibrated it turns out to be is the argument for
    replacing it.

    An answer outside the offered options -- a ``WATCH_CLEAN`` on a question that
    does not offer it, or an empty string where nothing was recorded -- becomes
    uniform and is counted as not covered. Scoring it as a wrong answer would
    make the existing regex look terrible on the logs it was never meant to read,
    when the useful fact about it is how many it does read.
    """
    distributions: list[Distribution] = []
    covered = 0
    for example in examples:
        answer = example.baseline(name)
        options = example.question.options
        width = len(options) or 1
        if answer is None or answer not in options:
            distributions.append(tuple((option, 1.0 / width) for option in options))
            continue
        covered += 1
        mass = _recorded_confidence(example, name, width)
        rest = (1.0 - mass) / (width - 1) if width > 1 else 0.0
        distributions.append(
            tuple((option, mass if option == answer else rest) for option in options)
        )
    return RecordedBaseline(
        name=name,
        distributions=tuple(distributions),
        covered=covered,
        total=len(examples),
    )


def _recorded_confidence(example: LabelledExample, name: str, width: int) -> float:
    """The confidence the old system reported, or near-certainty if it reported none.

    Clamped away from 1.0 because a hard baseline that is wrong at probability
    exactly 1.0 has an unbounded NLL, so a single mistake would dominate every
    aggregate it appears in. The clamp is loose enough to change no ranking.
    """
    prefix = name.split("_")[0]
    raw = example.baseline(f"{prefix}_confidence")
    floor = 1.0 / width
    try:
        value = float(raw) if raw else 0.99
    except ValueError:
        value = 0.99
    return min(0.99, max(floor, value))


def available_baselines(examples: Sequence[LabelledExample]) -> list[str]:
    """Which recorded baselines this corpus actually carries an answer for."""
    return sorted(
        name
        for name in RECORDED_BASELINES
        if any(example.baseline(name) for example in examples)
    )


# ── token census ───────────────────────────────────────────────────────────


@dataclass
class TokenCensus:
    """How long the states are, in the tokens the model will actually count."""

    model_id: str
    counts: list[int] = field(default_factory=list)

    @property
    def median(self) -> float:
        return statistics.median(self.counts) if self.counts else 0.0

    @property
    def largest(self) -> int:
        return max(self.counts) if self.counts else 0

    def fraction_over(self, window: int) -> float:
        if not self.counts:
            return 0.0
        return sum(1 for count in self.counts if count > window) / len(self.counts)

    def summary(self) -> str:
        parts = [f"over {window}: {self.fraction_over(window):.1%}" for window in CONTEXT_WINDOWS]
        return (
            f"{len(self.counts)} state(s), median {self.median:.0f} tokens, "
            f"longest {self.largest}; " + ", ".join(parts)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "states": len(self.counts),
            "median_tokens": self.median,
            "max_tokens": self.largest,
            "fraction_over": {
                str(window): round(self.fraction_over(window), 4) for window in CONTEXT_WINDOWS
            },
        }


def token_census(
    examples: Sequence[LabelledExample], tokenizer: Tokenizing, *, model_id: str = ""
) -> TokenCensus:
    """Count the tokens in every distinct state.

    Distinct, because a log-finder listing produces one example per file and all
    of them share one state. Counting per example would report a directory of
    forty files as forty long states and overstate the fraction over the window
    by exactly the imbalance of that corpus.
    """
    counter = getattr(tokenizer, "token_count", None)
    if counter is None:
        raise NotMeasurableError(
            f"{type(tokenizer).__name__} cannot count tokens, and a rule of thumb about "
            "characters per token is calibrated on prose rather than on logs full of "
            "timestamps and hex addresses. Load a real checkpoint to take this measurement."
        )
    seen: dict[str, int] = {}
    for example in examples:
        if example.state not in seen:
            seen[example.state] = counter(example.state)
    return TokenCensus(model_id=model_id, counts=list(seen.values()))


# ── latency ────────────────────────────────────────────────────────────────


@dataclass
class LatencySample:
    """Wall-clock cost of one batch size, timed on this machine."""

    batch: int
    questions: int
    seconds: list[float] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.seconds) * 1000.0 if self.seconds else 0.0

    @property
    def per_state_ms(self) -> float:
        return self.median_ms / self.batch if self.batch else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch,
            "questions_per_state": self.questions,
            "repeats": len(self.seconds),
            "median_ms": round(self.median_ms, 2),
            "per_state_ms": round(self.per_state_ms, 2),
        }


def measure_latency(
    predictor: DecisionPredictor,
    states: Sequence[str],
    questions: Sequence[Question],
    *,
    batches: Sequence[int] = LATENCY_BATCHES,
    repeats: int = 5,
) -> list[LatencySample]:
    """Time real calls at each batch size. Needs a model.

    One warm-up call is taken and discarded before anything is timed. A cold
    checkpoint build costs seconds -- the card measures a 7.4 s median reload on
    CPU -- and charging that to the first batch would report the load as the
    inference cost: flattering for a warm server, punishing for a poll loop, and
    wrong for both.

    ``per_state_ms`` is reported beside the total because it is the number that
    decides whether reranking thirty chunks is affordable, and it is not expected
    to fall with the batch size: N states are N forward passes.
    """
    assert_measurable(predictor)
    if not states:
        raise NotMeasurableError("no states to time")
    samples: list[LatencySample] = []
    predictor.ask([states[0]], list(questions))
    for batch in batches:
        if batch > len(states):
            # Recycling states to reach the batch size would time one encoding
            # repeatedly, and a cache anywhere in the stack would then report a
            # batch of thirty as cheaper per state than a batch of five.
            continue
        chosen = list(states[:batch])
        sample = LatencySample(batch=batch, questions=len(questions))
        for _ in range(max(1, repeats)):
            started = time.perf_counter()
            predictor.ask(chosen, list(questions))
            sample.seconds.append(time.perf_counter() - started)
        samples.append(sample)
    return samples


# ── the verdict ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DecisionVerdict:
    """Whether one decision cleared its bar, and against what."""

    decision: str
    metric: str
    candidate: float
    baselines: tuple[tuple[str, float], ...]
    passed: bool
    reason: str


@dataclass
class GateVerdict:
    """The Phase 1 outcome, per decision and overall.

    Per decision, because the plan is explicit that a track may pass while Watch
    fails and that the right response is to re-sequence around whichever track
    cleared. A single boolean would throw away the only information that says
    what to do next.
    """

    decisions: list[DecisionVerdict] = field(default_factory=list)

    @property
    def cleared(self) -> list[str]:
        return [item.decision for item in self.decisions if item.passed]

    @property
    def passed(self) -> bool:
        """Whether any decision cleared. One is enough to sequence a track around."""
        return bool(self.cleared)

    def summary(self) -> str:
        if not self.decisions:
            return "no decisions scored, so nothing was decided"
        if not self.cleared:
            return (
                "no decision beat its baselines: write up the negative result and close, "
                "per the kill criterion"
            )
        return f"cleared: {', '.join(self.cleared)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "cleared": self.cleared,
            "summary": self.summary(),
            "decisions": [
                {
                    "decision": item.decision,
                    "metric": item.metric,
                    "candidate": round(item.candidate, 4),
                    "baselines": {name: round(value, 4) for name, value in item.baselines},
                    "passed": item.passed,
                    "reason": item.reason,
                }
                for item in self.decisions
            ],
        }


def gate_verdict(candidate: EvalResult, baselines: dict[str, EvalResult]) -> GateVerdict:
    """Compare a candidate against its baselines, per decision. Pure.

    The kill criterion written as code: a decision clears only by beating every
    baseline that answered it. Strictly -- a tie is not a win, because the
    baselines cost no weights, no extra dependency and no CPU, and matching them
    means paying all three for nothing.

    Direction comes from :data:`LOWER_IS_BETTER` rather than from the metric's
    name, so "the strongest baseline" and "beats" both mean the right thing for a
    rate of mistakes as well as for a rate of successes.
    """
    verdict = GateVerdict()
    baseline_groups = {name: result.by_decision() for name, result in baselines.items()}
    for decision, result in candidate.by_decision().items():
        metric = PRIMARY_METRIC.get(decision, "accuracy")
        lower_is_better = metric in LOWER_IS_BETTER
        value = float(getattr(result, metric))
        scores: list[tuple[str, float]] = []
        for name, grouped in baseline_groups.items():
            group = grouped.get(decision)
            if group is None or not group.rows:
                continue
            scores.append((name, float(getattr(group, metric))))
        scores.sort()
        if not scores:
            verdict.decisions.append(
                DecisionVerdict(
                    decision=decision,
                    metric=metric,
                    candidate=value,
                    baselines=(),
                    passed=False,
                    reason=(
                        "no baseline answered this decision, so there is nothing to have "
                        "beaten. A figure with no floor under it is not a gate."
                    ),
                )
            )
            continue
        # The strongest baseline is the hardest one to beat, which is the smallest
        # value when a smaller number is the better one.
        best_name, best_value = (min if lower_is_better else max)(
            scores, key=lambda pair: pair[1]
        )
        passed = value < best_value if lower_is_better else value > best_value
        verdict.decisions.append(
            DecisionVerdict(
                decision=decision,
                metric=metric,
                candidate=value,
                baselines=tuple(scores),
                passed=passed,
                reason=(
                    f"{metric} {value:.3f} beats the strongest baseline "
                    f"({best_name} at {best_value:.3f})"
                    if passed
                    else f"{metric} {value:.3f} does not beat {best_name} at {best_value:.3f}"
                ),
            )
        )
    return verdict


# ── the whole gate ─────────────────────────────────────────────────────────


@dataclass
class GateResult:
    """Everything Phase 1 has to report, in one object."""

    candidate: EvalResult
    baselines: dict[str, EvalResult] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    verdict: GateVerdict = field(default_factory=GateVerdict)
    census: TokenCensus | None = None
    latency: list[LatencySample] = field(default_factory=list)
    #: The checkpoint's own temperature table, and which buckets of it the library
    #: refused to apply as shipped.
    #:
    #: Recorded beside our refit rather than instead of it, because Decision 22
    #: asks a report to name the fit that produced a probability and there are two
    #: of them: the one baked into the checkpoint, which this is, and the one
    #: :func:`aorta.local_classifier.eval.fit_temperatures` derives on our corpus, which lands
    #: in ``candidate["temperatures"]``. A result naming only ours would attribute
    #: a clamped bucket's uncalibrated answers to a fit that had nothing to do
    #: with them.
    calibration: Calibration | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "checkpoint_calibration": (
                self.calibration.to_dict() if self.calibration else None
            ),
            "baselines": {
                name: result.to_dict() for name, result in sorted(self.baselines.items())
            },
            "baseline_coverage": {
                name: round(value, 4) for name, value in sorted(self.coverage.items())
            },
            "verdict": self.verdict.to_dict(),
            "token_census": self.census.to_dict() if self.census else None,
            "latency": [sample.to_dict() for sample in self.latency],
        }

    def summary(self) -> str:
        lines = [f"candidate: {self.candidate.summary()}"]
        if self.calibration is not None and not self.calibration.known:
            lines.append(f"calibration: unknown -- {self.calibration.unknown_because}")
        elif self.calibration is not None and self.calibration.clamped:
            clamped = ", ".join(
                f"{entry.bucket} {entry.shipped:.4g}->{entry.applied:.4g}"
                for entry in self.calibration.clamped
            )
            # Printed rather than left in the JSON, because a clamped bucket is the
            # one fact about this run that changes how every probability under it
            # should be read, and the person running the gate is reading stdout.
            lines.append(f"calibration: laya clamped {clamped}")
        for name, result in sorted(self.baselines.items()):
            coverage = self.coverage.get(name)
            suffix = f", coverage {coverage:.1%}" if coverage is not None else ""
            lines.append(f"baseline {name}: {result.summary()}{suffix}")
        if self.census:
            lines.append(f"tokens: {self.census.summary()}")
        for sample in self.latency:
            lines.append(
                f"latency batch {sample.batch}: {sample.median_ms:.0f} ms median, "
                f"{sample.per_state_ms:.0f} ms per state"
            )
        lines.append(f"verdict: {self.verdict.summary()}")
        return "\n".join(lines)


def run_gate(
    examples: Sequence[LabelledExample],
    predictor: DecisionPredictor,
    *,
    holdout: float | None = DEFAULT_HOLDOUT,
    with_latency: bool = True,
    with_census: bool = True,
) -> GateResult:
    """Score a candidate against every baseline the corpus supports. Needs a model.

    The candidate goes through :func:`aorta.local_classifier.eval.evaluate`, so its
    temperatures are fitted on the held-in rows and its scores reported on the
    held-out ones. The baselines are scored on that same held-out set, which is
    what :func:`aorta.local_classifier.eval.split_examples` being deterministic buys: both
    sides compute the split independently and cannot disagree about it.

    ``holdout=None`` scores everything and fits in-sample, for a corpus too small
    to split. The majority baseline then has its prior fitted on the rows it is
    scored against, which flatters it -- so it is the *candidate* that is being
    given the harder comparison, which is the right way round for a gate.

    The census and the latency run are individually skippable, because both need
    a loaded checkpoint and either can be the slow part. Neither is optional for
    the gate decision itself: a result whose ``census`` is None has not answered
    the chunking question.
    """
    assert_measurable(predictor)
    candidate = evaluate(examples, predictor, holdout=holdout)
    if holdout is None:
        fit_set, report_set = list(examples), list(examples)
    else:
        fit_set, report_set = split_examples(examples, holdout=holdout)

    baselines: dict[str, EvalResult] = {
        "majority": score(
            report_set, prior_distributions(fit_set, report_set), model_id="majority-class prior"
        )
    }
    coverage: dict[str, float] = {}
    for name in available_baselines(report_set):
        recorded = recorded_distributions(name, report_set)
        baselines[name] = score(
            report_set, list(recorded.distributions), model_id=RECORDED_BASELINES[name]
        )
        coverage[name] = recorded.coverage

    result = GateResult(
        candidate=candidate,
        baselines=baselines,
        coverage=coverage,
        verdict=gate_verdict(candidate, baselines),
        # Read after the candidate has answered, so the checkpoint is loaded and
        # the table is there to read. Asked before would report "not loaded yet",
        # which is true and useless.
        calibration=predictor.calibration(),
    )

    if with_census:
        result.census = token_census(examples, predictor, model_id=predictor.model_id())
    if with_latency:
        states = list(dict.fromkeys(example.state for example in examples))
        questions = [examples[0].question]
        result.latency = measure_latency(predictor, states, questions)
    return result


__all__ = [
    "CONTEXT_WINDOWS",
    "LATENCY_BATCHES",
    "LOWER_IS_BETTER",
    "PRIMARY_METRIC",
    "RECORDED_BASELINES",
    "DecisionVerdict",
    "GateResult",
    "GateVerdict",
    "LatencySample",
    "NotMeasurableError",
    "RecordedBaseline",
    "TokenCensus",
    "assert_measurable",
    "available_baselines",
    "gate_verdict",
    "measure_latency",
    "prior_distributions",
    "recorded_distributions",
    "run_gate",
    "token_census",
]
