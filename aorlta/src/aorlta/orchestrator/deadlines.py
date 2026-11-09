"""Deadlines and the scored-fraction policy (design Section 6.6; decision D7).

- Generation deadline: max(generation floor, 3 x the rolling median of past generation phases).
- Scoring deadline, after the generation barrier: max(scoring floor, 2 x the rolling median of past scoring phases).
  The floors are `orchestrator.deadlines.{generation_s, scoring_s}` (1,800 s by default).
- D7: train if at least `min_scored_fraction` of the iteration's samples are scored; otherwise re-check the
  unsettled legs once, with the scoring deadline x 1.5; if the fraction is still short, stop the run.

The loop takes the fraction over the candidates handed to the reward service: a chain
an engine lost, or one the leak check quarantined, never reaches the service and is reported as a warning instead,
so one engine death that C1 survives does not stop the run. It is compared exactly, as a rational number, so 9 of
10 meets a threshold of 0.9.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

ROLLING_WINDOW = 10  # past iterations a rolling median looks at
GENERATION_FACTOR = 3.0
SCORING_FACTOR = 2.0
RETRY_DEADLINE_FACTOR = 1.5


def rolling_median(history: Sequence[float], window: int = ROLLING_WINDOW) -> float | None:
    recent = [float(x) for x in list(history)[-window:]]
    return statistics.median(recent) if recent else None


def generation_deadline_s(floor_s: float, history: Sequence[float]) -> float:
    med = rolling_median(history)
    return max(floor_s, GENERATION_FACTOR * med) if med is not None else floor_s


def scoring_deadline_s(floor_s: float, history: Sequence[float]) -> float:
    med = rolling_median(history)
    return max(floor_s, SCORING_FACTOR * med) if med is not None else floor_s


def push_history(history: Sequence[float], value: float, window: int = ROLLING_WINDOW) -> list[float]:
    return [float(x) for x in list(history)[-(window - 1):]] + [float(value)]


def _threshold(min_fraction: float) -> Fraction:
    # The decimal the configuration holds (repr), not the binary float: Fraction(0.9) is slightly above 9/10.
    return Fraction(repr(float(min_fraction)))


def fraction_met(scored: int, total: int, min_fraction: float) -> bool:
    if total <= 0:
        return False
    return Fraction(scored, total) >= _threshold(min_fraction)


@dataclass(frozen=True)
class PolicyDecision:
    decision: str  # "train" | "retry" | "stop"
    scored: int
    total: int
    retryable: int
    min_fraction: float
    retried: bool
    why: str

    @property
    def fraction(self) -> float:
        return self.scored / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {"decision": self.decision, "scored": self.scored, "total": self.total,
                "scored_fraction": self.fraction, "retryable": self.retryable,
                "min_scored_fraction": self.min_fraction, "retried": self.retried, "why": self.why}


def decide(*, scored: int, total: int, retryable: int, min_fraction: float, retried: bool) -> PolicyDecision:
    """D7. `retryable` counts the unscored samples that have at least one unsettled leg; `retried` says whether
    the one retry was already spent. A retry is skipped when even scoring every retryable sample could not reach
    the threshold: the run would stop either way, so waiting 1.5 x the deadline would buy nothing."""
    if scored < 0 or retryable < 0 or scored + retryable > total:
        raise ValueError(f"inconsistent counts: scored {scored}, retryable {retryable}, total {total}")
    if fraction_met(scored, total, min_fraction):
        return PolicyDecision("train", scored, total, retryable, min_fraction, retried, "scored fraction met")
    if retried:
        return PolicyDecision("stop", scored, total, retryable, min_fraction, retried,
                              "scored fraction still below the threshold after the retry")
    if retryable == 0:
        return PolicyDecision("stop", scored, total, retryable, min_fraction, retried,
                              "scored fraction below the threshold and no unsettled leg to retry")
    if not fraction_met(scored + retryable, total, min_fraction):
        return PolicyDecision("stop", scored, total, retryable, min_fraction, retried,
                              "scored fraction below the threshold even if every unsettled sample scored")
    return PolicyDecision("retry", scored, total, retryable, min_fraction, retried,
                          "scored fraction below the threshold; re-checking the unsettled legs once")
