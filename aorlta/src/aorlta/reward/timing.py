"""Timing a candidate against a reference: whether the candidate is faster by a required ratio, decided only when the
measurement can decide it (docs/REWARDS.md, "Timing").

A timing check gives `judge` two lists of per-launch times, the candidate's and the reference's, taken interleaved on
one GPU (so a clock change or a neighbour's load moves both), each starting with `warmup` samples that are dropped.
From the rest:

- each side's median, and its relative median absolute deviation, `median(|x - median|) / median`;
- `noisy` when either side's relative MAD is above `max_rel_mad`: the GPU was not quiet enough to compare on;
- the noise floor, the larger of `min_noise_floor` and the two relative MADs added;
- the speedup `s = reference median / candidate median`, which is `faster` when `s >= ratio * (1 + floor)`,
  `not_faster` when `s <= ratio * (1 - floor)`, and `within_noise` in between, where the noise could decide it.

Only `faster` is a clean leg and only `not_faster` is settled (`fail`, reason `not_faster`); everything else,
including too few samples or a sample that is not a positive finite number, is unsettled, never clean. A settled
verdict carries a measurement record among its findings (kind `measurement`, which `adapters/base.py` allows on a
clean leg, as it allows a trusted build fact); `measurement` reads it back and `speedup_base` grades it.

The check must establish that the candidate is correct before it times it: a wrong kernel can be arbitrarily fast.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .adapters.base import MEASUREMENT  # the kind of a measurement record among a leg's findings

SPEEDUP = "speedup"  # its `metric`
CLEAN_BASE = 0.75  # the lowest base a clean speedup grades to; a settled slower one grades below 0.5
VERDICTS = ("faster", "not_faster", "within_noise", "noisy", "too_few", "invalid")


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    if not s:
        raise ValueError("median of no samples")
    mid = len(s) // 2
    return float(s[mid]) if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def rel_mad(xs: Sequence[float]) -> float:
    """The median absolute deviation from the median, as a share of the median."""
    m = median(xs)
    return median([abs(x - m) for x in xs]) / m


def _positive_number(name: str, value: Any, *, allow_zero: bool = False) -> float:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if not ok or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be a finite number {'>=' if allow_zero else '>'} 0, got {value!r}")
    return float(value)


def _count(name: str, value: Any, least: int) -> int:
    if not (isinstance(value, int) and not isinstance(value, bool) and value >= least):
        raise ValueError(f"{name} must be an int >= {least}, got {value!r}")
    return value


@dataclass(frozen=True)
class TimingPolicy:
    """What a timing check asks for, from its leg settings (the cache key holds them). `ratio`: the speedup to clear;
    `warmup`: samples dropped from the front of each side; `repeats`: samples kept, at least `MIN_REPEATS`;
    `max_rel_mad`: the largest relative MAD either side may show; `min_noise_floor`: the smallest noise floor."""

    ratio: float = 1.1
    warmup: int = 5
    repeats: int = 30
    max_rel_mad: float = 0.05
    min_noise_floor: float = 0.02

    MIN_REPEATS = 5

    def __post_init__(self) -> None:
        _positive_number("ratio", self.ratio)
        _count("warmup", self.warmup, 0)
        _count("repeats", self.repeats, self.MIN_REPEATS)
        if not 0 < _positive_number("max_rel_mad", self.max_rel_mad) < 1:
            raise ValueError(f"max_rel_mad must be in (0, 1), got {self.max_rel_mad!r}")
        if not 0 <= _positive_number("min_noise_floor", self.min_noise_floor, allow_zero=True) < 1:
            raise ValueError(f"min_noise_floor must be in [0, 1), got {self.min_noise_floor!r}")

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> TimingPolicy:
        """The policy a leg's settings give; a key the policy does not know is ignored (it is the check's own)."""
        names = ("ratio", "warmup", "repeats", "max_rel_mad", "min_noise_floor")
        return cls(**{k: settings[k] for k in names if k in settings})


@dataclass(frozen=True)
class Speedup:
    """One judged measurement. `speedup` and the medians are None when there was nothing to compare."""

    verdict: str  # one of VERDICTS
    ratio: float
    speedup: float | None = None
    candidate_median: float | None = None
    reference_median: float | None = None
    candidate_rel_mad: float | None = None
    reference_rel_mad: float | None = None
    noise_floor: float | None = None
    samples: int = 0


def judge(candidate: Sequence[float], reference: Sequence[float], policy: TimingPolicy) -> Speedup:
    """The verdict on two interleaved series of per-launch times (module docstring). Both series must hold `warmup`
    samples and then at least `repeats`."""
    ratio = policy.ratio
    for xs in (candidate, reference):
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in xs):
            return Speedup("invalid", ratio)
    cand, ref = list(candidate)[policy.warmup:], list(reference)[policy.warmup:]
    n = min(len(cand), len(ref))
    if n < policy.repeats:
        return Speedup("too_few", ratio, samples=n)
    mc, mr = median(cand), median(ref)
    dc, dr = rel_mad(cand), rel_mad(ref)
    common = dict(candidate_median=mc, reference_median=mr, candidate_rel_mad=dc, reference_rel_mad=dr, samples=n)
    if dc > policy.max_rel_mad or dr > policy.max_rel_mad:
        return Speedup("noisy", ratio, **common)
    floor = max(policy.min_noise_floor, dc + dr)
    s = mr / mc
    if s >= ratio * (1.0 + floor):
        verdict = "faster"
    elif s <= ratio * (1.0 - floor):
        verdict = "not_faster"
    else:
        verdict = "within_noise"
    return Speedup(verdict, ratio, speedup=s, noise_floor=floor, **common)


def record(sp: Speedup) -> dict:
    """The measurement record a settled verdict carries among its findings: numbers only."""
    return {"kind": MEASUREMENT, "metric": SPEEDUP, "speedup": sp.speedup, "ratio": sp.ratio,
            "noise_floor": sp.noise_floor, "candidate_median_us": sp.candidate_median,
            "reference_median_us": sp.reference_median, "candidate_rel_mad": sp.candidate_rel_mad,
            "reference_rel_mad": sp.reference_rel_mad, "samples": sp.samples}


def leg_outcome(sp: Speedup) -> tuple[str, str | None, list[dict]]:
    """A leg's (state, reason, findings) for a verdict: `faster` is clean, `not_faster` is `fail` with reason
    `not_faster`, each with its record; anything else is unsettled with reason `timing_<verdict>` and no findings."""
    if sp.verdict == "faster":
        return "clean", None, [record(sp)]
    if sp.verdict == "not_faster":
        return "fail", "not_faster", [record(sp)]
    return "unsettled", f"timing_{sp.verdict}", []


def measurement(leg: Any) -> dict | None:
    """The speedup record of a leg result: its one `measurement` finding with metric `speedup`, a positive finite
    `speedup` and `ratio`, and a noise floor in [0, 1); else None."""
    found = [f for f in getattr(leg, "findings", None) or [] if isinstance(f, Mapping) and f.get("kind") == MEASUREMENT]
    if len(found) != 1 or found[0].get("metric") != SPEEDUP:
        return None
    rec = found[0]
    try:
        _positive_number("speedup", rec.get("speedup"))
        _positive_number("ratio", rec.get("ratio"))
        if not 0 <= _positive_number("noise_floor", rec.get("noise_floor"), allow_zero=True) < 1:
            return None
    except ValueError:
        return None
    return dict(rec)


def speedup_base(state: str, rec: Mapping[str, Any]) -> float | None:
    """The base a settled speedup leg grades to, from its state and record, or None when the two disagree:

    - clean (the speedup cleared `ratio` past the noise floor): `0.75 + 0.25 * min(1, log2(speedup / ratio))`, so
      from 0.75 just past the threshold to 1.0 at twice the required ratio;
    - fail `not_faster` (measured and correct, but not faster by the ratio): `0.5 * speedup / ratio`, below 0.5, so a
      correct kernel scores above every wrong one and below every one that clears the ratio."""
    s, ratio, floor = float(rec["speedup"]), float(rec["ratio"]), float(rec["noise_floor"])
    if state == "clean" and s >= ratio * (1.0 + floor):
        return CLEAN_BASE + (1.0 - CLEAN_BASE) * min(1.0, math.log2(s / ratio))
    if state == "fail" and s <= ratio * (1.0 - floor):
        return 0.5 * s / ratio
    return None
