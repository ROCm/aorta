"""C9 comparison report: AORLTA against a baseline run, on the pre-registered criteria of decision D1
(design Section 6.9, "Metrics" and "The comparison protocol"; Section 9, D1).

Three criteria, all pre-registered in a protocol file committed before the first comparison run:

1. Wall time per 256 scored samples (per256.py), with phase timings: AORLTA's statistic over
   iterations 2..N must beat the baseline's by more than the protocol's factor.
2. Learning parity: the per-iteration difference in mean training reward, over the same prompts in
   the same order, with a t-interval across iterations that must lie inside +-tolerance.
3. Held-out no worse: per eval mode, items paired one for one, with a cluster bootstrap over items;
   the lower bound of the difference must lie above -margin.

The comparison counts only if the fields the protocol holds equal are recorded and equal in both runs,
both runs cover iterations 1..N, and the held-out evals are at iteration N on both sides. Compute
curves are descriptive only. Every threshold comes from the protocol: nothing here has a default.

Inputs are two run bundles in a neutral schema (`aorlta.measure.bundle/1`). `bundle_from_run` builds
AORLTA's from its spans and episode records; the baseline's from the baseline run's own files.

`python -m aorlta.measure.compare sample --out DIR` writes a report from synthetic data, to show the
format; every page of it says so.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..atomic import atomic_write_bytes, atomic_write_json, atomic_write_text, read_bytes, read_text
from ..config import ConfigError, parse_yaml, plain, strict_decode
from ..ids import canonical_json, sha256_hex, sha256_json
from ..spans import Span
from . import MeasureError, curves, per256, timeline, verify70

BUNDLE_SCHEMA = "aorlta.measure.bundle/1"
PROTOCOL_SCHEMA = "aorlta.measure.protocol/1"
REPORT_SCHEMA = "aorlta.measure.report/1"
TRUNCATION_WARNING = 0.10  # the paper treats > 10-15% truncation as a warning of instability [P A.15]
GIT = "/usr/bin/git"


class CompareError(MeasureError):
    pass


# --- inputs ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    wall_s: float
    samples: int
    scored: int
    stalled: bool
    stall_reason: str | None = None
    mean_train_reward: float | None = None  # over scored samples
    prompts_sha256: str | None = None  # the iteration's prompt set (prompt_set_sha256)
    unscored_by_reason: dict[str, int] = field(default_factory=dict)
    truncated: int | None = None  # final answers that ended on the length cap
    noop: bool | None = None  # the landing verdict was NOOP, or no optimizer step was taken
    category_coverage_s: dict[str, float] = field(default_factory=dict)  # wall-clock coverage by category
    category_critical_s: dict[str, float] = field(default_factory=dict)  # critical-path seconds by category
    check_share: tuple[float, float] | None = None  # verify70 lower and upper bounds
    check_status: Literal["confirmed", "refuted", "undetermined"] | None = None

    def __post_init__(self) -> None:
        self.stats()  # the per-256 rules
        if sum(self.unscored_by_reason.values()) not in (0, self.samples - self.scored):
            raise CompareError(f"iteration {self.iteration}: unscored by reason sums to "
                               f"{sum(self.unscored_by_reason.values())}, not {self.samples - self.scored}")
        if self.truncated is not None and not 0 <= self.truncated <= self.samples:
            raise CompareError(f"iteration {self.iteration}: truncated {self.truncated} of {self.samples}")

    def stats(self) -> per256.IterationStats:
        return per256.IterationStats(self.iteration, self.wall_s, self.samples, self.scored, self.stalled,
                                     self.stall_reason)


@dataclass(frozen=True)
class Allocation:
    gpus_held: int  # every GPU held for the run, partner ASan GPUs included when dedicated
    cpu_nodes_held: int
    note: str = ""


@dataclass(frozen=True)
class Heldout:
    iteration: int
    eval_version: str
    modes: dict[str, dict[str, tuple[int, ...]]]  # mode -> item id -> per-sample verified repair (0 or 1)

    def __post_init__(self) -> None:
        for mode, items in self.modes.items():
            for item, outcomes in items.items():
                if not outcomes or any(x not in (0, 1) for x in outcomes):
                    raise CompareError(f"held-out {mode}/{item}: outcomes must be a non-empty list of 0 and 1")


@dataclass(frozen=True)
class ValidationPoint:
    iteration: int
    gpu_hours: float
    wall_hours: float
    reward: float


@dataclass(frozen=True)
class Validation:
    r0: float  # validation reward of the merged base (v0), before training
    samples_per_point: int
    points: tuple[ValidationPoint, ...]


@dataclass(frozen=True)
class Bundle:
    schema: Literal["aorlta.measure.bundle/1"]
    system: str
    synthetic: bool
    held_equal: dict[str, Any]
    allocation: Allocation
    iterations: tuple[IterationRecord, ...]
    heldout: Heldout | None = None
    validation: Validation | None = None

    def by_iteration(self) -> dict[int, IterationRecord]:
        out: dict[int, IterationRecord] = {}
        for r in self.iterations:
            if r.iteration in out:
                raise CompareError(f"{self.system}: iteration {r.iteration} appears twice")
            out[r.iteration] = r
        return out


@dataclass(frozen=True)
class SpeedRule:
    statistic: Literal["median", "mean_without_stalls", "mean_with_stalls"]
    min_speedup: float  # passes if baseline / AORLTA is more than this

    def __post_init__(self) -> None:
        if not self.min_speedup >= 1.0:
            raise CompareError("speed.min_speedup must be at least 1")


@dataclass(frozen=True)
class ParityRule:
    tolerance: float  # in training-reward units
    ci_level: float

    def __post_init__(self) -> None:
        if not (self.tolerance > 0 and 0 < self.ci_level < 1):
            raise CompareError("parity needs tolerance > 0 and ci_level in (0, 1)")


@dataclass(frozen=True)
class HeldoutRule:
    modes: tuple[str, ...]
    margin: float  # non-inferiority margin, in verified-repair rate
    ci_level: float
    resamples: int
    seed: int

    def __post_init__(self) -> None:
        if not (self.modes and self.margin >= 0 and 0 < self.ci_level < 1 and self.resamples >= 1000):
            raise CompareError("heldout needs modes, margin >= 0, ci_level in (0, 1) and resamples >= 1000")


@dataclass(frozen=True)
class CurveRule:
    burn_in_gpu_hours: float
    burn_in_wall_hours: float
    a_range: tuple[float, float]
    gpu_hours_cmid_range: tuple[float, float]
    wall_hours_cmid_range: tuple[float, float]


@dataclass(frozen=True)
class Protocol:
    schema: Literal["aorlta.measure.protocol/1"]
    title: str
    iterations: int  # N: both runs cover iterations 1..N
    held_equal: tuple[str, ...]
    speed: SpeedRule
    parity: ParityRule
    heldout: HeldoutRule
    curves: CurveRule | None = None

    def __post_init__(self) -> None:
        if self.iterations < 2:
            raise CompareError("protocol.iterations must be at least 2 (the per-256 statistics start at 2)")
        if not self.held_equal:
            raise CompareError("protocol.held_equal names nothing: the comparison would hold nothing equal")


def _decode(tp: type, data: Any, where: str) -> Any:
    try:
        return strict_decode(tp, data, where)
    except ConfigError as exc:
        raise CompareError(str(exc)) from None


def load_protocol(path: str | Path) -> Protocol:
    try:
        data = parse_yaml(read_text(path), str(path))
    except ConfigError as exc:
        raise CompareError(str(exc)) from None
    return _decode(Protocol, data, "protocol")


def load_bundle(path: str | Path) -> Bundle:
    try:
        data = json.loads(read_text(path))
    except ValueError as exc:
        raise CompareError(f"{path}: not JSON ({exc})") from None
    return _decode(Bundle, data, "bundle")


def prompt_set_sha256(prompt_ids: Sequence[str]) -> str:
    """The prompts of one iteration, as a set: parity pairs iterations that drew the same prompts."""
    return sha256_json(sorted(set(prompt_ids)))


# --- statistics --------------------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            delta = d * c
            h *= delta
        if abs(delta - 1.0) < 1e-15:
            return h
    raise CompareError("incomplete beta did not converge")


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    tail = 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_quantile(p: float, df: float) -> float:
    """Student t inverse CDF, by bisection on t_cdf."""
    if not (0 < p < 1 and df > 0):
        raise CompareError(f"t quantile needs p in (0, 1) and df > 0, got {p}, {df}")
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        lo, hi = hi, 2.0 * hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * hi:
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Interval:
    estimate: float
    lo: float
    hi: float


def paired_t_interval(diffs: Sequence[float], level: float) -> tuple[Interval, float, int]:
    """Mean of the paired differences with a two-sided t-interval; returns (interval, sd, df)."""
    n = len(diffs)
    if n < 2:
        raise CompareError("a t-interval needs at least 2 paired differences")
    mean, sd = statistics.fmean(diffs), statistics.stdev(diffs)
    half = t_quantile(0.5 + level / 2.0, n - 1) * sd / math.sqrt(n)
    return Interval(mean, mean - half, mean + half), sd, n - 1


def _quantile(sorted_xs: Sequence[float], p: float) -> float:
    """Linear interpolation between order statistics (numpy's default, 'inclusive')."""
    h = (len(sorted_xs) - 1) * p
    k = math.floor(h)
    if k + 1 >= len(sorted_xs):
        return sorted_xs[-1]
    return sorted_xs[k] + (h - k) * (sorted_xs[k + 1] - sorted_xs[k])


def cluster_bootstrap(per_item: Sequence[float], *, resamples: int, seed: int, level: float) -> Interval:
    """Percentile interval for the mean of per-item paired differences. Items are the clusters: each
    resample draws whole items with replacement, so samples of one item are never split up."""
    n = len(per_item)
    if n < 2:
        raise CompareError("a bootstrap over items needs at least 2 items")
    rng = random.Random(seed)
    means = sorted(math.fsum(per_item[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    alpha = 1.0 - level
    return Interval(statistics.fmean(per_item), _quantile(means, alpha / 2.0), _quantile(means, 1.0 - alpha / 2.0))


# --- evaluation --------------------------------------------------------------------------------------

PASS, FAIL, NOT_EVALUATED = "PASS", "FAIL", "NOT EVALUATED"


@dataclass(frozen=True)
class Criterion:
    key: str
    rule: str  # the pre-registered rule, in words
    verdict: str  # PASS | FAIL | NOT EVALUATED
    why: str
    numbers: dict[str, Any]


def _check_validity(p: Protocol, a: Bundle, b: Bundle) -> tuple[bool, list[str], list[dict[str, Any]]]:
    problems, fields = [], []
    for key in p.held_equal:
        va, vb = a.held_equal.get(key), b.held_equal.get(key)
        absent = [x.system for x in (a, b) if key not in x.held_equal]
        equal = not absent and canonical_json(va) == canonical_json(vb)
        fields.append({"field": key, "aorlta": va, "baseline": vb, "equal": equal})
        if absent:
            problems.append(f"held-equal field {key!r} is not recorded by {' and '.join(absent)}")
        elif not equal:
            problems.append(f"held-equal field {key!r} differs")
    for bundle in (a, b):
        its = bundle.by_iteration()
        missing = [n for n in range(1, p.iterations + 1) if n not in its]
        if missing:
            problems.append(f"{bundle.system} lacks iterations {missing} of 1..{p.iterations}")
    ia, ib = a.by_iteration(), b.by_iteration()
    differing = [n for n in range(1, p.iterations + 1) if n in ia and n in ib and ia[n].prompts_sha256 is not None
                 and ib[n].prompts_sha256 is not None and ia[n].prompts_sha256 != ib[n].prompts_sha256]
    if differing:
        problems.append(f"the prompt sets differ at iterations {differing}: not the same prompts in the same order")
    for bundle in (a, b):
        if bundle.heldout is not None and bundle.heldout.iteration != p.iterations:
            problems.append(f"{bundle.system}'s held-out eval is at iteration {bundle.heldout.iteration}, "
                            f"not {p.iterations}")
    return not problems, problems, fields


def _speed(p: Protocol, a: Bundle, b: Bundle) -> Criterion:
    rule = (f"{p.speed.statistic.replace('_', ' ')} wall time per 256 scored samples over iterations "
            f"{per256.FIRST_REPORTED}..{p.iterations}: baseline / AORLTA above {p.speed.min_speedup:g}")
    sa = per256.summarize([r.stats() for r in a.iterations if r.iteration <= p.iterations], last=p.iterations)
    sb = per256.summarize([r.stats() for r in b.iterations if r.iteration <= p.iterations], last=p.iterations)
    va, vb = sa.statistic(p.speed.statistic), sb.statistic(p.speed.statistic)
    numbers: dict[str, Any] = {
        "aorlta": sa.to_json(), "baseline": sb.to_json(), "statistic": p.speed.statistic,
        "resources": {"aorlta": per256.resources(sa, gpus_held=a.allocation.gpus_held,
                                                 cpu_nodes_held=a.allocation.cpu_nodes_held),
                      "baseline": per256.resources(sb, gpus_held=b.allocation.gpus_held,
                                                   cpu_nodes_held=b.allocation.cpu_nodes_held)},
        "paired_ratio": {str(n): sb.per256_s[n] / sa.per256_s[n] for n in sa.iterations}}
    if va is None or vb is None:
        return Criterion("speed", rule, NOT_EVALUATED, "the statistic is undefined: every reported iteration stalled",
                         numbers)
    ratio = vb / va
    numbers.update(aorlta_s=va, baseline_s=vb, speedup=ratio)
    verdict = PASS if ratio > p.speed.min_speedup else FAIL
    return Criterion("speed", rule, verdict, f"x{ratio:.2f} ({va / 60:.1f} min against {vb / 60:.1f} min)", numbers)


def _parity(p: Protocol, a: Bundle, b: Bundle) -> Criterion:
    tol, level = p.parity.tolerance, p.parity.ci_level
    rule = (f"mean training reward per iteration, iterations 1..{p.iterations}, same prompts: the "
            f"{level:.0%} t-interval of the mean paired difference inside +-{tol:g}")
    ia, ib = a.by_iteration(), b.by_iteration()
    its = range(1, p.iterations + 1)
    no_reward = [n for n in its if ia[n].mean_train_reward is None or ib[n].mean_train_reward is None]
    no_prompts = [n for n in its if ia[n].prompts_sha256 is None or ib[n].prompts_sha256 is None]
    if no_reward or no_prompts:
        why = []
        if no_reward:
            why.append(f"no mean training reward at iterations {no_reward}")
        if no_prompts:
            why.append(f"no prompt-set hash at iterations {no_prompts}, so the pairing is unverified")
        return Criterion("parity", rule, NOT_EVALUATED, "; ".join(why), {})
    diffs = [ia[n].mean_train_reward - ib[n].mean_train_reward for n in its]
    ci, sd, df = paired_t_interval(diffs, level)
    numbers = {"diffs": {str(n): d for n, d in zip(its, diffs)}, "mean": ci.estimate, "lo": ci.lo, "hi": ci.hi,
               "sd": sd, "df": df, "tolerance": tol, "ci_level": level,
               "iterations_outside_tolerance": [n for n, d in zip(its, diffs) if abs(d) >= tol]}
    verdict = PASS if -tol < ci.lo and ci.hi < tol else FAIL
    return Criterion("parity", rule, verdict, f"{ci.estimate:+.4f} [{ci.lo:+.4f}, {ci.hi:+.4f}]", numbers)


def _heldout(p: Protocol, a: Bundle, b: Bundle) -> list[Criterion]:
    r = p.heldout
    out = []
    for mode in r.modes:
        rule = (f"held-out ({mode}) at iteration {p.iterations}, items paired: the {r.ci_level:.0%} "
                f"cluster-bootstrap lower bound of AORLTA minus baseline above -{r.margin:g}")
        if a.heldout is None or b.heldout is None:
            out.append(Criterion(f"heldout.{mode}", rule, NOT_EVALUATED, "no held-out result for "
                                 + " and ".join(x.system for x in (a, b) if x.heldout is None), {}))
            continue
        ma, mb = a.heldout.modes.get(mode), b.heldout.modes.get(mode)
        if ma is None or mb is None:
            out.append(Criterion(f"heldout.{mode}", rule, NOT_EVALUATED, f"mode {mode!r} missing", {}))
            continue
        if set(ma) != set(mb):
            out.append(Criterion(f"heldout.{mode}", rule, NOT_EVALUATED,
                                 f"the item sets differ ({len(set(ma) ^ set(mb))} items not on both sides)", {}))
            continue
        items = sorted(ma)
        d = [statistics.fmean(ma[i]) - statistics.fmean(mb[i]) for i in items]
        ci = cluster_bootstrap(d, resamples=r.resamples, seed=r.seed, level=r.ci_level)
        rate = lambda m: math.fsum(sum(v) for v in m.values()) / sum(len(v) for v in m.values())  # noqa: E731
        numbers = {"items": len(items), "samples": {"aorlta": sum(len(ma[i]) for i in items),
                                                    "baseline": sum(len(mb[i]) for i in items)},
                   "rate": {"aorlta": rate(ma), "baseline": rate(mb)}, "mean_diff": ci.estimate, "lo": ci.lo,
                   "hi": ci.hi, "margin": r.margin, "resamples": r.resamples, "seed": r.seed, "ci_level": r.ci_level}
        verdict = PASS if ci.lo > -r.margin else FAIL
        out.append(Criterion(f"heldout.{mode}", rule, verdict, f"{ci.estimate:+.3f} [{ci.lo:+.3f}, {ci.hi:+.3f}]",
                             numbers))
    return out


def _median_by_key(records: Sequence[IterationRecord], attr: str) -> dict[str, float]:
    keys = sorted({k for r in records for k in getattr(r, attr)})
    return {k: statistics.median(getattr(r, attr).get(k, 0.0) for r in records) for k in keys}


def _health(bundle: Bundle, n_last: int) -> dict[str, Any]:
    rs = [r for r in bundle.iterations if r.iteration <= n_last]
    samples = sum(r.samples for r in rs)
    by_reason: dict[str, int] = defaultdict(int)
    for r in rs:
        for k, v in r.unscored_by_reason.items():
            by_reason[k] += v
    trunc = None if any(r.truncated is None for r in rs) else sum(r.truncated for r in rs) / samples
    noop = None if any(r.noop is None for r in rs) else [r.iteration for r in rs if r.noop]
    statuses: dict[str, int] = defaultdict(int)
    for r in rs:
        statuses[r.check_status or "not measured"] += 1
    return {"samples": samples, "unscored_fraction": (samples - sum(r.scored for r in rs)) / samples,
            "unscored_by_reason": dict(sorted(by_reason.items())), "truncation_rate": trunc,
            "truncation_flag": trunc is not None and trunc > TRUNCATION_WARNING, "noop_iterations": noop,
            "stalls": {str(r.iteration): r.stall_reason for r in rs if r.stalled},
            "check_share_status": dict(statuses)}


def _curves(p: Protocol, bundle: Bundle) -> dict[str, Any]:
    if bundle.validation is None:
        return {"points": [], "fits": {}, "note": "no validation points"}
    v = bundle.validation
    pts = sorted(v.points, key=lambda x: x.iteration)
    fits: dict[str, Any] = {}
    for axis, attr in (("gpu_hours", "gpu_hours"), ("wall_hours", "wall_hours")):
        if p.curves is None:
            fits[axis] = {"status": "refused", "reason": "no curve rule in the protocol"}
            continue
        burn = p.curves.burn_in_gpu_hours if axis == "gpu_hours" else p.curves.burn_in_wall_hours
        cmid = p.curves.gpu_hours_cmid_range if axis == "gpu_hours" else p.curves.wall_hours_cmid_range
        fit = curves.fit_sigmoid([curves.Point(getattr(x, attr), x.reward) for x in pts], r0=v.r0, burn_in=burn,
                                 validation_samples=v.samples_per_point, a_range=p.curves.a_range, cmid_range=cmid)
        fits[axis] = fit.to_json()
    return {"r0": v.r0, "samples_per_point": v.samples_per_point, "fits": fits,
            "points": [{"iteration": x.iteration, "gpu_hours": x.gpu_hours, "wall_hours": x.wall_hours,
                        "reward": x.reward} for x in pts]}


@dataclass(frozen=True)
class Comparison:
    protocol: Protocol
    aorlta: Bundle
    baseline: Bundle
    valid: bool
    problems: tuple[str, ...]
    held_equal: tuple[dict[str, Any], ...]
    criteria: tuple[Criterion, ...]
    overall: str  # PASS | FAIL | INCOMPLETE | INVALID
    overall_why: str
    descriptive: dict[str, Any]
    provenance: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"schema": REPORT_SCHEMA, "synthetic": self.aorlta.synthetic or self.baseline.synthetic,
                "provenance": self.provenance, "valid": self.valid, "problems": list(self.problems),
                "held_equal": list(self.held_equal), "overall": self.overall, "overall_why": self.overall_why,
                "criteria": [{"key": c.key, "rule": c.rule, "verdict": c.verdict, "why": c.why, "numbers": c.numbers}
                             for c in self.criteria], "descriptive": self.descriptive}


def compare(p: Protocol, a: Bundle, b: Bundle, *, provenance: Mapping[str, Any] | None = None) -> Comparison:
    valid, problems, fields = _check_validity(p, a, b)
    criteria: list[Criterion] = []
    if valid:
        criteria = [_speed(p, a, b), _parity(p, a, b), *_heldout(p, a, b)]
        verdicts = [c.verdict for c in criteria]
        if FAIL in verdicts:
            overall = "FAIL"
            failed = [c.key for c in criteria if c.verdict == FAIL]
            why = f"failed: {', '.join(failed)}"
            if "parity" in failed and next(c for c in criteria if c.key == "speed").verdict == PASS:
                why += "; a speed result means nothing while learning per iteration differs"
        elif NOT_EVALUATED in verdicts:
            overall, why = "INCOMPLETE", "not evaluated: " + ", ".join(c.key for c in criteria
                                                                     if c.verdict == NOT_EVALUATED)
        else:
            overall, why = "PASS", "every pre-registered criterion passes on a valid comparison"
    else:
        overall, why = "INVALID", "; ".join(problems)
    descriptive = {}
    if valid:
        for key, bundle in (("aorlta", a), ("baseline", b)):
            reported = [r for r in bundle.iterations if per256.FIRST_REPORTED <= r.iteration <= p.iterations]
            descriptive[key] = {"health": _health(bundle, p.iterations), "curves": _curves(p, bundle),
                                "phase_coverage_s_median": _median_by_key(reported, "category_coverage_s"),
                                "phase_critical_s_median": _median_by_key(reported, "category_critical_s")}
    return Comparison(p, a, b, valid, tuple(problems), tuple(fields), tuple(criteria), overall, why, descriptive,
                      dict(provenance or {}))


def verify_preregistration(protocol_path: str | Path, commit: str, repo: str | Path) -> str:
    """The protocol used must be byte-identical to the one committed at `commit` (the pre-registration
    commit, made before any comparison run). Returns a one-line statement; raises if it differs."""
    path = Path(protocol_path).resolve()
    top = subprocess.run([GIT, "-C", str(repo), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode != 0:
        raise CompareError(f"{repo} is not a git repository")
    root = Path(top.stdout.strip()).resolve()
    try:
        rel = path.relative_to(root)
    except ValueError:
        raise CompareError(f"the protocol {path} is not inside {root}") from None
    shown = subprocess.run([GIT, "-C", str(root), "show", f"{commit}:{rel.as_posix()}"], capture_output=True)
    if shown.returncode != 0:
        raise CompareError(f"commit {commit} has no {rel}")
    if shown.stdout != read_bytes(path):
        raise CompareError(f"the protocol differs from the copy pre-registered at {commit}")
    full = subprocess.run([GIT, "-C", str(root), "rev-parse", commit], capture_output=True, text=True).stdout.strip()
    return f"identical to {rel.as_posix()} at commit {full[:12]}"


# --- building AORLTA's bundle --------------------------------------------------------------------------


def bundle_from_run(spans: Sequence[Span], episodes: Sequence[Mapping[str, Any]], *, system: str,
                    held_equal: Mapping[str, Any], allocation: Allocation, stalls: Mapping[int, str],
                    mode: str = "v1", landing_verdicts: Mapping[int, Sequence[str]] | None = None,
                    heldout: Heldout | None = None, validation: Validation | None = None,
                    synthetic: bool = False) -> Bundle:
    """AORLTA's bundle from its spans (wall, phase coverage, critical path, check share) and episode
    records (scored counts, unscored by reason, truncation, mean reward, prompt sets). The stalls and,
    where the landing spans carry none, the landing verdicts come from the iteration summaries."""
    if mode not in ("v1", "v2"):
        raise CompareError(f"mode must be v1 or v2, got {mode!r}")
    tls = {t.iteration: t for t in timeline.summarize_run(spans, mode, verdicts=landing_verdicts)}
    counts = per256.counts_from_episodes(episodes)
    stats = {s.iteration: s for s in per256.iteration_stats({n: t.wall_s for n, t in tls.items()}, counts, stalls)}
    phases, classes = verify70.phases_from_spans(spans, mode, verdicts=landing_verdicts)
    shares = {r.iteration: r for r in verify70.verify(phases, classes).rows}
    prompts: dict[int, list[str]] = defaultdict(list)
    for rec in episodes:
        pid = rec.get("prompt_id")
        if not isinstance(pid, str):
            raise CompareError(f"episode {rec.get('chain_id')!r} has no prompt_id")
        prompts[rec["iteration"]].append(pid)
    records = []
    for n, t in sorted(tls.items()):
        c, s, row = counts[n], stats[n], shares[n]
        records.append(IterationRecord(
            iteration=n, wall_s=t.wall_s, samples=c.samples, scored=c.scored, stalled=s.stalled,
            stall_reason=s.stall_reason, mean_train_reward=c.mean_train_reward,
            prompts_sha256=prompt_set_sha256(prompts[n]), unscored_by_reason=dict(c.unscored_by_reason),
            truncated=c.truncated, noop=t.bounds.end_from in ("landing:NOOP", timeline.NO_STEP_SPAN),
            category_coverage_s={k: v for k, v in t.category_coverage_s.items()},
            category_critical_s=dict(t.category_critical_s), check_share=(row.lower, row.upper),
            check_status=row.status))
    return Bundle(BUNDLE_SCHEMA, system, synthetic, dict(held_equal), allocation, tuple(records), heldout, validation)


# --- the report --------------------------------------------------------------------------------------


def _min(s: float | None) -> str:
    return "n/a" if s is None else f"{s / 60:.1f}"


def _fmt(x: Any) -> str:
    text = canonical_json(x).decode()
    return f"`{text if len(text) <= 48 else text[:45] + '...'}`"


def markdown(c: Comparison, *, plots: Mapping[str, str] | None = None) -> str:
    a, b, p = c.aorlta, c.baseline, c.protocol
    synthetic = a.synthetic or b.synthetic
    L: list[str] = [f"# {p.title}", ""]
    if synthetic:
        L += ["> **SYNTHETIC DATA.** This report was generated from made-up runs to show the format. "
              "No number in it is a measurement of AORLTA, of Miles or of anything else.", ""]
    prov = c.provenance
    L += ["| | |", "|---|---|",
          f"| AORLTA run | {a.system}{' (SYNTHETIC)' if a.synthetic else ''} |",
          f"| baseline run | {b.system}{' (SYNTHETIC)' if b.synthetic else ''} |",
          f"| protocol | sha256 `{prov.get('protocol_sha256', '?')[:16]}` |",
          f"| pre-registration | {prov.get('preregistration', 'NOT VERIFIED: no pre-registration commit given')} |",
          f"| bundles | AORLTA sha256 `{prov.get('aorlta_sha256', '?')[:16]}`, "
          f"baseline sha256 `{prov.get('baseline_sha256', '?')[:16]}` |",
          f"| iterations | 1..{p.iterations} on both runs; per-256 statistics over "
          f"{per256.FIRST_REPORTED}..{p.iterations} |", ""]

    L += ["## Verdict", "", "| criterion, as pre-registered | result | verdict |", "|---|---|---|"]
    held = sum(1 for f in c.held_equal if f["equal"])
    L.append(f"| valid comparison: the {len(c.held_equal)} held-equal fields recorded and equal; iterations "
             f"1..{p.iterations} on both; same prompt sets; held-out at iteration {p.iterations} | "
             f"{held} of {len(c.held_equal)} fields equal | {'VALID' if c.valid else 'INVALID'} |")
    for cr in c.criteria:
        L.append(f"| {cr.rule} | {cr.why} | {cr.verdict} |")
    L += ["", f"**Overall: {c.overall}.** {c.overall_why}.", ""]
    if not c.valid:
        L += ["The criteria were not evaluated, because the comparison is not valid:", ""]
        L += [f"- {x}" for x in c.problems] + [""]
    L += ["### Fields held equal", "", "| field | AORLTA | baseline | equal |", "|---|---|---|---|"]
    L += [f"| {f['field']} | {_fmt(f['aorlta'])} | {_fmt(f['baseline'])} | {'yes' if f['equal'] else '**no**'} |"
          for f in c.held_equal]
    L.append("")
    if not c.valid:
        return "\n".join(L) + "\n"

    crit = {cr.key: cr for cr in c.criteria}
    sp = crit["speed"].numbers
    sa, sb = sp["aorlta"], sp["baseline"]
    L += ["## 1. Wall time per 256 scored samples", "",
          "`256 x wall(n) / scored(n)`; wall(n) runs from prompt selection to the passed landing check for "
          "AORLTA. Minutes.", "",
          "| | AORLTA | baseline |", "|---|---|---|",
          f"| median | {_min(sa['median_s'])} | {_min(sb['median_s'])} |",
          f"| IQR (q1 to q3) | {_min(sa['q1_s'])} to {_min(sa['q3_s'])} | {_min(sb['q1_s'])} to {_min(sb['q3_s'])} |",
          f"| mean, stalls included | {_min(sa['mean_with_stalls_s'])} | {_min(sb['mean_with_stalls_s'])} |",
          f"| mean, stalls excluded | {_min(sa['mean_without_stalls_s'])} | {_min(sb['mean_without_stalls_s'])} |",
          f"| stalled iterations | {', '.join(map(str, sa['stalled'])) or 'none'} | "
          f"{', '.join(map(str, sb['stalled'])) or 'none'} |",
          f"| scored fraction | {sa['scored_fraction']:.1%} | {sb['scored_fraction']:.1%} |"]
    ra, rb = sp["resources"]["aorlta"], sp["resources"]["baseline"]
    L += [f"| GPU-hours per 256 scored, median ({a.allocation.gpus_held} vs {b.allocation.gpus_held} GPUs held) | "
          f"{ra['gpu_hours_per256_median']:.2f} | {rb['gpu_hours_per256_median']:.2f} |",
          f"| CPU-node hours per 256 scored, median | {ra['cpu_node_hours_per256_median']:.2f} | "
          f"{rb['cpu_node_hours_per256_median']:.2f} |", ""]
    ia, ib = a.by_iteration(), b.by_iteration()
    L += ["| iteration | AORLTA wall | scored | per 256 | baseline wall | scored | per 256 | baseline / AORLTA |",
          "|---|---|---|---|---|---|---|---|"]
    for n in range(1, p.iterations + 1):
        x, y = ia[n], ib[n]
        ratio = sp["paired_ratio"].get(str(n))
        stall = lambda r: " (stall)" if r.stalled else ""  # noqa: E731
        L.append(f"| {n}{' (not reported)' if n < per256.FIRST_REPORTED else ''} | {_min(x.wall_s)}{stall(x)} | "
                 f"{x.scored}/{x.samples} | {_min(sa['per256_s'][str(n)])} | {_min(y.wall_s)}{stall(y)} | "
                 f"{y.scored}/{y.samples} | {_min(sb['per256_s'][str(n)])} | "
                 f"{'' if ratio is None else f'x{ratio:.2f}'} |")
    L.append("")
    da, db = c.descriptive["aorlta"], c.descriptive["baseline"]
    cats = sorted(set(da["phase_coverage_s_median"]) | set(db["phase_coverage_s_median"]))
    if cats:
        L += ["**Phase timings**, median over the reported iterations, minutes. Coverage is wall-clock time "
              "with the phase running (overlap counts once, so a column does not sum to the wall); the "
              "critical path does sum to it.", "",
              "| phase | AORLTA coverage | AORLTA critical path | baseline coverage |", "|---|---|---|---|"]
        for k in cats:
            L.append(f"| {k} | {_min(da['phase_coverage_s_median'].get(k))} | "
                     f"{_min(da['phase_critical_s_median'].get(k))} | {_min(db['phase_coverage_s_median'].get(k))} |")
        L.append("")
    sta, stb = da["health"]["check_share_status"], db["health"]["check_share_status"]
    L += [f"**Checks >= 70% of the iteration (verify70):** AORLTA {_statuses(sta)}; baseline {_statuses(stb)}. "
          "The share is the wall-clock time with a check running and no generation or training, over the wall; "
          "a range means a phase holds both check and generation time (a compile-repair envelope).", "",
          "| iteration | AORLTA check share | status | baseline check share | status |", "|---|---|---|---|---|"]
    for n in range(1, p.iterations + 1):
        L.append(f"| {n} | {_share(ia[n].check_share)} | {ia[n].check_status or 'not measured'} | "
                 f"{_share(ib[n].check_share)} | {ib[n].check_status or 'not measured'} |")
    L.append("")

    pa = crit["parity"]
    L += ["## 2. Learning parity", ""]
    if pa.verdict == NOT_EVALUATED:
        L += [f"Not evaluated: {pa.why}.", ""]
    else:
        nb = pa.numbers
        L += ["| iteration | AORLTA mean reward | baseline | difference |", "|---|---|---|---|"]
        for n in range(1, p.iterations + 1):
            L.append(f"| {n} | {ia[n].mean_train_reward:+.4f} | {ib[n].mean_train_reward:+.4f} | "
                     f"{nb['diffs'][str(n)]:+.4f} |")
        L += ["", f"Mean difference {nb['mean']:+.4f}, sd {nb['sd']:.4f}, t-interval ({nb['ci_level']:.0%}, "
              f"{nb['df']} df) [{nb['lo']:+.4f}, {nb['hi']:+.4f}] against +-{nb['tolerance']:g}: **{pa.verdict}**. "
              f"Iterations whose own difference reaches the tolerance: "
              f"{', '.join(map(str, nb['iterations_outside_tolerance'])) or 'none'} (descriptive).", ""]

    L += ["## 3. Held-out quality", ""]
    for cr in c.criteria:
        if not cr.key.startswith("heldout."):
            continue
        mode = cr.key.split(".", 1)[1]
        if cr.verdict == NOT_EVALUATED:
            L += [f"**{mode}:** not evaluated: {cr.why}.", ""]
            continue
        h = cr.numbers
        L += [f"**{mode}:** {h['items']} items; {h['samples']['aorlta']} AORLTA and {h['samples']['baseline']} "
              f"baseline samples. Verified-repair rate {h['rate']['aorlta']:.3f} against {h['rate']['baseline']:.3f}. "
              f"Mean per-item difference {h['mean_diff']:+.3f}, {h['ci_level']:.0%} cluster bootstrap over items "
              f"({h['resamples']} resamples, seed {h['seed']}) [{h['lo']:+.3f}, {h['hi']:+.3f}], margin "
              f"{h['margin']:g}: **{cr.verdict}**.", ""]

    L += ["## Compute curves (descriptive; not a criterion)", ""]
    if plots:
        L += [f"![{k}]({v})" for k, v in plots.items()] + [""]
    for key, bundle in (("aorlta", a), ("baseline", b)):
        cv = c.descriptive[key]["curves"]
        if not cv["points"]:
            L.append(f"- {bundle.system}: no validation points.")
            continue
        fits = "; ".join(f"against {axis.replace('_', '-')}: " + (
            f"A {f['a']:.3f}, B {f['b']:.2f}, C_mid {f['c_mid']:.1f}" if f.get("status") == "fitted"
            else f"not fitted ({f['reason']})") for axis, f in cv["fits"].items())
        L.append(f"- {bundle.system}: {len(cv['points'])} validation points, R_0 {cv['r0']:+.3f}, "
                 f"{cv['samples_per_point']} samples per point. {fits}.")
    L.append("")

    L += ["## Health", "", "| | AORLTA | baseline |", "|---|---|---|"]
    ha, hb = da["health"], db["health"]
    L.append(f"| unscored fraction | {ha['unscored_fraction']:.1%} | {hb['unscored_fraction']:.1%} |")
    L.append(f"| unscored by reason | {_reasons(ha)} | {_reasons(hb)} |")
    tr = lambda h: "n/a" if h["truncation_rate"] is None else (  # noqa: E731
        f"{h['truncation_rate']:.1%}{' **above 10%**' if h['truncation_flag'] else ''}")
    L.append(f"| truncation (final answers at the cap) | {tr(ha)} | {tr(hb)} |")
    nz = lambda h: "n/a" if h["noop_iterations"] is None else (", ".join(map(str, h["noop_iterations"])) or "none")  # noqa: E731
    L.append(f"| NOOP steps | {nz(ha)} | {nz(hb)} |")
    st = lambda h: "; ".join(f"{k}: {v}" for k, v in h["stalls"].items()) or "none"  # noqa: E731
    L.append(f"| stalls | {st(ha)} | {st(hb)} |")
    L.append("")
    return "\n".join(L) + "\n"


def _share(bounds: tuple[float, float] | None) -> str:
    if bounds is None:
        return "n/a"
    lo, hi = bounds
    return f"{lo:.1%}" if lo == hi else f"{lo:.1%} to {hi:.1%}"


def _statuses(counts: Mapping[str, int]) -> str:
    order = ("confirmed", "refuted", "undetermined", "not measured")
    return ", ".join(f"{k} in {counts[k]}" for k in order if counts.get(k))


def _reasons(h: Mapping[str, Any]) -> str:
    return ", ".join(f"{k} {v}" for k, v in h["unscored_by_reason"].items()) or "none"


def write_report(c: Comparison, out: str | Path) -> dict[str, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plots: dict[str, str] = {}
    series = {}
    for axis, label in (("gpu_hours", "GPU-hours"), ("wall_hours", "wall-hours")):
        series = {bundle.system: [(x[axis], x["reward"]) for x in c.descriptive[key]["curves"]["points"]]
                  for key, bundle in (("aorlta", c.aorlta), ("baseline", c.baseline))
                  if c.valid and c.descriptive[key]["curves"]["points"]}
        if series:
            name = f"validation-{axis.replace('_', '-')}.svg"
            title = "validation reward against " + label + (" (SYNTHETIC)" if c.aorlta.synthetic or c.baseline.synthetic else "")
            atomic_write_text(out / name, curves.svg_plot(series, title=title, x_label=label, y_label="validation reward"))
            plots[f"validation reward against {label}"] = name
    atomic_write_text(out / "report.md", markdown(c, plots=plots))
    atomic_write_json(out / "report.json", c.to_json())
    return {"report": out / "report.md", "json": out / "report.json", **{k: out / v for k, v in plots.items()}}


def run(protocol_path: str | Path, aorlta_path: str | Path, baseline_path: str | Path, out: str | Path, *,
        protocol_commit: str | None = None, repo: str | Path | None = None) -> Comparison:
    p, a, b = load_protocol(protocol_path), load_bundle(aorlta_path), load_bundle(baseline_path)
    prov = {"protocol_sha256": sha256_hex(read_bytes(protocol_path)), "aorlta_sha256": sha256_hex(read_bytes(aorlta_path)),
            "baseline_sha256": sha256_hex(read_bytes(baseline_path))}
    if protocol_commit is not None:
        prov["preregistration"] = verify_preregistration(protocol_path, protocol_commit,
                                                         repo if repo is not None else Path(protocol_path).parent)
    c = compare(p, a, b, provenance=prov)
    write_report(c, out)
    return c


# --- the synthetic sample (shows the format; never a result) ----------------------------------------------

SAMPLE_SEED = 20261102


def _synthetic_aorlta_run(rng: random.Random, n_iter: int, rewards: Sequence[float], prompt_sets: Sequence[list[str]]):
    """Spans and episode records for a made-up AORLTA run: 32 prompts x 8 chains, chains that end at
    different times, checks streaming on CPU workers, ASan in batches of up to 16 on 4 GPUs, gradients
    accumulated per group, then the sync tail. Iteration 6 hits the scoring deadline."""
    spans, episodes = [Span("engine.start", 0, 1.79e9 - 130, 1.79e9 - 2, {"engines": 4})], []
    origin = 1.79e9
    for n in range(1, n_iter + 1):
        o = origin
        spans.append(Span("select", n, o, o + 0.2, {}))
        chains = []
        for j, pid in enumerate(prompt_sets[n - 1]):
            for k in range(8):
                cid = f"sample/{n}/{pid}/{k}"
                t = o + 0.3
                turns = [min(rng.uniform(40, 330), 300.0)]
                if rng.random() < 0.55:
                    turns.append(rng.uniform(15, 90))
                    if rng.random() < 0.35:
                        turns.append(rng.uniform(10, 70))
                for i, d in enumerate(turns):
                    spans.append(Span("gen.turn0" if i == 0 else "gen.repair", n, t, t + d, {"chain_id": cid}))
                    t += d
                    spans.append(Span("compile", n, t, t + 0.6, {"chain_id": cid}))
                    t += 0.6
                chains.append((cid, pid, t, turns[-1] >= 300.0))
        barrier = max(t for _, _, t, _ in chains)
        resolved, asan_free = {}, [o] * 4
        queue = sorted(chains, key=lambda c: c[2])
        for start in range(0, len(queue), 16):
            batch = queue[start:start + 16]
            ready = []
            for cid, _, t, _ in batch:
                spans.append(Span("gate.parse", n, t, t + 0.2, {"candidate_id": cid}))
                for leg in ("ref.native", "san.consan", "san.waitcheck"):
                    spans.append(Span(leg, n, t + 0.2, t + 0.2 + rng.uniform(3, 16), {"candidate_id": cid}))
                ready.append(t + 0.2)
            gpu = min(range(4), key=lambda g: asan_free[g])
            begin = max(max(ready), asan_free[gpu])
            dur = rng.uniform(99, 180) * (40.0 if n == 6 and start == 112 else 1.0)  # iteration 6: a hung batch
            for (cid, _, _, _), r in zip(batch, ready):
                spans.append(Span("asan.queue", n, r, begin, {"candidate_id": cid, "request": f"r{n}.{start}"}))
            spans.append(Span("asan.run", n, begin, begin + dur, {"request": f"r{n}.{start}", "packages": len(batch)}))
            asan_free[gpu] = begin + dur
            for cid, _, _, _ in batch:
                resolved[cid] = begin + dur
        deadline = barrier + 1800.0 if n == 6 else math.inf
        last = min(max(resolved.values()), deadline)
        spans.append(Span("score.wait", n, barrier, last + 0.05, {}))
        train_t = o
        for j in range(32):
            group = [cid for cid, pid, _, _ in chains if pid == prompt_sets[n - 1][j]]
            done = max(min(resolved[c], deadline) for c in group)
            if done > last - 1:
                continue
            start_t, d = max(done, train_t), rng.uniform(4, 8)
            spans.append(Span("train.stream", n, start_t, start_t + d, {"group": prompt_sets[n - 1][j]}))
            train_t = start_t + d
        t = max(last + 0.05, train_t)
        for name, d in (("train.final", rng.uniform(15, 25)), ("merge", 1.2), ("write", 1.05), ("reload", 1.22)):
            spans.append(Span(name, n, t, t + d, {}))
            t += d
        spans.append(Span("landing", n, t, t + 1.8, {"verdict": ["LANDED"] * 4}))
        t += 1.8
        spans += [Span("record", n, t, t + 2.0, {}), Span("ckpt", n, t + 2.0, t + 4.5, {})]
        origin = t + 4.5 + 0.5
        for cid, pid, _, capped in chains:
            late = resolved[cid] > deadline
            infra = rng.random() < 0.025
            ok = not late and not infra
            episodes.append({"schema": per256.EPISODE_SCHEMA, "iteration": n, "chain_id": cid, "prompt_id": pid,
                             "status": "scored" if ok else "unscored",
                             "reward": max(-1.55, min(1.0, rewards[n - 1] + rng.gauss(0, 0.5))) if ok else None,
                             "unscored_reason": None if ok else ("scoring_deadline" if late else "asan_infra_error"),
                             "turns": [{"finish_reason": {"type": "length" if capped else "stop"}}], "final_turn": 0})
    return spans, episodes


def _synthetic_baseline(rng: random.Random, n_iter: int, rewards: Sequence[float], prompt_sets):
    """Per-iteration summaries for a made-up Miles-like baseline run with phase timestamps.
    Iteration 4 is a 10,912 s stall that the run's owner excluded."""
    summaries, records = [], []
    t = 1.70e9
    for n in range(1, n_iter + 1):
        phases, c = [], t
        plan = [("generate", rng.uniform(280, 520)), ("compile_repair", rng.uniform(120, 260)),
                ("gate", rng.uniform(60, 160)), ("scoring_wait", 10912.0 if n == 4 else rng.uniform(800, 1500))]
        for name, d in plan:
            phases.append({"phase": name, "start": round(c, 3), "end": round(c + d, 3)})
            c += d
        settle = rng.uniform(100, 300)
        phases.append({"phase": "asan_settle", "start": round(c - 0.5 * settle, 3), "end": round(c + 0.5 * settle, 3)})
        c += 0.5 * settle
        for name, d in (("train", rng.uniform(150, 260)), ("weight_sync", rng.uniform(25, 60))):
            phases.append({"phase": name, "start": round(c, 3), "end": round(c + d, 3)})
            c += d
        summaries.append({"iter": n, "start": round(t, 3), "end": round(c, 3), "phases": phases,
                          "reward_mean": rewards[n - 1]})
        unscored = rng.randint(6, 16)
        reasons = {"asan_timeout": unscored // 2, "sim_worker_error": unscored - unscored // 2}
        records.append({"iteration": n, "wall_s": round(c, 3) - round(t, 3), "samples": 256, "scored": 256 - unscored,
                        "stalled": n == 4, "stall_reason": "excluded by the run's owner" if n == 4 else None,
                        "mean_train_reward": rewards[n - 1], "prompts_sha256": prompt_set_sha256(prompt_sets[n - 1]),
                        "unscored_by_reason": reasons, "truncated": rng.randint(8, 22), "noop": None})
        t = c + rng.uniform(5, 20)
    return summaries, records


SAMPLE_MAPPING = """\
schema: aorlta.verify70.mapping/1
fields: {iteration: iter, t_start: start, t_end: end, wall_s: null, phases: phases, durations: null}
phase_entry: {name: phase, t_start: start, t_end: end}
classes:
  check: [gate, scoring_wait, asan_settle]
  mixed: [compile_repair]
  generation: [generate]
  training: [train]
  other: [weight_sync]
"""

SAMPLE_PROTOCOL = """\
schema: aorlta.measure.protocol/1
title: "SYNTHETIC SAMPLE: AORLTA against a Miles-like baseline"
iterations: 10
held_equal: [merged_base_sha256, prompt_pool_sha256, order_seed, P, K, reward_version, check_settings_sha256,
             sampling, lora, loss, lr, allocation]
speed: {statistic: median, min_speedup: 1.0}
parity: {tolerance: 0.05, ci_level: 0.95}
heldout: {modes: [full, evidence_only], margin: 0.05, ci_level: 0.95, resamples: 10000, seed: 20261102}
curves:
  burn_in_gpu_hours: 4.0
  burn_in_wall_hours: 0.5
  a_range: [0.30, 0.80]
  gpu_hours_cmid_range: [1.0, 200.0]
  wall_hours_cmid_range: [0.1, 20.0]
"""


def make_sample(out: str | Path, seed: int = SAMPLE_SEED) -> Comparison:
    """Write a synthetic protocol, both bundles, the baseline's summaries and mapping, and the report."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    n_iter = 10
    base_rewards = [-0.42 + 0.035 * n + rng.gauss(0, 0.015) for n in range(1, n_iter + 1)]
    prompt_sets = [[f"standin:k{rng.randrange(1000):03d}x{n}{j}" for j in range(32)] for n in range(1, n_iter + 1)]
    spans, episodes = _synthetic_aorlta_run(rng, n_iter, [r + rng.gauss(0, 0.01) for r in base_rewards], prompt_sets)
    summaries, base_records = _synthetic_baseline(rng, n_iter, base_rewards, prompt_sets)
    atomic_write_text(out / "baseline-mapping.yaml", SAMPLE_MAPPING)
    atomic_write_bytes(out / "baseline-summaries.json", (json.dumps(summaries, indent=1) + "\n").encode())
    v70 = verify70.run(out / "baseline-summaries.json", out / "baseline-mapping.yaml")
    for rec, row in zip(base_records, v70["rows"]):
        rec["check_share"] = [row["lower"], row["upper"]]
        rec["check_status"] = row["status"]
        sec = row["seconds"]
        rec["category_coverage_s"] = {k: sec[k] for k in verify70.CATEGORIES if sec.get(k)}
    held = {"merged_base_sha256": "0" * 64, "prompt_pool_sha256": "1" * 64, "order_seed": 7, "P": 32, "K": 8,
            "reward_version": "standin-v11", "check_settings_sha256": "2" * 64,
            "sampling": {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 4096}, "lora": {"r": 16, "alpha": 32},
            "loss": {"pg_loss": "ppo_clip", "eps": [0.2, 0.2]}, "lr": 1e-5,
            "allocation": "4 training GPUs + CPU node + partner ASan runner"}
    items = [f"item{i:02d}" for i in range(33)]
    p_item = [min(0.95, max(0.02, rng.betavariate(1.5, 3.5))) for _ in items]

    def eval_modes(shift: float) -> dict[str, dict[str, list[int]]]:
        return {mode: {it: [int(rng.random() < min(1.0, max(0.0, p + shift + delta))) for _ in range(4)]
                       for it, p in zip(items, p_item)} for mode, delta in (("full", 0.0), ("evidence_only", -0.08))}

    gpu_h = lambda walls, g: [sum(walls[:n]) * g / 3600 for n in range(1, n_iter + 1)]  # noqa: E731
    base_walls = [r["wall_s"] for r in base_records]
    base_val = Validation(r0=0.27, samples_per_point=1600, points=tuple(
        ValidationPoint(n, gh, sum(base_walls[:n]) / 3600, 0.27 + 0.012 * n + rng.gauss(0, 0.01))
        for n, gh in zip(range(1, n_iter + 1), gpu_h(base_walls, 8)) if n % 2 == 0))
    baseline = Bundle(BUNDLE_SCHEMA, "Miles-like baseline", True, held, Allocation(8, 1, "4 + 4 ASan"),
                      tuple(_decode(IterationRecord, r, f"baseline.iterations[{i}]") for i, r in enumerate(base_records)),
                      Heldout(n_iter, "synthetic-eval-1", {m: {k: tuple(v) for k, v in d.items()}
                                                           for m, d in eval_modes(0.0).items()}), base_val)
    a_tmp = bundle_from_run(spans, episodes, system="AORLTA", held_equal=held,
                            allocation=Allocation(8, 1, "4 + 4 ASan"), stalls={6: "scoring deadline"}, synthetic=True)
    a_walls = [r.wall_s for r in a_tmp.iterations]
    a_val = Validation(r0=0.27, samples_per_point=1600, points=tuple(
        ValidationPoint(n, gh, sum(a_walls[:n]) / 3600, base_val.points[n // 2 - 1].reward + rng.gauss(0, 0.01))
        for n, gh in zip(range(1, n_iter + 1), gpu_h(a_walls, 8)) if n % 2 == 0))
    aorlta = Bundle(BUNDLE_SCHEMA, a_tmp.system, True, held, a_tmp.allocation, a_tmp.iterations,
                    Heldout(n_iter, "synthetic-eval-1", {m: {k: tuple(v) for k, v in d.items()}
                                                         for m, d in eval_modes(0.01).items()}), a_val)
    atomic_write_text(out / "protocol.yaml", SAMPLE_PROTOCOL)
    atomic_write_json(out / "aorlta.bundle.json", plain(aorlta))
    atomic_write_json(out / "baseline.bundle.json", plain(baseline))
    return run(out / "protocol.yaml", out / "aorlta.bundle.json", out / "baseline.bundle.json", out)


# --- CLI ----------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aorlta.measure.compare", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="compare two bundles under a pre-registered protocol")
    r.add_argument("--protocol", type=Path, required=True)
    r.add_argument("--aorlta", type=Path, required=True)
    r.add_argument("--baseline", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--protocol-commit", help="the pre-registration commit; the protocol must be identical to it")
    r.add_argument("--repo", type=Path, help="the repository holding the pre-registration (default: the protocol's)")
    s = sub.add_parser("sample", help="write a report from synthetic data, to show the format")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "run":
            c = run(args.protocol, args.aorlta, args.baseline, args.out, protocol_commit=args.protocol_commit,
                    repo=args.repo)
        else:
            c = make_sample(args.out, args.seed)
    except MeasureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{c.overall}: {c.overall_why}")
    print(f"report: {Path(args.out) / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
