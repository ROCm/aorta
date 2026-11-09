"""The check planner (design Section 6.3, C3): what the selector is asked and what it answers.

`PlanRequest`, `CheckPlan` and `CheckPlanner` are the doc's interface. Around them:

- `plan_problem` decides whether an answer can be used. A plan names every leg it was offered and no
  other, each as {"decision": "run" | "skip", "p_finding": float in [0, 1] or None, "score": float
  or None}. Anything else (an unknown leg, a missing leg, a malformed entry or id) rejects the whole
  plan; a rejected plan is logged and treated as no plan, so every leg runs.
- `plan_json` is how a plan is logged: ids, decisions and numbers only. The selector reads the
  candidate source, so its `context` may quote it; only the context's sha256 and length are logged
  (Section 6.10 item 3).
- `audit_draw` and `audit_coin`: the C3 audit coin, a function of the run seed and the candidate id
  alone. It is drawn when the candidate is submitted, before any plan is requested, and nothing the
  planner sees or answers can move it.
- `wilson_interval` and `SafetySwitch`: plan mode's per-leg guard.
- `is_defect`: a leg found something when its state is `finding` or `fail`.
- Three test planners: skip nothing, skip everything, skip at random.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..ids import HashRng, sha256_text
from ..schemas import SETTLED_STATES, LegResult

DECISIONS = ("run", "skip")
LEG_ENTRY_KEYS = frozenset({"decision", "p_finding", "score"})
LEG_RE = re.compile(r"(?:gate|ref|san)\.[a-z0-9_]{1,48}")
ID_RE = re.compile(r"[\x21-\x7e]{1,128}")  # selector versions and plan ids: printable ASCII, no spaces
DEFECT_STATES = frozenset({"finding", "fail"})
AUDIT_SCHEMA = "aorlta.selector.audit/1"
Z95 = 1.959963984540054  # two-sided 95%


# --- the doc's records and protocol ------------------------------------------------------------------


@dataclass(frozen=True)
class PlanRequest:
    candidate_id: str
    source_sha256: str
    kernel_features: dict  # what the selector may read: the candidate source (inside the trusted boundary),
    #                        gate and compile results, static features
    available_legs: list[str]  # e.g. ["ref.native", "ref.edge", "san.consan", "san.waitcheck", "san.asan"]
    leg_cost_s: dict[str, float]  # running medians, so the selector can trade time against risk
    budget_s: float | None


@dataclass(frozen=True)
class CheckPlan:
    selector_version: str
    plan_id: str
    legs: dict[str, dict]  # leg -> {"decision": "run"|"skip", "p_finding": float|None, "score": float|None}
    context: str | None  # never injected into prompts in v1


@runtime_checkable
class CheckPlanner(Protocol):
    version: str

    def plan(self, req: PlanRequest) -> CheckPlan: ...  # must answer within plan_timeout_s (default 5 s)

    def score(self, bug: dict, candidates: list[dict]) -> list[dict]: ...  # the earlier (bug, candidates) ->
    #                                                                       {score, context} interface


# --- validating and logging a plan -----------------------------------------------------------------


def _number(x: Any) -> bool:
    return type(x) in (int, float) and math.isfinite(x)


def plan_problem(plan: Any, available: Sequence[str]) -> str | None:
    """Why `plan` cannot be used for a request that offered `available`, or None if it can."""
    if not isinstance(plan, CheckPlan):
        return "not_a_check_plan"
    for name in ("selector_version", "plan_id"):
        value = getattr(plan, name)
        if not (isinstance(value, str) and ID_RE.fullmatch(value)):
            return f"bad_{name}"
    if plan.context is not None and not isinstance(plan.context, str):
        return "bad_context"
    if not isinstance(plan.legs, Mapping):
        return "bad_legs"
    offered = set(available)
    if any(leg not in offered for leg in plan.legs):
        return "unknown_leg"
    if any(leg not in plan.legs for leg in offered):
        return "missing_leg"
    for entry in plan.legs.values():
        if not isinstance(entry, Mapping) or set(entry) != LEG_ENTRY_KEYS:
            return "bad_leg_entry"
        if entry["decision"] not in DECISIONS or type(entry["decision"]) is not str:
            return "bad_decision"
        p = entry["p_finding"]
        if p is not None and not (_number(p) and 0.0 <= p <= 1.0):
            return "bad_p_finding"
        if entry["score"] is not None and not _number(entry["score"]):
            return "bad_score"
    return None


def safe_leg_names(names: Iterable[Any]) -> list[str]:
    """Leg names fit for a log: a name in the leg vocabulary as is, anything else as a short hash."""
    out = []
    for name in names:
        if isinstance(name, str) and LEG_RE.fullmatch(name):
            out.append(name)
        else:
            out.append("sha256:" + sha256_text(repr(name))[:12])
    return sorted(out)


def plan_json(plan: CheckPlan) -> dict:
    """A valid plan as it is logged; the context only as its hash and length."""
    legs = {leg: {"decision": e["decision"],
                  "p_finding": None if e["p_finding"] is None else float(e["p_finding"]),
                  "score": None if e["score"] is None else float(e["score"])}
            for leg, e in sorted(plan.legs.items())}
    return {
        "selector_version": plan.selector_version,
        "plan_id": plan.plan_id,
        "legs": legs,
        "skip": [leg for leg, e in legs.items() if e["decision"] == "skip"],
        "context_sha256": None if plan.context is None else sha256_text(plan.context),
        "context_chars": None if plan.context is None else len(plan.context),
    }


def is_settled(result: LegResult) -> bool:
    return result.state in SETTLED_STATES


def is_defect(result: LegResult) -> bool:
    """The leg found something against the candidate: a sanitizer finding, or a fail it attributes to
    the candidate (a mismatch, a crash, a fault, a candidate timeout)."""
    return result.state in DEFECT_STATES


# --- the audit coin --------------------------------------------------------------------------------


def audit_draw(seed: int, candidate_id: str) -> float:
    """The candidate's audit draw, uniform in [0, 1): a hash of the run seed and the candidate id only."""
    if not (isinstance(seed, int) and not isinstance(seed, bool)):
        raise ValueError(f"the audit seed must be an int, got {seed!r}")
    if not (isinstance(candidate_id, str) and candidate_id):
        raise ValueError(f"candidate_id must be a non-empty str, got {candidate_id!r}")
    return HashRng([AUDIT_SCHEMA, seed, candidate_id]).uniform("audit")


def audit_coin(u: float, fraction: float) -> bool:
    """Audited at `fraction`: P(u < f) = f for a uniform u, and the coins at two fractions nest."""
    return u < fraction


# --- the Wilson interval and the safety switch -----------------------------------------------------


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval for k events in n trials; (0, 1) when n is 0."""
    if not (isinstance(k, int) and isinstance(n, int) and 0 <= k <= n):
        raise ValueError(f"need 0 <= k <= n, got k={k!r}, n={n!r}")
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lower = 0.0 if k == 0 else max(0.0, center - half)
    upper = 1.0 if k == n else min(1.0, center + half)
    return lower, upper


def wilson_upper(k: int, n: int, z: float = Z95) -> float:
    return wilson_interval(k, n, z)[1]


SWITCH_STATES = ("warming", "armed", "tripped")


@dataclass
class _LegSwitch:
    observations: int = 0
    misses: int = 0
    state: str = "warming"
    armed_at: int | None = None  # observation count when it armed
    tripped_at: int | None = None


class SafetySwitch:
    """Plan mode's per-leg guard (Section 6.3, Safety switch).

    Each audited sample with a usable plan and with the leg settled is one observation of that leg; it
    is a miss if the plan would have skipped the leg and the leg found a defect. The miss rate is misses
    per observation, read as its Wilson 95% upper bound, so a leg is trusted only while the evidence
    puts its miss rate at or below `max_miss_rate`.

    - `warming` (at the start, and after a reset): the plan may not skip the leg. The leg arms once the
      bound would still be at or below `max_miss_rate` if the next observation were a miss. Arming as
      soon as the bound itself reaches the limit would leave it on the limit, so the next miss would
      fire the switch for a selector missing at a fraction of the limit. With no misses arming takes
      279 observations at m = 2%.
    - `armed`: the plan may skip the leg. When the bound exceeds `max_miss_rate` the switch fires.
    - `tripped`: latched. The leg runs on every candidate until `reset(leg)`, by hand, which clears the
      leg's counts and returns it to `warming`.
    """

    def __init__(self, legs: Iterable[str], max_miss_rate: float, *, z: float = Z95):
        if not (_number(max_miss_rate) and 0.0 <= max_miss_rate <= 1.0):
            raise ValueError(f"max_miss_rate must be in [0, 1], got {max_miss_rate!r}")
        self.max_miss_rate = float(max_miss_rate)
        self.z = z
        self._legs = {leg: _LegSwitch() for leg in legs}
        for leg in self._legs:
            self._update(leg)

    @property
    def legs(self) -> tuple[str, ...]:
        return tuple(self._legs)

    def _update(self, leg: str) -> str:
        s = self._legs[leg]
        if s.state == "warming":
            if wilson_upper(s.misses + 1, s.observations + 1, self.z) <= self.max_miss_rate:
                s.state, s.armed_at = "armed", s.observations
        elif s.state == "armed" and wilson_upper(s.misses, s.observations, self.z) > self.max_miss_rate:
            s.state, s.tripped_at = "tripped", s.observations
        return s.state

    def observe(self, leg: str, *, would_skip: bool, defect: bool) -> str:
        """One audited observation of `leg`; returns the leg's state after it."""
        s = self._legs[leg]
        s.observations += 1
        s.misses += bool(would_skip and defect)
        return self._update(leg)

    def allows_skip(self, leg: str) -> bool:
        return self._legs[leg].state == "armed"

    def state(self, leg: str) -> str:
        return self._legs[leg].state

    def upper(self, leg: str) -> float:
        s = self._legs[leg]
        return wilson_upper(s.misses, s.observations, self.z)

    def reset(self, leg: str) -> None:
        """By hand only: a fresh evidence window for the leg."""
        self._legs[leg] = _LegSwitch()
        self._update(leg)

    def snapshot(self) -> dict:
        return {leg: {"state": s.state, "observations": s.observations, "misses": s.misses,
                      "upper": self.upper(leg), "armed_at": s.armed_at, "tripped_at": s.tripped_at}
                for leg, s in self._legs.items()}


# --- test planners ---------------------------------------------------------------------------------


def _entry(decision: str) -> dict:
    return {"decision": decision, "p_finding": None, "score": None}


def _plan_id(version: str, req: PlanRequest) -> str:
    return f"{version}:{sha256_text(req.candidate_id)[:16]}"


class _TestPlanner:
    version = "test/0"

    def decide(self, req: PlanRequest, leg: str) -> str:
        raise NotImplementedError

    def plan(self, req: PlanRequest) -> CheckPlan:
        return CheckPlan(self.version, _plan_id(self.version, req),
                         {leg: _entry(self.decide(req, leg)) for leg in req.available_legs}, None)

    def score(self, bug: dict, candidates: list[dict]) -> list[dict]:
        return [{"score": None, "context": None} for _ in candidates]


class SkipNothing(_TestPlanner):
    version = "test.skip-nothing/1"

    def decide(self, req: PlanRequest, leg: str) -> str:
        return "run"


class SkipEverything(_TestPlanner):
    version = "test.skip-everything/1"

    def decide(self, req: PlanRequest, leg: str) -> str:
        return "skip"


class SkipAtRandom(_TestPlanner):
    """Skips each leg with probability `p_skip` (one number, or one per leg), from a hash of its seed,
    the candidate id and the leg: the same request always gets the same plan."""

    SCHEMA = "aorlta.selector.skip_at_random/1"

    def __init__(self, seed: int = 0, p_skip: float | Mapping[str, float] = 0.5):
        rates = p_skip.values() if isinstance(p_skip, Mapping) else [p_skip]
        if not all(_number(p) and 0.0 <= p <= 1.0 for p in rates):
            raise ValueError(f"p_skip must be in [0, 1], got {p_skip!r}")
        self.seed = seed
        self.p_skip = dict(p_skip) if isinstance(p_skip, Mapping) else float(p_skip)
        self.version = f"test.skip-at-random/1:seed={seed}"

    def draw(self, candidate_id: str, leg: str) -> float:
        return HashRng([self.SCHEMA, self.seed, candidate_id, leg]).uniform("skip")

    def decide(self, req: PlanRequest, leg: str) -> str:
        p = self.p_skip.get(leg, 0.0) if isinstance(self.p_skip, dict) else self.p_skip
        return "skip" if self.draw(req.candidate_id, leg) < p else "run"
