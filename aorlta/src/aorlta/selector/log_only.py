"""The check planner inside the reward service (design Section 6.3, Modes; C3).

`SelectorHook` is what `RewardService` calls. Per candidate:

1. At submit, the audit coin is drawn (`base.audit_draw`), before any plan is requested.
2. When the candidate reaches its first planned stage, which is after S1, the planner is asked on one
   of the hook's own threads, with a deadline `plan_timeout_s` from the request. The planned legs
   are every profile leg that is not a gate: in the full stage layout ref.native, ref.edge, san.consan,
   san.waitcheck and san.asan.
3. `log_only` (v1) never waits for the plan, and every leg runs exactly as it would with no planner.
   When the candidate is decided while its plan is still on time, the result waits for the plan (at
   most `plan_timeout_s` from the request), so every plan that answers in time is logged. A missing,
   late, failed or rejected plan is logged as `plan_unavailable` with its reason.
4. The plan record goes into `RewardResult.plan`, the coin into `RewardResult.audit` (in log_only, the
   coin it would have drawn), and one line per candidate into the plan log, which the offline
   evaluator (`plan_eval`) reads.

Plan mode (v2) is built and off. The stage waits for the plan; a leg the plan skips becomes
`skipped_by_plan` unless the candidate is audited or the safety switch does not allow skipping that
leg; a plan that is unavailable skips nothing. It needs `allow_plan_mode=True`, which no run config
can set: `config.py` refuses `selector.mode: plan` until a reward version for it is agreed (D6). Only
scoring option A ("unscored": a skipped leg reaches the reward function as unsettled) exists; option
B ("capped") needs a reward version that implements the cap (Q8), and is refused.

All of the hook's state is changed on the service's coordinator thread. The planner threads only
call `planner.plan` and post the answer back as a ("plan", token, outcome) event.
"""

from __future__ import annotations

import itertools
import logging
import queue
import statistics
import threading
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_bytes
from ..config import SelectorSection
from ..ids import canonical_json
from ..schemas import SETTLED_STATES, Candidate, RewardResult, to_dict
from .base import (
    CheckPlanner,
    PlanRequest,
    SafetySwitch,
    audit_coin,
    audit_draw,
    is_defect,
    plan_json,
    plan_problem,
    safe_leg_names,
)

log = logging.getLogger(__name__)

PLAN_RECORD_SCHEMA = "aorlta.selector.plan_record/1"
PLAN_LOG_SCHEMA = "aorlta.selector.plan_log/1"
COST_WINDOW = 4096  # leg run times kept per leg for the running medians


@dataclass
class CandidatePlan:
    """One candidate's audit coin and plan, owned by the reward service's coordinator."""

    candidate_id: str
    source_sha256: str
    audit_u: float
    audit: bool
    drawn_at: float  # UTC, at submit
    status: str = "not_requested"  # not_requested | pending | ok | plan_unavailable
    reason: str | None = None
    token: int | None = None
    available_legs: tuple[str, ...] = ()
    leg_cost_s: dict = field(default_factory=dict)
    requested_at: float | None = None  # UTC
    t_requested: float | None = None  # service clock
    deadline: float | None = None  # service clock
    answered_after_s: float | None = None
    plan: dict | None = None  # the validated plan, as logged (`base.plan_json`)
    rejected_legs: list = field(default_factory=list)
    applied: bool = False  # plan mode: the plan's skips have been decided
    skipped: list = field(default_factory=list)  # plan mode: legs skipped
    withheld: list = field(default_factory=list)  # plan mode: skips not applied (audited, or the switch)


class _PlannerThreads:
    """A fixed pool of daemon threads that call the planner. A planner that hangs holds a thread, never
    the service; a request still queued at its deadline is dropped (the coordinator has expired it)."""

    def __init__(self, planner: CheckPlanner, n: int, now: Callable[[], float], post: Callable[[tuple], None],
                 max_queued: int):
        self.planner, self.now, self.post, self.max_queued = planner, now, post, max_queued
        self.q: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.threads = [threading.Thread(target=self._main, name=f"selector-planner-{i}", daemon=True) for i in range(n)]
        for t in self.threads:
            t.start()

    def submit(self, token: int, req: PlanRequest, deadline: float) -> bool:
        if self.stop.is_set() or self.q.qsize() >= self.max_queued:
            return False
        self.q.put((token, req, deadline))
        return True

    def _main(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            token, req, deadline = item
            if self.stop.is_set() or self.now() >= deadline:
                continue
            try:
                outcome = ("ok", self.planner.plan(req), self.now())
            except BaseException as exc:  # never let a planner take a thread down silently
                outcome = ("error", type(exc).__name__, self.now())
            if not self.stop.is_set():
                self.post(("plan", token, outcome))

    def close(self) -> None:
        self.stop.set()
        for _ in self.threads:
            self.q.put(None)


class SelectorHook:
    def __init__(
        self,
        *,
        mode: str,
        planner: CheckPlanner | None,
        audit_seed: int,
        audit_fraction: float = 0.2,
        plan_timeout_s: float = 5.0,
        max_miss_rate: float = 0.02,
        scoring: str = "unscored",
        allow_plan_mode: bool = False,
        threads: int = 8,
        max_queued: int = 1024,
    ):
        if mode not in ("log_only", "plan"):
            raise ValueError(f"mode must be 'log_only' or 'plan' (no hook is built for 'off'), got {mode!r}")
        if mode == "plan":
            if not allow_plan_mode:
                raise ValueError("plan mode is v2 (decision D6): it needs a reward version that supports it, and is "
                                 "built behind allow_plan_mode=True for tests only")
            if scoring != "unscored":
                raise ValueError(f"scoring {scoring!r}: only 'unscored' (option A) exists; a capped reward needs a "
                                 "reward version that implements the cap (Q8)")
        if not (isinstance(audit_seed, int) and not isinstance(audit_seed, bool)):
            raise ValueError(f"audit_seed must be an int (the run seed), got {audit_seed!r}")
        if not (0.0 <= audit_fraction <= 1.0):
            raise ValueError(f"audit_fraction must be in [0, 1], got {audit_fraction!r}")
        if not plan_timeout_s > 0:
            raise ValueError(f"plan_timeout_s must be > 0, got {plan_timeout_s!r}")
        if planner is not None and not (callable(getattr(planner, "plan", None)) and isinstance(getattr(planner, "version", None), str)):
            raise ValueError("the planner must have a str `version` and a `plan(PlanRequest)` method")
        self.mode = mode
        self.planner = planner
        self.audit_seed = audit_seed
        self.audit_fraction = float(audit_fraction)
        self.plan_timeout_s = float(plan_timeout_s)
        self.max_miss_rate = float(max_miss_rate)
        self.scoring = scoring
        self.n_threads = threads
        self.max_queued = max_queued
        self.switch: SafetySwitch | None = None
        self.planned_legs: tuple[str, ...] = ()
        self.plan_stage: str | None = None
        self._clock: Any = None
        self._post: Callable[[tuple], None] | None = None
        self._threads: _PlannerThreads | None = None
        self._tokens = itertools.count(1)
        self._pending: dict[int, CandidatePlan] = {}
        self._by_cid: dict[str, CandidatePlan] = {}
        self._costs: dict[str, deque] = {}
        self._lines: list[dict] = []
        self._counts: Counter = Counter()

    @property
    def plan_mode(self) -> bool:
        return self.mode == "plan"

    def attach(self, *, profile: Any, clock: Any, post: Callable[[tuple], None]) -> None:
        """Called once by the service that uses this hook."""
        if self._clock is not None:
            raise RuntimeError("a SelectorHook serves exactly one reward service")
        planned = tuple(leg for leg in profile.legs if not leg.startswith("gate."))
        if not planned:
            raise ValueError(f"profile {profile.name!r} has no leg after its gates for a plan to choose")
        self.planned_legs = planned
        self.plan_stage = next(st.name for st in profile.stages if any(leg in planned for leg in st.legs))
        self._clock, self._post = clock, post
        self._costs = {leg: deque(maxlen=COST_WINDOW) for leg in planned}
        if self.plan_mode:
            self.switch = SafetySwitch(planned, self.max_miss_rate)
        if self.planner is not None:
            self._threads = _PlannerThreads(self.planner, self.n_threads, clock.now, post, self.max_queued)

    # --- per candidate -----------------------------------------------------------------------------

    def on_submit(self, cand: Candidate, source_sha256: str) -> CandidatePlan:
        u = audit_draw(self.audit_seed, cand.candidate_id)
        cp = CandidatePlan(cand.candidate_id, source_sha256, u, audit_coin(u, self.audit_fraction), self._clock.wall())
        self._by_cid[cand.candidate_id] = cp
        self._counts["submitted"] += 1
        self._counts["audited"] += cp.audit
        return cp

    def coin(self, candidate_id: str) -> bool | None:
        """The audit coin already drawn for a candidate (None if it was never submitted)."""
        cp = self._by_cid.get(candidate_id)
        return None if cp is None else cp.audit

    def request(self, cp: CandidatePlan, cand: Candidate, kernel_features: dict) -> None:
        """Ask the planner; returns at once."""
        if cp.status != "not_requested":
            return
        now = self._clock.now()
        cp.available_legs = self.planned_legs
        cp.leg_cost_s = self.leg_cost_s()
        cp.requested_at, cp.t_requested, cp.deadline = self._clock.wall(), now, now + self.plan_timeout_s
        self._counts["requested"] += 1
        if self.planner is None:
            self._unavailable(cp, "no_planner")
            return
        req = PlanRequest(cand.candidate_id, cp.source_sha256, kernel_features, list(self.planned_legs),
                          dict(cp.leg_cost_s), None)
        cp.token = next(self._tokens)
        cp.status = "pending"
        self._pending[cp.token] = cp
        if not self._threads.submit(cp.token, req, cp.deadline):
            self._pending.pop(cp.token, None)
            self._unavailable(cp, "planner_busy")

    def on_answer(self, token: int, outcome: tuple) -> CandidatePlan | None:
        """The planner's answer (on the coordinator). None if the request was already expired or
        abandoned: a late answer is counted and dropped."""
        cp = self._pending.pop(token, None)
        if cp is None:
            self._counts["late_answers"] += 1
            return None
        kind, value, t_answer = outcome
        cp.answered_after_s = t_answer - cp.t_requested
        if t_answer > cp.deadline:
            self._counts["late_answers"] += 1
            self._unavailable(cp, "timeout")
        elif kind != "ok":
            self._unavailable(cp, f"planner_error:{value}")
        else:
            problem = plan_problem(value, cp.available_legs)
            if problem is not None:
                if problem in ("unknown_leg", "missing_leg"):
                    legs = getattr(value, "legs", {}) or {}
                    cp.rejected_legs = (safe_leg_names(k for k in legs if k not in cp.available_legs)
                                        if problem == "unknown_leg" else
                                        safe_leg_names(k for k in cp.available_legs if k not in legs))
                self._unavailable(cp, f"rejected:{problem}")
                log.warning("selector: plan for %s rejected (%s)", cp.candidate_id, problem)
            else:
                cp.plan = plan_json(value)
                cp.status = "ok"
                self._counts["ok"] += 1
        return cp

    def expire(self, now: float) -> list[CandidatePlan]:
        """Requests past their deadline become plan_unavailable(timeout); returns them."""
        out = []
        for token, cp in list(self._pending.items()):
            if now >= cp.deadline:
                del self._pending[token]
                self._unavailable(cp, "timeout")
                out.append(cp)
        return out

    def next_deadline(self) -> float | None:
        return min((cp.deadline for cp in self._pending.values()), default=None)

    def pending(self, cp: CandidatePlan) -> bool:
        return cp.status == "pending"

    def abandon(self, cp: CandidatePlan, reason: str) -> None:
        """The service stops waiting (drain deadline, close): a pending plan becomes unavailable."""
        if cp.status == "pending":
            self._pending.pop(cp.token, None)
            self._unavailable(cp, reason)
        elif cp.status == "not_requested" and cp.reason is None:
            cp.reason = reason

    def _unavailable(self, cp: CandidatePlan, reason: str) -> None:
        cp.status, cp.reason = "plan_unavailable", reason
        self._counts[f"plan_unavailable:{reason.split(':')[0]}"] += 1

    def skips(self, cp: CandidatePlan) -> list[str]:
        """Plan mode: the legs to mark `skipped_by_plan`. None on an audited candidate, none without a
        usable plan, and none the safety switch does not allow."""
        if not self.plan_mode:
            raise RuntimeError("skips() is plan mode only")
        cp.applied = True
        wanted = list(cp.plan["skip"]) if cp.status == "ok" else []
        if cp.audit:
            cp.skipped, cp.withheld = [], wanted
        else:
            cp.skipped = [leg for leg in wanted if self.switch.allows_skip(leg)]
            cp.withheld = [leg for leg in wanted if leg not in cp.skipped]
        return list(cp.skipped)

    # --- records -----------------------------------------------------------------------------------

    def record(self, cp: CandidatePlan, *, reward_cache_hit: bool = False) -> dict:
        """The plan record logged in RewardResult.plan."""
        return plan_record(cp, self.mode, reward_cache_hit=reward_cache_hit)

    def on_resolved(self, cp: CandidatePlan, cand: Candidate, result: RewardResult) -> None:
        """The candidate's result is final: append its plan-log line; in plan mode an audited sample
        with a usable plan is one observation for the safety switch, per settled planned leg."""
        self._lines.append(plan_log_line(cp, cand, result, self.audit_fraction))
        if self.plan_mode and cp.audit and cp.status == "ok":
            skip = set(cp.plan["skip"])
            for leg in self.planned_legs:
                r = result.legs.get(leg)
                if r is not None and r.state in SETTLED_STATES:
                    self.switch.observe(leg, would_skip=leg in skip, defect=is_defect(r))

    def log_lines(self) -> list[dict]:
        return list(self._lines)

    def write_plan_log(self, path: str | Path) -> Path:
        """The plan log as canonical-JSON lines, written atomically."""
        return atomic_write_bytes(path, b"".join(canonical_json(line) + b"\n" for line in self._lines))

    # --- leg costs ---------------------------------------------------------------------------------

    def observe_cost(self, leg: str, seconds: float) -> None:
        costs = self._costs.get(leg)
        if costs is not None and seconds >= 0:
            costs.append(float(seconds))

    def leg_cost_s(self) -> dict[str, float]:
        """Running medians of the planned legs' run times; a leg with no run yet is absent, not free."""
        return {leg: statistics.median(c) for leg, c in self._costs.items() if c}

    def stats(self) -> dict:
        return {"mode": self.mode, **dict(self._counts), "pending": len(self._pending),
                "switch": None if self.switch is None else self.switch.snapshot()}

    def close(self) -> None:
        if self._threads is not None:
            self._threads.close()


def plan_record(cp: CandidatePlan, mode: str, *, reward_cache_hit: bool = False) -> dict:
    """The plan record logged in RewardResult.plan: ids, decisions, numbers and timings, never text."""
    status, reason = cp.status, cp.reason
    if status == "not_requested" and reason is None:
        reason = "reward_cache_hit" if reward_cache_hit else "decided_before_plan"
    if status == "pending":  # the service never publishes a pending plan; if it did, it is not a plan
        status, reason = "plan_unavailable", "pending"
    plan = cp.plan if status == "ok" else None
    requested = cp.requested_at is not None
    return {
        "schema": PLAN_RECORD_SCHEMA,
        "mode": mode,
        "status": status,
        "reason": reason,
        "selector_version": None if plan is None else plan["selector_version"],
        "plan_id": None if plan is None else plan["plan_id"],
        "legs": None if plan is None else plan["legs"],
        "skip": [] if plan is None else list(plan["skip"]),
        "context_sha256": None if plan is None else plan["context_sha256"],
        "context_chars": None if plan is None else plan["context_chars"],
        "rejected_legs": list(cp.rejected_legs),
        "available_legs": list(cp.available_legs) if requested else None,
        "leg_cost_s": dict(cp.leg_cost_s) if requested else None,
        "budget_s": None,
        "requested_at": cp.requested_at,
        "answered_after_s": cp.answered_after_s,
        "applied": cp.applied,
        "skipped": list(cp.skipped),
        "withheld": list(cp.withheld),
    }


def plan_log_line(cp: CandidatePlan, cand: Candidate, result: RewardResult, audit_fraction: float) -> dict:
    """One plan-log line: the candidate's ids and the inputs its reward reads besides the legs (never
    its text), the audit coin with its draw and fraction, and the final RewardResult with every leg."""
    return {
        "schema": PLAN_LOG_SCHEMA,
        "candidate": {
            "candidate_id": cand.candidate_id, "run_id": cand.run_id, "iteration": cand.iteration,
            "prompt_id": cand.prompt_id, "k": cand.k, "policy_version": cand.policy_version,
            "check_profile": cand.check_profile, "source_sha256": cp.source_sha256,
            "response_tokens": cand.response_tokens, "max_new_tokens": cand.max_new_tokens,
            "task": to_dict(cand.task),
        },
        "audit": {"coin": cp.audit, "u": cp.audit_u, "fraction": float(audit_fraction), "drawn_at": cp.drawn_at},
        "result": to_dict(result),
    }


def selector_hook(section: SelectorSection, planner: CheckPlanner | None, *, audit_seed: int,
                  threads: int = 8) -> SelectorHook | None:
    """The hook a run config asks for: None for mode `off`, a log-only hook for `log_only`.

    `audit_seed` is the run seed (`orchestrator.run_seed`). With `endpoint` null and no planner, every
    plan is logged as plan_unavailable(no_planner) and the audit coins are still drawn. An endpoint
    needs a client for an external selector service, which does not exist yet, so one is refused rather than
    logged as unavailable."""
    if section.mode == "off":
        return None
    if section.mode != "log_only":
        raise ValueError(f"selector.mode {section.mode!r}: only 'off' and 'log_only' can come from a run config (D6)")
    if section.endpoint is not None and planner is None:
        raise ValueError("selector.endpoint is set, but no client for a selector endpoint exists yet (Q20)")
    return SelectorHook(mode="log_only", planner=planner, audit_seed=audit_seed, audit_fraction=section.audit_fraction,
                        plan_timeout_s=section.plan_timeout_s, max_miss_rate=section.max_miss_rate,
                        scoring=section.scoring, threads=threads)
