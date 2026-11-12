"""The reward service (design Section 6.2, C2).

Turns each final candidate into a reward, or None, as fast as possible, without ever scoring an
incomplete or untrusted check as clean.

    service = RewardService(reward_section, profile, adapter_factories, ...)
    fut = service.submit(cand)          # streams: call it the moment a chain ends
    report = service.drain(deadline)    # generation has ended; at the deadline every leg still
                                        # pending becomes unsettled(timeout)
    fut.result() -> RewardResult

One coordinator thread owns all state and runs the candidates' stage machines; workers only run
adapters and post results back, and the watchdog, ASan batch timers and drain deadline are checked
on the coordinator's tick. So every decision about a leg is made in one place, in order.

Per candidate: the stages of the profile run in order, each only while the reward function's
`decide` still answers NeedMore (early exit). Within a stage, legs run in parallel if the stage says
so (S2 by `speculative_parallel`), otherwise one at a time. A leg runs only after the legs it
requires are clean; one that never can is `not_run`. Every leg result is looked up in the cache first.

When the candidate is decided, the record the reward function sees is complete: every leg it names
is present, with `not_run` for legs that did not run. The reward is recomputed from that record with
`compute` and must agree with the decision; it must be None or a float in [-1.55, 1.0]; a clean
reward needs every required leg clean with its repeats; anything else is unscored with reason
`reward_invariant_violated` and raises an alarm.

With a `selector` (Section 6.3, `aorlta.selector.log_only.SelectorHook`), each candidate's audit
coin is drawn at submit and its check plan is requested when it reaches its first planned stage. In
log_only nothing waits for the plan and every leg runs as without one; a decided candidate's result
waits for a plan still on time, so the plan is logged beside the full results. Plan mode, which the
hook builds only behind its own flag, makes the stage wait and marks the skipped legs
`skipped_by_plan`.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import logging
import queue
import threading
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_bytes, atomic_write_json
from ..config import RewardSection
from ..ids import sha256_json, sha256_text
from ..schemas import SETTLED_STATES, Candidate, LegJob, LegResult, RewardResult, to_dict
from ..selector.log_only import SelectorHook
from ..spans import SpanRecorder
from . import batching, controls, watchdog
from .cache import RewardCache, leg_key_fields, reward_key_fields
from .clock import Clock
from .profiles import Profile, StageSpec
from .rewardfn.base import Decided, NeedMore, RewardFn, Unscored
from .workers import Item, Worker

log = logging.getLogger(__name__)

REWARD_MIN, REWARD_MAX = -1.55, 1.0
# Unsettled reasons that say nothing about the candidate and earn one retry on another worker.
INFRA_RETRY_REASONS = frozenset({"adapter_error", "worker_stalled", "worker_died", "control_dirty", "control_lost",
                                 "bracket_lost", "asan_unavailable"})
# Artifacts a leg leaves for the legs that require it: leg -> (input name, file name).
ARTIFACTS = {"gate.parse": ("source", "source.hip"), "gate.compile": ("object", "object.o")}


@dataclass(frozen=True)
class ServiceOptions:
    heartbeat_s: float = 10.0  # worker heartbeat period (Section 6.2, Warm workers)
    no_progress_s: float = 600.0  # alarm when no leg completes for this long with work pending
    tick_s: float = 1.0  # coordinator tick: watchdog, batch timers, deadlines
    max_dirty_streak: int = 3  # dirty controls in a row before a quarantined worker is restarted
    workers_default: int = 4  # workers in a pool that names none
    batch_runners_cap: int = 64  # runner threads for an unbounded (max_inflight None) ASan pool


@dataclass
class DrainReport:
    resolved: int
    scored: int
    unscored: int
    unscored_by_reason: dict[str, int]
    outstanding_at_deadline: list[dict]  # [{candidate_id, leg, where}] legs made unsettled(timeout)
    hit_deadline: bool
    t_drain: float
    t_done: float


@dataclass
class ServiceStats:
    submitted: int
    resolved: int
    scored: int
    unscored_by_reason: dict[str, int]
    leg_runs: dict[str, dict[str, int]]  # leg -> state -> runs
    retries: dict[str, int]  # reason -> retries scheduled
    worker_restarts: dict[str, int]  # reason -> restarts
    controls: dict
    asan_requests: dict
    cache: dict
    alarms: int
    resubmitted: int = 0  # D7 retries (`resubmit`); each is also counted in `submitted`


# --- per candidate state --------------------------------------------------------------------------


@dataclass
class _Slot:
    cand: _Cand
    leg: str
    pool: str
    key: str
    key_fields: dict
    attempt: int = 0
    status: str = "new"  # new | queued | running | provisional | done
    avoid: set = field(default_factory=set)
    retries_left: int = 0
    t_queued: float = 0.0  # clock seconds
    t_ready: float = 0.0  # when it joined the ASan queue
    job: LegJob | None = None
    pending_result: LegResult | None = None  # a provisional result awaiting its bracket
    worker_id: str | None = None
    orphan: bool = False  # the candidate is already decided; the result only feeds the cache


@dataclass
class _Cand:
    cand: Candidate
    future: concurrent.futures.Future
    t_submit: float
    response_sha256: str
    source_sha256: str
    pkg: Path | None
    results: dict[str, LegResult] = field(default_factory=dict)
    slots: dict[str, _Slot] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    stage_index: int = 0
    stage_times: dict = field(default_factory=dict)
    done: bool = False
    leg_keys: dict[str, str] = field(default_factory=dict)
    leg_fields: dict[str, dict] = field(default_factory=dict)
    sel: Any = None  # the selector's CandidatePlan (audit coin and check plan)
    held: RewardResult | None = None  # decided, waiting for a plan that is still on time
    prior_attempts: dict[str, int] = field(default_factory=dict)  # leg -> attempts before a resubmit


@dataclass
class _Pool:
    name: str
    spec: Any
    factory: Callable[[str], Any]
    timeout_s: float
    n_workers: int | None
    bracketed: bool
    batched: bool
    checker: tuple[str, str] = ("", "")
    control_specs: list = field(default_factory=list)
    workers: dict[str, Worker] = field(default_factory=dict)
    queue: deque = field(default_factory=deque)
    seq: int = 0
    inband: bool = False
    max_packages: int = 16


class RewardService:
    def __init__(
        self,
        reward: RewardSection,
        profile: Profile,
        rewardfn: RewardFn,
        adapter_factories: Mapping[str, Callable[[str], Any]],
        *,
        cache_dir: str | Path | None = None,
        work_dir: str | Path | None = None,
        clock: Clock | None = None,
        spans: SpanRecorder | None = None,
        options: ServiceOptions | None = None,
        selector: SelectorHook | None = None,
    ):
        if rewardfn.name != profile.reward_fn:
            raise ValueError(f"profile {profile.name!r} is scored by {profile.reward_fn!r}, not {rewardfn.name!r}")
        missing = sorted({p.adapter for p in profile.pools.values()} - set(adapter_factories))
        if missing:
            raise ValueError(f"no adapter factory for {missing}")
        if profile.materialize and work_dir is None:
            raise ValueError(f"profile {profile.name!r} writes packages and needs a work_dir")
        if reward.reuse_rollout_compile:
            raise ValueError("reward.reuse_rollout_compile: reusing the rollout's compile needs its settings proven "
                             "identical to the profile's, which nothing checks yet; leave it false")
        self.reward = reward
        self.profile = profile
        self.rewardfn = rewardfn
        self.clock = clock or Clock()
        self.spans = spans
        self.options = options or ServiceOptions()
        self.work_dir = None if work_dir is None else Path(work_dir)
        self.cache = RewardCache(cache_dir, enabled=reward.cache.enabled)
        self.ledger_controls = controls.ControlLedger()
        self._lock = threading.RLock()
        self._events: queue.Queue = queue.Queue()
        self._cands: dict[str, _Cand] = {}
        self._order: list[str] = []
        self._unresolved = 0
        self._draining = False
        self._drain_deadline: float | None = None
        self._drain_waiters: list[tuple[concurrent.futures.Future, float]] = []
        self._drain_outstanding: list[dict] = []
        self._closed = False
        self._alarms: list[dict] = []
        self._progress = watchdog.ProgressAlarm(self.options.no_progress_s)
        self._counts = {"leg_runs": Counter(), "retries": Counter(), "restarts": Counter(), "unscored": Counter(),
                        "asan_requests": 0, "asan_sizes": Counter(), "asan_max_wait_s": 0.0, "resolved": 0,
                        "scored": 0, "submitted": 0, "resubmitted": 0}
        self.events_log: list[dict] = []  # restarts, retries, requests: what the tests and reports read
        self._asan_ready: list[_Slot] = []
        self._inflight_batches = 0
        self._iteration = 0  # for spans not tied to one candidate (controls, ASan requests)
        self._controls_failing: dict[str, int] = {}
        timeouts = {"gate": reward.timeouts.gate, "compile": reward.timeouts.compile,
                    "sim_run": reward.timeouts.sim_run, "asan_request": reward.timeouts.asan_request}
        self._pools: dict[str, _Pool] = {}
        for name, spec in profile.pools.items():
            n = spec.workers
            if n is None and spec.resource == "sim_cpu":
                n = reward.sim_workers
            if n is None and not spec.batched:
                n = self.options.workers_default
            if spec.batched:
                n = reward.asan.max_inflight if reward.asan.max_inflight is not None else spec.workers
            elif n is not None:  # an adapter served from several places (reward.remote GPUs): n workers on each
                n *= int(getattr(adapter_factories[spec.adapter], "replicas", 1))
            pool = _Pool(name, spec, adapter_factories[spec.adapter], timeouts[spec.timeout], n, spec.controls,
                         spec.batched)
            probe = pool.factory(f"{name}/probe")
            pool.checker = (str(probe.name), str(probe.version))
            pool.control_specs = [c for c in probe.controls() if c.leg in self._pool_legs(name)] if spec.controls else []
            if spec.controls and not pool.control_specs:
                raise ValueError(f"pool {name!r} is bracketed but its adapter offers no control for its legs")
            if spec.batched:
                pool.inband = bool(getattr(probe, "inband_controls", True))
                pool.max_packages = min(int(getattr(probe, "max_packages", 16)), reward.asan.max_packages)
            close = getattr(probe, "close", None)
            if close is not None:
                close()
            self._pools[name] = pool
        inband = 2 * len(self._pools_batched_controls()) if self._pools_batched_inband() else 0
        self._batch_policy = batching.BatchPolicy(
            max_packages=min([p.max_packages for p in self._pools.values() if p.batched] or [reward.asan.max_packages]),
            max_wait_s=reward.asan.max_wait_s, max_inflight=reward.asan.max_inflight, inband_controls=inband)
        self.selector = selector
        if selector is not None:
            selector.attach(profile=profile, clock=self.clock, post=self._events.put)
        for pool in self._pools.values():
            for _ in range(pool.n_workers or (1 if pool.batched else 0)):
                self._spawn(pool)
        self._thread = threading.Thread(target=self._main, name="reward-coordinator", daemon=True)
        self._thread.start()

    # --- public API -------------------------------------------------------------------------------

    def submit(self, cand: Candidate) -> concurrent.futures.Future:
        with self._lock:
            existing = self._cands.get(getattr(cand, "candidate_id", None))
            if existing is not None:  # idempotent (resume)
                return existing.future
            fut: concurrent.futures.Future = concurrent.futures.Future()
            problem = self._candidate_problem(cand)
            if problem is None and (self._draining or self._closed):
                problem = "not_accepting"
            if problem is not None:
                if isinstance(getattr(cand, "candidate_id", None), str):
                    c = _Cand(cand, fut, self.clock.now(), "", "", None, done=True)
                    self._cands[cand.candidate_id] = c
                    self._order.append(cand.candidate_id)
                self._counts["submitted"] += 1
                fut.set_result(self._unscored_result(cand, problem))
                self._count_resolved(fut.result())
                return fut
            response_sha = sha256_text(cand.response_text)
            c = _Cand(cand, fut, self.clock.now(), response_sha, response_sha, None)
            if self.selector is not None:  # the audit coin is drawn now, before any plan is requested
                c.sel = self.selector.on_submit(cand, c.source_sha256)
            self._cands[cand.candidate_id] = c
            self._order.append(cand.candidate_id)
            self._counts["submitted"] += 1
            self._unresolved += 1
        self._events.put(("submit", cand.candidate_id))
        return fut

    def resubmit(self, cand: Candidate) -> concurrent.futures.Future:
        """D7's one retry (Section 6.6, step 7): check a resolved, unscored candidate again, on a fresh future. Its
        legs are looked up in the cache as at submit, so every leg that settled is reused and only the legs that
        ended unsettled (which are never cached) run again. A candidate still being checked, or already scored,
        keeps the future it has; one never submitted is submitted."""
        with self._lock:
            old = self._cands.get(getattr(cand, "candidate_id", None))
            if old is None:
                return self.submit(cand)
            if not old.future.done() or old.future.result().reward is not None:
                return old.future
            del self._cands[cand.candidate_id]
            self._order.remove(cand.candidate_id)
            self._counts["resubmitted"] += 1
            self.events_log.append({"event": "resubmit", "candidate_id": cand.candidate_id, "t": self.clock.now()})
            fut = self.submit(cand)
            new = self._cands.get(cand.candidate_id)
            if new is not None:  # a leg that runs again is the next attempt of the same check, as any retry is
                new.prior_attempts = {leg: max(s.attempt, old.prior_attempts.get(leg, 0))
                                      for leg, s in old.slots.items()}
            return fut

    def drain(self, deadline_monotonic: float) -> DrainReport:
        """Stop accepting candidates; flush the ASan queue; wait until every submitted candidate is
        resolved or the deadline passes (in this service's clock), when every leg still pending
        becomes unsettled(timeout)."""
        waiter: concurrent.futures.Future = concurrent.futures.Future()
        self._events.put(("drain", deadline_monotonic, waiter))
        # The coordinator resolves every candidate at the deadline; a minute of real time past it means
        # it could not, and every candidate still open is unscored rather than left waiting.
        limit = self.clock.real(max(0.0, deadline_monotonic - self.clock.now())) + 60.0
        try:
            return waiter.result(timeout=limit)
        except concurrent.futures.TimeoutError:
            with self._lock:
                self._alarm("drain_failed", {"unresolved": self._unresolved})
                self._force_resolve("drain_failed")
                self._check_drain_done()
            return waiter.result(timeout=10)

    def reopen(self) -> None:
        """Accept candidates again (the next iteration); warm workers and the cache carry over."""
        with self._lock:
            self._draining = False
            self._drain_deadline = None

    def stats(self) -> ServiceStats:
        with self._lock:
            runs: dict[str, dict[str, int]] = {}
            for (leg, state), n in self._counts["leg_runs"].items():
                runs.setdefault(leg, {})[state] = n
            return ServiceStats(
                submitted=self._counts["submitted"], resolved=self._counts["resolved"], scored=self._counts["scored"],
                unscored_by_reason=dict(self._counts["unscored"]), leg_runs=runs,
                retries=dict(self._counts["retries"]), worker_restarts=dict(self._counts["restarts"]),
                controls=self.ledger_controls.summary(),
                asan_requests={"count": self._counts["asan_requests"],
                               "sizes": dict(sorted(self._counts["asan_sizes"].items())),
                               "max_wait_s": self._counts["asan_max_wait_s"]},
                cache=self.cache.stats(), alarms=len(self._alarms), resubmitted=self._counts["resubmitted"])

    def health(self) -> dict:
        with self._lock:
            return {
                "profile": self.profile.name,
                "draining": self._draining,
                "closed": self._closed,
                "alarms": list(self._alarms),
                "pools": {name: {"workers": {w.worker_id: {"state": w.state, "quarantined": w.quarantined,
                                                           "silent_s": round(self.clock.now() - w.last_beat, 3)}
                                             for w in pool.workers.values()},
                                 "queued": len(pool.queue)} for name, pool in self._pools.items()},
                "asan_queued": len(self._asan_ready),
                "unresolved": self._unresolved,
                "selector": None if self.selector is None else self.selector.stats(),
            }

    def close(self) -> None:
        """Stop the coordinator, the workers and the cache's writer; any candidate still open is unscored
        (service_closed)."""
        with self._lock:
            self._closed = True
        self._events.put(("close",))
        self._thread.join(timeout=30)
        with self._lock:
            self._force_resolve("service_closed")
            for pool in self._pools.values():
                for w in list(pool.workers.values()):
                    w.abandon()
            if self.selector is not None:
                self.selector.close()
        self.cache.close()

    def _force_resolve(self, reason: str) -> None:
        for cid in self._order:
            c = self._cands[cid]
            if c.sel is not None and not c.future.done():
                self.selector.abandon(c.sel, reason)
                if c.held is not None:
                    self._publish(c, c.held)
            if not c.done:
                self._resolve(c, self._unscored_result(c.cand, reason))

    # --- coordinator ------------------------------------------------------------------------------

    def _main(self) -> None:
        while True:
            timeout = self._next_wait()
            try:
                ev = self._events.get(timeout=self.clock.real(timeout))
            except queue.Empty:
                ev = None
            with self._lock:
                try:
                    if ev is not None:
                        if ev[0] == "close":
                            return
                        self._handle(ev)
                        # take everything already waiting before ticking
                        while True:
                            try:
                                ev2 = self._events.get_nowait()
                            except queue.Empty:
                                break
                            if ev2[0] == "close":
                                return
                            self._handle(ev2)
                    self._tick()
                    self._dispatch()
                    self._check_drain_done()
                except Exception:
                    log.exception("reward coordinator: unexpected error")
                    self._alarm("coordinator_error", {})

    def _next_wait(self) -> float:
        # at least 2 ms of real time between idle ticks, however fast a test's clock runs
        wait = max(self.options.tick_s, 0.002 / self.clock.scale)
        now = self.clock.now()
        deadline = batching.next_deadline([s.t_ready for s in self._asan_ready], self._batch_policy)
        if deadline is not None:
            wait = min(wait, max(0.0, deadline - now))
        if self._drain_deadline is not None:
            wait = min(wait, max(0.0, self._drain_deadline - now))
        plan_deadline = None if self.selector is None else self.selector.next_deadline()
        if plan_deadline is not None:
            wait = min(wait, max(0.0, plan_deadline - now))
        return wait

    def _handle(self, ev: tuple) -> None:
        kind = ev[0]
        if kind == "submit":
            c = self._cands.get(ev[1])
            if c is not None and not c.done:
                self._start_candidate(c)
        elif kind == "item_done":
            _, worker_id, generation, item, results = ev
            self._item_done(worker_id, generation, item, results)
        elif kind == "drain":
            _, deadline, waiter = ev
            self._draining = True
            self._drain_deadline = deadline
            self._drain_outstanding = []
            self._drain_waiters.append((waiter, self.clock.now()))
        elif kind == "plan":
            _, token, outcome = ev
            cp = self.selector.on_answer(token, outcome)
            if cp is not None:
                self._plan_settled(cp)

    def _plan_settled(self, cp: Any) -> None:
        """A plan arrived, or became unavailable: release a result that was waiting for it, or (plan
        mode) let the candidate's stage go on."""
        c = self._cands.get(cp.candidate_id)
        if c is None:
            return
        if c.held is not None:
            self._publish(c, c.held)
        elif not c.done and self.selector.plan_mode:
            self._advance(c)

    # --- candidates -------------------------------------------------------------------------------

    def _candidate_problem(self, cand: Any) -> str | None:
        if not isinstance(cand, Candidate):
            return "bad_candidate"
        if cand.check_profile != self.profile.name:
            return "profile_mismatch"
        ok = (isinstance(cand.response_text, str) and isinstance(cand.max_new_tokens, int) and cand.max_new_tokens >= 1
              and isinstance(cand.response_tokens, int) and 0 <= cand.response_tokens <= cand.max_new_tokens
              and isinstance(cand.candidate_id, str) and cand.candidate_id)
        return None if ok else "bad_candidate"

    def _start_candidate(self, c: _Cand) -> None:
        cand = c.cand
        self._iteration = cand.iteration
        for leg in self.profile.legs:  # requirements come before their legs (profile_problems)
            pool = self._pools[self.profile.stage_of(leg).pool]
            fields = leg_key_fields(source_sha256=c.source_sha256, task_sha256=cand.task.task_sha256, leg=leg,
                                    leg_settings=dict(self.profile.leg_settings.get(leg, {})),
                                    checker_name=pool.checker[0], checker_version=pool.checker[1],
                                    requires={need: c.leg_keys[need] for need in self.profile.requires.get(leg, ())},
                                    bracketed=pool.bracketed)
            c.leg_keys[leg] = sha256_json(fields)
            c.leg_fields[leg] = fields
        rkey, rfields = self._reward_key(c)
        hit = self.cache.get_reward(rkey, rfields)
        if hit is not None:
            legs = {k: dataclasses.replace(v, cache_hit=True) for k, v in hit.legs.items()}
            result = dataclasses.replace(hit, candidate_id=cand.candidate_id, legs=legs,
                                         timings={"total_s": 0.0, "stages": {}, "reward_cache_hit": True})
            self._resolve(c, result)
            return
        if self.profile.materialize:
            c.pkg = self.work_dir / "pkg" / c.response_sha256[:2] / c.response_sha256
            c.pkg.mkdir(parents=True, exist_ok=True)
            response = c.pkg / "response.txt"
            if not response.exists():
                atomic_write_bytes(response, cand.response_text.encode("utf-8"))
        self._advance(c)

    def _reward_key(self, c: _Cand) -> tuple[str, dict]:
        rf = self._rewardfn_identity()
        fields = reward_key_fields(c.cand, c.leg_keys, rf, self.profile.name,
                                   reward_settings=dict(getattr(self.rewardfn, "settings", {})),
                                   clean_repeats=self.profile.clean_repeats,
                                   bracketed=[leg for leg, f in c.leg_fields.items() if f.get("bracketed") is True])
        return sha256_json(fields), fields

    def _rewardfn_identity(self) -> dict:
        return {"name": self.rewardfn.name, "version": self.rewardfn.version, "code_sha256": self.rewardfn.code_sha256}

    def _requirements(self, c: _Cand, leg: str) -> str:
        """'ok' when every leg it requires is clean, 'never' when one cannot be (it settled otherwise,
        or can itself never run), else 'wait'."""
        status = "ok"
        for need in self.profile.requires.get(leg, ()):
            r = c.results.get(need)
            if r is not None:
                if r.state != "clean":
                    return "never"
                continue
            if need in c.slots and c.slots[need].status != "done":
                status = "wait"
                continue
            sub = self._requirements(c, need)
            if sub == "never":
                return "never"
            status = "wait"
        return status

    def _partial(self, c: _Cand) -> dict[str, LegResult]:
        out: dict[str, LegResult] = {}
        for leg in self.rewardfn.legs:
            if leg in c.results:
                out[leg] = c.results[leg]
            elif leg not in self.profile.legs:
                out[leg] = self._not_run(leg, "not_in_profile")
            elif leg not in c.slots and self._requirements(c, leg) == "never":
                out[leg] = self._not_run(leg, "requirement_not_clean")
        return out

    def _not_run(self, leg: str, reason: str) -> LegResult:
        t = self.clock.wall()
        return LegResult(leg, "not_run", reason, [], 0, 0, None, None, False, t, None, None)

    def _parallel(self, stage: StageSpec) -> bool:
        if stage.parallel == "speculative":
            return bool(self.reward.speculative_parallel)
        return bool(stage.parallel)

    def _advance(self, c: _Cand) -> None:
        """Decide; while the answer is NeedMore, start what the current stage can start, move past
        stages with nothing left, and stop when legs are in flight. Never re-entered: results that
        arrive later call this again."""
        while not c.done:
            try:
                decision = self.rewardfn.decide(self._partial(c), c.cand)
            except Exception as exc:
                log.warning("reward fn decide raised %s for %s", type(exc).__name__, c.cand.candidate_id)
                self._finalize(c, Unscored("reward_fn_error"), error=True)
                return
            if not isinstance(decision, NeedMore):
                self._finalize(c, decision)
                return
            while True:
                if c.stage_index >= len(self.profile.stages):
                    # Every profile leg has a result or can never run, yet the record is undecided.
                    self._finalize(c, Unscored("reward_invariant_violated"), error=True)
                    return
                step = self._stage_step(c, self.profile.stages[c.stage_index])
                if step == "waiting":
                    return
                if step == "progressed":
                    break  # a cached result arrived: decide again
                c.stage_times[self.profile.stages[c.stage_index].name]["end"] = self.clock.now()
                c.stage_index += 1

    def _stage_step(self, c: _Cand, stage: StageSpec) -> str:
        """'waiting' (legs in flight), 'progressed' (a leg settled from the cache) or 'complete'."""
        c.stage_times.setdefault(stage.name, {"start": self.clock.now(), "end": None})
        if self.selector is not None and c.sel is not None and stage.name == self.selector.plan_stage:
            step = self._plan_step(c)
            if step is not None:
                return step
        left = [leg for leg in stage.legs if leg not in c.results]
        in_flight = [leg for leg in left if leg in c.slots and c.slots[leg].status != "done"]
        startable = [leg for leg in left if leg not in c.slots and self._requirements(c, leg) == "ok"]
        progressed = False
        parallel = self._parallel(stage)
        for leg in startable:
            if not parallel and in_flight:
                break
            hit = self._launch(c, leg)
            if hit is not None:
                c.results[leg] = hit
                progressed = True
                if not parallel:
                    break
            else:
                in_flight.append(leg)
        if progressed:
            return "progressed"
        if in_flight:
            return "waiting"
        waiting = [leg for leg in left if leg not in c.slots and self._requirements(c, leg) == "wait"]
        if waiting:
            return "waiting" if any(c.slots.get(n) and c.slots[n].status != "done"
                                    for leg in waiting for n in self.profile.requires.get(leg, ())) else "complete"
        return "complete"

    def _plan_step(self, c: _Cand) -> str | None:
        """The candidate reached its first planned stage (after S1): ask for its plan. log_only goes on
        at once. Plan mode waits for the plan (or its deadline), then marks the legs it skips."""
        sel = self.selector
        if c.sel.status == "not_requested":
            sel.request(c.sel, c.cand, self._kernel_features(c))
        if not sel.plan_mode or c.sel.applied:
            return None
        if sel.pending(c.sel):
            return "waiting"
        skipped = sel.skips(c.sel)
        for leg in skipped:
            c.results[leg] = self._skipped(leg)
        return "progressed" if skipped else None

    def _kernel_features(self, c: _Cand) -> dict:
        """What the planner may read (inside the trusted boundary): the candidate's answer and, when the
        profile writes packages, its parsed source file; the gate and compile results so far; the task.
        Never logged."""
        cand = c.cand
        return {"response_text": cand.response_text, "source_path": c.artifacts.get("source"),
                "legs": {leg: {"state": r.state, "reason": r.reason, "findings": len(r.findings)}
                         for leg, r in c.results.items()},
                "task": to_dict(cand.task), "response_tokens": cand.response_tokens,
                "max_new_tokens": cand.max_new_tokens}

    def _skipped(self, leg: str) -> LegResult:
        t = self.clock.wall()
        return LegResult(leg, "skipped_by_plan", None, [], 0, 0, None, None, False, t, None, None)

    def _launch(self, c: _Cand, leg: str) -> LegResult | None:
        """Start a leg: a cached result is returned at once; otherwise the leg is queued."""
        stage = self.profile.stage_of(leg)
        pool = self._pools[stage.pool]
        key, fields = c.leg_keys[leg], c.leg_fields[leg]
        slot = _Slot(c, leg, pool.name, key, fields, retries_left=self.reward.retries_infra,
                     attempt=c.prior_attempts.get(leg, 0))
        c.slots[leg] = slot
        hit = self.cache.get_leg(key, fields)
        if hit is not None and leg in ARTIFACTS:
            out = self._output_dir(c, leg, key)
            _, fname = ARTIFACTS[leg]
            if hit.state == "clean" and out is not None and not (out / fname).is_file():
                hit = None  # the artifact this result vouches for is gone: run it again
        if hit is not None:
            t = self.clock.wall()
            result = dataclasses.replace(hit, cache_hit=True, attempts=0, worker_id=None, t_queued=t, t_start=None,
                                         t_end=t)
            slot.status = "done"
            self._record_artifact(c, leg, key, result)
            return result
        self._enqueue(slot)
        return None

    def _output_dir(self, c: _Cand, leg: str, key: str) -> Path | None:
        return None if c.pkg is None else c.pkg / f"{leg}-{key[:16]}"

    def _record_artifact(self, c: _Cand, leg: str, key: str, result: LegResult) -> None:
        if leg in ARTIFACTS and result.state == "clean" and c.pkg is not None:
            name, fname = ARTIFACTS[leg]
            c.artifacts[name] = str(self._output_dir(c, leg, key) / fname)

    def _enqueue(self, slot: _Slot) -> None:
        slot.attempt += 1
        slot.status = "queued"
        slot.t_queued = self.clock.now()
        pool = self._pools[slot.pool]
        if pool.batched:
            slot.t_ready = slot.t_queued
            self._asan_ready.append(slot)
        else:
            pool.queue.append(slot)

    def _make_job(self, slot: _Slot, worker_id: str) -> LegJob:
        c = slot.cand
        out = self._output_dir(c, slot.leg, slot.key)
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
        inputs = dict(c.artifacts)
        if c.pkg is not None:
            inputs["response"] = str(c.pkg / "response.txt")
        return LegJob(job_id=f"{c.cand.candidate_id}#{slot.leg}#a{slot.attempt}", leg=slot.leg,
                      candidate_id=c.cand.candidate_id, control_id=None, source_sha256=c.source_sha256,
                      task_sha256=c.cand.task.task_sha256, attempt=slot.attempt, worker_id=worker_id,
                      leg_settings=dict(self.profile.leg_settings.get(slot.leg, {})),
                      t_queued=self.clock.to_wall(slot.t_queued), package=None if c.pkg is None else str(c.pkg),
                      inputs=inputs, output_dir=None if out is None else str(out))

    def _deliver(self, c: _Cand, leg: str, result: LegResult) -> None:
        """A leg's final result for this candidate (settled with its bracket, or unsettled for good)."""
        if c.done:
            return
        c.results[leg] = result
        self._advance(c)

    # --- leg results ------------------------------------------------------------------------------

    def _item_done(self, worker_id: str, generation: int, item: Item, results: list[LegResult] | None) -> None:
        pool, worker = self._find_worker(worker_id)
        if worker is None or worker.generation != generation or worker.state == "dead":
            return  # an abandoned worker's late answer: ignored
        self._progress.progress(self.clock.now())
        worker.state = "idle"
        worker.item = None
        if results is None or len(results) != len(item.jobs):
            results = [LegResult(j.leg, "unsettled", "adapter_error", [], 0, j.attempt, worker_id, None, False,
                                 j.t_queued or self.clock.wall(), self.clock.wall(), self.clock.wall()) for j in item.jobs]
        if item.kind == "job":
            slot = item.payload
            worker.jobs_since_control += 1
            worker.jobs_run += 1
            self._job_result(pool, worker, slot, results[0])
            if pool.bracketed and controls.due(jobs_since=worker.jobs_since_control, t_last=worker.t_last_control,
                                               now=self.clock.now(), every_jobs=self.reward.controls.every_jobs,
                                               every_s=self.reward.controls.every_s):
                worker.needs_control = True
        elif item.kind == "control":
            self._control_result(pool, worker, item, results)
        elif item.kind == "batch":
            self._batch_result(pool, worker, item, results)

    def _find_worker(self, worker_id: str) -> tuple[_Pool | None, Worker | None]:
        for pool in self._pools.values():
            w = pool.workers.get(worker_id)
            if w is not None:
                return pool, w
        return None, None

    def _job_result(self, pool: _Pool, worker: Worker, slot: _Slot, result: LegResult) -> None:
        result = dataclasses.replace(result, worker_id=worker.worker_id, attempts=slot.attempt)
        self._count_run(slot.leg, result)
        self._span(slot, result)
        if result.state not in SETTLED_STATES:
            self._unsettled(slot, result, worker.worker_id)
            return
        if pool.bracketed:
            slot.status = "provisional"
            slot.pending_result = result
            slot.worker_id = worker.worker_id
            worker.provisional.append(slot)
            return
        self._settle(slot, result, bracketed=False)

    def _settle(self, slot: _Slot, result: LegResult, *, bracketed: bool) -> None:
        slot.status = "done"
        c = slot.cand
        artifacts = {}
        if slot.leg in ARTIFACTS and result.state == "clean" and c.pkg is not None:
            name, fname = ARTIFACTS[slot.leg]
            artifacts[name] = fname
        self.cache.put_leg(slot.key, slot.key_fields, result, bracketed=bracketed, artifacts=artifacts)
        if slot.orphan:
            return
        self._record_artifact(c, slot.leg, slot.key, result)
        self._deliver(c, slot.leg, result)

    def _unsettled(self, slot: _Slot, result: LegResult, worker_id: str | None) -> None:
        reason = result.reason or "unsettled"
        if worker_id is not None:
            slot.avoid.add(worker_id)
        if (not slot.orphan and not slot.cand.done and reason in INFRA_RETRY_REASONS and slot.retries_left > 0
                and not self._past_drain_deadline()):
            slot.retries_left -= 1
            self._counts["retries"][reason] += 1
            self.events_log.append({"event": "retry", "candidate_id": slot.cand.cand.candidate_id, "leg": slot.leg,
                                    "reason": reason, "from_worker": worker_id, "attempt": slot.attempt + 1,
                                    "t": self.clock.now()})
            self._enqueue(slot)
            return
        slot.status = "done"
        # the cache's own check refuses it; every final result goes through the one admission rule
        self.cache.put_leg(slot.key, slot.key_fields, result, bracketed=self._pools[slot.pool].bracketed)
        if not slot.orphan:
            self._deliver(slot.cand, slot.leg, result)

    def _past_drain_deadline(self) -> bool:
        return self._drain_deadline is not None and self.clock.now() >= self._drain_deadline

    # --- controls ---------------------------------------------------------------------------------

    def _start_control(self, pool: _Pool, worker: Worker) -> None:
        worker.control_seq += 1
        run_id = f"control:{worker.worker_id}:{worker.control_seq}"
        jobs = []
        for spec in pool.control_specs:
            out = None
            if self.work_dir is not None and spec.inputs:
                out = self.work_dir / "controls" / run_id.replace(":", "_").replace("/", "_") / spec.leg
                out.mkdir(parents=True, exist_ok=True)
            jobs.append(LegJob(job_id=f"{run_id}:{spec.leg}", leg=spec.leg, candidate_id=None,
                               control_id=spec.control_id, source_sha256=spec.source_sha256,
                               task_sha256=spec.task_sha256, attempt=1, worker_id=worker.worker_id,
                               leg_settings=dict(self.profile.leg_settings.get(spec.leg, {})), t_queued=self.clock.wall(),
                               inputs=dict(spec.inputs), output_dir=None if out is None else str(out)))
        worker.needs_control = False
        worker.give(Item("control", run_id, jobs, pool.timeout_s * max(1, len(jobs))))

    def _control_result(self, pool: _Pool, worker: Worker, item: Item, results: list[LegResult]) -> None:
        run_id = item.payload
        clean = controls.control_clean(results)
        provisional, worker.provisional = worker.provisional, []
        before = worker.last_clean_control
        self.ledger_controls.add(controls.ControlRecord(
            run_id, worker.worker_id, pool.name, self.clock.to_wall(item.t_start), self.clock.wall(), clean,
            {r.leg: r.state for r in results}, len(provisional)))
        if self.spans is not None:
            self._record_span("control", self.clock.to_wall(item.t_start), self.clock.wall(),
                              worker=worker.worker_id, clean=clean, closes=len(provisional))
        br = controls.bracket(before, run_id, clean)
        if clean:
            self._controls_failing.pop(pool.name, None)
            worker.last_clean_control = run_id
            worker.quarantined = False
            worker.dirty_streak = 0
            worker.jobs_since_control = 0
            worker.t_last_control = self.clock.now()
            for slot in provisional:
                if slot.status != "provisional":
                    continue
                result = dataclasses.replace(slot.pending_result, bracket=br)
                self._settle(slot, result, bracketed=True)
            return
        worker.quarantined = True
        worker.needs_control = True
        worker.dirty_streak += 1
        self._control_failed(pool, run_id)
        worker.jobs_since_control = 0
        worker.t_last_control = self.clock.now()
        self.events_log.append({"event": "control_dirty", "worker": worker.worker_id, "control": run_id,
                                "closes": [s.cand.cand.candidate_id + "#" + s.leg for s in provisional],
                                "t": self.clock.now()})
        for slot in provisional:
            if slot.status != "provisional":
                continue
            self._unbracket(slot, "control_dirty", br, worker.worker_id)
        if worker.dirty_streak >= self.options.max_dirty_streak:
            self._restart(pool, worker, "control_dirty_streak")

    def _control_failed(self, pool: _Pool, run_id: str) -> None:
        """Count failed controls in a row per pool; at `max_dirty_streak` raise one alarm: the checker
        itself is suspect, and nothing on that pool can settle until a control is clean again."""
        n = self._controls_failing.get(pool.name, 0) + 1
        self._controls_failing[pool.name] = n
        if n == self.options.max_dirty_streak:
            self._alarm("controls_failing", {"pool": pool.name, "in_a_row": n, "last": run_id})

    def _unbracket(self, slot: _Slot, reason: str, br: dict | None, worker_id: str | None) -> None:
        r = slot.pending_result
        slot.pending_result = None
        t = self.clock.wall()
        result = LegResult(slot.leg, "unsettled", reason, [], 0, slot.attempt, worker_id, br, False,
                           r.t_queued if r else t, r.t_start if r else None, r.t_end if r else t)
        self._unsettled(slot, result, worker_id)

    # --- the ASan endpoint ------------------------------------------------------------------------

    def _pools_batched_inband(self) -> bool:
        return any(p.batched and p.bracketed and p.inband for p in self._pools.values())

    def _pools_batched_controls(self) -> list:
        for p in self._pools.values():
            if p.batched and p.bracketed:
                return p.control_specs
        return []

    def _dispatch_batches(self, pool: _Pool) -> None:
        while self._asan_ready:
            n = batching.take_batch([s.t_ready for s in self._asan_ready], now=self.clock.now(),
                                    policy=self._batch_policy, generation_ended=self._draining,
                                    inflight=self._inflight_batches)
            if n == 0:
                return
            worker = next((w for w in sorted(pool.workers.values(), key=lambda w: w.worker_id)
                           if w.state in ("idle", "fresh")), None)
            if worker is None:
                limit = self.reward.asan.max_inflight
                if limit is None and len(pool.workers) < self.options.batch_runners_cap:
                    worker = self._spawn(pool)
                else:
                    return
            slots, self._asan_ready = self._asan_ready[:n], self._asan_ready[n:]
            self._send_batch(pool, worker, slots)

    def _send_batch(self, pool: _Pool, worker: Worker, slots: list[_Slot]) -> None:
        now = self.clock.now()
        jobs, ctl_before, ctl_after = [], [], []
        worker.control_seq += 1
        req_id = f"asan:{worker.worker_id}:{worker.control_seq}"
        if pool.bracketed:
            for edge, bucket in (("before", ctl_before), ("after", ctl_after)):
                for spec in pool.control_specs:
                    out = None
                    if self.work_dir is not None and spec.inputs:
                        out = self.work_dir / "controls" / req_id.replace(":", "_").replace("/", "_") / edge
                        out.mkdir(parents=True, exist_ok=True)
                    bucket.append(LegJob(job_id=f"{req_id}:control-{edge}:{spec.leg}", leg=spec.leg, candidate_id=None,
                                         control_id=spec.control_id, source_sha256=spec.source_sha256,
                                         task_sha256=spec.task_sha256, attempt=1, worker_id=worker.worker_id,
                                         t_queued=self.clock.wall(), inputs=dict(spec.inputs),
                                         output_dir=None if out is None else str(out)))
        for slot in slots:
            slot.status = "running"
            slot.worker_id = worker.worker_id
            slot.job = self._make_job(slot, worker.worker_id)
            wait = now - slot.t_ready
            self._counts["asan_max_wait_s"] = max(self._counts["asan_max_wait_s"], wait)
            if self.spans is not None:
                self._record_span("asan.queue", self.clock.to_wall(slot.t_ready), self.clock.to_wall(now),
                                  candidate_id=slot.cand.cand.candidate_id, request=req_id)
        jobs = ctl_before + [s.job for s in slots] + ctl_after
        separate = pool.bracketed and not pool.inband  # controls as their own requests before and after
        sizes = [len(ctl_before), len(slots), len(ctl_after)] if separate else [len(jobs)]
        self._counts["asan_requests"] += 1
        for size in sizes:
            self._counts["asan_sizes"][size] += 1
        self._inflight_batches += 1
        self.events_log.append({"event": "asan_request", "request": req_id, "worker": worker.worker_id,
                                "packages": max(sizes), "request_sizes": sizes,
                                "candidates": [s.cand.cand.candidate_id for s in slots],
                                "waits": [now - s.t_ready for s in slots], "t": now,
                                "generation_ended": self._draining})
        payload = {"request": req_id, "slots": slots, "n_before": len(ctl_before), "n_after": len(ctl_after),
                   "separate": separate}
        worker.give(Item("batch", payload, jobs, pool.timeout_s * (3 if separate else 1)))

    def _batch_result(self, pool: _Pool, worker: Worker, item: Item, results: list[LegResult]) -> None:
        self._inflight_batches -= 1
        p = item.payload
        nb, na = p["n_before"], p["n_after"]
        before, pkgs, after = results[:nb], results[nb:len(results) - na], results[len(results) - na:]
        clean = (not pool.bracketed) or (controls.control_clean(before) and controls.control_clean(after))
        if self.spans is not None:
            self._record_span("asan.run", self.clock.to_wall(item.t_start), self.clock.wall(), request=p["request"],
                              packages=len(item.jobs), clean_bracket=clean)
        if pool.bracketed:
            for edge, rs in (("before", before), ("after", after)):
                self.ledger_controls.add(controls.ControlRecord(
                    f"{p['request']}:control-{edge}", worker.worker_id, pool.name, self.clock.to_wall(item.t_start),
                    self.clock.wall(), controls.control_clean(rs), {r.leg: r.state for r in rs}, len(pkgs)))
        br = controls.bracket(f"{p['request']}:control-before", f"{p['request']}:control-after", clean) \
            if pool.bracketed else None
        for slot, res in zip(p["slots"], pkgs):
            res = dataclasses.replace(res, worker_id=worker.worker_id, attempts=slot.attempt)
            self._count_run(slot.leg, res)
            self._span(slot, res)
            if res.state not in SETTLED_STATES:
                self._unsettled(slot, res, worker.worker_id)
            elif not clean:
                slot.pending_result = res
                self._unbracket(slot, "control_dirty", br, worker.worker_id)
            else:
                self._settle(slot, dataclasses.replace(res, bracket=br), bracketed=pool.bracketed)

    # --- workers and the watchdog -----------------------------------------------------------------

    def _spawn(self, pool: _Pool) -> Worker:
        pool.seq += 1
        wid = f"{pool.name}/{pool.seq}"
        adapter = pool.factory(wid)
        w = Worker(wid, pool.name, adapter, self.clock, self._events.put, heartbeat_s=self.options.heartbeat_s,
                   batched=pool.batched)
        w.needs_control = pool.bracketed and not pool.batched
        pool.workers[wid] = w
        w.start()
        if not pool.bracketed or pool.batched:
            w.state = "idle"
        return w

    def _restart(self, pool: _Pool, worker: Worker, reason: str) -> None:
        item = worker.item
        worker.abandon()
        pool.workers.pop(worker.worker_id, None)
        self._counts["restarts"][reason] += 1
        replacement = self._spawn(pool)
        self.events_log.append({"event": "restart", "worker": worker.worker_id, "replacement": replacement.worker_id,
                                "reason": reason, "t": self.clock.now(),
                                "item_started": item.t_start if item is not None else None})
        unsettle = "timeout" if reason == "timeout" else ("worker_stalled" if reason == "worker_stalled" else "worker_died")
        if item is not None:
            if item.kind == "job":
                slot = item.payload
                t = self.clock.wall()
                res = LegResult(slot.leg, "unsettled", unsettle, [], 0, slot.attempt, worker.worker_id, None, False,
                                self.clock.to_wall(slot.t_queued), self.clock.to_wall(item.t_start), t)
                self._count_run(slot.leg, res)
                self._span(slot, res)
                self._unsettled(slot, res, worker.worker_id)
            elif item.kind == "batch":
                self._inflight_batches -= 1
                for slot in item.payload["slots"]:
                    t = self.clock.wall()
                    res = LegResult(slot.leg, "unsettled", unsettle, [], 0, slot.attempt, worker.worker_id, None, False,
                                    self.clock.to_wall(slot.t_queued), self.clock.to_wall(item.t_start), t)
                    self._count_run(slot.leg, res)
                    self._span(slot, res)
                    self._unsettled(slot, res, worker.worker_id)
            elif item.kind == "control":
                self.ledger_controls.add(controls.ControlRecord(
                    item.payload, worker.worker_id, pool.name, self.clock.to_wall(item.t_start), self.clock.wall(),
                    False, {}, len(worker.provisional)))
                self._control_failed(pool, item.payload)
                for slot in list(worker.provisional):
                    if slot.status == "provisional":
                        self._unbracket(slot, "control_lost", None, worker.worker_id)
                worker.provisional = []
        for slot in list(worker.provisional):
            if slot.status == "provisional":
                self._unbracket(slot, "bracket_lost", None, worker.worker_id)
        worker.provisional = []

    def _tick(self) -> None:
        now = self.clock.now()
        for pool in self._pools.values():
            views = [w.view() for w in pool.workers.values() if w.state != "dead"]
            for verdict in watchdog.check_workers(views, now=now, stall_s=self.reward.stall_s,
                                                  silent_after_s=max(3 * self._beat_period(), 0.1 / self.clock.scale)):
                worker = pool.workers.get(verdict.worker_id)
                if worker is not None:
                    log.warning("watchdog: worker %s %s (running %.1fs, silent %.1fs)", worker.worker_id,
                                verdict.reason, verdict.running_s, verdict.silent_s)
                    self._restart(pool, worker, verdict.reason)
        pending = self._unresolved > 0
        if self._progress.check(now, pending):
            self._alarm("no_progress", {"no_progress_s": self.options.no_progress_s})
        if self.selector is not None:
            for cp in self.selector.expire(now):  # plan_timeout_s passed: plan_unavailable(timeout)
                self._plan_settled(cp)
        if self._drain_deadline is not None and now >= self._drain_deadline:
            self._expire_pending()

    def _beat_period(self) -> float:
        """The heartbeat period in clock seconds (workers never beat faster than every 20 ms real)."""
        return max(self.options.heartbeat_s, 0.02 / self.clock.scale)

    def _expire_pending(self) -> None:
        """The drain deadline has passed: every leg still pending becomes unsettled(timeout), and a plan
        still pending becomes unavailable."""
        for cid in self._order:
            c = self._cands[cid]
            if c.sel is not None and not c.future.done():
                self.selector.abandon(c.sel, "drain_deadline")
                if c.held is not None:
                    self._publish(c, c.held)
            if c.done:
                continue
            for leg, slot in list(c.slots.items()):
                if slot.status in ("queued", "running", "provisional", "new"):
                    where = slot.status
                    self._drain_outstanding.append({"candidate_id": cid, "leg": leg, "where": where,
                                                    "worker": slot.worker_id})
                    self._remove_from_queues(slot)
                    slot.orphan = slot.status in ("running", "provisional")
                    slot.status = "done" if not slot.orphan else slot.status
                    t = self.clock.wall()
                    res = LegResult(leg, "unsettled", "timeout", [], 0, slot.attempt, slot.worker_id, None, False,
                                    self.clock.to_wall(slot.t_queued), None, t)
                    c.results[leg] = res
            if not c.done:
                # legs never started can no longer run: decide on what is there
                self._advance_after_deadline(c)

    def _advance_after_deadline(self, c: _Cand) -> None:
        for leg in self.profile.legs:
            if leg not in c.results and leg not in c.slots:
                c.results[leg] = self._not_run(leg, "drain_deadline")
        try:
            decision = self.rewardfn.decide(self._partial(c), c.cand)
        except Exception:
            decision = Unscored("reward_fn_error")
        if isinstance(decision, NeedMore):
            decision = Unscored("reward_invariant_violated")
        self._finalize(c, decision)

    def _remove_from_queues(self, slot: _Slot) -> None:
        pool = self._pools[slot.pool]
        try:
            pool.queue.remove(slot)
        except ValueError:
            pass
        if slot in self._asan_ready:
            self._asan_ready.remove(slot)

    def _dispatch(self) -> None:
        for pool in self._pools.values():
            if pool.batched:
                self._dispatch_batches(pool)
                continue
            starved = False
            for worker in sorted(pool.workers.values(), key=lambda w: w.worker_id):
                if worker.state not in ("idle", "fresh"):
                    continue
                if pool.bracketed and (worker.needs_control or worker.quarantined):
                    self._start_control(pool, worker)
                    continue
                if not pool.queue:
                    if pool.bracketed and any(s.status == "provisional" for s in worker.provisional):
                        self._start_control(pool, worker)  # idle: close the bracket now
                    continue
                head = pool.queue[0]
                if worker.worker_id not in head.avoid:
                    slot = pool.queue.popleft()
                else:
                    slot = next((s for s in pool.queue if worker.worker_id not in s.avoid), None)
                    if slot is None:
                        starved = True
                        if pool.bracketed and any(s.status == "provisional" for s in worker.provisional):
                            self._start_control(pool, worker)
                        continue
                    pool.queue.remove(slot)
                slot.status = "running"
                slot.worker_id = worker.worker_id
                slot.job = self._make_job(slot, worker.worker_id)
                worker.give(Item("job", slot, [slot.job], pool.timeout_s))
            # a slot every live worker must avoid gets a fresh worker
            if starved:
                live = [w for w in pool.workers.values() if w.state != "dead"]
                blocked = [s for s in pool.queue if all(w.worker_id in s.avoid for w in live)]
                idle = [w for w in live if w.state == "idle" and not w.provisional]
                if blocked and idle:
                    self._restart(pool, idle[0], "retry_needs_another_worker")

    # --- finalisation -----------------------------------------------------------------------------

    def _final_record(self, c: _Cand) -> dict[str, LegResult]:
        out = {}
        for leg in self.rewardfn.legs:
            if leg in c.results:
                out[leg] = c.results[leg]
            elif leg not in self.profile.legs:
                out[leg] = self._not_run(leg, "not_in_profile")
            elif leg in c.slots and c.slots[leg].status != "done":
                out[leg] = self._not_run(leg, "decided_before_result")
            elif self._requirements(c, leg) == "never":
                out[leg] = self._not_run(leg, "requirement_not_clean")
            else:
                out[leg] = self._not_run(leg, "decided_before_run")
        return out

    def _finalize(self, c: _Cand, decision: Any, *, error: bool = False) -> None:
        if c.done:
            return
        record = self._final_record(c)
        reason = None
        reward = None
        parts = None
        violations: list[str] = []
        if error:
            reason = decision.reason if isinstance(decision, Unscored) else "reward_fn_error"
        else:
            try:
                reward, raw = self.rewardfn.compute(record, c.cand)
            except Exception as exc:
                log.warning("reward fn compute raised %s for %s", type(exc).__name__, c.cand.candidate_id)
                reward, raw, reason = None, {}, "reward_fn_error"
            if reason is None:
                if reward is None:
                    reason = raw.get("reason") if isinstance(raw, dict) else None
                    if isinstance(decision, Decided):
                        violations.append("decided a reward the full record leaves unscored")
                    elif isinstance(decision, Unscored) and decision.reason != reason:
                        violations.append(f"decide said {decision.reason!r}, compute said {reason!r}")
                else:
                    parts = raw
                    if not isinstance(decision, Decided) or decision.reward != reward or decision.parts != parts:
                        violations.append("decide and compute disagree")
                violations += self._invariant_problems(reward, parts, reason, record, c.cand)
        if violations:
            log.error("reward invariant violated for %s: %s", c.cand.candidate_id, "; ".join(violations))
            self._alarm("reward_invariant_violated", {"candidate_id": c.cand.candidate_id, "problems": violations})
            reward, parts, reason = None, None, "reward_invariant_violated"
        elif reason in ("reward_fn_error", "reward_invariant_violated"):
            self._alarm(reason, {"candidate_id": c.cand.candidate_id})
        full_check = not any(r.state == "skipped_by_plan" for r in record.values())
        rkey, rfields = self._reward_key(c)
        now = self.clock.now()
        timings = {"total_s": now - c.t_submit,
                   "stages": {name: {"start_s": t["start"] - c.t_submit,
                                     "end_s": None if t["end"] is None else t["end"] - c.t_submit}
                              for name, t in c.stage_times.items()}}
        result = RewardResult(
            candidate_id=c.cand.candidate_id, reward=reward, status="scored" if reward is not None else "unscored",
            reason=None if reward is not None else (reason or "unscored"), parts=parts,
            reward_fn=self._rewardfn_identity(), legs=record, full_check=full_check, plan=None, audit=False,
            cache_key=rkey, timings=timings)
        self.cache.put_reward(rkey, rfields, result)
        self._resolve(c, result)

    def _invariant_problems(self, reward, parts, reason, record: dict[str, LegResult], cand: Candidate) -> list[str]:
        out = []
        if reward is None:
            if not (isinstance(reason, str) and reason):
                out.append("an unscored result carries no reason")
            return out
        if type(reward) is not float or not (REWARD_MIN <= reward <= REWARD_MAX):
            out.append(f"reward {reward!r} is not a float in [{REWARD_MIN}, {REWARD_MAX}]")
            return out
        if not (isinstance(parts, dict) and set(parts) == {"base", "broke_healthy", "length"}
                and all(type(v) is float for v in parts.values())):
            out.append(f"parts {parts!r} are not base, broke_healthy and length")
            return out
        if parts["base"] + parts["broke_healthy"] + parts["length"] != reward:
            out.append("parts do not sum to the reward")
        if not (-0.05 <= parts["length"] <= 0.0):
            out.append("length term outside [-0.05, 0]")
        if parts["broke_healthy"] not in (0.0, -0.5):
            out.append("broke_healthy term is neither 0 nor -0.5")
        if self.rewardfn.claims_clean(reward, parts):
            k = self.profile.clean_repeats
            exempt = self._not_applicable(record, cand)
            for leg in self.profile.legs:
                r = record[leg]
                if leg in exempt:
                    continue
                if r.state != "clean":
                    out.append(f"clean reward but required leg {leg} is {r.state}")
                elif not leg.startswith("gate.") and r.repeats < k:
                    out.append(f"clean reward but {leg} has {r.repeats} of {k} repeats")
        return out

    def _not_applicable(self, record: dict[str, LegResult], cand: Candidate) -> frozenset[str]:
        """Legs the reward function itself reports as not applicable for this record (for example ConSan on a
        launch with no LDS), which a clean reward may stand without. None if it has no such rule or fails."""
        rule = getattr(self.rewardfn, "not_applicable_legs", None)
        if rule is None:
            return frozenset()
        try:
            return frozenset(rule(record, cand))
        except Exception:  # noqa: BLE001 - no exemption: the invariant then flags the clean reward
            return frozenset()

    def _resolve(self, c: _Cand, result: RewardResult) -> None:
        c.done = True
        for slot in c.slots.values():
            if slot.status in ("queued", "new"):
                self._remove_from_queues(slot)
                slot.status = "done"
            elif slot.status in ("running", "provisional"):
                slot.orphan = True
        if c.sel is not None and self.selector.pending(c.sel):
            c.held = result  # its plan is still on time: published when it arrives or its deadline passes
            return
        self._publish(c, result)

    def _publish(self, c: _Cand, result: RewardResult) -> None:
        """The candidate's final result, with its own plan record and audit coin, never another
        candidate's (a reward-cache hit is another candidate's result)."""
        c.held = None
        if c.sel is not None:
            result = dataclasses.replace(
                result, plan=self.selector.record(c.sel, reward_cache_hit=bool(result.timings.get("reward_cache_hit"))),
                audit=c.sel.audit)
            self.selector.on_resolved(c.sel, c.cand, result)
        else:
            result = dataclasses.replace(result, plan=None, audit=False)
        self._unresolved -= 1
        self._count_resolved(result)
        if not c.future.done():
            c.future.set_result(result)

    def _count_resolved(self, result: RewardResult) -> None:
        self._counts["resolved"] += 1
        if result.reward is not None:
            self._counts["scored"] += 1
        else:
            self._counts["unscored"][result.reason or "unscored"] += 1

    def _unscored_result(self, cand: Any, reason: str) -> RewardResult:
        cid = getattr(cand, "candidate_id", "") if isinstance(getattr(cand, "candidate_id", None), str) else ""
        return RewardResult(cid, None, "unscored", reason, None, self._rewardfn_identity(), {}, True, None, False, "",
                            {"total_s": 0.0, "stages": {}})

    def _check_drain_done(self) -> None:
        if not self._drain_waiters or self._unresolved > 0:
            return
        now = self.clock.now()
        for waiter, t0 in self._drain_waiters:
            if waiter.done():
                continue
            results = [c.future.result() for c in self._cands.values() if c.future.done()]
            by_reason = Counter(r.reason or "unscored" for r in results if r.reward is None)
            waiter.set_result(DrainReport(
                resolved=len(results), scored=sum(r.reward is not None for r in results),
                unscored=sum(r.reward is None for r in results), unscored_by_reason=dict(by_reason),
                outstanding_at_deadline=list(self._drain_outstanding),
                hit_deadline=bool(self._drain_outstanding), t_drain=t0, t_done=now))
        self._drain_waiters = []

    # --- records ----------------------------------------------------------------------------------

    def _count_run(self, leg: str, result: LegResult) -> None:
        self._counts["leg_runs"][(leg, result.state)] += 1
        if (self.selector is not None and result.state in SETTLED_STATES and result.t_start is not None
                and result.t_end is not None):
            self.selector.observe_cost(leg, result.t_end - result.t_start)  # the planner's running medians

    def _span(self, slot: _Slot, result: LegResult) -> None:
        if self.spans is None or result.t_start is None or result.t_end is None:
            return
        self._record_span(slot.leg, result.t_start, max(result.t_end, result.t_start),
                          candidate_id=slot.cand.cand.candidate_id, state=result.state,
                          reason=result.reason, attempt=slot.attempt, worker=result.worker_id,
                          iteration_of=slot.cand.cand.iteration)

    def _record_span(self, name: str, t_start: float, t_end: float, **attrs: Any) -> None:
        iteration = attrs.pop("iteration_of", None)
        if iteration is None:
            iteration = self._iteration
        try:
            self.spans.record(name, iteration, t_start, max(t_end, t_start),
                              **{k: v for k, v in attrs.items() if v is not None})
        except Exception:
            log.warning("span %s not recorded", name)

    def _alarm(self, kind: str, detail: dict) -> None:
        entry = {"kind": kind, "t": self.clock.now(), **detail}
        self._alarms.append(entry)
        log.error("reward service alarm: %s %s", kind, detail)
        if kind == "no_progress" and self.work_dir is not None:
            try:
                path = self.work_dir / "alarms" / f"no_progress-{int(self.clock.wall())}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(path, {"alarm": entry, "health": self.health_unlocked()})
            except OSError:
                log.warning("could not dump the service state")

    def health_unlocked(self) -> dict:
        return {"pools": {name: {w.worker_id: {"state": w.state, "silent_s": self.clock.now() - w.last_beat,
                                               "item": None if w.item is None else w.item.kind}
                                 for w in pool.workers.values()} for name, pool in self._pools.items()},
                "unresolved": [cid for cid in self._order if not self._cands[cid].done][:100],
                "asan_queued": len(self._asan_ready)}

    def results_json(self) -> list[dict]:
        """Every resolved RewardResult as JSON values, in submission order."""
        with self._lock:
            return [to_dict(self._cands[cid].future.result()) for cid in self._order if self._cands[cid].future.done()]

    def _pool_legs(self, pool_name: str) -> set[str]:
        return {leg for st in self.profile.stages if st.pool == pool_name for leg in st.legs}


def dump_results(path: str | Path, service: RewardService) -> Path:
    return atomic_write_bytes(path, json.dumps(service.results_json(), sort_keys=True).encode("utf-8"))
