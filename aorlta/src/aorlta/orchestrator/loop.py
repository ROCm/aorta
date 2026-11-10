"""C6: the v1 loop (design Section 6.6; decisions D5 and D7).

    for n in iterations:
      1  prompts   = select(pool, n, run_seed)                     deterministic; resume-safe cursor
      2  engines   : all at version v(n-1), landing check passed
      3  generate  : C1 for all P x K chains; candidates stream to C2 as chains end
      4  when every member of a group has a final result: C4 accumulates that group's gradient (streaming);
         groups with < 2 scored members contribute nothing
      5  generation_done barrier (engines idle from here)
      6  scoring   : wait until all futures resolve or the scoring deadline passes
      7  policy    : scored_fraction >= min_scored_fraction? yes -> continue; no -> re-check the unsettled legs once
                     with deadline x 1.5; still no -> stop (the iteration is not trained; gradients are discarded)
      8  step      : finish the remaining groups' backward; one optimizer step; checks (C4)
      9  sync      : merge -> write -> reload -> landing check (C5)
     10  record    : episodes, iteration summary, TensorBoard (C8); checkpoint, atomically
     11  every N   : eval request (C7), asynchronous

Streaming (step 4) is exact: a group is accumulated only once all K chains have ended and every candidate has its
final result. A group with a member that is unscored because a leg is unsettled waits for step 7, since D7's retry
could still score that member and change the group's mean. Groups go to one trainer thread, in completion order.

Commit order of iteration n (step 10): the episodes, summary and spans are written into
`iterations/iter-<n>.partial/` and the checkpoint into `ckpt/iter-<n>.partial/`; the checkpoint is renamed into
place, then the iteration directory. That second rename is the commit point: before it, no reader can see the
iteration, and a resume discards a checkpoint whose iteration directory never appeared. The TensorBoard file is
derived from the committed summary and written after the commit (resume rewrites it if the crash came first).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import queue
import random
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from .. import config as config_mod
from ..atomic import atomic_write_bytes, atomic_write_json, read_text
from ..config import RunConfig
from ..episodes import tensorboard as tb
from ..episodes.schema import build_episode
from ..episodes.summary import build_summary
from ..episodes.writer import IterationOutputs, jsonl_bytes
from ..ids import chain_id, policy_version, sha256_json, sha256_text
from ..schemas import Candidate, RewardResult
from ..spans import SpanRecorder
from . import deadlines as dl
from .ports import (BarrierViolation, ChainEnd, GroupInput, GroupMember, GroupOutcome, MonotonicClock, RunStopped,
                    describe_error)
from .resume import prepare_resume
from .state import (RUN_STATE_SCHEMA, RunState, cursor_for, prune_checkpoints, python_rng_state, restore_python_rng,
                    select_prompts, write_checkpoint)

log = logging.getLogger("aorlta.orchestrator")

RUN_RESULT_SCHEMA = "aorlta.run_result/1"
RUN_LOCK_NAME = "run.lock"  # held by the one process that owns the run directory (Orchestrator._lock_run)
RESULT_WAIT_S = 60.0  # real seconds to wait, after a drain, for result callbacks still in flight
NOOP_WARN, NOOP_STOP = 2, 5  # consecutive NOOP iterations: warn at 2, stop at 5 (Section 6.6, failure modes)


def is_retryable(result: RewardResult | None) -> bool:
    """Unscored because at least one leg is unsettled: D7's retry can still score it."""
    return (result is not None and result.status == "unscored"
            and any(lr.state == "unsettled" for lr in result.legs.values()))


def unscored_result(cand_id: str, reason: str) -> RewardResult:
    return RewardResult(candidate_id=cand_id, reward=None, status="unscored", reason=reason, parts=None,
                        reward_fn={}, legs={}, full_check=False, plan=None, audit=False, cache_key="", timings={})


# --- the generation barrier -------------------------------------------------------------------------------------


class GenerationBarrier:
    """Engines idle from the generation barrier on (A7: a flushing reload is refused while requests run)."""

    def __init__(self, in_flight: Callable[[], int]):
        self._in_flight = in_flight

    def in_flight(self) -> int:
        return int(self._in_flight())

    def assert_idle(self, where: str) -> None:
        busy = self.in_flight()
        if busy != 0:
            raise BarrierViolation(f"{where}: {busy} requests still in flight; a reload now would be refused or "
                                   f"would serve requests across a weight change")


# --- one iteration's ledger ---------------------------------------------------------------------------------------


class IterationLedger:
    """Tracks every chain of the iteration: its record, sample, candidate and reward result, and releases each group
    to the trainer once it is complete and final."""

    def __init__(self, run_id: str, iteration: int, prompts: Sequence[Any], K: int, reward: Any,
                 on_group: Callable[[GroupInput], None], point: Callable[[str, int], None]):
        self.run_id, self.n, self.K = run_id, iteration, K
        self.reward, self.on_group, self.point = reward, on_group, point
        self.prompt_ids = [p.prompt_id for p in prompts]
        self.expected = {chain_id(run_id, iteration, pid, k): (pid, k) for pid in self.prompt_ids for k in range(K)}
        self.records: dict[str, Any] = {}
        self.samples: dict[str, Any] = {}
        self.candidates: dict[str, Candidate] = {}
        self.futures: dict[str, Future] = {}
        self.results: dict[str, RewardResult] = {}
        self.attempt: dict[str, int] = {}
        self.released: set[str] = set()
        self.hold_retryable = True
        self.problems: list[str] = []
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)

    @property
    def total(self) -> int:
        return len(self.expected)

    def _known(self, cid: str, what: str) -> bool:
        if cid not in self.expected:
            self.problems.append(f"{what} for a chain this iteration did not start: {cid}")
            return False
        return True

    # events from the rollout (any thread)

    def on_candidate(self, cand: Candidate) -> None:
        cid = cand.candidate_id
        with self._lock:
            if not self._known(cid, "candidate") or cid in self.candidates:
                return
            self.candidates[cid] = cand
            self.attempt[cid] = 1
        fut = self.reward.submit(cand)
        with self._lock:
            self.futures[cid] = fut
        self.point("gen.candidate", self.n)
        fut.add_done_callback(lambda f, cid=cid, a=1: self._on_result(cid, a, f))

    def on_chain(self, end: ChainEnd) -> None:
        cid = end.record.chain_id
        with self._lock:
            if not self._known(cid, "chain record") or cid in self.records:
                return
            self.records[cid] = end.record
            if end.sample is not None:
                self.samples[cid] = end.sample
        if end.candidate is not None:
            self.on_candidate(end.candidate)
        self.point("gen.chain_end", self.n)
        self._check_group(self.expected[cid][0])

    def generation_done(self, outcome: Any) -> None:
        cands = {c.candidate_id: c for c in getattr(outcome, "candidates", [])}
        samples = {s.chain_id: s for s in getattr(outcome, "samples", [])}
        for rec in getattr(outcome, "chains", []):
            self.on_chain(ChainEnd(rec, cands.get(rec.chain_id), samples.get(rec.chain_id)))
        for cid, cand in cands.items():  # a candidate the rollout did not tie to a chain record
            if cid not in self.candidates:
                self.on_candidate(cand)
        missing = sorted(set(self.expected) - set(self.records))
        if missing:
            self.problems.append(f"{len(missing)} chains have no record, e.g. {missing[0]}")

    # results (reward-service threads)

    def _on_result(self, cid: str, attempt: int, fut: Future) -> None:
        try:
            result = fut.result()
            if not isinstance(result, RewardResult) or result.candidate_id != cid:
                result = unscored_result(cid, "reward_result_invalid")
        except Exception:  # noqa: BLE001 - a failed future is an unscored sample, never a default reward
            result = unscored_result(cid, "reward_future_error")
        with self._cond:
            if self.attempt.get(cid) != attempt:
                return  # superseded by a retry
            self.results[cid] = result
            self._cond.notify_all()
        self.point("score.result", self.n)
        self._check_group(self.expected[cid][0])

    def wait_results(self, timeout_s: float) -> list[str]:
        """After a drain: wait for callbacks still running; a future still unresolved after that is unscored
        (`reward_unresolved`), never left waiting or defaulted."""
        with self._cond:
            self._cond.wait_for(lambda: set(self.futures) <= set(self.results), timeout=timeout_s)
            late = sorted(set(self.futures) - set(self.results))
            for cid in late:
                self.results[cid] = unscored_result(cid, "reward_unresolved")
        for cid in late:
            self._check_group(self.expected[cid][0])
        return late

    # groups

    def _members(self, pid: str) -> list[str]:
        return [chain_id(self.run_id, self.n, pid, k) for k in range(self.K)]

    def _group_input(self, pid: str) -> GroupInput:
        members = tuple(GroupMember(cid, k, self.records[cid], self.samples.get(cid),
                                    self.results.get(cid) if cid in self.candidates else None)
                        for k, cid in enumerate(self._members(pid)))
        return GroupInput(self.n, pid, members)

    def _final(self, cid: str) -> bool:
        if cid not in self.records:
            return False
        if cid not in self.candidates:
            return True
        r = self.results.get(cid)
        return r is not None and not (self.hold_retryable and is_retryable(r))

    def _check_group(self, pid: str) -> None:
        with self._lock:
            if pid in self.released or not all(self._final(c) for c in self._members(pid)):
                return
            self.released.add(pid)
            group = self._group_input(pid)
        self.on_group(group)

    def release_all(self) -> None:
        """Step 7 has decided to train: members still unscored are final now, and every group goes to the trainer."""
        with self._lock:
            self.hold_retryable = False
        for pid in self.prompt_ids:
            self._check_group(pid)

    # the policy

    def handed_over(self) -> int:
        """D7's denominator: the candidates handed to the reward service. A chain an engine lost, or one the leak
        check quarantined, never reaches it."""
        with self._lock:
            return len(self.candidates)

    def scored(self) -> int:
        with self._lock:
            return sum(1 for r in self.results.values() if r.status == "scored")

    def retryable(self) -> list[Candidate]:
        with self._lock:
            return [self.candidates[cid] for cid in sorted(self.results) if is_retryable(self.results[cid])]

    def resubmit(self, cands: Sequence[Candidate]) -> None:
        for cand in cands:
            cid = cand.candidate_id
            with self._lock:
                self.attempt[cid] = self.attempt.get(cid, 1) + 1
                a = self.attempt[cid]
                self.results.pop(cid, None)
            fut = self.reward.resubmit(cand)
            with self._lock:
                self.futures[cid] = fut
            fut.add_done_callback(lambda f, cid=cid, a=a: self._on_result(cid, a, f))

    def unsettled_detail(self) -> list[dict]:
        """Which legs were unsettled, and on which workers (ids only)."""
        out = []
        with self._lock:
            for cid in sorted(self.results):
                r = self.results[cid]
                for leg, lr in sorted(r.legs.items()):
                    if lr.state == "unsettled":
                        out.append({"candidate_id": cid, "leg": leg, "reason": lr.reason, "worker_id": lr.worker_id})
        return out


# --- the trainer thread --------------------------------------------------------------------------------------------


class TrainerWorker:
    """Feeds complete groups to the trainer, one at a time and in completion order (`train.stream` spans). A trainer
    with `records_train_spans` hands each group to one of its ranks and times it there (one span per rank's group),
    so the hand-over here is not a span."""

    def __init__(self, trainer: Any, spans: SpanRecorder, iteration: int, point: Callable[[str, int], None]):
        self.trainer, self.spans, self.n, self.point = trainer, spans, iteration, point
        self.outcomes: dict[str, GroupOutcome] = {}
        self.error: BaseException | None = None
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=f"trainer-stream-{iteration}", daemon=True)
        self._thread.start()

    def submit(self, group: GroupInput) -> None:
        self._q.put(group)

    def _run(self) -> None:
        while True:
            group = self._q.get()
            if group is None:
                return
            if self.error is not None:
                continue
            try:
                if getattr(self.trainer, "records_train_spans", False):
                    out = self.trainer.add_group(group)
                else:
                    with self.spans.span("train.stream", self.n, prompt_id=group.prompt_id,
                                         members=len(group.members)):
                        out = self.trainer.add_group(group)
                self.outcomes[group.prompt_id] = out
                self.point("train.group", self.n)
            except BaseException as exc:  # noqa: BLE001 - reported at join, before any step
                self.error = exc

    def join(self) -> dict[str, GroupOutcome]:
        self._q.put(None)
        self._thread.join()
        if self.error is not None:
            raise self.error
        return self.outcomes


# --- the orchestrator -----------------------------------------------------------------------------------------------


class Orchestrator:
    def __init__(self, cfg: RunConfig, *, pool: Any, rollout: Any, reward: Any, trainer: Any, sync: Any,
                 eval_hook: Any = None, clock: Any = None, spans: SpanRecorder | None = None,
                 point: Callable[[str, int], None] | None = None, commit: str | None = None,
                 package_dir: str | Path | None = None, config_path: str | Path | None = None):
        self.cfg = cfg
        self.run_dir = cfg.run_dir
        self.source, self.run_id = cfg.run.source, cfg.run.run_id
        self.pool, self.rollout, self.reward, self.trainer, self.sync = pool, rollout, reward, trainer, sync
        self.eval_hook = eval_hook
        self.clock = clock or MonotonicClock()
        self.spans = spans or SpanRecorder(monotonic=self.clock.now)
        self._point = point or (lambda name, iteration: None)
        self.commit_id, self.package_dir, self.config_path = commit, package_dir, config_path
        self.barrier = GenerationBarrier(rollout.in_flight)
        self._eval_seen = 0
        self._reward_stats: Any = None  # the reward service's stats at the previous summary
        self.history: dict[str, list[float]] = {"generation_s": [], "scoring_s": []}
        self.noop_streak = 0
        self.engine_version: int | None = None
        self.base_sha256: str | None = None
        self.iteration = 0  # the last committed iteration
        self.attempt = 1
        self.resumed_from: int | None = None
        self._owns_run = False  # set once this process has written (or, resuming, verified) the run plan
        self._run_lock: int | None = None  # the descriptor holding RUN_LOCK_NAME

    def point(self, name: str, iteration: int) -> None:
        """A named place in the loop. Tests hook it (the kill -9 test dies at one); it does nothing by default."""
        self._point(name, iteration)

    # --- the run -------------------------------------------------------------------------------------------------

    def run(self, *, resume: bool = False) -> dict:
        try:
            start = self._setup(resume)
            for n in range(start, self.cfg.orchestrator.iterations + 1):
                self._iteration(n)
            self._finish()
            return self._write_result("completed")
        except RunStopped as stop:
            log.error("run stopped at iteration %d: %s", self.iteration + 1, stop.reason)
            return self._write_result("stopped", stop)
        except Exception as exc:
            self._write_result("stopped", RunStopped("unexpected_error", {"error": describe_error(exc, self.source)}))
            raise
        finally:
            self._close()

    def _setup(self, resume: bool) -> int:
        o = self.cfg.orchestrator
        with self.spans.span("setup.plan", 0, resume=resume):
            if resume:
                plan_path = self.run_dir / config_mod.RUN_PLAN_NAME
                if not plan_path.exists():
                    raise RunStopped("no_run_plan", {"run_dir": str(self.run_dir)})
                self._lock_run()
                config_mod.write_run_plan(self.cfg, exist_ok=True, commit=self.commit_id, package_dir=self.package_dir)
                self._owns_run = True
                rp = prepare_resume(self.run_dir, o.keep_ckpts)
                state = rp.state
                self.attempt = self._previous_attempts() + 1
            else:
                config_mod.write_run_plan(self.cfg, config_path=self.config_path, commit=self.commit_id,
                                          package_dir=self.package_dir, extra_files=self._base_files())
                self._lock_run()  # after the plan, which is a run directory's first file
                self._owns_run = True
                state, rp = None, None
            self.base_sha256 = state.base_sha256 if state is not None else self._base_sha256_from_plan()
            self.iteration = state.iteration if state is not None else 0
            self.resumed_from = self.iteration if resume else None
            self._write_result("running")
        self.prompt_records = list(self.pool.records())
        for r in self.prompt_records:
            if r.source != self.source:
                raise RunStopped("mixed_sources", {"prompt_id": r.prompt_id, "prompt_source": r.source})
        P, seed = self.cfg.rollout.P, o.run_seed
        if state is not None and cursor_for(state.iteration + 1, self.prompt_records, seed, P) != state.next_cursor:
            raise RunStopped("prompt_pool_changed", {"iteration": state.iteration + 1})
        first, _ = select_prompts(self.prompt_records, self.iteration + 1, seed, P)
        with self.spans.span("setup.engines", 0):
            self.rollout.start(first[:4])
        if self.cfg.sync.round_trip_at_start:
            with self.spans.span("setup.round_trip", 0):
                rt = self.sync.round_trip()
            if rt.get("verdict") != "PROVEN":
                raise RunStopped("round_trip_failed", {"verdict": rt.get("verdict")})
        if state is not None and state.iteration >= 1:
            with self.spans.span("setup.restore", 0, restored=state.iteration):
                self.trainer.load_state(rp.checkpoint / "trainer")
                if self.trainer.state_digest() != state.trainer_digest:
                    raise RunStopped("checkpoint_trainer_mismatch", {"iteration": state.iteration})
                restored = self.sync.restore(state.iteration)
            bad = {e: v for e, v in (restored.get("verdicts") or {}).items() if v not in ("LANDED", "NOOP")}
            if bad or not restored.get("verdicts"):
                raise RunStopped("restore_landing_failed", {"verdicts": restored.get("verdicts")})
        if self.cfg.trainer.startup_warmup:  # after any restore, so a resumed process is warmed on its own state
            warmup = getattr(self.trainer, "warmup", None)
            if warmup is None:
                raise RunStopped("warmup_unavailable", {"trainer": type(self.trainer).__name__})
            with self.spans.span("setup.warmup", 0) as h:
                try:
                    w = warmup()
                except Exception as exc:
                    raise RunStopped("warmup_failed", {"error": describe_error(exc, self.source)}) from None
                c, sec = w.get("counts") or {}, w.get("seconds") or {}
                h.set(microbatches=c.get("microbatches"), forward_tokens=c.get("forward_tokens"),
                      loss_tokens=c.get("loss_tokens"), forward_s=sec.get("forward"), backward_s=sec.get("backward"),
                      state=w.get("state"), max_allocated_gib=(w.get("memory") or {}).get("max_allocated_gib"))
        if state is not None:
            self.history = {k: list(v) for k, v in state.history.items()}
            self.noop_streak = state.noop_streak
            if state.rng.get("python_random") is not None:
                restore_python_rng(state.rng["python_random"])
        else:  # a new run: Python's own RNG starts from the run seed, so the state each checkpoint saves is the run's
            random.seed(int(sha256_json(["aorlta.python_random/1", o.run_seed])[:16], 16))
        self.engine_version = self.iteration
        if self.eval_hook is not None:
            self.eval_hook.base_sha256 = self.base_sha256
            with self.spans.span("setup.eval_reconcile", 0):
                self.eval_hook.reconcile(self.iteration, can_export=lambda m: m == self.engine_version)
            self._eval_seen = len(self.eval_hook.events)
        attempt_dir = self.run_dir / "attempts"
        attempt_dir.mkdir(exist_ok=True)
        atomic_write_bytes(attempt_dir / f"attempt-{self.attempt:03d}-setup-spans.jsonl",
                           jsonl_bytes(s.to_json() for s in self.spans.spans(0)))
        return self.iteration + 1

    # --- one iteration -----------------------------------------------------------------------------------------

    def _iteration(self, n: int) -> None:
        cfg, o = self.cfg, self.cfg.orchestrator
        warnings: list[str] = []
        self.point("select", n)
        t_start = self.spans.now()
        with self.spans.span("select", n):
            prompts, cursor = select_prompts(self.prompt_records, n, o.run_seed, cfg.rollout.P)
        if self.engine_version != n - 1:  # step 2
            raise RunStopped("engines_not_at_previous_version", {"engines": self.engine_version, "iteration": n})
        sampled = policy_version(self.run_id, n - 1)
        self.reward.reopen()
        worker = TrainerWorker(self.trainer, self.spans, n, self.point)
        self.trainer.begin(n)
        ledger = IterationLedger(self.run_id, n, prompts, cfg.rollout.K, self.reward, worker.submit, self.point)
        try:
            return self._iteration_body(n, prompts, cursor, sampled, ledger, worker, warnings, t_start)
        except BaseException:
            try:
                worker.join()
            except BaseException:  # noqa: BLE001 - the first error is the one reported
                pass
            self.trainer.discard()
            raise

    def _iteration_body(self, n, prompts, cursor, sampled, ledger, worker, warnings, t_start) -> None:
        o = self.cfg.orchestrator
        # 3: generate, candidates streaming to the reward service
        gen_s = dl.generation_deadline_s(o.deadlines.generation_s, self.history["generation_s"])
        t_gen = self.clock.now()
        try:
            outcome = self.rollout.run(prompts, n, sampled, on_candidate=ledger.on_candidate, on_chain=ledger.on_chain,
                                       deadline=t_gen + gen_s)
        except RunStopped:
            raise
        except Exception as exc:
            raise RunStopped("rollout_failed", {"error": describe_error(exc, self.source)}) from None
        # 5: the generation barrier
        ledger.generation_done(outcome)
        self.point("gen.done", n)
        self._barrier("generation barrier", n)
        t_barrier = self.clock.now()
        gen_dur = t_barrier - t_gen
        stops = [self._safe(s) for s in getattr(outcome, "stop_reasons", [])]
        if stops or ledger.problems:
            raise RunStopped("rollout_stop", {"stop_reasons": stops, "problems": [self._safe(p) for p in ledger.problems]})
        gen_hit = bool(getattr(outcome, "deadline_hit", False))
        cut = set(getattr(outcome, "outstanding", []) or [])
        handed = ledger.handed_over()
        died = sorted(cid for cid, rec in ledger.records.items() if rec.end_reason == "engine_error" and cid not in cut)
        if died:  # chains an engine lost: not handed over, so not in D7's fraction; said here instead
            warnings.append("engine_error_chains")
            log.warning("iteration %d: %d chains ended engine_error and were not checked", n, len(died))
        # 6: scoring, bounded by the deadline
        score_s = dl.scoring_deadline_s(o.deadlines.scoring_s, self.history["scoring_s"])
        with self.spans.span("score.wait", n, deadline_s=score_s):
            drain = self.reward.drain(t_barrier + score_s)
            late = ledger.wait_results(RESULT_WAIT_S)
        score_dur = self.clock.now() - t_barrier
        self.point("score.drained", n)
        if late:
            warnings.append("reward_unresolved")
        outstanding = list(getattr(drain, "outstanding_at_deadline", []) or [])
        score_hit = bool(getattr(drain, "hit_deadline", False))
        # 7: the policy (D7), over the candidates handed to the reward service
        decision = dl.decide(scored=ledger.scored(), total=handed, retryable=len(ledger.retryable()),
                             min_fraction=o.min_scored_fraction, retried=False)
        policy = {"first": decision.as_dict(), "retry": None, "denominator": "candidates_handed_over",
                  "chains": ledger.total, "candidates": handed,
                  "not_handed_over": {"engine_error": len(died), "deadline": len(cut),
                                      "quarantined": sum(1 for r in ledger.records.values()
                                                         if r.end_reason == "quarantined")}}
        if decision.decision == "retry":
            cands = ledger.retryable()
            retry_s = dl.RETRY_DEADLINE_FACTOR * score_s
            with self.spans.span("score.retry", n, candidates=len(cands), deadline_s=retry_s):
                self.reward.reopen()
                ledger.resubmit(cands)
                t_retry = self.clock.now()
                drain2 = self.reward.drain(t_retry + retry_s)
                ledger.wait_results(RESULT_WAIT_S)
            outstanding += list(getattr(drain2, "outstanding_at_deadline", []) or [])
            decision = dl.decide(scored=ledger.scored(), total=handed, retryable=len(ledger.retryable()),
                                 min_fraction=o.min_scored_fraction, retried=True)
            policy["retry"] = {"candidates": len(cands), "deadline_s": retry_s, "t_start": t_retry,
                               "t_end": self.clock.now(), **decision.as_dict()}
        policy["decision"] = decision.decision
        self.point("policy.decided", n)
        if decision.decision == "stop":
            raise RunStopped("scored_fraction_below_threshold",
                             {"iteration": n, "policy": policy, "outstanding_at_deadline": outstanding,
                              "unsettled": ledger.unsettled_detail()})
        # 8: the step
        ledger.release_all()
        try:
            outcomes = worker.join()
        except Exception as exc:
            raise RunStopped("trainer_failed", {"error": describe_error(exc, self.source)}) from None
        trained = sum(len(g.trained) for g in outcomes.values())
        step = noop_report = None
        if trained == 0:
            self.trainer.discard()
            warnings.append("no_trainable_samples")
        elif not any(g.advantages.get(cid, 0.0) != 0.0 for g in outcomes.values() for cid in g.trained):
            # Every trained sample's group scored all its members alike: the policy gradient is zero, and the
            # trainer refuses a zero step. A NOOP iteration, not a stop.
            log.warning("iteration %d: every group has zero advantage (%d samples); no step (NOOP)", n, trained)
            warnings.append("zero_advantage")
            report_fn = getattr(self.trainer, "noop_report", None)
            if report_fn is not None:
                try:
                    noop_report = report_fn()
                except Exception as exc:  # noqa: BLE001 - only the record of a step not taken
                    log.warning("iteration %d: no gap report: %s", n, describe_error(exc, self.source))
            self.trainer.discard()
        else:
            try:
                with self.spans.span("train.final", n, groups=len(outcomes), samples=trained):
                    step = self.trainer.finish()
            except Exception as exc:
                raise RunStopped("trainer_check_failed", {"error": describe_error(exc, self.source)}) from None
            self.point("train.step", n)
        # 9: sync
        if self.eval_hook is not None:
            self.eval_hook.wait_exports()  # an export reads v(n-1)'s slot, which this publication removes
        sync_report = None
        if step is not None:
            self._barrier("reload", n)
            try:
                sync_report = self.sync.publish(n, in_flight=self.barrier.in_flight)
            except BarrierViolation as exc:
                raise RunStopped("barrier_violation", {"iteration": n, "where": "sync", "error": str(exc)[:300]}) from None
            except Exception as exc:
                raise RunStopped("landing_failed", {"error": describe_error(exc, self.source)}) from None
            verdicts = sync_report.get("verdicts") or {}
            if not verdicts or any(v not in ("LANDED", "NOOP") for v in verdicts.values()):
                raise RunStopped("landing_failed", {"verdicts": verdicts})
            self.point("sync.published", n)
        noop = step is None or all(v == "NOOP" for v in sync_report["verdicts"].values())
        self.noop_streak = self.noop_streak + 1 if noop else 0
        if self.noop_streak >= NOOP_WARN:
            warnings.append("noop_steps")
        self.engine_version = n
        t_landed = self.spans.now()
        if step is None:  # nothing was published, so no landing span ends the iteration: this marker does
            self.spans.record("no_step", n, t_landed, t_landed,
                              reason="no_trainable_samples" if trained == 0 else "zero_advantage")
        self.history = {"generation_s": dl.push_history(self.history["generation_s"], gen_dur),
                        "scoring_s": dl.push_history(self.history["scoring_s"], score_dur)}
        deadlines = {"generation_s": gen_s, "generation_hit": gen_hit,
                     "generation_outstanding": list(getattr(outcome, "outstanding", []) or []),
                     "scoring_s": score_s, "scoring_hit": score_hit, "outstanding_at_deadline": outstanding,
                     "generation_dur_s": gen_dur, "scoring_dur_s": score_dur}
        # 10: record and commit
        self._record(n, cursor, ledger, outcomes, step, sync_report, policy, deadlines, warnings, outcome, t_start,
                     t_landed, sampled, trainer_noop=noop_report)
        # 11: eval, asynchronous
        if self.eval_hook is not None:
            try:
                self.eval_hook.after_commit(n)
                self.eval_hook.poll(n)
            except Exception as exc:  # noqa: BLE001 - the eval being down never stops training
                log.warning("eval hook: %s", describe_error(exc, self.source))
        if self.noop_streak >= NOOP_STOP:
            raise RunStopped("noop_streak", {"iterations": self.noop_streak,
                                             "why": "the merged update rounded away on every engine; it is not learning"})

    def _barrier(self, where: str, n: int) -> None:
        try:
            self.barrier.assert_idle(where)
        except BarrierViolation as exc:  # a programming error: stop before anything is reloaded (6.5 failure modes)
            raise RunStopped("barrier_violation", {"iteration": n, "where": where, "error": str(exc)[:300]}) from None

    # --- step 10 ------------------------------------------------------------------------------------------------

    def _episodes(self, n: int, ledger: IterationLedger, outcomes: dict[str, GroupOutcome], step: dict | None) -> list[dict]:
        gated = set((step or {}).get("excluded") or [])
        out = []
        for cid in sorted(ledger.expected):
            pid, k = ledger.expected[cid]
            rec, cand = ledger.records[cid], ledger.candidates.get(cid)
            result = ledger.results.get(cid) if cand is not None else None
            g = outcomes.get(pid)
            advantage = g.advantages.get(cid) if g is not None and not g.dropped else None
            trained, why = False, None
            if rec.end_reason in ("quarantined", "engine_error"):
                why = rec.end_reason
            elif result is None or result.status != "scored":
                why = "unscored"
            elif g is None or g.dropped:
                why = "group_dropped"
            elif cid not in g.trained:
                why = "logprob_error" if rec.end_reason == "logprob_error" else "no_sample"
            elif step is None:
                why = "no_step"
            elif cid in gated:
                why = "gap_gate"
            else:
                trained = True
            out.append(build_episode(source=self.source, run_id=self.run_id, iteration=n, record=rec, candidate=cand,
                                     result=result, advantage=advantage, trained=trained, excluded_reason=why))
        return out

    def _record(self, n, cursor, ledger, outcomes, step, sync_report, policy, deadlines, warnings, outcome, t_start,
                t_landed, sampled, trainer_noop=None) -> None:
        o = self.cfg.orchestrator
        trained_version = policy_version(self.run_id, n)
        with self.spans.span("record", n):
            episodes = self._episodes(n, ledger, outcomes, step)
            outs = IterationOutputs(self.run_dir, n)
            outs.write_episodes(episodes)
        self.point("record.episodes", n)
        next_cursor = cursor_for(n + 1, self.prompt_records, o.run_seed, self.cfg.rollout.P)
        state = RunState(
            schema=RUN_STATE_SCHEMA, run_id=self.run_id, source=self.source, iteration=n,
            policy_version=trained_version, config_sha256=sha256_json(config_mod.plain(self.cfg)),
            next_cursor=next_cursor, engines_slot=self.sync.held_slot(), history=self.history,
            noop_streak=self.noop_streak, trainer_digest=self.trainer.state_digest(), base_sha256=self.base_sha256,
            rng={"python_random": python_rng_state()},
        )
        with self.spans.span("ckpt", n):
            pending = write_checkpoint(self.run_dir / "ckpt", state, self.trainer.save_state,
                                       point=lambda name: self.point(name, n))
        eval_events = []
        if self.eval_hook is not None:  # what the eval did since the previous summary (results, requests issued)
            eval_events = list(self.eval_hook.events[self._eval_seen:])
            self._eval_seen += len(eval_events)
        reward_stats = self.reward.stats()
        summary = build_summary(
            run_id=self.run_id, source=self.source, iteration=n, sampled=sampled, trained=trained_version,
            episodes=episodes, results={cid: r for cid, r in ledger.results.items() if cid in ledger.candidates},
            total=ledger.total, reward_stats=reward_stats, reward_stats_before=self._reward_stats,
            groups=outcomes, step=step, sync=sync_report,
            policy=policy, deadlines=deadlines, spans=self.spans.spans(n), t_start=t_start, t_landed=t_landed,
            noop_streak=self.noop_streak, warnings=warnings, eval_events=eval_events,
            rollout_metrics=self._safe_metrics(getattr(outcome, "metrics", {}) or {}), trainer_noop=trainer_noop,
        )
        self._reward_stats = reward_stats
        outs.write_summary(summary)
        outs.write_json("cursor.json", cursor)
        outs.write_spans(self.spans.spans(n))
        self.point("record.summary", n)
        pending.commit()  # in place, but not committed until the iteration directory is
        self.point("ckpt.renamed", n)
        outs.commit()  # the commit point of iteration n
        self.iteration = n
        self.point("record.committed", n)
        tb.write_iteration(self.run_dir / "tb", n, tb.scalars_from_summary(summary), summary["timing"]["t_landed"])
        self.point("record.tb", n)
        prune_checkpoints(self.run_dir / "ckpt", o.keep_ckpts)
        self.point("ckpt.pruned", n)
        log.info("iteration %d committed: scored %d of %d, trained %d, noop streak %d", n,
                 summary["samples"]["scored"], ledger.total, summary["samples"]["trained"], self.noop_streak)

    # --- run records ---------------------------------------------------------------------------------------------

    def _finish(self) -> None:
        if self.eval_hook is not None:
            try:
                self.eval_hook.wait_exports()
                self.eval_hook.poll(self.iteration)
            except Exception as exc:  # noqa: BLE001
                log.warning("eval hook at the end of the run: %s", describe_error(exc, self.source))

    def _previous_attempts(self) -> int:
        path = self.run_dir / "run_result.json"
        if not path.exists():
            return 0
        try:
            return len(json.loads(read_text(path)).get("attempts", []))
        except ValueError:
            return 0

    def _write_result(self, status: str, stop: RunStopped | None = None) -> dict:
        path = self.run_dir / "run_result.json"
        old = {}
        if path.exists():
            try:
                old = json.loads(read_text(path))
            except ValueError:
                old = {}
        attempts = [a for a in old.get("attempts", []) if a.get("attempt") != self.attempt]
        attempts.append({"attempt": self.attempt, "status": status, "resumed_from": self.resumed_from,
                         "iterations_completed": self.iteration,
                         "stop": None if stop is None else {"reason": stop.reason, "detail": stop.detail}})
        result = {
            "schema": RUN_RESULT_SCHEMA, "run_id": self.run_id, "source": self.source, "status": status,
            "iterations_planned": self.cfg.orchestrator.iterations, "iterations_completed": self.iteration,
            "policy_version": policy_version(self.run_id, self.iteration),
            "stop": None if stop is None else {"reason": stop.reason, "detail": stop.detail},
            "attempts": attempts,
            "eval": self.eval_hook.status() if self.eval_hook is not None else None,
        }
        if self._owns_run:  # a process refused at the run plan never touches the run's records
            atomic_write_json(path, result)
        return result

    def _lock_run(self) -> None:
        """Hold the run directory's lock until the run closes: one process owns a run at a time, so a resume started
        while the run (or another resume) is alive is refused before it touches the run directory."""
        fd = os.open(self.run_dir / RUN_LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise RunStopped("run_in_use", {"run_dir": str(self.run_dir)}) from None
        self._run_lock = fd

    def _close(self) -> None:
        try:
            if not self._owns_run:  # refused at the run plan, it started nothing; the run's node files are another's
                return
            for name, fn in (("reward", getattr(self.reward, "close", None)),
                             ("rollout", getattr(self.rollout, "shutdown", None))):
                if fn is not None:
                    try:
                        fn()
                    except Exception as exc:  # noqa: BLE001
                        log.warning("closing %s: %s", name, describe_error(exc, self.source))
        finally:
            if self._run_lock is not None:  # closing the descriptor releases the lock
                os.close(self._run_lock)
                self._run_lock = None

    # --- data rules ---------------------------------------------------------------------------------------------

    def _safe(self, text: str) -> str:
        """A component's message for the run records: kept for stand-in runs, hashed for any other source (6.10)."""
        return str(text)[:500] if self.source == "standin" else f"sha256:{sha256_text(str(text))[:16]}"

    def _safe_metrics(self, metrics: dict) -> dict:
        m = dict(metrics)
        if "deaths" in m:
            m["deaths"] = [{k: (self._safe(v) if k == "error" else v) for k, v in d.items()} for d in m["deaths"]]
        return m

    def _base_files(self) -> dict[str, Path]:
        base = Path(self.cfg.model.base_dir)
        if not base.is_dir():
            raise RunStopped("base_dir_missing", {"base_dir": str(base)})
        files = {}
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(base)
            if any(part.startswith(".") for part in rel.parts) or not p.is_file():
                continue
            files[f"base/{rel.as_posix()}"] = p
        if not files:
            raise RunStopped("base_dir_empty", {"base_dir": str(base)})
        return files

    def _base_sha256_from_plan(self) -> str:
        plan = json.loads(read_text(self.run_dir / config_mod.RUN_PLAN_NAME))
        base = {k: v["sha256"] for k, v in plan.get("extra_files", {}).items() if k.startswith("base/")}
        return sha256_json(base)
