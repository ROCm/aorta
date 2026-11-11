"""Thin fakes of the components the orchestrator drives, and a fake-run harness (tests; the kill -9 test, C6.T1).

Time is a discrete-event `VirtualClock`: nothing sleeps. The fake rollout runs every chain through WP0's
`FakeEngine` (real seeds from `IterationSeeds`, compile-repair turns, feedback, leak check, quarantine, engine
errors, misaligned log-probs) and reports chain ends in virtual-time order, advancing the clock to each; the fake
reward service schedules each candidate's result on the same clock, so results arrive while chains still run,
exactly the streaming the loop has to handle. A stalled check (10^4 s) is resolved by `drain` at its deadline.

Every fake is deterministic in its inputs: the same prompts, seeds and weights give the same chains, the same
candidates give the same rewards, and the fake trainer's update is a function of the trained samples' contents,
summed in chain-id order, so the order groups arrive in cannot change it. Its state (latent, Adam moments, step
count, RNG) round-trips exactly through JSON. That is what lets C6.T1 compare a killed-and-resumed run with an
uninterrupted one bit for bit.

The harness runs as a subprocess: `python -m aorlta.orchestrator.fakes --spec SPEC.json [--resume]
[--kill-at NAME:ITERATION:K] [--points-out FILE]`. `--kill-at` sends SIGKILL to its own process at the K-th
time the loop reaches point NAME in ITERATION.
"""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import json
import logging
import math
import os
import random
import signal
import sys
import threading
import types
from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..atomic import atomic_write_json, read_text
from ..config import REWARD_MAX, REWARD_MIN, RunConfig, config_from_dict
from ..eval.hook import EvalHook, FakeEval
from ..ids import HashRng, IterationSeeds, chain_id, sha256_ids, sha256_json, sha256_text
from ..rollout.fake_engine import (COMPILE_ERROR_MARKER, STOP_TOKEN_ID, ByteTokenizer, FakeEngine, FakeEngineConfig,
                                   FakeEngineError)
from ..schemas import (Candidate, ChainRecord, CompileResult, LegResult, PromptRecord, RewardResult, TaskRef,
                       TrainingSample, TurnRecord)
from ..spans import SpanRecorder
from .loop import Orchestrator
from .ports import BarrierViolation, ChainEnd, GroupInput, GroupOutcome

FAKES_SCHEMA = "aorlta.orchestrator_fakes/1"
SCORED_ENDS = frozenset({"compiled", "no_code_block", "turns_exhausted", "context_limit", "logprob_error"})
TRAINED_ENDS = frozenset({"compiled", "no_code_block", "turns_exhausted", "context_limit"})


# --- time -------------------------------------------------------------------------------------------------------


class VirtualClock:
    """Monotonic seconds that move only when advanced; events scheduled on it run in (time, order) sequence."""

    def __init__(self, start: float = 1000.0):
        self._t = float(start)
        self._heap: list = []
        self._seq = 0
        self._lock = threading.RLock()

    def now(self) -> float:
        return self._t

    def schedule(self, t: float, fn) -> None:
        with self._lock:
            heapq.heappush(self._heap, (float(t), self._seq, fn))
            self._seq += 1

    def advance_to(self, t: float) -> None:
        with self._lock:
            while self._heap and self._heap[0][0] <= t:
                et, _, fn = heapq.heappop(self._heap)
                self._t = max(self._t, et)
                fn()
            self._t = max(self._t, float(t))


def wall_of(spans: SpanRecorder, clock: VirtualClock, t: float) -> float:
    """The UTC time on `spans`' timeline of virtual monotonic time `t` (spans reads `clock.now`)."""
    return spans.now() - (clock.now() - t)


# --- prompts ------------------------------------------------------------------------------------------------------


@dataclass
class FakePool:
    source: str
    _records: list[PromptRecord]

    def records(self) -> list[PromptRecord]:
        return list(self._records)


def make_pool(n: int, *, source: str = "standin", canary: str | None = None) -> FakePool:
    """`n` prompts with their own turn-0 ids. With `canary`, each prompt's text (and so its ids) carries it."""
    tok = ByteTokenizer()
    out = []
    for i in range(n):
        pid = f"{source}:p{i:03d}"
        text = f"Fix this kernel {i}." + (f" {canary}-{i}" if canary else "")
        ids = [248045] + tok.encode(text) + [248046, 198, 248045]
        task = TaskRef(contract_id=f"c{i:03d}", parent_id=None, parent_healthy=(i % 3 == 0), reference_kind="frozen",
                       task_sha256=sha256_text(f"{FAKES_SCHEMA}/task/{i}"))
        out.append(PromptRecord(prompt_id=pid, turn0_ids=ids,
                                messages=[{"role": "user", "content": text}] if canary else None,
                                prompt_ids_sha256=sha256_ids(ids), task=task, source=source))
    return FakePool(source, out)


# --- the rollout ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeRolloutConfig:
    engines: int = 2
    engine: FakeEngineConfig = field(default_factory=lambda: FakeEngineConfig(
        seed=7, min_tokens=60, max_tokens=240, latency_base_s=4.0, latency_per_token_s=0.05, latency_jitter_s=6.0,
        no_code_block_rate=0.08, compile_error_rate=0.45, error_rate=0.01, bad_logprobs_rate=0.02))
    quarantine_rate: float = 0.04  # with a leak check: feedback that fails it
    compile_s: float = 0.6
    max_model_len: int | None = None  # a repair turn that would not fit ends the chain context_limit
    canary_source: str | None = None  # planted in every reply's code (C8.T3)
    canary_feedback: str | None = None  # planted in every feedback text (C8.T3)
    raise_at_iteration: int | None = None  # the rollout raises, with a canary in its message (C8.T3)


@dataclass
class _End:
    record: ChainRecord
    candidate: Candidate | None
    sample: TrainingSample | None
    t_end: float
    spans: list
    aborted: bool


class FakeRollout:
    def __init__(self, cfg: RunConfig, clock: VirtualClock, spans: SpanRecorder, fake: FakeRolloutConfig | None = None):
        self.cfg, self.clock, self.spans = cfg, clock, spans
        self.fake = fake or FakeRolloutConfig()
        self.tok = ByteTokenizer()
        self.engines = [FakeEngine(self.fake.engine, engine_id=i, sampling_backend="triton", enable_output_logprobs=True)
                        for i in range(self.fake.engines)]
        self.leak = 0  # requests "still in flight" after run returns (C6.T3)
        self.started = False
        self.runs: list[int] = []

    def start(self, selftest_prompts) -> dict:
        differ = 0
        for i, p in enumerate(list(selftest_prompts)[:4]):
            outs = []
            for s in (1, 2):
                sp = {"temperature": 0.7, "top_p": 0.95, "top_k": -1, "max_new_tokens": 128,
                      "seed": 1_000_003 * (i + 1) + s, "stop_token_ids": [STOP_TOKEN_ID]}
                outs.append(self.engines[0].generate(input_ids=list(p.turn0_ids), sampling_params=sp)["output_ids"])
            differ += outs[0] != outs[1]
        if differ < 3:
            raise RuntimeError(f"fake self-test: seeds differ on {differ} of 4")
        self.started = True
        return {"engines": len(self.engines), "self_test_differ": differ}

    def set_weights(self, tag: str) -> None:
        for e in self.engines:
            e.set_weights(tag)

    def in_flight(self) -> int:
        return self.leak

    def shutdown(self) -> None:
        self.started = False

    def _wall(self, t: float) -> float:
        return wall_of(self.spans, self.clock, t)

    def _chain(self, p: PromptRecord, k: int, engine_i: int, seeds: IterationSeeds, n: int, version: str, t0: float,
               deadline: float, leak_check: bool) -> _End:
        r = self.cfg.rollout
        eng = self.engines[engine_i]
        cid = chain_id(self.cfg.run.run_id, n, p.prompt_id, k)
        ids = list(p.turn0_ids)
        t, turns, spans, inputs = t0, [], [], []
        end, aborted = None, False
        for turn in range(r.max_repair_turns + 1):
            seed = seeds.seed_for(p.prompt_id, k, turn)
            sp = {"temperature": r.temperature, "top_p": r.top_p, "top_k": r.top_k, "max_new_tokens": r.max_new_tokens,
                  "seed": seed, "stop_token_ids": [STOP_TOKEN_ID]}
            name = "gen.turn0" if turn == 0 else "gen.repair"
            try:
                res = eng.generate(input_ids=ids, sampling_params=sp, return_logprob=True)
            except FakeEngineError:
                t_fail = min(t + self.fake.engine.latency_base_s, deadline)
                spans.append((name, t, t_fail, {"chain_id": cid, "turn": turn, "error": "engine_error"}))
                t, end = t_fail, "engine_error"
                break
            out_ids = list(res["output_ids"])
            triples = list(res["meta_info"]["output_token_logprobs"])
            if self.fake.canary_source:
                extra = self.tok.encode(f"// {self.fake.canary_source}\n")
                at = len(out_ids) - 1 if out_ids and out_ids[-1] == STOP_TOKEN_ID else len(out_ids)
                out_ids[at:at] = extra
                triples[at:at] = [(-0.5, x, None) for x in extra]
            t_done = t + float(res["meta_info"]["e2e_latency"])
            if t_done > deadline:
                spans.append((name, t, deadline, {"chain_id": cid, "turn": turn, "error": "generation_deadline"}))
                t, end, aborted = deadline, "engine_error", True
                break
            aligned = len(triples) == len(out_ids) and all(tid == oid for (_, tid, _), oid in zip(triples, out_ids))
            rec = TurnRecord(turn=turn, seed=seed, input_ids_sha256=sha256_ids(ids), input_len=len(ids),
                             response_ids=out_ids, response_logprobs=[float(lp) for lp, _, _ in triples] if aligned else [],
                             finish_reason=dict(res["meta_info"]["finish_reason"]),
                             cached_tokens=int(res["meta_info"]["cached_tokens"]), ideal_cached_tokens=0,
                             t_submit=self._wall(t), t_done=self._wall(t_done), compile=None, feedback_sha256=None,
                             leak_check="not_run")
            turns.append(rec)
            inputs.append(ids)
            spans.append((name, t, t_done, {"chain_id": cid, "turn": turn, "output_len": len(out_ids)}))
            t = t_done
            if not aligned:
                end = "logprob_error"
                break
            text = self.tok.decode(out_ids, skip_special_tokens=True)
            if "```cpp" not in text:
                end = "no_code_block"
                break
            ok = COMPILE_ERROR_MARKER not in text
            stderr = "" if ok else f"kernel.hip:6:22: error: use of undeclared identifier '{COMPILE_ERROR_MARKER}'"
            rec.compile = CompileResult(ok=ok, rc=0 if ok else 1, seconds=self.fake.compile_s,
                                        errors=[] if ok else [{"cls": "use of undeclared identifier <name>", "line": 6,
                                                               "col": 22, "token": COMPILE_ERROR_MARKER}],
                                        raw_stderr_sha256=sha256_text(stderr))
            spans.append(("compile", t, t + self.fake.compile_s, {"chain_id": cid, "turn": turn, "ok": ok}))
            t += self.fake.compile_s
            if ok:
                end = "compiled"
                break
            if turn == r.max_repair_turns:
                end = "turns_exhausted"
                break
            feedback = ("Your code did not compile (hipcc --offload-arch=gfx950 -c). Errors:\n1. use of undeclared "
                        f"identifier <name> at line 6, column 22: `{COMPILE_ERROR_MARKER}`"
                        + (f" {self.fake.canary_feedback}" if self.fake.canary_feedback else ""))
            rec.feedback_sha256 = sha256_text(feedback)
            if leak_check:
                passed = HashRng([FAKES_SCHEMA, "leak", cid, turn]).uniform("leak") >= self.fake.quarantine_rate
                rec.leak_check = "pass" if passed else "fail"
                if not passed:
                    end = "quarantined"
                    break
            ids_next = (ids + out_ids + ([] if out_ids[-1] == STOP_TOKEN_ID else [STOP_TOKEN_ID])
                        + self.tok.encode(f"\n<|im_start|>user\n{feedback}<|im_end|>\n<|im_start|>assistant\n"))
            if self.fake.max_model_len is not None and len(ids_next) + r.max_new_tokens > self.fake.max_model_len:
                end = "context_limit"
                break
            ids = ids_next
        assert end is not None
        if end == "quarantined":
            turns = [replace(tr, response_ids=[], response_logprobs=[]) for tr in turns]
        record = ChainRecord(chain_id=cid, prompt_id=p.prompt_id, k=k, engine_id=engine_i, policy_version=version,
                             turns=turns, final_turn=len(turns) - 1, end_reason=end)
        cand = sample = None
        if end in SCORED_ENDS and turns:
            final = turns[-1]
            body = final.response_ids[:-1] if final.response_ids and final.response_ids[-1] == STOP_TOKEN_ID else final.response_ids
            cand = Candidate(candidate_id=cid, run_id=self.cfg.run.run_id, iteration=n, prompt_id=p.prompt_id, k=k,
                             policy_version=version, response_text=self.tok.decode(body),
                             response_tokens=len(final.response_ids), max_new_tokens=r.max_new_tokens, task=p.task,
                             check_profile=self.cfg.reward.profile, rollout_compile=final.compile)
            if end in TRAINED_ENDS:
                sample = TrainingSample(chain_id=cid, iteration=n, prompt_id=p.prompt_id, prompt_ids=list(inputs[-1]),
                                        response_ids=list(final.response_ids),
                                        response_logprobs=list(final.response_logprobs))
        return _End(record, cand, sample, t, spans, aborted)

    def run(self, prompts, iteration, policy_version, *, on_candidate, on_chain, deadline):
        if not self.started:
            raise RuntimeError("the fake rollout was not started")
        if self.fake.raise_at_iteration == iteration:
            text = prompts[0].messages[0]["content"] if prompts[0].messages else prompts[0].prompt_id
            raise RuntimeError(f"fake rollout failure while handling: {text}")
        self.runs.append(iteration)
        r = self.cfg.rollout
        seeds = IterationSeeds(self.cfg.orchestrator.run_seed, iteration, [p.prompt_id for p in prompts], r.K,
                               r.max_repair_turns + 1)
        leak_check = self.cfg.rollout.leak_check_adapter != "none"
        t0 = self.clock.now()
        ends = [self._chain(p, k, g % len(self.engines), seeds, iteration, policy_version, t0, deadline, leak_check)
                for g, p in enumerate(prompts) for k in range(r.K)]
        for e in sorted(ends, key=lambda e: (e.t_end, e.record.chain_id)):
            self.clock.advance_to(e.t_end)
            for name, a, b, attrs in e.spans:
                self.spans.record(name, iteration, self._wall(a), self._wall(b), **attrs)
            if e.candidate is not None:
                on_candidate(e.candidate)
            on_chain(ChainEnd(e.record, e.candidate, e.sample))
        ends.sort(key=lambda e: e.record.chain_id)
        outstanding = [e.record.chain_id for e in ends if e.aborted]
        return types.SimpleNamespace(
            iteration=iteration, policy_version=policy_version, chains=[e.record for e in ends],
            candidates=[e.candidate for e in ends if e.candidate is not None],
            samples=[e.sample for e in ends if e.sample is not None],
            metrics={"chains": len(ends), "end_reasons": dict(Counter(e.record.end_reason for e in ends)),
                     "generation_wall_s": max((e.t_end for e in ends), default=t0) - t0},
            stop_reasons=[], deadline_hit=bool(outstanding), outstanding=outstanding)


# --- the reward service -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeRewardConfig:
    seed: int = 0
    latency_s: float = 30.0
    jitter_s: float = 90.0
    unsettled_rate: float = 0.0  # per candidate and attempt: one leg unsettled, so the sample is unscored
    unsettled: frozenset = frozenset()  # candidate ids forced unsettled
    unscored: frozenset = frozenset()  # candidate ids unscored for a candidate reason, with no unsettled leg
    stall: frozenset = frozenset()  # candidate ids whose check stalls for stall_s
    stall_rate: float = 0.0
    stall_s: float = 1.0e4
    faulty_attempts: tuple = (1,)  # attempts on which injected stalls and unsettled legs happen
    workers: int = 4
    constant: float | None = None  # every scored candidate of `constant_prompts` (all, if empty) gets this reward
    constant_prompts: frozenset = frozenset()
    not_applicable_legs: frozenset = frozenset()  # legs the score report counts as not applicable


RUNGS = (-1.0, -0.9, -0.8, -0.7, -0.5, -1.0 / 3.0, -1.0 / 6.0, 0.5, 1.0)
REWARD_FN = {"name": "fake.v11", "version": "11", "code_sha256": sha256_text(f"{FAKES_SCHEMA}/reward")}


def _leg(leg: str, state: str, reason: str | None, worker: str | None, t: float, *, cache_hit: bool = False) -> LegResult:
    return LegResult(leg=leg, state=state, reason=reason, findings=[] if state != "finding" else [{"kind": "fake"}],
                     repeats=1 if state in ("clean", "finding", "fail") else 0, attempts=1, worker_id=worker,
                     bracket=None, cache_hit=cache_hit, t_queued=t, t_start=t, t_end=t)


class FakeReward:
    def __init__(self, clock: VirtualClock, config: FakeRewardConfig | None = None):
        self.clock = clock
        self.c = config or FakeRewardConfig()
        self.cands: dict[str, dict] = {}
        self.accepting = True
        self.resubmits: list[str] = []
        self.drains: list[dict] = []
        self._asan = {"count": 0, "sizes": Counter(), "resolved_at_last_drain": 0}
        self._lock = threading.RLock()

    def _worker(self, cid: str) -> str:
        return f"fake-w{HashRng([FAKES_SCHEMA, 'worker', cid]).randint('w', 0, self.c.workers - 1)}"

    def submit(self, cand: Candidate) -> Future:
        with self._lock:
            st = self.cands.get(cand.candidate_id)
            if st is not None:
                return st["future"]
            st = self.cands[cand.candidate_id] = {"cand": cand, "attempt": 1, "future": Future(), "pending": False}
            if not self.accepting:
                st["future"].set_result(self._unscored(cand, "not_accepting", {}))
                return st["future"]
            self._schedule(cand.candidate_id)
            return st["future"]

    def resubmit(self, cand: Candidate) -> Future:
        with self._lock:
            st = self.cands[cand.candidate_id]
            st["attempt"] += 1
            st["future"] = Future()
            self.resubmits.append(cand.candidate_id)
            self._schedule(cand.candidate_id)
            return st["future"]

    def _schedule(self, cid: str) -> None:
        st = self.cands[cid]
        a = st["attempt"]
        rng = HashRng([FAKES_SCHEMA, self.c.seed, "infra", cid, a])
        faulty = a in self.c.faulty_attempts
        st["stalled"] = faulty and (cid in self.c.stall or rng.uniform("stall") < self.c.stall_rate)
        st["unsettled"] = faulty and (cid in self.c.unsettled or rng.uniform("unsettled") < self.c.unsettled_rate)
        delay = self.c.stall_s if st["stalled"] else self.c.latency_s + self.c.jitter_s * rng.uniform("jitter")
        st["t_ready"], st["delay"], st["pending"], st["token"] = self.clock.now() + delay, delay, True, object()
        token = st["token"]
        self.clock.schedule(st["t_ready"], lambda: self._resolve(cid, token))

    def _resolve(self, cid: str, token: object) -> None:
        with self._lock:
            st = self.cands[cid]
            if st.get("token") is not token or not st["pending"]:
                return
            st["pending"] = False
            if st["stalled"]:  # the stalled worker finally answers: unsettled, never clean
                result = self._unscored(st["cand"], "sanitizer_not_settled",
                                        {"san.asan": _leg("san.asan", "unsettled", "worker_stalled", self._worker(cid),
                                                          self.clock.now())})
            else:
                result = self._scored_or_unsettled(st)
        st["future"].set_result(result)

    def _scored_or_unsettled(self, st: dict) -> RewardResult:
        cand, a = st["cand"], st["attempt"]
        t = self.clock.now()
        verdict = HashRng([FAKES_SCHEMA, self.c.seed, "verdict", sha256_text(cand.response_text), cand.task.task_sha256])
        base = verdict.choice("base", RUNGS)
        legs = {"gate.parse": _leg("gate.parse", "clean", None, "fake-gates", t, cache_hit=a > 1),
                "gate.compile": _leg("gate.compile", "fail" if base == -0.9 else "clean",
                                     "compile_error" if base == -0.9 else None, "fake-gates", t, cache_hit=a > 1)}
        if base != -0.9:
            legs["san.waitcheck"] = _leg("san.waitcheck", "finding" if base < -0.1 and base > -0.6 else "clean", None,
                                         self._worker(cand.candidate_id), t, cache_hit=a > 1)
            legs["san.asan"] = _leg("san.asan", "clean", None, self._worker(cand.candidate_id), t)
        if st["unsettled"]:
            legs["san.asan"] = _leg("san.asan", "unsettled", "adapter_error", self._worker(cand.candidate_id), t)
            return self._unscored(cand, "sanitizer_not_settled", legs)
        if cand.candidate_id in self.c.unscored:  # every leg settled, and still no reward: a retry cannot help
            return self._unscored(cand, "clean_not_proven", legs)
        length = -0.05 * (cand.response_tokens / cand.max_new_tokens)
        broke = -0.5 if (cand.task.parent_healthy and base < 0.5) else 0.0
        if self.c.constant is not None and (not self.c.constant_prompts or cand.prompt_id in self.c.constant_prompts):
            base, broke, length = float(self.c.constant), 0.0, 0.0  # a group whose rewards are all equal
        reward = max(REWARD_MIN, min(REWARD_MAX, base + broke + length))
        return RewardResult(candidate_id=cand.candidate_id, reward=reward, status="scored", reason=None,
                            parts={"base": base, "broke_healthy": broke, "length": length}, reward_fn=dict(REWARD_FN),
                            legs=legs, full_check=True, plan=None, audit=False,
                            cache_key=sha256_json([sha256_text(cand.response_text), cand.task.task_sha256]),
                            timings={"total": st["delay"]})

    def _unscored(self, cand: Candidate, reason: str, legs: dict) -> RewardResult:
        return RewardResult(candidate_id=cand.candidate_id, reward=None, status="unscored", reason=reason, parts=None,
                            reward_fn=dict(REWARD_FN), legs=legs, full_check=False, plan=None, audit=False,
                            cache_key="", timings={})

    def drain(self, deadline: float):
        with self._lock:
            self.accepting = False
            pending = [st for st in self.cands.values() if st["pending"]]
            t_last = max((st["t_ready"] for st in pending), default=self.clock.now())
        t_drain = self.clock.now()
        self.clock.advance_to(min(t_last, deadline))
        outstanding = []
        with self._lock:
            late = sorted(cid for cid, st in self.cands.items() if st["pending"])
            for cid in late:
                st = self.cands[cid]
                st["pending"] = False
                outstanding.append({"candidate_id": cid, "leg": "san.asan", "where": self._worker(cid)})
        for cid in late:
            st = self.cands[cid]
            st["future"].set_result(self._unscored(st["cand"], "sanitizer_not_settled",
                                                   {"san.asan": _leg("san.asan", "unsettled", "timeout",
                                                                     self._worker(cid), self.clock.now())}))
        report = types.SimpleNamespace(outstanding_at_deadline=outstanding, hit_deadline=bool(late), t_drain=t_drain,
                                       t_done=self.clock.now(), resolved=len(self.cands))
        self.drains.append({"deadline": deadline, "t_drain": t_drain, "t_done": report.t_done, "late": len(late)})
        with self._lock:  # ASan requests of up to 14 candidates (plus 2 in-band controls), counted per drain
            done = sum(1 for st in self.cands.values() if st["future"].done())
            n = done - self._asan["resolved_at_last_drain"]
            self._asan["resolved_at_last_drain"] = done
            full, rest = divmod(n, 14)
            self._asan["count"] += full + (1 if rest else 0)
            self._asan["sizes"]["14"] += full
            if rest:
                self._asan["sizes"][str(rest)] += 1
        return report

    def reopen(self) -> None:
        with self._lock:
            self.accepting = True

    def stats(self) -> dict:
        with self._lock:
            done = [st["future"].result() for st in self.cands.values() if st["future"].done()]
            unscored = Counter(r.reason for r in done if r.status != "scored")
            return {"submitted": len(self.cands), "resolved": len(done),
                    "scored": sum(1 for r in done if r.status == "scored"), "unscored_by_reason": dict(unscored),
                    "retries": {"resubmitted": len(self.resubmits)},
                    "asan_requests": {"count": self._asan["count"], "sizes": dict(self._asan["sizes"]),
                                      "max_wait_s": 20.0}}

    def close(self) -> None:
        with self._lock:
            open_ = [st for st in self.cands.values() if not st["future"].done()]
        for st in open_:
            st["future"].set_result(self._unscored(st["cand"], "service_closed", {}))

    def claims_clean(self, result: RewardResult) -> bool:
        """As standin.v11: clean exactly when the base is positive (+1.0 or +0.5)."""
        return result.status == "scored" and isinstance(result.parts, dict) and result.parts.get("base", 0.0) > 0.0

    def not_applicable(self, result: RewardResult, candidate: Candidate | None) -> tuple:
        """The legs named in `not_applicable_legs` (none by default), wherever such a leg has a result."""
        return tuple(sorted(leg for leg in self.c.not_applicable_legs if leg in result.legs))


class FakeServedModel:
    """A score run's model on the fake engines: `prepare` reads the run plan's hashes, `confirm` is FakeSync's round
    trip (perturb, every draw changes; restore, every draw as before). `serves` names what the engines restore to:
    anything but "v0", the weights they started on, fails the check, as engines serving another model would."""

    def __init__(self, cfg: RunConfig, rollout: FakeRollout, *, serves: str = "v0"):
        self.cfg, self.rollout, self.serves = cfg, rollout, serves
        self.prepared: dict | None = None
        self.confirms = 0

    def prepare(self, recorded: dict) -> dict:
        from .score import files_digest

        files = {k.partition("/")[2]: v for k, v in recorded.items() if k.startswith("base/")}
        kind = "merged_dir" if self.cfg.model.adapter_dir is None else "base_plus_adapter"
        self.prepared = {"kind": kind, "dir": self.cfg.model.base_dir, "files": files,
                         "files_sha256": files_digest(files)}
        return self.prepared

    def confirm(self) -> dict:
        self.confirms += 1
        probe = list(range(10_000, 10_064))
        sp = {"temperature": 0.7, "top_p": 0.95, "top_k": -1, "max_new_tokens": 32, "seed": 99,
              "stop_token_ids": [STOP_TOKEN_ID]}
        base = [e.generate(input_ids=probe, sampling_params=sp)["output_ids"] for e in self.rollout.engines]
        self.rollout.set_weights("roundtrip-perturbed")
        moved = [e.generate(input_ids=probe, sampling_params=sp)["output_ids"] for e in self.rollout.engines]
        self.rollout.set_weights(self.serves)
        back = [e.generate(input_ids=probe, sampling_params=sp)["output_ids"] for e in self.rollout.engines]
        proven = all(a != b for a, b in zip(base, moved)) and back == base
        return {"verdict": "PROVEN" if proven else "NOT_PROVEN", "engines": len(self.rollout.engines),
                "files_sha256": self.prepared["files_sha256"] if self.prepared else None}


# --- the trainer -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeTrainerConfig:
    dim: int = 8
    lr: float = 0.05
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    seed: int = 0
    min_scored: int = 2
    gap_exclude: frozenset = frozenset()  # chain ids the step's gap gate excludes
    fail_check_at_step: int | None = None  # the step whose post-step check fails


class FakeTrainerError(RuntimeError):
    pass


def _features(sample: TrainingSample, dim: int) -> list[float]:
    h = sha256_json([sha256_ids(sample.prompt_ids), sha256_ids(sample.response_ids)])
    return [(int(h[2 * j:2 * j + 2], 16) - 127.5) / 127.5 for j in range(dim)]


class FakeTrainer:
    """A deterministic stand-in for WP4's trainer: a `dim`-vector latent, Adam moments, a step count and an RNG
    (a per-step mask), all saved and loaded. The gradient is a fixed function of each trained sample's ids and
    advantage, summed in chain-id order with math.fsum, so it does not depend on the order groups arrive in."""

    def __init__(self, config: FakeTrainerConfig | None = None):
        self.c = config or FakeTrainerConfig()
        self.theta = [0.0] * self.c.dim
        self.m = [0.0] * self.c.dim
        self.v = [0.0] * self.c.dim
        self.steps = 0
        self.rng = random.Random(self.c.seed)
        self._acc: dict | None = None
        self.added: list[str] = []

    def begin(self, iteration: int) -> None:
        self._acc = {}

    def warmup(self) -> dict:
        """`trainer.startup_warmup`: the fake has no kernels to warm, so it trains nothing and keeps its state."""
        return {"counts": {"microbatches": 0, "forward_tokens": 0, "loss_tokens": 0}, "seconds": {},
                "state": "unchanged"}

    def add_group(self, group: GroupInput) -> GroupOutcome:
        if self._acc is None:
            raise FakeTrainerError("add_group before begin")
        scored, excluded = [], {}
        for m in group.members:
            if m.result is None:
                excluded[m.chain_id] = m.record.end_reason
            elif m.result.status != "scored":
                excluded[m.chain_id] = "unscored"
            else:
                scored.append((m.chain_id, float(m.result.reward)))
        self.added.append(group.prompt_id)
        if len(scored) < self.c.min_scored:
            return GroupOutcome(group.prompt_id, len(scored), None, f"fewer_than_{self.c.min_scored}_scored", {}, (),
                                excluded)
        mean = math.fsum(r for _, r in scored) / len(scored)
        adv = {cid: r - mean for cid, r in scored}
        trained = []
        for m in group.members:
            if m.chain_id in adv:
                if m.sample is None:
                    excluded[m.chain_id] = "no_sample"
                    continue
                trained.append(m.chain_id)
                self._acc[m.chain_id] = (adv[m.chain_id], _features(m.sample, self.c.dim), len(m.sample.response_ids))
        return GroupOutcome(group.prompt_id, len(scored), mean, None, adv, tuple(trained), excluded)

    def finish(self) -> dict:
        if not self._acc:
            raise FakeTrainerError("nothing to train")
        ids = sorted(self._acc)
        gated = [cid for cid in ids if cid in self.c.gap_exclude]
        use = [cid for cid in ids if cid not in self.c.gap_exclude]
        N = len(ids)
        g = [math.fsum(self._acc[cid][0] * self._acc[cid][1][j] for cid in use) / N for j in range(self.c.dim)]
        norm = math.sqrt(math.fsum(x * x for x in g))
        if norm == 0.0:
            self._acc = None
            raise FakeTrainerError("the gradient norm is zero")
        b1, b2 = self.c.betas
        mask = [self.rng.random() < 0.9 for _ in range(self.c.dim)]
        self.steps += 1
        for j in range(self.c.dim):
            self.m[j] = b1 * self.m[j] + (1 - b1) * g[j]
            self.v[j] = b2 * self.v[j] + (1 - b2) * g[j] * g[j]
            mhat, vhat = self.m[j] / (1 - b1 ** self.steps), self.v[j] / (1 - b2 ** self.steps)
            if mask[j]:
                self.theta[j] -= self.c.lr * mhat / (math.sqrt(vhat) + self.c.eps)
        if self.c.fail_check_at_step == self.steps:
            self._acc = None
            raise FakeTrainerError(f"post-step check failed at step {self.steps}")
        tokens = sum(self._acc[cid][2] for cid in ids)
        self._acc = None
        return {"step": self.steps, "loss": -math.fsum(g), "grad_norm": norm, "excluded": gated,
                "counts": {"samples": N, "excluded": len(gated), "loss_tokens": tokens, "clipped_tokens": 0},
                "gap": {"median": 0.0, "p99": 0.0}, "checks": {"finite": {"passed": True}}}

    def discard(self) -> None:
        self._acc = None

    def state(self) -> dict:
        version, internal, gauss = self.rng.getstate()
        return {"theta": list(self.theta), "m": list(self.m), "v": list(self.v), "steps": self.steps,
                "rng": [version, list(internal), gauss]}

    def weights(self) -> list[float]:
        return list(self.theta)

    def save_state(self, directory: Path) -> None:
        atomic_write_json(Path(directory) / "trainer_state.json", self.state(), canonical=True)

    def load_state(self, directory: Path) -> None:
        s = json.loads(read_text(Path(directory) / "trainer_state.json"))
        self.theta, self.m, self.v, self.steps = list(s["theta"]), list(s["m"]), list(s["v"]), int(s["steps"])
        version, internal, gauss = s["rng"]
        self.rng.setstate((version, tuple(internal), gauss))

    def state_digest(self) -> str:
        return sha256_json(self.state())


# --- sync and export ----------------------------------------------------------------------------------------------


class FakeSync:
    """Publishes the fake trainer's weights to the fake engines: changed(n) > 0 gives LANDED, a bitwise-equal update
    NOOP. Like WP3's `reload_all`, a reload with requests in flight raises BarrierViolation."""

    def __init__(self, trainer: FakeTrainer, rollout: FakeRollout, clock: VirtualClock, spans: SpanRecorder, *,
                 run_id: str, fail_publish_at: dict | None = None, check_in_flight: bool = True):
        self.trainer, self.rollout, self.clock, self.spans = trainer, rollout, clock, spans
        self.run_id = run_id
        self.fail_publish_at = dict(fail_publish_at or {})
        self.check_in_flight = check_in_flight
        self.calls: list[tuple] = []
        self.version = 0
        self._digest = sha256_json([0.0] * trainer.c.dim)
        self.slot_weights: dict[int, list[float]] = {0: [0.0] * trainer.c.dim}

    def round_trip(self) -> dict:
        probe = list(range(10_000, 10_064))
        sp = {"temperature": 0.7, "top_p": 0.95, "top_k": -1, "max_new_tokens": 32, "seed": 99,
              "stop_token_ids": [STOP_TOKEN_ID]}
        base = [e.generate(input_ids=probe, sampling_params=sp)["output_ids"] for e in self.rollout.engines]
        self.rollout.set_weights("roundtrip-perturbed")
        moved = [e.generate(input_ids=probe, sampling_params=sp)["output_ids"] for e in self.rollout.engines]
        self.rollout.set_weights("v0")
        back = [e.generate(input_ids=probe, sampling_params=sp)["output_ids"] for e in self.rollout.engines]
        proven = all(a != b for a, b in zip(base, moved)) and back == base
        self.calls.append(("round_trip", proven))
        return {"verdict": "PROVEN" if proven else "FAILED"}

    def _load(self, version: int, phases: bool) -> str:
        w = self.trainer.weights()
        digest = sha256_json(w)
        changed = digest != self._digest
        if phases:
            t = self.clock.now()
            for name, d in (("merge", 0.8), ("write", 1.1), ("reload", 1.2), ("landing", 0.9)):
                self.clock.advance_to(t + d)
                self.spans.record(name, version, wall_of(self.spans, self.clock, t), self.spans.now(), version=version)
                t = self.clock.now()
        if changed:
            self.rollout.set_weights(digest[:16])
            self._digest = digest
        self.version = version
        self.slot_weights[version] = list(w)
        for old in [v for v in self.slot_weights if v < version - 1]:
            del self.slot_weights[old]
        return "LANDED" if changed else "NOOP"

    def publish(self, version: int, *, in_flight) -> dict:
        self.calls.append(("publish", version))
        if self.check_in_flight and in_flight() != 0:
            raise BarrierViolation(f"reload with {in_flight()} requests in flight")
        if version in self.fail_publish_at:
            raise RuntimeError(f"landing check failed at version {version}: {self.fail_publish_at[version]}")
        verdict = self._load(version, phases=True)
        return {"verdicts": {e.engine_id: verdict for e in self.rollout.engines},
                "changed": int(verdict == "LANDED"), "coverage": 1.0 if verdict == "LANDED" else 0.0,
                "changed_frac": 1.0 if verdict == "LANDED" else 0.0,
                "seconds": {"merge": 0.8, "write": 1.1, "reload": 1.2, "landing": 0.9}, "slot": self.held_slot()}

    def restore(self, version: int) -> dict:
        self.calls.append(("restore", version))
        self._digest = sha256_json([0.0] * self.trainer.c.dim)  # fresh engines hold the base
        verdict = self._load(version, phases=False)
        return {"verdicts": {e.engine_id: verdict for e in self.rollout.engines}}

    def held_slot(self) -> str:
        return f"/dev/shm/aorlta/{self.run_id}/w{self.version % 2}"


class FakeExporter:
    """Exports the weights the fake sync holds for a version, as WP3's slot holds them for the real export."""

    def __init__(self, sync: FakeSync):
        self.sync = sync
        self.exports: list[int] = []

    def export(self, version: int, merged_dir: Path, adapter_dir: Path) -> dict:
        w = self.sync.slot_weights.get(version)
        if w is None:
            raise RuntimeError(f"no slot holds version {version}")
        atomic_write_json(Path(merged_dir) / "config.json", {"model_type": "fake", "dim": len(w)})
        atomic_write_json(Path(merged_dir) / "model.safetensors.json", {"theta": w}, canonical=True)
        atomic_write_json(Path(adapter_dir) / "adapter_model.json", {"theta": w}, canonical=True)
        atomic_write_json(Path(adapter_dir) / "adapter_config.json", {"r": 16, "lora_alpha": 32})
        self.exports.append(version)
        return {"tensors": 1}


class FakeEvalHook(EvalHook):
    """The eval hook with the fake partner eval answering every request before each poll."""

    def __init__(self, *args, fake_eval_status: str = "done", answer: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_eval = FakeEval(self.inbox, self.outbox, status=fake_eval_status)
        self.answer = answer

    def poll(self, current_iteration: int = 0) -> list[dict]:
        self._dirs()
        if self.answer:
            self.fake_eval.process()
        return super().poll(current_iteration)


# --- the fake run -----------------------------------------------------------------------------------------------


@dataclass
class FakeRunSpec:
    root: str  # absolute; the run directory is <root>/<source>/<run_id>
    run_id: str = "fake-run"
    source: str = "standin"
    iterations: int = 3
    P: int = 4
    K: int = 8
    pool_size: int = 8
    run_seed: int = 20261102
    eval_every: int | None = 2
    min_scored_fraction: float = 0.9
    generation_s: float = 1800.0
    scoring_s: float = 1800.0
    keep_ckpts: int = 2
    max_repair_turns: int = 2
    max_new_tokens: int = 300
    rollout: dict = field(default_factory=dict)  # FakeRolloutConfig overrides
    engine: dict = field(default_factory=dict)  # FakeEngineConfig overrides
    reward: dict = field(default_factory=dict)  # FakeRewardConfig overrides
    trainer: dict = field(default_factory=dict)  # FakeTrainerConfig overrides
    fail_publish_at: dict = field(default_factory=dict)
    eval_status: str = "done"
    eval_answer: bool = True
    canary: str | None = None
    mode: str = "v1"  # "score": the score run (orchestrator/score.py) over the same fakes, without trainer or sync
    score: dict | None = None  # orchestrator.score, in mode score
    served: str = "v0"  # FakeServedModel's `serves`: what the engines restore to in the landing check
    leak_check: bool = False  # run the fake leak check on every repair turn's feedback (rollout.quarantine_rate)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> FakeRunSpec:
        return cls(**data)


def fake_config(spec: FakeRunSpec) -> RunConfig:
    root = Path(spec.root)
    return config_from_dict({
        "run": {"run_id": spec.run_id, "source": spec.source, "root": str(root)},
        "model": {"base_dir": str(root.parent / "fake-base"), "source": spec.source},
        "data": {"prompt_pools": [{"path": str(root.parent / "fake-pool"), "source": spec.source}]},
        "rollout": {"engines": [0, 1], "K": spec.K, "P": spec.P, "max_repair_turns": spec.max_repair_turns,
                    "max_new_tokens": spec.max_new_tokens,
                    "compile_adapter": "fake.compile", "parse_adapter": "standin.parse",
                    "leak_check_adapter": "fake.leak_check" if spec.leak_check else "none",
                    "feedback_template": "standin"},
        "reward": {"profile": "fake", "reward_fn": "standin.v11"},
        "orchestrator": {"run_seed": spec.run_seed, "keep_ckpts": spec.keep_ckpts,
                         "min_scored_fraction": spec.min_scored_fraction,
                         "deadlines": {"generation_s": spec.generation_s, "scoring_s": spec.scoring_s},
                         **({"mode": "score", "score": spec.score} if spec.mode == "score" else
                            {"iterations": spec.iterations, "eval_every": spec.eval_every})},
    })


def ensure_fake_base(cfg: RunConfig) -> Path:
    base = Path(cfg.model.base_dir)
    base.mkdir(parents=True, exist_ok=True)
    for name, data in (("config.json", b'{"model_type": "fake"}\n'), ("model.safetensors", b"\x00" * 64),
                       ("tokenizer.json", b'{"fake": true}\n')):
        if not (base / name).exists():
            (base / name).write_bytes(data)
    return base


@dataclass
class FakeRun:
    spec: FakeRunSpec
    cfg: RunConfig
    clock: VirtualClock
    spans: SpanRecorder
    pool: FakePool
    rollout: FakeRollout
    reward: FakeReward
    trainer: FakeTrainer
    sync: FakeSync
    exporter: FakeExporter
    hook: FakeEvalHook | None
    orchestrator: Orchestrator


def build_fake_run(spec: FakeRunSpec, *, point=None, clock: VirtualClock | None = None) -> FakeRun:
    cfg = fake_config(spec)
    ensure_fake_base(cfg)
    clock = clock or VirtualClock()
    spans = SpanRecorder(monotonic=clock.now, wall=lambda: 1.8e9)
    pool = make_pool(spec.pool_size, source=spec.source, canary=spec.canary)
    engine = dataclasses.replace(FakeRolloutConfig().engine, **spec.engine)
    rollout = FakeRollout(cfg, clock, spans, FakeRolloutConfig(**{"engine": engine, **spec.rollout}))
    reward_kw = dict(spec.reward)
    for k in ("unsettled", "unscored", "stall", "constant_prompts", "not_applicable_legs"):
        if k in reward_kw:
            reward_kw[k] = frozenset(reward_kw[k])
    if "faulty_attempts" in reward_kw:
        reward_kw["faulty_attempts"] = tuple(reward_kw["faulty_attempts"])
    reward = FakeReward(clock, FakeRewardConfig(**reward_kw))
    if spec.mode == "score":
        from .score import Scorer

        served = FakeServedModel(cfg, rollout, serves=spec.served)
        scorer = Scorer(cfg, pool=pool, rollout=rollout, reward=reward, model=served, clock=clock, spans=spans,
                        point=point)
        return FakeRun(spec, cfg, clock, spans, pool, rollout, reward, None, None, None, None, scorer)
    trainer_kw = dict(spec.trainer)
    if "gap_exclude" in trainer_kw:
        trainer_kw["gap_exclude"] = frozenset(trainer_kw["gap_exclude"])
    trainer = FakeTrainer(FakeTrainerConfig(**trainer_kw))
    sync = FakeSync(trainer, rollout, clock, spans, run_id=spec.run_id,
                    fail_publish_at={int(k): v for k, v in spec.fail_publish_at.items()})
    exporter = FakeExporter(sync)
    hook = None
    point = point or (lambda name, iteration: None)
    if spec.eval_every is not None:
        hook = FakeEvalHook(cfg.run_dir / "eval", run_id=spec.run_id, every=spec.eval_every,
                            final_iteration=spec.iterations, exporter=exporter, base_sha256=None, spans=spans,
                            wall=lambda: 1.8e9, point=point, fake_eval_status=spec.eval_status,
                            answer=spec.eval_answer, source=cfg.run.source)
    orch = Orchestrator(cfg, pool=pool, rollout=rollout, reward=reward, trainer=trainer, sync=sync, eval_hook=hook,
                        clock=clock, spans=spans, point=point)
    return FakeRun(spec, cfg, clock, spans, pool, rollout, reward, trainer, sync, exporter, hook, orch)


# --- the subprocess entry point (C6.T1) ----------------------------------------------------------------------------


def kill_point(target: str | None, seen: list):
    want = None
    if target:
        name, it, k = target.rsplit(":", 2)
        want = (name, int(it), int(k))
    counts: Counter = Counter()
    lock = threading.Lock()

    def point(name: str, iteration: int) -> None:
        with lock:
            counts[(name, iteration)] += 1
            c = counts[(name, iteration)]
            seen.append([name, iteration, c])
        if want is not None and (name, iteration, c) == want:
            os.kill(os.getpid(), signal.SIGKILL)

    return point


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aorlta.orchestrator.fakes", description="a fake AORLTA run (C6.T1)")
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--kill-at", default=None, help="NAME:ITERATION:K - SIGKILL at the K-th time NAME is reached")
    ap.add_argument("--points-out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(name)s: %(message)s")
    spec = FakeRunSpec.from_json(json.loads(args.spec.read_text()))
    seen: list = []
    run = build_fake_run(spec, point=kill_point(args.kill_at, seen))
    result = run.orchestrator.run(resume=args.resume)
    if args.points_out is not None:
        args.points_out.write_text(json.dumps(seen))
    print(json.dumps({"status": result["status"], "iterations_completed": result["iterations_completed"],
                      "trainer_digest": run.trainer.state_digest() if run.trainer is not None else None,
                      "stop": result.get("stop")}))
    return 0 if result["status"] == "completed" else 3


if __name__ == "__main__":
    sys.exit(main())
