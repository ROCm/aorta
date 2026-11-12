"""Shared builders for the selector tests (design Section 6.3)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
import time
from pathlib import Path

from aorlta.config import RewardSection
from aorlta.ids import candidate_id
from aorlta.reward.adapters.base import SequentialBatch
from aorlta.reward.adapters.fake import FakeAdapterConfig, FakeCheckAdapter, FakeLegProfile, FakeOutcome
from aorlta.reward.clock import Clock
from aorlta.reward.profiles import load_profile
from aorlta.reward.rewardfn.standin_v11 import StandinV11
from aorlta.reward.service import RewardService
from aorlta.schemas import Candidate, TaskRef
from aorlta.selector.base import CheckPlan
from aorlta.selector.log_only import SelectorHook
from aorlta.spans import SpanRecorder

RUN = "wp6-test"
SEED = 20261102
# For the tests that compare two runs, or that need every plan in time: a slower clock and a longer plan deadline than
# the defaults, so the 60 s gate timeout and the plan deadline are each 3 s of real time, room for a stall of the
# process on a busy host that would otherwise time a leg or a plan out. Those tests check what a run does with a plan,
# not the deadline.
SCALE, PLAN_TIMEOUT_S = 0.05, 60.0
PLANNED = ("ref.native", "ref.edge", "san.consan", "san.waitcheck", "san.asan")
GATE_LEGS = ("gate.parse", "gate.contract", "gate.static", "gate.reference", "gate.compile", "gate.no_artifact",
             "gate.spec_invalid", "gate.static_reject", "gate.no_change_on_failing_parent")

def outcomes(cands, *, early_exit: bool = False) -> FakeAdapterConfig:
    """Fake verdicts for these candidates, the same in every run.

    Without early exit after S1, references always pass and sanitizers find or do not, so every leg
    after a clean compile runs whatever the timing. With it, reference failures and attributed
    sanitizer faults decide some candidates in S2. Every ninth candidate's waitcheck fails to settle.
    That is injected per candidate, not as an error rate: the fake applies a leg's error rate to its
    trusted controls too, and a dirty control bracket makes which legs are retried depend on timing."""
    per_leg = {
        "gate.compile": FakeLegProfile(fail_rate=0.15),
        "san.consan": FakeLegProfile(finding_rate=0.25),
        "san.waitcheck": FakeLegProfile(finding_rate=0.15),
        "san.asan": FakeLegProfile(finding_rate=0.2),
    }
    if early_exit:
        per_leg.update({
            "ref.native": FakeLegProfile(fail_rate=0.1, fail_reasons=("wrong_output", "crash")),
            "ref.edge": FakeLegProfile(fail_rate=0.05, fail_reasons=("edge_mismatch",)),
            "san.consan": FakeLegProfile(finding_rate=0.25, fail_rate=0.03, fail_reasons=("fault",)),
        })
    overrides = {(c.candidate_id, "san.waitcheck"): FakeOutcome("error", "adapter_error") for c in cands[4::9]}
    return FakeAdapterConfig(seed=6 if early_exit else 5, per_leg=per_leg, overrides=overrides)


def restated_draw(seed: int, cid: str) -> float:
    """The audit draw written out again: sha256 of the canonical JSON of [schema, seed, candidate id],
    then of [that hex, "audit", 0]; the first 8 bytes, little-endian, over 2^64."""
    key = hashlib.sha256(json.dumps(["aorlta.selector.audit/1", seed, cid], separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()
    digest = hashlib.sha256(json.dumps([key, "audit", 0], separators=(",", ":")).encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2.0**64


def restated_coin(seed: int, cid: str, fraction: float) -> bool:
    return restated_draw(seed, cid) < fraction


def cand(i: int, *, iteration: int = 1, text: str | None = None, tokens: int | None = None, max_new: int = 4096,
         parent_healthy: bool | None = None, reference_kind: str = "frozen", profile: str = "fake") -> Candidate:
    """Candidate i: prompt i // 8, chain i % 8, so every 8 consecutive candidates form a group."""
    pid, k = f"standin:p{i // 8}", i % 8
    return Candidate(candidate_id(RUN, iteration, pid, k), RUN, iteration, pid, k, f"{RUN}:v{iteration - 1}",
                     text if text is not None else f"```cpp\n// kernel {iteration}.{i}\n```",
                     tokens if tokens is not None else 300 + (37 * i) % 3000, max_new,
                     TaskRef("contract-1", None, parent_healthy, reference_kind, "1" * 64), profile, None)


def hook(mode: str = "log_only", planner=None, **kw) -> SelectorHook:
    kw.setdefault("audit_seed", SEED)
    if mode == "plan":
        kw.setdefault("allow_plan_mode", True)
    return SelectorHook(mode=mode, planner=planner, **kw)


@dataclasses.dataclass
class Rig:
    service: RewardService
    clock: Clock
    spans: SpanRecorder
    selector: SelectorHook | None

    def close(self) -> None:
        self.service.close()

    def run(self, cands, *, drain_s: float = 7200.0, timeout: float = 120.0):
        """Submit every candidate, drain (the generation barrier), and return the results in order."""
        futures = [self.service.submit(c) for c in cands]
        self.service.drain(self.clock.now() + drain_s)
        deadline = time.monotonic() + timeout
        return [f.result(timeout=max(0.1, deadline - time.monotonic())) for f in futures]


def rig(tmp_path: Path | None = None, *, selector: SelectorHook | None = None, clock: Clock | None = None,
        scale: float = 0.002, adapter: FakeAdapterConfig | None = None, reward: RewardSection | None = None,
        cache: bool = True, inband: bool = True, profile="fake", rewardfn=None) -> Rig:
    clock = clock or Clock(scale)
    spans = SpanRecorder(monotonic=clock.now, wall=clock.wall)
    base_cfg = dataclasses.replace(adapter or FakeAdapterConfig(), time_scale=clock.scale)
    asan_cfg = dataclasses.replace(
        base_cfg, legs=("san.asan",), per_leg={k: v for k, v in base_cfg.per_leg.items() if k == "san.asan"},
        overrides={k: v for k, v in base_cfg.overrides.items() if k[1] == "san.asan"})
    section = reward or RewardSection(profile="fake", reward_fn="standin.v11", sim_workers=8)
    if not cache:
        section = dataclasses.replace(section, cache=dataclasses.replace(section.cache, enabled=False))
    svc = RewardService(
        section, load_profile(profile) if isinstance(profile, str) else profile, rewardfn or StandinV11(),
        {"fake": lambda w: FakeCheckAdapter(base_cfg, clock=clock.wall),
         "fake-asan": lambda w: SequentialBatch(FakeCheckAdapter(asan_cfg, clock=clock.wall), max_packages=16,
                                                inband_controls=inband)},
        cache_dir=None if tmp_path is None else tmp_path / "cache",
        work_dir=None if tmp_path is None else tmp_path / "work", clock=clock, spans=spans, selector=selector)
    return Rig(svc, clock, spans, selector)


def leg_view(result) -> dict:
    """What a leg's result says, without its timing: for comparing two runs of the same candidates."""
    return {name: (r.state, r.reason, r.repeats, r.attempts, len(r.findings)) for name, r in result.legs.items()}


def reward_view(result) -> tuple:
    return (result.reward, result.status, result.reason, result.parts)


# --- planners for failure modes ----------------------------------------------------------------------


class Raising:
    version = "test.raising/1"
    CANARY = "kernel-source-canary-in-an-exception"

    def plan(self, req):
        raise RuntimeError(f"boom: {self.CANARY}")

    def score(self, bug, candidates):
        return []


class Answers:
    """Answers with whatever `make(req)` returns (a malformed plan, a plan with an unknown leg, ...)."""

    version = "test.answers/1"

    def __init__(self, make):
        self.make = make

    def plan(self, req):
        return self.make(req)

    def score(self, bug, candidates):
        return []


class Sleeper:
    """Wraps a planner and answers `delay_s` seconds of the service clock late."""

    def __init__(self, inner, clock: Clock, delay_s: float):
        self.inner, self.clock, self.delay_s = inner, clock, delay_s
        self.version = inner.version

    def plan(self, req):
        self.clock.sleep(self.delay_s)
        return self.inner.plan(req)

    def score(self, bug, candidates):
        return self.inner.score(bug, candidates)


class Gated:
    """Wraps a planner and answers only once `release` is set (real time, at most 60 s)."""

    def __init__(self, inner):
        self.inner = inner
        self.version = inner.version
        self.release = threading.Event()
        self.calls = 0  # requests the hook's threads handed over, before their deadline
        self._lock = threading.Lock()

    def plan(self, req):
        with self._lock:
            self.calls += 1
        self.release.wait(60)
        return self.inner.plan(req)

    def score(self, bug, candidates):
        return self.inner.score(bug, candidates)


class Hanging:
    """Never answers until released."""

    version = "test.hanging/1"

    def __init__(self):
        self.release = threading.Event()

    def plan(self, req):
        self.release.wait(60)
        raise RuntimeError("released")

    def score(self, bug, candidates):
        return []


class Recording:
    """Records, at plan time, the audit coin the hook already holds for the candidate."""

    def __init__(self, inner):
        self.inner = inner
        self.version = inner.version
        self.hook: SelectorHook | None = None
        self.seen: dict[str, object] = {}
        self.requests: dict[str, object] = {}

    def plan(self, req):
        self.seen[req.candidate_id] = self.hook.coin(req.candidate_id)
        self.requests[req.candidate_id] = req
        return self.inner.plan(req)

    def score(self, bug, candidates):
        return self.inner.score(bug, candidates)


def plan_with(req, decisions: dict) -> CheckPlan:
    return CheckPlan("test.scripted/1", f"scripted:{req.candidate_id}",
                     {leg: {"decision": decisions.get(leg, "run"), "p_finding": None, "score": None}
                      for leg in req.available_legs}, None)
