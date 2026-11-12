"""Shared builders for the reward-service tests."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

from aorlta.config import RewardSection
from aorlta.ids import candidate_id, sha256_text
from aorlta.reward.adapters.base import SequentialBatch
from aorlta.reward.adapters.fake import FakeAdapterConfig, FakeCheckAdapter, FakeLegProfile
from aorlta.reward.clock import Clock
from aorlta.reward.profiles import load_profile
from aorlta.reward.rewardfn.standin_v11 import StandinV11
from aorlta.reward.service import RewardService, ServiceOptions
from aorlta.schemas import Candidate, LegResult, TaskRef
from aorlta.spans import SpanRecorder

RUN = "wp5-test"
TASK_SHA = sha256_text("task")
SAN_LEGS = ("san.consan", "san.waitcheck", "san.asan")
SIM_LEGS = ("ref.native", "ref.edge", "san.consan", "san.waitcheck")


def task(*, parent_healthy=None, reference_kind="frozen", task_sha=TASK_SHA) -> TaskRef:
    return TaskRef("contract-1", None, parent_healthy, reference_kind, task_sha)


def cand(i: int, *, text: str | None = None, tokens: int = 1024, max_new: int = 4096, profile: str = "fake",
         iteration: int = 1, **task_kw) -> Candidate:
    pid = f"standin:k{i}"
    return Candidate(
        candidate_id=candidate_id(RUN, iteration, pid, 0), run_id=RUN, iteration=iteration, prompt_id=pid, k=0,
        policy_version=f"{RUN}:v0", response_text=text if text is not None else f"```cpp\n// kernel {i}\n```",
        response_tokens=tokens, max_new_tokens=max_new, task=task(**task_kw), check_profile=profile,
        rollout_compile=None)


def reward_section(**kw) -> RewardSection:
    base = dict(profile="fake", reward_fn="standin.v11")
    base.update(kw)
    return RewardSection(**base)


@dataclasses.dataclass
class Rig:
    service: RewardService
    clock: Clock
    spans: SpanRecorder
    adapters: dict  # worker id -> adapter

    def close(self):
        self.service.close()


def fake_rig(tmp_path: Path | None = None, *, scale: float = 0.01, adapter: FakeAdapterConfig | None = None,
             asan_adapter: FakeAdapterConfig | None = None, reward: RewardSection | None = None,
             options: ServiceOptions | None = None, cache: bool = True, profile="fake",
             inband: bool = True, rewardfn=None) -> Rig:
    clock = Clock(scale)
    spans = SpanRecorder(monotonic=clock.now, wall=clock.wall)
    adapters: dict = {}
    base_cfg = adapter or FakeAdapterConfig()
    asan_cfg = asan_adapter or dataclasses.replace(
        base_cfg, legs=("san.asan",), per_leg={k: v for k, v in base_cfg.per_leg.items() if k == "san.asan"},
        overrides={k: v for k, v in base_cfg.overrides.items() if k[1] == "san.asan"})

    def make(cfg):
        cfg = dataclasses.replace(cfg, time_scale=clock.scale)

        def factory(worker_id: str):
            a = FakeCheckAdapter(cfg, clock=clock.wall)
            adapters[worker_id] = a
            return a
        return factory

    asan_inner = make(asan_cfg)

    def asan_factory(worker_id: str):
        return SequentialBatch(asan_inner(worker_id), max_packages=16, inband_controls=inband)

    section = reward or reward_section(sim_workers=4)
    if not cache:
        section = dataclasses.replace(section, cache=dataclasses.replace(section.cache, enabled=False))
    prof = load_profile(profile) if isinstance(profile, str) else profile
    svc = RewardService(section, prof, rewardfn or StandinV11(), {"fake": make(base_cfg), "fake-asan": asan_factory},
                        cache_dir=None if tmp_path is None else tmp_path / "cache",
                        work_dir=None if tmp_path is None else tmp_path / "work", clock=clock, spans=spans,
                        options=options)
    return Rig(svc, clock, spans, adapters)


def wait_all(futures, timeout: float = 120.0):
    deadline = time.monotonic() + timeout
    out = []
    for f in futures:
        out.append(f.result(timeout=max(0.1, deadline - time.monotonic())))
    return out


def leg(name: str, state: str = "clean", reason: str | None = None, *, repeats: int = 1, bracket=None,
        findings=None) -> LegResult:
    settled = state in ("clean", "finding", "fail")
    if state == "finding" and findings is None:
        findings = [{"kind": "fake_finding"}]
    return LegResult(name, state, reason, list(findings or []), repeats if settled else 0, 1, "w", bracket, False,
                     0.0, 0.0, 1.0)
