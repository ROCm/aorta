"""A deterministic fake check adapter (design Section 6.2, `fake.*`; Section 8, WP0).

Latency, finding, failure and stall injection from a seed, with two kinds of draw kept apart, as the
doc keeps the two kinds of failure apart:

- The verdict (clean, finding, or fail with a reason) is drawn from the job's content: seed, leg,
  source and task hashes, and leg settings. The same content gets the same verdict on every attempt,
  worker and candidate, as from a deterministic checker; that is what the cache relies on.
- Infrastructure events (an adapter error, a stall) and latency are drawn per attempt: from the
  seed, the leg, the candidate (or, for a control, the job), the source hash and the attempt number.
  A retry gets a fresh draw and may succeed. Worker ids are not used, because which worker runs a
  job is not reproducible.

`fail` is a result the checker attributes to the candidate (Section 6.2, "Two kinds of timeout");
an injected adapter error or stall is `unsettled`, never clean. Like every CheckAdapter, `run`
never raises: anything unexpected, including a malformed job, becomes `unsettled` with reason
"adapter_error".
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ...ids import HashRng, sha256_text
from ...schemas import SETTLED_STATES, ControlSpec, LegJob, LegResult

log = logging.getLogger(__name__)

FAKE_ADAPTER_SCHEMA = "aorlta.fake_adapter/1"
GATES = ("parse", "contract", "static", "reference", "compile", "no_artifact", "spec_invalid", "static_reject",
         "no_change_on_failing_parent")
ALL_LEGS = tuple(f"gate.{g}" for g in GATES) + ("ref.native", "ref.edge", "san.consan", "san.waitcheck", "san.asan")
LEG_RE = re.compile(r"(gate|ref|san)\.[a-z0-9_]+")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
Resource = Literal["local_cpu", "sim_cpu", "external_gpu", "own_gpu"]


def default_fail_reasons(leg: str) -> tuple[str, ...]:
    """Reasons a `fail` draw picks from when a profile names none (Appendix B's ladder)."""
    if leg == "gate.compile":
        return ("compile_error",)
    if leg.startswith("gate."):
        return (leg.removeprefix("gate."),)
    if leg == "ref.native":
        return ("wrong_output", "crash", "timeout")
    if leg == "ref.edge":
        return ("edge_mismatch", "fault", "timeout")
    if leg.startswith("san."):
        return ("crash", "fault", "timeout")
    raise ValueError(f"not a leg name: {leg!r}")


def _check_rate(name: str, value: Any) -> None:
    if not (isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _check_seconds(name: str, value: Any) -> None:
    if not (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0):
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")


@dataclass(frozen=True)
class FakeLegProfile:
    """Per-leg knobs. finding + fail and error + stall are each at most 1; clean is the rest."""

    finding_rate: float = 0.0  # san.* legs only
    fail_rate: float = 0.0
    error_rate: float = 0.0
    stall_rate: float = 0.0
    latency_s: float = 0.0  # simulated seconds per run: latency_s + U[0, jitter_s)
    jitter_s: float = 0.0
    fail_reasons: tuple[str, ...] = ()  # empty: the leg's defaults (default_fail_reasons)

    def __post_init__(self) -> None:
        for name in ("finding_rate", "fail_rate", "error_rate", "stall_rate"):
            _check_rate(name, getattr(self, name))
        if self.finding_rate + self.fail_rate > 1.0:
            raise ValueError("finding_rate + fail_rate must be <= 1")
        if self.error_rate + self.stall_rate > 1.0:
            raise ValueError("error_rate + stall_rate must be <= 1")
        _check_seconds("latency_s", self.latency_s)
        _check_seconds("jitter_s", self.jitter_s)
        if not all(isinstance(r, str) and r for r in self.fail_reasons):
            raise ValueError("fail_reasons must be non-empty strings")


@dataclass(frozen=True)
class FakeOutcome:
    """A forced outcome for one job: clean, finding, fail (with a reason), error (unsettled) or stall.
    `latency_s`, if set, replaces the job's drawn latency (a slow but live job, unlike a stall)."""

    state: Literal["clean", "finding", "fail", "error", "stall"]
    reason: str | None = None
    latency_s: float | None = None

    def __post_init__(self) -> None:
        if self.state not in ("clean", "finding", "fail", "error", "stall"):
            raise ValueError(f"unknown fake outcome {self.state!r}")
        if self.latency_s is not None:
            _check_seconds("latency_s", self.latency_s)


@dataclass(frozen=True)
class FakeAdapterConfig:
    seed: int = 0
    name: str = "fake.checker"
    version: str = "1"
    legs: tuple[str, ...] = ALL_LEGS
    resource: Resource = "local_cpu"
    default: FakeLegProfile = field(default_factory=FakeLegProfile)
    per_leg: Mapping[str, FakeLegProfile] = field(default_factory=dict)
    control_dirty_rate: float = 0.0  # share of trusted-control runs that come back dirty
    stall_s: float = 1.0e4  # simulated length of a stall (C2.T8 uses 10^4 s)
    time_scale: float = 0.0  # real seconds slept per simulated second; 0 never sleeps
    repeats: int = 1  # settled repeats reported on a settled result
    # (key, leg) or (key, leg, attempt) -> outcome; key matches a job's candidate_id, control_id or job_id.
    # An attempt-specific entry wins over the generic one.
    overrides: Mapping[tuple, FakeOutcome] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (isinstance(self.seed, int) and not isinstance(self.seed, bool)):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        if not (isinstance(self.name, str) and re.fullmatch(r"fake(\.[A-Za-z0-9_.-]+)?", self.name)):
            raise ValueError(f"a fake adapter's name is 'fake' or 'fake.<name>', got {self.name!r}")
        if not (isinstance(self.version, str) and self.version):
            raise ValueError("version must be a non-empty str")
        legs = tuple(self.legs)
        if not legs or len(set(legs)) != len(legs) or not all(isinstance(g, str) and LEG_RE.fullmatch(g) for g in legs):
            raise ValueError(f"legs must be unique gate.*, ref.* or san.* names, got {legs!r}")
        object.__setattr__(self, "legs", legs)
        if self.resource not in ("local_cpu", "sim_cpu", "external_gpu", "own_gpu"):
            raise ValueError(f"unknown resource {self.resource!r}")
        extra = sorted(set(self.per_leg) - set(legs))
        if extra:
            raise ValueError(f"per_leg names legs this adapter does not run: {extra}")
        for leg in legs:
            profile = self.per_leg.get(leg, self.default)
            if profile.finding_rate > 0 and not leg.startswith("san."):
                raise ValueError(f"{leg}: only sanitizer legs report findings")
        _check_rate("control_dirty_rate", self.control_dirty_rate)
        _check_seconds("stall_s", self.stall_s)
        _check_seconds("time_scale", self.time_scale)
        if not (isinstance(self.repeats, int) and not isinstance(self.repeats, bool) and self.repeats >= 1):
            raise ValueError(f"repeats must be an int >= 1, got {self.repeats!r}")
        for key, outcome in self.overrides.items():
            ok = isinstance(key, tuple) and len(key) in (2, 3) and isinstance(key[0], str) and key[1] in legs
            if ok and len(key) == 3:
                ok = isinstance(key[2], int) and not isinstance(key[2], bool) and key[2] >= 1
            if not ok:
                raise ValueError(f"override keys are (key, leg) or (key, leg, attempt) for a leg of this adapter, got {key!r}")
            if not isinstance(outcome, FakeOutcome):
                raise ValueError(f"override {key!r} must map to a FakeOutcome")
            if outcome.state == "finding" and not key[1].startswith("san."):
                raise ValueError(f"override {key!r}: only sanitizer legs report findings")
        object.__setattr__(self, "per_leg", types.MappingProxyType(dict(self.per_leg)))
        object.__setattr__(self, "overrides", types.MappingProxyType(dict(self.overrides)))


@dataclass(frozen=True)
class FakePlan:
    """What `run` will do for a job, decided before it runs (so tests can predict it)."""

    state: str  # the LegResult state
    reason: str | None
    findings: tuple[dict, ...]
    sleep_s: float  # simulated seconds the run takes
    stalled: bool
    origin: str  # "override" | "rates" | "control" | "invalid"


def _job_problem(job: Any, legs: tuple[str, ...]) -> str | None:
    if not isinstance(job, LegJob):
        return f"not a LegJob: {type(job).__name__}"
    if job.leg not in legs:
        return f"leg {job.leg!r} is not run by this adapter"
    if (job.candidate_id is None) == (job.control_id is None):
        return "exactly one of candidate_id and control_id must be set"
    if not (isinstance(job.attempt, int) and not isinstance(job.attempt, bool) and job.attempt >= 1):
        return f"attempt must be an int >= 1, got {job.attempt!r}"
    for name in ("source_sha256", "task_sha256"):
        if not (isinstance(getattr(job, name), str) and SHA256_RE.fullmatch(getattr(job, name))):
            return f"{name} is not a sha256 hex digest"
    return None


class FakeCheckAdapter:
    """Implements the CheckAdapter protocol of Section 6.2 (name, version, legs, resource, run,
    controls, health)."""

    def __init__(
        self,
        config: FakeAdapterConfig | None = None,
        *,
        sleep: Callable[[float], Any] | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.config = config if config is not None else FakeAdapterConfig()
        self._stop = threading.Event()
        self._sleep = sleep if sleep is not None else self._stop.wait
        self._clock = clock
        self._lock = threading.Lock()
        self._runs = 0
        self._stalling = 0

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def legs(self) -> tuple[str, ...]:
        return self.config.legs

    @property
    def resource(self) -> str:
        return self.config.resource

    def _profile(self, leg: str) -> FakeLegProfile:
        return self.config.per_leg.get(leg, self.config.default)

    def _override(self, job: LegJob) -> FakeOutcome | None:
        keys = [k for k in (job.candidate_id, job.control_id, job.job_id) if k is not None]
        for key in keys:
            hit = self.config.overrides.get((key, job.leg, job.attempt))
            if hit is not None:
                return hit
        for key in keys:
            hit = self.config.overrides.get((key, job.leg))
            if hit is not None:
                return hit
        return None

    def plan(self, job: LegJob) -> FakePlan:
        problem = _job_problem(job, self.legs)
        if problem is not None:
            return FakePlan("unsettled", "adapter_error", (), 0.0, False, "invalid")
        cfg, profile = self.config, self._profile(job.leg)
        who = job.candidate_id if job.candidate_id is not None else job.job_id
        infra = HashRng([FAKE_ADAPTER_SCHEMA, cfg.seed, "infra", job.leg, who, job.source_sha256, job.attempt])
        latency = profile.latency_s + profile.jitter_s * infra.uniform("jitter")
        verdict = HashRng([FAKE_ADAPTER_SCHEMA, cfg.seed, "verdict", job.leg, job.source_sha256, job.task_sha256,
                           dict(job.leg_settings)])
        reasons = profile.fail_reasons or default_fail_reasons(job.leg)
        finding = ({"kind": "fake_finding", "leg": job.leg, "id": verdict.hex("finding", 12)},)

        forced = self._override(job)
        if forced is not None:
            if forced.latency_s is not None:
                latency = forced.latency_s
            if forced.state == "stall":
                return FakePlan("unsettled", forced.reason or "worker_stalled", (), cfg.stall_s, True, "override")
            if forced.state == "error":
                return FakePlan("unsettled", forced.reason or "adapter_error", (), latency, False, "override")
            if forced.state == "fail":
                return FakePlan("fail", forced.reason or reasons[0], (), latency, False, "override")
            if forced.state == "finding":
                return FakePlan("finding", forced.reason, finding, latency, False, "override")
            return FakePlan("clean", None, (), latency, False, "override")

        u = infra.uniform("infra")
        if u < profile.stall_rate:
            return FakePlan("unsettled", "worker_stalled", (), cfg.stall_s, True, "rates")
        if u < profile.stall_rate + profile.error_rate:
            return FakePlan("unsettled", "adapter_error", (), latency, False, "rates")

        if job.control_id is not None:
            control = HashRng([FAKE_ADAPTER_SCHEMA, cfg.seed, "control", job.job_id, job.attempt])
            if control.uniform("dirty") < cfg.control_dirty_rate:
                dirty = ({"kind": "fake_control_dirty", "leg": job.leg, "control_id": job.control_id},)
                state = "finding" if job.leg.startswith("san.") else "fail"
                return FakePlan(state, "control_dirty", dirty, latency, False, "control")
            return FakePlan("clean", None, (), latency, False, "control")

        v = verdict.uniform("verdict")
        if v < profile.finding_rate:
            return FakePlan("finding", None, finding, latency, False, "rates")
        if v < profile.finding_rate + profile.fail_rate:
            return FakePlan("fail", verdict.choice("reason", reasons), (), latency, False, "rates")
        return FakePlan("clean", None, (), latency, False, "rates")

    def run(self, job: LegJob) -> LegResult:
        t_start = self._clock()
        try:
            if self._stop.is_set():
                raise RuntimeError("adapter is closed")
            plan = self.plan(job)
            with self._lock:
                self._runs += 1
                self._stalling += plan.stalled
            try:
                if self.config.time_scale > 0 and plan.sleep_s > 0:
                    self._sleep(plan.sleep_s * self.config.time_scale)
            finally:
                with self._lock:
                    self._stalling -= plan.stalled
            settled = plan.state in SETTLED_STATES
            return LegResult(
                leg=job.leg,
                state=plan.state,
                reason=plan.reason,
                findings=[dict(f) for f in plan.findings],
                repeats=self.config.repeats if settled else 0,
                attempts=job.attempt,
                worker_id=job.worker_id,
                bracket=None,
                cache_hit=False,
                t_queued=job.t_queued if job.t_queued is not None else t_start,
                t_start=t_start,
                t_end=self._clock(),
            )
        except Exception as exc:  # the protocol: errors become state="unsettled", never an exception
            log.warning("fake adapter %s: %s on a job, reported as unsettled", self.name, type(exc).__name__)
            leg = getattr(job, "leg", None)
            attempt = getattr(job, "attempt", 0)
            return LegResult(
                leg=leg if isinstance(leg, str) else "unknown",
                state="unsettled",
                reason="adapter_error",
                findings=[],
                repeats=0,
                attempts=attempt if isinstance(attempt, int) and not isinstance(attempt, bool) else 0,
                worker_id=None,
                bracket=None,
                cache_hit=False,
                t_queued=t_start,
                t_start=t_start,
                t_end=self._clock(),
            )

    def controls(self) -> list[ControlSpec]:
        seed = self.config.seed
        return [
            ControlSpec(
                control_id=f"{self.name}/control/{leg}",
                leg=leg,
                source_sha256=sha256_text(f"{FAKE_ADAPTER_SCHEMA}/control-source/{seed}/{leg}"),
                task_sha256=sha256_text(f"{FAKE_ADAPTER_SCHEMA}/control-task/{seed}/{leg}"),
            )
            for leg in self.legs
        ]

    def health(self) -> dict:
        with self._lock:
            return {"name": self.name, "version": self.version, "resource": self.resource, "legs": list(self.legs),
                    "closed": self._stop.is_set(), "runs": self._runs, "stalls_in_progress": self._stalling}

    def close(self) -> None:
        """Interrupt any stall in progress (with the default sleep); later runs return unsettled."""
        self._stop.set()
