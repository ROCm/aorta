"""The CheckAdapter protocol (design Section 6.2, Plug-in adapters) and what every
adapter shares: building results, running one job without ever raising, a run's own names, a killable
subprocess runner, and the mapping from aorta's sanitizer verdicts to leg states.

The one rule every adapter keeps: `run` never raises. Anything unexpected becomes
`state="unsettled"`, never clean. Only an adapter may report a timeout attributed to the candidate
(`state="fail", reason="timeout"`); the service and its watchdog only ever report
`unsettled(timeout)` (Section 6.2, "Two kinds of timeout").
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ...ids import sha256_text
from ...schemas import SETTLED_STATES, ControlSpec, LegJob, LegResult

log = logging.getLogger(__name__)

RESOURCES = ("local_cpu", "sim_cpu", "external_gpu", "own_gpu")
# The kind of a build fact a clean `gate.compile` may carry among its findings (adapters/standin_hipcc.py): the
# launched kernels' LDS, read from their descriptors. It is a record of the build, not a finding.
TRUSTED_BUILD = "trusted_build"


@runtime_checkable
class CheckAdapter(Protocol):
    name: str
    version: str
    legs: tuple[str, ...]
    resource: str  # "local_cpu" | "sim_cpu" | "external_gpu" | "own_gpu"

    def run(self, job: LegJob) -> LegResult: ...  # never raises: errors become state="unsettled"

    def controls(self) -> list[ControlSpec]: ...

    def health(self) -> dict: ...  # must return promptly; it is the worker's heartbeat probe


class BatchCheckAdapter(CheckAdapter, Protocol):
    """An endpoint that checks several packages per request (the ASan endpoint, Section 6.2)."""

    max_packages: int
    inband_controls: bool  # controls travel as the first and last packages of a request

    def run_batch(self, jobs: Sequence[LegJob]) -> list[LegResult]: ...


def responsive(health: Any) -> bool:
    """Whether an adapter's health answer counts as a heartbeat. An adapter is silent when it says
    so (`responsive: False`) or reports a stall in progress (the fake's `stalls_in_progress`)."""
    if not isinstance(health, Mapping):
        return False
    if health.get("responsive") is False:
        return False
    stalls = health.get("stalls_in_progress", 0)
    return not (isinstance(stalls, int) and stalls > 0)


def make_result(job: LegJob, state: str, reason: str | None = None, *, findings: Sequence[dict] = (),
                repeats: int | None = None, t_start: float, t_end: float) -> LegResult:
    settled = state in SETTLED_STATES
    return LegResult(
        leg=job.leg,
        state=state,
        reason=reason,
        findings=[dict(f) for f in findings],
        repeats=(1 if repeats is None else repeats) if settled else 0,
        attempts=job.attempt,
        worker_id=job.worker_id,
        bracket=None,
        cache_hit=False,
        t_queued=job.t_queued if job.t_queued is not None else t_start,
        t_start=t_start,
        t_end=t_end,
    )


def unsettled(job: LegJob, reason: str, *, t_start: float, t_end: float) -> LegResult:
    return make_result(job, "unsettled", reason, t_start=t_start, t_end=t_end)


def run_tag(job: LegJob) -> str:
    """A name for one run of a job, unique across workers and processes. A job's output directory is content-addressed,
    so jobs with identical content share it, and a retry can overlap the retired worker's run of the same job."""
    return f"a{job.attempt}-{os.getpid()}-{secrets.token_hex(4)}"


def run_dir(job: LegJob) -> Path:
    """A new directory under the job's output directory, for this run's files alone (`run_tag`)."""
    d = Path(job.output_dir) / run_tag(job)
    d.mkdir(parents=True)
    return d


def safe_run(adapter: Any, job: LegJob, clock: Callable[[], float] = time.time) -> LegResult:
    """`adapter.run(job)`, checked. An exception, a non-LegResult, a result for another leg or a state
    outside the schema becomes `unsettled(adapter_error)`."""
    t0 = clock()
    try:
        result = adapter.run(job)
    except Exception as exc:  # the protocol forbids it; fail closed anyway
        log.warning("adapter %s raised %s on %s", getattr(adapter, "name", "?"), type(exc).__name__, job.leg)
        return unsettled(job, "adapter_error", t_start=t0, t_end=clock())
    return checked_result(job, result, t0, clock())


def inconsistency(result: LegResult) -> str | None:
    """Why a leg's state contradicts its findings, or None: a finding needs findings, and a clean leg has none
    (a trusted build fact is not a finding)."""
    if result.state == "finding" and not result.findings:
        return "finding_without_findings"
    if result.state == "clean" and any(not (isinstance(f, dict) and f.get("kind") == TRUSTED_BUILD)
                                       for f in result.findings):
        return "clean_with_findings"
    return None


def checked_result(job: LegJob, result: Any, t0: float, t1: float) -> LegResult:
    ok = (isinstance(result, LegResult) and result.leg == job.leg
          and result.state in ("clean", "finding", "fail", "unsettled")
          and isinstance(result.findings, list))
    if not ok:
        return unsettled(job, "adapter_error", t_start=t0, t_end=t1)
    why = inconsistency(result)
    return unsettled(job, why, t_start=t0, t_end=t1) if why else result


def safe_run_batch(adapter: Any, jobs: Sequence[LegJob], clock: Callable[[], float] = time.time) -> list[LegResult]:
    """`adapter.run_batch(jobs)`, checked: one result per job, in order, or every job unsettled."""
    t0 = clock()
    try:
        results = adapter.run_batch(list(jobs))
    except Exception as exc:
        log.warning("adapter %s raised %s on a batch", getattr(adapter, "name", "?"), type(exc).__name__)
        results = None
    t1 = clock()
    if not isinstance(results, list) or len(results) != len(jobs):
        return [unsettled(j, "adapter_error", t_start=t0, t_end=t1) for j in jobs]
    return [checked_result(j, r, t0, t1) for j, r in zip(jobs, results)]


class SequentialBatch:
    """A batch endpoint over a one-job adapter: a request runs its packages one after another. Used
    for the fake ASan endpoint and the stand-in ASan adapter."""

    def __init__(self, adapter: Any, *, max_packages: int = 16, inband_controls: bool = True):
        self.adapter = adapter
        self.max_packages = max_packages
        self.inband_controls = inband_controls

    @property
    def name(self) -> str:
        return self.adapter.name

    @property
    def version(self) -> str:
        return self.adapter.version

    @property
    def legs(self) -> tuple[str, ...]:
        return tuple(self.adapter.legs)

    @property
    def resource(self) -> str:
        return self.adapter.resource

    def run(self, job: LegJob) -> LegResult:
        return self.adapter.run(job)

    def run_batch(self, jobs: Sequence[LegJob]) -> list[LegResult]:
        return [self.adapter.run(j) for j in jobs]

    def controls(self) -> list[ControlSpec]:
        return self.adapter.controls()

    def health(self) -> dict:
        return self.adapter.health()

    def close(self) -> None:
        close = getattr(self.adapter, "close", None)
        if close is not None:
            close()


# --- subprocesses ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcResult:
    returncode: int | None
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool
    killed: bool  # killed by close()
    launch_error: str | None


class ProcessRunner:
    """Runs one subprocess at a time in its own process group, so `close()` from the watchdog can
    kill it (and everything it started) when the worker is restarted."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._closed = threading.Event()

    def run(self, argv: Sequence[str], *, timeout_s: float, env: Mapping[str, str] | None = None,
            cwd: str | None = None) -> ProcResult:
        t0 = time.monotonic()
        if self._closed.is_set():
            return ProcResult(None, "", "", 0.0, False, True, "runner closed")
        try:
            proc = subprocess.Popen(list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    errors="replace", env=dict(env) if env is not None else None, cwd=cwd,
                                    start_new_session=True)
        except OSError as exc:
            return ProcResult(None, "", "", time.monotonic() - t0, False, False, f"{type(exc).__name__}: {exc}")
        with self._lock:
            if self._closed.is_set():  # close() came after the check above, and found nothing to kill
                self._kill(proc)
            else:
                self._proc = proc
        try:
            out, err = proc.communicate(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            self._kill(proc)
            out, err = proc.communicate()
            timed_out = True
        finally:
            with self._lock:
                self._proc = None
        return ProcResult(proc.returncode, out, err, time.monotonic() - t0, timed_out, self._closed.is_set(), None)

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            proc = self._proc
        if proc is not None:
            self._kill(proc)


# --- aorta verdicts --------------------------------------------------------------------------------

REASON_CODE = re.compile(r"[a-z][a-z0-9_]*")


def reason_code(reason: str) -> str:
    """An aorta reason as its leading code, with anything after the code replaced by `sha256:` and 12 hex digits of
    the whole reason: the rest can quote the checked program's output (ConSan's `consan_output_parse_error: ...` quotes
    the value it could not parse), and that output stays in aorta's logs in the run's directory."""
    m = REASON_CODE.match(reason)
    code = m.group(0) if m else ""
    if code == reason:
        return code
    return (f"{code}: " if code else "") + "sha256:" + sha256_text(reason)[:12]


def map_aorta_check(check: Any) -> tuple[str, str | None]:
    """An aorta `CheckResult` as a leg state (Section 6.2, the aorta stand-ins): PASS -> clean;
    FAIL with findings -> finding; waitcheck WARN with findings -> finding; anything else (ERROR,
    NOT_CHECKED, TIMED_OUT, a verdict with no findings to back it) -> unsettled, with aorta's reason as
    `reason_code` gives it. An ASan PASS on a node whose canary is NOT_VERIFIED is already NOT_CHECKED in aorta, so it
    maps to unsettled."""
    state = getattr(getattr(check, "state", None), "value", None)
    verdict = getattr(getattr(check, "verdict", None), "value", None)
    findings = getattr(check, "findings", ())
    reason = getattr(check, "reason", None)
    if state == "ran" and verdict == "pass" and not findings:
        return "clean", None
    if state == "ran" and verdict in ("fail", "warn") and findings:
        return "finding", None
    detail = f"{state}/{verdict}" + (f": {reason_code(str(reason))}" if reason else "")
    return "unsettled", f"checker_not_settled:{detail}"[:300]


def aorta_findings(check: Any, limit: int = 32) -> list[dict]:
    """Findings as sanitised dicts: codes, severities, kernel names and offsets, no message text
    (a message can quote source or output, Section 6.10 item 3)."""
    out = []
    for f in list(getattr(check, "findings", ()))[:limit]:
        out.append({
            "kind": str(getattr(f, "code", "finding"))[:120],
            "sanitizer": str(getattr(f, "sanitizer", ""))[:40],
            "severity": str(getattr(getattr(f, "severity", None), "value", ""))[:40],
            "kernel": (str(f.kernel_name)[:200] if getattr(f, "kernel_name", None) else None),
        })
    return out
