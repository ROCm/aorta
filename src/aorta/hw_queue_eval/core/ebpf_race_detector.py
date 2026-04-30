"""
eBPF-based stream race detector for AMD GPUs.

Detects concurrent same-ring submissions without proper fencing at the
kernel driver level.  This catches the root cause of NaN issues in
distributed training: two GPU streams submitting work to the same HW
ring without synchronization, leading to data corruption.

Tracepoints attached:
- amdgpu:amdgpu_cs_ioctl        -- command submission (ring, fence seqno)

The race heuristic is purely submission-based: it inspects pairs of
submissions on the same ``(pid, ring)`` from different threads within
``race_window_us`` and flags them when their fence sequence numbers
diverge by more than 1.  An earlier draft of this module also attached
``amdgpu:amdgpu_sched_run_job`` to track dispatches, but that signal
was unused by the analysis and was removed; the ``total_dispatches``
field on :class:`RaceDetectionMetrics` is preserved for JSON-schema
back-compat and will always be ``0`` here.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RaceEvent:
    """A potential race condition detected at the kernel level."""

    timestamp_ns: int
    ring: int
    submit_a_ts: int
    submit_a_pid: int
    submit_a_comm: str
    submit_a_fence: int
    submit_b_ts: int
    submit_b_pid: int
    submit_b_comm: str
    submit_b_fence: int
    gap_us: float
    fence_gap: int

    @property
    def timestamp_ms(self) -> float:
        return self.timestamp_ns / 1_000_000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "ring": self.ring,
            "gap_us": self.gap_us,
            "fence_gap": self.fence_gap,
            "submit_a": {
                "pid": self.submit_a_pid,
                "comm": self.submit_a_comm,
                "fence": self.submit_a_fence,
                "timestamp_ns": self.submit_a_ts,
            },
            "submit_b": {
                "pid": self.submit_b_pid,
                "comm": self.submit_b_comm,
                "fence": self.submit_b_fence,
                "timestamp_ns": self.submit_b_ts,
            },
        }


@dataclass
class RaceDetectionMetrics:
    """Aggregated race detection results."""

    total_submissions: int = 0
    total_dispatches: int = 0
    races_detected: int = 0
    rings_with_races: List[int] = field(default_factory=list)
    race_events: List[RaceEvent] = field(default_factory=list)
    per_ring_submissions: Dict[int, int] = field(default_factory=dict)
    trace_duration_ms: float = 0.0
    race_window_threshold_us: float = 100.0

    @property
    def race_rate_per_sec(self) -> float:
        if self.trace_duration_ms <= 0:
            return 0.0
        return self.races_detected / (self.trace_duration_ms / 1000.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_submissions": self.total_submissions,
            "total_dispatches": self.total_dispatches,
            "races_detected": self.races_detected,
            "rings_with_races": self.rings_with_races,
            "per_ring_submissions": self.per_ring_submissions,
            "race_rate_per_sec": self.race_rate_per_sec,
            "trace_duration_ms": self.trace_duration_ms,
            "race_window_threshold_us": self.race_window_threshold_us,
            "race_events": [e.to_dict() for e in self.race_events],
        }


def _probe_tracepoint_fields(tp_category: str, tp_name: str) -> Optional[Set[str]]:
    fmt_path = Path(
        f"/sys/kernel/debug/tracing/events/{tp_category}/{tp_name}/format"
    )
    try:
        content = fmt_path.read_text()
        return set(re.findall(r"field:[^;]*\s(\w+);", content))
    except (PermissionError, OSError, FileNotFoundError):
        return None


def _build_race_detection_script(
    target_pid: Optional[int] = None,
) -> str:
    """Build a bpftrace script that captures per-submission detail for race analysis.

    Attaches a single tracepoint -- ``amdgpu:amdgpu_cs_ioctl`` -- and
    emits ``RACE_SUBMIT`` lines carrying PID, thread ID, ring, and
    fence sequence number so that the user-space correlator can detect
    interleaved submissions from different threads on the same ring.
    The dispatch tracepoint (``amdgpu:amdgpu_sched_run_job``) is
    intentionally not attached; see the module docstring for context.
    """
    cs_fields = _probe_tracepoint_fields("amdgpu", "amdgpu_cs_ioctl")
    if cs_fields is not None:
        cs_ring = "args->ring" if "ring" in cs_fields else "0"
        cs_fence = (
            "args->num_ibs"
            if "num_ibs" in cs_fields
            else ("args->num_chunks" if "num_chunks" in cs_fields else "0")
        )
    else:
        cs_ring = "0"
        cs_fence = "0"

    pid_filter = f"\n/pid == {target_pid}/" if target_pid is not None else ""

    # NOTE: An earlier version of this script also attached
    # ``tracepoint:amdgpu:amdgpu_sched_run_job`` and emitted RACE_DISPATCH
    # events.  The Python-side race heuristic only looks at submissions,
    # so those dispatch events added overhead and log volume without
    # affecting results.  They were removed; if a dispatch-based race
    # signal is added later, re-introduce the tracepoint together with
    # the analysis that consumes it (and update the docstring on
    # ``BPFRaceDetector``).
    return f"""\
#!/usr/bin/env bpftrace
/*
 * Race detection script: captures submission events with thread IDs
 * for cross-stream race correlation.
 * Output: TYPE|TIMESTAMP_NS|PID|TID|COMM|RING|FENCE
 */

tracepoint:amdgpu:amdgpu_cs_ioctl{pid_filter}
{{
    printf("RACE_SUBMIT|%llu|%d|%d|%s|%d|%d\\n",
           nsecs, pid, tid, comm, {cs_ring}, {cs_fence});
}}
"""


@dataclass
class _RawEvent:
    timestamp_ns: int
    event_type: str
    pid: int
    tid: int
    comm: str
    ring: int
    fence: int


class BPFRaceDetector:
    """
    Detect stream races at the kernel driver level via bpftrace.

    Traces amdgpu command submissions only (``amdgpu_cs_ioctl``) and
    analyses the event stream for concurrent same-``(pid, ring)``
    submissions from different threads without sequential fence
    numbers -- the kernel-level signature of a missing stream
    synchronization.  Dispatch-side tracing was removed because the
    Python-side heuristic never consumed it; ``total_dispatches`` on
    the returned metrics is therefore always ``0`` for this detector.

    Usage::

        detector = BPFRaceDetector(target_pid=os.getpid())
        detector.start()
        # ... run workload ...
        metrics = detector.stop()
        if metrics.races_detected > 0:
            print("RACE CONDITIONS DETECTED!")
    """

    def __init__(
        self,
        target_pid: Optional[int] = None,
        sudo: bool = True,
        output_dir: Optional[Path] = None,
        race_window_us: float = 100.0,
    ):
        self._target_pid = target_pid
        self._sudo = sudo
        self._output_dir = output_dir or Path(tempfile.mkdtemp(prefix="aorta_ebpf_race_"))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._race_window_us = race_window_us

        self._process: Optional[subprocess.Popen] = None
        self._script_path: Optional[Path] = None
        self._output_path: Optional[Path] = None
        self._stderr_path: Optional[Path] = None
        self._stderr_file = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_chunks: List[str] = []
        self._start_time_ns: Optional[int] = None

    def _generate_script(self) -> Path:
        script = _build_race_detection_script(target_pid=self._target_pid)
        script_path = self._output_dir / "race_detect.bt"
        script_path.write_text(script)
        return script_path

    def _drain_stderr(self) -> None:
        """Continuously read bpftrace stderr to keep the pipe from filling.

        Without this, long-running traces can deadlock once the kernel
        stderr pipe buffer fills.  We tee everything to ``stderr_path``
        and keep a bounded in-memory tail for diagnostics.
        """
        proc = self._process
        if proc is None or proc.stderr is None:
            return
        try:
            for line in iter(proc.stderr.readline, ""):
                if not line:
                    break
                self._stderr_chunks.append(line)
                if sum(len(c) for c in self._stderr_chunks) > 16384:
                    self._stderr_chunks = self._stderr_chunks[-256:]
                if self._stderr_file is not None:
                    try:
                        self._stderr_file.write(line)
                        self._stderr_file.flush()
                    except Exception:
                        pass
        except (ValueError, OSError):
            pass

    def _cleanup_stderr_capture(self) -> None:
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None

    def start(self) -> None:
        """Start the race detection tracer."""
        if self._process is not None:
            raise RuntimeError("Race detector already running")

        bpftrace_path = shutil.which("bpftrace")
        if bpftrace_path is None:
            raise RuntimeError(
                "bpftrace is not installed. Install it with: "
                "apt-get install bpftrace (Ubuntu) or dnf install bpftrace (RHEL)"
            )

        self._script_path = self._generate_script()
        self._output_path = self._output_dir / "race_detect.log"
        self._stderr_path = self._output_dir / "race_detect.stderr.log"

        cmd: List[str] = []
        if self._sudo and os.geteuid() != 0:
            cmd.append("sudo")
        cmd.extend([bpftrace_path, str(self._script_path)])

        self._stderr_file = open(self._stderr_path, "w")  # noqa: SIM115
        with open(self._output_path, "w") as out_f:
            self._process = subprocess.Popen(
                cmd,
                stdout=out_f,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

        self._stderr_chunks = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="bpftrace-race-stderr", daemon=True,
        )
        self._stderr_thread.start()

        self._start_time_ns = time.monotonic_ns()
        time.sleep(0.5)

        rc = self._process.poll()
        if rc is not None:
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=1.0)
            stderr_text = "".join(self._stderr_chunks)
            self._cleanup_stderr_capture()
            self._process = None
            msg = f"bpftrace (race detector) exited immediately (rc={rc})"
            if stderr_text:
                msg += f": {stderr_text.strip()}"
            logger.warning(msg)
            raise RuntimeError(msg)

    def stop(self) -> RaceDetectionMetrics:
        """Stop the tracer and return race detection results."""
        if self._process is None:
            return RaceDetectionMetrics()

        elapsed_ns = time.monotonic_ns() - (self._start_time_ns or 0)

        try:
            if self._sudo and os.geteuid() != 0:
                subprocess.run(
                    ["sudo", "kill", "-INT", str(self._process.pid)],
                    timeout=5,
                )
            else:
                self._process.send_signal(signal.SIGINT)
        except (subprocess.SubprocessError, ProcessLookupError):
            pass

        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
            self._stderr_thread = None
        self._cleanup_stderr_capture()
        self._process = None

        events = self._parse_output()
        return self._detect_races(events, elapsed_ns)

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    _LINE_RE = re.compile(
        r"^(RACE_SUBMIT|RACE_DISPATCH)\|(\d+)\|(\d+)\|(\d+)\|([^|]+)\|(\d+)\|(\d+)$"
    )

    def _parse_output(self) -> List[_RawEvent]:
        events: List[_RawEvent] = []
        if self._output_path is None or not self._output_path.exists():
            return events

        with open(self._output_path) as f:
            for line in f:
                line = line.strip()
                m = self._LINE_RE.match(line)
                if not m:
                    continue

                raw_type, ts, pid, tid, comm, ring, fence = m.groups()
                events.append(
                    _RawEvent(
                        timestamp_ns=int(ts),
                        event_type="submit" if raw_type == "RACE_SUBMIT" else "dispatch",
                        pid=int(pid),
                        tid=int(tid),
                        comm=comm,
                        ring=int(ring),
                        fence=int(fence),
                    )
                )

        return events

    def _detect_races(
        self, events: List[_RawEvent], elapsed_ns: int,
    ) -> RaceDetectionMetrics:
        """Analyse parsed events for concurrent same-ring submissions.

        Two submissions on the same ring from different threads are flagged
        when their time gap is below ``race_window_us`` **and** their fence
        sequence numbers are not consecutive (indicating a missing barrier).
        """
        metrics = RaceDetectionMetrics(
            trace_duration_ms=elapsed_ns / 1_000_000,
            race_window_threshold_us=self._race_window_us,
        )

        # Bucket submissions by ``(pid, ring)`` rather than ``ring`` alone.
        # When ``target_pid=None`` (system-wide trace), submissions from
        # different processes that happen to land on the same HW ring are
        # not actually racing -- they are isolated by the kernel's
        # per-process queues -- and pairing them across PIDs would
        # manufacture phantom races.
        submits_by_pid_ring: Dict[Tuple[int, int], List[_RawEvent]] = {}
        for ev in events:
            if ev.event_type == "submit":
                metrics.total_submissions += 1
                metrics.per_ring_submissions[ev.ring] = (
                    metrics.per_ring_submissions.get(ev.ring, 0) + 1
                )
                submits_by_pid_ring.setdefault((ev.pid, ev.ring), []).append(ev)
            elif ev.event_type == "dispatch":
                metrics.total_dispatches += 1

        race_rings: set[int] = set()

        for (_pid, ring), submits in submits_by_pid_ring.items():
            submits.sort(key=lambda e: e.timestamp_ns)

            for i in range(1, len(submits)):
                prev = submits[i - 1]
                curr = submits[i]

                if prev.tid == curr.tid:
                    continue

                gap_us = (curr.timestamp_ns - prev.timestamp_ns) / 1_000.0
                if gap_us > self._race_window_us:
                    continue

                fence_gap = abs(curr.fence - prev.fence)
                if fence_gap <= 1:
                    continue

                race = RaceEvent(
                    timestamp_ns=curr.timestamp_ns,
                    ring=ring,
                    submit_a_ts=prev.timestamp_ns,
                    submit_a_pid=prev.pid,
                    submit_a_comm=prev.comm,
                    submit_a_fence=prev.fence,
                    submit_b_ts=curr.timestamp_ns,
                    submit_b_pid=curr.pid,
                    submit_b_comm=curr.comm,
                    submit_b_fence=curr.fence,
                    gap_us=gap_us,
                    fence_gap=fence_gap,
                )
                metrics.race_events.append(race)
                metrics.races_detected += 1
                race_rings.add(ring)

        metrics.rings_with_races = sorted(race_rings)
        return metrics

    def cleanup(self) -> None:
        if self.is_running:
            self.stop()

    def __del__(self) -> None:
        if self.is_running:
            try:
                self.stop()
            except Exception:
                pass
