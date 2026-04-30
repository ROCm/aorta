"""
eBPF-based RCCL collective kernel tracer for AMD GPUs.

Traces GPU kernel submissions and identifies RCCL collective kernels
by their process/comm name to detect when collective output buffers
are read before the collective has completed -- the driver-level
signature of collective-compute races that produce NaN.

Key tracepoint:
- amdgpu:amdgpu_cs_ioctl       -- command submission with ring ID

RCCL kernels are identified from the submitting thread's comm name
(typically containing ``nccl`` or ``rccl`` prefixes) together with
the submitted ring ID observed at command submission time.
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
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

RCCL_COMM_PATTERNS = re.compile(
    r"(nccl|rccl|NCCL|RCCL|ncclKern|allreduce|all_reduce|allgather|"
    r"reduce_scatter|all_to_all|sendrecv|broadcast)",
    re.IGNORECASE,
)


@dataclass
class CollectiveRaceEvent:
    """A potential race between an RCCL collective and a compute submission."""

    timestamp_ns: int
    collective_ring: int
    collective_ts: int
    collective_pid: int
    collective_comm: str
    compute_ring: int
    compute_ts: int
    compute_pid: int
    compute_comm: str
    gap_us: float
    same_ring: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "gap_us": self.gap_us,
            "same_ring": self.same_ring,
            "collective": {
                "ring": self.collective_ring,
                "timestamp_ns": self.collective_ts,
                "pid": self.collective_pid,
                "comm": self.collective_comm,
            },
            "compute": {
                "ring": self.compute_ring,
                "timestamp_ns": self.compute_ts,
                "pid": self.compute_pid,
                "comm": self.compute_comm,
            },
        }


@dataclass
class RCCLTraceMetrics:
    """Aggregated RCCL collective tracing results."""

    total_submissions: int = 0
    collective_submissions: int = 0
    compute_submissions: int = 0
    collective_rings: List[int] = field(default_factory=list)
    compute_rings: List[int] = field(default_factory=list)
    # ``races_detected`` counts unambiguous same-ring data hazards: a
    # compute submission on the same HW ring as a collective within
    # ``race_window_us``.  Cross-ring observations live in
    # ``cross_ring_observations`` because reasoning about WAR hazards
    # there requires fence/barrier inspection that this tracer does not
    # do today.  ``race_events`` retains every observation (both kinds)
    # for downstream inspection; consumers can filter by ``same_ring``.
    races_detected: int = 0
    cross_ring_observations: int = 0
    race_events: List[CollectiveRaceEvent] = field(default_factory=list)
    trace_duration_ms: float = 0.0
    race_window_us: float = 500.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_submissions": self.total_submissions,
            "collective_submissions": self.collective_submissions,
            "compute_submissions": self.compute_submissions,
            "collective_rings": self.collective_rings,
            "compute_rings": self.compute_rings,
            "races_detected": self.races_detected,
            "cross_ring_observations": self.cross_ring_observations,
            "trace_duration_ms": self.trace_duration_ms,
            "race_window_us": self.race_window_us,
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


def _build_rccl_trace_script(target_pid: Optional[int] = None) -> str:
    """Build a bpftrace script that captures all submissions with thread comm.

    RCCL collective kernels are submitted by threads whose ``comm``
    contains rccl/nccl identifiers.  The script captures every
    submission with full comm strings so user-space can classify
    collective vs compute submissions.
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

    return f"""\
#!/usr/bin/env bpftrace
/*
 * RCCL collective tracer: captures all command submissions with thread
 * comm names for collective-vs-compute classification.
 * Output: TYPE|TIMESTAMP_NS|PID|TID|COMM|RING|FENCE
 */

tracepoint:amdgpu:amdgpu_cs_ioctl{pid_filter}
{{
    printf("RCCL_CS|%llu|%d|%d|%s|%d|%d\\n",
           nsecs, pid, tid, comm, {cs_ring}, {cs_fence});
}}
"""


@dataclass
class _RawRCCLEvent:
    timestamp_ns: int
    pid: int
    tid: int
    comm: str
    ring: int
    fence: int
    is_collective: bool


class BPFRCCLTracer:
    """
    Trace RCCL collective kernel submissions and detect collective-compute races.

    Identifies RCCL collective submissions by thread comm name patterns
    (nccl/rccl) and flags when compute submissions appear on the same or
    adjacent rings without proper fencing -- indicating that application
    kernels may read collective output buffers before they are written.

    Usage::

        tracer = BPFRCCLTracer(target_pid=os.getpid())
        tracer.start()
        # ... run workload with RCCL collectives ...
        metrics = tracer.stop()
        if metrics.races_detected > 0:
            print("Collective-compute race detected!")
    """

    def __init__(
        self,
        target_pid: Optional[int] = None,
        sudo: bool = True,
        output_dir: Optional[Path] = None,
        race_window_us: float = 500.0,
    ):
        self._target_pid = target_pid
        self._sudo = sudo
        self._output_dir = output_dir or Path(tempfile.mkdtemp(prefix="aorta_ebpf_rccl_"))
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
        script = _build_rccl_trace_script(target_pid=self._target_pid)
        script_path = self._output_dir / "rccl_trace.bt"
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
        """Start the RCCL collective tracer."""
        if self._process is not None:
            raise RuntimeError("RCCL tracer already running")

        bpftrace_path = shutil.which("bpftrace")
        if bpftrace_path is None:
            raise RuntimeError(
                "bpftrace is not installed. Install it with: "
                "apt-get install bpftrace (Ubuntu) or dnf install bpftrace (RHEL)"
            )

        self._script_path = self._generate_script()
        self._output_path = self._output_dir / "rccl_trace.log"
        self._stderr_path = self._output_dir / "rccl_trace.stderr.log"

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
            target=self._drain_stderr, name="bpftrace-rccl-stderr", daemon=True,
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
            msg = f"bpftrace (RCCL tracer) exited immediately (rc={rc})"
            if stderr_text:
                msg += f": {stderr_text.strip()}"
            logger.warning(msg)
            raise RuntimeError(msg)

    def stop(self) -> RCCLTraceMetrics:
        """Stop the tracer and return collective race detection results."""
        if self._process is None:
            return RCCLTraceMetrics()

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
        r"^RCCL_CS\|(\d+)\|(\d+)\|(\d+)\|([^|]+)\|(\d+)\|(\d+)$"
    )

    def _parse_output(self) -> List[_RawRCCLEvent]:
        events: List[_RawRCCLEvent] = []
        if self._output_path is None or not self._output_path.exists():
            return events

        with open(self._output_path) as f:
            for line in f:
                line = line.strip()
                m = self._LINE_RE.match(line)
                if not m:
                    continue

                ts, pid, tid, comm, ring, fence = m.groups()
                is_coll = bool(RCCL_COMM_PATTERNS.search(comm))

                events.append(
                    _RawRCCLEvent(
                        timestamp_ns=int(ts),
                        pid=int(pid),
                        tid=int(tid),
                        comm=comm,
                        ring=int(ring),
                        fence=int(fence),
                        is_collective=is_coll,
                    )
                )

        return events

    def _detect_races(
        self, events: List[_RawRCCLEvent], elapsed_ns: int,
    ) -> RCCLTraceMetrics:
        """Detect compute submissions that follow a collective within the race window.

        Two outcomes are recorded:

        * **Race (same ring)** -- a compute submission lands on the same
          HW ring as a collective inside ``race_window_us``.  This is an
          unambiguous data hazard and increments ``races_detected``.
        * **Cross-ring observation** -- compute lands on a different ring
          inside the same window.  Whether this is actually a WAR hazard
          depends on fence / barrier state that this tracer does not
          currently inspect, so it is recorded in
          ``cross_ring_observations`` (and added to ``race_events`` with
          ``same_ring=False``) without inflating ``races_detected``.
        """
        metrics = RCCLTraceMetrics(
            trace_duration_ms=elapsed_ns / 1_000_000,
            race_window_us=self._race_window_us,
        )

        collective_events: List[_RawRCCLEvent] = []
        compute_events: List[_RawRCCLEvent] = []
        collective_ring_set: set[int] = set()
        compute_ring_set: set[int] = set()

        for ev in events:
            metrics.total_submissions += 1
            if ev.is_collective:
                metrics.collective_submissions += 1
                collective_events.append(ev)
                collective_ring_set.add(ev.ring)
            else:
                metrics.compute_submissions += 1
                compute_events.append(ev)
                compute_ring_set.add(ev.ring)

        metrics.collective_rings = sorted(collective_ring_set)
        metrics.compute_rings = sorted(compute_ring_set)

        collective_events.sort(key=lambda e: e.timestamp_ns)
        compute_events.sort(key=lambda e: e.timestamp_ns)

        window_ns = int(self._race_window_us * 1_000)

        coll_idx = 0
        for comp in compute_events:
            while coll_idx < len(collective_events) and \
                    collective_events[coll_idx].timestamp_ns < comp.timestamp_ns - window_ns:
                coll_idx += 1

            for ci in range(coll_idx, len(collective_events)):
                coll = collective_events[ci]
                if coll.timestamp_ns > comp.timestamp_ns:
                    break

                gap_ns = comp.timestamp_ns - coll.timestamp_ns
                if gap_ns < 0 or gap_ns > window_ns:
                    continue

                # System-wide traces (target_pid=None) can interleave
                # collectives from process A with computes from process
                # B; pairing them across PIDs would invent races.
                if coll.pid != comp.pid:
                    continue

                same_ring = coll.ring == comp.ring

                race = CollectiveRaceEvent(
                    timestamp_ns=comp.timestamp_ns,
                    collective_ring=coll.ring,
                    collective_ts=coll.timestamp_ns,
                    collective_pid=coll.pid,
                    collective_comm=coll.comm,
                    compute_ring=comp.ring,
                    compute_ts=comp.timestamp_ns,
                    compute_pid=comp.pid,
                    compute_comm=comp.comm,
                    gap_us=gap_ns / 1_000.0,
                    same_ring=same_ring,
                )
                metrics.race_events.append(race)
                if same_ring:
                    metrics.races_detected += 1
                else:
                    metrics.cross_ring_observations += 1

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
