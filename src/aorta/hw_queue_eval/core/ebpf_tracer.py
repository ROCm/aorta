"""
eBPF-based hardware queue tracer for AMD GPUs.

Uses bpftrace to attach to amdgpu/amdkfd kernel tracepoints and capture
ground-truth command submission and dispatch timing at the driver level.
This complements the user-space CUDA-event-based measurements in metrics.py.

Key tracepoints:
- amdgpu:amdgpu_cs_ioctl        -- command submission (ring/queue ID)
- amdgpu:amdgpu_sched_run_job   -- job dispatched to HW queue
- amdgpu:amdgpu_vm_bo_map       -- buffer object mapping
- amdgpu:amdgpu_vm_bo_unmap     -- buffer object unmapping
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EBPFCapabilities:
    """Available eBPF capabilities on the current system."""

    kernel_version: str = ""
    bpftrace_path: Optional[str] = None
    bpftrace_version: Optional[str] = None
    has_amdgpu_tracepoints: bool = False
    has_amdkfd_tracepoints: bool = False
    amdgpu_tracepoints: List[str] = field(default_factory=list)
    amdkfd_tracepoints: List[str] = field(default_factory=list)
    has_root_or_cap: bool = False

    @property
    def available(self) -> bool:
        """Whether eBPF tracing is usable on this system."""
        return (
            self.bpftrace_path is not None
            and (self.has_amdgpu_tracepoints or self.has_amdkfd_tracepoints)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel_version": self.kernel_version,
            "bpftrace_version": self.bpftrace_version,
            "has_amdgpu_tracepoints": self.has_amdgpu_tracepoints,
            "has_amdkfd_tracepoints": self.has_amdkfd_tracepoints,
            "amdgpu_tracepoints": self.amdgpu_tracepoints,
            "amdkfd_tracepoints": self.amdkfd_tracepoints,
            "available": self.available,
        }


@dataclass
class DriverQueueEvent:
    """A single driver-level queue event captured via eBPF."""

    timestamp_ns: int
    event_type: str  # "submit", "dispatch", "complete"
    pid: int
    comm: str  # process name
    ring: int = 0  # HW ring / queue index
    fence: int = 0  # fence sequence number
    device_id: int = 0

    @property
    def timestamp_ms(self) -> float:
        return self.timestamp_ns / 1_000_000


@dataclass
class DriverQueueMetrics:
    """Aggregated driver-level queue metrics from eBPF tracing."""

    total_submissions: int = 0
    total_dispatches: int = 0
    submission_to_dispatch_us: List[float] = field(default_factory=list)
    per_ring_submissions: Dict[int, int] = field(default_factory=dict)
    per_ring_dispatches: Dict[int, int] = field(default_factory=dict)
    trace_duration_ms: float = 0.0
    events: List[DriverQueueEvent] = field(default_factory=list)

    @property
    def avg_submit_to_dispatch_us(self) -> float:
        if not self.submission_to_dispatch_us:
            return 0.0
        return sum(self.submission_to_dispatch_us) / len(self.submission_to_dispatch_us)

    @property
    def p99_submit_to_dispatch_us(self) -> float:
        if not self.submission_to_dispatch_us:
            return 0.0
        sorted_vals = sorted(self.submission_to_dispatch_us)
        idx = int(len(sorted_vals) * 0.99)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def rings_used(self) -> List[int]:
        all_rings = set(self.per_ring_submissions.keys()) | set(
            self.per_ring_dispatches.keys()
        )
        return sorted(all_rings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_submissions": self.total_submissions,
            "total_dispatches": self.total_dispatches,
            "avg_submit_to_dispatch_us": self.avg_submit_to_dispatch_us,
            "p99_submit_to_dispatch_us": self.p99_submit_to_dispatch_us,
            "per_ring_submissions": self.per_ring_submissions,
            "per_ring_dispatches": self.per_ring_dispatches,
            "rings_used": self.rings_used,
            "trace_duration_ms": self.trace_duration_ms,
        }


# ---------------------------------------------------------------------------
# bpftrace script templates
# ---------------------------------------------------------------------------

_QUEUE_TRACE_SCRIPT = """\
#!/usr/bin/env bpftrace
/*
 * Trace amdgpu command submission and dispatch for a target PID.
 * Output is machine-parseable: TYPE|TIMESTAMP_NS|PID|COMM|RING|FENCE
 */

tracepoint:amdgpu:amdgpu_cs_ioctl
/pid == {pid}/
{{
    printf("SUBMIT|%llu|%d|%s|%d|%d\\n",
           nsecs, pid, comm, args->ring, args->num_chunks);
}}

tracepoint:amdgpu:amdgpu_sched_run_job
/pid == {pid}/
{{
    printf("DISPATCH|%llu|%d|%s|%d|%d\\n",
           nsecs, pid, comm, args->ring, args->seqno);
}}
"""

_QUEUE_TRACE_SCRIPT_ALL_PIDS = """\
#!/usr/bin/env bpftrace
/*
 * Trace amdgpu command submission and dispatch for all PIDs.
 * Output is machine-parseable: TYPE|TIMESTAMP_NS|PID|COMM|RING|FENCE
 */

tracepoint:amdgpu:amdgpu_cs_ioctl
{{
    printf("SUBMIT|%llu|%d|%s|%d|%d\\n",
           nsecs, pid, comm, args->ring, args->num_chunks);
}}

tracepoint:amdgpu:amdgpu_sched_run_job
{{
    printf("DISPATCH|%llu|%d|%s|%d|%d\\n",
           nsecs, pid, comm, args->ring, args->seqno);
}}
"""


def check_ebpf_capabilities() -> EBPFCapabilities:
    """Detect available eBPF capabilities on the current system."""
    caps = EBPFCapabilities()

    # Kernel version
    try:
        result = subprocess.run(
            ["uname", "-r"], capture_output=True, text=True, timeout=5
        )
        caps.kernel_version = result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # bpftrace
    bpftrace_path = shutil.which("bpftrace")
    if bpftrace_path:
        caps.bpftrace_path = bpftrace_path
        try:
            result = subprocess.run(
                [bpftrace_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            caps.bpftrace_version = result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # amdgpu tracepoints
    amdgpu_tp_dir = Path("/sys/kernel/debug/tracing/events/amdgpu")
    try:
        if amdgpu_tp_dir.is_dir():
            caps.amdgpu_tracepoints = sorted(
                e.name for e in amdgpu_tp_dir.iterdir() if e.is_dir()
            )
            caps.has_amdgpu_tracepoints = len(caps.amdgpu_tracepoints) > 0
    except (PermissionError, OSError):
        pass

    # amdkfd tracepoints
    amdkfd_tp_dir = Path("/sys/kernel/debug/tracing/events/amdkfd")
    try:
        if amdkfd_tp_dir.is_dir():
            caps.amdkfd_tracepoints = sorted(
                e.name for e in amdkfd_tp_dir.iterdir() if e.is_dir()
            )
            caps.has_amdkfd_tracepoints = len(caps.amdkfd_tracepoints) > 0
    except (PermissionError, OSError):
        pass

    # Root / CAP_BPF check
    caps.has_root_or_cap = os.geteuid() == 0

    return caps


class BPFQueueTracer:
    """
    Trace AMD GPU hardware queue submissions and dispatches via bpftrace.

    Wraps a bpftrace subprocess that attaches to amdgpu kernel tracepoints
    and streams machine-parseable events.  The tracer is designed to run
    alongside a benchmark workload: call ``start()`` before the workload
    and ``stop()`` after it finishes.

    Requires:
    - Linux kernel >=5.x with amdgpu driver loaded
    - bpftrace installed and accessible (usually requires root/sudo)
    - amdgpu tracepoints present in debugfs

    Usage::

        tracer = BPFQueueTracer(target_pid=os.getpid())
        tracer.start()
        # ... run workload ...
        metrics = tracer.stop()
        print(metrics.to_dict())
    """

    def __init__(
        self,
        target_pid: Optional[int] = None,
        sudo: bool = True,
        output_dir: Optional[Path] = None,
    ):
        self._target_pid = target_pid
        self._sudo = sudo
        self._output_dir = output_dir or Path(tempfile.mkdtemp(prefix="aorta_ebpf_"))
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._process: Optional[subprocess.Popen] = None
        self._script_path: Optional[Path] = None
        self._output_path: Optional[Path] = None
        self._start_time_ns: Optional[int] = None

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def _generate_script(self) -> Path:
        """Generate the bpftrace script and write it to a temp file."""
        if self._target_pid is not None:
            script = _QUEUE_TRACE_SCRIPT.format(pid=self._target_pid)
        else:
            script = _QUEUE_TRACE_SCRIPT_ALL_PIDS

        script_path = self._output_dir / "queue_trace.bt"
        script_path.write_text(script)
        return script_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the bpftrace tracer in the background."""
        if self._process is not None:
            raise RuntimeError("Tracer already running")

        caps = check_ebpf_capabilities()
        if caps.bpftrace_path is None:
            raise RuntimeError(
                "bpftrace is not installed. Install it with: "
                "apt-get install bpftrace (Ubuntu) or dnf install bpftrace (RHEL)"
            )

        self._script_path = self._generate_script()
        self._output_path = self._output_dir / "queue_trace.log"

        cmd: List[str] = []
        if self._sudo and os.geteuid() != 0:
            cmd.append("sudo")
        cmd.extend([caps.bpftrace_path, str(self._script_path)])

        with open(self._output_path, "w") as out_f:
            self._process = subprocess.Popen(
                cmd,
                stdout=out_f,
                stderr=subprocess.PIPE,
                text=True,
            )

        self._start_time_ns = time.monotonic_ns()
        # Give bpftrace time to attach probes
        time.sleep(0.5)

    def stop(self) -> DriverQueueMetrics:
        """Stop the tracer and return parsed metrics."""
        if self._process is None:
            return DriverQueueMetrics()

        elapsed_ns = time.monotonic_ns() - (self._start_time_ns or 0)

        # Send SIGINT to bpftrace for graceful shutdown
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
            self._process.wait(timeout=5)

        self._process = None

        events = self._parse_output()
        return self._compute_metrics(events, elapsed_ns)

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    _LINE_RE = re.compile(
        r"^(SUBMIT|DISPATCH)\|(\d+)\|(\d+)\|([^|]+)\|(\d+)\|(\d+)$"
    )

    def _parse_output(self) -> List[DriverQueueEvent]:
        """Parse the bpftrace output log into structured events."""
        events: List[DriverQueueEvent] = []
        if self._output_path is None or not self._output_path.exists():
            return events

        with open(self._output_path) as f:
            for line in f:
                line = line.strip()
                m = self._LINE_RE.match(line)
                if not m:
                    continue

                event_type_raw, ts, pid, comm, ring, fence = m.groups()
                event_type = "submit" if event_type_raw == "SUBMIT" else "dispatch"

                events.append(
                    DriverQueueEvent(
                        timestamp_ns=int(ts),
                        event_type=event_type,
                        pid=int(pid),
                        comm=comm,
                        ring=int(ring),
                        fence=int(fence),
                    )
                )

        return events

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        events: List[DriverQueueEvent],
        elapsed_ns: int,
    ) -> DriverQueueMetrics:
        """Aggregate raw events into DriverQueueMetrics."""
        if not events:
            return DriverQueueMetrics(trace_duration_ms=elapsed_ns / 1_000_000)

        metrics = DriverQueueMetrics(
            trace_duration_ms=elapsed_ns / 1_000_000,
            events=events,
        )

        submit_by_ring: Dict[int, List[DriverQueueEvent]] = {}
        dispatch_by_ring: Dict[int, List[DriverQueueEvent]] = {}

        for ev in events:
            if ev.event_type == "submit":
                metrics.total_submissions += 1
                metrics.per_ring_submissions[ev.ring] = (
                    metrics.per_ring_submissions.get(ev.ring, 0) + 1
                )
                submit_by_ring.setdefault(ev.ring, []).append(ev)
            elif ev.event_type == "dispatch":
                metrics.total_dispatches += 1
                metrics.per_ring_dispatches[ev.ring] = (
                    metrics.per_ring_dispatches.get(ev.ring, 0) + 1
                )
                dispatch_by_ring.setdefault(ev.ring, []).append(ev)

        # Pair submit→dispatch by ring to estimate submission-to-dispatch latency.
        # Within each ring, events are chronologically ordered; we pair them
        # positionally (first submit → first dispatch, etc.).
        for ring, submits in submit_by_ring.items():
            dispatches = dispatch_by_ring.get(ring, [])
            for sub, disp in zip(submits, dispatches):
                delta_us = (disp.timestamp_ns - sub.timestamp_ns) / 1_000
                if delta_us >= 0:
                    metrics.submission_to_dispatch_us.append(delta_us)

        return metrics

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Stop tracer if running and remove temporary files."""
        if self.is_running:
            self.stop()
        # Leave output_dir for inspection; caller can delete if desired.

    def __del__(self) -> None:
        if self.is_running:
            try:
                self.stop()
            except Exception:
                pass
