"""
eBPF-based GPU memory profiler for AMD GPUs.

Traces UVM page faults, memory migrations, and process eviction/restore events
at the kernel driver level via amdkfd and amdgpu tracepoints.  This provides
driver-level visibility into memory behaviour that user-space tools
(torch.cuda.max_memory_allocated) cannot capture.

Key tracepoints:
- amdkfd:kfd_evict_process       -- process evicted from GPU (memory pressure)
- amdkfd:kfd_restore_process     -- process restored after eviction
- amdgpu:amdgpu_vm_bo_map        -- buffer object mapped into VM
- amdgpu:amdgpu_vm_bo_unmap      -- buffer object unmapped from VM
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MemoryTraceEvent:
    """A single memory-related event captured via eBPF."""

    timestamp_ns: int
    event_type: str  # "fault", "evict", "restore", "bo_map", "bo_unmap"
    pid: int
    comm: str
    size_bytes: int = 0
    latency_ns: int = 0
    device_id: int = 0

    @property
    def timestamp_ms(self) -> float:
        return self.timestamp_ns / 1_000_000


@dataclass
class MemoryTraceMetrics:
    """Aggregated memory trace metrics from eBPF tracing."""

    total_faults: int = 0
    total_evictions: int = 0
    total_restores: int = 0
    total_bo_maps: int = 0
    total_bo_unmaps: int = 0
    fault_rate_per_sec: float = 0.0
    avg_fault_latency_us: float = 0.0
    migration_bytes: int = 0
    pages_prefetched: int = 0
    trace_duration_ms: float = 0.0
    events: List[MemoryTraceEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_faults": self.total_faults,
            "total_evictions": self.total_evictions,
            "total_restores": self.total_restores,
            "total_bo_maps": self.total_bo_maps,
            "total_bo_unmaps": self.total_bo_unmaps,
            "fault_rate_per_sec": self.fault_rate_per_sec,
            "avg_fault_latency_us": self.avg_fault_latency_us,
            "migration_bytes": self.migration_bytes,
            "pages_prefetched": self.pages_prefetched,
            "trace_duration_ms": self.trace_duration_ms,
        }


# ---------------------------------------------------------------------------
# bpftrace script template for memory tracing
# ---------------------------------------------------------------------------

_MEMORY_TRACE_SCRIPT = """\
#!/usr/bin/env bpftrace
/*
 * Trace AMD GPU memory events (evictions, restores, BO map/unmap).
 * Output: TYPE|TIMESTAMP_NS|PID|COMM|SIZE_BYTES
 */

tracepoint:amdgpu:amdgpu_vm_bo_map
/pid == {pid}/
{{
    printf("BO_MAP|%llu|%d|%s|%d\\n",
           nsecs, pid, comm, args->bo_size);
}}

tracepoint:amdgpu:amdgpu_vm_bo_unmap
/pid == {pid}/
{{
    printf("BO_UNMAP|%llu|%d|%s|%d\\n",
           nsecs, pid, comm, args->bo_size);
}}

tracepoint:amdkfd:kfd_evict_process
{{
    printf("EVICT|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}}

tracepoint:amdkfd:kfd_restore_process
{{
    printf("RESTORE|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}}
"""

_MEMORY_TRACE_SCRIPT_ALL_PIDS = """\
#!/usr/bin/env bpftrace
/*
 * Trace AMD GPU memory events for all processes.
 * Output: TYPE|TIMESTAMP_NS|PID|COMM|SIZE_BYTES
 */

tracepoint:amdgpu:amdgpu_vm_bo_map
{{
    printf("BO_MAP|%llu|%d|%s|%d\\n",
           nsecs, pid, comm, args->bo_size);
}}

tracepoint:amdgpu:amdgpu_vm_bo_unmap
{{
    printf("BO_UNMAP|%llu|%d|%s|%d\\n",
           nsecs, pid, comm, args->bo_size);
}}

tracepoint:amdkfd:kfd_evict_process
{{
    printf("EVICT|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}}

tracepoint:amdkfd:kfd_restore_process
{{
    printf("RESTORE|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}}
"""


class BPFMemoryTracer:
    """
    Trace AMD GPU memory events via bpftrace.

    Captures buffer object mapping/unmapping, process evictions (due to
    memory pressure), and process restores.  Useful for diagnosing memory
    thrashing in multi-GPU workloads and validating prefetch effectiveness.

    Requires:
    - Linux kernel >=5.x with amdgpu/amdkfd drivers loaded
    - bpftrace installed (usually requires root/sudo)
    - amdgpu and amdkfd tracepoints present in debugfs

    Usage::

        tracer = BPFMemoryTracer(target_pid=os.getpid())
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
        self._output_dir = output_dir or Path(tempfile.mkdtemp(prefix="aorta_ebpf_mem_"))
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._process: Optional[subprocess.Popen] = None
        self._script_path: Optional[Path] = None
        self._output_path: Optional[Path] = None
        self._start_time_ns: Optional[int] = None

    def _generate_script(self) -> Path:
        if self._target_pid is not None:
            script = _MEMORY_TRACE_SCRIPT.format(pid=self._target_pid)
        else:
            script = _MEMORY_TRACE_SCRIPT_ALL_PIDS

        script_path = self._output_dir / "memory_trace.bt"
        script_path.write_text(script)
        return script_path

    def start(self) -> None:
        """Start the memory tracer in the background."""
        if self._process is not None:
            raise RuntimeError("Memory tracer already running")

        bpftrace_path = shutil.which("bpftrace")
        if bpftrace_path is None:
            raise RuntimeError(
                "bpftrace is not installed. Install it with: "
                "apt-get install bpftrace (Ubuntu) or dnf install bpftrace (RHEL)"
            )

        self._script_path = self._generate_script()
        self._output_path = self._output_dir / "memory_trace.log"

        cmd: List[str] = []
        if self._sudo and os.geteuid() != 0:
            cmd.append("sudo")
        cmd.extend([bpftrace_path, str(self._script_path)])

        with open(self._output_path, "w") as out_f:
            self._process = subprocess.Popen(
                cmd,
                stdout=out_f,
                stderr=subprocess.PIPE,
                text=True,
            )

        self._start_time_ns = time.monotonic_ns()
        time.sleep(0.5)

    def stop(self) -> MemoryTraceMetrics:
        """Stop the memory tracer and return parsed metrics."""
        if self._process is None:
            return MemoryTraceMetrics()

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
            self._process.wait(timeout=5)

        self._process = None

        events = self._parse_output()
        return self._compute_metrics(events, elapsed_ns)

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    _LINE_RE = re.compile(
        r"^(BO_MAP|BO_UNMAP|EVICT|RESTORE)\|(\d+)\|(\d+)\|([^|]+)\|(\d+)$"
    )

    _EVENT_TYPE_MAP = {
        "BO_MAP": "bo_map",
        "BO_UNMAP": "bo_unmap",
        "EVICT": "evict",
        "RESTORE": "restore",
    }

    def _parse_output(self) -> List[MemoryTraceEvent]:
        events: List[MemoryTraceEvent] = []
        if self._output_path is None or not self._output_path.exists():
            return events

        with open(self._output_path) as f:
            for line in f:
                line = line.strip()
                m = self._LINE_RE.match(line)
                if not m:
                    continue

                raw_type, ts, pid, comm, size = m.groups()
                events.append(
                    MemoryTraceEvent(
                        timestamp_ns=int(ts),
                        event_type=self._EVENT_TYPE_MAP.get(raw_type, raw_type),
                        pid=int(pid),
                        comm=comm,
                        size_bytes=int(size),
                    )
                )

        return events

    @staticmethod
    def _compute_metrics(
        events: List[MemoryTraceEvent], elapsed_ns: int
    ) -> MemoryTraceMetrics:
        trace_duration_ms = elapsed_ns / 1_000_000
        if not events:
            return MemoryTraceMetrics(trace_duration_ms=trace_duration_ms)

        metrics = MemoryTraceMetrics(
            trace_duration_ms=trace_duration_ms,
            events=events,
        )

        evict_timestamps: List[int] = []

        for ev in events:
            if ev.event_type == "bo_map":
                metrics.total_bo_maps += 1
                metrics.migration_bytes += ev.size_bytes
            elif ev.event_type == "bo_unmap":
                metrics.total_bo_unmaps += 1
            elif ev.event_type == "evict":
                metrics.total_evictions += 1
                evict_timestamps.append(ev.timestamp_ns)
            elif ev.event_type == "restore":
                metrics.total_restores += 1
                if evict_timestamps:
                    latency_ns = ev.timestamp_ns - evict_timestamps.pop(0)
                    metrics.total_faults += 1
                    ev.latency_ns = latency_ns

        # Fault rate
        trace_sec = trace_duration_ms / 1000.0
        if trace_sec > 0:
            metrics.fault_rate_per_sec = metrics.total_evictions / trace_sec

        # Average fault latency from evict→restore pairs
        latencies_us = [
            ev.latency_ns / 1000.0
            for ev in events
            if ev.event_type == "restore" and ev.latency_ns > 0
        ]
        if latencies_us:
            metrics.avg_fault_latency_us = sum(latencies_us) / len(latencies_us)

        return metrics

    def cleanup(self) -> None:
        """Stop tracer if running."""
        if self.is_running:
            self.stop()

    def __del__(self) -> None:
        if self.is_running:
            try:
                self.stop()
            except Exception:
                pass
