"""
eBPF-based GPU memory profiler for AMD GPUs.

Traces buffer object migrations, memory mappings, and process
eviction/restore events at the kernel driver level via amdkfd and amdgpu
tracepoints.  This provides driver-level visibility into memory behaviour
that user-space tools (torch.cuda.max_memory_allocated) cannot capture.

Key tracepoints:
- amdgpu:amdgpu_bo_move          -- buffer migration between memory domains
- amdgpu:amdgpu_vm_bo_map        -- buffer object mapped into VM
- amdgpu:amdgpu_vm_bo_unmap      -- buffer object unmapped from VM
- amdkfd:kfd_evict_process       -- process evicted from GPU (memory pressure)
- amdkfd:kfd_restore_process     -- process restored after eviction
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# #region agent log
_DBG_LOG_PATH = str(Path(__file__).resolve().parents[4] / ".cursor" / "debug-8e5cb7.log")
def _mdbg(location, message, data=None, hypothesis=None):
    import json as _j, time as _t
    entry = {"sessionId": "8e5cb7", "location": location, "message": message,
             "data": data or {}, "timestamp": int(_t.time() * 1000),
             "hypothesisId": hypothesis or "", "runId": "run1"}
    try:
        Path(_DBG_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(_DBG_LOG_PATH, "a") as _f:
            _f.write(_j.dumps(entry) + "\n")
    except Exception:
        pass
# #endregion


@dataclass
class MemoryTraceEvent:
    """A single memory-related event captured via eBPF."""

    timestamp_ns: int
    event_type: str  # "bo_move", "bo_map", "bo_unmap", "evict", "restore"
    pid: int
    comm: str
    size_bytes: int = 0
    latency_ns: int = 0
    device_id: int = 0
    old_domain: str = ""
    new_domain: str = ""

    @property
    def timestamp_ms(self) -> float:
        return self.timestamp_ns / 1_000_000


@dataclass
class MemoryTraceMetrics:
    """Aggregated memory trace metrics from eBPF tracing."""

    total_bo_moves: int = 0
    total_bo_maps: int = 0
    total_bo_unmaps: int = 0
    total_evictions: int = 0
    total_restores: int = 0
    total_faults: int = 0
    fault_rate_per_sec: float = 0.0
    avg_fault_latency_us: float = 0.0
    migration_bytes: int = 0
    bo_move_rate_per_sec: float = 0.0
    pages_prefetched: int = 0
    trace_duration_ms: float = 0.0
    bpftrace_stderr: str = ""
    events: List[MemoryTraceEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_bo_moves": self.total_bo_moves,
            "total_bo_maps": self.total_bo_maps,
            "total_bo_unmaps": self.total_bo_unmaps,
            "total_evictions": self.total_evictions,
            "total_restores": self.total_restores,
            "total_faults": self.total_faults,
            "fault_rate_per_sec": self.fault_rate_per_sec,
            "avg_fault_latency_us": self.avg_fault_latency_us,
            "migration_bytes": self.migration_bytes,
            "bo_move_rate_per_sec": self.bo_move_rate_per_sec,
            "pages_prefetched": self.pages_prefetched,
            "trace_duration_ms": self.trace_duration_ms,
        }


# ---------------------------------------------------------------------------
# Tracepoint field probing (shared helper)
# ---------------------------------------------------------------------------

def _probe_tracepoint_fields(tp_category: str, tp_name: str) -> Optional[Set[str]]:
    """Read available field names from the debugfs format file.

    Returns a set of field names, or ``None`` if the format file cannot be
    read (e.g. no debugfs access, tracepoint does not exist).
    """
    fmt_path = Path(
        f"/sys/kernel/debug/tracing/events/{tp_category}/{tp_name}/format"
    )
    try:
        content = fmt_path.read_text()
        return set(re.findall(r"field:[^;]*\s(\w+);", content))
    except (PermissionError, OSError, FileNotFoundError):
        return None


def _check_tracepoint_exists(tp_category: str, tp_name: str) -> bool:
    """Check whether a tracepoint directory exists in debugfs.

    Only returns ``False`` when the category directory exists but the
    specific tracepoint does not. When debugfs is not mounted or
    permissions prevent reading, returns ``True`` (include the tracepoint
    and let bpftrace report the error via the health check).
    """
    category_dir = Path(f"/sys/kernel/debug/tracing/events/{tp_category}")
    tp_dir = category_dir / tp_name
    try:
        if not category_dir.is_dir():
            return True
        return tp_dir.is_dir()
    except (PermissionError, OSError):
        return True


# ---------------------------------------------------------------------------
# bpftrace script generation for memory tracing
# ---------------------------------------------------------------------------

def _build_memory_trace_script() -> str:
    """Build a bpftrace script that traces amdgpu memory events.

    Field names vary across kernel versions (e.g. ``bo_size`` may not
    exist on kernel 6.x where ``amdgpu_bo_move`` only has ``bo``,
    ``new_placement``, ``old_placement``).  We probe debugfs format files
    to determine correct field names and fall back to ``0`` when probing
    is unavailable.

    On MES-based GPUs, ``amdgpu_vm_bo_map``/``unmap`` may not fire for
    KFD compute.  KFD memory mapping tracepoints
    (``kfd_map_memory_to_gpu_start``/``end``) are used instead.
    """
    # --- amdgpu_bo_move fields ---
    bo_move_fields = _probe_tracepoint_fields("amdgpu", "amdgpu_bo_move")
    if bo_move_fields is not None:
        bo_size_expr = (
            "args->bo_size"
            if "bo_size" in bo_move_fields
            else ("args->size" if "size" in bo_move_fields else "0")
        )
    else:
        bo_size_expr = "0"

    sections: List[str] = []

    sections.append(f"""\
tracepoint:amdgpu:amdgpu_bo_move
{{
    printf("BO_MOVE|%llu|%d|%s|%d\\n",
           nsecs, pid, comm, {bo_size_expr});
}}""")

    sections.append("""\
tracepoint:amdgpu:amdgpu_vm_bo_map
{
    printf("BO_MAP|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}""")

    sections.append("""\
tracepoint:amdgpu:amdgpu_vm_bo_unmap
{
    printf("BO_UNMAP|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}""")

    # KFD tracepoints -- use correct names (with _worker_ suffix)
    if _check_tracepoint_exists("amdkfd", "kfd_evict_process_worker_start"):
        sections.append("""\
tracepoint:amdkfd:kfd_evict_process_worker_start
{
    printf("EVICT|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}""")

    if _check_tracepoint_exists("amdkfd", "kfd_restore_process_worker_start"):
        sections.append("""\
tracepoint:amdkfd:kfd_restore_process_worker_start
{
    printf("RESTORE|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}""")

    # KFD memory mapping tracepoints (fire for compute workloads on MES GPUs)
    if _check_tracepoint_exists("amdkfd", "kfd_map_memory_to_gpu_start"):
        sections.append("""\
tracepoint:amdkfd:kfd_map_memory_to_gpu_start
{
    printf("KFD_MAP_START|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}""")

    if _check_tracepoint_exists("amdkfd", "kfd_map_memory_to_gpu_end"):
        sections.append("""\
tracepoint:amdkfd:kfd_map_memory_to_gpu_end
{
    printf("KFD_MAP_END|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}""")

    header = """\
#!/usr/bin/env bpftrace
/*
 * Trace AMD GPU memory events (BO moves, map/unmap, evictions).
 * Output: TYPE|TIMESTAMP_NS|PID|COMM|SIZE_BYTES
 *
 * No PID filters: these tracepoints fire from kernel threads.
 */
"""
    return header + "\n".join(sections) + "\n"


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
        script = _build_memory_trace_script()
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

        # #region agent log
        _mdbg("ebpf_memory_tracer.py:start", "generated_script", {
            "script_path": str(self._script_path),
            "script_content": self._script_path.read_text()[:2000],
        }, hypothesis="MEM1")
        # #endregion

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

        rc = self._process.poll()
        # #region agent log
        _mdbg("ebpf_memory_tracer.py:start", "health_check", {
            "poll_rc": rc,
            "pid": self._process.pid if self._process else None,
        }, hypothesis="MEM1")
        # #endregion
        if rc is not None:
            stderr_text = ""
            try:
                stderr_text = self._process.stderr.read()  # type: ignore[union-attr]
            except Exception:
                pass
            self._process = None
            msg = f"bpftrace (memory) exited immediately (rc={rc})"
            if stderr_text:
                msg += f": {stderr_text.strip()}"
            # #region agent log
            _mdbg("ebpf_memory_tracer.py:start", "health_check_FAILED", {
                "rc": rc, "stderr": stderr_text[:500],
            }, hypothesis="MEM1")
            # #endregion
            logger.warning(msg)
            raise RuntimeError(msg)

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

        stderr_text = ""
        try:
            _, stderr_text = self._process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            _, stderr_text = self._process.communicate(timeout=5)

        self._process = None

        # #region agent log
        post_size = self._output_path.stat().st_size if self._output_path and self._output_path.exists() else -1
        raw_content = ""
        if self._output_path and self._output_path.exists() and post_size > 0:
            raw_content = self._output_path.read_text()[:2000]
        _mdbg("ebpf_memory_tracer.py:stop", "post_stop_state", {
            "stderr": (stderr_text or "")[:500],
            "output_file_size_bytes": post_size,
            "raw_output_first_2k": raw_content,
        }, hypothesis="MEM1")
        # #endregion

        events = self._parse_output()
        metrics = self._compute_metrics(events, elapsed_ns)
        # #region agent log
        _mdbg("ebpf_memory_tracer.py:stop", "parsed_metrics", {
            "num_events": len(events),
            "bo_moves": metrics.total_bo_moves,
            "bo_maps": metrics.total_bo_maps,
            "bo_unmaps": metrics.total_bo_unmaps,
            "evictions": metrics.total_evictions,
            "restores": metrics.total_restores,
            "migration_bytes": metrics.migration_bytes,
        }, hypothesis="MEM1")
        # #endregion
        if stderr_text:
            metrics.bpftrace_stderr = stderr_text.strip()
        return metrics

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    _LINE_RE = re.compile(
        r"^(BO_MOVE|BO_MAP|BO_UNMAP|EVICT|RESTORE|KFD_MAP_START|KFD_MAP_END)\|(\d+)\|(\d+)\|([^|]+)\|(\d+)$"
    )

    _EVENT_TYPE_MAP = {
        "BO_MOVE": "bo_move",
        "BO_MAP": "bo_map",
        "BO_UNMAP": "bo_unmap",
        "EVICT": "evict",
        "RESTORE": "restore",
        "KFD_MAP_START": "bo_map",
        "KFD_MAP_END": "bo_unmap",
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
            if ev.event_type == "bo_move":
                metrics.total_bo_moves += 1
                metrics.migration_bytes += ev.size_bytes
            elif ev.event_type == "bo_map":
                metrics.total_bo_maps += 1
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

        trace_sec = trace_duration_ms / 1000.0
        if trace_sec > 0:
            metrics.fault_rate_per_sec = metrics.total_evictions / trace_sec
            metrics.bo_move_rate_per_sec = metrics.total_bo_moves / trace_sec

        # Average fault latency from evict->restore pairs
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
