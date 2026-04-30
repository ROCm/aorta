"""
eBPF-based GPU memory profiler for AMD GPUs.

Traces buffer object migrations, memory mappings, and process
eviction/restore events at the kernel driver level via amdkfd and amdgpu
tracepoints. This provides driver-level visibility into memory behaviour
that user-space tools (torch.cuda.max_memory_allocated) cannot capture.

Note: this module does **not** attach to GPU UVM page fault tracepoints.
The metrics here describe BO map/unmap activity and eviction/restore
cycles that signal driver-level memory pressure -- not literal GPU page
faults.  ``MemoryTraceMetrics.total_faults`` (and the associated
``avg_fault_latency_us``) counts evict -> restore round trips and is more
accurately thought of as ``eviction_restore_pairs``; the legacy name is
kept for backward compatibility with existing dashboards/JSON consumers.

Key tracepoints (the actual symbols probed at runtime; older docs may
refer to ``kfd_evict_process``/``kfd_restore_process``):

- amdgpu:amdgpu_bo_move                       -- BO migration between memory domains
- amdgpu:amdgpu_vm_bo_map                     -- BO mapped into VM
- amdgpu:amdgpu_vm_bo_unmap                   -- BO unmapped from VM
- amdkfd:kfd_evict_process_worker_start       -- process evicted from GPU (memory pressure)
- amdkfd:kfd_restore_process_worker_start     -- process restored after eviction
- amdkfd:kfd_map_memory_to_gpu_start/_end     -- KFD compute path memory mapping
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
    """Aggregated memory trace metrics from eBPF tracing.

    Field semantics:

    - ``total_bo_moves`` / ``total_bo_maps`` / ``total_bo_unmaps``: counts
      of ``amdgpu_bo_move`` / ``amdgpu_vm_bo_map`` / ``amdgpu_vm_bo_unmap``
      tracepoint hits.
    - ``total_evictions`` / ``total_restores``: counts of KFD process
      eviction/restore worker invocations.  These signal driver-level
      memory pressure (the GPU ran out of room, so the process was
      paged out and brought back).
    - ``total_eviction_restore_pairs`` (alias ``total_faults``): number of
      matched evict -> restore pairs observed during the trace.  This is
      *not* a count of GPU UVM page faults; the legacy ``total_faults``
      / ``fault_rate_per_sec`` / ``avg_fault_latency_us`` names are kept
      for backward compatibility with existing dashboards but the
      ``eviction_restore_*`` aliases are preferred for new code.
    """

    total_bo_moves: int = 0
    total_bo_maps: int = 0
    total_bo_unmaps: int = 0
    total_evictions: int = 0
    total_restores: int = 0
    total_eviction_restore_pairs: int = 0
    eviction_restore_rate_per_sec: float = 0.0
    avg_eviction_restore_latency_us: float = 0.0
    migration_bytes: int = 0
    bo_move_rate_per_sec: float = 0.0
    pages_prefetched: int = 0
    trace_duration_ms: float = 0.0
    bpftrace_stderr: str = ""
    events: List[MemoryTraceEvent] = field(default_factory=list)

    # ----- Backward-compatible aliases (deprecated names) -----
    @property
    def total_faults(self) -> int:
        """Deprecated alias for ``total_eviction_restore_pairs``."""
        return self.total_eviction_restore_pairs

    @total_faults.setter
    def total_faults(self, value: int) -> None:
        self.total_eviction_restore_pairs = value

    @property
    def fault_rate_per_sec(self) -> float:
        """Deprecated alias for ``eviction_restore_rate_per_sec``."""
        return self.eviction_restore_rate_per_sec

    @fault_rate_per_sec.setter
    def fault_rate_per_sec(self, value: float) -> None:
        self.eviction_restore_rate_per_sec = value

    @property
    def avg_fault_latency_us(self) -> float:
        """Deprecated alias for ``avg_eviction_restore_latency_us``."""
        return self.avg_eviction_restore_latency_us

    @avg_fault_latency_us.setter
    def avg_fault_latency_us(self, value: float) -> None:
        self.avg_eviction_restore_latency_us = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_bo_moves": self.total_bo_moves,
            "total_bo_maps": self.total_bo_maps,
            "total_bo_unmaps": self.total_bo_unmaps,
            "total_evictions": self.total_evictions,
            "total_restores": self.total_restores,
            "total_eviction_restore_pairs": self.total_eviction_restore_pairs,
            "eviction_restore_rate_per_sec": self.eviction_restore_rate_per_sec,
            "avg_eviction_restore_latency_us": self.avg_eviction_restore_latency_us,
            # --- legacy aliases preserved for dashboard / JSON consumers ---
            "total_faults": self.total_eviction_restore_pairs,
            "fault_rate_per_sec": self.eviction_restore_rate_per_sec,
            "avg_fault_latency_us": self.avg_eviction_restore_latency_us,
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

def _build_memory_trace_script(target_pid: Optional[int] = None) -> str:
    """Build a bpftrace script that traces amdgpu memory events.

    Field names vary across kernel versions (e.g. ``bo_size`` may not
    exist on kernel 6.x where ``amdgpu_bo_move`` only has ``bo``,
    ``new_placement``, ``old_placement``).  We probe debugfs format files
    to determine correct field names and fall back to ``0`` when probing
    is unavailable.

    On MES-based GPUs, ``amdgpu_vm_bo_map``/``unmap`` may not fire for
    KFD compute.  KFD memory mapping tracepoints
    (``kfd_map_memory_to_gpu_start``/``end``) are used instead.

    When ``target_pid`` is provided we attach a ``/pid == N/`` predicate
    to every tracepoint that the kernel fires from the originating user
    task -- BO map/unmap, KFD memory mapping, and the eviction/restore
    worker start probes.  ``amdgpu_bo_move`` and the KFD worker bodies
    can fire from kernel threads on behalf of the process, so we filter
    them as well: the eviction/restore *_worker_start probes carry the
    original task's PID via the work item's ``mm`` reference, which is
    the closest stable signal we have for "owned by this process".
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

    pid_filter = f" /pid == {target_pid}/" if target_pid is not None else ""

    sections: List[str] = []

    sections.append(f"""\
tracepoint:amdgpu:amdgpu_bo_move{pid_filter}
{{
    printf("BO_MOVE|%llu|%d|%s|%d\\n",
           nsecs, pid, comm, {bo_size_expr});
}}""")

    sections.append(f"""\
tracepoint:amdgpu:amdgpu_vm_bo_map{pid_filter}
{{
    printf("BO_MAP|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}}""")

    sections.append(f"""\
tracepoint:amdgpu:amdgpu_vm_bo_unmap{pid_filter}
{{
    printf("BO_UNMAP|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}}""")

    if _check_tracepoint_exists("amdkfd", "kfd_evict_process_worker_start"):
        sections.append(f"""\
tracepoint:amdkfd:kfd_evict_process_worker_start{pid_filter}
{{
    printf("EVICT|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}}""")

    if _check_tracepoint_exists("amdkfd", "kfd_restore_process_worker_start"):
        sections.append(f"""\
tracepoint:amdkfd:kfd_restore_process_worker_start{pid_filter}
{{
    printf("RESTORE|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}}""")

    # KFD memory mapping tracepoints (fire for compute workloads on MES GPUs)
    if _check_tracepoint_exists("amdkfd", "kfd_map_memory_to_gpu_start"):
        sections.append(f"""\
tracepoint:amdkfd:kfd_map_memory_to_gpu_start{pid_filter}
{{
    printf("KFD_MAP_START|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}}""")

    if _check_tracepoint_exists("amdkfd", "kfd_map_memory_to_gpu_end"):
        sections.append(f"""\
tracepoint:amdkfd:kfd_map_memory_to_gpu_end{pid_filter}
{{
    printf("KFD_MAP_END|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}}""")

    header = (
        "#!/usr/bin/env bpftrace\n"
        "/*\n"
        " * Trace AMD GPU memory events (BO moves, map/unmap, evictions).\n"
        " * Output: TYPE|TIMESTAMP_NS|PID|COMM|SIZE_BYTES\n"
    )
    if target_pid is not None:
        header += f" *\n * Scoped to PID {target_pid}.\n"
    header += " */\n"
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

    # Time we wait after spawning bpftrace before considering attach
    # complete.  Used to (a) detect immediate crashes and (b) advance
    # ``_start_time_ns`` past the probe-attach window so that
    # duration-derived rates measure the actual tracing window only.
    _ATTACH_DELAY_SEC: float = 0.5

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
        self._stderr_path: Optional[Path] = None
        self._stderr_file = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_chunks: List[str] = []
        self._start_time_ns: Optional[int] = None

    def _generate_script(self) -> Path:
        script = _build_memory_trace_script(target_pid=self._target_pid)
        script_path = self._output_dir / "memory_trace.bt"
        script_path.write_text(script)
        return script_path

    def _drain_stderr(self) -> None:
        """Continuously read stderr to keep the pipe from filling up.

        bpftrace can emit several lines per second when probes mismatch;
        leaving stderr unread can stall the subprocess.  We tee everything
        to a file in ``output_dir`` and also keep the last few KiB in
        memory so ``stop()`` can surface it via ``MemoryTraceMetrics.bpftrace_stderr``.
        """
        proc = self._process
        if proc is None or proc.stderr is None:
            return
        try:
            for line in iter(proc.stderr.readline, ""):
                if not line:
                    break
                self._stderr_chunks.append(line)
                # Cap in-memory buffer at ~16 KiB
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
        self._stderr_path = self._output_dir / "memory_trace.stderr.log"

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

        # Start a background thread that drains stderr so the pipe never
        # fills up and blocks bpftrace.
        self._stderr_chunks = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="bpftrace-mem-stderr", daemon=True,
        )
        self._stderr_thread.start()

        time.sleep(self._ATTACH_DELAY_SEC)

        rc = self._process.poll()
        if rc is not None:
            # Wait briefly for the drain thread to flush whatever stderr
            # the dying process produced before snapshotting it.
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=1.0)
            stderr_text = "".join(self._stderr_chunks)
            self._cleanup_stderr_capture()
            self._process = None
            msg = f"bpftrace (memory) exited immediately (rc={rc})"
            if stderr_text:
                msg += f": {stderr_text.strip()}"
            logger.warning(msg)
            raise RuntimeError(msg)

        # Capture start time AFTER attach completes so that the reported
        # ``trace_duration_ms`` is the actual measurement window, not
        # measurement window + arbitrary attach delay.
        self._start_time_ns = time.monotonic_ns()

    def _cleanup_stderr_capture(self) -> None:
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None

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
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
            self._stderr_thread = None

        stderr_text = "".join(self._stderr_chunks)
        self._cleanup_stderr_capture()
        self._process = None

        events = self._parse_output()
        metrics = self._compute_metrics(events, elapsed_ns)
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
                    metrics.total_eviction_restore_pairs += 1
                    ev.latency_ns = latency_ns

        trace_sec = trace_duration_ms / 1000.0
        if trace_sec > 0:
            metrics.eviction_restore_rate_per_sec = (
                metrics.total_evictions / trace_sec
            )
            metrics.bo_move_rate_per_sec = metrics.total_bo_moves / trace_sec

        latencies_us = [
            ev.latency_ns / 1000.0
            for ev in events
            if ev.event_type == "restore" and ev.latency_ns > 0
        ]
        if latencies_us:
            metrics.avg_eviction_restore_latency_us = (
                sum(latencies_us) / len(latencies_us)
            )

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
