"""
eBPF-based H2D DMA completion tracer for AMD GPUs.

Detects when compute kernel submissions arrive while a buffer object
migration (H2D DMA copy) is still in-flight -- the kernel-level
signature of the H2D race pattern that produces NaN in distributed
training.

Key tracepoints:
- amdgpu:amdgpu_bo_move        -- buffer migration start (H2D / GTT->VRAM)
- amdgpu:amdgpu_cs_ioctl       -- compute command submission
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
class DMAOverlapEvent:
    """A compute submission that overlapped with an active DMA transfer."""

    timestamp_ns: int
    bo_move_start_ns: int
    compute_submit_ns: int
    overlap_us: float
    bo_move_pid: int
    bo_move_comm: str
    bo_move_size: int
    compute_pid: int
    compute_comm: str
    compute_ring: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "overlap_us": self.overlap_us,
            "bo_move": {
                "start_ns": self.bo_move_start_ns,
                "pid": self.bo_move_pid,
                "comm": self.bo_move_comm,
                "size_bytes": self.bo_move_size,
            },
            "compute_submit": {
                "timestamp_ns": self.compute_submit_ns,
                "pid": self.compute_pid,
                "comm": self.compute_comm,
                "ring": self.compute_ring,
            },
        }


@dataclass
class DMATraceMetrics:
    """Aggregated DMA/H2D overlap detection metrics."""

    total_bo_moves: int = 0
    total_compute_submits: int = 0
    overlaps_detected: int = 0
    overlap_events: List[DMAOverlapEvent] = field(default_factory=list)
    max_overlap_us: float = 0.0
    avg_overlap_us: float = 0.0
    migration_bytes: int = 0
    trace_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_bo_moves": self.total_bo_moves,
            "total_compute_submits": self.total_compute_submits,
            "overlaps_detected": self.overlaps_detected,
            "max_overlap_us": self.max_overlap_us,
            "avg_overlap_us": self.avg_overlap_us,
            "migration_bytes": self.migration_bytes,
            "trace_duration_ms": self.trace_duration_ms,
            "overlap_events": [e.to_dict() for e in self.overlap_events],
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


def _build_dma_trace_script(target_pid: Optional[int] = None) -> str:
    """Build a bpftrace script that captures BO moves and compute submissions.

    The script traces both buffer migrations and command submissions so
    that the user-space analyser can detect H2D-compute overlap -- a
    compute kernel reading a buffer that is still being DMA'd.
    """
    bo_move_fields = _probe_tracepoint_fields("amdgpu", "amdgpu_bo_move")
    if bo_move_fields is not None:
        bo_size_expr = (
            "args->bo_size"
            if "bo_size" in bo_move_fields
            else ("args->size" if "size" in bo_move_fields else "0")
        )
    else:
        bo_size_expr = "0"

    cs_fields = _probe_tracepoint_fields("amdgpu", "amdgpu_cs_ioctl")
    if cs_fields is not None:
        cs_ring = "args->ring" if "ring" in cs_fields else "0"
    else:
        cs_ring = "0"

    pid_filter = f"\n/pid == {target_pid}/" if target_pid is not None else ""

    return f"""\
#!/usr/bin/env bpftrace
/*
 * DMA/H2D overlap detector: traces buffer migrations alongside compute
 * submissions to detect H2D-compute races.
 * Output: TYPE|TIMESTAMP_NS|PID|COMM|RING_OR_SIZE
 */

tracepoint:amdgpu:amdgpu_bo_move{pid_filter}
{{
    printf("DMA_MOVE|%llu|%d|%s|%d\\n",
           nsecs, pid, comm, {bo_size_expr});
}}

tracepoint:amdgpu:amdgpu_cs_ioctl{pid_filter}
{{
    printf("DMA_CS|%llu|%d|%s|%d\\n",
           nsecs, pid, comm, {cs_ring});
}}
"""


@dataclass
class _RawDMAEvent:
    timestamp_ns: int
    event_type: str
    pid: int
    comm: str
    value: int  # size_bytes for DMA_MOVE, ring for DMA_CS


class BPFDMATracer:
    """
    Detect H2D DMA-compute overlap at the kernel driver level.

    Traces ``amdgpu_bo_move`` (buffer migration) and ``amdgpu_cs_ioctl``
    (compute submission) events.  When a compute submission arrives while
    a BO move from the same process is recent and potentially still
    in-flight, it flags the overlap -- the hardware-level confirmation
    of the H2D race pattern.

    Usage::

        tracer = BPFDMATracer(target_pid=os.getpid())
        tracer.start()
        # ... run workload with H2D copies ...
        metrics = tracer.stop()
        if metrics.overlaps_detected > 0:
            print("H2D DMA overlaps detected!")
    """

    def __init__(
        self,
        target_pid: Optional[int] = None,
        sudo: bool = True,
        output_dir: Optional[Path] = None,
        overlap_window_us: float = 500.0,
    ):
        self._target_pid = target_pid
        self._sudo = sudo
        self._output_dir = output_dir or Path(tempfile.mkdtemp(prefix="aorta_ebpf_dma_"))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._overlap_window_us = overlap_window_us

        self._process: Optional[subprocess.Popen] = None
        self._script_path: Optional[Path] = None
        self._output_path: Optional[Path] = None
        self._stderr_path: Optional[Path] = None
        self._stderr_file = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_chunks: List[str] = []
        self._start_time_ns: Optional[int] = None

    def _generate_script(self) -> Path:
        script = _build_dma_trace_script(target_pid=self._target_pid)
        script_path = self._output_dir / "dma_trace.bt"
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
        """Start the DMA overlap tracer."""
        if self._process is not None:
            raise RuntimeError("DMA tracer already running")

        bpftrace_path = shutil.which("bpftrace")
        if bpftrace_path is None:
            raise RuntimeError(
                "bpftrace is not installed. Install it with: "
                "apt-get install bpftrace (Ubuntu) or dnf install bpftrace (RHEL)"
            )

        self._script_path = self._generate_script()
        self._output_path = self._output_dir / "dma_trace.log"
        self._stderr_path = self._output_dir / "dma_trace.stderr.log"

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
            target=self._drain_stderr, name="bpftrace-dma-stderr", daemon=True,
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
            msg = f"bpftrace (DMA tracer) exited immediately (rc={rc})"
            if stderr_text:
                msg += f": {stderr_text.strip()}"
            logger.warning(msg)
            raise RuntimeError(msg)

    def stop(self) -> DMATraceMetrics:
        """Stop the tracer and return overlap detection results."""
        if self._process is None:
            return DMATraceMetrics()

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
        return self._detect_overlaps(events, elapsed_ns)

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    _LINE_RE = re.compile(
        r"^(DMA_MOVE|DMA_CS)\|(\d+)\|(\d+)\|([^|]+)\|(\d+)$"
    )

    def _parse_output(self) -> List[_RawDMAEvent]:
        events: List[_RawDMAEvent] = []
        if self._output_path is None or not self._output_path.exists():
            return events

        with open(self._output_path) as f:
            for line in f:
                line = line.strip()
                m = self._LINE_RE.match(line)
                if not m:
                    continue

                raw_type, ts, pid, comm, value = m.groups()
                events.append(
                    _RawDMAEvent(
                        timestamp_ns=int(ts),
                        event_type="bo_move" if raw_type == "DMA_MOVE" else "compute",
                        pid=int(pid),
                        comm=comm,
                        value=int(value),
                    )
                )

        return events

    def _detect_overlaps(
        self, events: List[_RawDMAEvent], elapsed_ns: int,
    ) -> DMATraceMetrics:
        """Find compute submissions that fall within the overlap window of a BO move.

        A BO move duration is not directly observable from tracepoints
        (there is no ``bo_move_end``), so we use a configurable time
        window: any compute submission within ``overlap_window_us`` after
        a BO move start is flagged as a potential overlap.

        Implementation: walk both event streams in chronological order
        with a sliding deque of currently-active moves (those whose
        timestamp is within ``window_ns`` of the latest event).  Expired
        moves are popped from the front of the deque in O(1).  For each
        compute event we then scan only the active set, so the total
        cost is O(N + sum_of_active_set_sizes), bounded by the time
        window rather than the full move count.  The previous index-
        based form had the same intent but the inner ``for`` loop was
        easy to misread as O(N*M).
        """
        from collections import deque

        metrics = DMATraceMetrics(
            trace_duration_ms=elapsed_ns / 1_000_000,
        )

        moves: List[_RawDMAEvent] = []
        computes: List[_RawDMAEvent] = []

        for ev in events:
            if ev.event_type == "bo_move":
                metrics.total_bo_moves += 1
                metrics.migration_bytes += ev.value
                moves.append(ev)
            elif ev.event_type == "compute":
                metrics.total_compute_submits += 1
                computes.append(ev)

        moves.sort(key=lambda e: e.timestamp_ns)
        computes.sort(key=lambda e: e.timestamp_ns)

        window_ns = int(self._overlap_window_us * 1_000)
        overlap_us_values: List[float] = []

        active: "deque[_RawDMAEvent]" = deque()
        move_iter = iter(moves)
        next_move: Optional[_RawDMAEvent] = next(move_iter, None)

        for cs in computes:
            # 1. Admit any moves that started at or before this compute
            #    event's timestamp.  Moves later than ``cs`` cannot be
            #    in-flight when ``cs`` was submitted.
            while next_move is not None and next_move.timestamp_ns <= cs.timestamp_ns:
                active.append(next_move)
                next_move = next(move_iter, None)

            # 2. Evict moves that started more than ``window_ns`` ago --
            #    by definition they are no longer considered in-flight
            #    for this compute event.  Both lists are sorted, so once
            #    a move ages out for ``cs`` it stays aged out for every
            #    later compute event too; pop from the front in O(1).
            cutoff_ns = cs.timestamp_ns - window_ns
            while active and active[0].timestamp_ns < cutoff_ns:
                active.popleft()

            # 3. Inspect the active set.  When tracing system-wide
            #    (``target_pid=None``) the same window can contain a BO
            #    move from one process and a compute submission from
            #    another -- those are not real overlaps, only same-PID
            #    pairs are.
            for mv in active:
                if mv.pid != cs.pid:
                    continue
                overlap_us = (cs.timestamp_ns - mv.timestamp_ns) / 1_000.0
                metrics.overlap_events.append(DMAOverlapEvent(
                    timestamp_ns=cs.timestamp_ns,
                    bo_move_start_ns=mv.timestamp_ns,
                    compute_submit_ns=cs.timestamp_ns,
                    overlap_us=overlap_us,
                    bo_move_pid=mv.pid,
                    bo_move_comm=mv.comm,
                    bo_move_size=mv.value,
                    compute_pid=cs.pid,
                    compute_comm=cs.comm,
                    compute_ring=cs.value,
                ))
                metrics.overlaps_detected += 1
                overlap_us_values.append(overlap_us)

        if overlap_us_values:
            metrics.max_overlap_us = max(overlap_us_values)
            metrics.avg_overlap_us = sum(overlap_us_values) / len(overlap_us_values)

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
