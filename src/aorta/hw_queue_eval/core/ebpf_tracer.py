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
        """Whether eBPF tracing is usable on this system.

        bpftrace must be installed, at least one of the amdgpu/amdkfd
        tracepoint trees must be visible, *and* the current process must
        have privileges to attach (root or CAP_BPF/CAP_PERFMON).  We do
        not auto-trigger ``sudo`` here: callers that report
        ``available=True`` should be able to start the tracer without an
        interactive password prompt.
        """
        return (
            self.bpftrace_path is not None
            and (self.has_amdgpu_tracepoints or self.has_amdkfd_tracepoints)
            and self.has_root_or_cap
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel_version": self.kernel_version,
            "bpftrace_version": self.bpftrace_version,
            "has_amdgpu_tracepoints": self.has_amdgpu_tracepoints,
            "has_amdkfd_tracepoints": self.has_amdkfd_tracepoints,
            "amdgpu_tracepoints": self.amdgpu_tracepoints,
            "amdkfd_tracepoints": self.amdkfd_tracepoints,
            "has_root_or_cap": self.has_root_or_cap,
            "available": self.available,
        }


@dataclass
class DriverQueueEvent:
    """A single driver-level queue event captured via eBPF."""

    timestamp_ns: int
    event_type: str  # "submit", "dispatch", "complete", "irq"
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
    """Aggregated driver-level queue metrics from eBPF tracing.

    On modern AMD GPUs with MES (MI200/MI300), the MES firmware handles
    job dispatch.  In that case, ``dispatch`` events come from MES
    kprobes and ``complete`` events mark MES round-trip completion.
    ``submission_to_dispatch_us`` then represents MES round-trip latency.

    On older GPUs or the DRM/graphics path, ``submit`` events come
    from ``amdgpu_cs_ioctl`` and ``dispatch`` from
    ``amdgpu_sched_run_job``.
    """

    total_submissions: int = 0
    total_dispatches: int = 0
    submission_to_dispatch_us: List[float] = field(default_factory=list)
    inter_dispatch_gap_us: List[float] = field(default_factory=list)
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
    def avg_inter_dispatch_gap_us(self) -> float:
        if not self.inter_dispatch_gap_us:
            return 0.0
        return sum(self.inter_dispatch_gap_us) / len(self.inter_dispatch_gap_us)

    @property
    def p99_inter_dispatch_gap_us(self) -> float:
        if not self.inter_dispatch_gap_us:
            return 0.0
        sorted_vals = sorted(self.inter_dispatch_gap_us)
        idx = int(len(sorted_vals) * 0.99)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def dispatch_rate_per_sec(self) -> float:
        if self.trace_duration_ms <= 0:
            return 0.0
        return self.total_dispatches / (self.trace_duration_ms / 1000.0)

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
            "avg_inter_dispatch_gap_us": self.avg_inter_dispatch_gap_us,
            "p99_inter_dispatch_gap_us": self.p99_inter_dispatch_gap_us,
            "dispatch_rate_per_sec": self.dispatch_rate_per_sec,
            "per_ring_submissions": self.per_ring_submissions,
            "per_ring_dispatches": self.per_ring_dispatches,
            "rings_used": self.rings_used,
            "trace_duration_ms": self.trace_duration_ms,
        }


# ---------------------------------------------------------------------------
# Tracepoint format probing
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


# ---------------------------------------------------------------------------
# MES (Micro Engine Scheduler) detection
# ---------------------------------------------------------------------------

def _detect_mes_kprobe(bpftrace_path: str) -> Optional[str]:
    """Find an available MES submit kprobe on this kernel.

    Modern AMD GPUs (MI200/MI300) use MES firmware for job dispatch,
    bypassing the kernel DRM scheduler entirely.  Returns the kprobe
    symbol name (e.g. ``mes_v12_0_submit_pkt_and_poll_completion``) or
    ``None`` if MES kprobes are not available.
    """
    for ver in ("v12_0", "v11_0"):
        sym = f"mes_{ver}_submit_pkt_and_poll_completion"
        try:
            r = subprocess.run(
                [bpftrace_path, "-l", f"kprobe:{sym}"],
                capture_output=True, text=True, timeout=5,
            )
            if r.stdout.strip():
                return sym
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    return None


# ---------------------------------------------------------------------------
# bpftrace script generation
# ---------------------------------------------------------------------------

def _build_queue_trace_script(
    target_pid: Optional[int] = None,
    bpftrace_path: Optional[str] = None,
) -> str:
    """Build a bpftrace script that traces amdgpu queue events.

    The function auto-detects which probing strategy works on the
    running kernel:

    1. **MES mode** (modern MI200/MI300 GPUs): Uses kprobes on the MES
       firmware submit function plus ``amdgpu_iv`` interrupts.  The
       kernel DRM scheduler is bypassed on these GPUs, so the
       ``amdgpu_sched_run_job`` tracepoint never fires.

    2. **Legacy DRM-scheduler mode**: Uses ``amdgpu_cs_ioctl`` and
       ``amdgpu_sched_run_job`` tracepoints (older GPUs / graphics path).

    Field names in tracepoints vary across kernel versions; the function
    probes debugfs format files and falls back to ``0`` constants.
    """
    mes_sym = None
    if bpftrace_path:
        mes_sym = _detect_mes_kprobe(bpftrace_path)

    if mes_sym is not None:
        return _build_mes_trace_script(mes_sym, target_pid)
    return _build_legacy_trace_script(target_pid)


def _build_mes_trace_script(
    mes_symbol: str,
    target_pid: Optional[int] = None,
) -> str:
    """Build a bpftrace script for MES-based GPUs (MI200/MI300).

    On MES GPUs, steady-state compute dispatch goes through user-space
    doorbell writes directly to GPU firmware -- the kernel is not on
    the data path.  We attach to:

    - ``kprobe:<mes_symbol>``: kernel-mediated MES submissions
      (e.g. ``mes_v12_0_submit_pkt_and_poll_completion``).  This catches
      management/control packets, not steady-state compute, but lets
      the tracer correlate kernel side and IRQ side timings.
    - ``amdgpu_iv``: GPU interrupt completions
    - ``amdgpu_device_wreg``: kernel-side register writes (management ops)

    Output: ``TYPE|TIMESTAMP_NS|PID|COMM|RING|FENCE``
    """
    pid_filter = f" /pid == {target_pid}/" if target_pid is not None else ""
    sections: List[str] = []

    if mes_symbol:
        sections.append(f"""\
kprobe:{mes_symbol}{pid_filter}
{{
    @mes_in[tid] = nsecs;
    printf("SUBMIT|%llu|%d|%s|0|0\\n", nsecs, pid, comm);
}}

kretprobe:{mes_symbol}{pid_filter}
{{
    $start = @mes_in[tid];
    if ($start) {{
        printf("COMPLETE|%llu|%d|%s|0|0\\n", nsecs, pid, comm);
        delete(@mes_in[tid]);
    }}
}}""")

    sections.append("""\
tracepoint:amdgpu:amdgpu_iv
{
    printf("IRQ|%llu|%d|%s|%d|%d\\n",
           nsecs, pid, comm, args->ring_id, args->src_id);
}""")

    sections.append("""\
tracepoint:amdgpu:amdgpu_device_wreg
{
    printf("WREG|%llu|%d|%s|%d|%d\\n",
           nsecs, pid, comm, args->reg, args->did);
}""")

    header = (
        "#!/usr/bin/env bpftrace\n"
        "/*\n"
        " * Trace AMD GPU queue events on MES-based GPUs (MI200/MI300).\n"
        " *\n"
        " * Output: TYPE|TIMESTAMP_NS|PID|COMM|RING|FENCE\n"
    )
    if mes_symbol:
        header += f" * MES kprobe attached: {mes_symbol}\n"
    if target_pid is not None:
        header += f" * Scoped to PID {target_pid}.\n"
    header += " */\n"
    return header + "\n".join(sections) + "\n"


def _build_legacy_trace_script(
    target_pid: Optional[int] = None,
) -> str:
    """Build a bpftrace script using DRM-scheduler tracepoints (legacy)."""
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

    sched_fields = _probe_tracepoint_fields("amdgpu", "amdgpu_sched_run_job")
    if sched_fields is not None:
        sched_ring = "args->ring" if "ring" in sched_fields else "0"
        sched_seqno = (
            "args->seqno"
            if "seqno" in sched_fields
            else ("args->sched_job_id" if "sched_job_id" in sched_fields else "0")
        )
    else:
        sched_ring = "0"
        sched_seqno = "0"

    pid_filter = f"\n/pid == {target_pid}/" if target_pid is not None else ""

    return f"""\
#!/usr/bin/env bpftrace
/*
 * Trace amdgpu command submission and dispatch (legacy DRM-scheduler path).
 * Output: TYPE|TIMESTAMP_NS|PID|COMM|RING|FENCE
 */

tracepoint:amdgpu:amdgpu_cs_ioctl{pid_filter}
{{
    printf("SUBMIT|%llu|%d|%s|%d|%d\\n",
           nsecs, pid, comm, {cs_ring}, {cs_fence});
}}

tracepoint:amdgpu:amdgpu_sched_run_job
{{
    printf("DISPATCH|%llu|%d|%s|%d|%d\\n",
           nsecs, pid, comm, {sched_ring}, {sched_seqno});
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

    # Root / CAP_BPF / CAP_PERFMON check.  bpftrace requires either
    # uid 0 or one of the modern eBPF-attach capabilities.  The
    # capability bits live in /proc/self/status under "CapEff".
    caps.has_root_or_cap = _check_ebpf_privilege()

    return caps


# Capability bit offsets defined in <linux/capability.h>.
_CAP_SYS_ADMIN = 21
_CAP_BPF = 39
_CAP_PERFMON = 38


def _check_ebpf_privilege() -> bool:
    """Return True if the current process can attach eBPF probes.

    Either uid 0 or one of {CAP_SYS_ADMIN, CAP_BPF, CAP_PERFMON} in the
    effective capability set qualifies.  Errors reading
    ``/proc/self/status`` (e.g. on non-Linux platforms) fall back to the
    uid check only.
    """
    try:
        if os.geteuid() == 0:
            return True
    except AttributeError:
        return False

    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("CapEff:"):
                    cap_eff = int(line.split()[1], 16)
                    needed = (
                        (1 << _CAP_SYS_ADMIN)
                        | (1 << _CAP_BPF)
                        | (1 << _CAP_PERFMON)
                    )
                    return bool(cap_eff & needed)
    except (OSError, ValueError):
        pass
    return False


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
        self._output_dir = output_dir or Path(tempfile.mkdtemp(prefix="aorta_ebpf_"))
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._process: Optional[subprocess.Popen] = None
        self._script_path: Optional[Path] = None
        self._output_path: Optional[Path] = None
        self._stderr_path: Optional[Path] = None
        self._stderr_file = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_chunks: List[str] = []
        self._start_time_ns: Optional[int] = None

    def _drain_stderr(self) -> None:
        """Continuously read stderr to keep the pipe from filling up.

        bpftrace may emit warnings or per-probe diagnostics throughout
        the run; if we never read the pipe, the kernel buffer fills up
        and the subprocess can stall indefinitely.  We tee everything to
        ``self._stderr_path`` and keep a bounded in-memory buffer so
        ``stop()`` can surface the recent stderr to the caller.
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

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def _generate_script(self, bpftrace_path: Optional[str] = None) -> Path:
        """Generate the bpftrace script and write it to a temp file."""
        script = _build_queue_trace_script(
            target_pid=self._target_pid,
            bpftrace_path=bpftrace_path,
        )
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

        self._script_path = self._generate_script(bpftrace_path=caps.bpftrace_path)
        self._output_path = self._output_dir / "queue_trace.log"
        self._stderr_path = self._output_dir / "queue_trace.stderr.log"

        cmd: List[str] = []
        if self._sudo and os.geteuid() != 0:
            cmd.append("sudo")
        cmd.extend([caps.bpftrace_path, str(self._script_path)])

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
            target=self._drain_stderr, name="bpftrace-queue-stderr", daemon=True,
        )
        self._stderr_thread.start()

        time.sleep(self._ATTACH_DELAY_SEC)

        rc = self._process.poll()
        if rc is not None:
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=1.0)
            stderr_text = "".join(self._stderr_chunks)
            self._cleanup_stderr_capture()
            self._process = None
            msg = f"bpftrace exited immediately (rc={rc})"
            if stderr_text:
                msg += f": {stderr_text.strip()}"
            logger.warning(msg)
            raise RuntimeError(msg)

        # Capture start time AFTER attach completes so that the reported
        # ``trace_duration_ms`` reflects the actual tracing window, not
        # the probe-attach delay.
        self._start_time_ns = time.monotonic_ns()

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

        if stderr_text.strip():
            import warnings
            warnings.warn(f"bpftrace stderr: {stderr_text.strip()}")

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
        r"^(SUBMIT|DISPATCH|COMPLETE|IRQ|WREG)\|(\d+)\|(\d+)\|([^|]+)\|(\d+)\|(\d+)$"
    )

    _EVENT_TYPE_MAP = {
        "SUBMIT": "submit",
        "DISPATCH": "dispatch",
        "COMPLETE": "complete",
        "IRQ": "irq",
        "WREG": "wreg",
    }

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
                event_type = self._EVENT_TYPE_MAP.get(event_type_raw, event_type_raw.lower())

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
    def _group_irq_completions(
        irq_events: List[DriverQueueEvent],
        window_us: float = 500.0,
    ) -> List[DriverQueueEvent]:
        """Group nearby IRQ events into single completion events.

        GPU completions often deliver multiple interrupts within a small
        window (one per CPU/node).  This deduplicates them, keeping the
        earliest timestamp from each group as the canonical completion
        time.

        We compare each IRQ to the *previous* IRQ's timestamp (not the
        group head's), so a long but evenly-spaced burst stays in one
        group even when the burst spans more than ``window_us``.
        """
        if not irq_events:
            return []
        window_ns = window_us * 1_000.0
        sorted_irqs = sorted(irq_events, key=lambda e: e.timestamp_ns)
        groups: List[DriverQueueEvent] = [sorted_irqs[0]]
        last_seen_ns = sorted_irqs[0].timestamp_ns
        for ev in sorted_irqs[1:]:
            gap_ns = ev.timestamp_ns - last_seen_ns
            if gap_ns > window_ns:
                groups.append(ev)
            last_seen_ns = ev.timestamp_ns
        return groups

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
        complete_events: List[DriverQueueEvent] = []
        irq_events: List[DriverQueueEvent] = []

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
            elif ev.event_type == "complete":
                complete_events.append(ev)
            elif ev.event_type == "irq":
                irq_events.append(ev)

        # --- Legacy mode: pair submit->dispatch ---
        for ring, submits in submit_by_ring.items():
            dispatches = dispatch_by_ring.get(ring, [])
            for sub, disp in zip(submits, dispatches):
                delta_us = (disp.timestamp_ns - sub.timestamp_ns) / 1_000
                if delta_us >= 0:
                    metrics.submission_to_dispatch_us.append(delta_us)

        # --- MES mode: pair dispatch->complete for round-trip latency ---
        if complete_events and not submit_by_ring:
            all_dispatches = sorted(
                (ev for evs in dispatch_by_ring.values() for ev in evs),
                key=lambda e: e.timestamp_ns,
            )
            for disp, comp in zip(all_dispatches, complete_events):
                delta_us = (comp.timestamp_ns - disp.timestamp_ns) / 1_000
                if delta_us >= 0:
                    metrics.submission_to_dispatch_us.append(delta_us)

        # --- Inter-dispatch gaps (legacy / explicit dispatch events) ---
        if dispatch_by_ring and not submit_by_ring:
            all_dispatches_sorted = sorted(
                (ev for evs in dispatch_by_ring.values() for ev in evs),
                key=lambda e: e.timestamp_ns,
            )
            for i in range(1, len(all_dispatches_sorted)):
                gap_us = (
                    all_dispatches_sorted[i].timestamp_ns
                    - all_dispatches_sorted[i - 1].timestamp_ns
                ) / 1_000
                if gap_us >= 0:
                    metrics.inter_dispatch_gap_us.append(gap_us)

        # --- IRQ-based completion metrics (MES/doorbell systems) ---
        # When no dispatch/submit events exist, IRQ completions are the
        # only signal.  Group nearby IRQs and treat each group as one
        # GPU completion event.
        if irq_events and not dispatch_by_ring and not submit_by_ring:
            completion_groups = BPFQueueTracer._group_irq_completions(irq_events)
            metrics.total_dispatches = len(completion_groups)
            metrics.per_ring_dispatches[0] = len(completion_groups)
            for i in range(1, len(completion_groups)):
                gap_us = (
                    completion_groups[i].timestamp_ns
                    - completion_groups[i - 1].timestamp_ns
                ) / 1_000
                if gap_us >= 0:
                    metrics.inter_dispatch_gap_us.append(gap_us)

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
