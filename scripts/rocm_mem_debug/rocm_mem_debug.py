#!/usr/bin/env python3
"""
Standalone AMD GPU memory access debugger.

Uses bpftrace to trace kernel-level GPU memory operations (BO migrations,
VM mappings, process evictions, GPU interrupts) and simultaneously monitors
dmesg for GPU fault messages.  Correlates both event sources to help
diagnose intermittent "illegal memory access" (hipErrorIllegalAddress)
errors on ROCm.

Requirements:
  - Linux with amdgpu/amdkfd drivers loaded
  - bpftrace installed (apt install bpftrace / dnf install bpftrace)
  - Root or CAP_BPF privileges
  - Python 3.8+

Usage:
  sudo python3 rocm_mem_debug.py --pid <PID> --duration 300
  sudo python3 rocm_mem_debug.py --duration 120
  sudo python3 rocm_mem_debug.py --check-only
  sudo python3 rocm_mem_debug.py --pid 12345 --duration 60 --output report.json

No dependencies beyond Python stdlib and bpftrace.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()

def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m" if _USE_COLOR else s

def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m" if _USE_COLOR else s

def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m" if _USE_COLOR else s

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _USE_COLOR else s


# ---------------------------------------------------------------------------
# Privilege detection
# ---------------------------------------------------------------------------

# Capability bit offsets defined in <linux/capability.h>.
_CAP_SYS_ADMIN = 21
_CAP_PERFMON = 38
_CAP_BPF = 39


def _can_attach_ebpf() -> bool:
    """Return True if the current process can attach eBPF probes.

    Either uid 0 or one of {CAP_SYS_ADMIN, CAP_BPF, CAP_PERFMON} in the
    effective capability set qualifies.  Falls back to the uid check if
    ``/proc/self/status`` cannot be read.
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


# ---------------------------------------------------------------------------
# System check
# ---------------------------------------------------------------------------

def system_check() -> Dict[str, Any]:
    """Gather environment info: ROCm, GPU, XNACK, bpftrace, tracepoints."""
    info: Dict[str, Any] = {}

    # Kernel
    info["kernel"] = _run_cmd("uname -r")

    # bpftrace
    bp = shutil.which("bpftrace")
    info["bpftrace_path"] = bp
    info["bpftrace_version"] = _run_cmd(f"{bp} --version") if bp else None

    # ROCm
    info["rocm_smi_version"] = _run_cmd("rocm-smi --version 2>/dev/null") or "not found"
    info["hipcc_version"] = _run_cmd("hipcc --version 2>/dev/null | head -1") or "not found"

    # GPU info via rocm-smi
    gpu_info = _run_cmd("rocm-smi --showproductname 2>/dev/null")
    info["gpu_info"] = gpu_info.strip() if gpu_info else "not available"

    vram = _run_cmd("rocm-smi --showmeminfo vram 2>/dev/null")
    info["vram_info"] = vram.strip() if vram else "not available"

    # XNACK
    info["HSA_XNACK"] = os.environ.get("HSA_XNACK", "not set")

    # bpftrace requires either uid 0 or one of CAP_BPF / CAP_PERFMON /
    # CAP_SYS_ADMIN.  We keep the legacy ``is_root`` key (the printed
    # label is "Root/CAP_BPF") so existing JSON consumers don't break.
    info["is_root"] = _can_attach_ebpf()

    # Tracepoints
    amdgpu_tps = _list_tracepoints("amdgpu")
    amdkfd_tps = _list_tracepoints("amdkfd")
    info["amdgpu_tracepoints"] = amdgpu_tps
    info["amdkfd_tracepoints"] = amdkfd_tps
    info["has_amdgpu_tracepoints"] = len(amdgpu_tps) > 0
    info["has_amdkfd_tracepoints"] = len(amdkfd_tps) > 0

    # Env vars
    env_vars = {}
    for var in ("HSA_XNACK", "HSA_ENABLE_SDMA", "GPU_MAX_HW_QUEUES",
                "AMD_LOG_LEVEL", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                "AMD_SERIALIZE_KERNEL", "AMD_SERIALIZE_COPY"):
        val = os.environ.get(var)
        if val is not None:
            env_vars[var] = val
    info["env_vars"] = env_vars

    return info


def print_system_check(info: Dict[str, Any]) -> None:
    print(_bold("=" * 60))
    print(_bold("AMD GPU MEMORY ACCESS DEBUGGER -- SYSTEM CHECK"))
    print(_bold("=" * 60))
    print()
    print(f"  Kernel:           {info['kernel']}")
    print(f"  bpftrace:         {info.get('bpftrace_version') or _red('NOT INSTALLED')}")
    print(f"  Root/CAP_BPF:     {'yes' if info['is_root'] else _red('NO -- required')}")
    print(f"  ROCm (rocm-smi):  {info['rocm_smi_version']}")
    print(f"  hipcc:            {info['hipcc_version']}")
    print(f"  HSA_XNACK:        {info['HSA_XNACK']}")
    print()

    if info["gpu_info"] and info["gpu_info"] != "not available":
        print("  GPU:")
        for line in info["gpu_info"].splitlines():
            if line.strip():
                print(f"    {line.strip()}")
        print()

    if info["vram_info"] and info["vram_info"] != "not available":
        print("  VRAM:")
        for line in info["vram_info"].splitlines():
            if line.strip():
                print(f"    {line.strip()}")
        print()

    print(f"  amdgpu tracepoints: {len(info['amdgpu_tracepoints'])} found")
    if info["amdgpu_tracepoints"]:
        for tp in info["amdgpu_tracepoints"][:10]:
            print(f"    - {tp}")
        if len(info["amdgpu_tracepoints"]) > 10:
            print(f"    ... and {len(info['amdgpu_tracepoints']) - 10} more")
    else:
        print(f"    {_red('(none -- mount debugfs or run as root)')}")

    print(f"  amdkfd tracepoints: {len(info['amdkfd_tracepoints'])} found")
    if info["amdkfd_tracepoints"]:
        for tp in info["amdkfd_tracepoints"]:
            print(f"    - {tp}")
    else:
        print(f"    {_red('(none)')}")

    if info["env_vars"]:
        print()
        print("  Relevant env vars:")
        for k, v in info["env_vars"].items():
            print(f"    {k}={v}")

    print()

    # Readiness
    problems = []
    if not info["bpftrace_path"]:
        problems.append("bpftrace is not installed")
    if not info["is_root"]:
        problems.append(
            "no eBPF privilege (need root or CAP_BPF/CAP_PERFMON/CAP_SYS_ADMIN)"
        )
    if not info["has_amdgpu_tracepoints"] and not info["has_amdkfd_tracepoints"]:
        problems.append("no amdgpu/amdkfd tracepoints found")

    if problems:
        print(_red("  ISSUES:"))
        for p in problems:
            print(_red(f"    - {p}"))
        print()
    else:
        print(_green("  System is ready for GPU memory debugging."))
        print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cmd(cmd: str) -> Optional[str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _list_tracepoints(category: str) -> List[str]:
    tp_dir = Path(f"/sys/kernel/debug/tracing/events/{category}")
    try:
        if tp_dir.is_dir():
            return sorted(e.name for e in tp_dir.iterdir() if e.is_dir())
    except (PermissionError, OSError):
        pass
    return []


def _check_tp(category: str, name: str) -> bool:
    cat_dir = Path(f"/sys/kernel/debug/tracing/events/{category}")
    tp_dir = cat_dir / name
    try:
        if not cat_dir.is_dir():
            return True
        return tp_dir.is_dir()
    except (PermissionError, OSError):
        return True


def _probe_fields(category: str, name: str) -> Optional[Set[str]]:
    fmt = Path(f"/sys/kernel/debug/tracing/events/{category}/{name}/format")
    try:
        content = fmt.read_text()
        return set(re.findall(r"field:[^;]*\s(\w+);", content))
    except (PermissionError, OSError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# bpftrace script generation
# ---------------------------------------------------------------------------

def build_bpftrace_script(target_pid: Optional[int] = None) -> str:
    """Generate a bpftrace script that traces all available memory-related tracepoints."""
    sections: List[str] = []

    # --- amdgpu_bo_move ---
    bo_fields = _probe_fields("amdgpu", "amdgpu_bo_move")
    if bo_fields is not None:
        bo_size = ("args->bo_size" if "bo_size" in bo_fields
                   else ("args->size" if "size" in bo_fields else "0"))
    else:
        bo_size = "0"

    sections.append(f"""\
tracepoint:amdgpu:amdgpu_bo_move
{{
    printf("BO_MOVE|%llu|%d|%s|%d\\n",
           nsecs, pid, comm, {bo_size});
}}""")

    # --- amdgpu_vm_bo_map / unmap ---
    sections.append("""\
tracepoint:amdgpu:amdgpu_vm_bo_map
{
    printf("VM_MAP|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}""")

    sections.append("""\
tracepoint:amdgpu:amdgpu_vm_bo_unmap
{
    printf("VM_UNMAP|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}""")

    # --- amdgpu_vm_set_ptes (if available) ---
    if _check_tp("amdgpu", "amdgpu_vm_set_ptes"):
        sections.append("""\
tracepoint:amdgpu:amdgpu_vm_set_ptes
{
    printf("VM_PTE|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}""")

    # --- amdgpu_iv (GPU interrupts) ---
    if _check_tp("amdgpu", "amdgpu_iv"):
        iv_fields = _probe_fields("amdgpu", "amdgpu_iv")
        if iv_fields and "src_id" in iv_fields:
            iv_extra = "args->src_id"
        else:
            iv_extra = "0"
        sections.append(f"""\
tracepoint:amdgpu:amdgpu_iv
{{
    printf("GPU_IRQ|%llu|%d|%s|%d\\n",
           nsecs, pid, comm, {iv_extra});
}}""")

    # --- KFD eviction / restore ---
    for tp_name, label in [
        ("kfd_evict_process_worker_start", "EVICT"),
        ("kfd_restore_process_worker_start", "RESTORE"),
    ]:
        if _check_tp("amdkfd", tp_name):
            sections.append(f"""\
tracepoint:amdkfd:{tp_name}
{{
    printf("{label}|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}}""")

    # --- KFD memory mapping ---
    for tp_name, label in [
        ("kfd_map_memory_to_gpu_start", "KFD_MAP_START"),
        ("kfd_map_memory_to_gpu_end", "KFD_MAP_END"),
    ]:
        if _check_tp("amdkfd", tp_name):
            sections.append(f"""\
tracepoint:amdkfd:{tp_name}
{{
    printf("{label}|%llu|%d|%s|0\\n",
           nsecs, pid, comm);
}}""")

    header = """\
#!/usr/bin/env bpftrace
/*
 * AMD GPU memory access debugger -- traces BO moves, VM map/unmap,
 * page table updates, GPU interrupts, process evictions.
 * Output: TYPE|TIMESTAMP_NS|PID|COMM|EXTRA
 */
"""
    return header + "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# Event data structures
# ---------------------------------------------------------------------------

@dataclass
class MemEvent:
    timestamp_ns: int
    event_type: str
    pid: int
    comm: str
    extra: int = 0

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ns / 1_000_000_000


@dataclass
class DmesgFault:
    timestamp_s: float
    line: str
    fault_type: str = ""


# ---------------------------------------------------------------------------
# Event parser
# ---------------------------------------------------------------------------

_EVENT_RE = re.compile(
    r"^(BO_MOVE|VM_MAP|VM_UNMAP|VM_PTE|GPU_IRQ|EVICT|RESTORE|"
    r"KFD_MAP_START|KFD_MAP_END)\|(\d+)\|(\d+)\|([^|]+)\|(\d+)$"
)


def parse_bpf_line(line: str) -> Optional[MemEvent]:
    line = line.strip()
    m = _EVENT_RE.match(line)
    if not m:
        return None
    etype, ts, pid, comm, extra = m.groups()
    return MemEvent(
        timestamp_ns=int(ts),
        event_type=etype,
        pid=int(pid),
        comm=comm,
        extra=int(extra),
    )


_DMESG_FAULT_PATTERNS = re.compile(
    r"(amdgpu.*fault|GPU fault|VM fault|illegal memory|page fault|"
    r"[Xx]nack|(?<![0-9a-fA-F])\bECC\b(?! [a-z])|"
    r"\bRAS\b.*(error|checksum|table)|gpu_recover|"
    r"ring .* timeout|amdgpu:.*error|sdma.*error|"
    r"gfx.*error|compute.*error)",
)

# Lines that match the above patterns but are known-benign boot/init messages
_DMESG_FALSE_POSITIVE = re.compile(
    r"(MCE.*decoding enabled|systemd\[|"
    r"Final command line|Freeing .* buffer)",
    re.IGNORECASE,
)

_DMESG_TS_RE = re.compile(r"^\[([^\]]+)\]")


def parse_dmesg_line(line: str) -> Optional[DmesgFault]:
    if not _DMESG_FAULT_PATTERNS.search(line):
        return None
    if _DMESG_FALSE_POSITIVE.search(line):
        return None
    ts_match = _DMESG_TS_RE.search(line)
    ts = 0.0
    if ts_match:
        try:
            ts = float(ts_match.group(1).strip())
        except ValueError:
            ts = time.time()
    else:
        ts = time.time()

    fault_type = "unknown"
    lower = line.lower()
    if "illegal" in lower:
        fault_type = "illegal_access"
    elif "vm fault" in lower or "page fault" in lower:
        fault_type = "vm_fault"
    elif re.search(r"\bECC\b", line):
        fault_type = "ecc_error"
    elif re.search(r"\bRAS\b", line):
        fault_type = "ras_error"
    elif "timeout" in lower:
        fault_type = "ring_timeout"
    elif "gpu_recover" in lower:
        fault_type = "gpu_recovery"
    elif "xnack" in lower:
        fault_type = "xnack_fault"
    elif "fault" in lower:
        fault_type = "gpu_fault"

    return DmesgFault(timestamp_s=ts, line=line.strip(), fault_type=fault_type)


# ---------------------------------------------------------------------------
# Fault correlator and pattern detector
# ---------------------------------------------------------------------------

class FaultCorrelator:
    """Correlate dmesg faults with eBPF memory events."""

    def __init__(self, window_ms: float = 500.0):
        self._window_ms = window_ms
        self.events: List[MemEvent] = []
        self.faults: List[DmesgFault] = []
        self.counters: Dict[str, int] = defaultdict(int)
        self.migration_bytes: int = 0
        self._start_time_ns: Optional[int] = None
        self._evict_times: List[float] = []

    def add_event(self, ev: MemEvent) -> None:
        self.events.append(ev)
        self.counters[ev.event_type] += 1
        if ev.event_type == "BO_MOVE":
            self.migration_bytes += ev.extra
        if ev.event_type == "EVICT":
            self._evict_times.append(ev.timestamp_s)
        if self._start_time_ns is None:
            self._start_time_ns = ev.timestamp_ns

    def add_fault(self, fault: DmesgFault) -> None:
        self.faults.append(fault)

    def detect_patterns(self) -> Dict[str, Any]:
        patterns: Dict[str, Any] = {}

        # Eviction storm: >5 evictions within any 1-second window
        eviction_storms = 0
        for i, t in enumerate(self._evict_times):
            count = sum(1 for t2 in self._evict_times[i:]
                        if t2 - t <= 1.0)
            if count >= 5:
                eviction_storms += 1
        patterns["eviction_storm"] = eviction_storms > 0
        patterns["eviction_storm_count"] = eviction_storms

        # BO migration rate
        duration_s = self._duration_s()
        if duration_s > 0:
            patterns["bo_move_rate_per_sec"] = self.counters["BO_MOVE"] / duration_s
            patterns["vm_map_rate_per_sec"] = self.counters["VM_MAP"] / duration_s
            patterns["irq_rate_per_sec"] = self.counters["GPU_IRQ"] / duration_s
        else:
            patterns["bo_move_rate_per_sec"] = 0.0
            patterns["vm_map_rate_per_sec"] = 0.0
            patterns["irq_rate_per_sec"] = 0.0

        # High migration volume
        patterns["migration_bytes"] = self.migration_bytes
        patterns["migration_mb"] = self.migration_bytes / (1024 * 1024)

        # IRQ spike detection
        irq_events = [e for e in self.events if e.event_type == "GPU_IRQ"]
        irq_spike = False
        if len(irq_events) > 10:
            for i in range(len(irq_events) - 10):
                window = irq_events[i + 10].timestamp_s - irq_events[i].timestamp_s
                if window > 0 and 10 / window > 1000:
                    irq_spike = True
                    break
        patterns["irq_spike"] = irq_spike

        # VM mapping churn
        vm_ops = self.counters["VM_MAP"] + self.counters["VM_UNMAP"]
        patterns["vm_mapping_churn"] = vm_ops > 100 and duration_s > 0 and vm_ops / duration_s > 50

        return patterns

    def correlate_faults(self) -> List[Dict[str, Any]]:
        """For each dmesg fault, find eBPF events in the preceding window."""
        results = []
        for fault in self.faults:
            window_s = self._window_ms / 1000.0
            lo = fault.timestamp_s - window_s
            hi = fault.timestamp_s

            events_in_window = [
                e for e in self.events
                if lo <= e.timestamp_s <= hi
            ]

            ev_counts: Dict[str, int] = defaultdict(int)
            for e in events_in_window:
                ev_counts[e.event_type] += 1

            correlation = {
                "fault_type": fault.fault_type,
                "fault_line": fault.line,
                "fault_timestamp_s": fault.timestamp_s,
                "window_ms": self._window_ms,
                "events_in_window": len(events_in_window),
                "event_counts": dict(ev_counts),
                "had_evictions": ev_counts.get("EVICT", 0) > 0,
                "had_bo_moves": ev_counts.get("BO_MOVE", 0) > 0,
                "had_irq_burst": ev_counts.get("GPU_IRQ", 0) > 10,
            }

            causes = []
            if ev_counts.get("EVICT", 0) > 0:
                causes.append("memory_pressure_eviction")
            if ev_counts.get("BO_MOVE", 0) > 5:
                causes.append("high_bo_migration")
            if ev_counts.get("GPU_IRQ", 0) > 10:
                causes.append("interrupt_burst")
            if ev_counts.get("VM_MAP", 0) + ev_counts.get("VM_UNMAP", 0) > 10:
                causes.append("vm_mapping_churn")
            correlation["probable_causes"] = causes
            results.append(correlation)

        return results

    def generate_report(self, sys_info: Dict[str, Any],
                        duration_s: float) -> Dict[str, Any]:
        patterns = self.detect_patterns()
        correlations = self.correlate_faults()

        evict_restore_latencies = self._evict_restore_latencies()

        recommendations = self._recommendations(sys_info, patterns)

        report = {
            "system": {
                "kernel": sys_info.get("kernel", ""),
                "rocm_version": sys_info.get("rocm_smi_version", ""),
                "gpu": sys_info.get("gpu_info", ""),
                "xnack": sys_info.get("HSA_XNACK", ""),
                "env_vars": sys_info.get("env_vars", {}),
            },
            "duration_s": duration_s,
            "event_counts": dict(self.counters),
            "faults_detected": len(self.faults),
            "fault_details": [
                {"type": f.fault_type, "line": f.line, "timestamp_s": f.timestamp_s}
                for f in self.faults
            ],
            "patterns": patterns,
            "fault_correlations": correlations,
            "evict_restore_latencies_us": evict_restore_latencies,
            "recommendations": recommendations,
        }
        return report

    def _duration_s(self) -> float:
        if not self.events:
            return 0.0
        return (self.events[-1].timestamp_ns - self.events[0].timestamp_ns) / 1e9

    def _evict_restore_latencies(self) -> List[float]:
        evict_ts = []
        latencies = []
        for e in self.events:
            if e.event_type == "EVICT":
                evict_ts.append(e.timestamp_ns)
            elif e.event_type == "RESTORE" and evict_ts:
                lat_us = (e.timestamp_ns - evict_ts.pop(0)) / 1000.0
                if lat_us > 0:
                    latencies.append(lat_us)
        return latencies

    def _recommendations(self, sys_info: Dict[str, Any],
                         patterns: Dict[str, Any]) -> List[str]:
        recs = []
        if patterns.get("eviction_storm"):
            recs.append(
                "Eviction storm detected: GPU memory pressure is high. "
                "Reduce batch size or model size, or check for memory leaks."
            )
        if patterns.get("irq_spike"):
            recs.append(
                "GPU interrupt spike detected: may indicate hardware fault "
                "interrupts. Check dmesg for detailed fault addresses."
            )
        if patterns.get("vm_mapping_churn"):
            recs.append(
                "High VM map/unmap rate: frequent buffer remapping may "
                "cause race conditions. Consider pinned allocations."
            )
        if sys_info.get("HSA_XNACK") == "1" and self.faults:
            recs.append(
                "XNACK is enabled and faults were detected. Try running "
                "with HSA_XNACK=0 to see if faults disappear."
            )
        if sys_info.get("HSA_XNACK") == "not set":
            recs.append(
                "HSA_XNACK is not explicitly set. For debugging, try both "
                "HSA_XNACK=0 and HSA_XNACK=1 to isolate page fault behavior."
            )
        if not self.faults and not patterns.get("eviction_storm"):
            recs.append(
                "No faults or anomalies detected in this capture window. "
                "If the issue is intermittent, try a longer duration."
            )
        if patterns.get("migration_mb", 0) > 1024:
            recs.append(
                f"Large migration volume ({patterns['migration_mb']:.0f} MB). "
                "Heavy VRAM<->GTT traffic may indicate memory pressure."
            )
        return recs


# ---------------------------------------------------------------------------
# Main tracing engine
# ---------------------------------------------------------------------------

class GPUMemoryDebugger:
    """Orchestrate bpftrace and dmesg monitoring."""

    def __init__(self, target_pid: Optional[int] = None,
                 duration_s: float = 60.0,
                 output_file: Optional[str] = None,
                 verbose: bool = False):
        self._target_pid = target_pid
        self._duration_s = duration_s
        self._output_file = output_file
        self._verbose = verbose

        self._bpf_proc: Optional[subprocess.Popen] = None
        self._dmesg_proc: Optional[subprocess.Popen] = None
        self._tmpdir: Optional[str] = None
        self._correlator = FaultCorrelator()
        self._stop_event = threading.Event()
        self._bpf_output_path: Optional[Path] = None
        self._sys_info: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """Run the full debugging session."""
        # System check
        self._sys_info = system_check()
        print_system_check(self._sys_info)

        if not self._sys_info["bpftrace_path"]:
            print(_red("Cannot proceed: bpftrace is not installed."))
            sys.exit(1)
        if not self._sys_info["is_root"]:
            print(_red(
                "Cannot proceed: need root or CAP_BPF/CAP_PERFMON/"
                "CAP_SYS_ADMIN to attach eBPF probes."
            ))
            sys.exit(1)

        self._tmpdir = tempfile.mkdtemp(prefix="rocm_memdebug_")

        # Set up signal handlers
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            self._start_bpftrace()
            self._start_dmesg_monitor()

            print(_bold(f"Tracing for {self._duration_s:.0f} seconds..."))
            if self._target_pid:
                print(f"  Target PID: {self._target_pid}")
            print("  Press Ctrl+C to stop early")
            print()

            start = time.time()
            while not self._stop_event.is_set():
                elapsed = time.time() - start
                if elapsed >= self._duration_s:
                    break
                self._stop_event.wait(timeout=1.0)
                self._read_bpf_output()
                self._print_status_line(elapsed)

            # Final read
            self._read_bpf_output()

        finally:
            self._stop_all()
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

        actual_duration = time.time() - start
        report = self._correlator.generate_report(self._sys_info, actual_duration)
        self._print_report(report)

        if self._output_file:
            with open(self._output_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {self._output_file}")

        return report

    def _signal_handler(self, signum, frame):
        print("\n\nStopping capture...")
        self._stop_event.set()

    def _start_bpftrace(self) -> None:
        script = build_bpftrace_script(self._target_pid)
        script_path = Path(self._tmpdir) / "mem_debug.bt"
        script_path.write_text(script)

        self._bpf_output_path = Path(self._tmpdir) / "bpf_output.log"

        bp = self._sys_info["bpftrace_path"]
        cmd = [bp, str(script_path)]

        self._bpf_outfile = open(self._bpf_output_path, "w")
        self._bpf_proc = subprocess.Popen(
            cmd,
            stdout=self._bpf_outfile,
            stderr=subprocess.PIPE,
            text=True,
        )

        time.sleep(0.5)
        rc = self._bpf_proc.poll()
        if rc is not None:
            stderr = ""
            try:
                stderr = self._bpf_proc.stderr.read()
            except Exception:
                pass
            print(_red(f"bpftrace failed to start (rc={rc})"))
            if stderr:
                print(_red(f"  stderr: {stderr.strip()[:500]}"))
            sys.exit(1)

        print(_green("  bpftrace started"))

    def _start_dmesg_monitor(self) -> None:
        # Use --since to skip historical boot messages and only capture
        # new kernel messages from now onwards.
        since_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        cmd = ["dmesg", "--follow", "-T", "--since", since_ts]
        try:
            self._dmesg_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            # Make stdout non-blocking
            import fcntl
            fd = self._dmesg_proc.stdout.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
            print(_green("  dmesg monitor started"))
        except Exception as e:
            # Fall back to plain --follow if --since is not supported
            try:
                self._dmesg_proc = subprocess.Popen(
                    ["dmesg", "--follow", "-T"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                import fcntl
                fd = self._dmesg_proc.stdout.fileno()
                fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                print(_yellow("  dmesg monitor started (without --since, may show historical messages)"))
            except Exception as e2:
                print(_yellow(f"  dmesg monitor failed: {e2} (continuing without it)"))
                self._dmesg_proc = None

    def _read_bpf_output(self) -> None:
        if not self._bpf_output_path or not self._bpf_output_path.exists():
            return
        try:
            with open(self._bpf_output_path) as f:
                f.seek(getattr(self, '_bpf_read_offset', 0))
                for line in f:
                    ev = parse_bpf_line(line)
                    if ev:
                        self._correlator.add_event(ev)
                        if self._verbose:
                            print(f"  [{ev.event_type}] pid={ev.pid} comm={ev.comm}")
                self._bpf_read_offset = f.tell()
        except Exception:
            pass

        # Also read dmesg
        if self._dmesg_proc and self._dmesg_proc.stdout:
            try:
                while True:
                    line = self._dmesg_proc.stdout.readline()
                    if not line:
                        break
                    fault = parse_dmesg_line(line)
                    if fault:
                        self._correlator.add_fault(fault)
                        print(_red(f"  FAULT: {fault.line.strip()}"))
            except (IOError, OSError):
                pass

    def _print_status_line(self, elapsed: float) -> None:
        c = self._correlator.counters
        faults = len(self._correlator.faults)
        remaining = max(0, self._duration_s - elapsed)
        parts = [
            f"[{elapsed:.0f}s / {self._duration_s:.0f}s]",
            f"BO_MOVE={c['BO_MOVE']}",
            f"VM_MAP={c['VM_MAP']}",
            f"EVICT={c['EVICT']}",
            f"IRQ={c['GPU_IRQ']}",
        ]
        if faults > 0:
            parts.append(_red(f"FAULTS={faults}"))
        status = "  " + "  ".join(parts)
        print(f"\r{status}", end="", flush=True)

    def _stop_all(self) -> None:
        print()

        if self._bpf_proc and self._bpf_proc.poll() is None:
            try:
                self._bpf_proc.send_signal(signal.SIGINT)
                self._bpf_proc.wait(timeout=5)
            except Exception:
                try:
                    self._bpf_proc.kill()
                    self._bpf_proc.wait(timeout=3)
                except Exception:
                    pass
        if hasattr(self, '_bpf_outfile'):
            try:
                self._bpf_outfile.close()
            except Exception:
                pass

        if self._dmesg_proc and self._dmesg_proc.poll() is None:
            try:
                self._dmesg_proc.terminate()
                self._dmesg_proc.wait(timeout=3)
            except Exception:
                try:
                    self._dmesg_proc.kill()
                except Exception:
                    pass

        # Final parse of all output
        self._bpf_read_offset = 0
        self._read_bpf_output()

    def _print_report(self, report: Dict[str, Any]) -> None:
        print()
        print(_bold("=" * 60))
        print(_bold("DIAGNOSTIC REPORT"))
        print(_bold("=" * 60))
        print()

        print(f"  Duration:    {report['duration_s']:.1f} s")
        print()

        # Event counts
        print(_bold("  EVENT COUNTS:"))
        for etype, count in sorted(report["event_counts"].items()):
            print(f"    {etype:<20} {count:>8}")
        print()

        # Faults
        faults = report["faults_detected"]
        if faults > 0:
            print(_red(f"  FAULTS DETECTED: {faults}"))
            for fd in report["fault_details"]:
                print(_red(f"    [{fd['type']}] {fd['line']}"))
            print()
        else:
            print(_green("  FAULTS DETECTED: 0"))
            print()

        # Patterns
        patterns = report["patterns"]
        print(_bold("  PATTERNS:"))
        if patterns.get("eviction_storm"):
            print(_red("    EVICTION STORM detected"))
        if patterns.get("irq_spike"):
            print(_red("    GPU INTERRUPT SPIKE detected"))
        if patterns.get("vm_mapping_churn"):
            print(_yellow("    HIGH VM MAPPING CHURN detected"))
        if patterns.get("migration_mb", 0) > 0:
            print(f"    Migration volume: {patterns['migration_mb']:.1f} MB")
        print(f"    BO move rate:    {patterns.get('bo_move_rate_per_sec', 0):.1f} /sec")
        print(f"    VM map rate:     {patterns.get('vm_map_rate_per_sec', 0):.1f} /sec")
        print(f"    IRQ rate:        {patterns.get('irq_rate_per_sec', 0):.1f} /sec")
        print()

        # Fault correlations
        if report["fault_correlations"]:
            print(_bold("  FAULT CORRELATIONS:"))
            for i, corr in enumerate(report["fault_correlations"]):
                print(f"    Fault #{i+1} ({corr['fault_type']}):")
                print(f"      Events in {corr['window_ms']}ms window: {corr['events_in_window']}")
                if corr["probable_causes"]:
                    print(f"      Probable causes: {', '.join(corr['probable_causes'])}")
                else:
                    print("      No clear kernel-level precursor events")
            print()

        # Eviction latencies
        latencies = report.get("evict_restore_latencies_us", [])
        if latencies:
            avg = sum(latencies) / len(latencies)
            mx = max(latencies)
            print(f"  EVICTION LATENCIES:")
            print(f"    Count: {len(latencies)}")
            print(f"    Avg:   {avg:.1f} us")
            print(f"    Max:   {mx:.1f} us")
            print()

        # Recommendations
        print(_bold("  RECOMMENDATIONS:"))
        for rec in report.get("recommendations", []):
            print(f"    - {rec}")
        print()

        print(_bold("=" * 60))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AMD GPU memory access debugger using eBPF + dmesg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  sudo python3 %(prog)s --pid 12345 --duration 300
  sudo python3 %(prog)s --duration 120
  sudo python3 %(prog)s --check-only
  sudo python3 %(prog)s --pid 12345 -d 60 -o report.json
""",
    )
    parser.add_argument("--pid", "-p", type=int, default=None,
                        help="PID of the target process (default: all)")
    parser.add_argument("--duration", "-d", type=float, default=60.0,
                        help="Capture duration in seconds (default: 60)")
    parser.add_argument("--output", "-o", default=None,
                        help="Save JSON report to file")
    parser.add_argument("--check-only", action="store_true",
                        help="Only run system check, no tracing")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print every eBPF event")

    args = parser.parse_args()

    if args.check_only:
        info = system_check()
        print_system_check(info)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(info, f, indent=2)
            print(f"System info saved to: {args.output}")
        return

    debugger = GPUMemoryDebugger(
        target_pid=args.pid,
        duration_s=args.duration,
        output_file=args.output,
        verbose=args.verbose,
    )
    debugger.run()


if __name__ == "__main__":
    main()
