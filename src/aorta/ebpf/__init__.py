"""eBPF kernel-tracing integration for AORTA.

This package wraps the bpftrace scripts originally developed in the
``ebpfaultline`` project, exposing them as a reusable Python API so that
any AORTA workload can attach kernel-level GPU/KFD observability alongside
the existing user-space profilers (``StreamProfiler``, ``ROCmProfiler``).

Public surface:
    - ``BpftraceRunner`` -- subprocess lifecycle wrapper for bpftrace.
    - ``BpftraceConfig`` -- runner configuration dataclass.
    - ``BpftraceScriptVariant`` -- enum of vendored ``.bt`` script variants.
    - ``BpftraceLogParser`` -- structured parser for bpftrace stdout.
    - ``KernelEvent`` / ``KernelEventType`` -- typed event records.

The vendored scripts under ``scripts/`` target AMD ROCm KFD/amdgpu paths;
probe symbols may need adjustment for non-tested kernel versions.
"""

from .events import KernelEvent, KernelEventType
from .parser import BpftraceLogParser
from .runner import (
    SCRIPTS_DIR,
    BpftraceConfig,
    BpftraceRunner,
    BpftraceScriptVariant,
    BpftraceUnavailableError,
)

__all__ = [
    "BpftraceConfig",
    "BpftraceLogParser",
    "BpftraceRunner",
    "BpftraceScriptVariant",
    "BpftraceUnavailableError",
    "KernelEvent",
    "KernelEventType",
    "SCRIPTS_DIR",
]
