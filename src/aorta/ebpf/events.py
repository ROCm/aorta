"""Kernel event types emitted by bpftrace log parsing."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class KernelEventType(enum.Enum):
    """Event types observed by the bpftrace GPU tracing scripts."""

    KFD_EVICT = "KFD_EVICT_QUEUES"
    KFD_RESTORE = "KFD_RESTORE_QUEUES"
    SVM_EVICT = "SVM_RANGE_EVICT_WORKER"
    BO_MOVE = "BO_MOVE"
    VM_FLUSH = "VM_FLUSH"
    VM_UNMAP_TICK = "VM_UNMAP_TICK"
    IOCTL_ERROR = "IOCTL_ERROR"
    LONG_IOCTL = "LONG_IOCTL"
    MMAP = "MMAP"
    MUNMAP = "MUNMAP"
    VM_HANDLE_MOVED = "VM_HANDLE_MOVED"
    KFD_INTERRUPT = "KFD_INTERRUPT"
    SIGNAL = "SIGNAL"
    HEARTBEAT = "HEARTBEAT"
    TICK = "TICK"


@dataclass
class KernelEvent:
    """A single parsed event from bpftrace output.

    Attributes:
        timestamp_ns: Kernel-clock timestamp in nanoseconds.
        event_type: Classified event type.
        payload: Event-specific key-value data (e.g. pid, size, placement).
        raw_line: The original unparsed log line.
    """

    timestamp_ns: int
    event_type: KernelEventType
    payload: dict[str, Any] = field(default_factory=dict)
    raw_line: str = ""


__all__ = ["KernelEventType", "KernelEvent"]
