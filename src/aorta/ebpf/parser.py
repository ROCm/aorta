"""Parse structured log lines emitted by the vendored bpftrace scripts."""

from __future__ import annotations

import logging
import re

from .events import KernelEvent, KernelEventType

log = logging.getLogger(__name__)

_TS = r"(\d+)"

_PATTERNS: list[tuple[re.Pattern[str], KernelEventType, list[str]]] = [
    (
        re.compile(rf"^{_TS} \*\*\* KFD_EVICT_QUEUES pid=(\d+) comm=(\S+) \*\*\*"),
        KernelEventType.KFD_EVICT,
        ["pid", "comm"],
    ),
    (
        re.compile(rf"^{_TS} \*\*\* KFD_RESTORE_QUEUES pid=(\d+) comm=(\S+) \*\*\*"),
        KernelEventType.KFD_RESTORE,
        ["pid", "comm"],
    ),
    (
        re.compile(rf"^{_TS} \*\*\* SVM_RANGE_EVICT_WORKER pid=(\d+) comm=(\S+) \*\*\*"),
        KernelEventType.SVM_EVICT,
        ["pid", "comm"],
    ),
    (
        re.compile(rf"^{_TS} \*\*\* KFD_EVICT_QUEUES pid=(\d+) \*\*\*"),
        KernelEventType.KFD_EVICT,
        ["pid"],
    ),
    (
        re.compile(rf"^{_TS} \*\*\* AMDGPU_BO_MOVE size=(\d+) old=(\d+) new=(\d+) \*\*\*"),
        KernelEventType.BO_MOVE,
        ["size", "old_placement", "new_placement"],
    ),
    (
        re.compile(rf"^{_TS} BO_MOVE size=(\d+) old=(\d+) new=(\d+)"),
        KernelEventType.BO_MOVE,
        ["size", "old_placement", "new_placement"],
    ),
    (
        re.compile(rf"^{_TS} AMDGPU_VM_FLUSH vmid=(\d+) hub=(\d+) pd_addr=(0x[0-9a-fA-F]+)"),
        KernelEventType.VM_FLUSH,
        ["vmid", "hub", "pd_addr"],
    ),
    (
        re.compile(rf"^{_TS} VM_FLUSH vmid=(\d+) hub=(\d+)"),
        KernelEventType.VM_FLUSH,
        ["vmid", "hub"],
    ),
    (
        re.compile(rf"^{_TS} VM_UNMAP_COUNT=(\d+) VM_PTE_UPDATE_COUNT=(\d+)"),
        KernelEventType.VM_UNMAP_TICK,
        ["unmap_count", "pte_count"],
    ),
    (
        re.compile(rf"^{_TS} TICK unmaps=(\d+) ptes=(\d+) openats=(\d+)"),
        KernelEventType.TICK,
        ["unmaps", "ptes", "openats"],
    ),
    (
        re.compile(rf"^{_TS} TICK unmaps=(\d+) ptes=(\d+)"),
        KernelEventType.TICK,
        ["unmaps", "ptes"],
    ),
    (
        re.compile(rf"^{_TS} IOCTL_ERROR cmd=(0x[0-9a-fA-F]+) ret=(-?\d+) dur=(\d+)us"),
        KernelEventType.IOCTL_ERROR,
        ["cmd", "ret", "dur_us"],
    ),
    (
        re.compile(rf"^{_TS} IOCTL_ERR ret=(-?\d+)"),
        KernelEventType.IOCTL_ERROR,
        ["ret"],
    ),
    (
        re.compile(rf"^{_TS} LONG_IOCTL cmd=(0x[0-9a-fA-F]+) dur=(\d+)us"),
        KernelEventType.LONG_IOCTL,
        ["cmd", "dur_us"],
    ),
    (
        re.compile(rf"^{_TS} MMAP len=(\d+) prot=(0x[0-9a-fA-F]+) flags=(0x[0-9a-fA-F]+)"),
        KernelEventType.MMAP,
        ["len", "prot", "flags"],
    ),
    (
        re.compile(rf"^{_TS} MUNMAP addr=(0x[0-9a-fA-F]+) len=(\d+)"),
        KernelEventType.MUNMAP,
        ["addr", "len"],
    ),
    (
        re.compile(rf"^{_TS} AMDGPU_VM_HANDLE_MOVED pid=(\d+)"),
        KernelEventType.VM_HANDLE_MOVED,
        ["pid"],
    ),
    (
        re.compile(rf"^{_TS} KFD_INTERRUPT pid=(\d+)"),
        KernelEventType.KFD_INTERRUPT,
        ["pid"],
    ),
    (
        re.compile(rf"^{_TS} SIGNAL sig=(\d+)"),
        KernelEventType.SIGNAL,
        ["sig"],
    ),
    (
        re.compile(rf"^{_TS} SIG=(\d+)"),
        KernelEventType.SIGNAL,
        ["sig"],
    ),
    (
        re.compile(rf"^{_TS} HEARTBEAT alive"),
        KernelEventType.HEARTBEAT,
        [],
    ),
]


class BpftraceLogParser:
    """Stateless parser that converts bpftrace stdout lines into typed events.

    Handles all vendored script variants (full, light, tp_only, 1kprobe,
    unrelated_kprobe) by trying patterns in priority order.
    """

    def parse_line(self, line: str) -> KernelEvent | None:
        """Parse a single bpftrace output line.

        Returns a ``KernelEvent`` if the line matches a known pattern,
        or ``None`` for unrecognised / blank lines.
        """
        stripped = line.strip()
        if not stripped:
            return None

        for pattern, event_type, field_names in _PATTERNS:
            m = pattern.match(stripped)
            if m is None:
                continue

            groups = m.groups()
            ts_ns = int(groups[0])
            payload: dict[str, object] = {}
            for i, name in enumerate(field_names):
                raw = groups[i + 1]
                if raw.startswith("0x"):
                    payload[name] = raw
                else:
                    try:
                        payload[name] = int(raw)
                    except ValueError:
                        payload[name] = raw

            return KernelEvent(
                timestamp_ns=ts_ns,
                event_type=event_type,
                payload=payload,
                raw_line=stripped,
            )

        log.debug("Unrecognised bpftrace line: %s", stripped)
        return None


__all__ = ["BpftraceLogParser"]
