"""Smoke tests for ``aorta.ebpf.events``.

These pin the public surface (enum values, dataclass shape) so that the
lower-level parser/runner tests keep meaning what they look like.
"""

from __future__ import annotations

from aorta.ebpf import KernelEvent, KernelEventType


class TestKernelEventType:
    def test_enum_values_match_log_tokens(self):
        # The parser keys patterns off these literal strings; if they
        # drift, parser tests will fail in confusing ways. Lock them
        # explicitly.
        assert KernelEventType.KFD_EVICT.value == "KFD_EVICT_QUEUES"
        assert KernelEventType.KFD_RESTORE.value == "KFD_RESTORE_QUEUES"
        assert KernelEventType.SVM_EVICT.value == "SVM_RANGE_EVICT_WORKER"
        assert KernelEventType.BO_MOVE.value == "BO_MOVE"
        assert KernelEventType.VM_FLUSH.value == "VM_FLUSH"
        assert KernelEventType.IOCTL_ERROR.value == "IOCTL_ERROR"

    def test_full_membership_is_stable(self):
        # Adding/removing event types should be a deliberate change with
        # parser + summary keymap updates. This keeps that contract tight.
        names = {member.name for member in KernelEventType}
        assert names == {
            "KFD_EVICT",
            "KFD_RESTORE",
            "SVM_EVICT",
            "BO_MOVE",
            "VM_FLUSH",
            "VM_UNMAP_TICK",
            "IOCTL_ERROR",
            "LONG_IOCTL",
            "MMAP",
            "MUNMAP",
            "VM_HANDLE_MOVED",
            "KFD_INTERRUPT",
            "SIGNAL",
            "HEARTBEAT",
            "TICK",
        }


class TestKernelEvent:
    def test_defaults(self):
        ev = KernelEvent(timestamp_ns=42, event_type=KernelEventType.HEARTBEAT)
        assert ev.timestamp_ns == 42
        assert ev.event_type is KernelEventType.HEARTBEAT
        assert ev.payload == {}
        assert ev.raw_line == ""

    def test_payload_is_independent_per_instance(self):
        a = KernelEvent(timestamp_ns=1, event_type=KernelEventType.TICK)
        b = KernelEvent(timestamp_ns=2, event_type=KernelEventType.TICK)
        a.payload["x"] = 1
        assert "x" not in b.payload, "default_factory must not share state"
