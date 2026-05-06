"""Tests for ``aorta.ebpf.parser.BpftraceLogParser``.

The parser is a list of regex patterns iterated in order; ordering is
load-bearing (specific patterns must match before their fallbacks). These
tests pin both the per-pattern behaviour and the ordering contract --
addressing review item C6.
"""

from __future__ import annotations

import pytest

from aorta.ebpf import BpftraceLogParser, KernelEventType


@pytest.fixture
def parser() -> BpftraceLogParser:
    return BpftraceLogParser()


class TestParseSingleEvents:
    def test_kfd_evict_with_comm(self, parser):
        ev = parser.parse_line("1700000000123 *** KFD_EVICT_QUEUES pid=4321 comm=python ***")
        assert ev is not None
        assert ev.event_type is KernelEventType.KFD_EVICT
        assert ev.timestamp_ns == 1700000000123
        assert ev.payload == {"pid": 4321, "comm": "python"}

    def test_kfd_restore(self, parser):
        ev = parser.parse_line("10 *** KFD_RESTORE_QUEUES pid=99 comm=foo ***")
        assert ev is not None
        assert ev.event_type is KernelEventType.KFD_RESTORE
        assert ev.payload["pid"] == 99

    def test_svm_evict(self, parser):
        ev = parser.parse_line("5 *** SVM_RANGE_EVICT_WORKER pid=11 comm=worker ***")
        assert ev is not None
        assert ev.event_type is KernelEventType.SVM_EVICT

    def test_bo_move_amdgpu_form(self, parser):
        ev = parser.parse_line("12 *** AMDGPU_BO_MOVE size=4096 old=2 new=4 ***")
        assert ev is not None
        assert ev.event_type is KernelEventType.BO_MOVE
        assert ev.payload == {"size": 4096, "old_placement": 2, "new_placement": 4}

    def test_bo_move_short_form(self, parser):
        ev = parser.parse_line("13 BO_MOVE size=128 old=1 new=3")
        assert ev is not None
        assert ev.event_type is KernelEventType.BO_MOVE
        assert ev.payload["size"] == 128

    def test_vm_flush_with_pd_addr(self, parser):
        ev = parser.parse_line("20 AMDGPU_VM_FLUSH vmid=2 hub=0 pd_addr=0xffff800012345000")
        assert ev is not None
        assert ev.event_type is KernelEventType.VM_FLUSH
        assert ev.payload["pd_addr"] == "0xffff800012345000"
        assert ev.payload["vmid"] == 2

    def test_tick_with_openats(self, parser):
        ev = parser.parse_line("99 TICK unmaps=3 ptes=44 openats=7")
        assert ev is not None
        assert ev.event_type is KernelEventType.TICK
        assert ev.payload == {"unmaps": 3, "ptes": 44, "openats": 7}

    def test_tick_without_openats(self, parser):
        ev = parser.parse_line("99 TICK unmaps=3 ptes=44")
        assert ev is not None
        assert ev.event_type is KernelEventType.TICK
        assert ev.payload == {"unmaps": 3, "ptes": 44}

    def test_ioctl_error_long_form(self, parser):
        ev = parser.parse_line("31 IOCTL_ERROR cmd=0xc0186444 ret=-22 dur=120us")
        assert ev is not None
        assert ev.event_type is KernelEventType.IOCTL_ERROR
        assert ev.payload == {"cmd": "0xc0186444", "ret": -22, "dur_us": 120}

    def test_ioctl_error_short_form(self, parser):
        ev = parser.parse_line("32 IOCTL_ERR ret=-5")
        assert ev is not None
        assert ev.event_type is KernelEventType.IOCTL_ERROR
        assert ev.payload == {"ret": -5}

    def test_long_ioctl(self, parser):
        ev = parser.parse_line("40 LONG_IOCTL cmd=0x1234 dur=2500us")
        assert ev is not None
        assert ev.event_type is KernelEventType.LONG_IOCTL
        assert ev.payload == {"cmd": "0x1234", "dur_us": 2500}

    def test_mmap_munmap(self, parser):
        mmap = parser.parse_line("50 MMAP len=4096 prot=0x3 flags=0x22")
        assert mmap is not None
        assert mmap.event_type is KernelEventType.MMAP
        assert mmap.payload == {"len": 4096, "prot": "0x3", "flags": "0x22"}

        munmap = parser.parse_line("51 MUNMAP addr=0x7fff00000000 len=2048")
        assert munmap is not None
        assert munmap.event_type is KernelEventType.MUNMAP
        assert munmap.payload == {"addr": "0x7fff00000000", "len": 2048}

    def test_signal_long_and_short_form(self, parser):
        long_form = parser.parse_line("60 SIGNAL sig=11")
        short_form = parser.parse_line("61 SIG=9")
        assert long_form is not None and long_form.event_type is KernelEventType.SIGNAL
        assert long_form.payload == {"sig": 11}
        assert short_form is not None
        assert short_form.event_type is KernelEventType.SIGNAL
        assert short_form.payload == {"sig": 9}

    def test_heartbeat(self, parser):
        ev = parser.parse_line("70 HEARTBEAT alive")
        assert ev is not None
        assert ev.event_type is KernelEventType.HEARTBEAT
        assert ev.payload == {}


class TestUnknownAndEmpty:
    def test_blank_line_returns_none(self, parser):
        assert parser.parse_line("") is None
        assert parser.parse_line("    \n") is None

    def test_unknown_line_returns_none(self, parser):
        assert parser.parse_line("Attaching 12 probes...") is None
        assert parser.parse_line("WARNING: missing kernel symbol") is None


class TestRawLineCapture:
    def test_raw_line_is_stripped_form(self, parser):
        ev = parser.parse_line("  10 HEARTBEAT alive  \n")
        assert ev is not None
        assert ev.raw_line == "10 HEARTBEAT alive"


class TestPatternOrderingIsLoadBearing:
    """C6: pattern order matters because more-specific patterns must win.

    The parser declares `_PATTERNS` as a list iterated in-order. If a
    refactor accidentally re-orders or de-duplicates entries, callers
    silently lose ``comm=`` or ``pd_addr=`` payload fields. These tests
    pin that contract.
    """

    def test_specific_kfd_evict_with_comm_wins_over_short_form(self, parser):
        # Both patterns match a "KFD_EVICT_QUEUES pid=N" prefix; only the
        # specific one captures `comm`. If the bare-pid pattern moved
        # ahead of it, `comm` would silently disappear.
        ev = parser.parse_line("1 *** KFD_EVICT_QUEUES pid=42 comm=trainer ***")
        assert ev is not None
        assert ev.payload.get("comm") == "trainer", (
            "with-comm pattern must precede the bare-pid pattern in _PATTERNS"
        )

    def test_amdgpu_bo_move_form_wins_over_short_form(self, parser):
        # Both ``*** AMDGPU_BO_MOVE size=... ***`` and the bare ``BO_MOVE
        # size=...`` line should classify as BO_MOVE; the bare form is a
        # fallback for stripped-down scripts. Verify both still win the
        # right class regardless of which appears first in input.
        amdgpu = parser.parse_line("1 *** AMDGPU_BO_MOVE size=64 old=1 new=2 ***")
        bare = parser.parse_line("2 BO_MOVE size=64 old=1 new=2")
        assert amdgpu is not None and amdgpu.event_type is KernelEventType.BO_MOVE
        assert bare is not None and bare.event_type is KernelEventType.BO_MOVE

    def test_long_form_ioctl_error_keeps_dur_us(self, parser):
        # If ``IOCTL_ERR ret=-N`` (short form) preceded the long form
        # in _PATTERNS, the dur_us would be lost on long-form lines.
        ev = parser.parse_line("3 IOCTL_ERROR cmd=0x1 ret=-22 dur=99us")
        assert ev is not None
        assert "dur_us" in ev.payload, (
            "long-form IOCTL_ERROR pattern must precede the short ret-only one"
        )
        assert ev.payload["dur_us"] == 99
