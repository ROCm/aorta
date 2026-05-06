"""Tests for ``aorta.profiling.kernel_profiler.KernelTraceProfiler``.

The profiler is loaded directly via ``importlib.util`` to bypass
``aorta.profiling.__init__``'s torch-dependent ``StreamProfiler`` import,
keeping the tests usable in venvs without PyTorch.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aorta.ebpf import KernelEvent, KernelEventType

_PROF_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "aorta" / "profiling" / "kernel_profiler.py"
)
_spec = importlib.util.spec_from_file_location("kernel_profiler", _PROF_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

KernelTraceConfig = _mod.KernelTraceConfig
KernelTraceProfiler = _mod.KernelTraceProfiler


def _make_event(event_type: KernelEventType, **payload) -> KernelEvent:
    return KernelEvent(
        timestamp_ns=0,
        event_type=event_type,
        payload=dict(payload),
        raw_line="",
    )


# ---------------------------------------------------------------------------
# enabled=False is a true no-op
# ---------------------------------------------------------------------------


class TestDisabledProfilerIsNoop:
    def test_start_does_nothing_when_disabled(self):
        profiler = KernelTraceProfiler(KernelTraceConfig(enabled=False))
        profiler.start()
        assert profiler.enabled is False

    def test_iteration_methods_no_op_when_disabled(self):
        profiler = KernelTraceProfiler(KernelTraceConfig(enabled=False))
        profiler.start()
        profiler.start_iteration(0)
        # end_iteration must return None and never crash, even with no
        # paired start_iteration on the disabled path.
        assert profiler.end_iteration() is None

    def test_stop_returns_empty_list_when_disabled(self):
        profiler = KernelTraceProfiler(KernelTraceConfig(enabled=False))
        profiler.start()
        assert profiler.stop() == []


# ---------------------------------------------------------------------------
# skip_if_unavailable: bpftrace missing should silently disable, not crash
# ---------------------------------------------------------------------------


class TestUnavailableHandling:
    def test_skip_if_unavailable_disables_silently(self, monkeypatch):
        # Simulate a host without bpftrace by stubbing
        # BpftraceRunner.is_bpftrace_available -> False.
        from aorta.ebpf import runner as runner_mod

        monkeypatch.setattr(
            runner_mod.BpftraceRunner, "is_bpftrace_available", staticmethod(lambda *a, **kw: False)
        )
        profiler = KernelTraceProfiler(
            KernelTraceConfig(enabled=True, skip_if_unavailable=True, target_pid=os.getpid())
        )
        # Should not raise; profiler.enabled becomes False because the
        # runner never started.
        profiler.start()
        assert profiler.enabled is False
        # end_iteration on the disabled path returns None, not a record.
        assert profiler.end_iteration() is None

    def test_skip_if_unavailable_false_raises(self, monkeypatch):
        from aorta.ebpf import runner as runner_mod

        monkeypatch.setattr(
            runner_mod.BpftraceRunner, "is_bpftrace_available", staticmethod(lambda *a, **kw: False)
        )
        profiler = KernelTraceProfiler(
            KernelTraceConfig(enabled=True, skip_if_unavailable=False, target_pid=os.getpid())
        )
        with pytest.raises(RuntimeError, match="bpftrace binary not available"):
            profiler.start()


# ---------------------------------------------------------------------------
# Per-iteration bucketing (with a fake runner injected)
# ---------------------------------------------------------------------------


class _FakeRunner:
    """Stand-in for ``BpftraceRunner`` controlled from the test."""

    def __init__(self):
        self._queue: list[KernelEvent] = []
        self.start_called = False
        self.stop_called = False

    def queue(self, *events: KernelEvent) -> None:
        self._queue.extend(events)

    def start(self) -> None:
        self.start_called = True

    def stop(self):
        self.stop_called = True
        events, self._queue = self._queue, []
        return events

    def drain_events(self):
        events, self._queue = self._queue, []
        return events


@pytest.fixture
def injected_profiler(monkeypatch):
    """Yield a profiler whose runner is a controllable ``_FakeRunner``."""
    fake = _FakeRunner()

    # Inject the fake by patching BpftraceRunner construction *and*
    # availability check within the kernel_profiler module's import scope.
    monkeypatch.setattr(_mod, "BpftraceRunner", MagicMock(return_value=fake))
    # `is_bpftrace_available` is called as a staticmethod on the original
    # class; patch it on the MagicMock too.
    _mod.BpftraceRunner.is_bpftrace_available = staticmethod(lambda *a, **kw: True)

    profiler = KernelTraceProfiler(
        KernelTraceConfig(enabled=True, skip_if_unavailable=False, target_pid=os.getpid())
    )
    profiler.start()
    yield profiler, fake
    profiler.stop()


class TestIterationBucketing:
    def test_summary_counts_per_iteration(self, injected_profiler):
        profiler, fake = injected_profiler

        profiler.start_iteration(0)
        fake.queue(
            _make_event(KernelEventType.KFD_EVICT, pid=1),
            _make_event(KernelEventType.KFD_EVICT, pid=2),
            _make_event(KernelEventType.SVM_EVICT, pid=3),
            _make_event(KernelEventType.BO_MOVE, size=4096),
        )
        record = profiler.end_iteration()
        assert record is not None
        assert record["index"] == 0
        assert record["summary"]["kfd_evict"] == 2
        assert record["summary"]["svm_evict"] == 1
        assert record["summary"]["bo_move"] == 1
        # No raw events kept by default.
        assert "events" not in record
        assert record["event_count"] == 4

    def test_tick_events_aggregate_into_pte_and_unmap_counters(self, injected_profiler):
        profiler, fake = injected_profiler

        profiler.start_iteration(7)
        fake.queue(
            _make_event(KernelEventType.TICK, unmaps=2, ptes=10),
            _make_event(KernelEventType.TICK, unmaps=3, ptes=20),
            _make_event(KernelEventType.VM_UNMAP_TICK, unmap_count=4, pte_count=40),
        )
        record = profiler.end_iteration()
        assert record is not None
        assert record["summary"]["pte_updates"] == 70
        assert record["summary"]["vm_unmaps"] == 9

    def test_keep_raw_events_serialises_event_list(self, monkeypatch):
        fake = _FakeRunner()
        monkeypatch.setattr(_mod, "BpftraceRunner", MagicMock(return_value=fake))
        _mod.BpftraceRunner.is_bpftrace_available = staticmethod(lambda *a, **kw: True)

        profiler = KernelTraceProfiler(
            KernelTraceConfig(
                enabled=True,
                skip_if_unavailable=False,
                target_pid=os.getpid(),
                keep_raw_events=True,
            )
        )
        profiler.start()
        try:
            profiler.start_iteration(0)
            fake.queue(_make_event(KernelEventType.HEARTBEAT))
            record = profiler.end_iteration()
        finally:
            profiler.stop()

        assert record is not None
        assert record["events"], "keep_raw_events=True must serialise events list"
        assert record["events"][0]["event_type"] == "HEARTBEAT"

    def test_double_start_iteration_raises(self, injected_profiler):
        profiler, _ = injected_profiler
        profiler.start_iteration(0)
        with pytest.raises(RuntimeError, match="not finalised"):
            profiler.start_iteration(1)

    def test_pre_iteration_events_are_drained_not_attributed(self, injected_profiler):
        # If the profiler started and bpftrace queued events before
        # start_iteration(0), those events must NOT be charged to step 0.
        profiler, fake = injected_profiler

        # Pre-iteration noise
        fake.queue(_make_event(KernelEventType.KFD_EVICT, pid=99))

        profiler.start_iteration(0)
        # No new events for this iteration
        record = profiler.end_iteration()
        assert record is not None
        assert record["summary"]["kfd_evict"] == 0, (
            "events queued before start_iteration must be drained, not credited to the iteration"
        )


class TestStopFlushesEvents:
    def test_stop_returns_remaining_runner_events(self, monkeypatch):
        fake = _FakeRunner()
        monkeypatch.setattr(_mod, "BpftraceRunner", MagicMock(return_value=fake))
        _mod.BpftraceRunner.is_bpftrace_available = staticmethod(lambda *a, **kw: True)

        profiler = KernelTraceProfiler(
            KernelTraceConfig(enabled=True, skip_if_unavailable=False, target_pid=os.getpid())
        )
        profiler.start()
        fake.queue(_make_event(KernelEventType.HEARTBEAT))
        leftover = profiler.stop()
        assert len(leftover) == 1
        assert profiler.all_events()[-1].event_type is KernelEventType.HEARTBEAT
