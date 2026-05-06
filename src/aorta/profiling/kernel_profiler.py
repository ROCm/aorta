"""Per-iteration kernel-event profiler backed by bpftrace.

``KernelTraceProfiler`` is the user-space peer of ``StreamProfiler``: it
manages a long-lived ``BpftraceRunner`` and slices the resulting kernel
events into per-iteration buckets so they can be merged into the same
JSONL metrics stream produced by training loops.

Lifecycle (mirrors ``StreamProfiler``):

    profiler = KernelTraceProfiler(KernelTraceConfig(target_pid=os.getpid()))
    profiler.start()
    for step in range(N):
        profiler.start_iteration(step)
        # ... training work ...
        record = profiler.end_iteration()
    profiler.stop()

The per-iteration record schema:

    {
      "index": <step>,
      "elapsed_ns": <int>,
      "events": [<KernelEvent.__dict__>, ...],
      "summary": {
        "kfd_evict": <int>,
        "kfd_restore": <int>,
        "svm_evict": <int>,
        "bo_move": <int>,
        "vm_flush": <int>,
        "ioctl_error": <int>,
        "signal": <int>,
        "pte_updates": <int>,
        "vm_unmaps": <int>,
      },
    }
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from aorta.ebpf import (
    BpftraceConfig,
    BpftraceRunner,
    BpftraceScriptVariant,
    KernelEvent,
    KernelEventType,
)

log = logging.getLogger(__name__)


@dataclass
class KernelTraceConfig:
    """Configuration for a ``KernelTraceProfiler``.

    Attributes:
        enabled: If False, the profiler is a no-op (useful for guarded
            training loops where the flag is config-driven).
        target_pid: PID to attach to. Defaults to the current process.
        variant: bpftrace script variant; ``TP_ONLY`` is the recommended
            default to minimise the Heisenberg effect on GPU memory races.
        use_sudo: Whether to prefix the bpftrace command with ``sudo``.
        bpftrace_path: Optional override for the bpftrace binary path.
        keep_raw_events: If False, the per-iteration record only stores
            the aggregated summary, not the full event list. Useful for
            long runs where event volume could blow up the JSONL file.
        skip_if_unavailable: If True, the profiler logs a warning and
            silently disables itself when bpftrace is missing rather than
            raising. Defaults to True so kernel tracing remains an opt-in
            best-effort feature.
    """

    enabled: bool = False
    target_pid: int | None = None
    variant: BpftraceScriptVariant = BpftraceScriptVariant.TP_ONLY
    use_sudo: bool = True
    bpftrace_path: str | None = None
    keep_raw_events: bool = False
    skip_if_unavailable: bool = True


@dataclass
class _IterationState:
    index: int
    start_ns: int
    events: list[KernelEvent] = field(default_factory=list)


_SUMMARY_COUNTER_TYPES: dict[str, KernelEventType] = {
    "kfd_evict": KernelEventType.KFD_EVICT,
    "kfd_restore": KernelEventType.KFD_RESTORE,
    "svm_evict": KernelEventType.SVM_EVICT,
    "bo_move": KernelEventType.BO_MOVE,
    "vm_flush": KernelEventType.VM_FLUSH,
    "ioctl_error": KernelEventType.IOCTL_ERROR,
    "signal": KernelEventType.SIGNAL,
    "vm_handle_moved": KernelEventType.VM_HANDLE_MOVED,
    "kfd_interrupt": KernelEventType.KFD_INTERRUPT,
    "long_ioctl": KernelEventType.LONG_IOCTL,
    "mmap": KernelEventType.MMAP,
    "munmap": KernelEventType.MUNMAP,
}


class KernelTraceProfiler:
    """Bucket bpftrace kernel events by training iteration.

    Designed to run alongside ``StreamProfiler`` with the same call
    pattern; output is appended to the same per-iteration metrics dict
    written by ``MetricsLogger``.

    The profiler is safe to construct with ``enabled=False`` -- in that
    mode every method becomes a no-op and ``end_iteration`` returns
    ``None``. This makes it cheap to wire into existing training loops
    behind a configuration flag.
    """

    def __init__(self, config: KernelTraceConfig) -> None:
        self.config = config
        self._runner: BpftraceRunner | None = None
        self._iteration: _IterationState | None = None
        self._started = False
        self._all_events: list[KernelEvent] = []

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self._runner is not None

    def start(self) -> None:
        """Spawn the bpftrace process. Safe to call when disabled."""
        if not self.config.enabled:
            log.debug("KernelTraceProfiler disabled; skipping start")
            return
        if self._started:
            return

        if not BpftraceRunner.is_bpftrace_available(self.config.bpftrace_path):
            msg = "bpftrace binary not available; kernel tracing will be skipped"
            if self.config.skip_if_unavailable:
                log.warning(msg)
                return
            raise RuntimeError(msg)

        import os

        target_pid = self.config.target_pid or os.getpid()
        bpftrace_cfg = BpftraceConfig(
            target_pid=target_pid,
            variant=self.config.variant,
            use_sudo=self.config.use_sudo,
            bpftrace_path=self.config.bpftrace_path,
        )
        runner = BpftraceRunner(bpftrace_cfg)
        try:
            runner.start()
        except Exception as exc:
            if self.config.skip_if_unavailable:
                log.warning("Failed to start bpftrace (%s); kernel tracing disabled", exc)
                return
            raise

        self._runner = runner
        self._started = True
        log.info(
            "KernelTraceProfiler started | pid=%d variant=%s",
            target_pid,
            self.config.variant.name,
        )

    def stop(self) -> list[KernelEvent]:
        """Stop the bpftrace process and return any remaining events."""
        if self._runner is None:
            return []
        events = self._runner.stop()
        self._all_events.extend(events)
        self._runner = None
        self._started = False
        return events

    def start_iteration(self, index: int) -> None:
        if not self.enabled:
            return
        if self._iteration is not None:
            raise RuntimeError("Previous kernel-trace iteration not finalised")
        # Drain any events queued before the iteration to avoid attributing
        # them to this step.
        assert self._runner is not None
        self._runner.drain_events()
        self._iteration = _IterationState(index=index, start_ns=time.monotonic_ns())

    def end_iteration(self) -> dict[str, Any] | None:
        if not self.enabled or self._iteration is None:
            self._iteration = None
            return None
        assert self._runner is not None

        end_ns = time.monotonic_ns()
        events = self._runner.drain_events()
        self._iteration.events = events
        self._all_events.extend(events)

        record = self._serialize_iteration(self._iteration, end_ns)
        self._iteration = None
        return record

    def _serialize_iteration(self, state: _IterationState, end_ns: int) -> dict[str, Any]:
        summary: dict[str, int] = dict.fromkeys(_SUMMARY_COUNTER_TYPES, 0)
        summary["pte_updates"] = 0
        summary["vm_unmaps"] = 0

        for event in state.events:
            for key, ev_type in _SUMMARY_COUNTER_TYPES.items():
                if event.event_type is ev_type:
                    summary[key] += 1
                    break
            if event.event_type in (
                KernelEventType.TICK,
                KernelEventType.VM_UNMAP_TICK,
            ):
                summary["pte_updates"] += int(
                    event.payload.get("ptes") or event.payload.get("pte_count") or 0
                )
                summary["vm_unmaps"] += int(
                    event.payload.get("unmaps") or event.payload.get("unmap_count") or 0
                )

        record: dict[str, Any] = {
            "index": state.index,
            "elapsed_ns": end_ns - state.start_ns,
            "summary": summary,
        }
        if self.config.keep_raw_events:
            record["events"] = [_event_to_dict(ev) for ev in state.events]
        else:
            record["event_count"] = len(state.events)
        return record

    def all_events(self) -> list[KernelEvent]:
        """Return a copy of every event observed across the profiler's life."""
        return list(self._all_events)

    def __enter__(self) -> KernelTraceProfiler:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def _event_to_dict(event: KernelEvent) -> dict[str, Any]:
    data = asdict(event)
    data["event_type"] = event.event_type.value
    return data


__all__ = ["KernelTraceConfig", "KernelTraceProfiler"]
