"""Multi-stream profiling utilities with overlap computation."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

log = logging.getLogger(__name__)


StreamName = str


@dataclass
class RangeRecord:
    stream: StreamName
    tag: str
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarkerRecord:
    stream: StreamName
    tag: str
    event: torch.cuda.Event
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamProfiler:
    """Track activity across multiple CUDA/HIP streams with precise timing."""

    def __init__(self, device: torch.device, stream_names: Optional[Iterable[StreamName]] = None) -> None:
        if not torch.cuda.is_available():  # pragma: no cover - runtime guard
            raise RuntimeError("StreamProfiler requires CUDA/HIP availability")

        self.device = device
        names = list(stream_names or ("compute", "allreduce", "reducescatter", "aux"))
        if len(set(names)) != len(names):
            raise ValueError("Stream names must be unique")

        self.streams: Dict[StreamName, torch.cuda.Stream] = {
            name: torch.cuda.Stream(device=self.device) for name in names
        }

        self.iteration_records: List[Dict[str, Any]] = []
        self._current_iteration: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Iteration lifecycle
    # ------------------------------------------------------------------
    def start_iteration(self, iteration: int) -> None:
        if self._current_iteration is not None:
            raise RuntimeError("Previous iteration not finalized")
        start_event = torch.cuda.Event(enable_timing=True, blocking=False)
        start_event.record(torch.cuda.current_stream(self.device))
        self._current_iteration = {
            "index": iteration,
            "start_event": start_event,
            "ranges": [],
            "markers": [],
        }

    def end_iteration(self) -> Dict[str, Any]:
        if self._current_iteration is None:
            raise RuntimeError("No iteration is active")

        end_event = torch.cuda.Event(enable_timing=True, blocking=False)
        end_event.record(torch.cuda.current_stream(self.device))

        # Ensure all streams are complete before evaluating timings.
        for stream in self.streams.values():
            stream.synchronize()
        torch.cuda.synchronize(self.device)

        record = self._finalize_iteration(self._current_iteration, end_event)
        self.iteration_records.append(record)
        self._current_iteration = None
        return record

    # ------------------------------------------------------------------
    # Stream helpers
    # ------------------------------------------------------------------
    def stream(self, name: StreamName) -> torch.cuda.Stream:
        return self.streams[name]

    @contextlib.contextmanager
    def range(self, stream_name: StreamName, tag: str, *, use_stream_context: bool = True, metadata: Optional[Dict[str, Any]] = None):
        self._require_iteration()
        stream = self.streams[stream_name]
        start = torch.cuda.Event(enable_timing=True, blocking=False)
        end = torch.cuda.Event(enable_timing=True, blocking=False)

        context = (
            torch.cuda.stream(stream) if use_stream_context else contextlib.nullcontext(stream)
        )
        with context:
            start.record(stream)
            yield stream
            end.record(stream)

        self._register_range(stream_name, tag, start, end, metadata or {})

    def record_marker(self, stream_name: StreamName, tag: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._require_iteration()
        stream = self.streams[stream_name]
        event = torch.cuda.Event(enable_timing=True, blocking=False)
        event.record(stream)
        marker = MarkerRecord(stream=stream_name, tag=tag, event=event, metadata=metadata or {})
        self._current_iteration["markers"].append(marker)

    def register_external_range(
        self,
        stream_name: StreamName,
        tag: str,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._require_iteration()
        self._register_range(stream_name, tag, start_event, end_event, metadata or {})

    # ------------------------------------------------------------------
    # Distributed instrumentation
    # ------------------------------------------------------------------
    def intercept_distributed_ops(self) -> "DistributedOpsInterceptor":
        return DistributedOpsInterceptor(self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _register_range(
        self,
        stream_name: StreamName,
        tag: str,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
        metadata: Dict[str, Any],
    ) -> None:
        record = RangeRecord(stream=stream_name, tag=tag, start_event=start_event, end_event=end_event, metadata=metadata)
        self._current_iteration["ranges"].append(record)

    def _require_iteration(self) -> None:
        if self._current_iteration is None:
            raise RuntimeError("start_iteration must be called before recording ranges")

    def _finalize_iteration(self, iteration_state: Dict[str, Any], end_event: torch.cuda.Event) -> Dict[str, Any]:
        start_event: torch.cuda.Event = iteration_state["start_event"]
        ranges: List[RangeRecord] = iteration_state["ranges"]
        markers: List[MarkerRecord] = iteration_state["markers"]

        total_ms = start_event.elapsed_time(end_event)

        serialized_ranges = []
        for record in ranges:
            start_ms = start_event.elapsed_time(record.start_event)
            end_ms = start_event.elapsed_time(record.end_event)
            duration = max(end_ms - start_ms, 0.0)
            serialized_ranges.append(
                {
                    "stream": record.stream,
                    "tag": record.tag,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": duration,
                    "metadata": record.metadata,
                }
            )

        serialized_markers = []
        for marker in markers:
            time_ms = start_event.elapsed_time(marker.event)
            serialized_markers.append(
                {
                    "stream": marker.stream,
                    "tag": marker.tag,
                    "time_ms": time_ms,
                    "metadata": marker.metadata,
                }
            )

        overlap_summary = self._compute_overlap(serialized_ranges, total_ms)

        return {
            "index": iteration_state["index"],
            "total_ms": total_ms,
            "ranges": serialized_ranges,
            "markers": serialized_markers,
            "overlap": overlap_summary,
        }

    def _compute_overlap(self, ranges: List[Dict[str, Any]], total_ms: float) -> Dict[str, Any]:
        events: List[Tuple[float, str, str]] = []  # time, kind, stream
        for rng in ranges:
            events.append((rng["start_ms"], "start", rng["stream"]))
            events.append((rng["end_ms"], "end", rng["stream"]))
        events.sort(key=lambda item: (item[0], 0 if item[1] == "start" else 1))

        active_streams: set[str] = set()
        last_time = 0.0
        stream_durations = defaultdict(float)
        overlap_durations = defaultdict(float)
        active_segments: List[Dict[str, Any]] = []
        active_union = 0.0

        for time, kind, stream in events:
            delta = max(time - last_time, 0.0)
            if delta > 0:
                if active_streams:
                    active_union += delta
                    for active in active_streams:
                        stream_durations[active] += delta
                    if "compute" in active_streams:
                        if "allreduce" in active_streams:
                            overlap_durations["compute_allreduce"] += delta
                        if "reducescatter" in active_streams:
                            overlap_durations["compute_reducescatter"] += delta
                        if any(name in active_streams for name in ("allreduce", "reducescatter")):
                            overlap_durations["compute_comm"] += delta
                else:
                    overlap_durations["idle"] += delta

                active_segments.append(
                    {
                        "start_ms": last_time,
                        "end_ms": time,
                        "active_streams": sorted(active_streams),
                    }
                )

            if kind == "start":
                active_streams.add(stream)
            else:
                active_streams.discard(stream)
            last_time = time

        idle_total = max(total_ms - active_union, 0.0)
        overlap_durations.setdefault("idle", idle_total)

        return {
            "per_stream_ms": dict(stream_durations),
            "overlap_ms": dict(overlap_durations),
            "active_segments": active_segments,
            "utilization": {
                stream: stream_durations.get(stream, 0.0) / total_ms if total_ms > 0 else 0.0
                for stream in self.streams.keys()
            },
        }


class DistributedOpsInterceptor:
    """Monkey-patch key distributed collectives to bind them to profiling streams."""

    def __init__(self, profiler: StreamProfiler) -> None:
        self.profiler = profiler
        self._originals: Dict[str, Any] = {}

    def __enter__(self) -> None:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():  # pragma: no cover - runtime guard
            log.warning("Distributed backend not initialised; comm interception disabled")
            return None

        self._patch(dist, "all_reduce", "allreduce")
        if hasattr(dist, "reduce_scatter_tensor"):
            self._patch(dist, "reduce_scatter_tensor", "reducescatter")
        if hasattr(dist, "all_gather_into_tensor"):
            self._patch(dist, "all_gather_into_tensor", "aux")
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        # #region agent log
        import json
        import logging
        log = logging.getLogger(__name__)
        log.info("[INSTR_H1] DistributedOpsInterceptor.__exit__ | exc_type=%s", exc_type.__name__ if exc_type else None)
        try:
            with open("/apps/oyazdanb/.cursor/debug.log", "a") as f:
                f.write(json.dumps({"location": "stream_profiler.py:280", "message": "DistributedOpsInterceptor exit", "data": {"exc_type": exc_type.__name__ if exc_type else None, "exc_msg": str(exc)[:200] if exc else None}, "timestamp": __import__("time").time() * 1000, "sessionId": "debug-session", "hypothesisId": "H1"}) + "\n")
        except: pass
        # #endregion
        for module, name, original in self._originals.values():
            setattr(module, name, original)
        self._originals.clear()

    def _patch(self, module: Any, name: str, stream_name: str) -> None:
        if (module, name) in self._originals:
            return
        original = getattr(module, name)
        self._originals[(module, name)] = (module, name, original)

        def wrapper(*args, **kwargs):
            async_requested = kwargs.get("async_op", False)
            kwargs["async_op"] = True
            stream = self.profiler.stream(stream_name)
            start = torch.cuda.Event(enable_timing=True, blocking=False)
            end = torch.cuda.Event(enable_timing=True, blocking=False)
            with torch.cuda.stream(stream):
                start.record(stream)
                work = original(*args, **kwargs)
                end.record(stream)
            tag = kwargs.get("tag", name)
            metadata = {"function": name}
            self.profiler.register_external_range(stream_name, f"{tag}", start, end, metadata=metadata)
            if not async_requested:
                work.wait()
                return None
            return work

        setattr(module, name, wrapper)


__all__ = ["StreamProfiler", "DistributedOpsInterceptor"]
