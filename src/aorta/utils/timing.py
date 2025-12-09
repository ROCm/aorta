"""GPU timing utilities using CUDA/HIP events.

This module provides utilities for:
- Event-based GPU timing
- Timing context managers
- Multi-stream timing coordination
- CPU-side timing
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import torch


@dataclass
class EventTiming:
    """Timing data from a pair of CUDA/HIP events."""

    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    stream_id: int
    kernel_name: Optional[str] = None

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds between start and end events."""
        return self.start_event.elapsed_time(self.end_event)


@dataclass
class TimingContext:
    """Context manager for timing GPU operations.

    Usage:
        with TimingContext(stream, stream_id=0) as ctx:
            # GPU operations
            ...
        elapsed = ctx.elapsed_ms()
    """

    stream: torch.cuda.Stream
    stream_id: int = 0
    kernel_name: Optional[str] = None
    _start_event: torch.cuda.Event = field(init=False)
    _end_event: torch.cuda.Event = field(init=False)
    _completed: bool = field(init=False, default=False)

    def __post_init__(self):
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self) -> "TimingContext":
        self._start_event.record(self.stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._end_event.record(self.stream)
        self._completed = True

    def elapsed_ms(self, sync: bool = True) -> float:
        """Get elapsed time in milliseconds.

        Args:
            sync: If True, synchronize the stream before computing elapsed time

        Returns:
            Elapsed time in milliseconds
        """
        if not self._completed:
            raise RuntimeError("Cannot get elapsed time before context manager exits")
        if sync:
            self._end_event.synchronize()
        return self._start_event.elapsed_time(self._end_event)

    def to_event_timing(self) -> EventTiming:
        """Convert to EventTiming object."""
        return EventTiming(
            start_event=self._start_event,
            end_event=self._end_event,
            stream_id=self.stream_id,
            kernel_name=self.kernel_name,
        )


class StreamTimer:
    """Timer for recording multiple kernel timings across streams.

    Usage:
        timer = StreamTimer(num_streams=4)

        # In workload execution
        start, end = timer.create_events(stream_id=0)
        start.record(stream)
        # ... GPU operations ...
        end.record(stream)

        # After synchronization
        timings = timer.get_all_timings()
    """

    def __init__(self, num_streams: int):
        self.num_streams = num_streams
        self._events: List[List[Tuple[torch.cuda.Event, torch.cuda.Event, Optional[str]]]] = [
            [] for _ in range(num_streams)
        ]

    def create_events(
        self, stream_id: int, kernel_name: Optional[str] = None
    ) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        """Create a pair of timing events for a stream.

        Args:
            stream_id: Index of the stream
            kernel_name: Optional name for the kernel being timed

        Returns:
            Tuple of (start_event, end_event)
        """
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        self._events[stream_id].append((start, end, kernel_name))
        return start, end

    def get_stream_timings(self, stream_id: int) -> List[Tuple[float, Optional[str]]]:
        """Get all timings for a specific stream.

        Args:
            stream_id: Index of the stream

        Returns:
            List of (elapsed_ms, kernel_name) tuples
        """
        timings = []
        for start, end, name in self._events[stream_id]:
            elapsed = start.elapsed_time(end)
            timings.append((elapsed, name))
        return timings

    def get_all_timings(self) -> List[List[Tuple[float, Optional[str]]]]:
        """Get all timings for all streams.

        Returns:
            List of stream timings, where each element is a list of (elapsed_ms, kernel_name)
        """
        return [self.get_stream_timings(i) for i in range(self.num_streams)]

    def get_total_time_per_stream(self) -> List[float]:
        """Get total elapsed time for each stream."""
        totals = []
        for stream_id in range(self.num_streams):
            stream_total = sum(t[0] for t in self.get_stream_timings(stream_id))
            totals.append(stream_total)
        return totals

    def clear(self) -> None:
        """Clear all recorded events."""
        self._events = [[] for _ in range(self.num_streams)]


class CPUTimer:
    """Simple CPU-side timer for host-side measurements."""

    def __init__(self):
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start(self) -> None:
        """Start the timer."""
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop the timer."""
        self._end_time = time.perf_counter()

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self._start_time is None or self._end_time is None:
            raise RuntimeError("Timer not started/stopped")
        return (self._end_time - self._start_time) * 1000

    @contextmanager
    def measure(self) -> Generator["CPUTimer", None, None]:
        """Context manager for CPU timing."""
        self.start()
        try:
            yield self
        finally:
            self.stop()


__all__ = [
    "EventTiming",
    "TimingContext",
    "StreamTimer",
    "CPUTimer",
]
