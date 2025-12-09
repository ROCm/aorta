"""
Performance metrics collection and aggregation.

This module provides:
- LatencyMetrics: Statistical latency measurements
- SwitchLatencyMetrics: Queue switch overhead estimation
- MetricsCollector: Collection and aggregation of performance data
- Throughput computation utilities
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class LatencyMetrics:
    """Statistical latency metrics in milliseconds."""

    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float = 0.0
    count: int = 0

    @classmethod
    def from_samples(cls, samples: List[float]) -> "LatencyMetrics":
        """
        Compute latency metrics from a list of samples.

        Args:
            samples: List of latency values in milliseconds

        Returns:
            LatencyMetrics object with computed statistics
        """
        if not samples:
            return cls(
                mean_ms=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                std_ms=0.0,
                count=0,
            )

        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        def percentile(p: float) -> float:
            """Compute percentile using linear interpolation."""
            if n == 1:
                return sorted_samples[0]
            idx = (n - 1) * p / 100
            lower = int(idx)
            upper = min(lower + 1, n - 1)
            frac = idx - lower
            return sorted_samples[lower] * (1 - frac) + sorted_samples[upper] * frac

        return cls(
            mean_ms=statistics.mean(samples),
            p50_ms=percentile(50),
            p95_ms=percentile(95),
            p99_ms=percentile(99),
            min_ms=min(samples),
            max_ms=max(samples),
            std_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
            count=len(samples),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SwitchLatencyMetrics:
    """
    Metrics for measuring queue switch overhead.

    These metrics help identify the cost of switching between hardware queues:
    - inter_stream_gap_ms: Average gap between kernels on different streams
    - intra_stream_gap_ms: Average gap between kernels on the same stream
    - estimated_switch_overhead_ms: Difference suggests queue switch cost
    """

    inter_stream_gap_ms: float  # Average gap between kernels on different streams
    intra_stream_gap_ms: float  # Average gap between kernels on same stream
    estimated_switch_overhead_ms: float  # Difference suggests switch cost
    inter_stream_samples: int = 0
    intra_stream_samples: int = 0

    @classmethod
    def from_kernel_timings(
        cls,
        kernel_timings: List[Tuple[int, float, float]],  # (stream_id, start_ms, end_ms)
    ) -> "SwitchLatencyMetrics":
        """
        Compute switch latency metrics from kernel timing data.

        Args:
            kernel_timings: List of (stream_id, start_time_ms, end_time_ms) tuples
                           Times should be relative to a common baseline.

        Returns:
            SwitchLatencyMetrics with computed values
        """
        if len(kernel_timings) < 2:
            return cls(
                inter_stream_gap_ms=0.0,
                intra_stream_gap_ms=0.0,
                estimated_switch_overhead_ms=0.0,
            )

        # Sort by end time to find sequential kernel pairs
        sorted_timings = sorted(kernel_timings, key=lambda x: x[2])

        inter_stream_gaps = []
        intra_stream_gaps = []

        for i in range(1, len(sorted_timings)):
            prev_stream, prev_start, prev_end = sorted_timings[i - 1]
            curr_stream, curr_start, curr_end = sorted_timings[i]

            # Gap is the time between end of previous kernel and start of next
            gap = curr_start - prev_end

            # Only consider positive gaps (overlapping kernels don't count)
            if gap > 0:
                if prev_stream == curr_stream:
                    intra_stream_gaps.append(gap)
                else:
                    inter_stream_gaps.append(gap)

        inter_avg = statistics.mean(inter_stream_gaps) if inter_stream_gaps else 0.0
        intra_avg = statistics.mean(intra_stream_gaps) if intra_stream_gaps else 0.0

        return cls(
            inter_stream_gap_ms=inter_avg,
            intra_stream_gap_ms=intra_avg,
            estimated_switch_overhead_ms=max(0.0, inter_avg - intra_avg),
            inter_stream_samples=len(inter_stream_gaps),
            intra_stream_samples=len(intra_stream_gaps),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ThroughputMetrics:
    """Throughput metrics with configurable units."""

    value: float
    unit: str
    raw_count: int
    duration_sec: float

    @classmethod
    def compute(
        cls, count: int, duration_sec: float, unit: str = "ops/sec"
    ) -> "ThroughputMetrics":
        """
        Compute throughput from count and duration.

        Args:
            count: Number of operations/samples/tokens
            duration_sec: Total duration in seconds
            unit: Unit string for display

        Returns:
            ThroughputMetrics object
        """
        value = count / duration_sec if duration_sec > 0 else 0.0
        return cls(value=value, unit=unit, raw_count=count, duration_sec=duration_sec)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MemoryMetrics:
    """GPU memory usage metrics."""

    peak_allocated_gb: float
    peak_reserved_gb: float
    final_allocated_gb: float
    final_reserved_gb: float

    @classmethod
    def capture(cls, device: str = "cuda:0") -> "MemoryMetrics":
        """Capture current memory state."""
        device_idx = torch.device(device).index or 0
        return cls(
            peak_allocated_gb=torch.cuda.max_memory_allocated(device_idx) / (1024**3),
            peak_reserved_gb=torch.cuda.max_memory_reserved(device_idx) / (1024**3),
            final_allocated_gb=torch.cuda.memory_allocated(device_idx) / (1024**3),
            final_reserved_gb=torch.cuda.memory_reserved(device_idx) / (1024**3),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class KernelTiming:
    """Timing information for a single kernel execution."""

    stream_id: int
    kernel_name: Optional[str]
    start_ms: float  # Relative to baseline
    end_ms: float  # Relative to baseline
    duration_ms: float

    @property
    def elapsed_ms(self) -> float:
        return self.duration_ms


class MetricsCollector:
    """
    Collect and aggregate performance metrics across streams and iterations.

    This collector tracks:
    - Per-kernel timing data
    - Per-stream aggregate timing
    - Per-iteration timing
    - Overall throughput and latency metrics

    Usage:
        collector = MetricsCollector(num_streams=4)

        for iteration in range(num_iterations):
            collector.start_iteration()

            # Record kernel timings
            collector.record_kernel_timing(stream_id=0, start_event, end_event)

            collector.end_iteration()

        # Get aggregated metrics
        latency = collector.compute_latency_metrics()
        switch = collector.compute_switch_latency()
    """

    def __init__(self, num_streams: int, device: str = "cuda:0"):
        """
        Initialize the metrics collector.

        Args:
            num_streams: Number of streams being used
            device: Target device
        """
        self.num_streams = num_streams
        self.device = device

        # Timing data storage
        self._kernel_timings: List[List[KernelTiming]] = []  # Per-iteration kernel timings
        self._iteration_times_ms: List[float] = []
        self._per_stream_times_ms: List[List[float]] = []  # Per-stream times per iteration

        # Current iteration state
        self._current_iteration_kernels: List[KernelTiming] = []
        self._iteration_start_event: Optional[torch.cuda.Event] = None
        self._iteration_end_event: Optional[torch.cuda.Event] = None
        self._baseline_event: Optional[torch.cuda.Event] = None

        # Event pairs for kernel timing
        self._pending_events: List[
            Tuple[int, torch.cuda.Event, torch.cuda.Event, Optional[str]]
        ] = []

    def start_iteration(self) -> None:
        """Mark the start of an iteration."""
        self._current_iteration_kernels = []
        self._pending_events = []

        # Create baseline event for relative timing
        self._baseline_event = torch.cuda.Event(enable_timing=True)
        self._baseline_event.record()

        self._iteration_start_event = torch.cuda.Event(enable_timing=True)
        self._iteration_start_event.record()

    def record_kernel_timing(
        self,
        stream_id: int,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
        kernel_name: Optional[str] = None,
    ) -> None:
        """
        Record timing for a kernel execution.

        Args:
            stream_id: ID of the stream the kernel ran on
            start_event: Event recorded before kernel
            end_event: Event recorded after kernel
            kernel_name: Optional name/identifier for the kernel
        """
        self._pending_events.append((stream_id, start_event, end_event, kernel_name))

    def end_iteration(self, sync: bool = True) -> None:
        """
        Mark the end of an iteration and process timings.

        Args:
            sync: If True, synchronize device before processing timings
        """
        self._iteration_end_event = torch.cuda.Event(enable_timing=True)
        self._iteration_end_event.record()

        if sync:
            torch.cuda.synchronize(self.device)

        # Process pending kernel timings
        for stream_id, start_event, end_event, kernel_name in self._pending_events:
            # Get times relative to baseline
            start_ms = self._baseline_event.elapsed_time(start_event)
            end_ms = self._baseline_event.elapsed_time(end_event)
            duration_ms = start_event.elapsed_time(end_event)

            timing = KernelTiming(
                stream_id=stream_id,
                kernel_name=kernel_name,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=duration_ms,
            )
            self._current_iteration_kernels.append(timing)

        self._kernel_timings.append(self._current_iteration_kernels)

        # Record iteration time
        iteration_time = self._iteration_start_event.elapsed_time(self._iteration_end_event)
        self._iteration_times_ms.append(iteration_time)

        # Compute per-stream times for this iteration
        stream_times = [0.0] * self.num_streams
        for kt in self._current_iteration_kernels:
            if 0 <= kt.stream_id < self.num_streams:
                stream_times[kt.stream_id] += kt.duration_ms
        self._per_stream_times_ms.append(stream_times)

    def compute_latency_metrics(self) -> LatencyMetrics:
        """
        Compute latency metrics from recorded iteration times.

        Returns:
            LatencyMetrics object with computed statistics
        """
        return LatencyMetrics.from_samples(self._iteration_times_ms)

    def compute_switch_latency(self) -> SwitchLatencyMetrics:
        """
        Compute queue switch latency metrics.

        This analyzes kernel timing patterns across streams to estimate
        the overhead of switching between hardware queues.

        Returns:
            SwitchLatencyMetrics object with computed values
        """
        # Flatten all kernel timings from all iterations
        all_timings = []
        for iteration_kernels in self._kernel_timings:
            for kt in iteration_kernels:
                all_timings.append((kt.stream_id, kt.start_ms, kt.end_ms))

        return SwitchLatencyMetrics.from_kernel_timings(all_timings)

    def compute_throughput(
        self, count_per_iteration: int, unit: str = "ops/sec"
    ) -> ThroughputMetrics:
        """
        Compute throughput from recorded timings.

        Args:
            count_per_iteration: Number of operations/samples per iteration
            unit: Unit string for the throughput metric

        Returns:
            ThroughputMetrics object
        """
        total_count = count_per_iteration * len(self._iteration_times_ms)
        total_time_sec = sum(self._iteration_times_ms) / 1000.0

        return ThroughputMetrics.compute(total_count, total_time_sec, unit)

    def get_per_stream_times(self) -> List[List[float]]:
        """Get per-stream times for each iteration."""
        return self._per_stream_times_ms

    def get_iteration_times(self) -> List[float]:
        """Get per-iteration times in milliseconds."""
        return self._iteration_times_ms

    def get_total_time_ms(self) -> float:
        """Get total time across all iterations in milliseconds."""
        return sum(self._iteration_times_ms)

    def get_kernel_timings(self) -> List[List[KernelTiming]]:
        """Get all recorded kernel timings by iteration."""
        return self._kernel_timings

    def clear(self) -> None:
        """Clear all recorded data."""
        self._kernel_timings = []
        self._iteration_times_ms = []
        self._per_stream_times_ms = []
        self._current_iteration_kernels = []
        self._pending_events = []

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all collected metrics.

        Returns:
            Dictionary with metric summaries
        """
        latency = self.compute_latency_metrics()
        switch = self.compute_switch_latency()

        return {
            "iterations": len(self._iteration_times_ms),
            "total_time_ms": self.get_total_time_ms(),
            "latency": latency.to_dict(),
            "switch_latency": switch.to_dict(),
            "per_stream_total_ms": [
                sum(times[i] for times in self._per_stream_times_ms)
                for i in range(self.num_streams)
            ]
            if self._per_stream_times_ms
            else [],
        }

    def export_to_json(self, filepath: str | Path) -> None:
        """
        Export collected metrics to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "num_streams": self.num_streams,
            "device": self.device,
            "summary": self.get_summary(),
            "iteration_times_ms": self._iteration_times_ms,
            "per_stream_times_ms": self._per_stream_times_ms,
            "kernel_timings": [
                [
                    {
                        "stream_id": kt.stream_id,
                        "kernel_name": kt.kernel_name,
                        "start_ms": kt.start_ms,
                        "end_ms": kt.end_ms,
                        "duration_ms": kt.duration_ms,
                    }
                    for kt in iteration_kernels
                ]
                for iteration_kernels in self._kernel_timings
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


@dataclass
class ScalingAnalysis:
    """Analysis of throughput scaling across stream counts."""

    stream_counts: List[int]
    throughputs: List[float]
    efficiencies: List[float]  # Relative to ideal linear scaling
    inflection_point: Optional[int]  # Stream count where scaling breaks down
    peak_stream_count: int  # Stream count with best throughput

    @classmethod
    def from_sweep_results(
        cls, results: List[Tuple[int, float]]
    ) -> "ScalingAnalysis":
        """
        Analyze scaling from sweep results.

        Args:
            results: List of (stream_count, throughput) tuples

        Returns:
            ScalingAnalysis with computed metrics
        """
        if not results:
            return cls(
                stream_counts=[],
                throughputs=[],
                efficiencies=[],
                inflection_point=None,
                peak_stream_count=0,
            )

        # Sort by stream count
        sorted_results = sorted(results, key=lambda x: x[0])
        stream_counts = [r[0] for r in sorted_results]
        throughputs = [r[1] for r in sorted_results]

        # Compute efficiency (relative to linear scaling from single-stream)
        base_throughput = throughputs[0] / stream_counts[0] if throughputs else 0
        efficiencies = []
        for sc, tp in zip(stream_counts, throughputs):
            ideal = base_throughput * sc
            efficiency = tp / ideal if ideal > 0 else 0
            efficiencies.append(efficiency)

        # Find peak throughput
        peak_idx = throughputs.index(max(throughputs)) if throughputs else 0
        peak_stream_count = stream_counts[peak_idx] if stream_counts else 0

        # Find inflection point (where efficiency drops significantly)
        inflection_point = None
        for i in range(1, len(efficiencies)):
            if efficiencies[i] < 0.8 * efficiencies[i - 1]:  # 20% drop
                inflection_point = stream_counts[i]
                break

        return cls(
            stream_counts=stream_counts,
            throughputs=throughputs,
            efficiencies=efficiencies,
            inflection_point=inflection_point,
            peak_stream_count=peak_stream_count,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def compare_results(
    baseline: Dict[str, Any],
    test: Dict[str, Any],
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Compare baseline and test results for regressions.

    Args:
        baseline: Baseline results dictionary
        test: Test results dictionary
        threshold: Regression threshold (fraction, e.g., 0.05 = 5%)

    Returns:
        Dictionary with comparison results including any regressions
    """
    comparison = {
        "baseline": baseline,
        "test": test,
        "regressions": [],
        "improvements": [],
        "unchanged": [],
    }

    # Compare throughputs if available
    if "throughput" in baseline and "throughput" in test:
        baseline_tp = baseline["throughput"]
        test_tp = test["throughput"]

        if isinstance(baseline_tp, dict):
            baseline_tp = baseline_tp.get("value", 0)
        if isinstance(test_tp, dict):
            test_tp = test_tp.get("value", 0)

        if baseline_tp > 0:
            change = (test_tp - baseline_tp) / baseline_tp
            metric_info = {
                "metric": "throughput",
                "baseline": baseline_tp,
                "test": test_tp,
                "change_pct": change * 100,
            }

            if change < -threshold:
                comparison["regressions"].append(metric_info)
            elif change > threshold:
                comparison["improvements"].append(metric_info)
            else:
                comparison["unchanged"].append(metric_info)

    # Compare latencies
    for latency_metric in ["p50_ms", "p95_ms", "p99_ms"]:
        baseline_lat = baseline.get("latency", {}).get(latency_metric, 0)
        test_lat = test.get("latency", {}).get(latency_metric, 0)

        if baseline_lat > 0:
            # For latency, increase is regression
            change = (test_lat - baseline_lat) / baseline_lat
            metric_info = {
                "metric": f"latency_{latency_metric}",
                "baseline": baseline_lat,
                "test": test_lat,
                "change_pct": change * 100,
            }

            if change > threshold:  # Latency increase is bad
                comparison["regressions"].append(metric_info)
            elif change < -threshold:  # Latency decrease is good
                comparison["improvements"].append(metric_info)
            else:
                comparison["unchanged"].append(metric_info)

    comparison["has_regressions"] = len(comparison["regressions"]) > 0
    comparison["summary"] = (
        f"{len(comparison['regressions'])} regressions, "
        f"{len(comparison['improvements'])} improvements, "
        f"{len(comparison['unchanged'])} unchanged"
    )

    return comparison
