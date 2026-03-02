"""
Parameterized test harness for running workloads with configurable stream counts.

This module provides:
- HarnessConfig: Configuration for test runs
- HarnessResult: Results from a test run
- StreamHarness: Main harness for executing workloads
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch

from aorta.hw_queue_eval.core.metrics import (
    LatencyMetrics,
    MemoryMetrics,
    MetricsCollector,
    ScalingAnalysis,
    SwitchLatencyMetrics,
    ThroughputMetrics,
)
from aorta.utils import (
    GPUControlConfig,
    GPUControlManager,
    create_multi_gpu_streams,
    create_streams,
    get_available_devices,
    get_device_properties,
    get_driver_info,
    get_rocm_env_info,
    get_system_info,
    reset_memory_stats,
    sync_all_streams,
    warmup_all_gpus,
    warmup_gpu,
)

if TYPE_CHECKING:
    from aorta.hw_queue_eval.workloads.base import BaseWorkload


@dataclass
class HarnessConfig:
    """Configuration for the stream harness."""

    stream_count: int
    warmup_iterations: int = 10
    measurement_iterations: int = 100
    sync_mode: str = "per_iteration"  # "per_iteration", "end_only", "none"
    device: str = "cuda:0"
    collect_kernel_timings: bool = True
    warmup_gpu_before_run: bool = True
    reset_memory_stats_before_run: bool = True
    use_multi_gpu: bool = True  # If True, distribute streams across all available GPUs
    devices: Optional[List[str]] = None  # Explicit list of devices (auto-detected if None)
    gpu_control: Optional[GPUControlConfig] = None  # GPU power/frequency control via Magpie

    def __post_init__(self):
        if self.stream_count < 1:
            raise ValueError("stream_count must be at least 1")
        if self.sync_mode not in ("per_iteration", "end_only", "none"):
            raise ValueError(f"Invalid sync_mode: {self.sync_mode}")
        # Auto-detect devices if multi-GPU is enabled but no explicit list provided
        if self.use_multi_gpu and self.devices is None:
            self.devices = get_available_devices() or [self.device]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HarnessResult:
    """Results from a harness run."""

    # Core metrics
    throughput: float  # ops/sec or samples/sec (depends on workload)
    throughput_unit: str
    latency_ms: Dict[str, float]  # {"mean": x, "p50": x, "p95": x, "p99": x}
    total_time_ms: float
    stream_count: int

    # Detailed metrics
    per_stream_times_ms: List[float]
    iteration_times_ms: List[float]
    switch_latency: Optional[Dict[str, float]] = None
    memory: Optional[Dict[str, float]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    workload_name: str = ""
    timestamp: str = ""

    @classmethod
    def from_metrics(
        cls,
        throughput_metrics: ThroughputMetrics,
        latency_metrics: LatencyMetrics,
        collector: MetricsCollector,
        stream_count: int,
        workload_name: str = "",
        memory_metrics: Optional[MemoryMetrics] = None,
        switch_metrics: Optional[SwitchLatencyMetrics] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> "HarnessResult":
        """Create HarnessResult from metric objects."""
        return cls(
            throughput=throughput_metrics.value,
            throughput_unit=throughput_metrics.unit,
            latency_ms={
                "mean": latency_metrics.mean_ms,
                "p50": latency_metrics.p50_ms,
                "p95": latency_metrics.p95_ms,
                "p99": latency_metrics.p99_ms,
                "min": latency_metrics.min_ms,
                "max": latency_metrics.max_ms,
                "std": latency_metrics.std_ms,
            },
            total_time_ms=collector.get_total_time_ms(),
            stream_count=stream_count,
            per_stream_times_ms=[
                sum(times[i] for times in collector.get_per_stream_times())
                for i in range(stream_count)
            ]
            if collector.get_per_stream_times()
            else [],
            iteration_times_ms=collector.get_iteration_times(),
            switch_latency=switch_metrics.to_dict() if switch_metrics else None,
            memory=memory_metrics.to_dict() if memory_metrics else None,
            metadata=extra_metadata or {},
            workload_name=workload_name,
            timestamp=datetime.now().isoformat(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, filepath: Optional[str | Path] = None) -> str:
        """
        Convert to JSON string, optionally saving to file.

        Args:
            filepath: Optional path to save JSON file

        Returns:
            JSON string
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
        return json_str


class StreamHarness:
    """
    Base harness for running workloads with parameterized stream counts.

    This harness handles:
    - Stream creation and management
    - Warmup and measurement phases
    - Timing data collection
    - Result aggregation

    Usage:
        config = HarnessConfig(stream_count=8)
        harness = StreamHarness(config)

        # Run with a workload object
        result = harness.run_workload(workload)

        # Or run with a callable
        result = harness.run(my_function, arg1, arg2)

        # Sweep across stream counts
        results = harness.sweep(my_function, [1, 2, 4, 8, 16])
    """

    def __init__(self, config: HarnessConfig):
        """
        Initialize the harness.

        Args:
            config: HarnessConfig with run parameters
        """
        self.config = config
        self.streams: List[torch.cuda.Stream] = []
        self.stream_to_device: Dict[int, str] = {}  # Maps stream index to device
        self.devices: List[str] = []  # List of devices in use
        self._initialized = False
        self._metrics_collector: Optional[MetricsCollector] = None
        self._gpu_control = GPUControlManager(
            config.gpu_control or GPUControlConfig()
        )

    def _initialize(self) -> None:
        """Initialize streams and prepare for run."""
        if self._initialized:
            return

        # Create streams (single-GPU or multi-GPU based on config)
        if self.config.use_multi_gpu and self.config.devices:
            self.devices = self.config.devices
            self.streams, self.stream_to_device = create_multi_gpu_streams(
                total_stream_count=self.config.stream_count,
                devices=self.config.devices,
            )
        else:
            self.devices = [self.config.device]
            self.streams = create_streams(self.config.stream_count, self.config.device)
            self.stream_to_device = {i: self.config.device for i in range(self.config.stream_count)}

        # Initialize metrics collector (uses primary device)
        self._metrics_collector = MetricsCollector(
            num_streams=self.config.stream_count,
            device=self.config.device,
        )

        # GPU warmup if requested
        if self.config.warmup_gpu_before_run:
            if self.config.use_multi_gpu and len(self.devices) > 1:
                warmup_all_gpus(devices=self.devices)
            else:
                warmup_gpu(self.config.device)

        # Reset memory stats if requested (on all devices)
        if self.config.reset_memory_stats_before_run:
            for device in self.devices:
                reset_memory_stats(device)

        self._initialized = True

    def _cleanup(self) -> None:
        """Cleanup after run."""
        # Sync all streams
        sync_all_streams(self.streams)

        # Synchronize all devices
        for device in self.devices:
            torch.cuda.synchronize(device)

        # Clear stream references
        self.streams = []
        self.stream_to_device = {}
        self._initialized = False

    def run(
        self,
        workload_fn: Callable[[List[torch.cuda.Stream]], None],
        throughput_fn: Optional[Callable[[int, float], float]] = None,
        throughput_unit: str = "iterations/sec",
        workload_name: str = "custom",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> HarnessResult:
        """
        Run a workload function with the configured stream count.

        Args:
            workload_fn: Function that takes a list of streams and executes work
            throughput_fn: Optional function to compute throughput (iterations, time_sec) -> value
            throughput_unit: Unit string for throughput
            workload_name: Name for the workload
            extra_metadata: Additional metadata to include in results

        Returns:
            HarnessResult with timing and metrics
        """
        self._initialize()

        # Apply GPU hardware control (lock clocks, set power) before benchmark
        gpu_hw_snapshot = self._gpu_control.apply()

        collector = self._metrics_collector
        collector.clear()

        # Warmup phase
        for _ in range(self.config.warmup_iterations):
            workload_fn(self.streams)
            if self.config.sync_mode == "per_iteration":
                sync_all_streams(self.streams)

        # Sync after warmup (all devices)
        sync_all_streams(self.streams)
        for device in self.devices:
            torch.cuda.synchronize(device)

        # Reset memory stats after warmup (all devices)
        if self.config.reset_memory_stats_before_run:
            for device in self.devices:
                reset_memory_stats(device)

        # Measurement phase
        for _ in range(self.config.measurement_iterations):
            collector.start_iteration()

            workload_fn(self.streams)

            if self.config.sync_mode == "per_iteration":
                sync_all_streams(self.streams)

            collector.end_iteration(sync=(self.config.sync_mode == "per_iteration"))

        # Final sync if using end_only mode (all devices)
        if self.config.sync_mode in ("end_only", "none"):
            sync_all_streams(self.streams)
            for device in self.devices:
                torch.cuda.synchronize(device)

        # Compute metrics (capture from primary device, but note multi-GPU in metadata)
        latency_metrics = collector.compute_latency_metrics()
        switch_metrics = collector.compute_switch_latency()
        memory_metrics = MemoryMetrics.capture(self.config.device)

        # Compute throughput
        total_time_sec = collector.get_total_time_ms() / 1000.0
        if throughput_fn:
            throughput_value = throughput_fn(
                self.config.measurement_iterations, total_time_sec
            )
        else:
            throughput_value = self.config.measurement_iterations / total_time_sec

        throughput_metrics = ThroughputMetrics(
            value=throughput_value,
            unit=throughput_unit,
            raw_count=self.config.measurement_iterations,
            duration_sec=total_time_sec,
        )

        # Build result with comprehensive system info
        system_info = get_system_info()
        metadata = {
            "config": self.config.to_dict(),
            "device_info": get_device_properties(self.config.device).__dict__,
            "rocm_info": get_rocm_env_info(),
            "driver_info": get_driver_info(),
            "system_info": {
                "hostname": system_info.get("hostname"),
                "kernel": system_info.get("driver", {}).get("kernel"),
                "driver_type": system_info.get("driver", {}).get("driver_type"),
            },
            "multi_gpu": {
                "enabled": self.config.use_multi_gpu,
                "devices": self.devices,
                "num_gpus": len(self.devices),
                "stream_to_device": self.stream_to_device,
            },
        }
        if gpu_hw_snapshot:
            metadata.update(gpu_hw_snapshot)
        if extra_metadata:
            metadata.update(extra_metadata)

        result = HarnessResult.from_metrics(
            throughput_metrics=throughput_metrics,
            latency_metrics=latency_metrics,
            collector=collector,
            stream_count=self.config.stream_count,
            workload_name=workload_name,
            memory_metrics=memory_metrics,
            switch_metrics=switch_metrics,
            extra_metadata=metadata,
        )

        # Reset GPU settings after benchmark
        self._gpu_control.reset()
        self._cleanup()
        return result

    def run_workload(
        self,
        workload: "BaseWorkload",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> HarnessResult:
        """
        Run a BaseWorkload object with the configured stream count.

        Args:
            workload: Workload object implementing BaseWorkload
            extra_metadata: Additional metadata to include in results

        Returns:
            HarnessResult with timing and metrics
        """
        self._initialize()

        # Apply GPU hardware control (lock clocks, set power) before benchmark
        gpu_hw_snapshot = self._gpu_control.apply()

        # Setup workload
        workload.setup(self.config.stream_count, self.config.device)

        collector = self._metrics_collector
        collector.clear()

        # Warmup phase
        for _ in range(self.config.warmup_iterations):
            workload.run_iteration(self.streams)
            if self.config.sync_mode == "per_iteration":
                sync_all_streams(self.streams)

        # Sync after warmup (all devices)
        sync_all_streams(self.streams)
        for device in self.devices:
            torch.cuda.synchronize(device)

        # Reset memory stats after warmup (all devices)
        if self.config.reset_memory_stats_before_run:
            for device in self.devices:
                reset_memory_stats(device)

        # Measurement phase
        for _ in range(self.config.measurement_iterations):
            collector.start_iteration()

            workload.run_iteration(self.streams)

            if self.config.sync_mode == "per_iteration":
                sync_all_streams(self.streams)

            collector.end_iteration(sync=(self.config.sync_mode == "per_iteration"))

        # Final sync (all devices)
        if self.config.sync_mode in ("end_only", "none"):
            sync_all_streams(self.streams)
            for device in self.devices:
                torch.cuda.synchronize(device)

        # Compute metrics
        latency_metrics = collector.compute_latency_metrics()
        switch_metrics = collector.compute_switch_latency()
        memory_metrics = MemoryMetrics.capture(self.config.device)

        # Compute throughput using workload's method
        total_time_sec = collector.get_total_time_ms() / 1000.0
        throughput_value = workload.compute_throughput(
            self.config.measurement_iterations, total_time_sec
        )

        throughput_metrics = ThroughputMetrics(
            value=throughput_value,
            unit=workload.get_throughput_unit(),
            raw_count=self.config.measurement_iterations,
            duration_sec=total_time_sec,
        )

        # Build result with comprehensive system info
        system_info = get_system_info()
        metadata = {
            "config": self.config.to_dict(),
            "device_info": get_device_properties(self.config.device).__dict__,
            "rocm_info": get_rocm_env_info(),
            "driver_info": get_driver_info(),
            "system_info": {
                "hostname": system_info.get("hostname"),
                "kernel": system_info.get("driver", {}).get("kernel"),
                "driver_type": system_info.get("driver", {}).get("driver_type"),
            },
            "workload_config": workload.get_config() if hasattr(workload, "get_config") else {},
            "multi_gpu": {
                "enabled": self.config.use_multi_gpu,
                "devices": self.devices,
                "num_gpus": len(self.devices),
                "stream_to_device": self.stream_to_device,
            },
        }
        if gpu_hw_snapshot:
            metadata.update(gpu_hw_snapshot)
        if extra_metadata:
            metadata.update(extra_metadata)

        result = HarnessResult.from_metrics(
            throughput_metrics=throughput_metrics,
            latency_metrics=latency_metrics,
            collector=collector,
            stream_count=self.config.stream_count,
            workload_name=workload.name,
            memory_metrics=memory_metrics,
            switch_metrics=switch_metrics,
            extra_metadata=metadata,
        )

        # Cleanup workload, reset GPU settings, then cleanup harness
        workload.cleanup()
        self._gpu_control.reset()
        self._cleanup()

        return result

    def sweep(
        self,
        workload_fn: Callable[[List[torch.cuda.Stream]], None],
        stream_counts: List[int],
        throughput_fn: Optional[Callable[[int, float], float]] = None,
        throughput_unit: str = "iterations/sec",
        workload_name: str = "custom",
    ) -> List[HarnessResult]:
        """
        Run workload across multiple stream counts.

        Args:
            workload_fn: Function that takes a list of streams and executes work
            stream_counts: List of stream counts to test
            throughput_fn: Optional function to compute throughput
            throughput_unit: Unit string for throughput
            workload_name: Name for the workload

        Returns:
            List of HarnessResult objects, one per stream count
        """
        results = []

        for count in stream_counts:
            # Create new config with this stream count
            config = HarnessConfig(
                stream_count=count,
                warmup_iterations=self.config.warmup_iterations,
                measurement_iterations=self.config.measurement_iterations,
                sync_mode=self.config.sync_mode,
                device=self.config.device,
                collect_kernel_timings=self.config.collect_kernel_timings,
                warmup_gpu_before_run=self.config.warmup_gpu_before_run,
                reset_memory_stats_before_run=self.config.reset_memory_stats_before_run,
                use_multi_gpu=self.config.use_multi_gpu,
                devices=self.config.devices,
            )

            harness = StreamHarness(config)
            result = harness.run(
                workload_fn=workload_fn,
                throughput_fn=throughput_fn,
                throughput_unit=throughput_unit,
                workload_name=workload_name,
            )
            results.append(result)

        return results

    def sweep_workload(
        self,
        workload: "BaseWorkload",
        stream_counts: Optional[List[int]] = None,
    ) -> List[HarnessResult]:
        """
        Run a BaseWorkload across multiple stream counts.

        Args:
            workload: Workload object implementing BaseWorkload
            stream_counts: List of stream counts to test.
                          If None, uses [1, 2, 4, 8, 16, 32] filtered by workload limits.

        Returns:
            List of HarnessResult objects, one per stream count
        """
        if stream_counts is None:
            stream_counts = [1, 2, 4, 8, 16, 32]

        # Filter by workload's stream limits
        stream_counts = [
            c for c in stream_counts
            if workload.min_streams <= c <= workload.max_streams
        ]

        results = []

        for count in stream_counts:
            # Create new config with this stream count
            config = HarnessConfig(
                stream_count=count,
                warmup_iterations=self.config.warmup_iterations,
                measurement_iterations=self.config.measurement_iterations,
                sync_mode=self.config.sync_mode,
                device=self.config.device,
                collect_kernel_timings=self.config.collect_kernel_timings,
                warmup_gpu_before_run=self.config.warmup_gpu_before_run,
                reset_memory_stats_before_run=self.config.reset_memory_stats_before_run,
                use_multi_gpu=self.config.use_multi_gpu,
                devices=self.config.devices,
            )

            harness = StreamHarness(config)
            result = harness.run_workload(workload)
            results.append(result)

        return results


def analyze_sweep_results(results: List[HarnessResult]) -> ScalingAnalysis:
    """
    Analyze results from a stream count sweep.

    Args:
        results: List of HarnessResult from a sweep

    Returns:
        ScalingAnalysis with scaling characteristics
    """
    sweep_data = [(r.stream_count, r.throughput) for r in results]
    return ScalingAnalysis.from_sweep_results(sweep_data)


def format_results_table(results: List[HarnessResult]) -> str:
    """
    Format sweep results as a text table.

    Args:
        results: List of HarnessResult from a sweep

    Returns:
        Formatted table string
    """
    lines = []
    lines.append(
        f"{'Streams':>8} {'Throughput':>15} {'P50 (ms)':>12} {'P95 (ms)':>12} "
        f"{'P99 (ms)':>12} {'Total (ms)':>12}"
    )
    lines.append("-" * 75)

    for r in results:
        lines.append(
            f"{r.stream_count:>8} {r.throughput:>15.2f} {r.latency_ms['p50']:>12.3f} "
            f"{r.latency_ms['p95']:>12.3f} {r.latency_ms['p99']:>12.3f} "
            f"{r.total_time_ms:>12.2f}"
        )

    return "\n".join(lines)


def save_sweep_results(
    results: List[HarnessResult],
    filepath: str | Path,
    include_analysis: bool = True,
    include_system_info: bool = True,
) -> None:
    """
    Save sweep results to JSON file.

    Args:
        results: List of HarnessResult from a sweep
        filepath: Path to output file
        include_analysis: If True, include scaling analysis
        include_system_info: If True, include system/driver info at top level
    """
    data = {
        "timestamp": datetime.now().isoformat(),
        "workload": results[0].workload_name if results else "unknown",
        "results": [r.to_dict() for r in results],
    }

    if include_system_info:
        system_info = get_system_info()
        data["environment"] = {
            "hostname": system_info.get("hostname"),
            "kernel": system_info.get("driver", {}).get("kernel"),
            "dkms_version": system_info.get("driver", {}).get("dkms_version"),
            "driver_type": system_info.get("driver", {}).get("driver_type"),
            "hip_version": system_info.get("rocm", {}).get("hip_version"),
            "torch_version": system_info.get("rocm", {}).get("torch_version"),
            "gpu_count": system_info.get("gpu_count"),
            "gpus": system_info.get("gpus", []),
        }

    if include_analysis:
        analysis = analyze_sweep_results(results)
        data["analysis"] = analysis.to_dict()

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
