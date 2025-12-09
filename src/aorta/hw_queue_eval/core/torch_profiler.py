"""
PyTorch Profiler integration for detailed kernel and memory analysis.

This module provides:
- TorchProfilerWrapper: Easy-to-use profiler with trace export
- Profile context managers for workload profiling
- Trace export in Chrome/TensorBoard formats
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
from torch.profiler import ProfilerActivity, profile


@dataclass
class ProfilerConfig:
    """Configuration for PyTorch profiler."""

    output_dir: Path
    activities: List[str] = field(default_factory=lambda: ["cpu", "cuda"])
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False  # Can be slow, enable for debugging
    with_flops: bool = True
    with_modules: bool = True

    # Schedule settings (for multi-iteration profiling)
    wait_iterations: int = 1
    warmup_iterations: int = 1
    active_iterations: int = 3
    repeat: int = 1

    # Export formats
    export_chrome_trace: bool = True
    export_tensorboard: bool = True
    export_stacks: bool = False
    export_memory_timeline: bool = True

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_activities(self) -> List[ProfilerActivity]:
        """Convert activity strings to ProfilerActivity enums."""
        activity_map = {
            "cpu": ProfilerActivity.CPU,
            "cuda": ProfilerActivity.CUDA,
        }
        return [activity_map[a.lower()] for a in self.activities if a.lower() in activity_map]


@dataclass
class ProfilerResult:
    """Results from profiling a workload."""

    chrome_trace_path: Optional[Path] = None
    tensorboard_dir: Optional[Path] = None
    memory_timeline_path: Optional[Path] = None
    stacks_path: Optional[Path] = None

    # Summary statistics
    total_cuda_time_ms: float = 0.0
    total_cpu_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    num_cuda_kernels: int = 0

    # Top operations
    top_cuda_ops: List[Dict[str, Any]] = field(default_factory=list)
    top_cpu_ops: List[Dict[str, Any]] = field(default_factory=list)

    # Memory events
    memory_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chrome_trace_path": str(self.chrome_trace_path) if self.chrome_trace_path else None,
            "tensorboard_dir": str(self.tensorboard_dir) if self.tensorboard_dir else None,
            "memory_timeline_path": str(self.memory_timeline_path) if self.memory_timeline_path else None,
            "total_cuda_time_ms": self.total_cuda_time_ms,
            "total_cpu_time_ms": self.total_cpu_time_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "num_cuda_kernels": self.num_cuda_kernels,
            "top_cuda_ops": self.top_cuda_ops,
            "top_cpu_ops": self.top_cpu_ops,
        }


class TorchProfilerWrapper:
    """
    Wrapper around PyTorch profiler for easy workload profiling.

    Usage:
        profiler = TorchProfilerWrapper(output_dir="profiles/")

        # Profile a workload
        result = profiler.profile_workload(workload, streams, iterations=10)

        # Or use context manager
        with profiler.profile_context("my_run") as prof:
            # ... run workload ...

        print(f"Chrome trace: {result.chrome_trace_path}")
        print(f"TensorBoard: {result.tensorboard_dir}")
    """

    def __init__(
        self,
        output_dir: str | Path = "profiles",
        config: Optional[ProfilerConfig] = None,
    ):
        """
        Initialize the profiler wrapper.

        Args:
            output_dir: Directory for output files
            config: Optional ProfilerConfig (will be created if not provided)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if config is None:
            config = ProfilerConfig(output_dir=self.output_dir)
        self.config = config

    def profile_workload(
        self,
        workload_fn: Callable,
        name: str = "workload",
        iterations: int = 5,
        warmup: int = 2,
    ) -> ProfilerResult:
        """
        Profile a workload function.

        Args:
            workload_fn: Function to profile (called repeatedly)
            name: Name for output files
            iterations: Number of iterations to profile
            warmup: Warmup iterations before profiling

        Returns:
            ProfilerResult with paths to generated traces
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{name}_{timestamp}"

        result = ProfilerResult()

        # Setup TensorBoard directory if enabled
        if self.config.export_tensorboard:
            tb_dir = self.output_dir / "tensorboard" / run_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            result.tensorboard_dir = tb_dir

        # Run warmup iterations without profiling
        for _ in range(warmup):
            workload_fn()

        # Run profiler without on_trace_ready to allow manual export
        # (on_trace_ready saves traces automatically which prevents manual export)
        with profile(
            activities=self.config.get_activities(),
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            with_modules=self.config.with_modules,
        ) as prof:
            for _ in range(iterations):
                workload_fn()

        # Export Chrome trace (can only be called once per profiler session)
        # ROCm/HIP may auto-save traces, so handle gracefully
        chrome_path = None
        if self.config.export_chrome_trace:
            chrome_path = self.output_dir / f"{run_name}_chrome_trace.json"
            try:
                prof.export_chrome_trace(str(chrome_path))
                result.chrome_trace_path = chrome_path
            except RuntimeError as e:
                if "already saved" in str(e).lower():
                    # Trace was auto-saved by ROCm/HIP profiler
                    # Try to find auto-saved trace files
                    import glob
                    auto_traces = glob.glob(str(self.output_dir / "*.json"))
                    if auto_traces:
                        # Use the most recent one
                        auto_traces.sort(key=os.path.getmtime, reverse=True)
                        result.chrome_trace_path = Path(auto_traces[0])
                    # Still extract statistics from profiler
                else:
                    raise

        # Copy chrome trace to TensorBoard directory if needed
        if self.config.export_tensorboard and result.tensorboard_dir and result.chrome_trace_path:
            import shutil
            tb_trace = result.tensorboard_dir / f"{run_name}.pt.trace.json"
            try:
                shutil.copy(result.chrome_trace_path, tb_trace)
            except Exception:
                pass  # Non-critical if copy fails

        # Export stacks if enabled
        if self.config.export_stacks and self.config.with_stack:
            try:
                stacks_path = self.output_dir / f"{run_name}_stacks.txt"
                prof.export_stacks(str(stacks_path), "self_cuda_time_total")
                result.stacks_path = stacks_path
            except RuntimeError:
                pass  # Stack export may fail if trace already saved

        # Export memory timeline if enabled
        if self.config.export_memory_timeline and self.config.profile_memory:
            try:
                memory_path = self.output_dir / f"{run_name}_memory_timeline.html"
                prof.export_memory_timeline(str(memory_path))
                result.memory_timeline_path = memory_path
            except Exception:
                # Memory timeline export may not be available in all versions
                pass

        # Extract summary statistics
        result = self._extract_statistics(prof, result)

        return result

    def _extract_statistics(
        self, prof: profile, result: ProfilerResult
    ) -> ProfilerResult:
        """Extract summary statistics from profiler."""

        # Get key averages
        key_averages = prof.key_averages()

        # Calculate totals
        total_cuda_time = 0
        total_cpu_time = 0
        num_cuda_kernels = 0

        cuda_ops = []
        cpu_ops = []

        for event in key_averages:
            if event.device_type == torch.device("cuda").type:
                total_cuda_time += event.cuda_time_total
                num_cuda_kernels += event.count
                cuda_ops.append({
                    "name": event.key,
                    "cuda_time_ms": event.cuda_time_total / 1000,
                    "cpu_time_ms": event.cpu_time_total / 1000,
                    "count": event.count,
                    "flops": getattr(event, "flops", 0),
                })
            else:
                total_cpu_time += event.cpu_time_total
                cpu_ops.append({
                    "name": event.key,
                    "cpu_time_ms": event.cpu_time_total / 1000,
                    "count": event.count,
                })

        # Sort by time
        cuda_ops.sort(key=lambda x: x["cuda_time_ms"], reverse=True)
        cpu_ops.sort(key=lambda x: x["cpu_time_ms"], reverse=True)

        result.total_cuda_time_ms = total_cuda_time / 1000
        result.total_cpu_time_ms = total_cpu_time / 1000
        result.num_cuda_kernels = num_cuda_kernels
        result.top_cuda_ops = cuda_ops[:10]
        result.top_cpu_ops = cpu_ops[:10]

        # Get peak memory
        if torch.cuda.is_available():
            result.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return result

    @contextmanager
    def profile_context(
        self, name: str = "profile"
    ) -> Generator[profile, None, ProfilerResult]:
        """
        Context manager for profiling.

        Args:
            name: Name for output files

        Yields:
            PyTorch profiler object

        Returns:
            ProfilerResult after context exits
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{name}_{timestamp}"

        with profile(
            activities=self.config.get_activities(),
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            with_modules=self.config.with_modules,
        ) as prof:
            yield prof

        # Export traces
        result = ProfilerResult()

        if self.config.export_chrome_trace:
            chrome_path = self.output_dir / f"{run_name}_chrome_trace.json"
            prof.export_chrome_trace(str(chrome_path))
            result.chrome_trace_path = chrome_path

        if self.config.export_tensorboard:
            tb_dir = self.output_dir / "tensorboard" / run_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            # Note: For single-shot profiling, we export chrome trace to TB dir
            chrome_tb = tb_dir / "trace.json"
            prof.export_chrome_trace(str(chrome_tb))
            result.tensorboard_dir = tb_dir

        return self._extract_statistics(prof, result)


def profile_workload_run(
    workload,
    streams: List[torch.cuda.Stream],
    output_dir: str | Path,
    iterations: int = 10,
    warmup: int = 3,
    name: Optional[str] = None,
) -> ProfilerResult:
    """
    Convenience function to profile a workload.

    Args:
        workload: BaseWorkload instance (must be setup)
        streams: List of CUDA streams
        output_dir: Output directory for traces
        iterations: Profiling iterations
        warmup: Warmup iterations
        name: Optional name for output files

    Returns:
        ProfilerResult with paths to traces
    """
    if name is None:
        name = workload.name

    profiler = TorchProfilerWrapper(output_dir=output_dir)

    def run_iteration():
        workload.run_iteration(streams)
        torch.cuda.synchronize()

    return profiler.profile_workload(
        run_iteration,
        name=name,
        iterations=iterations,
        warmup=warmup,
    )


def generate_profile_summary(result: ProfilerResult) -> str:
    """
    Generate a human-readable summary of profiling results.

    Args:
        result: ProfilerResult from profiling

    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("PROFILING SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    lines.append("TIMING:")
    lines.append(f"  Total CUDA time: {result.total_cuda_time_ms:.2f} ms")
    lines.append(f"  Total CPU time:  {result.total_cpu_time_ms:.2f} ms")
    lines.append(f"  CUDA kernels:    {result.num_cuda_kernels}")
    lines.append("")

    lines.append("MEMORY:")
    lines.append(f"  Peak GPU memory: {result.peak_memory_mb:.2f} MB")
    lines.append("")

    if result.top_cuda_ops:
        lines.append("TOP CUDA OPERATIONS (by time):")
        for i, op in enumerate(result.top_cuda_ops[:5], 1):
            flops_str = ""
            if op.get("flops", 0) > 0:
                gflops = op["flops"] / 1e9
                flops_str = f" ({gflops:.2f} GFLOPS)"
            lines.append(
                f"  {i}. {op['name'][:40]:<40} "
                f"{op['cuda_time_ms']:>8.2f} ms "
                f"(x{op['count']}){flops_str}"
            )
        lines.append("")

    lines.append("OUTPUT FILES:")
    if result.chrome_trace_path:
        lines.append(f"  Chrome trace:     {result.chrome_trace_path}")
        lines.append(f"    View: chrome://tracing and load the JSON file")
    if result.tensorboard_dir:
        lines.append(f"  TensorBoard:      {result.tensorboard_dir}")
        lines.append(f"    View: tensorboard --logdir={result.tensorboard_dir}")
    if result.memory_timeline_path:
        lines.append(f"  Memory timeline:  {result.memory_timeline_path}")
        lines.append(f"    View: Open in browser")
    if result.stacks_path:
        lines.append(f"  Stack traces:     {result.stacks_path}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def export_kernel_summary(result: ProfilerResult, output_path: Path) -> None:
    """
    Export kernel summary to JSON file.

    Args:
        result: ProfilerResult from profiling
        output_path: Path for output JSON
    """
    data = {
        "summary": {
            "total_cuda_time_ms": result.total_cuda_time_ms,
            "total_cpu_time_ms": result.total_cpu_time_ms,
            "peak_memory_mb": result.peak_memory_mb,
            "num_cuda_kernels": result.num_cuda_kernels,
        },
        "top_cuda_ops": result.top_cuda_ops,
        "top_cpu_ops": result.top_cpu_ops,
        "output_files": {
            "chrome_trace": str(result.chrome_trace_path) if result.chrome_trace_path else None,
            "tensorboard": str(result.tensorboard_dir) if result.tensorboard_dir else None,
            "memory_timeline": str(result.memory_timeline_path) if result.memory_timeline_path else None,
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
