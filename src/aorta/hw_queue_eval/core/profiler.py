"""
ROCm profiler integration for hardware queue visibility.

This module provides:
- ROCmProfiler: Wrapper for rocprof/roctracer
- Trace parsing utilities for queue information
- Timeline generation for visualization

Environment variables used:
- AMD_LOG_LEVEL=4: For queue-level logging
- HSA_TOOLS_LIB: For profiler injection
- ROCPROF_COUNTERS_PATH: Custom counter definitions
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aorta.utils import IS_ROCM


@dataclass
class KernelTrace:
    """Information about a traced kernel execution."""

    kernel_name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    queue_id: Optional[int] = None
    stream_id: Optional[int] = None
    grid_size: Optional[Tuple[int, int, int]] = None
    block_size: Optional[Tuple[int, int, int]] = None
    device_id: int = 0

    @property
    def start_ms(self) -> float:
        return self.start_ns / 1_000_000

    @property
    def end_ms(self) -> float:
        return self.end_ns / 1_000_000

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000


@dataclass
class QueueInfo:
    """Information about hardware queue usage."""

    num_queues: int
    queue_ids: List[int]
    kernels_per_queue: Dict[int, int]
    queue_utilization: Dict[int, float]  # Fraction of time each queue was active

    @classmethod
    def from_traces(cls, traces: List[KernelTrace]) -> "QueueInfo":
        """Compute queue info from kernel traces."""
        if not traces:
            return cls(
                num_queues=0,
                queue_ids=[],
                kernels_per_queue={},
                queue_utilization={},
            )

        queue_ids = set()
        kernels_per_queue: Dict[int, int] = {}
        queue_active_time: Dict[int, int] = {}

        for trace in traces:
            qid = trace.queue_id or 0
            queue_ids.add(qid)
            kernels_per_queue[qid] = kernels_per_queue.get(qid, 0) + 1
            queue_active_time[qid] = queue_active_time.get(qid, 0) + trace.duration_ns

        # Compute utilization
        total_time = max(t.end_ns for t in traces) - min(t.start_ns for t in traces)
        queue_utilization = {}
        for qid in queue_ids:
            if total_time > 0:
                queue_utilization[qid] = queue_active_time.get(qid, 0) / total_time
            else:
                queue_utilization[qid] = 0.0

        return cls(
            num_queues=len(queue_ids),
            queue_ids=sorted(queue_ids),
            kernels_per_queue=kernels_per_queue,
            queue_utilization=queue_utilization,
        )


@dataclass
class ProfilerConfig:
    """Configuration for profiling sessions."""

    output_dir: Path
    metrics: List[str] = field(default_factory=list)
    trace_hip_api: bool = True
    trace_hsa_api: bool = False
    trace_kernels: bool = True
    flush_interval_ms: int = 100
    verbose: bool = False

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class ROCmProfiler:
    """
    Wrapper for ROCm profiling tools (rocprof, roctracer).

    Provides functionality to:
    - Profile commands and collect kernel traces
    - Parse profiler output for queue information
    - Generate timeline visualizations
    """

    # Common hardware counters for queue analysis
    DEFAULT_METRICS = [
        "GRBM_COUNT",
        "GRBM_GUI_ACTIVE",
        "SQ_WAVES",
        "SQ_INSTS_VALU",
    ]

    def __init__(self, output_dir: Path | str):
        """
        Initialize the profiler.

        Args:
            output_dir: Directory for profiler output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._check_rocm_tools()

    def _check_rocm_tools(self) -> Dict[str, bool]:
        """Check availability of ROCm profiling tools."""
        tools = {
            "rocprof": shutil.which("rocprof") is not None,
            "roctracer": shutil.which("roctracer") is not None,
            "rocprofv2": shutil.which("rocprofv2") is not None,
        }
        self._available_tools = tools
        return tools

    @property
    def rocprof_available(self) -> bool:
        """Check if rocprof is available."""
        return self._available_tools.get("rocprof", False)

    @property
    def roctracer_available(self) -> bool:
        """Check if roctracer is available."""
        return self._available_tools.get("roctracer", False)

    def profile_with_rocprof(
        self,
        command: List[str],
        metrics: Optional[List[str]] = None,
        output_name: Optional[str] = None,
        hip_trace: bool = True,
        hsa_trace: bool = False,
    ) -> Path:
        """
        Run command under rocprof and return path to results.

        Args:
            command: Command to profile as list of strings
            metrics: Optional list of hardware counters to collect
            output_name: Optional name for output file (without extension)
            hip_trace: Enable HIP API tracing
            hsa_trace: Enable HSA API tracing

        Returns:
            Path to rocprof output file

        Raises:
            RuntimeError: If rocprof is not available or profiling fails
        """
        if not self.rocprof_available:
            raise RuntimeError("rocprof is not available. Please install ROCm profiler.")

        # Generate output filename
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"rocprof_{timestamp}"

        output_file = self.output_dir / f"{output_name}.csv"

        # Build rocprof command
        rocprof_cmd = ["rocprof"]

        # Add tracing options
        if hip_trace:
            rocprof_cmd.append("--hip-trace")
        if hsa_trace:
            rocprof_cmd.append("--hsa-trace")

        # Add metrics if specified
        if metrics:
            metrics_file = self._create_metrics_file(metrics, output_name)
            rocprof_cmd.extend(["-i", str(metrics_file)])

        # Output file
        rocprof_cmd.extend(["-o", str(output_file)])

        # Add the command to profile
        rocprof_cmd.extend(command)

        # Set environment for better tracing
        env = os.environ.copy()
        env["AMD_LOG_LEVEL"] = "4"

        try:
            result = subprocess.run(
                rocprof_cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"rocprof failed with code {result.returncode}:\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

        except subprocess.TimeoutExpired:
            raise RuntimeError("Profiling timed out after 10 minutes")

        return output_file

    def _create_metrics_file(self, metrics: List[str], name: str) -> Path:
        """Create a metrics input file for rocprof."""
        metrics_file = self.output_dir / f"{name}_metrics.txt"

        with open(metrics_file, "w") as f:
            f.write("# Hardware counters for queue evaluation\n")
            f.write("pmc: " + " ".join(metrics) + "\n")

        return metrics_file

    def parse_rocprof_csv(self, csv_file: Path) -> List[KernelTrace]:
        """
        Parse rocprof CSV output to extract kernel traces.

        Args:
            csv_file: Path to rocprof CSV output

        Returns:
            List of KernelTrace objects
        """
        traces = []

        if not csv_file.exists():
            return traces

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Handle different rocprof output formats
                    kernel_name = row.get("KernelName", row.get("Name", "unknown"))

                    # Time fields (in nanoseconds)
                    start_ns = int(row.get("BeginNs", row.get("Start", 0)))
                    end_ns = int(row.get("EndNs", row.get("End", 0)))

                    # Queue/stream info
                    queue_id = int(row.get("Queue", row.get("queue-id", 0)))
                    stream_id = int(row.get("Stream", row.get("stream-id", 0)))

                    # Grid/block dimensions
                    grid_size = None
                    block_size = None

                    if "grd" in row:
                        grid_str = row["grd"]
                        if grid_str:
                            parts = grid_str.split(",")
                            if len(parts) >= 3:
                                grid_size = (int(parts[0]), int(parts[1]), int(parts[2]))

                    if "wgr" in row:
                        block_str = row["wgr"]
                        if block_str:
                            parts = block_str.split(",")
                            if len(parts) >= 3:
                                block_size = (int(parts[0]), int(parts[1]), int(parts[2]))

                    trace = KernelTrace(
                        kernel_name=kernel_name,
                        start_ns=start_ns,
                        end_ns=end_ns,
                        duration_ns=end_ns - start_ns,
                        queue_id=queue_id,
                        stream_id=stream_id,
                        grid_size=grid_size,
                        block_size=block_size,
                        device_id=int(row.get("gpu-id", 0)),
                    )
                    traces.append(trace)

                except (ValueError, KeyError) as e:
                    # Skip malformed rows
                    continue

        return traces

    def parse_hip_trace(self, trace_file: Path) -> List[Dict[str, Any]]:
        """
        Parse HIP API trace file.

        Args:
            trace_file: Path to HIP trace file

        Returns:
            List of HIP API call records
        """
        calls = []

        if not trace_file.exists():
            return calls

        with open(trace_file, "r") as f:
            for line in f:
                # Parse roctracer HIP output format
                # Format: <pid>:<tid> <timestamp> <duration> <api_name>:<args>
                match = re.match(
                    r"(\d+):(\d+)\s+(\d+)\s+(\d+)\s+(\w+)(?::(.*))?",
                    line.strip()
                )
                if match:
                    calls.append({
                        "pid": int(match.group(1)),
                        "tid": int(match.group(2)),
                        "timestamp_ns": int(match.group(3)),
                        "duration_ns": int(match.group(4)),
                        "api_name": match.group(5),
                        "args": match.group(6) if match.group(6) else "",
                    })

        return calls

    def parse_queue_info(self, trace_file: Path) -> QueueInfo:
        """
        Parse profiler output to extract hardware queue information.

        Args:
            trace_file: Path to trace file (CSV or other format)

        Returns:
            QueueInfo with queue usage statistics
        """
        # Determine file type and parse accordingly
        if trace_file.suffix == ".csv":
            traces = self.parse_rocprof_csv(trace_file)
        else:
            # Try to parse as JSON
            try:
                with open(trace_file, "r") as f:
                    data = json.load(f)
                # Convert JSON format to traces
                traces = self._json_to_traces(data)
            except (json.JSONDecodeError, KeyError):
                traces = []

        return QueueInfo.from_traces(traces)

    def _json_to_traces(self, data: Dict[str, Any]) -> List[KernelTrace]:
        """Convert JSON profiler output to kernel traces."""
        traces = []

        # Handle Chrome trace format (used by some profilers)
        if "traceEvents" in data:
            for event in data["traceEvents"]:
                if event.get("ph") == "X":  # Complete event
                    traces.append(KernelTrace(
                        kernel_name=event.get("name", "unknown"),
                        start_ns=int(event.get("ts", 0) * 1000),  # us to ns
                        end_ns=int((event.get("ts", 0) + event.get("dur", 0)) * 1000),
                        duration_ns=int(event.get("dur", 0) * 1000),
                        queue_id=event.get("args", {}).get("queue_id"),
                        stream_id=event.get("args", {}).get("stream_id"),
                    ))

        return traces

    def generate_timeline(
        self,
        trace_file: Path,
        output_file: Optional[Path] = None,
    ) -> Path:
        """
        Generate visual timeline of queue utilization.

        Args:
            trace_file: Path to trace file
            output_file: Optional path for output HTML file

        Returns:
            Path to generated timeline file
        """
        traces = self.parse_rocprof_csv(trace_file)

        if output_file is None:
            output_file = self.output_dir / f"{trace_file.stem}_timeline.html"

        # Generate Chrome trace format JSON
        chrome_events = []

        for trace in traces:
            event = {
                "name": trace.kernel_name,
                "cat": "kernel",
                "ph": "X",  # Complete event
                "ts": trace.start_ns / 1000,  # Convert to microseconds
                "dur": trace.duration_ns / 1000,
                "pid": trace.device_id,
                "tid": trace.queue_id or 0,
                "args": {
                    "queue_id": trace.queue_id,
                    "stream_id": trace.stream_id,
                },
            }
            chrome_events.append(event)

        # Create HTML viewer
        html_content = self._create_timeline_html(chrome_events)

        with open(output_file, "w") as f:
            f.write(html_content)

        return output_file

    def _create_timeline_html(self, events: List[Dict[str, Any]]) -> str:
        """Create HTML file with embedded trace viewer."""
        trace_json = json.dumps({"traceEvents": events})

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>GPU Queue Timeline</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .info {{ margin: 10px 0; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>GPU Queue Timeline</h1>
    <div class="info">
        <p>This trace file can be viewed in Chrome's tracing tool:</p>
        <ol>
            <li>Open Chrome and navigate to <code>chrome://tracing</code></li>
            <li>Click "Load" and select the JSON file, or paste the JSON below</li>
        </ol>
    </div>
    <h2>Trace Data (JSON)</h2>
    <pre>{trace_json}</pre>
    <script>
        // Save trace as downloadable JSON
        const blob = new Blob(['{trace_json}'], {{type: 'application/json'}});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'trace.json';
        document.body.appendChild(a);
    </script>
</body>
</html>
"""

    def get_queue_env_vars(self) -> Dict[str, str]:
        """
        Get environment variables to set for queue debugging.

        Returns:
            Dictionary of environment variable names and values
        """
        return {
            "AMD_LOG_LEVEL": "4",
            "HSA_TOOLS_LIB": "",  # Disable default tools
            "GPU_MAX_HW_QUEUES": "",  # Let system decide
            "HIP_VISIBLE_DEVICES": "0",  # Single GPU
            "AMD_SERIALIZE_KERNEL": "3",  # Serialize for debugging
        }


def profile_python_command(
    script_path: str,
    args: List[str] = None,
    output_dir: str = "profiles",
    metrics: Optional[List[str]] = None,
) -> Tuple[Path, QueueInfo]:
    """
    Convenience function to profile a Python script.

    Args:
        script_path: Path to Python script
        args: Optional arguments for the script
        output_dir: Directory for output files
        metrics: Optional hardware metrics to collect

    Returns:
        Tuple of (trace_file_path, queue_info)
    """
    profiler = ROCmProfiler(output_dir)

    command = ["python", script_path]
    if args:
        command.extend(args)

    try:
        trace_file = profiler.profile_with_rocprof(command, metrics=metrics)
        queue_info = profiler.parse_queue_info(trace_file)
        return trace_file, queue_info
    except RuntimeError as e:
        # Return empty results if profiling fails
        print(f"Warning: Profiling failed: {e}")
        return Path(output_dir) / "empty.csv", QueueInfo(
            num_queues=0,
            queue_ids=[],
            kernels_per_queue={},
            queue_utilization={},
        )


def create_profiling_script(
    workload_name: str,
    stream_count: int,
    output_dir: Path,
) -> Path:
    """
    Create a standalone Python script for profiling a workload.

    Args:
        workload_name: Name of the workload to profile
        stream_count: Number of streams to use
        output_dir: Directory for output

    Returns:
        Path to the created script
    """
    script_content = f'''#!/usr/bin/env python3
"""Auto-generated profiling script for {workload_name}"""

import torch
from aorta.hw_queue_eval.workloads import get_workload
from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

def main():
    # Setup
    workload = get_workload("{workload_name}")
    config = HarnessConfig(
        stream_count={stream_count},
        warmup_iterations=5,
        measurement_iterations=50,
    )

    # Run
    harness = StreamHarness(config)
    result = harness.run_workload(workload)

    # Print summary
    print(f"Throughput: {{result.throughput:.2f}} {{result.throughput_unit}}")
    print(f"P99 Latency: {{result.latency_ms['p99']:.3f}} ms")

if __name__ == "__main__":
    main()
'''

    script_path = output_dir / f"profile_{workload_name}_{stream_count}s.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    script_path.chmod(0o755)
    return script_path
