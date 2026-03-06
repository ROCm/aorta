"""
HW Queue Eval JSON loader and data classes.

This module provides:
- SingleRunData: Parsed single run result
- SweepData: Parsed sweep result with list of runs
- HWQueueLoader: Static methods to load and validate JSON files
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class LatencyData:
    """Latency metrics in milliseconds."""

    mean: float
    p50: float
    p95: float
    p99: float
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LatencyData":
        """Create from dictionary."""
        return cls(
            mean=data.get("mean", 0.0),
            p50=data.get("p50", 0.0),
            p95=data.get("p95", 0.0),
            p99=data.get("p99", 0.0),
            min=data.get("min", 0.0),
            max=data.get("max", 0.0),
            std=data.get("std", 0.0),
        )


@dataclass
class SwitchLatencyData:
    """Queue switch latency metrics."""

    inter_stream_gap_ms: float
    intra_stream_gap_ms: float
    estimated_switch_overhead_ms: float
    inter_stream_samples: int = 0
    intra_stream_samples: int = 0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["SwitchLatencyData"]:
        """Create from dictionary, returns None if data is None."""
        if data is None:
            return None
        return cls(
            inter_stream_gap_ms=data.get("inter_stream_gap_ms", 0.0),
            intra_stream_gap_ms=data.get("intra_stream_gap_ms", 0.0),
            estimated_switch_overhead_ms=data.get("estimated_switch_overhead_ms", 0.0),
            inter_stream_samples=data.get("inter_stream_samples", 0),
            intra_stream_samples=data.get("intra_stream_samples", 0),
        )


@dataclass
class MemoryData:
    """GPU memory metrics."""

    peak_allocated_gb: float
    peak_reserved_gb: float
    final_allocated_gb: float = 0.0
    final_reserved_gb: float = 0.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["MemoryData"]:
        """Create from dictionary, returns None if data is None."""
        if data is None:
            return None
        return cls(
            peak_allocated_gb=data.get("peak_allocated_gb", 0.0),
            peak_reserved_gb=data.get("peak_reserved_gb", 0.0),
            final_allocated_gb=data.get("final_allocated_gb", 0.0),
            final_reserved_gb=data.get("final_reserved_gb", 0.0),
        )


@dataclass
class SingleRunData:
    """Parsed single run result."""

    workload_name: str
    stream_count: int
    throughput: float
    throughput_unit: str
    latency: LatencyData
    total_time_ms: float
    per_stream_times_ms: List[float] = field(default_factory=list)
    iteration_times_ms: List[float] = field(default_factory=list)
    switch_latency: Optional[SwitchLatencyData] = None
    memory: Optional[MemoryData] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SingleRunData":
        """Create from dictionary."""
        return cls(
            workload_name=data.get("workload_name", "unknown"),
            stream_count=data.get("stream_count", 0),
            throughput=data.get("throughput", 0.0),
            throughput_unit=data.get("throughput_unit", "ops/sec"),
            latency=LatencyData.from_dict(data.get("latency_ms", {})),
            total_time_ms=data.get("total_time_ms", 0.0),
            per_stream_times_ms=data.get("per_stream_times_ms", []),
            iteration_times_ms=data.get("iteration_times_ms", []),
            switch_latency=SwitchLatencyData.from_dict(data.get("switch_latency")),
            memory=MemoryData.from_dict(data.get("memory")),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class EnvironmentData:
    """Environment/system information."""

    hostname: str = ""
    kernel: str = ""
    dkms_version: str = ""
    driver_type: str = ""
    hip_version: str = ""
    torch_version: str = ""
    gpu_count: int = 0
    gpus: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EnvironmentData":
        """Create from dictionary."""
        if data is None:
            return cls()
        return cls(
            hostname=data.get("hostname", ""),
            kernel=data.get("kernel", ""),
            dkms_version=data.get("dkms_version", ""),
            driver_type=data.get("driver_type", ""),
            hip_version=data.get("hip_version", ""),
            torch_version=data.get("torch_version", ""),
            gpu_count=data.get("gpu_count", 0),
            gpus=data.get("gpus", []),
        )

    @classmethod
    def from_environment_info(cls, data: Optional[Dict[str, Any]]) -> "EnvironmentData":
        """Create from a nested environment_info.json dictionary.

        The environment_info.json produced by ``aorta.utils.device.log_environment_info``
        uses a nested structure (``driver`` and ``rocm`` sub-dicts) rather than the
        flat structure stored inside sweep result files.  This factory method handles
        that nested layout so callers do not need to flatten it manually.
        """
        if data is None:
            return cls()
        driver = data.get("driver") or {}
        rocm = data.get("rocm") or {}
        return cls(
            hostname=data.get("hostname", ""),
            kernel=driver.get("kernel", ""),
            dkms_version=driver.get("dkms_version", ""),
            driver_type=driver.get("driver_type", ""),
            hip_version=rocm.get("hip_version", ""),
            torch_version=rocm.get("torch_version", ""),
            gpu_count=data.get("gpu_count", 0),
            gpus=data.get("gpus", []),
        )


@dataclass
class ScalingAnalysisData:
    """Scaling analysis from sweep."""

    stream_counts: List[int] = field(default_factory=list)
    throughputs: List[float] = field(default_factory=list)
    efficiencies: List[float] = field(default_factory=list)
    inflection_point: Optional[int] = None
    peak_stream_count: int = 0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ScalingAnalysisData":
        """Create from dictionary."""
        if data is None:
            return cls()
        return cls(
            stream_counts=data.get("stream_counts", []),
            throughputs=data.get("throughputs", []),
            efficiencies=data.get("efficiencies", []),
            inflection_point=data.get("inflection_point"),
            peak_stream_count=data.get("peak_stream_count", 0),
        )


@dataclass
class SweepData:
    """Parsed sweep result with list of runs."""

    workload_name: str
    results: List[SingleRunData]
    environment: EnvironmentData
    analysis: ScalingAnalysisData
    timestamp: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SweepData":
        """Create from dictionary."""
        results = [SingleRunData.from_dict(r) for r in data.get("results", [])]
        return cls(
            workload_name=data.get("workload", "unknown"),
            results=results,
            environment=EnvironmentData.from_dict(data.get("environment")),
            analysis=ScalingAnalysisData.from_dict(data.get("analysis")),
            timestamp=data.get("timestamp", ""),
        )

    def get_best_throughput(self) -> Tuple[int, float]:
        """Get stream count and throughput for best result."""
        if not self.results:
            return (0, 0.0)
        best = max(self.results, key=lambda r: r.throughput)
        return (best.stream_count, best.throughput)

    def get_result_by_stream_count(self, stream_count: int) -> Optional[SingleRunData]:
        """Get result for a specific stream count."""
        for r in self.results:
            if r.stream_count == stream_count:
                return r
        return None


class HWQueueLoaderError(Exception):
    """Exception raised for loader errors."""

    pass


class HWQueueLoader:
    """Static methods to load and validate hw_queue_eval JSON files."""

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """Load JSON file with error handling."""
        if not path.exists():
            raise HWQueueLoaderError(f"File not found: {path}")
        if not path.is_file():
            raise HWQueueLoaderError(f"Not a file: {path}")

        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            raise HWQueueLoaderError(f"Failed to load JSON from {path}: {e}") from e

    @staticmethod
    def _is_sweep_format(data: Any) -> bool:
        """Check if data is in sweep format (has 'results' array)."""
        return isinstance(data, dict) and "results" in data and isinstance(data["results"], list)

    @staticmethod
    def _is_single_run_format(data: Any) -> bool:
        """Check if data is in single run format."""
        return (
            isinstance(data, dict)
            and "throughput" in data
            and "stream_count" in data
            and "results" not in data
        )

    @staticmethod
    def load_single_run(path: Path) -> SingleRunData:
        """
        Load a single run result JSON file.

        Args:
            path: Path to single run JSON file

        Returns:
            SingleRunData object

        Raises:
            HWQueueLoaderError: If file is invalid or not single run format
        """
        data = HWQueueLoader._load_json(path)

        if not HWQueueLoader._is_single_run_format(data):
            raise HWQueueLoaderError(
                f"File {path} is not in single run format. "
                "Expected 'throughput' and 'stream_count' fields."
            )

        return SingleRunData.from_dict(data)

    @staticmethod
    def load_sweep(path: Path) -> SweepData:
        """
        Load a sweep result JSON file.

        Args:
            path: Path to sweep JSON file

        Returns:
            SweepData object

        Raises:
            HWQueueLoaderError: If file is invalid or not sweep format
        """
        data = HWQueueLoader._load_json(path)

        if not HWQueueLoader._is_sweep_format(data):
            raise HWQueueLoaderError(
                f"File {path} is not in sweep format. " "Expected 'results' array."
            )

        return SweepData.from_dict(data)

    @staticmethod
    def load_auto(path: Path) -> Tuple[str, SingleRunData | SweepData]:
        """
        Auto-detect format and load JSON file. This is used to load a single run or a sweep.

        Args:
            path: Path to JSON file

        Returns:
            Tuple of (format_type, data) where format_type is 'single_run' or 'sweep'

        Raises:
            HWQueueLoaderError: If file format cannot be determined
        """
        data = HWQueueLoader._load_json(path)

        if HWQueueLoader._is_sweep_format(data):
            return ("sweep", SweepData.from_dict(data))
        elif HWQueueLoader._is_single_run_format(data):
            return ("single_run", SingleRunData.from_dict(data))
        else:
            raise HWQueueLoaderError(
                f"Cannot determine format of {path}. "
                "Expected either sweep format (with 'results' array) "
                "or single run format (with 'throughput' and 'stream_count')."
            )

    @staticmethod
    def _load_environment_from_dir(path: Path) -> EnvironmentData:
        """Load environment information from a directory-level environment_info.json.

        When a run directory is produced by ``run-priority`` (or similar tooling),
        an ``environment_info.json`` file written by
        ``aorta.utils.device.log_environment_info`` is typically present alongside
        the per-workload result files.  This helper reads that file and returns an
        :class:`EnvironmentData` instance so that single-run wrappers in
        :meth:`load_directory` can carry accurate system metadata instead of an
        empty placeholder.

        Args:
            path: Directory that may contain ``environment_info.json``.

        Returns:
            :class:`EnvironmentData` populated from the file, or an empty
            :class:`EnvironmentData` when the file is absent or unreadable.
        """
        env_file = path / "environment_info.json"
        if not env_file.is_file():
            return EnvironmentData()
        try:
            data = HWQueueLoader._load_json(env_file)
            return EnvironmentData.from_environment_info(data)
        except Exception as exc:
            log.warning("Failed to load environment_info.json from %s: %s", path, exc)
            return EnvironmentData()

    @staticmethod
    def load_directory(path: Path) -> Dict[str, SweepData]:
        """
        Load all workload result JSON files from a directory.

        Expects files named `*_results.json`. Each file may be either:
        - Sweep format (with a top-level 'results' array), or
        - Single-run format (with fields like 'throughput' and 'stream_count'),
          which will be wrapped into a SweepData containing a single run.

        If an ``environment_info.json`` file is present in the directory (as
        written by ``aorta.utils.device.log_environment_info``), its contents
        are used to populate the :class:`EnvironmentData` of any single-run
        wrappers so that system metadata is preserved consistently.

        Args:
            path: Path to directory containing result files

        Returns:
            Dictionary mapping workload name to SweepData. Single-run files are
            normalized into SweepData instances with one SingleRunData entry.

        Raises:
            HWQueueLoaderError: If directory doesn't exist or contains no valid files
        """
        if not path.exists():
            raise HWQueueLoaderError(f"Directory not found: {path}")
        if not path.is_dir():
            raise HWQueueLoaderError(f"Not a directory: {path}")

        results: Dict[str, SweepData] = {}
        errors: List[str] = []

        # Load directory-level environment info once for use in single-run wrappers.
        dir_environment = HWQueueLoader._load_environment_from_dir(path)

        # Find all *_results.json files
        json_files = sorted(path.glob("*_results.json"))

        if not json_files:
            raise HWQueueLoaderError(
                f"No *_results.json files found in {path}"
            )

        for json_file in json_files:
            try:
                # Extract workload name from filename (e.g., hetero_kernels_results.json -> hetero_kernels)
                workload_name = json_file.stem.replace("_results", "")

                data = HWQueueLoader._load_json(json_file)

                if HWQueueLoader._is_sweep_format(data):
                    sweep_data = SweepData.from_dict(data)
                    # Normalize workload_name to match the filename-derived key
                    try:
                        setattr(sweep_data, "workload_name", workload_name)
                    except Exception:
                        # If SweepData does not have workload_name for some reason,
                        # we still proceed using the filename-derived dict key.
                        pass
                    results[workload_name] = sweep_data
                elif HWQueueLoader._is_single_run_format(data):
                    # Wrap single run in sweep format for consistency, using the
                    # directory-level environment info when available.
                    single_run = SingleRunData.from_dict(data)
                    sweep_data = SweepData(
                        workload_name=workload_name,
                        timestamp=getattr(single_run, "timestamp", ""),
                        results=[single_run],
                        environment=dir_environment,
                        analysis=ScalingAnalysisData(
                            stream_counts=[single_run.stream_count],
                            throughputs=[single_run.throughput],
                            efficiencies=[1.0],
                            peak_stream_count=single_run.stream_count,
                        ),
                    )
                    results[workload_name] = sweep_data
                else:
                    errors.append(f"{json_file.name}: Unknown format")

            except HWQueueLoaderError as e:
                errors.append(str(e))
            except Exception as e:
                errors.append(f"{json_file.name}: {e}")

        if not results:
            error_msg = "No valid result files loaded."
            if errors:
                error_msg += f" Errors: {'; '.join(errors)}"
            raise HWQueueLoaderError(error_msg)

        return results

    @staticmethod
    def find_common_workloads(
        baseline_dir: Path, test_dir: Path
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Find common and missing workloads between baseline and test directories.

        Args:
            baseline_dir: Path to baseline results directory
            test_dir: Path to test results directory

        Returns:
            Tuple of (common_workloads, baseline_only, test_only)

        Raises:
            HWQueueLoaderError: If directories are invalid
        """
        # Get workload names from file names
        def get_workloads(dir_path: Path) -> set:
            if not dir_path.exists() or not dir_path.is_dir():
                raise HWQueueLoaderError(f"Invalid directory: {dir_path}")
            workloads = set()
            for f in dir_path.glob("*_results.json"):
                workload_name = f.stem.replace("_results", "")
                workloads.add(workload_name)
            return workloads

        baseline_workloads = get_workloads(baseline_dir)
        test_workloads = get_workloads(test_dir)

        common = sorted(baseline_workloads & test_workloads)
        baseline_only = sorted(baseline_workloads - test_workloads)
        test_only = sorted(test_workloads - baseline_workloads)

        return (common, baseline_only, test_only)

    @staticmethod
    def load_comparison_data(
        baseline_dir: Path, test_dir: Path
    ) -> Tuple[Dict[str, SweepData], Dict[str, SweepData], List[str], List[str], List[str]]:
        """
        Load data for comparison mode (Mode C).

        Args:
            baseline_dir: Path to baseline results directory
            test_dir: Path to test results directory

        Returns:
            Tuple of (baseline_data, test_data, common_workloads, baseline_only, test_only)
        """
        common, baseline_only, test_only = HWQueueLoader.find_common_workloads(
            baseline_dir, test_dir
        )

        # TODO: Optimize by passing an allowlist to load_directory() or pre-filtering
        # the result file list so only common workloads are loaded, avoiding unnecessary
        # I/O and JSON parsing when result directories are large.
        baseline_data = HWQueueLoader.load_directory(baseline_dir)
        test_data = HWQueueLoader.load_directory(test_dir)

        # Filter to only common workloads
        baseline_common = {k: v for k, v in baseline_data.items() if k in common}
        test_common = {k: v for k, v in test_data.items() if k in common}

        return (baseline_common, test_common, common, baseline_only, test_only)
