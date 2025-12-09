"""
Tests for the core harness functionality.

Tests:
- Stream creation and management
- Timing accuracy
- Result aggregation
- Sweep functionality
"""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestHarnessConfig:
    """Tests for HarnessConfig."""

    def test_valid_config(self):
        """Test creating a valid config."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig

        config = HarnessConfig(stream_count=4)
        assert config.stream_count == 4
        assert config.warmup_iterations == 10
        assert config.measurement_iterations == 100

    def test_invalid_stream_count(self):
        """Test that stream_count < 1 raises error."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig

        with pytest.raises(ValueError):
            HarnessConfig(stream_count=0)

    def test_invalid_sync_mode(self):
        """Test that invalid sync_mode raises error."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig

        with pytest.raises(ValueError):
            HarnessConfig(stream_count=4, sync_mode="invalid")

    def test_config_to_dict(self):
        """Test config serialization."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig

        config = HarnessConfig(stream_count=8, warmup_iterations=5)
        d = config.to_dict()

        assert d["stream_count"] == 8
        assert d["warmup_iterations"] == 5


class TestStreamHarness:
    """Tests for StreamHarness."""

    def test_harness_creates_correct_stream_count(self):
        """Test that harness creates the correct number of streams."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        config = HarnessConfig(
            stream_count=4,
            warmup_iterations=1,
            measurement_iterations=2,
        )
        harness = StreamHarness(config)
        harness._initialize()

        assert len(harness.streams) == 4
        harness._cleanup()

    def test_simple_workload_run(self):
        """Test running a simple workload function."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        config = HarnessConfig(
            stream_count=2,
            warmup_iterations=1,
            measurement_iterations=5,
        )
        harness = StreamHarness(config)

        # Simple workload that does some GPU work
        def simple_workload(streams):
            for stream in streams:
                with torch.cuda.stream(stream):
                    a = torch.randn(100, 100, device="cuda")
                    b = torch.mm(a, a)

        result = harness.run(simple_workload, workload_name="test")

        assert result.stream_count == 2
        assert result.throughput > 0
        assert result.total_time_ms > 0
        assert "mean" in result.latency_ms

    def test_harness_timing_accuracy(self):
        """Verify timing is within expected tolerance."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        config = HarnessConfig(
            stream_count=1,
            warmup_iterations=2,
            measurement_iterations=10,
        )
        harness = StreamHarness(config)

        # Workload with known approximate duration
        def timed_workload(streams):
            with torch.cuda.stream(streams[0]):
                # Do enough work to have measurable time
                a = torch.randn(1000, 1000, device="cuda")
                for _ in range(10):
                    a = torch.mm(a, a)

        result = harness.run(timed_workload)

        # Total time should be positive and reasonable
        assert result.total_time_ms > 0
        assert result.total_time_ms < 60000  # Less than 60 seconds

        # Iteration times should be consistent
        times = result.iteration_times_ms
        assert len(times) == 10
        assert all(t > 0 for t in times)

    def test_sweep_returns_results_for_all_counts(self):
        """Test that sweep returns results for all stream counts."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        config = HarnessConfig(
            stream_count=1,  # Will be overridden
            warmup_iterations=1,
            measurement_iterations=3,
        )
        harness = StreamHarness(config)

        def simple_workload(streams):
            for stream in streams:
                with torch.cuda.stream(stream):
                    torch.randn(50, 50, device="cuda")

        stream_counts = [1, 2, 4]
        results = harness.sweep(simple_workload, stream_counts)

        assert len(results) == 3
        assert results[0].stream_count == 1
        assert results[1].stream_count == 2
        assert results[2].stream_count == 4


class TestHarnessResult:
    """Tests for HarnessResult."""

    def test_result_to_json(self):
        """Test result JSON serialization."""
        from aorta.hw_queue_eval.core.harness import HarnessResult
        import json

        result = HarnessResult(
            throughput=1000.0,
            throughput_unit="ops/sec",
            latency_ms={"mean": 10.0, "p50": 9.0, "p95": 15.0, "p99": 20.0},
            total_time_ms=1000.0,
            stream_count=4,
            per_stream_times_ms=[250.0, 250.0, 250.0, 250.0],
            iteration_times_ms=[100.0] * 10,
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["throughput"] == 1000.0
        assert parsed["stream_count"] == 4

    def test_result_to_dict(self):
        """Test result dictionary conversion."""
        from aorta.hw_queue_eval.core.harness import HarnessResult

        result = HarnessResult(
            throughput=500.0,
            throughput_unit="samples/sec",
            latency_ms={"mean": 5.0, "p50": 4.0, "p95": 8.0, "p99": 10.0},
            total_time_ms=2000.0,
            stream_count=8,
            per_stream_times_ms=[250.0] * 8,
            iteration_times_ms=[20.0] * 100,
        )

        d = result.to_dict()

        assert d["throughput"] == 500.0
        assert d["throughput_unit"] == "samples/sec"
        assert d["stream_count"] == 8


class TestAnalysisUtilities:
    """Tests for result analysis utilities."""

    def test_analyze_sweep_results(self):
        """Test scaling analysis from sweep results."""
        from aorta.hw_queue_eval.core.harness import HarnessResult, analyze_sweep_results

        # Create mock results with diminishing returns
        results = []
        for streams, tp in [(1, 100), (2, 190), (4, 350), (8, 400), (16, 410)]:
            results.append(HarnessResult(
                throughput=tp,
                throughput_unit="ops/sec",
                latency_ms={"mean": 10.0, "p50": 9.0, "p95": 15.0, "p99": 20.0},
                total_time_ms=1000.0,
                stream_count=streams,
                per_stream_times_ms=[100.0] * streams,
                iteration_times_ms=[10.0] * 100,
            ))

        analysis = analyze_sweep_results(results)

        assert analysis.stream_counts == [1, 2, 4, 8, 16]
        assert analysis.peak_stream_count in [8, 16]  # Best throughput
        assert len(analysis.efficiencies) == 5

    def test_format_results_table(self):
        """Test table formatting."""
        from aorta.hw_queue_eval.core.harness import HarnessResult, format_results_table

        results = [
            HarnessResult(
                throughput=100.0,
                throughput_unit="ops/sec",
                latency_ms={"mean": 10.0, "p50": 9.0, "p95": 15.0, "p99": 20.0, "min": 5.0, "max": 25.0, "std": 3.0},
                total_time_ms=1000.0,
                stream_count=4,
                per_stream_times_ms=[250.0] * 4,
                iteration_times_ms=[10.0] * 100,
            )
        ]

        table = format_results_table(results)

        assert "Streams" in table
        assert "Throughput" in table
        assert "100.00" in table
