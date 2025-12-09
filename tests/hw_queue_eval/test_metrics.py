"""
Tests for the metrics collection module.

Tests:
- Latency metrics computation
- Switch latency estimation
- Metrics collector functionality
"""

import pytest


class TestLatencyMetrics:
    """Tests for LatencyMetrics."""

    def test_from_samples_empty(self):
        """Test with empty sample list."""
        from aorta.hw_queue_eval.core.metrics import LatencyMetrics

        metrics = LatencyMetrics.from_samples([])

        assert metrics.count == 0
        assert metrics.mean_ms == 0.0

    def test_from_samples_single(self):
        """Test with single sample."""
        from aorta.hw_queue_eval.core.metrics import LatencyMetrics

        metrics = LatencyMetrics.from_samples([10.0])

        assert metrics.count == 1
        assert metrics.mean_ms == 10.0
        assert metrics.p50_ms == 10.0
        assert metrics.std_ms == 0.0

    def test_from_samples_multiple(self):
        """Test with multiple samples."""
        from aorta.hw_queue_eval.core.metrics import LatencyMetrics

        samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        metrics = LatencyMetrics.from_samples(samples)

        assert metrics.count == 10
        assert metrics.mean_ms == 5.5
        assert metrics.min_ms == 1.0
        assert metrics.max_ms == 10.0
        assert metrics.p50_ms == pytest.approx(5.5, rel=0.1)

    def test_to_dict(self):
        """Test dictionary conversion."""
        from aorta.hw_queue_eval.core.metrics import LatencyMetrics

        metrics = LatencyMetrics(
            mean_ms=10.0,
            p50_ms=9.0,
            p95_ms=15.0,
            p99_ms=20.0,
            min_ms=5.0,
            max_ms=25.0,
            count=100,
        )

        d = metrics.to_dict()

        assert d["mean_ms"] == 10.0
        assert d["p99_ms"] == 20.0
        assert d["count"] == 100


class TestSwitchLatencyMetrics:
    """Tests for SwitchLatencyMetrics."""

    def test_from_kernel_timings_empty(self):
        """Test with empty timing list."""
        from aorta.hw_queue_eval.core.metrics import SwitchLatencyMetrics

        metrics = SwitchLatencyMetrics.from_kernel_timings([])

        assert metrics.inter_stream_gap_ms == 0.0
        assert metrics.intra_stream_gap_ms == 0.0

    def test_from_kernel_timings_same_stream(self):
        """Test with kernels on same stream."""
        from aorta.hw_queue_eval.core.metrics import SwitchLatencyMetrics

        # (stream_id, start_ms, end_ms)
        timings = [
            (0, 0.0, 10.0),
            (0, 11.0, 20.0),
            (0, 21.0, 30.0),
        ]

        metrics = SwitchLatencyMetrics.from_kernel_timings(timings)

        assert metrics.intra_stream_samples > 0
        assert metrics.intra_stream_gap_ms == pytest.approx(1.0, rel=0.1)

    def test_from_kernel_timings_different_streams(self):
        """Test with kernels on different streams."""
        from aorta.hw_queue_eval.core.metrics import SwitchLatencyMetrics

        # Alternating streams
        timings = [
            (0, 0.0, 10.0),
            (1, 12.0, 22.0),
            (0, 25.0, 35.0),
            (1, 38.0, 48.0),
        ]

        metrics = SwitchLatencyMetrics.from_kernel_timings(timings)

        assert metrics.inter_stream_samples > 0
        assert metrics.inter_stream_gap_ms > 0


class TestThroughputMetrics:
    """Tests for ThroughputMetrics."""

    def test_compute_basic(self):
        """Test basic throughput computation."""
        from aorta.hw_queue_eval.core.metrics import ThroughputMetrics

        metrics = ThroughputMetrics.compute(1000, 10.0, "ops/sec")

        assert metrics.value == 100.0
        assert metrics.unit == "ops/sec"
        assert metrics.raw_count == 1000
        assert metrics.duration_sec == 10.0

    def test_compute_zero_duration(self):
        """Test with zero duration."""
        from aorta.hw_queue_eval.core.metrics import ThroughputMetrics

        metrics = ThroughputMetrics.compute(1000, 0.0, "ops/sec")

        assert metrics.value == 0.0


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_collector_basic(self):
        """Test basic metrics collection."""
        import torch
        from aorta.hw_queue_eval.core.metrics import MetricsCollector

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        collector = MetricsCollector(num_streams=2)

        for _ in range(5):
            collector.start_iteration()
            # Simulate some work
            collector.end_iteration(sync=False)

        times = collector.get_iteration_times()
        assert len(times) == 5

    def test_collector_compute_latency(self):
        """Test latency computation."""
        import torch
        from aorta.hw_queue_eval.core.metrics import MetricsCollector

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        collector = MetricsCollector(num_streams=1)

        for _ in range(10):
            collector.start_iteration()
            # Do some GPU work
            a = torch.randn(100, 100, device="cuda")
            _ = torch.mm(a, a)
            collector.end_iteration()

        latency = collector.compute_latency_metrics()

        assert latency.count == 10
        assert latency.mean_ms > 0

    def test_collector_clear(self):
        """Test clearing collected data."""
        import torch
        from aorta.hw_queue_eval.core.metrics import MetricsCollector

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        collector = MetricsCollector(num_streams=1)

        collector.start_iteration()
        collector.end_iteration(sync=False)

        assert len(collector.get_iteration_times()) == 1

        collector.clear()

        assert len(collector.get_iteration_times()) == 0


class TestScalingAnalysis:
    """Tests for ScalingAnalysis."""

    def test_from_sweep_results_empty(self):
        """Test with empty results."""
        from aorta.hw_queue_eval.core.metrics import ScalingAnalysis

        analysis = ScalingAnalysis.from_sweep_results([])

        assert analysis.stream_counts == []
        assert analysis.peak_stream_count == 0

    def test_from_sweep_results_scaling(self):
        """Test with scaling results."""
        from aorta.hw_queue_eval.core.metrics import ScalingAnalysis

        # (stream_count, throughput) - linear scaling initially, then drops
        results = [
            (1, 100),
            (2, 195),
            (4, 380),
            (8, 500),  # Starts to plateau
            (16, 520),
        ]

        analysis = ScalingAnalysis.from_sweep_results(results)

        assert analysis.stream_counts == [1, 2, 4, 8, 16]
        assert len(analysis.efficiencies) == 5
        # Efficiency should decrease as scaling breaks down
        assert analysis.efficiencies[0] > analysis.efficiencies[-1]


class TestCompareResults:
    """Tests for result comparison."""

    def test_no_regression(self):
        """Test comparison with no regression."""
        from aorta.hw_queue_eval.core.metrics import compare_results

        baseline = {"throughput": 100.0, "latency": {"p50_ms": 10.0, "p95_ms": 15.0, "p99_ms": 20.0}}
        test = {"throughput": 102.0, "latency": {"p50_ms": 10.0, "p95_ms": 15.0, "p99_ms": 20.0}}

        comparison = compare_results(baseline, test, threshold=0.05)

        assert not comparison["has_regressions"]

    def test_throughput_regression(self):
        """Test detection of throughput regression."""
        from aorta.hw_queue_eval.core.metrics import compare_results

        baseline = {"throughput": 100.0}
        test = {"throughput": 90.0}  # 10% regression

        comparison = compare_results(baseline, test, threshold=0.05)

        assert comparison["has_regressions"]
        assert len(comparison["regressions"]) > 0

    def test_latency_regression(self):
        """Test detection of latency regression."""
        from aorta.hw_queue_eval.core.metrics import compare_results

        baseline = {"latency": {"p99_ms": 10.0}}
        test = {"latency": {"p99_ms": 12.0}}  # 20% increase

        comparison = compare_results(baseline, test, threshold=0.05)

        assert comparison["has_regressions"]

    def test_improvement_detected(self):
        """Test detection of improvements."""
        from aorta.hw_queue_eval.core.metrics import compare_results

        baseline = {"throughput": 100.0}
        test = {"throughput": 120.0}  # 20% improvement

        comparison = compare_results(baseline, test, threshold=0.05)

        assert not comparison["has_regressions"]
        assert len(comparison["improvements"]) > 0
