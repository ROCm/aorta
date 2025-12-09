"""
Tests for workload implementations.

Tests each workload to ensure:
- Runs without error at various stream counts
- Produces valid metrics
- Correctness validation passes (where implemented)
"""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


def get_workload(name: str, **kwargs):
    """Get a workload by name."""
    from aorta.hw_queue_eval.workloads.registry import get_workload
    return get_workload(name, **kwargs)


class TestHeterogeneousKernelWorkload:
    """Tests for the hetero_kernels workload."""

    @pytest.mark.parametrize("stream_count", [2, 4, 8])
    def test_runs_at_various_stream_counts(self, stream_count):
        """Test workload runs without error."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        workload = get_workload("hetero_kernels")

        config = HarnessConfig(
            stream_count=stream_count,
            warmup_iterations=2,
            measurement_iterations=5,
        )
        harness = StreamHarness(config)
        result = harness.run_workload(workload)

        assert result.throughput > 0
        assert result.stream_count == stream_count

    def test_produces_valid_metrics(self):
        """Test that metrics are valid."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        workload = get_workload("hetero_kernels")

        config = HarnessConfig(
            stream_count=4,
            warmup_iterations=2,
            measurement_iterations=10,
        )
        harness = StreamHarness(config)
        result = harness.run_workload(workload)

        assert result.latency_ms["mean"] > 0
        assert result.latency_ms["p50"] > 0
        assert result.latency_ms["p99"] >= result.latency_ms["p50"]
        assert result.throughput_unit == "GFLOPS"

    def test_correctness_validation(self):
        """Test correctness validation."""
        workload = get_workload("hetero_kernels")
        workload.setup(stream_count=4, device="cuda:0")

        is_correct, message = workload.validate_correctness(None, None)

        assert is_correct
        workload.cleanup()


class TestTinyKernelStressWorkload:
    """Tests for tiny_kernel_stress workload."""

    @pytest.mark.parametrize("stream_count", [1, 4, 8, 16])
    def test_runs_at_various_stream_counts(self, stream_count):
        """Test workload runs at high stream counts."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        workload = get_workload("tiny_kernel_stress")

        config = HarnessConfig(
            stream_count=stream_count,
            warmup_iterations=2,
            measurement_iterations=5,
        )
        harness = StreamHarness(config)
        result = harness.run_workload(workload)

        assert result.throughput > 0


class TestFSDPTPWorkload:
    """Tests for fsdp_tp workload."""

    @pytest.mark.parametrize("stream_count", [4, 8, 10])
    def test_runs_at_various_stream_counts(self, stream_count):
        """Test FSDP+TP workload."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        workload = get_workload("fsdp_tp", model_size="small")

        config = HarnessConfig(
            stream_count=stream_count,
            warmup_iterations=2,
            measurement_iterations=5,
        )
        harness = StreamHarness(config)
        result = harness.run_workload(workload)

        assert result.throughput > 0
        assert result.throughput_unit == "samples/sec"


class TestMoEWorkload:
    """Tests for moe workload."""

    @pytest.mark.parametrize("stream_count", [4, 8, 16])
    def test_runs_at_various_stream_counts(self, stream_count):
        """Test MoE workload with multiple experts."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        workload = get_workload(
            "moe",
            num_experts=8,
            hidden_size=512,
            batch_size=2,
            seq_length=128,
        )

        config = HarnessConfig(
            stream_count=stream_count,
            warmup_iterations=2,
            measurement_iterations=5,
        )
        harness = StreamHarness(config)
        result = harness.run_workload(workload)

        assert result.throughput > 0
        assert result.throughput_unit == "tokens/sec"


class TestSpeculativeDecodeWorkload:
    """Tests for speculative_decode workload."""

    @pytest.mark.parametrize("stream_count", [4, 6, 8])
    def test_runs_at_various_stream_counts(self, stream_count):
        """Test speculative decoding workload."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        workload = get_workload(
            "speculative_decode",
            draft_hidden_size=128,
            draft_num_layers=2,
            main_hidden_size=256,
            main_num_layers=4,
        )

        config = HarnessConfig(
            stream_count=stream_count,
            warmup_iterations=2,
            measurement_iterations=5,
        )
        harness = StreamHarness(config)
        result = harness.run_workload(workload)

        assert result.throughput > 0


class TestContinuousBatchWorkload:
    """Tests for continuous_batch workload."""

    @pytest.mark.parametrize("stream_count", [4, 6, 8])
    def test_runs_at_various_stream_counts(self, stream_count):
        """Test continuous batching workload."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        workload = get_workload(
            "continuous_batch",
            hidden_size=256,
            num_layers=2,
            prefill_batch_size=1,
            decode_batch_size=4,
        )

        config = HarnessConfig(
            stream_count=stream_count,
            warmup_iterations=2,
            measurement_iterations=5,
        )
        harness = StreamHarness(config)
        result = harness.run_workload(workload)

        assert result.throughput > 0


class TestGraphSubgraphsWorkload:
    """Tests for graph_subgraphs workload."""

    @pytest.mark.parametrize("stream_count", [4, 8, 12])
    def test_runs_at_various_stream_counts(self, stream_count):
        """Test independent subgraph execution."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        workload = get_workload(
            "graph_subgraphs",
            num_subgraphs=4,
            hidden_size=512,
            batch_size=16,
        )

        config = HarnessConfig(
            stream_count=stream_count,
            warmup_iterations=2,
            measurement_iterations=5,
        )
        harness = StreamHarness(config)
        result = harness.run_workload(workload)

        assert result.throughput > 0


class TestWorkloadRegistry:
    """Tests for workload registry."""

    def test_list_all_workloads(self):
        """Test listing all workloads."""
        from aorta.hw_queue_eval.workloads.registry import list_workloads

        workloads = list_workloads()

        assert len(workloads) > 0
        assert "hetero_kernels" in workloads

    def test_get_workload_info(self):
        """Test getting workload info."""
        from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry

        info = WorkloadRegistry.get_info("hetero_kernels")

        assert info.name == "hetero_kernels"
        assert info.category == "latency_sensitive"
        assert info.switch_latency_sensitivity == "critical"

    def test_list_by_category(self):
        """Test filtering by category."""
        from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry

        latency_sensitive = WorkloadRegistry.list_by_category("latency_sensitive")

        assert "hetero_kernels" in latency_sensitive
        assert "graph_subgraphs" in latency_sensitive

    def test_unknown_workload_raises(self):
        """Test that unknown workload raises KeyError."""
        from aorta.hw_queue_eval.workloads.registry import get_workload

        with pytest.raises(KeyError):
            get_workload("nonexistent_workload")


class TestWorkloadCleanup:
    """Tests for workload cleanup."""

    def test_cleanup_releases_memory(self):
        """Test that cleanup releases GPU memory."""
        workload = get_workload("hetero_kernels")
        workload.setup(stream_count=4, device="cuda:0")

        # Record memory before cleanup
        before_cleanup = torch.cuda.memory_allocated()

        workload.cleanup()
        torch.cuda.empty_cache()

        # Memory should be reduced after cleanup
        after_cleanup = torch.cuda.memory_allocated()

        assert after_cleanup <= before_cleanup


class TestWorkloadStreamCompatibility:
    """Tests for stream count compatibility."""

    def test_supports_stream_count(self):
        """Test stream count support checking."""
        workload = get_workload("hetero_kernels")

        assert workload.supports_stream_count(4)
        assert workload.supports_stream_count(8)
        assert not workload.supports_stream_count(0)
        assert not workload.supports_stream_count(100)

    def test_workload_limits(self):
        """Test that workloads respect their limits."""
        workload = get_workload("fsdp_tp")

        assert workload.min_streams > 0
        assert workload.max_streams >= workload.min_streams
        assert workload.recommended_streams >= workload.min_streams
        assert workload.recommended_streams <= workload.max_streams
