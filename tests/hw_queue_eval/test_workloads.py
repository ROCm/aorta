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


class TestCommsComputeOverlapWorkload:
    """Tests for the comms_compute_overlap workload (simulated collectives)."""

    # Tests for real collectives (simulate_collectives=False) are not
    # included here because they require multi-process launch via torchrun
    # (torch.distributed.init_process_group needs RANK, WORLD_SIZE, etc.
    # env vars).  The existing test suite runs as a single pytest process.

    @pytest.mark.parametrize("stream_count", [2, 4, 8])
    def test_runs_at_various_stream_counts(self, stream_count):
        """Test workload runs without error at different stream counts."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        workload = get_workload(
            "comms_compute_overlap",
            mm_dim=(512, 512, 512),
            num_compute_per_iter=2,
            comm_size_bytes=1 * 1024 * 1024,
        )

        config = HarnessConfig(
            stream_count=stream_count,
            warmup_iterations=2,
            measurement_iterations=5,
        )
        harness = StreamHarness(config)
        result = harness.run_workload(workload)

        assert result.throughput > 0
        assert result.stream_count == stream_count

    @pytest.mark.parametrize("mode", ["compute_only", "comms_only", "comms_compute"])
    def test_all_modes(self, mode):
        """Test all three workload modes run without error."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        workload = get_workload(
            "comms_compute_overlap",
            mode=mode,
            mm_dim=(512, 512, 512),
            num_compute_per_iter=2,
            comm_size_bytes=1 * 1024 * 1024,
        )

        config = HarnessConfig(
            stream_count=4,
            warmup_iterations=2,
            measurement_iterations=5,
        )
        harness = StreamHarness(config)
        result = harness.run_workload(workload)

        assert result.throughput > 0

    def test_produces_valid_metrics(self):
        """Test that metrics are valid."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        workload = get_workload(
            "comms_compute_overlap",
            mm_dim=(512, 512, 512),
            num_compute_per_iter=2,
            comm_size_bytes=1 * 1024 * 1024,
        )

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
        assert result.throughput_unit == "TFLOPS"

    def test_comms_only_throughput_unit(self):
        """Test comms_only mode reports GB/s."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        workload = get_workload(
            "comms_compute_overlap",
            mode="comms_only",
            comm_size_bytes=1 * 1024 * 1024,
        )

        config = HarnessConfig(
            stream_count=4,
            warmup_iterations=2,
            measurement_iterations=5,
        )
        harness = StreamHarness(config)
        result = harness.run_workload(workload)

        assert result.throughput_unit == "GB/s"

    def test_correctness_validation(self):
        """Test correctness validation passes."""
        workload = get_workload(
            "comms_compute_overlap",
            mm_dim=(512, 512, 512),
            num_compute_per_iter=2,
            comm_size_bytes=1 * 1024 * 1024,
        )
        workload.setup(stream_count=4, device="cuda:0")

        is_correct, message = workload.validate_correctness(None, None)

        assert is_correct
        workload.cleanup()

    @pytest.mark.parametrize("compute_streams", [1, 2, 4])
    def test_explicit_compute_streams(self, compute_streams):
        """Test decoupled compute stream count."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        workload = get_workload(
            "comms_compute_overlap",
            mm_dim=(512, 512, 512),
            num_compute_per_iter=2,
            compute_streams=compute_streams,
            comm_size_bytes=1 * 1024 * 1024,
        )

        config = HarnessConfig(
            stream_count=8,
            warmup_iterations=2,
            measurement_iterations=5,
        )
        harness = StreamHarness(config)
        result = harness.run_workload(workload)

        assert result.throughput > 0

    @pytest.mark.parametrize("comp_dt,comm_dt", [
        ("float32", "float32"),
        ("bfloat16", "bfloat16"),
        ("float16", "float32"),
    ])
    def test_data_types(self, comp_dt, comm_dt):
        """Test various compute and comm data type combinations."""
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

        workload = get_workload(
            "comms_compute_overlap",
            mm_dim=(512, 512, 512),
            num_compute_per_iter=2,
            comp_data_type=comp_dt,
            comm_data_type=comm_dt,
            comm_size_bytes=1 * 1024 * 1024,
        )

        config = HarnessConfig(
            stream_count=4,
            warmup_iterations=2,
            measurement_iterations=5,
        )
        harness = StreamHarness(config)
        result = harness.run_workload(workload)

        assert result.throughput > 0

    def test_get_config(self):
        """Test that get_config returns expected keys."""
        workload = get_workload(
            "comms_compute_overlap",
            mm_dim=(1024, 1024, 1024),
            compute_streams=2,
            comp_data_type="bfloat16",
            comm_data_type="float16",
        )
        workload.setup(stream_count=4, device="cuda:0")

        config = workload.get_config()

        assert config["name"] == "comms_compute_overlap"
        assert config["mm_dim"] == (1024, 1024, 1024)
        assert config["num_compute_streams"] == 2
        assert "bfloat16" in config["comp_data_type"]
        assert "float16" in config["comm_data_type"]
        workload.cleanup()


class TestParseProcessGroups:
    """Tests for the process-group spec parser."""

    def test_single_group(self):
        from aorta.utils.distributed import parse_process_groups
        assert parse_process_groups("[0,1,2,3]") == {0: [0, 1, 2, 3]}

    def test_two_groups(self):
        from aorta.utils.distributed import parse_process_groups
        assert parse_process_groups("[0,1],[2,3]") == {0: [0, 1], 1: [2, 3]}

    def test_four_groups(self):
        from aorta.utils.distributed import parse_process_groups
        result = parse_process_groups("[0,1],[2,3],[4,5],[6,7]")
        assert result == {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}

    def test_single_rank_groups(self):
        from aorta.utils.distributed import parse_process_groups
        assert parse_process_groups("[0],[1],[2]") == {0: [0], 1: [1], 2: [2]}

    def test_whitespace_handling(self):
        from aorta.utils.distributed import parse_process_groups
        assert parse_process_groups(" [ 0 , 1 ] , [ 2 , 3 ] ") == {0: [0, 1], 1: [2, 3]}

    def test_empty_string_raises(self):
        from aorta.utils.distributed import parse_process_groups
        with pytest.raises(ValueError, match="non-empty"):
            parse_process_groups("")

    def test_whitespace_only_raises(self):
        from aorta.utils.distributed import parse_process_groups
        with pytest.raises(ValueError, match="non-empty"):
            parse_process_groups("   ")

    def test_empty_brackets_raises(self):
        from aorta.utils.distributed import parse_process_groups
        with pytest.raises(ValueError, match="does not contain any ranks"):
            parse_process_groups("[]")

    def test_empty_group_in_list_raises(self):
        from aorta.utils.distributed import parse_process_groups
        with pytest.raises(ValueError, match="Empty process group"):
            parse_process_groups("[0,1],[],[2,3]")

    def test_non_integer_rank_raises(self):
        from aorta.utils.distributed import parse_process_groups
        with pytest.raises(ValueError, match="Non-integer"):
            parse_process_groups("[0,abc,2]")

    def test_float_rank_raises(self):
        from aorta.utils.distributed import parse_process_groups
        with pytest.raises(ValueError, match="Non-integer"):
            parse_process_groups("[0,1.5,2]")

    def test_negative_rank_raises(self):
        from aorta.utils.distributed import parse_process_groups
        with pytest.raises(ValueError, match="Negative rank"):
            parse_process_groups("[0,-1,2]")

    def test_large_ranks(self):
        from aorta.utils.distributed import parse_process_groups
        assert parse_process_groups("[100,200,300]") == {0: [100, 200, 300]}

    def test_preserves_order(self):
        from aorta.utils.distributed import parse_process_groups
        assert parse_process_groups("[3,1,2,0]") == {0: [3, 1, 2, 0]}


class TestParseSize:
    """Tests for the CLI _parse_size helper."""

    def test_plain_integer(self):
        from aorta.hw_queue_eval.cli import _parse_size
        assert _parse_size("1024") == 1024

    def test_megabytes(self):
        from aorta.hw_queue_eval.cli import _parse_size
        assert _parse_size("128M") == 128 * 1024 * 1024

    def test_gigabytes(self):
        from aorta.hw_queue_eval.cli import _parse_size
        assert _parse_size("1G") == 1024 ** 3

    def test_lowercase_suffix(self):
        from aorta.hw_queue_eval.cli import _parse_size
        assert _parse_size("64m") == 64 * 1024 * 1024

    def test_fractional_with_suffix(self):
        from aorta.hw_queue_eval.cli import _parse_size
        assert _parse_size("1.5G") == int(1.5 * 1024 ** 3)

    def test_invalid_string_raises(self):
        from aorta.hw_queue_eval.cli import _parse_size
        with pytest.raises(ValueError, match="Invalid size"):
            _parse_size("abc")

    def test_empty_string_raises(self):
        from aorta.hw_queue_eval.cli import _parse_size
        with pytest.raises(ValueError, match="Invalid size"):
            _parse_size("")

    def test_suffix_only_raises(self):
        from aorta.hw_queue_eval.cli import _parse_size
        with pytest.raises(ValueError, match="Invalid size"):
            _parse_size("M")

    def test_negative_value_raises(self):
        from aorta.hw_queue_eval.cli import _parse_size
        with pytest.raises(ValueError, match="non-negative"):
            _parse_size("-10M")


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
