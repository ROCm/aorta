"""
Heterogeneous Kernel Mixing Workload

Pattern: Interleave tiny kernels (elementwise, ~10µs) with large GEMMs (~10ms)
Goal: Test scheduler's ability to dispatch small kernels without convoy effect

Stream assignment:
- First half of streams: Large GEMM operations
- Second half of streams: Tiny elementwise operations
- Interleaved dispatch to stress queue switching

Expected behavior with good queue mapping:
- Tiny kernels should not wait for large GEMMs on other queues
- Throughput should scale with stream count up to HW queue limit
- Switch latency overhead should be minimal

This is the most direct test for hardware queue switch latency issues.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from aorta.hw_queue_eval.workloads.base import SyntheticWorkload
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry


@WorkloadRegistry.register
class HeterogeneousKernelWorkload(SyntheticWorkload):
    """
    Mixed tiny and large kernels to test convoy effect and switch latency.

    This workload dispatches:
    - Large GEMM operations (configurable size, ~10ms each)
    - Tiny elementwise operations (configurable size, ~10µs each)

    The pattern interleaves these operations across streams to stress
    the hardware queue scheduler's ability to dispatch work efficiently.
    """

    name = "hetero_kernels"
    description = "Mixed tiny and large kernels to test convoy effect"
    category = "latency_sensitive"
    min_streams = 2
    max_streams = 64
    recommended_streams = 8
    switch_latency_sensitivity = "critical"
    memory_requirements_gb = 4.0

    def __init__(
        self,
        large_gemm_size: Tuple[int, int, int] = (4096, 4096, 4096),  # M, N, K
        small_kernel_size: int = 1024,
        large_to_small_ratio: int = 10,  # small kernels per large GEMM
        interleave_pattern: str = "alternating",  # "alternating", "batched", "random"
    ):
        """
        Initialize the heterogeneous kernel workload.

        Args:
            large_gemm_size: (M, N, K) dimensions for large GEMMs
            small_kernel_size: Size of elementwise tensors
            large_to_small_ratio: Number of small kernels per large GEMM
            interleave_pattern: How to interleave operations
        """
        super().__init__()
        self.large_gemm_size = large_gemm_size
        self.small_kernel_size = small_kernel_size
        self.large_to_small_ratio = large_to_small_ratio
        self.interleave_pattern = interleave_pattern

        # Will be set in setup()
        self._large_tensors_a: List[torch.Tensor] = []
        self._large_tensors_b: List[torch.Tensor] = []
        self._small_tensors: List[torch.Tensor] = []
        self._num_large_streams = 0
        self._num_small_streams = 0

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """
        Initialize tensors for large GEMMs and small elementwise ops.

        Args:
            stream_count: Number of streams to use
            device: Target device
        """
        self._stream_count = stream_count
        self._device = device
        self._is_setup = True

        # Split streams between large and small kernels
        self._num_large_streams = max(1, stream_count // 2)
        self._num_small_streams = max(1, stream_count - self._num_large_streams)

        m, n, k = self.large_gemm_size

        # Allocate tensors for large GEMMs (one pair per large stream)
        self._large_tensors_a = []
        self._large_tensors_b = []
        for i in range(self._num_large_streams):
            a = torch.randn(m, k, dtype=torch.float32, device=device)
            b = torch.randn(k, n, dtype=torch.float32, device=device)
            self._large_tensors_a.append(a)
            self._large_tensors_b.append(b)
            self._tensors[f"large_a_{i}"] = a
            self._tensors[f"large_b_{i}"] = b

        # Allocate tensors for small elementwise ops
        # Multiple tensors per small stream for the ratio
        self._small_tensors = []
        num_small_tensors = self._num_small_streams * self.large_to_small_ratio
        for i in range(num_small_tensors):
            t = torch.randn(self.small_kernel_size, dtype=torch.float32, device=device)
            self._small_tensors.append(t)
            self._tensors[f"small_{i}"] = t

        # Compute ops per iteration for throughput calculation
        # Large GEMM: 2*M*N*K FLOPs each
        large_flops = 2 * m * n * k * self._num_large_streams
        # Small ops: ~3 FLOPs per element (add, mul, add in fused op)
        small_flops = 3 * self.small_kernel_size * num_small_tensors
        self._ops_per_iteration = large_flops + small_flops

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Execute one iteration with interleaved large and small kernels.

        Args:
            streams: List of CUDA/HIP streams
        """
        if self.interleave_pattern == "alternating":
            self._run_alternating(streams)
        elif self.interleave_pattern == "batched":
            self._run_batched(streams)
        else:
            # Default to alternating
            self._run_alternating(streams)

    def _run_alternating(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Alternating pattern: dispatch large, then small, then large, etc.

        This pattern maximizes the chance of convoy effects if queue
        scheduling is poor.
        """
        large_streams = streams[: self._num_large_streams]
        small_streams = streams[self._num_large_streams :]

        if not small_streams:
            small_streams = large_streams  # Fall back if only one stream type

        small_idx = 0
        num_small_per_large = self.large_to_small_ratio

        # Interleave: for each large stream, dispatch GEMM then small kernels
        for i, large_stream in enumerate(large_streams):
            # Large GEMM on large stream
            with torch.cuda.stream(large_stream):
                a = self._large_tensors_a[i]
                b = self._large_tensors_b[i]
                # Use matmul to ensure GEMM kernel
                c = torch.mm(a, b)

            # Multiple small kernels on small streams
            for j in range(num_small_per_large):
                small_stream = small_streams[j % len(small_streams)]
                tensor_idx = small_idx % len(self._small_tensors)

                with torch.cuda.stream(small_stream):
                    t = self._small_tensors[tensor_idx]
                    # Fused elementwise operations (tiny kernel)
                    result = torch.add(torch.mul(t, 2.0), 1.0)

                small_idx += 1

    def _run_batched(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Batched pattern: all large GEMMs, then all small kernels.

        This pattern allows maximum overlap if queues are independent.
        """
        large_streams = streams[: self._num_large_streams]
        small_streams = streams[self._num_large_streams :]

        if not small_streams:
            small_streams = large_streams

        # First: all large GEMMs
        for i, large_stream in enumerate(large_streams):
            with torch.cuda.stream(large_stream):
                a = self._large_tensors_a[i]
                b = self._large_tensors_b[i]
                c = torch.mm(a, b)

        # Then: all small kernels
        for i, t in enumerate(self._small_tensors):
            small_stream = small_streams[i % len(small_streams)]
            with torch.cuda.stream(small_stream):
                result = torch.add(torch.mul(t, 2.0), 1.0)

    def get_throughput_unit(self) -> str:
        return "GFLOPS"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        """Compute throughput in GFLOPS."""
        if total_time_sec <= 0:
            return 0.0
        total_flops = iterations * self._ops_per_iteration
        return total_flops / (total_time_sec * 1e9)  # GFLOPS

    def get_config(self) -> Dict[str, Any]:
        """Get workload configuration."""
        return {
            "name": self.name,
            "large_gemm_size": self.large_gemm_size,
            "small_kernel_size": self.small_kernel_size,
            "large_to_small_ratio": self.large_to_small_ratio,
            "interleave_pattern": self.interleave_pattern,
            "num_large_streams": self._num_large_streams,
            "num_small_streams": self._num_small_streams,
            "stream_count": self._stream_count,
            "device": self._device,
        }

    def validate_correctness(
        self, baseline_result: Any, test_result: Any
    ) -> Tuple[bool, str]:
        """
        Validate correctness of kernel execution.

        For this synthetic workload, we verify that:
        1. Results are finite (no NaN/Inf)
        2. GEMM produces expected output shape
        """
        # Run a quick validation iteration
        if self._is_setup:
            # Check large GEMM result shape and finiteness
            for i in range(min(2, len(self._large_tensors_a))):
                a = self._large_tensors_a[i]
                b = self._large_tensors_b[i]
                c = torch.mm(a, b)

                if not torch.isfinite(c).all():
                    return False, f"GEMM {i} produced non-finite values"

                expected_shape = (self.large_gemm_size[0], self.large_gemm_size[1])
                if c.shape != expected_shape:
                    return False, f"GEMM {i} shape mismatch: {c.shape} vs {expected_shape}"

            # Check small tensor ops
            for i in range(min(2, len(self._small_tensors))):
                t = self._small_tensors[i]
                result = torch.add(torch.mul(t, 2.0), 1.0)

                if not torch.isfinite(result).all():
                    return False, f"Small op {i} produced non-finite values"

        return True, "Correctness validation passed"


@WorkloadRegistry.register
class TinyKernelStressWorkload(SyntheticWorkload):
    """
    Stress test with only tiny kernels across many streams.

    This tests the overhead of queue switching with minimal kernel time,
    making switch latency the dominant factor.
    """

    name = "tiny_kernel_stress"
    description = "Tiny kernels only - isolates queue switch overhead"
    category = "latency_sensitive"
    min_streams = 1
    max_streams = 64
    recommended_streams = 16
    switch_latency_sensitivity = "critical"
    memory_requirements_gb = 0.5

    def __init__(
        self,
        tensor_size: int = 256,
        ops_per_stream: int = 100,
    ):
        """
        Initialize tiny kernel stress workload.

        Args:
            tensor_size: Size of each tensor
            ops_per_stream: Operations per stream per iteration
        """
        super().__init__()
        self.tensor_size = tensor_size
        self.ops_per_stream = ops_per_stream
        self._per_stream_tensors: List[List[torch.Tensor]] = []

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """Setup tensors for each stream."""
        self._stream_count = stream_count
        self._device = device
        self._is_setup = True

        self._per_stream_tensors = []
        for stream_idx in range(stream_count):
            stream_tensors = []
            for op_idx in range(self.ops_per_stream):
                t = torch.randn(self.tensor_size, dtype=torch.float32, device=device)
                stream_tensors.append(t)
                self._tensors[f"s{stream_idx}_op{op_idx}"] = t
            self._per_stream_tensors.append(stream_tensors)

        # FLOPs: ~5 per element per op (mul, add, abs, clamp, add)
        self._ops_per_iteration = (
            5 * self.tensor_size * self.ops_per_stream * stream_count
        )

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """Execute tiny kernels across all streams."""
        for stream_idx, stream in enumerate(streams):
            tensors = self._per_stream_tensors[stream_idx]

            with torch.cuda.stream(stream):
                for t in tensors:
                    # Chain of tiny ops
                    result = torch.clamp(torch.abs(torch.add(torch.mul(t, 1.5), 0.5)), 0, 10)

    def get_throughput_unit(self) -> str:
        return "GFLOPS"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return (iterations * self._ops_per_iteration) / (total_time_sec * 1e9)


@WorkloadRegistry.register
class LargeGEMMOnlyWorkload(SyntheticWorkload):
    """
    Large GEMMs only - baseline for compute-bound behavior.

    This establishes baseline throughput without switch overhead concerns.
    """

    name = "large_gemm_only"
    description = "Large GEMMs only - compute-bound baseline"
    category = "latency_sensitive"
    min_streams = 1
    max_streams = 16
    recommended_streams = 4
    switch_latency_sensitivity = "low"
    memory_requirements_gb = 8.0

    def __init__(
        self,
        gemm_size: Tuple[int, int, int] = (8192, 8192, 8192),
    ):
        """
        Initialize large GEMM workload.

        Args:
            gemm_size: (M, N, K) dimensions
        """
        super().__init__()
        self.gemm_size = gemm_size
        self._matrices_a: List[torch.Tensor] = []
        self._matrices_b: List[torch.Tensor] = []

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """Setup GEMM matrices."""
        self._stream_count = stream_count
        self._device = device
        self._is_setup = True

        m, n, k = self.gemm_size

        self._matrices_a = []
        self._matrices_b = []

        for i in range(stream_count):
            a = torch.randn(m, k, dtype=torch.float32, device=device)
            b = torch.randn(k, n, dtype=torch.float32, device=device)
            self._matrices_a.append(a)
            self._matrices_b.append(b)
            self._tensors[f"a_{i}"] = a
            self._tensors[f"b_{i}"] = b

        # 2*M*N*K FLOPs per GEMM
        self._ops_per_iteration = 2 * m * n * k * stream_count

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """Execute large GEMMs on each stream."""
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                a = self._matrices_a[i]
                b = self._matrices_b[i]
                c = torch.mm(a, b)

    def get_throughput_unit(self) -> str:
        return "TFLOPS"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return (iterations * self._ops_per_iteration) / (total_time_sec * 1e12)
