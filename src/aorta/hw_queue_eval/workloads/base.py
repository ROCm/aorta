"""
Abstract base class for all workloads.

All workloads must implement this interface to be usable with the StreamHarness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class WorkloadInfo:
    """Metadata about a workload."""

    name: str
    description: str
    category: str  # "distributed", "inference", "pipeline", "latency_sensitive"
    min_streams: int
    max_streams: int
    recommended_streams: int
    switch_latency_sensitivity: str  # "low", "medium", "high", "critical"
    memory_requirements_gb: float = 0.0
    multi_gpu_capable: bool = False


class BaseWorkload(ABC):
    """
    Base class for all queue stress test workloads.

    Subclasses must implement:
    - setup(): Initialize workload state
    - run_iteration(): Execute one iteration
    - get_throughput_unit(): Return throughput unit string
    - compute_throughput(): Compute throughput from timing data

    Class attributes to set:
    - name: Short identifier for the workload
    - description: Human-readable description
    - min_streams: Minimum number of streams supported
    - max_streams: Maximum number of streams supported
    - recommended_streams: Optimal number of streams
    - switch_latency_sensitivity: How sensitive to queue switch latency
    """

    # Class attributes - override in subclasses
    name: str = "base"
    description: str = "Base workload - do not use directly"
    category: str = "base"
    min_streams: int = 1
    max_streams: int = 32
    recommended_streams: int = 4
    switch_latency_sensitivity: str = "medium"  # "low", "medium", "high", "critical"
    memory_requirements_gb: float = 1.0
    multi_gpu_capable: bool = False

    def __init__(self):
        """Initialize the workload."""
        self._device: str = "cuda:0"
        self._stream_count: int = 0
        self._is_setup: bool = False
        self._tensors: Dict[str, torch.Tensor] = {}

    @abstractmethod
    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """
        Initialize workload state, models, tensors.

        This method is called once before iterations begin. It should:
        - Allocate all necessary tensors
        - Initialize models (if any)
        - Setup any per-stream state

        Args:
            stream_count: Number of streams that will be used
            device: Target device (e.g., "cuda:0")
        """
        pass

    @abstractmethod
    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Execute one iteration of the workload across provided streams.

        This method should dispatch work to the provided streams.
        It should NOT synchronize streams - the harness handles that.

        Args:
            streams: List of CUDA/HIP streams to use
        """
        pass

    @abstractmethod
    def get_throughput_unit(self) -> str:
        """
        Return the unit for throughput measurement.

        Examples: "samples/sec", "tokens/sec", "GFLOPS", "ops/sec"

        Returns:
            Unit string for throughput
        """
        pass

    @abstractmethod
    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        """
        Compute throughput from iteration count and time.

        Args:
            iterations: Number of iterations completed
            total_time_sec: Total time in seconds

        Returns:
            Throughput value (in units returned by get_throughput_unit)
        """
        pass

    def cleanup(self) -> None:
        """
        Optional cleanup after workload completes.

        Override this to release resources, clear caches, etc.
        """
        self._tensors.clear()
        self._is_setup = False
        torch.cuda.empty_cache()

    def validate_correctness(
        self, baseline_result: Any, test_result: Any
    ) -> Tuple[bool, str]:
        """
        Compare results for numerical correctness.

        Override this to implement workload-specific correctness checks.

        Args:
            baseline_result: Result from baseline run
            test_result: Result from test run

        Returns:
            Tuple of (is_correct, message)
        """
        return True, "Correctness validation not implemented"

    def get_info(self) -> WorkloadInfo:
        """Get workload metadata."""
        return WorkloadInfo(
            name=self.name,
            description=self.description,
            category=self.category,
            min_streams=self.min_streams,
            max_streams=self.max_streams,
            recommended_streams=self.recommended_streams,
            switch_latency_sensitivity=self.switch_latency_sensitivity,
            memory_requirements_gb=self.memory_requirements_gb,
            multi_gpu_capable=self.multi_gpu_capable,
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Get workload configuration.

        Override to return workload-specific configuration.

        Returns:
            Dictionary with configuration parameters
        """
        return {
            "name": self.name,
            "stream_count": self._stream_count,
            "device": self._device,
        }

    def supports_stream_count(self, count: int) -> bool:
        """Check if workload supports the given stream count."""
        return self.min_streams <= count <= self.max_streams

    def _allocate_tensor(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """
        Allocate a tensor and track it for cleanup.

        Args:
            name: Identifier for the tensor
            shape: Tensor shape
            dtype: Data type
            requires_grad: Whether to track gradients

        Returns:
            Allocated tensor
        """
        tensor = torch.randn(
            shape, dtype=dtype, device=self._device, requires_grad=requires_grad
        )
        self._tensors[name] = tensor
        return tensor

    def _get_tensor(self, name: str) -> torch.Tensor:
        """Get a previously allocated tensor."""
        if name not in self._tensors:
            raise KeyError(f"Tensor '{name}' not found. Did you call setup()?")
        return self._tensors[name]


class SyntheticWorkload(BaseWorkload):
    """
    Base class for synthetic (non-model) workloads.

    Provides utilities for creating simple kernel patterns.
    """

    category = "synthetic"

    def __init__(self):
        super().__init__()
        self._ops_per_iteration: int = 0

    def get_throughput_unit(self) -> str:
        return "ops/sec"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return (iterations * self._ops_per_iteration) / total_time_sec


class ModelWorkload(BaseWorkload):
    """
    Base class for model-based workloads.

    Provides utilities for:
    - Model initialization
    - Batch processing
    - Gradient computation
    """

    category = "model"

    def __init__(self):
        super().__init__()
        self._model: Optional[torch.nn.Module] = None
        self._batch_size: int = 1
        self._samples_per_iteration: int = 0

    def get_throughput_unit(self) -> str:
        return "samples/sec"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return (iterations * self._samples_per_iteration) / total_time_sec

    def cleanup(self) -> None:
        """Cleanup model and tensors."""
        super().cleanup()
        self._model = None


class DistributedWorkload(BaseWorkload):
    """
    Base class for distributed training workloads.

    Provides utilities for:
    - Mock collective operations
    - Communication/compute overlap simulation
    """

    category = "distributed"
    multi_gpu_capable = True

    def __init__(self, simulate_collectives: bool = True):
        """
        Initialize distributed workload.

        Args:
            simulate_collectives: If True, mock collective operations with local copies.
                                 If False, use actual NCCL/RCCL operations.
        """
        super().__init__()
        self._simulate_collectives = simulate_collectives
        self._comm_buffers: Dict[str, torch.Tensor] = {}

    def _mock_all_reduce(
        self, tensor: torch.Tensor, stream: torch.cuda.Stream
    ) -> torch.Tensor:
        """
        Mock all-reduce operation.

        In simulation mode, this just copies the tensor.
        In real distributed mode, this would call NCCL/RCCL.

        Args:
            tensor: Tensor to reduce
            stream: Stream to use

        Returns:
            Reduced tensor
        """
        with torch.cuda.stream(stream):
            if self._simulate_collectives:
                # Simulate collective with a copy and small computation
                result = tensor.clone()
                result.mul_(1.0)  # Force kernel execution
            else:
                # Would use torch.distributed here
                result = tensor
        return result

    def _mock_all_gather(
        self,
        tensor: torch.Tensor,
        stream: torch.cuda.Stream,
        num_ranks: int = 8,
    ) -> torch.Tensor:
        """
        Mock all-gather operation.

        Args:
            tensor: Local tensor
            stream: Stream to use
            num_ranks: Number of ranks (for output size)

        Returns:
            Gathered tensor (simulated)
        """
        with torch.cuda.stream(stream):
            if self._simulate_collectives:
                # Simulate by repeating tensor
                result = tensor.repeat(num_ranks, *([1] * (tensor.dim() - 1)))
            else:
                result = tensor
        return result

    def cleanup(self) -> None:
        """Cleanup communication buffers."""
        super().cleanup()
        self._comm_buffers.clear()


class InferenceWorkload(BaseWorkload):
    """
    Base class for inference workloads.

    Provides utilities for:
    - Token generation patterns
    - KV cache management
    - Batch scheduling
    """

    category = "inference"

    def __init__(self):
        super().__init__()
        self._tokens_per_iteration: int = 0
        self._kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def get_throughput_unit(self) -> str:
        return "tokens/sec"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return (iterations * self._tokens_per_iteration) / total_time_sec

    def _init_kv_cache(
        self,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Initialize KV cache for transformer inference."""
        self._kv_cache = {}
        for layer_idx in range(num_layers):
            k_cache = torch.zeros(
                (batch_size, num_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=self._device,
            )
            v_cache = torch.zeros(
                (batch_size, num_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=self._device,
            )
            self._kv_cache[layer_idx] = (k_cache, v_cache)

    def cleanup(self) -> None:
        """Cleanup KV cache and other resources."""
        super().cleanup()
        self._kv_cache.clear()
