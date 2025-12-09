"""
FSDP + Tensor Parallelism (3D Parallelism) Workload

Pattern: Simulate overlapped communication and compute in distributed training
- All-reduce streams for data parallelism
- All-gather streams for FSDP shard gathering
- Compute streams for forward/backward
- Point-to-point streams for tensor parallelism

This can run in single-GPU simulation mode (mock collectives)
or multi-GPU mode with actual NCCL/RCCL operations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from aorta.hw_queue_eval.workloads.base import DistributedWorkload
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing."""

    def __init__(self, hidden_size: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_size)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        # MLP
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x

        return x


@WorkloadRegistry.register
class FSDPTPWorkload(DistributedWorkload):
    """
    FSDP + Tensor Parallelism with overlapped comm/compute.

    Simulates the stream usage pattern of 3D parallelism:
    - Compute streams: Model forward/backward
    - All-gather streams: FSDP shard gathering (prefetch)
    - All-reduce streams: Gradient synchronization
    - P2P streams: TP communication (if using TP)

    Stream assignment (for 10 streams):
    - Streams 0-1: Primary compute
    - Streams 2-3: All-gather (FSDP)
    - Streams 4-5: All-reduce (DP)
    - Streams 6-7: Point-to-point (TP)
    - Streams 8-9: Auxiliary/overlap
    """

    name = "fsdp_tp"
    description = "FSDP + Tensor Parallelism with overlapped comm/compute"
    category = "distributed"
    min_streams = 4
    max_streams = 16
    recommended_streams = 10
    switch_latency_sensitivity = "high"
    memory_requirements_gb = 4.0

    def __init__(
        self,
        model_size: str = "small",  # "small", "medium", "large"
        simulate_collectives: bool = True,
        overlap_comm_compute: bool = True,
        num_layers: int = 4,
        batch_size: int = 8,
        seq_length: int = 512,
    ):
        """
        Initialize FSDP+TP workload.

        Args:
            model_size: Model size preset
            simulate_collectives: Mock collectives for single-GPU testing
            overlap_comm_compute: Enable comm/compute overlap pattern
            num_layers: Number of transformer layers
            batch_size: Batch size
            seq_length: Sequence length
        """
        super().__init__(simulate_collectives)

        self.model_size = model_size
        self.overlap_comm_compute = overlap_comm_compute
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Size presets
        size_configs = {
            "small": {"hidden_size": 512, "num_heads": 8},
            "medium": {"hidden_size": 1024, "num_heads": 16},
            "large": {"hidden_size": 2048, "num_heads": 32},
        }
        config = size_configs.get(model_size, size_configs["small"])
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]

        # Stream indices (will be populated in setup)
        self._compute_streams: List[int] = []
        self._allgather_streams: List[int] = []
        self._allreduce_streams: List[int] = []
        self._p2p_streams: List[int] = []

        # Model layers
        self._layers: List[SimpleTransformerBlock] = []
        self._layer_weights: List[torch.Tensor] = []

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """Setup model layers and stream assignments."""
        self._stream_count = stream_count
        self._device = device
        self._is_setup = True

        # Assign streams to different roles
        # Compute gets priority (first 1/4)
        # Then all-gather (next 1/4)
        # Then all-reduce (next 1/4)
        # Then P2P/aux (remaining)
        quarter = max(1, stream_count // 4)

        self._compute_streams = list(range(0, quarter))
        self._allgather_streams = list(range(quarter, 2 * quarter))
        self._allreduce_streams = list(range(2 * quarter, 3 * quarter))
        self._p2p_streams = list(range(3 * quarter, stream_count))

        # Ensure at least one stream per type
        if not self._allgather_streams:
            self._allgather_streams = [0]
        if not self._allreduce_streams:
            self._allreduce_streams = [0]
        if not self._p2p_streams:
            self._p2p_streams = [0]

        # Create model layers
        self._layers = []
        self._layer_weights = []

        for i in range(self.num_layers):
            layer = SimpleTransformerBlock(
                self.hidden_size, self.num_heads
            ).to(device)
            layer.eval()  # No dropout for determinism
            self._layers.append(layer)

            # Create "shard" weights to gather (simulating FSDP)
            shard = torch.randn(
                self.hidden_size, self.hidden_size,
                dtype=torch.float32, device=device
            )
            self._layer_weights.append(shard)
            self._tensors[f"shard_{i}"] = shard

        # Input tensor
        self._input = torch.randn(
            self.batch_size, self.seq_length, self.hidden_size,
            dtype=torch.float32, device=device, requires_grad=True
        )
        self._tensors["input"] = self._input

        # Gradient buffer
        self._grad_buffer = torch.zeros(
            self.hidden_size * self.hidden_size,
            dtype=torch.float32, device=device
        )
        self._tensors["grad_buffer"] = self._grad_buffer

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Execute one training iteration with overlapped operations.

        Pattern:
        1. Prefetch all-gather for layer 0 weights
        2. For each layer:
           - Compute forward on compute stream
           - Prefetch next layer's weights on all-gather stream
        3. Backward pass with all-reduce overlap
        """
        if self.overlap_comm_compute:
            self._run_overlapped(streams)
        else:
            self._run_sequential(streams)

    def _run_overlapped(self, streams: List[torch.cuda.Stream]) -> None:
        """Run with comm/compute overlap."""
        compute_stream = streams[self._compute_streams[0]]
        allgather_stream = streams[self._allgather_streams[0]]
        allreduce_stream = streams[self._allreduce_streams[0]]

        x = self._input

        # Prefetch first layer weights
        gathered_weights = self._mock_all_gather(
            self._layer_weights[0], allgather_stream
        )

        # Forward pass with prefetching
        for i, layer in enumerate(self._layers):
            # Wait for this layer's weights (from previous prefetch)
            compute_stream.wait_stream(allgather_stream)

            # Compute forward
            with torch.cuda.stream(compute_stream):
                x = layer(x)

            # Prefetch next layer's weights (if not last layer)
            if i < len(self._layers) - 1:
                gathered_weights = self._mock_all_gather(
                    self._layer_weights[i + 1], allgather_stream
                )

        # Simulate backward pass with gradient computation
        with torch.cuda.stream(compute_stream):
            # Fake backward - just compute some gradients
            loss = x.sum()
            grad = torch.autograd.grad(loss, self._input, retain_graph=True)[0]

        # All-reduce gradients (overlapped with next batch's forward)
        with torch.cuda.stream(allreduce_stream):
            reduced_grad = self._mock_all_reduce(
                self._grad_buffer, allreduce_stream
            )

    def _run_sequential(self, streams: List[torch.cuda.Stream]) -> None:
        """Run without overlap (baseline)."""
        compute_stream = streams[self._compute_streams[0]]
        allreduce_stream = streams[self._allreduce_streams[0]]

        x = self._input

        # Forward pass
        with torch.cuda.stream(compute_stream):
            for layer in self._layers:
                x = layer(x)

        # Backward
        with torch.cuda.stream(compute_stream):
            loss = x.sum()
            grad = torch.autograd.grad(loss, self._input, retain_graph=True)[0]

        # All-reduce (after backward completes)
        compute_stream.synchronize()
        with torch.cuda.stream(allreduce_stream):
            reduced_grad = self._mock_all_reduce(
                self._grad_buffer, allreduce_stream
            )

    def get_throughput_unit(self) -> str:
        return "samples/sec"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return (iterations * self.batch_size) / total_time_sec

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_size": self.model_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "overlap_comm_compute": self.overlap_comm_compute,
            "simulate_collectives": self._simulate_collectives,
            "stream_assignment": {
                "compute": self._compute_streams,
                "allgather": self._allgather_streams,
                "allreduce": self._allreduce_streams,
                "p2p": self._p2p_streams,
            },
        }

    def cleanup(self) -> None:
        """Cleanup model and tensors."""
        super().cleanup()
        self._layers = []
        self._layer_weights = []
