"""
ZeRO-Offload Workload

Pattern: Overlap CPU offload/prefetch with GPU computation
- Offload streams: GPU to CPU transfers (optimizer states)
- Prefetch streams: CPU to GPU transfers (parameters)
- Compute streams: Forward/backward computation

This tests memory-bounded training scenarios with CPU offloading.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from aorta.hw_queue_eval.workloads.base import BaseWorkload
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry


class LargeLinearBlock(nn.Module):
    """Large linear block for memory-heavy computation."""

    def __init__(self, size: int):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@WorkloadRegistry.register
class ZeROOffloadWorkload(BaseWorkload):
    """
    ZeRO-Offload memory management patterns.

    Simulates the pattern where:
    1. Optimizer states are offloaded to CPU after update
    2. Parameters are prefetched to GPU before computation
    3. Gradients are computed while next layer prefetches

    Stream assignment:
    - Streams 0-1: Offload (GPU->CPU)
    - Streams 2-3: Prefetch (CPU->GPU)
    - Streams 4+: Compute
    """

    name = "zero_offload"
    description = "ZeRO-offload memory management patterns"
    category = "pipeline"
    min_streams = 4
    max_streams = 12
    recommended_streams = 6
    switch_latency_sensitivity = "medium"
    memory_requirements_gb = 8.0

    def __init__(
        self,
        num_layers: int = 4,
        layer_size: int = 4096,
        batch_size: int = 16,
    ):
        """
        Initialize ZeRO-Offload workload.

        Args:
            num_layers: Number of layers
            layer_size: Size of each layer
            batch_size: Batch size
        """
        super().__init__()
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.batch_size = batch_size

        self._layers: List[LargeLinearBlock] = []
        self._cpu_params: List[torch.Tensor] = []
        self._gpu_params: List[torch.Tensor] = []
        self._cpu_opt_states: List[torch.Tensor] = []

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """Setup layers and offload buffers."""
        self._stream_count = stream_count
        self._device = device
        self._is_setup = True

        # Stream assignments
        third = max(1, stream_count // 3)
        self._offload_streams = list(range(0, third))
        self._prefetch_streams = list(range(third, 2 * third))
        self._compute_streams = list(range(2 * third, stream_count))

        if not self._prefetch_streams:
            self._prefetch_streams = [0]
        if not self._compute_streams:
            self._compute_streams = [0]

        # Create layers (on GPU)
        self._layers = []
        for i in range(self.num_layers):
            layer = LargeLinearBlock(self.layer_size).to(device)
            self._layers.append(layer)

        # CPU parameter storage (pinned memory)
        self._cpu_params = []
        for i in range(self.num_layers):
            param = torch.randn(
                self.layer_size, self.layer_size, dtype=torch.float32
            ).pin_memory()
            self._cpu_params.append(param)

        # GPU parameter buffers
        self._gpu_params = []
        for i in range(self.num_layers):
            buf = torch.empty(
                self.layer_size, self.layer_size,
                dtype=torch.float32, device=device
            )
            self._gpu_params.append(buf)
            self._tensors[f"gpu_param_{i}"] = buf

        # CPU optimizer states (momentum, etc.)
        self._cpu_opt_states = []
        for i in range(self.num_layers):
            state = torch.zeros(
                self.layer_size, self.layer_size, dtype=torch.float32
            ).pin_memory()
            self._cpu_opt_states.append(state)

        # Input tensor
        self._input = torch.randn(
            self.batch_size, self.layer_size,
            dtype=torch.float32, device=device, requires_grad=True
        )
        self._tensors["input"] = self._input

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Execute one iteration with offload/prefetch pattern.

        Pattern:
        1. Prefetch layer N parameters while computing layer N-1
        2. Offload layer N-1 states after backward
        """
        offload_stream = streams[self._offload_streams[0]]
        prefetch_stream = streams[self._prefetch_streams[0]]
        compute_stream = streams[self._compute_streams[0]]

        x = self._input

        # Forward pass with prefetching
        for i in range(self.num_layers):
            # Prefetch current layer parameters
            with torch.cuda.stream(prefetch_stream):
                self._gpu_params[i].copy_(self._cpu_params[i], non_blocking=True)

            # Wait for prefetch then compute
            compute_stream.wait_stream(prefetch_stream)

            with torch.cuda.stream(compute_stream):
                # Update layer weights from prefetched buffer
                # (In real ZeRO, this would be more integrated)
                x = self._layers[i](x)

        # Backward pass with offloading
        with torch.cuda.stream(compute_stream):
            loss = x.sum()
            loss.backward()

        # Offload optimizer states
        for i in range(self.num_layers):
            # Wait for backward to complete for this layer
            offload_stream.wait_stream(compute_stream)

            with torch.cuda.stream(offload_stream):
                # Simulate offloading gradient/optimizer state
                for param in self._layers[i].parameters():
                    if param.grad is not None:
                        # Copy gradient to CPU (would update optimizer state)
                        grad_cpu = param.grad.cpu()
                        # Simulate optimizer step on CPU
                        self._cpu_opt_states[i].add_(grad_cpu[:self.layer_size, :self.layer_size] * 0.01)
                        param.grad = None

    def get_throughput_unit(self) -> str:
        return "samples/sec"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return (iterations * self.batch_size) / total_time_sec

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_layers": self.num_layers,
            "layer_size": self.layer_size,
            "batch_size": self.batch_size,
            "stream_assignment": {
                "offload": self._offload_streams,
                "prefetch": self._prefetch_streams,
                "compute": self._compute_streams,
            },
        }

    def cleanup(self) -> None:
        """Cleanup layers and buffers."""
        super().cleanup()
        self._layers = []
        self._cpu_params = []
        self._gpu_params = []
        self._cpu_opt_states = []
