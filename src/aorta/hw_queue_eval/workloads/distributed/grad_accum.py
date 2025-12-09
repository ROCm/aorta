"""
Gradient Accumulation with Early Reduction Workload

Pattern: Overlap gradient computation with reduction
- Compute streams for microbatch forward/backward
- Reduction streams for gradient accumulation
- Enables early all-reduce before full accumulation

This is common in large-batch training with gradient accumulation.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from aorta.hw_queue_eval.workloads.base import DistributedWorkload
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry


class SimpleMLP(nn.Module):
    """Simple MLP for gradient accumulation testing."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


@WorkloadRegistry.register
class GradientAccumulationWorkload(DistributedWorkload):
    """
    Gradient accumulation with early reduction pattern.

    Simulates the pattern where:
    - Multiple microbatches are processed
    - Gradients are accumulated across microbatches
    - All-reduce can start on early layers before all microbatches complete

    Stream assignment:
    - Streams 0-N/2: Compute (forward/backward)
    - Streams N/2-N: Reduction (all-reduce overlap)
    """

    name = "grad_accum"
    description = "Gradient accumulation with early reduction"
    category = "distributed"
    min_streams = 2
    max_streams = 16
    recommended_streams = 6
    switch_latency_sensitivity = "medium"
    memory_requirements_gb = 2.0

    def __init__(
        self,
        num_microbatches: int = 4,
        hidden_size: int = 2048,
        batch_size: int = 32,  # Per microbatch
        input_size: int = 1024,
        output_size: int = 1000,
        simulate_collectives: bool = True,
    ):
        """
        Initialize gradient accumulation workload.

        Args:
            num_microbatches: Number of microbatches to accumulate
            hidden_size: MLP hidden size
            batch_size: Batch size per microbatch
            input_size: Input feature size
            output_size: Output size
            simulate_collectives: Mock collective operations
        """
        super().__init__(simulate_collectives)

        self.num_microbatches = num_microbatches
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size

        self._model: nn.Module = None
        self._inputs: List[torch.Tensor] = []
        self._grad_buffers: Dict[str, torch.Tensor] = {}

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """Setup model and gradient buffers."""
        self._stream_count = stream_count
        self._device = device
        self._is_setup = True

        # Create model
        self._model = SimpleMLP(
            self.input_size, self.hidden_size, self.output_size
        ).to(device)

        # Stream assignments
        half = max(1, stream_count // 2)
        self._compute_streams = list(range(0, half))
        self._reduce_streams = list(range(half, stream_count))

        if not self._reduce_streams:
            self._reduce_streams = [0]

        # Create input tensors for each microbatch
        self._inputs = []
        for i in range(self.num_microbatches):
            inp = torch.randn(
                self.batch_size, self.input_size,
                dtype=torch.float32, device=device
            )
            self._inputs.append(inp)
            self._tensors[f"input_{i}"] = inp

        # Create gradient accumulation buffers
        self._grad_buffers = {}
        for name, param in self._model.named_parameters():
            buf = torch.zeros_like(param)
            self._grad_buffers[name] = buf

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Execute gradient accumulation with early reduction.

        Pattern:
        1. Process microbatches, accumulating gradients
        2. Start reducing early layers while later microbatches process
        """
        # Zero gradient buffers
        for buf in self._grad_buffers.values():
            buf.zero_()

        param_list = list(self._model.named_parameters())
        num_params = len(param_list)

        # Process microbatches
        for mb_idx in range(self.num_microbatches):
            compute_stream_idx = mb_idx % len(self._compute_streams)
            compute_stream = streams[self._compute_streams[compute_stream_idx]]

            with torch.cuda.stream(compute_stream):
                # Forward
                x = self._inputs[mb_idx]
                output = self._model(x)
                loss = output.sum() / self.num_microbatches

                # Backward
                loss.backward()

                # Accumulate gradients
                for name, param in self._model.named_parameters():
                    if param.grad is not None:
                        self._grad_buffers[name].add_(param.grad)
                        param.grad = None

            # Early reduction: reduce first half of params after half microbatches
            if mb_idx == self.num_microbatches // 2:
                reduce_stream = streams[self._reduce_streams[0]]
                reduce_stream.wait_stream(compute_stream)

                with torch.cuda.stream(reduce_stream):
                    for i, (name, _) in enumerate(param_list[:num_params // 2]):
                        buf = self._grad_buffers[name]
                        self._mock_all_reduce(buf, reduce_stream)

        # Final reduction for remaining params
        last_compute_stream = streams[self._compute_streams[-1]]
        reduce_stream = streams[self._reduce_streams[0]]
        reduce_stream.wait_stream(last_compute_stream)

        with torch.cuda.stream(reduce_stream):
            for name, _ in param_list[num_params // 2:]:
                buf = self._grad_buffers[name]
                self._mock_all_reduce(buf, reduce_stream)

    def get_throughput_unit(self) -> str:
        return "samples/sec"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        total_samples = iterations * self.num_microbatches * self.batch_size
        return total_samples / total_time_sec

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_microbatches": self.num_microbatches,
            "hidden_size": self.hidden_size,
            "batch_size": self.batch_size,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "stream_assignment": {
                "compute": self._compute_streams,
                "reduce": self._reduce_streams,
            },
        }

    def cleanup(self) -> None:
        """Cleanup model and buffers."""
        super().cleanup()
        self._model = None
        self._inputs = []
        self._grad_buffers = {}
