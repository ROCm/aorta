"""
Activation Checkpointing Workload

Pattern: Simulate activation checkpointing with recomputation streams
- Forward compute streams
- Recomputation streams during backward
- Gradient streams

This tests the overlap of recomputation with gradient computation.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from aorta.hw_queue_eval.workloads.base import DistributedWorkload
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry


class CheckpointBlock(nn.Module):
    """A block that can be checkpointed."""

    def __init__(self, hidden_size: int, expansion: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * expansion)
        self.fc2 = nn.Linear(hidden_size * expansion, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x + residual


@WorkloadRegistry.register
class ActivationCheckpointWorkload(DistributedWorkload):
    """
    Activation checkpointing with overlapped recomputation.

    Simulates the pattern where:
    - Forward pass saves only checkpoint boundaries
    - Backward recomputes activations on demand
    - Recomputation can overlap with gradient computation

    Stream assignment:
    - Streams 0-1: Forward compute
    - Streams 2-3: Recomputation during backward
    - Streams 4-5: Gradient computation
    """

    name = "activation_ckpt"
    description = "Activation checkpointing with recomputation overlap"
    category = "distributed"
    min_streams = 4
    max_streams = 12
    recommended_streams = 6
    switch_latency_sensitivity = "high"
    memory_requirements_gb = 2.0

    def __init__(
        self,
        num_blocks: int = 8,
        hidden_size: int = 1024,
        checkpoint_every: int = 2,  # Checkpoint every N blocks
        batch_size: int = 4,
        seq_length: int = 512,
        simulate_collectives: bool = True,
    ):
        """
        Initialize activation checkpointing workload.

        Args:
            num_blocks: Number of blocks in the model
            hidden_size: Hidden dimension
            checkpoint_every: Checkpoint frequency
            batch_size: Batch size
            seq_length: Sequence length
            simulate_collectives: Mock collective operations
        """
        super().__init__(simulate_collectives)

        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.checkpoint_every = checkpoint_every
        self.batch_size = batch_size
        self.seq_length = seq_length

        self._blocks: List[CheckpointBlock] = []
        self._checkpoints: Dict[int, torch.Tensor] = {}

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """Setup model blocks and stream assignments."""
        self._stream_count = stream_count
        self._device = device
        self._is_setup = True

        # Create blocks
        self._blocks = []
        for i in range(self.num_blocks):
            block = CheckpointBlock(self.hidden_size).to(device)
            block.eval()
            self._blocks.append(block)

        # Stream assignments
        third = max(1, stream_count // 3)
        self._forward_streams = list(range(0, third))
        self._recompute_streams = list(range(third, 2 * third))
        self._gradient_streams = list(range(2 * third, stream_count))

        # Ensure at least one stream per type
        if not self._recompute_streams:
            self._recompute_streams = [0]
        if not self._gradient_streams:
            self._gradient_streams = [0]

        # Input tensor
        self._input = torch.randn(
            self.batch_size, self.seq_length, self.hidden_size,
            dtype=torch.float32, device=device, requires_grad=True
        )
        self._tensors["input"] = self._input

        # Checkpoint storage
        self._checkpoints = {}

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Execute one iteration with checkpointing pattern.

        1. Forward pass: compute and save checkpoints
        2. Backward pass: recompute from checkpoints, compute gradients
        """
        forward_stream = streams[self._forward_streams[0]]
        recompute_stream = streams[self._recompute_streams[0]]
        gradient_stream = streams[self._gradient_streams[0]]

        x = self._input
        self._checkpoints = {0: x}

        # Forward pass with checkpointing
        with torch.cuda.stream(forward_stream):
            for i, block in enumerate(self._blocks):
                x = block(x)

                # Save checkpoint at intervals
                if (i + 1) % self.checkpoint_every == 0:
                    self._checkpoints[i + 1] = x.detach().clone()

            # Final output
            output = x.sum()

        # Simulate backward pass with recomputation
        # In real checkpointing, this happens automatically
        # Here we simulate the pattern

        # Wait for forward to complete
        recompute_stream.wait_stream(forward_stream)
        gradient_stream.wait_stream(forward_stream)

        # Backward: iterate blocks in reverse
        # Recompute intermediate activations from checkpoints
        num_segments = (self.num_blocks + self.checkpoint_every - 1) // self.checkpoint_every

        for seg in range(num_segments - 1, -1, -1):
            start_idx = seg * self.checkpoint_every
            end_idx = min(start_idx + self.checkpoint_every, self.num_blocks)

            # Get checkpoint for this segment
            ckpt_key = start_idx
            if ckpt_key in self._checkpoints:
                ckpt = self._checkpoints[ckpt_key]
            else:
                ckpt = self._input

            # Recompute activations for this segment
            with torch.cuda.stream(recompute_stream):
                recomputed = ckpt
                for i in range(start_idx, end_idx):
                    recomputed = self._blocks[i](recomputed)

            # Overlap: compute gradients while next segment recomputes
            recompute_stream.wait_stream(gradient_stream)

            with torch.cuda.stream(gradient_stream):
                # Simulate gradient computation
                if recomputed.requires_grad:
                    grad = torch.autograd.grad(
                        recomputed.sum(), ckpt, retain_graph=True,
                        allow_unused=True
                    )[0]

    def get_throughput_unit(self) -> str:
        return "samples/sec"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return (iterations * self.batch_size) / total_time_sec

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_blocks": self.num_blocks,
            "hidden_size": self.hidden_size,
            "checkpoint_every": self.checkpoint_every,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "stream_assignment": {
                "forward": self._forward_streams,
                "recompute": self._recompute_streams,
                "gradient": self._gradient_streams,
            },
        }

    def cleanup(self) -> None:
        """Cleanup model and checkpoints."""
        super().cleanup()
        self._blocks = []
        self._checkpoints = {}
