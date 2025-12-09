"""
torch.compile Multi-Region Workload

Pattern: Multiple compiled regions with stream-based coordination
- Each compiled region may use internal streams
- External coordination across regions via explicit streams

This tests how compiled code interacts with manual stream management.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from aorta.hw_queue_eval.workloads.base import BaseWorkload
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry


class CompiledRegion(nn.Module):
    """A module that can be compiled with torch.compile."""

    def __init__(self, hidden_size: int, expansion: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * expansion)
        self.fc2 = nn.Linear(hidden_size * expansion, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = torch.gelu(self.fc1(x))
        x = self.fc2(x)
        return x + residual


@WorkloadRegistry.register
class TorchCompileWorkload(BaseWorkload):
    """
    Multi-region torch.compile execution.

    Simulates a model with multiple compiled regions that
    need to be coordinated across streams:

    1. Region A computes independently
    2. Region B depends on Region A output
    3. Region C runs in parallel with Region B
    4. Final merge depends on B and C

    Stream assignment:
    - Streams 0-1: Region A
    - Streams 2-3: Region B
    - Streams 4-5: Region C
    - Streams 6+: Merge/coordination
    """

    name = "torch_compile"
    description = "Multi-region compiled execution"
    category = "pipeline"
    min_streams = 4
    max_streams = 12
    recommended_streams = 8
    switch_latency_sensitivity = "medium"
    memory_requirements_gb = 2.0

    def __init__(
        self,
        hidden_size: int = 1024,
        batch_size: int = 16,
        seq_length: int = 256,
        use_compile: bool = False,  # Disabled by default (may not work in all envs)
        compile_mode: str = "reduce-overhead",  # "default", "reduce-overhead", "max-autotune"
    ):
        """
        Initialize torch.compile workload.

        Args:
            hidden_size: Hidden dimension
            batch_size: Batch size
            seq_length: Sequence length
            use_compile: Whether to use torch.compile
            compile_mode: Compilation mode
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_compile = use_compile
        self.compile_mode = compile_mode

        self._region_a: Optional[nn.Module] = None
        self._region_b: Optional[nn.Module] = None
        self._region_c: Optional[nn.Module] = None
        self._merge_layer: Optional[nn.Module] = None

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """Setup compiled regions."""
        self._stream_count = stream_count
        self._device = device
        self._is_setup = True

        # Stream assignments
        quarter = max(1, stream_count // 4)
        self._region_a_streams = list(range(0, quarter))
        self._region_b_streams = list(range(quarter, 2 * quarter))
        self._region_c_streams = list(range(2 * quarter, 3 * quarter))
        self._merge_streams = list(range(3 * quarter, stream_count))

        for stream_list in [self._region_b_streams, self._region_c_streams, self._merge_streams]:
            if not stream_list:
                stream_list.append(0)

        # Create regions
        self._region_a = CompiledRegion(self.hidden_size).to(device)
        self._region_b = CompiledRegion(self.hidden_size).to(device)
        self._region_c = CompiledRegion(self.hidden_size).to(device)
        self._merge_layer = nn.Linear(self.hidden_size * 2, self.hidden_size).to(device)

        # Optionally compile
        if self.use_compile:
            try:
                self._region_a = torch.compile(self._region_a, mode=self.compile_mode)
                self._region_b = torch.compile(self._region_b, mode=self.compile_mode)
                self._region_c = torch.compile(self._region_c, mode=self.compile_mode)
            except Exception:
                # Compilation may fail in some environments
                pass

        # Set to eval mode
        self._region_a.eval()
        self._region_b.eval()
        self._region_c.eval()
        self._merge_layer.eval()

        # Input tensor
        self._input = torch.randn(
            self.batch_size, self.seq_length, self.hidden_size,
            dtype=torch.float32, device=device
        )
        self._tensors["input"] = self._input

        # Intermediate buffers
        self._buf_a = torch.empty_like(self._input)
        self._buf_b = torch.empty_like(self._input)
        self._buf_c = torch.empty_like(self._input)
        self._tensors["buf_a"] = self._buf_a
        self._tensors["buf_b"] = self._buf_b
        self._tensors["buf_c"] = self._buf_c

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Execute multi-region computation.

        Graph:
        input -> Region A -> Region B --\\
                         \\-> Region C -> Merge -> output
        """
        stream_a = streams[self._region_a_streams[0]]
        stream_b = streams[self._region_b_streams[0]]
        stream_c = streams[self._region_c_streams[0]]
        stream_merge = streams[self._merge_streams[0]]

        # Region A: process input
        with torch.cuda.stream(stream_a):
            self._buf_a = self._region_a(self._input)

        # Region B: depends on A
        stream_b.wait_stream(stream_a)
        with torch.cuda.stream(stream_b):
            self._buf_b = self._region_b(self._buf_a)

        # Region C: parallel to B, depends on A
        stream_c.wait_stream(stream_a)
        with torch.cuda.stream(stream_c):
            self._buf_c = self._region_c(self._buf_a)

        # Merge: depends on B and C
        stream_merge.wait_stream(stream_b)
        stream_merge.wait_stream(stream_c)

        with torch.cuda.stream(stream_merge):
            # Concatenate B and C outputs
            merged = torch.cat([self._buf_b, self._buf_c], dim=-1)
            output = self._merge_layer(merged)

    def get_throughput_unit(self) -> str:
        return "samples/sec"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return (iterations * self.batch_size) / total_time_sec

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "hidden_size": self.hidden_size,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "use_compile": self.use_compile,
            "compile_mode": self.compile_mode,
            "stream_assignment": {
                "region_a": self._region_a_streams,
                "region_b": self._region_b_streams,
                "region_c": self._region_c_streams,
                "merge": self._merge_streams,
            },
        }

    def cleanup(self) -> None:
        """Cleanup regions."""
        super().cleanup()
        self._region_a = None
        self._region_b = None
        self._region_c = None
        self._merge_layer = None
