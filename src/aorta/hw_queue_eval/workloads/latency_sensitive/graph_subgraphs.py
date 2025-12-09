"""
Independent Subgraph Execution Workload

Pattern: Multiple independent computation subgraphs
- Each subgraph can execute on its own stream
- No dependencies between subgraphs within an iteration
- Final aggregation combines results

This tests maximum parallel execution when there are no dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from aorta.hw_queue_eval.workloads.base import BaseWorkload
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry


class Subgraph(nn.Module):
    """A self-contained computation subgraph."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        super().__init__()
        layers = []
        prev_size = input_size
        for i in range(num_layers):
            next_size = hidden_size if i < num_layers - 1 else output_size
            layers.append(nn.Linear(prev_size, next_size))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
            prev_size = next_size
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@WorkloadRegistry.register
class GraphSubgraphsWorkload(BaseWorkload):
    """
    Independent subgraph execution pattern.

    Simulates a computation graph that splits into independent
    branches, each of which can execute in parallel:

    Input -> Split -> [Subgraph 1]
                  -> [Subgraph 2]
                  -> [Subgraph 3]
                  -> ...
                  -> [Subgraph N] -> Aggregate -> Output

    Each subgraph is assigned to its own stream for maximum
    parallelism. This tests the hardware's ability to
    execute truly independent work concurrently.

    Stream assignment:
    - Stream 0: Input/split
    - Streams 1 to N-1: One per subgraph
    - Stream N: Aggregation
    """

    name = "graph_subgraphs"
    description = "Independent subgraph parallel execution"
    category = "latency_sensitive"
    min_streams = 4
    max_streams = 32
    recommended_streams = 8
    switch_latency_sensitivity = "high"
    memory_requirements_gb = 2.0

    def __init__(
        self,
        num_subgraphs: int = 8,
        input_size: int = 1024,
        hidden_size: int = 2048,
        output_size: int = 512,
        batch_size: int = 32,
        subgraph_layers: int = 3,
    ):
        """
        Initialize subgraph workload.

        Args:
            num_subgraphs: Number of independent subgraphs
            input_size: Input dimension
            hidden_size: Hidden dimension in subgraphs
            output_size: Output dimension per subgraph
            batch_size: Batch size
            subgraph_layers: Number of layers per subgraph
        """
        super().__init__()
        self.num_subgraphs = num_subgraphs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.subgraph_layers = subgraph_layers

        self._subgraphs: List[Subgraph] = []
        self._aggregator: nn.Module = None
        self._subgraph_outputs: List[torch.Tensor] = []

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """Setup subgraphs and stream assignments."""
        self._stream_count = stream_count
        self._device = device
        self._is_setup = True

        # Create subgraphs
        self._subgraphs = []
        for i in range(self.num_subgraphs):
            subgraph = Subgraph(
                self.input_size,
                self.hidden_size,
                self.output_size,
                self.subgraph_layers,
            ).to(device)
            subgraph.eval()
            self._subgraphs.append(subgraph)

        # Aggregation layer
        self._aggregator = nn.Linear(
            self.output_size * self.num_subgraphs, self.output_size
        ).to(device)
        self._aggregator.eval()

        # Stream assignments
        # Reserve first and last for input/aggregate
        available_streams = stream_count - 2
        self._input_stream = 0
        self._subgraph_streams = []

        for i in range(self.num_subgraphs):
            # Round-robin assign subgraphs to available streams
            stream_idx = 1 + (i % max(1, available_streams))
            self._subgraph_streams.append(stream_idx)

        self._aggregate_stream = stream_count - 1

        # Input tensor
        self._input = torch.randn(
            self.batch_size, self.input_size,
            dtype=torch.float32, device=device
        )
        self._tensors["input"] = self._input

        # Output buffers for each subgraph
        self._subgraph_outputs = []
        for i in range(self.num_subgraphs):
            buf = torch.empty(
                self.batch_size, self.output_size,
                dtype=torch.float32, device=device
            )
            self._subgraph_outputs.append(buf)
            self._tensors[f"subgraph_out_{i}"] = buf

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Execute all subgraphs in parallel, then aggregate.
        """
        input_stream = streams[self._input_stream]
        aggregate_stream = streams[self._aggregate_stream]

        # Input processing (could include splitting/routing)
        with torch.cuda.stream(input_stream):
            # Simple pass-through (in real use, might route different data)
            x = self._input

        # Execute all subgraphs in parallel
        for i, (subgraph, stream_idx) in enumerate(
            zip(self._subgraphs, self._subgraph_streams)
        ):
            subgraph_stream = streams[stream_idx]
            subgraph_stream.wait_stream(input_stream)

            with torch.cuda.stream(subgraph_stream):
                self._subgraph_outputs[i] = subgraph(x)

        # Aggregate: wait for all subgraphs
        for stream_idx in set(self._subgraph_streams):
            aggregate_stream.wait_stream(streams[stream_idx])

        with torch.cuda.stream(aggregate_stream):
            # Concatenate all outputs
            concatenated = torch.cat(self._subgraph_outputs, dim=-1)
            output = self._aggregator(concatenated)

    def get_throughput_unit(self) -> str:
        return "samples/sec"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return (iterations * self.batch_size) / total_time_sec

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_subgraphs": self.num_subgraphs,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "batch_size": self.batch_size,
            "subgraph_layers": self.subgraph_layers,
            "stream_assignment": {
                "input": self._input_stream,
                "subgraphs": self._subgraph_streams,
                "aggregate": self._aggregate_stream,
            },
        }

    def cleanup(self) -> None:
        """Cleanup subgraphs."""
        super().cleanup()
        self._subgraphs = []
        self._aggregator = None
        self._subgraph_outputs = []
