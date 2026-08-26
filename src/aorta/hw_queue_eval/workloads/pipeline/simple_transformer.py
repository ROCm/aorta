"""
Simple Transformer Training Workload

Pattern: GPT-2-style decoder-only transformer with multi-stream pipelined training.
- Layers are split into groups, each assigned to a different stream
- Forward pass is pipelined: stream K+1 waits on stream K
- Loss + backward run on a dedicated stream
- Optimizer step overlaps with next iteration's data prep

This creates a chain of inter-stream dependencies that exercises hardware
queue switching under a realistic training workload.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from aorta.hw_queue_eval.workloads.base import ModelWorkload, MultiGPUMixin
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry


class SimpleTransformerModel(nn.Module):
    """Small GPT-2-style decoder-only transformer for single-GPU training."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_pos = nn.Embedding(max_seq_len, hidden_size)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_layers(
        self, x: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        """Run a subset of transformer layers (for pipelined execution)."""
        for layer in self.layers[start:end]:
            x = layer(x)
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.embed_tokens(input_ids) + self.embed_pos(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)


@WorkloadRegistry.register
class SimpleTransformerWorkload(MultiGPUMixin, ModelWorkload):
    """
    Single-GPU transformer training with multi-stream pipelining.

    Splits the model's layers into groups and pipelines the forward pass
    across CUDA streams.  After the final layer group, loss and backward
    run on a dedicated stream, then the optimizer step is issued on the
    first stream so it can overlap with the next iteration's data prep.

    Stream assignment (for stream_count=4 as example):
      - Stream 0: data prep / optimizer step
      - Stream 1: forward layers 0-1
      - Stream 2: forward layers 2-3
      - Stream 3: forward layers 4-5, loss, backward
    """

    name = "simple_transformer"
    description = "Simple transformer training with pipelined forward pass"
    category = "pipeline"
    min_streams = 2
    max_streams = 16
    recommended_streams = 4
    switch_latency_sensitivity = "medium"
    memory_requirements_gb = 2.0
    multi_gpu_capable = True

    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        batch_size: int = 8,
        seq_length: int = 128,
        vocab_size: int = 32000,
        learning_rate: float = 1e-3,
        use_multi_gpu: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self._batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.use_multi_gpu = use_multi_gpu

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._input_ids: Optional[torch.Tensor] = None
        self._labels: Optional[torch.Tensor] = None
        self._loss_fn: Optional[nn.Module] = None

        self._layer_groups: List[tuple] = []
        self._fwd_streams: List[int] = []
        self._data_stream: int = 0
        self._loss_stream: int = 0
        self._devices: List[str] = []
        self._stream_to_device: Dict[int, str] = {}

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        self._stream_count = stream_count
        self._is_setup = True

        self._setup_multi_gpu(stream_count, device, self.use_multi_gpu)

        primary_device = self._get_device_for_stream(0)

        self._model = SimpleTransformerModel(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            max_seq_len=self.seq_length,
        ).to(primary_device)
        self._model.train()

        self._optimizer = torch.optim.SGD(
            self._model.parameters(), lr=self.learning_rate
        )
        self._loss_fn = nn.CrossEntropyLoss()

        self._input_ids = torch.randint(
            0, self.vocab_size,
            (self._batch_size, self.seq_length),
            device=primary_device,
        )
        self._labels = torch.randint(
            0, self.vocab_size,
            (self._batch_size, self.seq_length),
            device=primary_device,
        )
        self._tensors["input_ids"] = self._input_ids
        self._tensors["labels"] = self._labels

        self._samples_per_iteration = self._batch_size

        # Partition layers across available forward streams.
        # Reserve stream 0 for data prep / optimizer; the rest are forward.
        num_fwd_streams = max(1, stream_count - 1)
        self._data_stream = 0
        self._fwd_streams = list(range(1, 1 + num_fwd_streams))
        if not self._fwd_streams:
            self._fwd_streams = [0]
        self._loss_stream = self._fwd_streams[-1]

        layers_per_group = math.ceil(self.num_layers / num_fwd_streams)
        self._layer_groups = []
        for i in range(num_fwd_streams):
            start = i * layers_per_group
            end = min(start + layers_per_group, self.num_layers)
            if start < self.num_layers:
                self._layer_groups.append((start, end))

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        model = self._model
        data_stream = streams[self._data_stream]

        # --- Data prep on stream 0 ---
        with torch.cuda.stream(data_stream):
            self._input_ids.random_(0, self.vocab_size)
            self._labels.random_(0, self.vocab_size)

        # --- Pipelined forward pass across streams ---
        seq_len = self._input_ids.size(1)
        positions = torch.arange(seq_len, device=self._input_ids.device)

        prev_stream = data_stream
        hidden = None

        for group_idx, (layer_start, layer_end) in enumerate(self._layer_groups):
            fwd_idx = self._fwd_streams[group_idx % len(self._fwd_streams)]
            fwd_stream = streams[fwd_idx]

            fwd_stream.wait_stream(prev_stream)

            with torch.cuda.stream(fwd_stream):
                if group_idx == 0:
                    hidden = (
                        model.embed_tokens(self._input_ids)
                        + model.embed_pos(positions)
                    )
                hidden = model.forward_layers(hidden, layer_start, layer_end)

            prev_stream = fwd_stream

        # --- Loss + backward on the last forward stream ---
        loss_stream = streams[self._loss_stream]
        loss_stream.wait_stream(prev_stream)

        with torch.cuda.stream(loss_stream):
            hidden = model.ln_f(hidden)
            logits = model.lm_head(hidden)
            loss = self._loss_fn(
                logits.view(-1, self.vocab_size), self._labels.view(-1)
            )
            loss.backward()

        # --- Optimizer step on stream 0, overlapping with next data prep ---
        data_stream.wait_stream(loss_stream)

        with torch.cuda.stream(data_stream):
            self._optimizer.step()
            self._optimizer.zero_grad()

    def get_config(self) -> Dict[str, Any]:
        config = {
            "name": self.name,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "batch_size": self._batch_size,
            "seq_length": self.seq_length,
            "vocab_size": self.vocab_size,
            "learning_rate": self.learning_rate,
            "layer_groups": self._layer_groups,
            "stream_assignment": {
                "data_optimizer": self._data_stream,
                "forward": self._fwd_streams,
                "loss_backward": self._loss_stream,
            },
        }
        config.update(self._get_multi_gpu_config())
        return config

    def cleanup(self) -> None:
        super().cleanup()
        self._optimizer = None
        self._loss_fn = None
        self._input_ids = None
        self._labels = None
