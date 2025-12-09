"""
Continuous Batching Workload

Pattern: Prefill and decode phases overlap for different requests
- Prefill streams: Process new requests (compute-heavy)
- Decode streams: Generate tokens for active requests (memory-bound)
- Scheduling stream: Batch management

This tests the ability to overlap compute-heavy and memory-bound operations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from aorta.hw_queue_eval.workloads.base import InferenceWorkload
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry


class AttentionBlock(nn.Module):
    """Single attention block for inference."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)

        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Scaled dot-product attention
        q = q.transpose(1, 2)  # B, H, S, D
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, S, self.hidden_size)
        out = self.proj(out)

        return residual + out


@WorkloadRegistry.register
class ContinuousBatchWorkload(InferenceWorkload):
    """
    Continuous batching with prefill/decode overlap.

    Simulates a serving system where:
    - New requests are prefilled (compute-heavy, long sequences)
    - Active requests decode one token at a time (memory-bound)
    - These phases overlap to maximize throughput

    Stream assignment:
    - Streams 0-2: Prefill (new requests)
    - Streams 3-5: Decode (active requests)
    - Streams 6-7: KV cache management
    - Streams 8-9: Batch scheduling
    """

    name = "continuous_batch"
    description = "Prefill/decode overlap in continuous batching"
    category = "inference"
    min_streams = 4
    max_streams = 16
    recommended_streams = 8
    switch_latency_sensitivity = "high"
    memory_requirements_gb = 6.0

    def __init__(
        self,
        hidden_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 8,
        prefill_batch_size: int = 2,
        prefill_seq_length: int = 512,
        decode_batch_size: int = 16,
        max_seq_length: int = 2048,
    ):
        """
        Initialize continuous batching workload.

        Args:
            hidden_size: Model hidden size
            num_layers: Number of attention layers
            num_heads: Number of attention heads
            prefill_batch_size: Batch size for prefill
            prefill_seq_length: Sequence length for prefill
            decode_batch_size: Batch size for decode
            max_seq_length: Maximum sequence length for KV cache
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefill_batch_size = prefill_batch_size
        self.prefill_seq_length = prefill_seq_length
        self.decode_batch_size = decode_batch_size
        self.max_seq_length = max_seq_length

        self._layers: List[AttentionBlock] = []

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """Setup model and stream assignments."""
        self._stream_count = stream_count
        self._device = device
        self._is_setup = True

        # Create attention layers
        self._layers = []
        for _ in range(self.num_layers):
            layer = AttentionBlock(self.hidden_size, self.num_heads).to(device)
            layer.eval()
            self._layers.append(layer)

        # Stream assignments
        third = max(1, stream_count // 3)
        self._prefill_streams = list(range(0, third))
        self._decode_streams = list(range(third, 2 * third))
        self._cache_streams = list(range(2 * third, stream_count))

        if not self._decode_streams:
            self._decode_streams = [0]
        if not self._cache_streams:
            self._cache_streams = [0]

        # Prefill input
        self._prefill_input = torch.randn(
            self.prefill_batch_size, self.prefill_seq_length, self.hidden_size,
            dtype=torch.float32, device=device
        )
        self._tensors["prefill_input"] = self._prefill_input

        # Decode input (single token per request)
        self._decode_input = torch.randn(
            self.decode_batch_size, 1, self.hidden_size,
            dtype=torch.float32, device=device
        )
        self._tensors["decode_input"] = self._decode_input

        # KV cache (simplified)
        self._init_kv_cache(
            num_layers=self.num_layers,
            batch_size=self.decode_batch_size + self.prefill_batch_size,
            max_seq_len=self.max_seq_length,
            num_heads=self.num_heads,
            head_dim=self.hidden_size // self.num_heads,
        )

        # Tokens per iteration: prefill + decode
        self._tokens_per_iteration = (
            self.prefill_batch_size * self.prefill_seq_length +
            self.decode_batch_size
        )

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Execute one iteration with overlapped prefill and decode.

        Prefill and decode run in parallel on different streams.
        """
        prefill_stream = streams[self._prefill_streams[0]]
        decode_stream = streams[self._decode_streams[0]]
        cache_stream = streams[self._cache_streams[0]]

        # Start prefill (compute-heavy)
        with torch.cuda.stream(prefill_stream):
            prefill_out = self._prefill_input
            for layer in self._layers:
                prefill_out = layer(prefill_out)

        # Decode in parallel (memory-bound)
        with torch.cuda.stream(decode_stream):
            decode_out = self._decode_input
            for layer in self._layers:
                decode_out = layer(decode_out)

        # KV cache update (overlapped)
        # Wait for both to complete for cache coherence
        cache_stream.wait_stream(prefill_stream)
        cache_stream.wait_stream(decode_stream)

        with torch.cuda.stream(cache_stream):
            # Simulate cache operations
            for layer_idx in range(min(2, self.num_layers)):
                k_cache, v_cache = self._kv_cache[layer_idx]
                # Small update operation
                k_cache[:, :, 0, :] = k_cache[:, :, 0, :] * 0.99 + 0.01

    def get_throughput_unit(self) -> str:
        return "tokens/sec"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return (iterations * self._tokens_per_iteration) / total_time_sec

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "prefill_batch_size": self.prefill_batch_size,
            "prefill_seq_length": self.prefill_seq_length,
            "decode_batch_size": self.decode_batch_size,
            "max_seq_length": self.max_seq_length,
            "stream_assignment": {
                "prefill": self._prefill_streams,
                "decode": self._decode_streams,
                "cache": self._cache_streams,
            },
        }

    def cleanup(self) -> None:
        """Cleanup model and cache."""
        super().cleanup()
        self._layers = []
