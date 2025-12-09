"""
Speculative Decoding Workload

Pattern: Draft model generates K tokens, main model verifies in parallel
- Draft stream(s): Small model forward passes
- Verify stream(s): Main model forward for verification
- Accept/reject stream: Token acceptance logic
- KV cache stream: Cache management

This has tight latency requirements - switch overhead directly impacts tokens/sec.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from aorta.hw_queue_eval.workloads.base import InferenceWorkload
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry


class SimpleLM(nn.Module):
    """Simple language model for speculative decoding simulation."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Forward pass returning logits."""
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


@WorkloadRegistry.register
class SpeculativeDecodeWorkload(InferenceWorkload):
    """
    Draft + verify speculative decoding simulation.

    Simulates the speculative decoding pattern:
    1. Draft model generates K tokens speculatively
    2. Main model verifies all K tokens in parallel
    3. Accept/reject logic determines how many tokens to keep
    4. KV cache is updated accordingly

    Stream assignment:
    - Streams 0-1: Draft model forward
    - Streams 2-3: Main model verification
    - Stream 4: Accept/reject computation
    - Stream 5: KV cache management
    """

    name = "speculative_decode"
    description = "Draft + verify speculative decoding"
    category = "inference"
    min_streams = 4
    max_streams = 12
    recommended_streams = 6
    switch_latency_sensitivity = "high"
    memory_requirements_gb = 4.0

    def __init__(
        self,
        draft_hidden_size: int = 256,
        draft_num_layers: int = 4,
        main_hidden_size: int = 1024,
        main_num_layers: int = 12,
        speculation_length: int = 4,
        batch_size: int = 1,
        vocab_size: int = 32000,
    ):
        """
        Initialize speculative decoding workload.

        Args:
            draft_hidden_size: Draft model hidden size
            draft_num_layers: Draft model layers
            main_hidden_size: Main model hidden size
            main_num_layers: Main model layers
            speculation_length: Number of tokens to speculate
            batch_size: Batch size
            vocab_size: Vocabulary size
        """
        super().__init__()

        self.draft_hidden_size = draft_hidden_size
        self.draft_num_layers = draft_num_layers
        self.main_hidden_size = main_hidden_size
        self.main_num_layers = main_num_layers
        self.speculation_length = speculation_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size

        self._draft_model: Optional[SimpleLM] = None
        self._main_model: Optional[SimpleLM] = None
        self._input_ids: Optional[torch.Tensor] = None

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """Setup draft and main models."""
        self._stream_count = stream_count
        self._device = device
        self._is_setup = True

        # Create models
        self._draft_model = SimpleLM(
            vocab_size=self.vocab_size,
            hidden_size=self.draft_hidden_size,
            num_layers=self.draft_num_layers,
            num_heads=4,
        ).to(device)
        self._draft_model.eval()

        self._main_model = SimpleLM(
            vocab_size=self.vocab_size,
            hidden_size=self.main_hidden_size,
            num_layers=self.main_num_layers,
            num_heads=8,
        ).to(device)
        self._main_model.eval()

        # Stream assignments
        sixth = max(1, stream_count // 6)
        self._draft_streams = list(range(0, min(2, stream_count)))
        self._verify_streams = list(range(2, min(4, stream_count)))
        self._accept_stream = min(4, stream_count - 1)
        self._cache_stream = min(5, stream_count - 1)

        if not self._verify_streams:
            self._verify_streams = [0]

        # Input: some context tokens
        context_len = 128
        self._input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, context_len),
            dtype=torch.long, device=device
        )
        self._tensors["input_ids"] = self._input_ids

        # Buffers for speculated tokens
        self._draft_tokens = torch.zeros(
            self.batch_size, self.speculation_length,
            dtype=torch.long, device=device
        )
        self._tensors["draft_tokens"] = self._draft_tokens

        # Tokens per iteration = speculation_length (on average, accept rate ~70%)
        self._tokens_per_iteration = int(self.speculation_length * 0.7 * self.batch_size)

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Execute one speculative decoding iteration.

        1. Draft model generates K tokens
        2. Main model verifies all K in parallel
        3. Accept/reject determines accepted tokens
        """
        draft_stream = streams[self._draft_streams[0]]
        verify_stream = streams[self._verify_streams[0]]
        accept_stream = streams[self._accept_stream]
        cache_stream = streams[self._cache_stream]

        context = self._input_ids

        # Step 1: Draft model generates K tokens autoregressively
        with torch.cuda.stream(draft_stream):
            draft_input = context
            draft_tokens = []

            for k in range(self.speculation_length):
                # Get logits for last position
                logits = self._draft_model(draft_input)[:, -1, :]
                # Greedy sampling
                next_token = logits.argmax(dim=-1, keepdim=True)
                draft_tokens.append(next_token)
                # Append for next iteration
                draft_input = torch.cat([draft_input, next_token], dim=1)

            self._draft_tokens = torch.cat(draft_tokens, dim=1)

        # Step 2: Main model verifies all K tokens in parallel
        verify_stream.wait_stream(draft_stream)

        with torch.cuda.stream(verify_stream):
            # Verify all draft tokens in one forward pass
            verify_input = torch.cat([context, self._draft_tokens], dim=1)
            main_logits = self._main_model(verify_input)

            # Get logits at draft positions
            draft_start = context.size(1)
            verify_logits = main_logits[:, draft_start - 1:-1, :]

        # Step 3: Accept/reject logic
        accept_stream.wait_stream(verify_stream)

        with torch.cuda.stream(accept_stream):
            # Simplified accept/reject: compare main vs draft predictions
            main_preds = verify_logits.argmax(dim=-1)

            # Find first mismatch
            matches = (main_preds == self._draft_tokens)
            # Count accepted tokens (all matching up to first mismatch)
            # For simplicity, just compute the mask
            accept_mask = matches.cumprod(dim=1)

        # Step 4: Update KV cache (simulated)
        cache_stream.wait_stream(accept_stream)

        with torch.cuda.stream(cache_stream):
            # In real implementation, this would update KV cache
            # Here we just do some computation to simulate cache ops
            cache_update = self._draft_tokens * accept_mask.long()

    def get_throughput_unit(self) -> str:
        return "tokens/sec"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return (iterations * self._tokens_per_iteration) / total_time_sec

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "draft_hidden_size": self.draft_hidden_size,
            "draft_num_layers": self.draft_num_layers,
            "main_hidden_size": self.main_hidden_size,
            "main_num_layers": self.main_num_layers,
            "speculation_length": self.speculation_length,
            "batch_size": self.batch_size,
            "vocab_size": self.vocab_size,
            "stream_assignment": {
                "draft": self._draft_streams,
                "verify": self._verify_streams,
                "accept": self._accept_stream,
                "cache": self._cache_stream,
            },
        }

    def cleanup(self) -> None:
        """Cleanup models."""
        super().cleanup()
        self._draft_model = None
        self._main_model = None
