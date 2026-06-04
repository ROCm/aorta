"""
FSDP mode reproducer (Fully Sharded Data Parallel pattern).

This mode simulates an FSDP-style workload with:
- H2D transfer for batch data (single- or double-buffered via --prefetch)
- Per-layer all_gather to reconstruct full parameters before compute
- Per-layer reduce_scatter to shard gradients after backward compute
- GEMMs interleaved with collectives (if compute enabled)

Unlike default (TorchRec) and DDP modes which use bulk collectives,
FSDP interleaves many small all_gather/reduce_scatter operations with
per-layer compute. This creates a fundamentally different overlap and
timing profile that may trigger different runtime bugs.

All FSDP collectives run on the default stream (no separate comm stream).
Overlap comes from NCCL internal pipelining and H2D on memcpy_stream.

Data Flow:
    memcpy_stream:   [H2D] ──────────────────────────────────────────────────┐
                                                                              │ wait
    default_stream:  [all_gather L0 → GEMM L0 → all_gather L1 → GEMM L1 ...]│
                     [... → GEMM bwd L1 → reduce_scatter L1 →                │
                            GEMM bwd L0 → reduce_scatter L0]                 │
                     [optimizer step]
"""

import logging
from typing import List, Optional

import torch
import torch.distributed as dist

from ..base import BaseReproducer
from ..config import ReproducerConfig

log = logging.getLogger(__name__)


class FSDPModeReproducer(BaseReproducer):
    """
    FSDP reproducer with per-layer all_gather + reduce_scatter.

    This mode tests the communication pattern where many small collectives
    are interleaved with compute, matching real FSDP training:
    - Forward: all_gather per layer → GEMM → (free full param)
    - Backward: all_gather per layer → GEMM backward → reduce_scatter

    H2D strategy is controlled by config.h2d_prefetch (base class).

    Verification checks:
    - H2D: batch_gpu == iteration % 1000
    - all_gather: after gathering rank-filled shards, chunk j == float(j)
    - reduce_scatter: after scattering rank-filled grads, output == sum(1..world_size)
    """

    def __init__(self, config: ReproducerConfig, rank: int, world_size: int):
        super().__init__(config, rank, world_size)

        if config.compute_type == "transformer":
            self.num_layers: int = config.num_layers
            self._dim: int = config.model_dim
        else:
            self.num_layers: int = config.gemm_layers
            self._dim: int = config.gemm_size
        self.shard_size: int = config.fsdp_shard_size

        # Per-layer parameter shards (each rank holds 1/world_size)
        self.param_shards: List[torch.Tensor] = []

        # Reusable buffers (shared across layers, like real FSDP)
        self.full_param: Optional[torch.Tensor] = None   # all_gather output
        self.full_grad: Optional[torch.Tensor] = None     # reduce_scatter input
        self.grad_shard: Optional[torch.Tensor] = None    # reduce_scatter output

        # Per-layer GEMM weights (only when compute is enabled)
        self.weight_matrices: List[torch.Tensor] = []
        self.activation: Optional[torch.Tensor] = None
        self.grad_buffer: Optional[torch.Tensor] = None

        # Shared-weight transformer: fixed reference input + per-layer checksums
        self.reference_input: Optional[torch.Tensor] = None
        self.layer_checksums: List[Optional[dict]] = []

    def _setup_compute(self) -> None:
        """
        Override base compute setup -- FSDP manages its own per-layer compute.

        FSDP interleaves collectives and compute per-layer, so it cannot use the
        base class's bulk compute simulator. Per-layer weights are allocated in
        setup_buffers() instead.

        Still validates h2d_tensor_size when compute is enabled.
        """
        if not self.config.simulate_compute:
            return

        # Validate buffer sizes based on compute type
        dim = self._dim
        min_h2d_size = dim * dim
        if self.config.h2d_tensor_size < min_h2d_size:
            log.warning(
                f"h2d_tensor_size ({self.config.h2d_tensor_size}) < {dim}² "
                f"({min_h2d_size}). Increasing to {min_h2d_size} for compute."
            )
            self.config.h2d_tensor_size = min_h2d_size

        # NOTE: We do NOT create a base compute simulator (self.compute stays None).
        # FSDP mode creates per-layer weight_matrices in setup_buffers() because
        # collectives and compute are interleaved per-layer.
        log.info("FSDP mode: per-layer compute managed internally (no base compute)")

    def setup_buffers(self) -> None:
        """Allocate FSDP-specific buffers: per-layer shards + reusable collective buffers."""
        cfg = self.config
        ws = self.world_size

        # Per-layer parameter shards (what each rank "owns")
        self.param_shards = [
            torch.empty(self.shard_size, dtype=self.dtype, device="cuda")
            for _ in range(self.num_layers)
        ]

        # Reusable all_gather output: full parameter = shard_size * world_size
        self.full_param = torch.empty(
            self.shard_size * ws, dtype=self.dtype, device="cuda"
        )

        # Reusable reduce_scatter buffers
        self.full_grad = torch.empty(
            self.shard_size * ws, dtype=self.dtype, device="cuda"
        )
        self.grad_shard = torch.empty(
            self.shard_size, dtype=self.dtype, device="cuda"
        )

        # Per-layer GEMM weights (only if compute simulation is enabled)
        # FSDP mode manages its own compute because collectives are interleaved
        # per-layer, unlike the base compute simulator which runs all layers at once.
        if cfg.simulate_compute:
            dim = self._dim
            use_shared = (
                cfg.shared_layer_weights and cfg.compute_type == "transformer"
            )
            if use_shared:
                # All layers share one weight matrix with a fixed seed so that
                # gelu(W @ reference_input) is identical for every layer.
                # Any divergence across layers indicates compute-path corruption.
                g = torch.Generator(device="cuda")
                g.manual_seed(0)
                shared_w = torch.randn(
                    dim, dim, dtype=self.dtype, device="cuda", generator=g
                )
                self.weight_matrices = [shared_w] * self.num_layers
                # Fixed reference input, same seed across all ranks and iterations
                g.manual_seed(1)
                self.reference_input = torch.randn(
                    dim, dim, dtype=self.dtype, device="cuda", generator=g
                )
                self.layer_checksums = [None] * self.num_layers
            else:
                self.weight_matrices = [
                    torch.randn(dim, dim, dtype=self.dtype, device="cuda")
                    for _ in range(self.num_layers)
                ]
            self.activation = torch.randn(
                dim, dim,
                dtype=self.dtype, device="cuda",
            )
            self.grad_buffer = torch.randn(
                dim, dim,
                dtype=self.dtype, device="cuda",
            )

        log.info(
            f"Allocated FSDP buffers: layers={self.num_layers}, "
            f"shard_size={self.shard_size}, "
            f"full_param_size={self.shard_size * ws}, "
            f"compute={'enabled' if cfg.simulate_compute else 'disabled'}"
        )

    def _fill_patterns(self) -> None:
        """Fill buffers with known patterns for verification."""
        # Each rank fills its shards with its rank number
        for shard in self.param_shards:
            shard.fill_(float(self.rank))

        # Each rank fills full_grad with rank + 1 (for reduce_scatter verification)
        self.full_grad.fill_(float(self.rank + 1))

    @staticmethod
    def _checksum(tensor: torch.Tensor) -> int:
        """
        Bitwise checksum: reinterpret-cast to int16 and sum.

        bf16 (or any 16-bit dtype) is viewed as int16 so every bit pattern
        contributes to the checksum with zero information loss -- no float
        rounding, no abs(), and NaN / denorm bit patterns are included.
        Accumulation is done in int64 to avoid overflow.
        """
        return tensor.view(torch.int16).to(torch.int64).sum().item()

    def _forward_layer(self, layer_idx: int) -> None:
        """
        Forward pass for a single FSDP layer.

        1. all_gather: reconstruct full parameter from shards across ranks
        2. GEMM: compute with full parameter (if enabled)

        Shared-weight path: every layer receives the same fixed reference_input
        so that outputs are analytically identical.  Input/output checksums are
        recorded for both the comm kernel (all_gather) and the compute kernel
        (GEMM + GELU) so _verify_layer_checksums() can pinpoint whether
        corruption entered during communication or compute.

        Chained path (default): layer 0 seeds activation from batch_gpu (H2D race
        opportunity) and each subsequent layer receives the previous layer's output.
        """
        use_shared = (
            self.config.shared_layer_weights
            and self.config.compute_type == "transformer"
            and self.reference_input is not None
        )

        # ── comm kernel: all_gather ──────────────────────────────────
        if use_shared:
            comm_input_cksum = self._checksum(self.param_shards[layer_idx])

        dist.all_gather_into_tensor(
            self.full_param, self.param_shards[layer_idx]
        )

        if use_shared:
            comm_output_cksum = self._checksum(self.full_param)

        # ── compute kernel: GEMM + GELU ──────────────────────────────
        if self.config.simulate_compute and self.weight_matrices:
            if use_shared:
                compute_input_cksum = self._checksum(self.reference_input)

                out = torch.mm(self.weight_matrices[layer_idx], self.reference_input)
                out = torch.nn.functional.gelu(out)

                compute_output_cksum = self._checksum(out)

                self.layer_checksums[layer_idx] = {
                    "comm_input": comm_input_cksum,
                    "comm_output": comm_output_cksum,
                    "compute_input": compute_input_cksum,
                    "compute_output": compute_output_cksum,
                }
                self.activation = out
            else:
                # Use batch_gpu for data dependency on first layer (H2D race opportunity)
                if layer_idx == 0:
                    dim = self._dim
                    batch_slice = self.batch_gpu[:dim * dim]
                    self.activation = batch_slice.view(dim, dim)
                self.activation = torch.mm(
                    self.weight_matrices[layer_idx], self.activation
                )
                self.activation = torch.nn.functional.gelu(self.activation)

    def _backward_layer(self, layer_idx: int) -> None:
        """
        Backward pass for a single FSDP layer.

        1. all_gather: reconstruct full parameter (freed after forward)
        2. GEMM backward: compute gradient (if enabled)
        3. reduce_scatter: shard gradients back across ranks
        """
        # all_gather: reconstruct full parameter for backward
        dist.all_gather_into_tensor(
            self.full_param, self.param_shards[layer_idx]
        )

        # GEMM backward (if compute enabled)
        if self.config.simulate_compute and self.weight_matrices:
            if self.config.include_backward_compute:
                self.grad_buffer = torch.mm(
                    self.weight_matrices[layer_idx].T, self.grad_buffer
                )

        # reduce_scatter: sum gradients across ranks, each rank gets its shard
        dist.reduce_scatter_tensor(self.grad_shard, self.full_grad)

    def run_iteration(self, iteration: int) -> bool:
        """
        Run one iteration of FSDP mode.

        Per-layer all_gather/reduce_scatter interleaved with compute,
        with H2D on memcpy_stream (single- or double-buffered).

        Returns True if verification passed (or not in verification phase).
        """
        # Fill buffers with known patterns
        self._fill_patterns()

        if self.config.h2d_prefetch:
            return self._run_iteration_prefetch(iteration)
        else:
            return self._run_iteration_single(iteration)

    def _run_iteration_single(self, iteration: int) -> bool:
        """Single-buffered iteration: transfer → wait → FSDP forward/backward."""
        # ─── H2D ─────────────────────────────────────────────────────
        self._h2d_transfer(iteration)
        self._h2d_wait()

        # ─── Forward: per-layer all_gather + GEMM ────────────────────
        for l in range(self.num_layers):
            self._forward_layer(l)

        # ─── Backward: per-layer all_gather + GEMM bwd + reduce_scatter
        for l in reversed(range(self.num_layers)):
            self._backward_layer(l)

        # ─── Optimizer step ──────────────────────────────────────────
        self._run_optimizer_step()

        # ─── Verify ──────────────────────────────────────────────────
        if self.in_verification_phase:
            torch.cuda.synchronize()
            return self._verify(iteration)

        return True

    def _run_iteration_prefetch(self, iteration: int) -> bool:
        """Double-buffered iteration: wait(prev) → FSDP fwd/bwd → prefetch next → swap."""
        # ─── Ensure current batch is ready ───────────────────────────
        if self._h2d_is_first_iteration:
            self._h2d_transfer(iteration)
            self._h2d_is_first_iteration = False

        self._h2d_wait()

        # ─── Forward: per-layer all_gather + GEMM ────────────────────
        for l in range(self.num_layers):
            self._forward_layer(l)

        # ─── Prefetch next batch (overlaps with backward) ────────────
        self._h2d_prefetch_next(iteration + 1)

        # ─── Backward: per-layer all_gather + GEMM bwd + reduce_scatter
        for l in reversed(range(self.num_layers)):
            self._backward_layer(l)

        # ─── Optimizer step ──────────────────────────────────────────
        self._run_optimizer_step()

        # ─── Verify (before swap) ────────────────────────────────────
        result = True
        if self.in_verification_phase:
            torch.cuda.synchronize()
            result = self._verify(iteration)

        # ─── Swap buffers ────────────────────────────────────────────
        self._h2d_swap_buffers()

        return result

    def _verify(self, iteration: int) -> bool:
        """Verify H2D, last all_gather, last reduce_scatter, and (if shared-weight
        transformer) cross-layer activation consistency."""
        all_correct = True

        # Check H2D result
        if not self._verify_h2d(self.batch_gpu, iteration):
            all_correct = False

        # Check last all_gather result (full_param from last backward layer = layer 0)
        if not self._verify_all_gather():
            all_correct = False

        # Check last reduce_scatter result
        if not self._verify_reduce_scatter():
            all_correct = False

        # Cross-layer checksum comparison (shared-weight transformer only)
        if (
            self.config.shared_layer_weights
            and self.config.compute_type == "transformer"
            and self.layer_checksums
        ):
            if not self._verify_layer_checksums(iteration):
                all_correct = False

        return all_correct

    def _verify_layer_checksums(self, iteration: int) -> bool:
        """
        Verify that per-kernel int16 checksums are identical across all layers.

        With shared weights and a fixed reference input every layer runs the
        same comm kernel (all_gather of rank-filled shard) and the same compute
        kernel (GEMM + GELU with shared W and fixed reference_input).  Both the
        input and output of each kernel are checksummed via reinterpret-cast to
        int16 → int64 sum, so every bit contributes with zero information loss.

        Four checksums per layer:
          comm_input    -- param shard before all_gather (should be identical:
                           every shard is filled with float(rank))
          comm_output   -- full_param after all_gather
          compute_input -- reference_input fed to GEMM (constant across layers)
          compute_output-- activation after GELU

        If comm_output diverges but comm_input matches, corruption is in the
        collective (RCCL / NIC path).  If compute_output diverges but
        comm_output matches, corruption is in the compute kernel (GPU ALU).
        """
        ref = self.layer_checksums[0]
        if ref is None:
            return True

        all_correct = True
        for i in range(1, len(self.layer_checksums)):
            cmp = self.layer_checksums[i]
            if cmp is None:
                continue
            for key in ("comm_input", "comm_output", "compute_input", "compute_output"):
                if cmp[key] != ref[key]:
                    log.error(
                        f"LAYER_CHECKSUM_MISMATCH ({key}): "
                        f"rank={self.rank} iter={iteration} "
                        f"layer_0={ref[key]} layer_{i}={cmp[key]}"
                    )
                    self.corruption_details.append({
                        "type": f"layer_checksum_mismatch_{key}",
                        "rank": self.rank,
                        "iteration": iteration,
                        "layer_ref": 0,
                        "layer_cmp": i,
                        "ref_checksum": ref[key],
                        "cmp_checksum": cmp[key],
                    })
                    all_correct = False

        return all_correct

    def _verify_all_gather(self) -> bool:
        """
        Verify all_gather result.

        Each rank filled its shard with float(rank). After all_gather,
        chunk j of full_param should be float(j).
        """
        all_correct = True

        for src_rank in range(self.world_size):
            start = src_rank * self.shard_size
            end = start + self.shard_size
            chunk = self.full_param[start:end]
            expected = float(src_rank)
            expected_tensor = torch.full_like(chunk, expected)

            if not torch.allclose(chunk, expected_tensor, rtol=1e-3, atol=1e-3):
                actual = chunk[0].item()
                log.error(
                    f"ALL_GATHER CORRUPTION (RUNTIME BUG!): "
                    f"rank={self.rank} src_rank={src_rank} "
                    f"expected={expected} actual={actual}"
                )
                self.corruption_details.append({
                    "type": "all_gather",
                    "rank": self.rank,
                    "src_rank": src_rank,
                    "expected": expected,
                    "actual": actual,
                })
                all_correct = False

        return all_correct

    def _verify_reduce_scatter(self) -> bool:
        """
        Verify reduce_scatter result.

        Each rank filled full_grad with float(rank + 1). After reduce_scatter
        with SUM, each rank's grad_shard should be sum(1..world_size).
        """
        expected = float(sum(range(1, self.world_size + 1)))
        expected_tensor = torch.full_like(self.grad_shard, expected)

        if not torch.allclose(
            self.grad_shard, expected_tensor, rtol=1e-3, atol=1e-3
        ):
            actual = self.grad_shard[0].item()
            log.error(
                f"REDUCE_SCATTER CORRUPTION (RUNTIME BUG!): "
                f"rank={self.rank} expected={expected} actual={actual}"
            )
            self.corruption_details.append({
                "type": "reduce_scatter",
                "rank": self.rank,
                "expected": expected,
                "actual": actual,
            })
            return False

        return True


__all__ = ["FSDPModeReproducer"]
