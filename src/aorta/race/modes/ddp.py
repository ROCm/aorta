"""
DDP mode reproducer (Distributed Data Parallel pattern).

This mode simulates a DDP-style workload with:
- H2D transfer (single- or double-buffered via --prefetch)
- Forward/backward compute with GEMMs
- Gradient all_reduce (no all_to_all)
- Cross-rank gradient consistency verification

Supports two gradient sync strategies via --bucketed:

Non-bucketed (default):
    Forward all layers → Backward all layers → one big all_reduce

Bucketed (--bucketed):
    Forward all layers → [Bwd layer N + all_reduce N] → [Bwd layer N-1 + all_reduce N-1] → ...

Data Flow (single-buffered):
    memcpy_stream:  [H2D] → batch_gpu
                              ↓
    default_stream:          [Forward] → [Backward] → [all_reduce grads]

Data Flow (bucketed, single-buffered):
    memcpy_stream:  [H2D] → batch_gpu
                              ↓
    default_stream:          [Forward] → [Bwd L2 + AR L2] → [Bwd L1 + AR L1] → [Bwd L0 + AR L0]

Data Flow (double-buffered, --prefetch):
    Iteration N:
        memcpy_stream:  [H2D batch_N+1] ────────────────────────┐
                                                                │ (prefetch overlaps)
        default_stream: [Forward(batch_N)] → [Backward] → [all_reduce grads]
                                                                │
                        ← swap buffers ─────────────────────────┘
"""

import logging
from typing import List, Optional

import torch
import torch.distributed as dist

from ..base import BaseReproducer
from ..config import ReproducerConfig

log = logging.getLogger(__name__)


class DDPModeReproducer(BaseReproducer):
    """
    DDP reproducer with gradient all_reduce.

    This mode tests the 2-stream pattern common in DDP training:
    - memcpy_stream: H2D data transfers
    - default_stream: compute + gradient all_reduce

    H2D strategy is controlled by config.h2d_prefetch:
    - False: single-buffered, copy-then-use at start of iteration
    - True (--prefetch): double-buffered, prefetch next batch during backward

    Gradient sync strategy is controlled by config.ddp_bucketed:
    - False (default): one bulk all_reduce after all of backward
    - True (--bucketed): per-layer all_reduce interleaved with backward

    Key features:
    - Deterministic: same seed across ranks for gradient verification
    - Gradient sync: all_reduce on actual computed gradients

    Verification checks:
    - H2D: batch_gpu == iteration % 1000
    - Gradient consistency: all ranks have identical gradient checksums
    """

    def __init__(self, config: ReproducerConfig, rank: int, world_size: int):
        super().__init__(config, rank, world_size)

        self.bucketed: bool = config.ddp_bucketed
        if config.compute_type == "transformer":
            self.num_layers: int = config.num_layers
            self._dim: int = config.model_dim
        else:
            self.num_layers: int = config.gemm_layers
            self._dim: int = config.gemm_size

        # Per-layer weight matrices (bucketed mode only, allocated in setup_buffers)
        self.weight_matrices: List[torch.Tensor] = []
        self.activation: Optional[torch.Tensor] = None
        self.grad_buffer: Optional[torch.Tensor] = None

    def _setup_compute(self) -> None:
        """
        Setup compute: use base class for non-bucketed, manage per-layer for bucketed.

        Bucketed mode needs per-layer backward control so it can interleave
        all_reduce between layers. The base class compute runs all layers in
        one shot, so we override to manage our own weight matrices.
        """
        if not self.bucketed:
            # Non-bucketed: use base class compute as before
            super()._setup_compute()
            return

        # Bucketed: manage per-layer weights internally (like FSDP mode)
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

        log.info("DDP bucketed mode: per-layer compute managed internally")

    def setup_buffers(self) -> None:
        """Allocate per-layer weight matrices for bucketed mode."""
        if not self.bucketed or not self.config.simulate_compute:
            return

        dim = self._dim

        self.weight_matrices = [
            torch.randn(
                dim, dim,
                dtype=self.dtype, device="cuda", requires_grad=True,
            )
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
            f"Allocated DDP bucketed buffers: layers={self.num_layers}, "
            f"dim={dim}"
        )

    def _setup_optimizer(self) -> None:
        """Setup optimizer using the appropriate parameters."""
        if not self.bucketed:
            # Non-bucketed: use base class optimizer setup (uses self.compute.parameters)
            super()._setup_optimizer()
            return

        # Bucketed: optimizer uses our per-layer weight_matrices
        cfg = self.config
        opt_name = cfg.optimizer.lower()

        if opt_name == "none" or not self.weight_matrices:
            self.optimizer = None
            return

        params = self.weight_matrices

        if opt_name == "adamw":
            log.info(f"Using AdamW optimizer (lr={cfg.optimizer_lr})")
            self.optimizer = torch.optim.AdamW(
                params, lr=cfg.optimizer_lr,
                weight_decay=cfg.optimizer_weight_decay,
                betas=cfg.optimizer_betas, eps=cfg.optimizer_eps,
            )
        elif opt_name == "sgd":
            log.info(f"Using SGD optimizer (lr={cfg.optimizer_lr})")
            self.optimizer = torch.optim.SGD(
                params, lr=cfg.optimizer_lr,
                weight_decay=cfg.optimizer_weight_decay, momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer for bucketed DDP: {opt_name}")

    # =====================================================================
    # Non-bucketed gradient sync (original behavior)
    # =====================================================================

    def _gradient_allreduce(self) -> None:
        """
        All-reduce actual gradients (DDP-style, non-bucketed).

        Averages gradients across all ranks by summing and dividing by world_size.
        """
        if self.compute is None:
            return

        for param in self.compute.parameters:
            if param.grad is not None:
                dist.all_reduce(param.grad)
                param.grad.div_(self.world_size)

    # =====================================================================
    # Bucketed per-layer forward / backward + all_reduce
    # =====================================================================

    def _forward_all_layers(self) -> None:
        """
        Forward pass through all layers sequentially.

        Uses batch_gpu for the initial activation (creates H2D data dependency),
        then runs through all GEMM layers with GELU activation.
        """
        if not self.config.simulate_compute or not self.weight_matrices:
            return

        dim = self._dim
        batch_slice = self.batch_gpu[:dim * dim]
        self.activation = batch_slice.view(dim, dim)

        for weight in self.weight_matrices:
            self.activation = torch.mm(weight, self.activation)
            self.activation = torch.nn.functional.gelu(self.activation)

    def _backward_layer_and_allreduce(self, layer_idx: int) -> None:
        """
        Backward GEMM for one layer, then all_reduce its gradient.

        This is the core bucketed pattern: compute gradient for layer L,
        immediately all_reduce it, then move to layer L-1. The all_reduce
        for layer L can overlap with backward compute for layer L-1 on
        the GPU (NCCL internal pipelining).

        Args:
            layer_idx: Index of the layer to process.
        """
        weight = self.weight_matrices[layer_idx]

        # Backward GEMM for this layer
        if self.config.include_backward_compute:
            self.grad_buffer = torch.mm(weight.T, self.grad_buffer)

        # Simulate gradient: use grad_buffer as the "gradient" for this layer
        # In real DDP with autograd, weight.grad is populated by backward().
        # Here we manually create a gradient from the backward computation.
        weight.grad = self.grad_buffer.clone()

        # All-reduce this layer's gradient immediately
        dist.all_reduce(weight.grad)
        weight.grad.div_(self.world_size)

    # =====================================================================
    # Iteration dispatch
    # =====================================================================

    def run_iteration(self, iteration: int) -> bool:
        """
        Run one iteration of DDP mode.

        Dispatches to the appropriate variant based on:
        - config.ddp_bucketed: bucketed vs. non-bucketed gradient sync
        - config.h2d_prefetch: single vs. double-buffered H2D

        Returns True if patterns verified correctly (or not in verification phase).
        """
        if self.bucketed:
            if self.config.h2d_prefetch:
                return self._run_iteration_bucketed_prefetch(iteration)
            else:
                return self._run_iteration_bucketed_single(iteration)
        else:
            if self.config.h2d_prefetch:
                return self._run_iteration_prefetch(iteration)
            else:
                return self._run_iteration_single(iteration)

    # =====================================================================
    # Non-bucketed iterations (unchanged from original)
    # =====================================================================

    def _run_iteration_single(self, iteration: int) -> bool:
        """
        Single-buffered iteration: transfer → wait → forward → backward → all_reduce.

        Data Flow:
          memcpy_stream:  [H2D] → batch_gpu
          default_stream:          [Forward] → [Backward] → [all_reduce grads]
        """
        # ─── Phase 1: H2D on memcpy_stream ───────────────────────────
        self._h2d_transfer(iteration)

        # ─── Phase 2: PROPER SYNC - wait for H2D ─────────────────────
        self._h2d_wait()

        # ─── Phase 3: Forward pass ───────────────────────────────────
        forward_output = None
        if self.compute:
            forward_output = self.compute.forward(self.batch_gpu)

        # ─── Phase 4: Backward pass (computes gradients) ─────────────
        if self.compute:
            self.compute.backward(
                forward_output,
                use_autograd=(self.optimizer is not None)
            )

        # ─── Phase 5: DDP gradient sync ──────────────────────────────
        self._gradient_allreduce()

        # ─── Phase 6: Optimizer step (if enabled) ────────────────────
        self._run_optimizer_step()

        # ─── Phase 7: Verify patterns ────────────────────────────────
        if self.in_verification_phase:
            torch.cuda.synchronize()
            return self._verify(iteration)

        return True

    def _run_iteration_prefetch(self, iteration: int) -> bool:
        """
        Double-buffered iteration: wait(prev) → forward → prefetch_next → ...

        Data Flow:
          memcpy_stream:  [H2D batch_N+1 (prefetch)] ──────────────────┐
                                                                        │ overlap
          default_stream: [Forward(batch_N)] → [Backward] → [all_reduce grads]
                                                                        │
                          ← swap buffers ───────────────────────────────┘
        """
        # ─── Phase 1: Ensure current batch is ready ──────────────────
        if self._h2d_is_first_iteration:
            self._h2d_transfer(iteration)
            self._h2d_is_first_iteration = False

        self._h2d_wait()

        # ─── Phase 2: Forward pass (uses current batch) ──────────────
        forward_output = None
        if self.compute:
            forward_output = self.compute.forward(self.batch_gpu)

        # ─── Phase 3: Start prefetching NEXT batch ───────────────────
        self._h2d_prefetch_next(iteration + 1)

        # ─── Phase 4: Backward pass (computes gradients) ─────────────
        if self.compute:
            self.compute.backward(
                forward_output,
                use_autograd=(self.optimizer is not None)
            )

        # ─── Phase 5: DDP gradient sync ──────────────────────────────
        self._gradient_allreduce()

        # ─── Phase 6: Optimizer step (if enabled) ────────────────────
        self._run_optimizer_step()

        # ─── Phase 7: Verify patterns (before swapping buffers) ──────
        result = True
        if self.in_verification_phase:
            torch.cuda.synchronize()
            result = self._verify(iteration)

        # ─── Phase 8: Swap buffers for next iteration ────────────────
        self._h2d_swap_buffers()

        return result

    # =====================================================================
    # Bucketed iterations (per-layer backward + all_reduce)
    # =====================================================================

    def _run_iteration_bucketed_single(self, iteration: int) -> bool:
        """
        Bucketed single-buffered: forward → per-layer [backward + all_reduce].

        Data Flow:
          memcpy_stream:  [H2D] → batch_gpu
          default_stream:          [Forward all layers]
                                   [Bwd L2 + AR L2] → [Bwd L1 + AR L1] → [Bwd L0 + AR L0]
        """
        # ─── H2D ─────────────────────────────────────────────────────
        self._h2d_transfer(iteration)
        self._h2d_wait()

        # ─── Forward all layers ──────────────────────────────────────
        self._forward_all_layers()

        # ─── Per-layer backward + all_reduce ─────────────────────────
        if self.config.simulate_compute and self.weight_matrices:
            for l in reversed(range(self.num_layers)):
                self._backward_layer_and_allreduce(l)

        # ─── Optimizer step ──────────────────────────────────────────
        self._run_optimizer_step()

        # ─── Verify ──────────────────────────────────────────────────
        if self.in_verification_phase:
            torch.cuda.synchronize()
            return self._verify(iteration)

        return True

    def _run_iteration_bucketed_prefetch(self, iteration: int) -> bool:
        """
        Bucketed double-buffered: wait → forward → prefetch → per-layer [bwd + AR] → swap.

        Data Flow:
          memcpy_stream:  [H2D batch_N+1 (prefetch)] ───────────────────────────────┐
                                                                                     │ overlap
          default_stream: [Forward all layers]                                       │
                          [Bwd L2 + AR L2] → [Bwd L1 + AR L1] → [Bwd L0 + AR L0]  │
                                                                                     │
                          ← swap buffers ────────────────────────────────────────────┘
        """
        # ─── Ensure current batch is ready ───────────────────────────
        if self._h2d_is_first_iteration:
            self._h2d_transfer(iteration)
            self._h2d_is_first_iteration = False

        self._h2d_wait()

        # ─── Forward all layers ──────────────────────────────────────
        self._forward_all_layers()

        # ─── Prefetch next batch (overlaps with backward) ────────────
        self._h2d_prefetch_next(iteration + 1)

        # ─── Per-layer backward + all_reduce ─────────────────────────
        if self.config.simulate_compute and self.weight_matrices:
            for l in reversed(range(self.num_layers)):
                self._backward_layer_and_allreduce(l)

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

    # =====================================================================
    # Verification
    # =====================================================================

    def _verify(self, iteration: int) -> bool:
        """Verify DDP patterns: H2D correctness and gradient consistency."""
        all_correct = True

        if not self._verify_h2d(self.batch_gpu, iteration):
            all_correct = False

        if not self._verify_gradient_consistency(iteration):
            all_correct = False

        return all_correct

    def _verify_gradient_consistency(self, iteration: int) -> bool:
        """
        Verify gradient consistency across ranks.

        Since all ranks use same input and same weights (deterministic mode),
        gradients should be identical after averaging.

        Works with both non-bucketed (self.compute.parameters) and bucketed
        (self.weight_matrices) parameter sources.
        """
        # Get parameter list from the appropriate source
        if self.bucketed:
            params = self.weight_matrices
        elif self.compute is not None:
            params = self.compute.parameters
        else:
            return True

        if not params:
            return True

        # Compute local gradient checksum (sum of all grad elements)
        local_checksum = torch.zeros(1, device="cuda", dtype=torch.float32)
        for param in params:
            if param.grad is not None:
                local_checksum += param.grad.sum().float()

        # Gather checksums from all ranks
        all_checksums = [
            torch.zeros(1, device="cuda", dtype=torch.float32)
            for _ in range(self.world_size)
        ]
        dist.all_gather(all_checksums, local_checksum)

        # All checksums should be identical (same input → same grad after avg)
        reference_checksum = all_checksums[0]
        all_correct = True

        for r, cs in enumerate(all_checksums):
            if not torch.allclose(cs, reference_checksum, rtol=1e-2, atol=1e-2):
                log.error(
                    f"GRADIENT DIVERGENCE (RUNTIME BUG!): iter={iteration} "
                    f"rank {r} checksum={cs.item():.6f} differs from "
                    f"rank 0 checksum={reference_checksum.item():.6f}"
                )
                self.corruption_details.append({
                    "type": "gradient_divergence",
                    "iteration": iteration,
                    "rank": self.rank,
                    "divergent_rank": r,
                    "expected_checksum": reference_checksum.item(),
                    "actual_checksum": cs.item(),
                })
                all_correct = False

        return all_correct


__all__ = ["DDPModeReproducer"]
