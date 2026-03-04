"""
Base reproducer class for RCCL race condition testing.

This module provides the abstract base class that all reproducer modes
inherit from, containing shared logic for setup, run loop, and utilities.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from .config import ReproducerConfig, ReproducerResult
from .compute import BaseCompute, create_compute

log = logging.getLogger(__name__)


class BaseReproducer(ABC):
    """
    Abstract base class for RCCL race condition reproducers.

    Subclasses must implement:
    - setup_buffers(): Allocate mode-specific buffers
    - run_iteration(): Run one iteration of the mode-specific data flow

    Shared functionality provided:
    - Stream creation
    - Compute simulation setup
    - Optimizer creation
    - Run loop (warmup + verification)
    - H2D verification utility

    Usage:
        class MyModeReproducer(BaseReproducer):
            def setup_buffers(self): ...
            def run_iteration(self, iteration): ...

        reproducer = MyModeReproducer(config, rank, world_size)
        result = reproducer.run()
    """

    def __init__(self, config: ReproducerConfig, rank: int, world_size: int):
        """
        Initialize the reproducer.

        Args:
            config: Reproducer configuration.
            rank: Current process rank.
            world_size: Total number of processes.
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size

        # Streams (shared)
        self.memcpy_stream: Optional[torch.cuda.Stream] = None
        self.default_stream: Optional[torch.cuda.Stream] = None

        # Compute simulator
        self.compute: Optional[BaseCompute] = None

        # Optimizer
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # H2D buffers (managed by base class for all modes)
        self.batch_cpu: Optional[torch.Tensor] = None
        self.batch_gpu: Optional[torch.Tensor] = None
        self._batch_cpu_next: Optional[torch.Tensor] = None   # double-buffer only
        self._batch_gpu_next: Optional[torch.Tensor] = None   # double-buffer only
        self._h2d_is_first_iteration: bool = True

        # State
        self.in_verification_phase: bool = False
        self.corruption_details: List[Dict] = []

        # Dtype
        self.dtype = self._get_dtype()

    def _get_dtype(self) -> torch.dtype:
        """Get torch dtype from config string."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.config.dtype, torch.bfloat16)

    # =========================================================================
    # Setup Methods
    # =========================================================================

    def _setup_env(self) -> None:
        """Set environment variables."""
        if self.config.gpu_max_hw_queues is not None:
            os.environ["GPU_MAX_HW_QUEUES"] = str(self.config.gpu_max_hw_queues)
            log.info(f"Set GPU_MAX_HW_QUEUES={self.config.gpu_max_hw_queues}")

    def _setup_deterministic(self) -> None:
        """
        Fix seeds for reproducible computation across all ranks.

        Required for cross-rank gradient consistency verification.
        """
        if not self.config.deterministic:
            return

        seed = self.config.deterministic_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log.info(f"Set deterministic mode with seed={seed}")

    def _setup_streams(self) -> None:
        """Create CUDA streams."""
        self.memcpy_stream = torch.cuda.Stream()
        self.default_stream = torch.cuda.current_stream()
        log.info("Created memcpy_stream and default_stream")

    def _setup_h2d_buffers(self) -> None:
        """
        Allocate H2D buffers based on config.h2d_prefetch.

        Single-buffered: batch_cpu + batch_gpu (copy-then-use each iteration).
        Double-buffered: + _batch_cpu_next + _batch_gpu_next (prefetch overlaps
        with compute; buffers are swapped at end of each iteration).

        All modes use self.batch_gpu for the current batch.
        """
        cfg = self.config
        pin = cfg.pin_memory

        # Always allocate current buffers
        self.batch_cpu = torch.empty(
            cfg.h2d_tensor_size, dtype=self.dtype,
            pin_memory=pin,
        )
        self.batch_gpu = torch.empty(
            cfg.h2d_tensor_size, dtype=self.dtype, device="cuda",
        )

        if cfg.h2d_prefetch:
            # Double-buffer: allocate next-batch buffers
            self._batch_cpu_next = torch.empty(
                cfg.h2d_tensor_size, dtype=self.dtype,
                pin_memory=pin,
            )
            self._batch_gpu_next = torch.empty(
                cfg.h2d_tensor_size, dtype=self.dtype, device="cuda",
            )
            log.info(
                f"H2D: double-buffered (prefetch) mode, size={cfg.h2d_tensor_size}"
            )
        else:
            log.info(f"H2D: single-buffered mode, size={cfg.h2d_tensor_size}")

    def _setup_compute(self) -> None:
        """Setup compute simulator if enabled."""
        if not self.config.simulate_compute:
            self.compute = None
            return

        # Validate buffer sizes based on compute type
        if self.config.compute_type == "transformer":
            dim = self.config.model_dim
        else:
            dim = self.config.gemm_size
        min_h2d_size = dim * dim
        if self.config.h2d_tensor_size < min_h2d_size:
            log.warning(
                f"h2d_tensor_size ({self.config.h2d_tensor_size}) < {dim}² ({min_h2d_size}). "
                f"Increasing to {min_h2d_size} for compute simulation."
            )
            self.config.h2d_tensor_size = min_h2d_size

        requires_grad = self.config.optimizer.lower() != "none"
        self.compute = create_compute(
            self.config.compute_type, self.config, self.dtype
        )
        self.compute.setup(requires_grad=requires_grad)
        log.info(f"Setup {self.config.compute_type} compute simulator")

    def _setup_optimizer(self) -> None:
        """Create optimizer if specified."""
        cfg = self.config
        opt_name = cfg.optimizer.lower()

        if opt_name == "none":
            self.optimizer = None
            return

        if self.compute is None or not self.compute.parameters:
            log.warning("No parameters to optimize (simulate_compute=False?)")
            self.optimizer = None
            return

        params = self.compute.parameters

        if opt_name == "adamw":
            log.info(f"Using AdamW optimizer (lr={cfg.optimizer_lr})")
            self.optimizer = torch.optim.AdamW(
                params,
                lr=cfg.optimizer_lr,
                weight_decay=cfg.optimizer_weight_decay,
                betas=cfg.optimizer_betas,
                eps=cfg.optimizer_eps,
            )
        elif opt_name == "sgd":
            log.info(f"Using SGD optimizer (lr={cfg.optimizer_lr})")
            self.optimizer = torch.optim.SGD(
                params,
                lr=cfg.optimizer_lr,
                weight_decay=cfg.optimizer_weight_decay,
                momentum=0.9,
            )
        elif opt_name == "shampoo":
            try:
                from distributed_shampoo import DDPDistributedConfig, DistributedShampoo

                distributed_config = DDPDistributedConfig(
                    communication_dtype=torch.float32,
                    num_trainers_per_group=-1,
                    communicate_params=False,
                )
                log.info(f"Using DistributedShampoo optimizer (lr={cfg.optimizer_lr})")
                self.optimizer = DistributedShampoo(
                    params,
                    lr=cfg.optimizer_lr,
                    betas=cfg.optimizer_betas,
                    epsilon=cfg.optimizer_eps,
                    weight_decay=cfg.optimizer_weight_decay,
                    distributed_config=distributed_config,
                )
            except ImportError:
                log.error(
                    "distributed_shampoo package not installed. "
                    "Install with: pip install distributed_shampoo"
                )
                raise
        else:
            raise ValueError(
                f"Unknown optimizer: {cfg.optimizer}. "
                f"Options: none, adamw, sgd, shampoo"
            )

    def setup(self) -> None:
        """
        Setup the reproducer.

        Subclasses should call super().setup() and then add mode-specific setup.
        """
        self._setup_env()
        self._setup_deterministic()
        self._setup_streams()
        self._setup_compute()
        self._setup_h2d_buffers()  # H2D buffers (shared by all modes)
        self.setup_buffers()       # Mode-specific (non-H2D buffers)
        self._setup_optimizer()

        log.info(
            f"Reproducer setup complete: mode={self.config.mode}, rank={self.rank}, "
            f"world_size={self.world_size}, h2d_size={self.config.h2d_tensor_size}, "
            f"h2d_prefetch={self.config.h2d_prefetch}, "
            f"dtype={self.config.dtype}, optimizer={self.config.optimizer}"
        )

    @abstractmethod
    def setup_buffers(self) -> None:
        """
        Allocate mode-specific buffers (NOT H2D buffers).

        H2D buffers (batch_cpu, batch_gpu, and double-buffer variants) are
        managed by the base class via _setup_h2d_buffers(). Subclasses only
        need to allocate buffers specific to their communication pattern
        (e.g., all_to_all send/recv buffers, all_reduce buffers).
        """
        pass

    # =========================================================================
    # Iteration Methods
    # =========================================================================

    @abstractmethod
    def run_iteration(self, iteration: int) -> bool:
        """
        Run one iteration of the reproducer.

        Implemented by subclasses to define their specific data flow.

        Args:
            iteration: Current iteration number.

        Returns:
            True if verification passed (or not in verification phase).
            False if corruption detected.
        """
        pass

    # =========================================================================
    # H2D Primitives (used by all modes)
    # =========================================================================

    def _h2d_transfer(self, iteration: int) -> None:
        """
        Fill current batch CPU buffer with known pattern and start async H2D.

        Used at the start of each iteration in single-buffered mode,
        and only for the first iteration in double-buffered mode (subsequent
        iterations rely on the prefetch from the previous iteration).

        Args:
            iteration: Current iteration number (pattern = iteration % 1000).
        """
        self.batch_cpu.fill_(float(iteration % 1000))
        with torch.cuda.stream(self.memcpy_stream):
            self.batch_gpu.copy_(self.batch_cpu, non_blocking=True)

    def _h2d_prefetch_next(self, next_iteration: int) -> None:
        """
        Start prefetching the next batch on memcpy_stream.

        No-op when h2d_prefetch is disabled (single-buffered mode).
        In double-buffered mode, fills _batch_cpu_next with the pattern for
        next_iteration and starts async copy to _batch_gpu_next.

        Call this where you want the prefetch to overlap (e.g., after forward,
        during backward).

        Args:
            next_iteration: Iteration number for the next batch.
        """
        if not self.config.h2d_prefetch:
            return
        self._batch_cpu_next.fill_(float(next_iteration % 1000))
        with torch.cuda.stream(self.memcpy_stream):
            self._batch_gpu_next.copy_(self._batch_cpu_next, non_blocking=True)

    def _h2d_wait(self) -> None:
        """Wait for the current H2D transfer to complete on the default stream."""
        self.default_stream.wait_stream(self.memcpy_stream)

    def _h2d_swap_buffers(self) -> None:
        """
        Swap current and next buffers for the next iteration.

        No-op when h2d_prefetch is disabled (single-buffered mode).
        In double-buffered mode, swaps batch_gpu <-> _batch_gpu_next and
        batch_cpu <-> _batch_cpu_next so that the prefetched data becomes
        the current batch for the next iteration.
        """
        if not self.config.h2d_prefetch:
            return
        self.batch_gpu, self._batch_gpu_next = (
            self._batch_gpu_next, self.batch_gpu
        )
        self.batch_cpu, self._batch_cpu_next = (
            self._batch_cpu_next, self.batch_cpu
        )

    # =========================================================================
    # Shared Utilities
    # =========================================================================

    def _run_optimizer_step(self) -> None:
        """Run optimizer step to update weights."""
        if self.optimizer is None:
            return
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _verify_h2d(self, batch_gpu: torch.Tensor, iteration: int) -> bool:
        """
        Verify H2D transfer correctness.

        Shared by all modes - checks that batch_gpu contains the expected
        known pattern value.

        Args:
            batch_gpu: GPU tensor to verify.
            iteration: Current iteration number.

        Returns:
            True if correct, False if corruption detected.
        """
        expected = float(iteration % 1000)
        expected_tensor = torch.full_like(batch_gpu, expected)

        if not torch.allclose(batch_gpu, expected_tensor, rtol=1e-3, atol=1e-3):
            actual = batch_gpu[0].item()
            log.error(
                f"H2D CORRUPTION (RUNTIME BUG!): iter={iteration} rank={self.rank} "
                f"expected={expected} actual={actual}"
            )
            self.corruption_details.append({
                "type": "h2d",
                "iteration": iteration,
                "rank": self.rank,
                "expected": expected,
                "actual": actual,
            })
            return False
        return True

    # =========================================================================
    # Run Loop
    # =========================================================================

    def run(self) -> ReproducerResult:
        """
        Run the full reproducer: warmup + verification.

        Returns:
            ReproducerResult with pass/fail status and details.
        """
        cfg = self.config

        # Setup
        self.setup()

        start_time = time.time()
        corruption_count = 0
        first_corruption_iter = None
        total_iterations = 0

        # ─────────────────────────────────────────────────────────────
        # Phase 1: Warmup (no verification)
        # ─────────────────────────────────────────────────────────────
        self.in_verification_phase = False
        log.info(f"Starting warmup phase: {cfg.warmup_iterations} iterations")

        warmup_start = time.time()
        for i in range(cfg.warmup_iterations):
            self.run_iteration(i)

            # CRITICAL: Synchronize during warmup to actually execute GPU work
            # Without this, iterations just queue work and finish instantly
            torch.cuda.synchronize()

            total_iterations += 1

            if (i + 1) % cfg.log_interval == 0:
                warmup_elapsed = time.time() - warmup_start
                avg_iter_ms = (warmup_elapsed * 1000) / (i + 1)
                log.info(
                    f"Warmup progress: {i + 1}/{cfg.warmup_iterations}, "
                    f"avg_step_time: {avg_iter_ms:.1f}ms"
                )

        warmup_total = time.time() - warmup_start
        warmup_avg_ms = (
            (warmup_total * 1000) / cfg.warmup_iterations
            if cfg.warmup_iterations > 0 else 0
        )
        log.info(
            f"Warmup complete: {warmup_total:.1f}s total, "
            f"{warmup_avg_ms:.1f}ms avg per step"
        )

        # Warn if timing is too fast (may not trigger timing-sensitive bugs)
        if cfg.simulate_compute and warmup_avg_ms < 400:
            log.warning(
                f"Step time ({warmup_avg_ms:.1f}ms) is faster than target (~500ms/step). "
                f"Consider increasing gemm_size or gemm_layers to match timing profile."
            )

        # ─────────────────────────────────────────────────────────────
        # Phase 2: Verification
        # ─────────────────────────────────────────────────────────────
        self.in_verification_phase = True
        log.info(f"Starting verification phase: {cfg.verify_iterations} iterations")

        for i in range(cfg.verify_iterations):
            iteration = cfg.warmup_iterations + i

            passed = self.run_iteration(iteration)
            total_iterations += 1

            if not passed:
                corruption_count += 1
                if first_corruption_iter is None:
                    first_corruption_iter = iteration

                if cfg.stop_on_first_corruption:
                    log.error(
                        f"Stopping on first corruption at iteration {iteration}"
                    )
                    break

            if (i + 1) % cfg.log_interval == 0:
                log.info(
                    f"Verification progress: {i + 1}/{cfg.verify_iterations}, "
                    f"corruptions={corruption_count}"
                )

        elapsed = time.time() - start_time
        avg_step_ms = (elapsed * 1000) / total_iterations if total_iterations > 0 else 0

        # Summary
        if corruption_count > 0:
            log.error(
                f"RUNTIME BUG DETECTED: {corruption_count} corruptions in "
                f"{total_iterations} iterations"
            )
            log.error(
                "Corruption occurred DESPITE proper synchronization - "
                "this is a bug in RCCL/HIP runtime"
            )
        else:
            log.info(
                f"PASSED: No corruption in {total_iterations} iterations "
                f"with proper synchronization"
            )
            log.info(
                "If corruption still occurs in real workloads, check for "
                "application-level synchronization issues."
            )

        return ReproducerResult(
            passed=(corruption_count == 0),
            total_iterations=total_iterations,
            corruption_count=corruption_count,
            first_corruption_iter=first_corruption_iter,
            corruption_details=self.corruption_details,
            elapsed_time_sec=elapsed,
            avg_step_time_ms=avg_step_ms,
        )


__all__ = ["BaseReproducer"]
