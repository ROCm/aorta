"""
Default mode reproducer (TorchRec-like pattern).

This mode simulates a TorchRec-style workload with:
- H2D transfer for batch data (single- or double-buffered via --prefetch)
- all_to_all for sparse embedding distribution
- Forward/backward compute with GEMMs
- all_reduce for gradient synchronization

Data Flow (single-buffered, default):
    memcpy_stream:  [H2D] → batch_gpu
                              ↓ (Forward READS batch_gpu)
    default_stream:          [Forward] → [Backward] → [all_reduce]
    datadist_stream:         [all_to_all]
                              (overlaps with backward)

Data Flow (double-buffered, --prefetch):
    memcpy_stream:  [H2D batch_N+1 (prefetch)] ──────────────────────┐
                                                                      │ overlap
    default_stream: [Forward(batch_N)] → [Backward] → [all_reduce]   │
    datadist_stream:                    [all_to_all]                  │
                    ← swap buffers ───────────────────────────────────┘
"""

import logging
from typing import Optional

import torch
import torch.distributed as dist

from ..base import BaseReproducer
from ..config import ReproducerConfig

log = logging.getLogger(__name__)


class DefaultModeReproducer(BaseReproducer):
    """
    TorchRec-like reproducer with all_to_all + all_reduce.

    This mode tests the 3-stream pattern common in recommendation models:
    - memcpy_stream: H2D data transfers
    - datadist_stream: all_to_all collectives (sparse embedding exchange)
    - default_stream: compute + all_reduce (gradient sync)

    H2D strategy is controlled by config.h2d_prefetch:
    - False (default): single-buffered, copy-then-use at start of iteration
    - True (--prefetch): double-buffered, prefetch next batch during backward

    Verification checks:
    - H2D: batch_gpu == iteration % 1000
    - all_to_all: recv_buf[j] == j (data from rank j)
    - all_reduce: reduce_buf == sum(1..world_size)
    """

    def __init__(self, config: ReproducerConfig, rank: int, world_size: int):
        super().__init__(config, rank, world_size)

        # Mode-specific stream
        self.datadist_stream: Optional[torch.cuda.Stream] = None

        # Mode-specific buffers (H2D buffers are in base class)
        self.send_buf: Optional[torch.Tensor] = None
        self.recv_buf: Optional[torch.Tensor] = None
        self.reduce_buf: Optional[torch.Tensor] = None

    def _setup_streams(self) -> None:
        """Create CUDA streams including datadist_stream."""
        super()._setup_streams()

        if self.config.same_stream_mode:
            # Same stream for H2D and datadist (definitive runtime bug test)
            self.datadist_stream = self.memcpy_stream
            log.info("Using SAME stream for H2D and datadist (same_stream_mode)")
        else:
            self.datadist_stream = torch.cuda.Stream()
            log.info("Using separate datadist_stream")

    def setup_buffers(self) -> None:
        """Allocate mode-specific buffers (all_to_all + all_reduce)."""
        cfg = self.config

        # all_to_all buffers
        self.send_buf = torch.empty(
            self.world_size, cfg.alltoall_tensor_size,
            dtype=self.dtype, device="cuda"
        )
        self.recv_buf = torch.empty_like(self.send_buf)

        # all_reduce buffer
        self.reduce_buf = torch.empty(
            cfg.allreduce_tensor_size, dtype=self.dtype, device="cuda"
        )

        log.info(
            f"Allocated default mode buffers: "
            f"a2a={cfg.alltoall_tensor_size}, ar={cfg.allreduce_tensor_size}"
        )

    def _fill_collective_patterns(self) -> None:
        """Fill collective buffers with known patterns for verification."""
        # all_to_all: send_buf[i] = rank for all i
        self.send_buf.fill_(float(self.rank))

        # all_reduce: reduce_buf = rank + 1
        self.reduce_buf.fill_(float(self.rank + 1))

    def _run_alltoall(self) -> dist.Work:
        """Run all_to_all on datadist_stream."""
        with torch.cuda.stream(self.datadist_stream):
            work = dist.all_to_all_single(
                self.recv_buf, self.send_buf, async_op=True
            )
        return work

    def _run_allreduce(self) -> None:
        """Run all_reduce on default stream."""
        dist.all_reduce(self.reduce_buf)

    def run_iteration(self, iteration: int) -> bool:
        """
        Run one iteration of the default (TorchRec-like) mode.

        Supports both single-buffered and double-buffered (prefetch) H2D
        via config.h2d_prefetch. The mode controls where H2D primitives are
        called; the base class handles buffer allocation and mechanics.

        Returns True if patterns verified correctly (or not in verification phase).
        """
        # Fill collective buffers with known patterns
        self._fill_collective_patterns()

        if self.config.h2d_prefetch:
            return self._run_iteration_prefetch(iteration)
        else:
            return self._run_iteration_single(iteration)

    def _run_iteration_single(self, iteration: int) -> bool:
        """
        Single-buffered iteration: transfer → wait → forward → ...

        Data Flow:
          memcpy_stream:  [H2D] → batch_gpu
          default_stream:          [Forward] → [Backward] → [all_reduce]
          datadist_stream:         [all_to_all] (overlaps with backward)
        """
        # ─── Phase 1: H2D on memcpy_stream ───────────────────────────
        self._h2d_transfer(iteration)

        # ─── Phase 2: PROPER SYNC - wait for H2D ─────────────────────
        self._h2d_wait()

        # ─── Phase 3: Forward pass (if enabled) ──────────────────────
        forward_output = None
        if self.compute:
            forward_output = self.compute.forward(self.batch_gpu)

        # ─── Phase 4: all_to_all on datadist_stream ──────────────────
        work_a2a = self._run_alltoall()

        # ─── Phase 5: Backward pass (if enabled) ─────────────────────
        if self.compute:
            self.compute.backward(
                forward_output,
                use_autograd=(self.optimizer is not None)
            )

        # ─── Phase 6: PROPER SYNC - wait for all_to_all ──────────────
        self.default_stream.wait_stream(self.datadist_stream)
        work_a2a.wait()

        # ─── Phase 7: Optimizer step (if enabled) ────────────────────
        self._run_optimizer_step()

        # ─── Phase 8: all_reduce on default stream ───────────────────
        self._run_allreduce()

        # ─── Phase 9: Verify patterns ────────────────────────────────
        if self.in_verification_phase:
            torch.cuda.synchronize()
            return self._verify(iteration)

        return True

    def _run_iteration_prefetch(self, iteration: int) -> bool:
        """
        Double-buffered iteration: wait(prev) → forward → prefetch_next → ...

        Data Flow:
          memcpy_stream:  [H2D batch_N+1 (prefetch)] ─────────────────┐
                                                                       │ overlap
          default_stream: [Forward(batch_N)] → [Backward] → [all_reduce]
          datadist_stream:                    [all_to_all]             │
                          ← swap buffers ──────────────────────────────┘
        """
        # ─── Phase 1: Ensure current batch is ready ──────────────────
        # First iteration: explicit transfer (no previous prefetch)
        if self._h2d_is_first_iteration:
            self._h2d_transfer(iteration)
            self._h2d_is_first_iteration = False

        # Wait for current batch H2D to complete
        self._h2d_wait()

        # ─── Phase 2: Forward pass (uses current batch) ──────────────
        forward_output = None
        if self.compute:
            forward_output = self.compute.forward(self.batch_gpu)

        # ─── Phase 3: Start prefetching NEXT batch ───────────────────
        self._h2d_prefetch_next(iteration + 1)

        # ─── Phase 4: all_to_all on datadist_stream ──────────────────
        work_a2a = self._run_alltoall()

        # ─── Phase 5: Backward pass (if enabled) ─────────────────────
        if self.compute:
            self.compute.backward(
                forward_output,
                use_autograd=(self.optimizer is not None)
            )

        # ─── Phase 6: PROPER SYNC - wait for all_to_all ──────────────
        self.default_stream.wait_stream(self.datadist_stream)
        work_a2a.wait()

        # ─── Phase 7: Optimizer step (if enabled) ────────────────────
        self._run_optimizer_step()

        # ─── Phase 8: all_reduce on default stream ───────────────────
        self._run_allreduce()

        # ─── Phase 9: Verify patterns (before swapping buffers) ──────
        result = True
        if self.in_verification_phase:
            torch.cuda.synchronize()
            result = self._verify(iteration)

        # ─── Phase 10: Swap buffers for next iteration ───────────────
        self._h2d_swap_buffers()

        return result

    def _verify(self, iteration: int) -> bool:
        """Verify all buffers contain expected patterns."""
        all_correct = True

        # Check H2D result
        if not self._verify_h2d(self.batch_gpu, iteration):
            all_correct = False

        # Check all_to_all result
        if not self._verify_alltoall():
            all_correct = False

        # Check all_reduce result
        if not self._verify_allreduce():
            all_correct = False

        return all_correct

    def _verify_alltoall(self) -> bool:
        """Verify all_to_all result: recv_buf[j] should be all j's."""
        all_correct = True

        for src_rank in range(self.world_size):
            expected = float(src_rank)
            expected_tensor = torch.full_like(self.recv_buf[src_rank], expected)

            if not torch.allclose(
                self.recv_buf[src_rank], expected_tensor, rtol=1e-3, atol=1e-3
            ):
                actual = self.recv_buf[src_rank, 0].item()
                log.error(
                    f"ALL_TO_ALL CORRUPTION (RUNTIME BUG!): "
                    f"rank={self.rank} src_rank={src_rank} "
                    f"expected={expected} actual={actual}"
                )
                self.corruption_details.append({
                    "type": "all_to_all",
                    "rank": self.rank,
                    "src_rank": src_rank,
                    "expected": expected,
                    "actual": actual,
                })
                all_correct = False

        return all_correct

    def _verify_allreduce(self) -> bool:
        """Verify all_reduce result: should be sum(1..world_size)."""
        expected = float(sum(range(1, self.world_size + 1)))
        expected_tensor = torch.full_like(self.reduce_buf, expected)

        if not torch.allclose(
            self.reduce_buf, expected_tensor, rtol=1e-3, atol=1e-3
        ):
            actual = self.reduce_buf[0].item()
            log.error(
                f"ALL_REDUCE CORRUPTION (RUNTIME BUG!): "
                f"rank={self.rank} expected={expected} actual={actual}"
            )
            self.corruption_details.append({
                "type": "all_reduce",
                "rank": self.rank,
                "expected": expected,
                "actual": actual,
            })
            return False

        return True


__all__ = ["DefaultModeReproducer"]
