"""
FSDP mode reproducer (Fully Sharded Data Parallel pattern).

NOTE: This *simulates* the FSDP communication pattern with EXPLICIT
``all_gather`` / ``reduce_scatter`` calls. It does NOT use ``torch.distributed``
FSDP (neither FSDP1 ``FullyShardedDataParallel`` nor FSDP2 ``fully_shard``).
When ``compute_type=transformer``, ``RepeatedTransformerBlock`` is reused only as
a compute kernel between the explicit collectives -- it is not wrapped in
``fully_shard``. Explicit collectives are what give the clean rank-fill +
shared-input checksum invariant. Real-FSDP coverage is a separate, deferred
workload (PR D).

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
from collections import Counter
from typing import List, Optional

import torch
import torch.distributed as dist

from aorta.models import BlockConfig, RepeatedTransformerBlock

from ..base import BaseReproducer
from ..config import ReproducerConfig

log = logging.getLogger(__name__)
_REDUCE_SCATTER_ORACLE_DTYPE = torch.float32
_MAX_EXACT_RANK_FILL = {
    torch.bfloat16: 256,
    torch.float16: 2_048,
    torch.float32: 16_777_216,
}


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

        # Effective (resolved) transformer block shape. num_heads/ffn_size are
        # auto-derived when 0, so the config value alone doesn't record what
        # actually ran -- store the resolved values for the startup log + metrics.
        self.eff_num_heads: Optional[int] = None
        self.eff_ffn_size: Optional[int] = None
        self.eff_seq_len: Optional[int] = None
        self.eff_batch_size: Optional[int] = None
        self.effective_h2d_tensor_size: int = config.h2d_tensor_size
        self.reduce_scatter_oracle_dtype: str = "float32"
        self._iteration_collectives_correct: bool = True

        # Real transformer block shared across all layers (shared-weight path)
        self.shared_block: Optional[RepeatedTransformerBlock] = None

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

        # Validate buffer sizes based on compute type.
        # NOTE: for the shared-weight transformer path the real compute size is
        # governed by batch_size × seq_len × model_dim (the block's activation),
        # not dim²; this min only sizes the H2D staging buffer and stays harmless.
        dim = self._dim
        min_h2d_size = dim * dim
        if self.config.h2d_tensor_size < min_h2d_size:
            log.warning(
                f"h2d_tensor_size ({self.config.h2d_tensor_size}) < {dim}² "
                f"({min_h2d_size}). Using effective size {min_h2d_size} for compute."
            )
            self.effective_h2d_tensor_size = min_h2d_size
        else:
            self.effective_h2d_tensor_size = self.config.h2d_tensor_size

        # NOTE: We do NOT create a base compute simulator (self.compute stays None).
        # FSDP mode creates per-layer weight_matrices in setup_buffers() because
        # collectives and compute are interleaved per-layer.
        log.info("FSDP mode: per-layer compute managed internally (no base compute)")

    def setup_buffers(self) -> None:
        """Allocate FSDP-specific buffers: per-layer shards + reusable collective buffers."""
        cfg = self.config
        ws = self.world_size
        max_exact_rank_fill = _MAX_EXACT_RANK_FILL[self.dtype]
        if ws > max_exact_rank_fill:
            raise ValueError(
                f"world_size={ws} exceeds the exact rank-fill capacity "
                f"({max_exact_rank_fill}) of {self.dtype}"
            )

        # Per-layer parameter shards (what each rank "owns")
        self.param_shards = [
            torch.empty(self.shard_size, dtype=self.dtype, device="cuda")
            for _ in range(self.num_layers)
        ]

        # Reusable all_gather output: full parameter = shard_size * world_size
        self.full_param = torch.empty(
            self.shard_size * ws, dtype=self.dtype, device="cuda"
        )

        # Reusable reduce_scatter correctness buffers. The model/all-gather/H2D
        # path stays at the requested workload dtype, but the synthetic
        # rank-coded SUM oracle must be exact. BF16 reduction order legitimately
        # produces ULP-sized variation at 24+ ranks and cannot be used as a
        # corruption oracle.
        self.full_grad = torch.empty(
            self.shard_size * ws,
            dtype=_REDUCE_SCATTER_ORACLE_DTYPE,
            device="cuda",
        )
        self.grad_shard = torch.empty(
            self.shard_size,
            dtype=_REDUCE_SCATTER_ORACLE_DTYPE,
            device="cuda",
        )

        # Per-layer GEMM weights (only if compute simulation is enabled)
        # FSDP mode manages its own compute because collectives are interleaved
        # per-layer, unlike the base compute simulator which runs all layers at once.
        if cfg.simulate_compute:
            dim = self._dim
            use_shared = (
                cfg.shared_layer_weights and cfg.compute_type == "transformer"
            )
            if cfg.compute_type == "transformer" and not cfg.shared_layer_weights:
                # Only the shared-weight transformer path is implemented; without
                # shared weights we fall back to GEMM. Warn loudly so this is not
                # a silent transformer->GEMM fallback (the thing this PR fixes).
                log.warning(
                    "race: compute_type='transformer' but shared_layer_weights=False "
                    "-- the non-shared transformer path is not implemented; running "
                    "the GEMM compute path instead."
                )
            if use_shared:
                # All layers share ONE real transformer block with deterministic,
                # rank-identical weights so block(reference_input) is analytically
                # identical for every layer and every rank. Any divergence across
                # layers indicates compute-path corruption.
                hidden = cfg.model_dim
                num_heads = cfg.num_heads or (hidden // 128)
                if num_heads < 1:
                    num_heads = 1
                ffn = cfg.ffn_size or (hidden * 4)
                if hidden % num_heads != 0:
                    raise ValueError(
                        f"model_dim ({hidden}) must be divisible by num_heads "
                        f"({num_heads}) for shared-weight transformer compute"
                    )
                # Record the resolved shape (num_heads/ffn may be auto-derived).
                self.eff_num_heads = num_heads
                self.eff_ffn_size = ffn
                self.eff_seq_len = cfg.seq_len
                self.eff_batch_size = cfg.batch_size
                block_cfg = BlockConfig(
                    hidden_size=hidden,
                    num_heads=num_heads,
                    num_layers=1,
                    ffn_size=ffn,
                    num_experts=1,
                    seq_len=cfg.seq_len,
                    vocab_size=16,  # embed is unused on this path; keep tiny
                )
                # fork_rng so we can fix the seed without perturbing global RNG.
                # RepeatedTransformerBlock initializes its params on CPU (nn.Linear
                # / LayerNorm use the CPU RNG) BEFORE .to("cuda"), so we must seed
                # the CPU RNG too -- seeding only CUDA would leave weights dependent
                # on each rank's CPU RNG state and break the rank-identical invariant
                # (every layer would still match within a rank, so the per-layer
                # checksum would falsely pass while ranks silently diverged).
                with torch.random.fork_rng(devices=["cuda"]):
                    torch.manual_seed(0)
                    torch.cuda.manual_seed(0)
                    self.shared_block = (
                        RepeatedTransformerBlock(block_cfg).to("cuda").to(self.dtype)
                    )
                self.shared_block.eval()

                # Fixed reference input, same seed across all ranks and iterations.
                g = torch.Generator(device="cuda")
                g.manual_seed(1)
                self.reference_input = torch.randn(
                    cfg.batch_size, cfg.seq_len, hidden,
                    dtype=self.dtype, device="cuda", generator=g,
                )
                self.weight_matrices = []
                self.layer_checksums = [None] * self.num_layers
                # activation/grad_buffer below are unused on the shared path
                # (forward sets activation to the block output; backward re-runs
                # the block), so skip those dim x dim allocations.
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

        # Startup line names the active compute path so a silent fallback
        # (e.g. transformer requested but GEMM ran) is greppable in logs.
        shared_active = self.shared_block is not None
        log.info(
            f"Allocated FSDP buffers: layers={self.num_layers}, "
            f"shard_size={self.shard_size}, "
            f"full_param_size={self.shard_size * ws}, "
            f"compute={'enabled' if cfg.simulate_compute else 'disabled'}, "
            f"compute_type={cfg.compute_type}, "
            f"shared_layer_weights={cfg.shared_layer_weights}, "
            f"transformer_block={'active' if shared_active else 'none'}, "
            f"layer_checksum_verify={'ON' if shared_active else 'OFF'}"
            + (
                f", resolved_shape=heads:{self.eff_num_heads} ffn:{self.eff_ffn_size} "
                f"seq:{self.eff_seq_len} batch:{self.eff_batch_size}"
                if shared_active else ""
            )
        )

    def _fill_patterns(self) -> None:
        """Fill buffers with known patterns for verification."""
        # Keep rank 0 nonzero so stale zero-filled memory cannot pass. Every
        # supported workload dtype represents these values injectively up to
        # the world-size guard in setup_buffers().
        for shard in self.param_shards:
            shard.fill_(float(self.rank + 1))

        # Each rank fills full_grad with rank + 1 (for reduce_scatter verification)
        self.full_grad.fill_(float(self.rank + 1))

    @staticmethod
    def _checksum(tensor: torch.Tensor) -> int:
        """
        Lightweight bit-pattern fingerprint.

        The int view matches the dtype's byte width and accumulation is int64,
        so floating-point rounding is avoided. This is not collision-free;
        communication correctness is established by exact rank-fill checks.
        The fingerprint remains useful for localizing compute divergence.
        """
        itemsize = tensor.element_size()
        int_view = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(itemsize)
        if int_view is None:
            raise ValueError(f"_checksum: unsupported element size {itemsize} bytes")
        return tensor.view(int_view).to(torch.int64).sum().item()

    @staticmethod
    def _mismatch_summary(
        actual: torch.Tensor,
        expected: torch.Tensor,
    ) -> dict[str, int | float]:
        """Return truthful details for an already-detected tensor mismatch."""
        mismatch = actual.ne(expected)
        flat_mismatch = mismatch.reshape(-1)
        mismatch_indices = flat_mismatch.nonzero(as_tuple=False)
        first_index = int(mismatch_indices[0].item())
        actual_flat = actual.reshape(-1)
        expected_flat = expected.reshape(-1)
        return {
            "first_mismatch_index": first_index,
            "actual": float(actual_flat[first_index].item()),
            "expected": float(expected_flat[first_index].item()),
            "mismatch_count": int(flat_mismatch.sum().item()),
            "max_abs_error": float(
                (actual.to(torch.float64) - expected.to(torch.float64))
                .abs()
                .max()
                .item()
            ),
        }

    def _forward_layer(self, layer_idx: int, iteration: int) -> None:
        """
        Forward pass for a single FSDP layer.

        1. all_gather: reconstruct full parameter from shards across ranks
        2. compute with the reconstructed parameter (if enabled)

        Shared-weight transformer path: every layer runs the same shared
        RepeatedTransformerBlock on the same fixed reference_input, so outputs are
        analytically identical.  Input/output checksums are recorded for both the
        comm kernel (all_gather) and the compute kernel (the transformer block) so
        _verify_layer_checksums() can pinpoint whether corruption entered during
        communication or compute.

        Chained path (default, GEMM): layer 0 seeds activation from batch_gpu (H2D
        race opportunity) and each subsequent layer receives the previous layer's
        output through a GEMM + GELU.
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

        if self.in_verification_phase and not self._verify_all_gather(
            iteration=iteration,
            layer_idx=layer_idx,
            phase="forward",
        ):
            self._iteration_collectives_correct = False

        if use_shared:
            comm_output_cksum = self._checksum(self.full_param)

        # ── compute kernel: transformer block (shared) or GEMM + GELU ──
        # Gate admits the shared-block path even though weight_matrices is now
        # empty on that path; the GEMM/chained else still requires weight_matrices.
        if self.config.simulate_compute and (
            self.shared_block is not None or self.weight_matrices
        ):
            if use_shared:
                compute_input_cksum = self._checksum(self.reference_input)

                with torch.no_grad():
                    out = self.shared_block(self.reference_input)

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

        Shared-transformer path: with config.real_backward=True, runs a genuine
        autograd backward (grad-enabled forward + loss.backward()) so real
        gradient kernels execute; grads are discarded. Otherwise re-runs the
        forward under no_grad as a backward timing proxy.
        """
        # all_gather: reconstruct full parameter for backward
        dist.all_gather_into_tensor(
            self.full_param, self.param_shards[layer_idx]
        )

        # Backward compute (if enabled)
        if self.config.simulate_compute and self.config.include_backward_compute:
            if self.shared_block is not None:
                if self.config.real_backward:
                    # Real backward: grad-enabled forward + loss.backward() so
                    # genuine gradient kernels run over the same shared block.
                    # Loss in fp32 to avoid bf16 underflow. Grads are discarded
                    # (not routed through reduce_scatter, no optimizer step) so
                    # the deterministic forward checksums and the rank-fill /
                    # reduce_scatter-sum invariants are untouched. requires_grad
                    # on the INPUT guarantees a real backward graph even if the
                    # block params were built with requires_grad=False.
                    ri = self.reference_input.detach().requires_grad_(True)
                    out = self.shared_block(ri)
                    # Accumulate the loss in fp32 (avoids bf16 underflow) WITHOUT
                    # materializing a full fp32 copy of the activation: sum(dtype=)
                    # casts inside the reduction.
                    loss = out.sum(dtype=torch.float32)
                    loss.backward()
                    self.shared_block.zero_grad(set_to_none=True)
                    # free graph references promptly (per-layer, so peak memory
                    # stays flat)
                    del out, loss, ri
                else:
                    # Shared-transformer path: re-run forward as a backward timing
                    # proxy (we don't train, so an exact bwd kernel isn't needed —
                    # only the comm/compute overlap timing matters here).
                    with torch.no_grad():
                        _ = self.shared_block(self.reference_input)
            elif self.weight_matrices:
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
        self._iteration_collectives_correct = True

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
        for layer_idx in range(self.num_layers):
            self._forward_layer(layer_idx, iteration)

        # ─── Backward: per-layer all_gather + GEMM bwd + reduce_scatter
        for layer_idx in reversed(range(self.num_layers)):
            self._backward_layer(layer_idx)

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
        for layer_idx in range(self.num_layers):
            self._forward_layer(layer_idx, iteration)

        # ─── Prefetch next batch (overlaps with backward) ────────────
        self._h2d_prefetch_next(iteration + 1)

        # ─── Backward: per-layer all_gather + GEMM bwd + reduce_scatter
        for layer_idx in reversed(range(self.num_layers)):
            self._backward_layer(layer_idx)

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
        all_correct = self._iteration_collectives_correct

        # Check H2D result
        if not self._verify_h2d(self.batch_gpu, iteration):
            all_correct = False

        # Check last all_gather result (full_param from last backward layer = layer 0)
        if not self._verify_all_gather(
            iteration=iteration,
            layer_idx=0,
            phase="backward_final",
        ):
            all_correct = False

        # Check last reduce_scatter result
        if not self._verify_reduce_scatter(iteration):
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
        Verify that per-kernel fingerprints agree with the modal layer value.

        With a shared transformer block and a fixed reference input every layer
        runs the same comm kernel (all_gather of rank-filled shard) and the same
        compute kernel (the shared RepeatedTransformerBlock on reference_input).
        Inputs and outputs are fingerprinted via a same-width integer view and
        int64 sum. This avoids floating-point rounding but is collision-prone;
        exact rank-fill validation separately establishes all-gather correctness.

        Four checksums per layer:
          comm_input    -- param shard before all_gather (should be identical:
                           every shard is filled with float(rank))
          comm_output   -- full_param after all_gather
          compute_input -- reference_input fed to the transformer block
                           (constant across layers)
          compute_output-- transformer block output

        Modal consensus prevents one bad layer 0 from being treated as the
        reference and mislabeling every agreeing later layer. If comm_output
        diverges but comm_input matches, corruption is in the collective path.
        If compute_output diverges but comm_output matches, corruption is in
        the compute path.
        """
        valid_layers = [
            (index, checksums)
            for index, checksums in enumerate(self.layer_checksums)
            if checksums is not None
        ]
        if len(valid_layers) < 2:
            return True

        self.layers_verified += len(valid_layers) - 1
        all_correct = True
        mismatched_layers: set[int] = set()
        for key in ("comm_input", "comm_output", "compute_input", "compute_output"):
            values = [checksums[key] for _, checksums in valid_layers]
            counts = Counter(values)
            reference_value, reference_count = counts.most_common(1)[0]
            if reference_count * 2 <= len(values):
                log.error(
                    f"LAYER_CHECKSUM_AMBIGUOUS ({key}): "
                    f"rank={self.rank} iter={iteration} groups={dict(counts)}"
                )
                self.corruption_details.append({
                    "type": f"layer_checksum_ambiguous_{key}",
                    "rank": self.rank,
                    "iteration": iteration,
                    "checksum_groups": dict(counts),
                })
                all_correct = False
                continue
            reference_index = next(
                index
                for index, checksums in valid_layers
                if checksums[key] == reference_value
            )
            for index, checksums in valid_layers:
                if checksums[key] != reference_value:
                    log.error(
                        f"LAYER_CHECKSUM_MISMATCH ({key}): "
                        f"rank={self.rank} iter={iteration} "
                        f"reference_layer={reference_index} "
                        f"reference={reference_value} "
                        f"layer_{index}={checksums[key]}"
                    )
                    self.corruption_details.append({
                        "type": f"layer_checksum_mismatch_{key}",
                        "rank": self.rank,
                        "iteration": iteration,
                        "layer_ref": reference_index,
                        "layer_cmp": index,
                        "ref_checksum": reference_value,
                        "cmp_checksum": checksums[key],
                    })
                    mismatched_layers.add(index)
                    all_correct = False

        self.layer_checksum_mismatches += len(mismatched_layers)

        return all_correct

    def _verify_all_gather(
        self,
        *,
        iteration: int,
        layer_idx: int,
        phase: str,
    ) -> bool:
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
            expected = float(src_rank + 1)
            expected_tensor = torch.full_like(chunk, expected)

            if not torch.equal(chunk, expected_tensor):
                summary = self._mismatch_summary(chunk, expected_tensor)
                log.error(
                    f"ALL_GATHER CORRUPTION (RUNTIME BUG!): "
                    f"rank={self.rank} src_rank={src_rank} "
                    f"iter={iteration} layer={layer_idx} phase={phase} "
                    f"expected={summary['expected']} actual={summary['actual']} "
                    f"index={summary['first_mismatch_index']} "
                    f"count={summary['mismatch_count']}"
                )
                self.corruption_details.append({
                    "type": "all_gather",
                    "rank": self.rank,
                    "src_rank": src_rank,
                    "iteration": iteration,
                    "layer": layer_idx,
                    "phase": phase,
                    **summary,
                })
                all_correct = False

        return all_correct

    def _verify_reduce_scatter(self, iteration: int) -> bool:
        """
        Verify reduce_scatter result.

        Each rank filled full_grad with float(rank + 1). After reduce_scatter
        with SUM, each rank's grad_shard should be sum(1..world_size).
        """
        expected = float(sum(range(1, self.world_size + 1)))
        expected_tensor = torch.full_like(self.grad_shard, expected)

        if not torch.equal(self.grad_shard, expected_tensor):
            summary = self._mismatch_summary(self.grad_shard, expected_tensor)
            log.error(
                f"REDUCE_SCATTER CORRUPTION (RUNTIME BUG!): "
                f"rank={self.rank} iter={iteration} "
                f"expected={summary['expected']} actual={summary['actual']} "
                f"index={summary['first_mismatch_index']} "
                f"count={summary['mismatch_count']}"
            )
            self.corruption_details.append({
                "type": "reduce_scatter",
                "rank": self.rank,
                "iteration": iteration,
                "oracle_dtype": self.reduce_scatter_oracle_dtype,
                **summary,
            })
            return False

        return True


__all__ = ["FSDPModeReproducer"]
