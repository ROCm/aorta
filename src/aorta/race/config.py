"""
Race experiment configuration dataclasses.

This module defines:
- ReproducerConfig: Settings for the standalone RCCL race condition reproducer
- ReproducerResult: Result from running the reproducer
- RaceConfig: Settings for race condition injection experiments (broader aorta system)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Standalone Reproducer Config
# =============================================================================


@dataclass
class ReproducerConfig:
    """
    Configuration for the standalone RCCL race condition reproducer.

    This reproducer tests for RUNTIME bugs (not application bugs) by:
    - Using proper synchronization everywhere
    - Using known-pattern data to detect any corruption
    - Simulating training timing profile

    If corruption occurs with proper syncs, it's a RUNTIME BUG in RCCL/HIP.
    """

    # =========================================================================
    # Mode selection
    # =========================================================================
    mode: str = "default"
    """
    Reproducer mode. Determines the communication pattern tested.

    Available modes:
    - "default": TorchRec-like pattern (H2D + all_to_all + all_reduce)
    - "ddp": DDP pattern (H2D with double-buffered prefetch + gradient all_reduce)
    - "minimal": Legacy monolithic reproducer (backward compat)
    """

    # =========================================================================
    # Iteration settings
    # =========================================================================
    warmup_iterations: int = 100
    """
    Number of iterations to run WITHOUT verification.

    Builds up RCCL/runtime state before checking for corruption.
    Runtime bugs may manifest after ~100+ steps due to internal state
    accumulation. Warmup helps reach similar conditions.
    """

    verify_iterations: int = 10000
    """
    Number of iterations to run WITH verification.

    After warmup, we check every iteration for corruption using
    known-pattern data verification.
    """

    stop_on_first_corruption: bool = True
    """Stop immediately when corruption is detected."""

    log_interval: int = 100
    """Log progress every N iterations."""

    # =========================================================================
    # Tensor sizes (stress factors)
    # =========================================================================
    h2d_tensor_size: int = 1_000_000
    """
    Size of H2D tensor (number of elements).

    Larger = longer DMA = more overlap opportunity.
    Recommended: 1M+ for stress testing.
    """

    alltoall_tensor_size: int = 100_000
    """
    Size of all_to_all tensor per rank (number of elements).

    Total transfer size = alltoall_tensor_size * world_size.
    Larger = longer collective = more overlap opportunity.
    Used by default mode only.
    """

    allreduce_tensor_size: int = 100_000
    """
    Size of all_reduce tensor (number of elements).

    Simulates gradient synchronization (e.g., FSDP-style).
    Used by default mode only; DDP mode all-reduces actual gradients.
    """

    fsdp_shard_size: int = 100_000
    """
    Size of each FSDP parameter shard per rank (number of elements).

    Used by FSDP mode for per-layer all_gather/reduce_scatter.
    Full parameter size = fsdp_shard_size * world_size.
    Larger = longer collectives = more overlap opportunity.
    """

    dtype: str = "bfloat16"
    """Data type for tensors. Options: bfloat16, float16, float32."""

    # =========================================================================
    # Compute simulation (match training timing)
    # =========================================================================
    simulate_compute: bool = True
    """
    Add GEMM work between collectives to match training timing.

    Without this, the reproducer runs ~100x faster than real training,
    which may not trigger the same timing conditions.
    """

    compute_type: str = "gemm"
    """Compute pattern type. Options: gemm, transformer."""

    # ----- GEMM compute settings -----
    gemm_size: int = 5120
    """
    Matrix size NxN for GEMM compute type.

    5120x5120 GEMM takes ~14ms on MI300X.
    Adjust based on GPU to achieve ~500ms/step target.
    """

    gemm_layers: int = 26
    """
    Number of GEMM layers for GEMM compute type.

    26 layers x 14ms = ~364ms per forward/backward pass.
    """

    # ----- Transformer compute settings -----
    model_dim: int = 2048
    """
    Hidden dimension (d_model) for transformer compute type.

    Controls the size of the self-attention and FFN layers.
    Number of attention heads = model_dim // 128.
    FFN intermediate size = model_dim * 4.
    """

    num_layers: int = 4
    """
    Number of transformer encoder layers for transformer compute type.

    Each layer contains multi-head self-attention + FFN + layer norm.
    4 layers at model_dim=2048 gives ~300ms/step on MI350X.
    """

    include_backward_compute: bool = True
    """Also simulate backward pass (doubles compute time)."""

    shared_layer_weights: bool = False
    """
    Use a single shared weight matrix for all layers (transformer compute type).

    When True, all num_layers layers use the same weight matrix initialized with
    a fixed seed, and each layer receives the same fixed reference input rather
    than chaining activations.  This makes per-layer forward outputs analytically
    identical so that cross-layer activation comparison can serve as a secondary
    corruption signal: any all_gather corruption that propagates through GEMM
    compute will produce a mismatch between layer outputs.

    Only meaningful when compute_type == 'transformer'.
    """

    # =========================================================================
    # Optimizer (used by modes that support it, e.g., DDP)
    # =========================================================================
    optimizer: str = "none"
    """
    Optimizer for weight updates. Options: none, adamw, sgd, shampoo.

    When set to 'none', no optimizer step is performed.
    DDP mode uses this to test gradient all_reduce + optimizer interaction.
    """

    optimizer_lr: float = 1e-4
    """Learning rate."""

    optimizer_weight_decay: float = 0.01
    """Weight decay."""

    optimizer_betas: Tuple[float, float] = (0.9, 0.999)
    """Adam betas."""

    optimizer_eps: float = 1e-8
    """Adam epsilon."""

    # =========================================================================
    # Deterministic mode (used by DDP for cross-rank gradient verification)
    # =========================================================================
    deterministic: bool = False
    """
    Enable deterministic mode with fixed seeds.

    Required for DDP gradient consistency verification across ranks.
    """

    deterministic_seed: int = 42
    """Seed for deterministic mode."""

    ddp_bucketed: bool = False
    """
    Use bucketed gradient all_reduce overlapping with backward.

    When enabled, per-layer gradient all_reduce runs concurrently with
    backward computation of earlier layers (real DDP behavior).
    When disabled, one bulk all_reduce runs after all of backward finishes.

    Only used by DDP mode.
    """

    # =========================================================================
    # Buffer management
    # =========================================================================
    reuse_buffers: bool = True
    """
    Reuse tensor buffers across iterations.

    Real training reuses buffers; fresh allocations each iteration
    may change memory layout and timing.
    """

    pin_memory: bool = True
    """Use pinned memory for H2D source tensors."""

    h2d_prefetch: bool = False
    """
    Use double-buffered H2D with prefetch.

    When enabled, the next batch is copied to GPU during the current iteration's
    backward pass, overlapping H2D with compute. When disabled, H2D is a blocking
    transfer at the start of each iteration.

    Double-buffered is the standard pattern in real DDP/FSDP training pipelines.
    Single-buffered is simpler and tests a different timing profile.
    """

    # =========================================================================
    # Stream configuration
    # =========================================================================
    same_stream_mode: bool = False
    """
    Put H2D and datadist on the SAME new stream (not separate streams).

    If corruption occurs in this mode, it's DEFINITIVE proof of a
    runtime bug because operations on the same stream are guaranteed
    to be ordered by CUDA/HIP specification.
    """

    # =========================================================================
    # Environment variables
    # =========================================================================
    gpu_max_hw_queues: Optional[int] = 4
    """
    Set GPU_MAX_HW_QUEUES environment variable.

    - 2: Reduced parallelism (may mask bugs)
    - 4+: Full parallelism (exposes timing-sensitive bugs)
    """


@dataclass
class ReproducerResult:
    """Result from running the reproducer."""

    passed: bool
    """True if no corruption was detected."""

    total_iterations: int
    """Total iterations run (warmup + verify)."""

    corruption_count: int
    """Number of corruptions detected."""

    first_corruption_iter: Optional[int]
    """Iteration where first corruption was detected (None if passed)."""

    corruption_details: List[Dict]
    """Details of each corruption detected."""

    elapsed_time_sec: float
    """Total elapsed time in seconds."""

    avg_step_time_ms: float
    """Average time per step in milliseconds."""


# =============================================================================
# Race Injection Config (broader aorta system)
# =============================================================================


@dataclass
class RaceConfig:
    """
    Configuration for race condition injection experiments.

    Four categories of race conditions are supported:

    1. H2D Race (realistic pattern):
       - h2d_memcpy_racing: Uses separate stream for H2D batch copy
       - h2d_skip_sync_before_forward: Skips wait_stream() before forward (causes race!)
       - h2d_racing_start_step: Step to start H2D racing

    2. Datadist Race (TorchRec-style all_to_all):
       - datadist_racing: Uses separate stream for all_to_all operations
       - datadist_skip_sync_before_collective: Skips wait_stream() before FSDP collective (causes race!)
       - datadist_racing_start_step: Step to start datadist racing

    3. Timing Skew Experiment (demonstrates NaN progression):
       - timing_skew_enabled: Enable controlled timing skew
       - timing_skew_mode: none, fixed, progressive, random
       - timing_skew_us: Delay in microseconds
       - timing_skew_ranks: Which ranks get delayed
       - timing_skew_start_step: Step to start skew

    Supporting options:
       - skip_training_warmup: Skip training warmup to maximize timing variability
       - skip_rccl_warmup: Skip RCCL communicator warmup before FSDP init
       - nan_check_collectives: Enable NaN checking around RCCL collectives
       - gpu_max_hw_queues: Set GPU_MAX_HW_QUEUES (4+ needed to expose race)
    """

    # =========================================================================
    # H2D memcpy racing (realistic pattern)
    # =========================================================================
    h2d_memcpy_racing: bool = False
    """Use separate memcpy_stream for H2D batch copy."""

    h2d_skip_sync_before_forward: bool = False
    """Skip wait_stream() before forward pass - THIS CAUSES THE RACE!"""

    h2d_racing_start_step: int = 0
    """Step to start H2D racing (0 = aggressive, from first step)."""

    # =========================================================================
    # Datadist racing (TorchRec-style all_to_all on separate stream)
    # =========================================================================
    datadist_racing: bool = False
    """Use separate datadist_stream for all_to_all operations."""

    datadist_skip_sync_before_collective: bool = False
    """Skip wait_stream() before FSDP collective - THIS CAUSES THE RACE!"""

    datadist_racing_start_step: int = 0
    """Step to start datadist racing (0 = aggressive, from first step)."""

    # =========================================================================
    # Supporting options
    # =========================================================================
    skip_training_warmup: bool = False
    """Skip training warmup to maximize timing variability for race testing."""

    training_warmup_steps: int = 1
    """Number of training warmup steps to run (if not skipped)."""

    warmup_batch_size: Optional[int] = None
    """
    Batch size for warmup steps. If None, uses the training batch_size.
    Set this smaller than training batch_size to speed up warmup while
    still exercising the collectives, then use larger batch during racing
    for wider race windows.
    """

    skip_rccl_warmup: bool = False
    """Skip RCCL communicator warmup before FSDP init to test race conditions."""

    rccl_warmup_iterations: int = 10
    """Number of RCCL warmup iterations (if not skipped). Higher = more stable but slower startup."""

    nan_check_collectives: bool = False
    """Enable NaN checking before/after RCCL collectives."""

    gpu_max_hw_queues: Optional[int] = None
    """
    Set GPU_MAX_HW_QUEUES environment variable.

    CRITICAL for race exposure:
    - 1-2: Streams share HW queues → implicit serialization → RACE MASKED
    - 4+: Each stream gets own HW queue → true parallelism → RACE EXPOSED

    Recommended: 4 for race testing, or 3 with torchrec_stream_layout.
    """

    torchrec_stream_layout: bool = False
    """
    Use TorchRec-style stream layout for accurate reproduction of the NaN issue.

    When enabled:
    - 3 streams only: default_stream, memcpy_stream, datadist_stream
    - Forward/backward/clip/optimizer/FSDP collectives run on default stream
    - Only H2D and datadist operations on separate racing streams
    - DistributedOpsInterceptor is bypassed (no stream redirection for collectives)

    This matches a typical TorchRec-style architecture where:
    - memcpy_stream races with default stream (H2D not synced before forward)
    - datadist_stream races with default stream (all_to_all not synced before collective)

    Recommended: Use with gpu_max_hw_queues: 3 for 1:1 stream-to-queue mapping.
    """

    # =========================================================================
    # Timing skew experiment (demonstrates NaN progression)
    # =========================================================================
    timing_skew_enabled: bool = False
    """Enable controlled timing skew experiment to show NaN progression."""

    timing_skew_mode: str = "none"
    """
    Skew mode:
    - none: No artificial skew
    - fixed: Fixed delay in microseconds
    - progressive: Increase delay each step (skew_us * step)
    - random: Random delay within range
    """

    timing_skew_us: int = 0
    """Base delay in microseconds (for fixed/progressive modes)."""

    timing_skew_ranks: List[int] = field(default_factory=list)
    """Which ranks get delayed. Empty = all ranks."""

    timing_skew_start_step: int = 3
    """Step to start introducing timing skew (after warmup)."""

    # =========================================================================
    # Debug instrumentation
    # =========================================================================
    timing_debug_logs: bool = False
    """
    Enable detailed timing logs around H2D, datadist, and forward operations.

    When enabled, logs wall-clock timestamps for:
    - H2D copy (memcpy_stream operations)
    - Datadist racing (all_to_all operations)
    - Timing skew delay
    - Forward pass

    Also logs gap/overlap between operations to verify if race windows exist.
    Negative gap = overlap = potential race.
    """

    gpu_event_timing: bool = False
    """
    Enable GPU event-based timing to measure actual stream overlap.

    Unlike timing_debug_logs which measures CPU-side timestamps, this uses
    CUDA/HIP events to measure actual GPU execution times and overlap.

    Event recording is non-blocking and doesn't interfere with race conditions.
    Timing calculations happen AFTER the iteration completes (when we sync anyway).

    Logs GPU-side durations and overlap between streams:
    - GPU_H2D_DUR: Actual GPU time for H2D copy
    - GPU_DD_DUR: Actual GPU time for datadist all_to_all
    - GPU_FWD_DUR: Actual GPU time for forward pass
    - GPU_OVERLAP: Time between stream operations (negative = overlap = race!)

    WARNING: Cross-stream timing (e.g., DD->FWD) is INACCURATE because events
    are on different streams. Use nccl_async_diagnostic for accurate overlap detection.
    """

    nccl_async_diagnostic: bool = False
    """
    Enable NCCL async behavior diagnostic.

    When enabled, checks if the datadist all_to_all work is still in-flight
    when forward pass starts. This uses work.is_completed() which is non-blocking
    and does NOT affect the race condition.

    Logs:
    - NCCL_DIAG: Whether all_to_all is PENDING (async, race possible) or
      COMPLETED (sync, race may be masked by NCCL internal sync)

    This is the authoritative way to verify if a race window exists, because
    it directly queries the NCCL work handle status rather than relying on
    GPU event timing which is inaccurate for cross-stream measurements.
    """

    datadist_use_real_dependency: bool = True
    """
    Make datadist output actually used by forward pass.

    When enabled, the all_to_all output tensor is added as noise to batch["dense"],
    creating a real data dependency that forward must read. This more accurately
    reproduces a TorchRec pattern where distributed embeddings race
    with the forward pass.

    Without this, datadist creates synthetic tensors that are discarded,
    so there's no actual data race with forward (just stream contention).
    """

    # =========================================================================
    # Controlled in-flight reads (no NaN poisoning)
    # =========================================================================
    h2d_split_dense_copy: bool = False
    """
    Copy the dense tensor in two chunks on the memcpy stream.

    The tail chunk is copied last to increase the chance it is still in-flight
    when forward starts. This helps ensure the forward reads a tensor that is
    actively being copied without injecting NaNs.
    """

    h2d_dense_tail_fraction: float = 0.5
    """
    Fraction of dense elements copied in the tail chunk (copied last).
    Clamped to (0, 1) and adjusted to keep at least one element in each chunk.
    """

    datadist_split_alltoall: bool = False
    """
    Split the all_to_all into two sequential collectives on the datadist stream.

    The tail chunk collective is launched last, so reading that tail immediately
    on the default stream is guaranteed to be in-flight.
    """

    datadist_tail_fraction: float = 0.5
    """
    Fraction of all_to_all elements assigned to the tail chunk (launched last).
    Clamped to (0, 1) and adjusted to be divisible by world_size.
    """

    datadist_read_tail_only: bool = False
    """
    Use only the tail chunk of the all_to_all output when creating the
    datadist->forward dependency. This guarantees the read targets the
    in-flight chunk without NaN poisoning.
    """

    # =========================================================================
    # In-flight read instability checks
    # =========================================================================
    inflight_read_check_enabled: bool = False
    """
    Enable repeated in-flight reads to detect instability/mismatch.

    When enabled, the tail region of in-flight tensors is read multiple times
    on the default stream while the racing stream is still writing. If the
    values change between reads, it indicates a torn read (race condition).
    """

    inflight_read_repeats: int = 0
    """
    Number of repeated reads to perform on the in-flight tail region.

    Set to 0 to disable repeated reads. Higher values increase the chance
    of detecting instability but add more GPU work on the default stream.
    Recommended: 3-10 for typical detection.
    """

    inflight_read_sample_size: int = 4096
    """
    Number of elements to sample from the tail region for instability checks.

    A smaller sample reduces GPU work; a larger sample increases detection
    probability. The sample is taken from the start of the tail region.
    """

    inflight_read_delay_size: int = 65536
    """
    Size of dummy tensor used for GPU-side delay work between reads.

    Between each repeated read, we issue GPU work (D2D copy + reduction) to
    spread reads across the race window. Larger values = more delay = higher
    detection probability but more GPU overhead.

    Recommended: 16384-131072 for typical workloads.
    """

    datadist_tensor_size: Optional[int] = 1_000_000
    """
    Fixed tensor size for datadist all_to_all operation.

    If set, uses this fixed size instead of basing it on dense.numel().
    This allows controlling the all_to_all duration independently of dense_dim.

    Recommended: 1M-10M elements for ~100-500ms all_to_all duration.
    Set to None to use dense.numel() (original behavior, but can be very slow
    with large dense_dim).
    """

    def is_any_race_enabled(self) -> bool:
        """Check if any race injection is enabled."""
        return (
            self.h2d_memcpy_racing
            or self.datadist_racing
            or self.timing_skew_enabled
        )

    def is_h2d_race_enabled(self) -> bool:
        """Check if H2D racing is enabled."""
        return self.h2d_memcpy_racing and self.h2d_skip_sync_before_forward

    def is_datadist_race_enabled(self) -> bool:
        """Check if datadist racing is enabled."""
        return self.datadist_racing and self.datadist_skip_sync_before_collective

    def is_timing_skew_active(self, step: int) -> bool:
        """Check if timing skew should be applied at this step."""
        return (
            self.timing_skew_enabled
            and self.timing_skew_mode != "none"
            and step >= self.timing_skew_start_step
        )
