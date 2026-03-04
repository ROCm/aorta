# RCCL Runtime Race Condition Reproducer

## What This Tool Does

This is a standalone test that checks whether **RCCL/HIP has a runtime-level bug** that silently corrupts data during multi-GPU distributed training.

The key idea: the test uses **correct synchronization everywhere**. If data corruption still occurs, the bug is **in the runtime itself** (RCCL or HIP), not in application code.

### The Problem It Detects

In multi-GPU training, different operations run on different GPU streams in parallel:

1. **H2D** ("Host-to-Device"): Copies a batch of data from CPU memory to GPU memory on a dedicated `memcpy_stream`.
2. **datadist** ("data distribution"): Runs `all_to_all` collective communication (exchanging sparse embeddings across GPUs, TorchRec-style) on a dedicated `datadist_stream`.
3. **Compute + gradient sync**: Forward pass, backward pass, and `all_reduce` on the `default_stream`.

These streams are supposed to be safely coordinated via `wait_stream()` calls. But under certain HW queue configurations (`GPU_MAX_HW_QUEUES=4`), a runtime bug in RCCL/HIP can cause the synchronization to be violated internally, leading to **silent data corruption** -- data that arrives on the GPU is stale, partial, or wrong.

### How It Detects Corruption

Every iteration, the test:
1. Fills all buffers with **known values** (e.g., `batch = iteration % 1000`, `send_buf = rank`, `reduce_buf = rank + 1`).
2. Runs the full multi-stream pipeline with proper `wait_stream()` synchronization.
3. Checks that every buffer still has the expected value after the pipeline completes.

If `batch_gpu` was supposed to be `42.0` but contains `41.0`, that is proof of data corruption despite correct synchronization -- a runtime bug.

## Key Concepts

### H2D (Host-to-Device)

The operation that copies training data from CPU ("host") to GPU ("device"). In the code, this is `batch_gpu.copy_(batch_cpu, non_blocking=True)` running on a separate `memcpy_stream`. This is how real training pipelines overlap data loading with compute.

All modes support two H2D strategies, controlled by `--prefetch`:

| Strategy | Flag | How It Works | Where in Code |
|----------|------|-------------|---------------|
| **Single-buffered** | (default) | Copy current batch at start of each iteration, wait, then use it | `base.py` → `_h2d_transfer()` |
| **Double-buffered** | `--prefetch` | Prefetch next batch during current backward pass, swap buffers at end | `base.py` → `_h2d_prefetch_next()`, `_h2d_swap_buffers()` |

Single-buffered is simpler and tests a different timing profile. Double-buffered matches real DDP/FSDP training pipelines where data loading overlaps with compute.

### datadist (Data Distribution)

The `all_to_all` collective communication that simulates TorchRec's distributed embedding exchange. In real recommendation models, each GPU holds a shard of the embedding table, and `all_to_all` redistributes lookup results across GPUs. This runs on a separate `datadist_stream`.

Only the **default** mode uses datadist. DDP mode does not use `all_to_all` -- it uses gradient `all_reduce` instead.

| Mode | Communication Pattern | Where in Code |
|------|----------------------|---------------|
| **default** | `all_to_all` + `all_reduce` | `modes/default.py` → `_run_alltoall()`, `_run_allreduce()` |
| **ddp** | gradient `all_reduce` only | `modes/ddp.py` → `_gradient_allreduce()` |
| **fsdp** | per-layer `all_gather` + `reduce_scatter` | `modes/fsdp.py` → `_forward_layer()`, `_backward_layer()` |

### Warmup Iterations (`--warmup N`)

Run N iterations of the full pipeline **without checking for corruption**.

Why? Runtime bugs in RCCL/HIP are often timing-sensitive. They only manifest after the runtime has built up internal state (signal pools, caches, HW queue assignments) over many iterations. Running warmup gets the runtime into the "hot" state where bugs are more likely to appear. Without warmup, the runtime may still be in a cold/serial state that masks the bug.

During warmup, the test still runs `torch.cuda.synchronize()` each step so GPU work actually executes (not just queued).

### Verify Iterations (`--verify N`)

Run N iterations of the full pipeline **and check every buffer after each step**. This is where corruption is actually detected. More iterations = higher confidence. Use `--verify 10000` or more for thorough testing.

## Modes (`--mode`)

The `--mode` flag selects which workload pattern to test. Different distributed training frameworks have different communication patterns, and each may trigger the bug differently.

### default (TorchRec-like)

```
 memcpy_stream:   [fill batch_cpu → copy to batch_gpu]
                                                        │ wait_stream()
                                                        ▼
  default_stream:                                      [Forward(batch_gpu)] → [Backward] ──────→ [all_reduce]
                                                                                            ▲
  datadist_stream:                                                          [all_to_all] ──┘ wait_stream()
```

This is the default. It simulates a TorchRec recommendation model with:
- H2D: batch data copied to GPU on `memcpy_stream`
- datadist: `all_to_all` on `datadist_stream` (overlaps with backward)
- Compute: forward/backward GEMMs on `default_stream`
- all_reduce: gradient sync on `default_stream`

Verifies: H2D correctness, all_to_all correctness, all_reduce correctness.

```bash
torchrun --nproc_per_node=8 -m aorta.race --mode default \
    --warmup 100 --verify 10000
```

### ddp (Distributed Data Parallel)

```
Single-buffered (default):
    memcpy_stream:  [H2D] → batch_gpu
    default_stream:          [Forward] → [Backward] → [all_reduce grads]

Double-buffered (--prefetch):
    Iteration N:
        memcpy_stream:  [H2D batch_N+1 (prefetch)] ────────────────────┐
                                                                        │ overlap
        default_stream: [Forward(batch_N)] → [Backward] → [all_reduce grads]
                                                                        │
                        ← swap buffers ─────────────────────────────────┘
```

Simulates DDP training with:
- H2D: single-buffered (default) or double-buffered prefetch (`--prefetch`)
- Compute: forward/backward GEMMs with autograd
- Gradient all_reduce: averages actual computed gradients across ranks
- No `all_to_all` (DDP doesn't use it)

Supports two gradient sync strategies:
- **Non-bucketed** (default): One bulk all_reduce after all of backward finishes
- **Bucketed** (`--bucketed`): Per-layer all_reduce interleaved with backward (matches real PyTorch DDP)

```
Bucketed (--bucketed):
    default_stream: [Forward] → [Bwd L2 + AR L2] → [Bwd L1 + AR L1] → [Bwd L0 + AR L0]
```

Verifies: H2D correctness, gradient consistency across ranks.

```bash
# Single-buffered (default)
torchrun --nproc_per_node=8 -m aorta.race --mode ddp \
    --warmup 100 --verify 10000 --deterministic

# Double-buffered (prefetch overlaps with backward)
torchrun --nproc_per_node=8 -m aorta.race --mode ddp --prefetch \
    --warmup 100 --verify 10000 --deterministic

# Bucketed (per-layer backward + all_reduce overlap)
torchrun --nproc_per_node=8 -m aorta.race --mode ddp --bucketed \
    --warmup 100 --verify 10000 --deterministic
```

### fsdp (Fully Sharded Data Parallel)

```
memcpy_stream:   [H2D] ─────────────────────────────────────────────────────┐
                                                                             │ wait
default_stream:  [all_gather L0 → GEMM L0 → all_gather L1 → GEMM L1 → ...]│
                 [... → GEMM bwd L1 → reduce_scatter L1 →                   │
                        GEMM bwd L0 → reduce_scatter L0]                    │
                 [optimizer step]
```

Simulates FSDP training with:
- H2D: single-buffered (default) or double-buffered prefetch (`--prefetch`)
- Per-layer `all_gather`: reconstructs full parameters from shards before compute
- Per-layer `reduce_scatter`: shards gradients back across ranks after backward
- GEMMs interleaved with collectives (if compute enabled)
- No separate communication stream (all collectives on default stream)

Unlike default and DDP modes which use one or two bulk collectives, FSDP interleaves many small `all_gather`/`reduce_scatter` operations with per-layer compute. This creates a fundamentally different overlap and timing profile that may trigger different runtime bugs.

Verifies: H2D correctness, all_gather correctness, reduce_scatter correctness.

```bash
# Single-buffered
torchrun --nproc_per_node=8 -m aorta.race --mode fsdp \
    --warmup 100 --verify 10000

# Double-buffered (prefetch overlaps with backward)
torchrun --nproc_per_node=8 -m aorta.race --mode fsdp --prefetch \
    --warmup 100 --verify 10000

# Custom shard size
torchrun --nproc_per_node=8 -m aorta.race --mode fsdp \
    --fsdp-shard-size 200000 --warmup 100 --verify 10000
```

## Quick Start

```bash
# Default mode (TorchRec-like) — most common test
GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 -m aorta.race \
    --warmup 10 --verify 100

# Default mode with double-buffered H2D prefetch
GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 -m aorta.race --prefetch \
    --warmup 10 --verify 100

# DDP mode (gradient all_reduce pattern)
torchrun --nproc_per_node=8 -m aorta.race --mode ddp \
    --warmup 100 --verify 10000 --deterministic

# FSDP mode (per-layer all_gather + reduce_scatter)
torchrun --nproc_per_node=8 -m aorta.race --mode fsdp \
    --warmup 10 --verify 100

# Same-stream mode (strongest proof of runtime bug)
torchrun --nproc_per_node=8 -m aorta.race --same-stream

# Multi-node (via launch script)
./scripts/multi_node/launch_reproducer.sh \
    --docker <container-name> \
    --hw-queues 4 \
    --warmup 100 \
    --verify 10000
```

## Test Configurations

| Test | Command | What It Does |
|------|---------|--------------|
| **Baseline** | `--hw-queues 4` | Full HW queue parallelism -- most likely to trigger the bug |
| **Serialized** | `--hw-queues 2` | Reduced parallelism -- if bug disappears, it's parallelism-related |
| **Same-Stream** | `--same-stream` | H2D + datadist on same stream. Corruption here = definitive runtime bug |
| **No Compute** | `--no-compute` | Skip GEMM simulation (~5ms/step). Fast iteration but may not hit timing window |
| **H2D Prefetch** | `--prefetch` | Double-buffered H2D overlapping with backward (works with any mode) |
| **DDP Mode** | `--mode ddp` | Tests gradient all_reduce pattern (different comm pattern) |
| **DDP Bucketed** | `--mode ddp --bucketed` | Per-layer backward + all_reduce overlap (real DDP pattern) |
| **FSDP Mode** | `--mode fsdp` | Tests per-layer all_gather + reduce_scatter (many small collectives) |
| **NCCL Implicit** | `--nccl-implicit-order` | Serialize NCCL ops via `NCCL_LAUNCH_ORDER_IMPLICIT=1` |

## Command-Line Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode MODE` | `default` | Workload mode: `default`, `ddp`, `fsdp` |
| `--warmup N` | 100 | Warmup iterations (runs pipeline, skips corruption checks) |
| `--verify N` | 10000 | Verification iterations (runs pipeline and checks for corruption) |
| `--no-compute` | - | Skip GEMM compute simulation (faster but less realistic timing) |
| `--same-stream` | - | Put H2D and datadist on same GPU stream |
| `--no-stop-on-first` | - | Continue running after first corruption (count total) |
| `--gemm-size N` | 5120 | GEMM matrix size (controls compute duration) |
| `--gemm-layers N` | 26 | Number of GEMM layers (controls compute duration) |
| `--optimizer OPT` | `none` | Optimizer: `none`, `adamw`, `sgd`, `shampoo` (for DDP mode) |
| `--deterministic` | - | Fixed seeds for cross-rank gradient verification (for DDP mode) |
| `--bucketed` | - | Per-layer gradient all_reduce overlapping with backward (for DDP mode) |
| `--fsdp-shard-size N` | 100000 | FSDP shard size per rank (for FSDP mode) |

### Environment Variable Flags

| Flag | Env Variable | Effect |
|------|--------------|--------|
| `--hw-queues N` | `GPU_MAX_HW_QUEUES=N` | Control HW queue count (4 = exposes bug, 2 = masks it) |
| `--nccl-implicit-order` | `NCCL_LAUNCH_ORDER_IMPLICIT=1` | Serialize NCCL ops |
| `--disable-sdma` | `HSA_ENABLE_SDMA=0` | Disable SDMA engine |
| `--signal-pool-size N` | `ROC_SIGNAL_POOL_SIZE=N` | HSA signal pool size |
| `--disable-cheap-fence` | `RCCL_GFX9_CHEAP_FENCE_OFF=1` | Disable fence optimization |

## Output

### Pass
```
PASSED: No corruption in 10100 iterations with proper synchronization
VERDICT: No runtime bug detected with current settings.
```

### Fail (Runtime Bug Detected)
```
RUNTIME BUG DETECTED: 15 corruptions in 5432 iterations
Corruption occurred DESPITE proper synchronization - this is a bug in RCCL/HIP runtime
VERDICT: RUNTIME BUG DETECTED!
```

## Interpreting Results

| Baseline (HW=4) | Serialized (HW=2) | Same-Stream | Conclusion |
|-----------------|-------------------|-------------|------------|
| Fail | Pass | Pass | Runtime bug triggered by HW queue parallelism |
| Fail | Pass | Fail | Runtime bug in stream ordering itself |
| Pass | Pass | Pass | No runtime bug detected |
| Fail | Fail | Fail | Possible hardware issue |

## Adding a New Mode

To test a new workload pattern, create a new mode file. Each mode controls its own H2D strategy, communication pattern, and verification checks. See `modes/fsdp.py` for a complete example.

1. Create `modes/your_mode.py` inheriting from `BaseReproducer`
2. Implement `setup_buffers()` and `run_iteration()`
3. Register in `modes/__init__.py`
4. Add `--mode your_mode` to CLI choices in `__main__.py`

See `developer_guide.md` for the full walkthrough.

## References

- **Background & Concepts:** `src/aorta/race/background.md` -- Detailed explanation of distributed training patterns, collectives, streams, and the race condition bug
- **Config Reference:** `config/race/`
- **Environment Variables:** `config/race/env_vars_reference.yaml`
- **Multi-node Scripts:** `scripts/multi_node/launch_reproducer.sh`
- **Developer Guide:** `src/aorta/race/developer_guide.md`
