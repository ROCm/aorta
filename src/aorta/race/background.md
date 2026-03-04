# Background: Distributed Training Patterns & Communication

This document explains the distributed training concepts behind each reproducer mode. It covers what happens in real training frameworks and how the reproducer simulates those patterns.

---

## Table of Contents

1. [GPU Programming Model](#1-gpu-programming-model)
2. [Communication Collectives](#2-communication-collectives)
3. [TorchRec Pattern (Default Mode)](#3-torchrec-pattern-default-mode)
4. [DDP Pattern](#4-ddp-pattern)
5. [FSDP Pattern](#5-fsdp-pattern)
6. [H2D Strategies](#6-h2d-strategies)
7. [The Race Condition Bug](#7-the-race-condition-bug)

---

## 1. GPU Programming Model

### GPU Streams

A **stream** is a sequence of GPU operations that execute in order. Operations on the *same* stream are sequential; operations on *different* streams can run concurrently:

```
Stream A:  [op1] → [op2] → [op3]       ← sequential within stream
Stream B:  [op4] → [op5]               ← sequential within stream
                                        ← A and B run concurrently
```

Real training uses multiple streams to overlap different kinds of work:

| Stream | Purpose | Typical Operations |
|--------|---------|-------------------|
| `default_stream` | Compute + some collectives | Forward GEMMs, backward GEMMs, all_reduce |
| `memcpy_stream` | Data loading | Host-to-Device copies (H2D) |
| `datadist_stream` | Data distribution (TorchRec) | all_to_all for embedding exchange |

### Stream Synchronization

Streams are coordinated with `wait_stream()`:

```python
# "default_stream, don't start until memcpy_stream finishes its current work"
default_stream.wait_stream(memcpy_stream)
```

This creates a **dependency edge** -- it does NOT block the CPU, it tells the GPU scheduler: "don't launch anything new on default_stream until memcpy_stream's queued work is done."

```
memcpy_stream:   [H2D copy ...........]
                                       │
                                       ▼ wait_stream()
default_stream:                        [Forward ........]
```

### Hardware Queues (`GPU_MAX_HW_QUEUES`)

The GPU has a limited number of **hardware dispatch queues**. Multiple GPU streams are multiplexed onto these queues:

```
                         ┌─── HW Queue 0 ──→ GPU compute units
GPU Streams  ───────────►├─── HW Queue 1 ──→ GPU compute units
(software)               ├─── HW Queue 2 ──→ GPU compute units
                         └─── HW Queue 3 ──→ GPU compute units
```

- `GPU_MAX_HW_QUEUES=4`: Full parallelism. Multiple streams dispatch to different HW queues simultaneously. **This is where the RCCL race bug manifests** -- the runtime's internal synchronization between queues can break.
- `GPU_MAX_HW_QUEUES=2`: Reduced parallelism. Streams share fewer queues, making overlaps less aggressive. Often masks the bug.

---

## 2. Communication Collectives

Collectives are operations where **all GPUs participate** to exchange or combine data. Each has a different communication pattern:

### all_to_all

Every GPU sends a different piece of data to every other GPU. It's a **personalized exchange**.

```
Before all_to_all:                After all_to_all:
  GPU 0: [A0 A1 A2 A3]            GPU 0: [A0 B0 C0 D0]
  GPU 1: [B0 B1 B2 B3]     →      GPU 1: [A1 B1 C1 D1]
  GPU 2: [C0 C1 C2 C3]            GPU 2: [A2 B2 C2 D2]
  GPU 3: [D0 D1 D2 D3]            GPU 3: [A3 B3 C3 D3]
```

Each GPU i sends chunk j to GPU j, and receives chunk j from GPU j. Used by **TorchRec** for redistributing embedding lookups.

### all_reduce

Every GPU contributes a tensor. The result (e.g., SUM) is placed on **all** GPUs.

```
Before all_reduce (SUM):          After all_reduce:
  GPU 0: [1, 1, 1]                 GPU 0: [10, 10, 10]
  GPU 1: [2, 2, 2]          →      GPU 1: [10, 10, 10]
  GPU 2: [3, 3, 3]                 GPU 2: [10, 10, 10]
  GPU 3: [4, 4, 4]                 GPU 3: [10, 10, 10]
```

All GPUs get the same result. Used by **DDP** and **default mode** for gradient synchronization.

### all_gather

Every GPU contributes a shard. The full concatenated result is placed on **all** GPUs.

```
Before all_gather:                After all_gather:
  GPU 0: [A]                       GPU 0: [A B C D]
  GPU 1: [B]              →        GPU 1: [A B C D]
  GPU 2: [C]                       GPU 2: [A B C D]
  GPU 3: [D]                       GPU 3: [A B C D]
```

Used by **FSDP** to reconstruct full parameter tensors from shards before compute.

### reduce_scatter

The inverse of all_gather. Every GPU contributes a full tensor. The SUM is computed and **scattered** -- each GPU gets a different shard of the result.

```
Before reduce_scatter (SUM):      After reduce_scatter:
  GPU 0: [1,2,3,4]                 GPU 0: [10]  (sum of position 0)
  GPU 1: [2,3,4,5]          →      GPU 1: [14]  (sum of position 1)
  GPU 2: [3,4,5,6]                 GPU 2: [18]  (sum of position 2)
  GPU 3: [4,5,6,7]                 GPU 3: [22]  (sum of position 3)
```

Used by **FSDP** to shard gradients back across GPUs after backward.

### Summary Table

| Collective | Input per GPU | Output per GPU | Used By |
|-----------|---------------|----------------|---------|
| `all_to_all` | Different chunk for each dest | Different chunk from each src | TorchRec (default mode) |
| `all_reduce` | Full tensor | Full tensor (summed) | DDP, default mode |
| `all_gather` | One shard | Full tensor (concatenated) | FSDP |
| `reduce_scatter` | Full tensor | One shard (summed) | FSDP |

---

## 3. TorchRec Pattern (Default Mode)

### What TorchRec Does

TorchRec is Meta's framework for recommendation models (e.g., DLRM). These models have two kinds of parameters:

1. **Embedding tables** (sparse, huge): Sharded across GPUs. Each GPU holds a slice of the table.
2. **Dense layers** (MLP, interaction): Replicated on every GPU.

A training iteration looks like:

```
1. Load batch from CPU → GPU                       (H2D)
2. Each GPU looks up embeddings in its local shard
3. Exchange embedding results across GPUs           (all_to_all)
4. Forward pass through dense layers                (GEMMs)
5. Backward pass                                    (GEMMs)
6. Synchronize dense gradients                      (all_reduce)
7. Optimizer step
```

The key insight: steps 3-5 can be **overlapped**. The `all_to_all` can run on a separate stream while backward runs on the default stream.

### Three-Stream Pipeline

```
memcpy_stream:    [H2D: copy batch to GPU] ─────────────────────────────────┐
                                                                             │ wait_stream()
                                                                             ▼
default_stream:                             [Forward: GEMMs read batch_gpu]
                                            [Backward: gradient GEMMs     ] ──────┐
                                                                                   │
datadist_stream:                            [all_to_all: exchange embeddings]      │
                                                                             │     │
                                            ┌──── wait_stream() ◄────────────┘     │
                                            ▼                                      ▼
default_stream (cont):                     [all_reduce: sync dense gradients]
                                           [Optimizer step                  ]
```

**Why three streams?** To maximize GPU utilization:
- While backward computes gradients (compute-bound), all_to_all exchanges embeddings (network-bound). These use different hardware resources, so overlapping them is free performance.
- H2D uses the DMA engine, which is independent of both compute and network.

### What the Reproducer Simulates

| Real TorchRec | Reproducer |
|---------------|-----------|
| Load batch from DataLoader | Fill `batch_cpu` with known pattern (`iteration % 1000`) |
| Embedding lookup + all_to_all | Fill `send_buf` with `float(rank)`, run all_to_all |
| Forward/backward through MLP | GEMMs of configurable size (`--gemm-size`, `--gemm-layers`) |
| Gradient all_reduce | Fill `reduce_buf` with `float(rank + 1)`, run all_reduce |

The reproducer uses **synthetic data with known patterns** instead of real model computations. This is intentional -- known patterns make corruption trivially detectable (just check if the value matches), while the timing profile (stream overlaps, collective sizes, compute duration) matches real training.

---

## 4. DDP Pattern

### What DDP Does

Distributed Data Parallel (DDP) is PyTorch's standard data-parallel training. Every GPU has a **full copy** of the model. Each GPU processes a different mini-batch, computes gradients, and then **averages gradients** across all GPUs before the optimizer step:

```
1. Load batch from CPU → GPU                       (H2D)
2. Forward pass (each GPU, different data)          (GEMMs)
3. Backward pass (each GPU, compute gradients)      (GEMMs)
4. Average gradients across all GPUs                (all_reduce)
5. Optimizer step (identical on all GPUs)
```

After step 4, every GPU has identical averaged gradients, so the optimizer step produces identical weights. The model replicas stay in sync.

### Two-Stream Pipeline

DDP is simpler than TorchRec -- no `all_to_all`, no `datadist_stream`:

```
memcpy_stream:    [H2D: copy batch to GPU] ──────────────────┐
                                                              │ wait_stream()
                                                              ▼
default_stream:                            [Forward(batch_gpu)]
                                           [Backward          ]
                                           [all_reduce(grads)  ]
                                           [Optimizer step     ]
```

In real DDP (PyTorch), the gradient all_reduce is actually **bucketed and overlapped** with backward -- as soon as a layer's gradients are ready, they're all_reduced while backward continues for earlier layers.

The reproducer supports both patterns via `--bucketed`:

**Non-bucketed** (default): one bulk all_reduce after all of backward. Simpler, tests the basic communication pattern.

**Bucketed** (`--bucketed`): per-layer all_reduce interleaved with backward. Matches real PyTorch DDP's overlap pattern:

```
default_stream:  [Forward all layers]
                 [Bwd Layer 2] → [all_reduce L2] → [Bwd Layer 1] → [all_reduce L1] → [Bwd Layer 0] → [all_reduce L0]
                                  ↑ overlaps with next layer's backward via NCCL pipelining
```

The bucketed pattern creates many concurrent NCCL operations interleaved with compute, which may trigger different timing-sensitive bugs than the bulk pattern.

### Data Dependency: Backward → all_reduce

Unlike the default mode where all_reduce uses synthetic data, DDP mode all_reduces **actual computed gradients**:

```
Forward(batch_gpu) → loss
           ↓
Backward(loss) → param.grad   ← real gradient tensor
           ↓
all_reduce(param.grad)         ← operates on real data from backward
           ↓
optimizer.step()               ← updates weights with averaged gradients
```

This means DDP mode tests a real **data dependency** between compute and communication. If the runtime corrupts the timing, the gradients will diverge across ranks (detected via checksum comparison).

### Deterministic Mode (`--deterministic`)

For gradient verification to work, all ranks must compute identical gradients (before averaging). This requires:
- Same random seed → same initial weights
- Same input data → same forward/backward

The `--deterministic` flag fixes all seeds so gradients are reproducible across ranks. After all_reduce averaging, all ranks should have bit-identical gradient checksums.

---

## 5. FSDP Pattern

### What FSDP Does

Fully Sharded Data Parallel (FSDP) is PyTorch's memory-efficient training strategy. Instead of replicating the full model on every GPU (like DDP), FSDP **shards parameters** across GPUs:

```
DDP (each GPU holds full model):
  GPU 0: [Layer0_full, Layer1_full, Layer2_full]    ← 3x memory
  GPU 1: [Layer0_full, Layer1_full, Layer2_full]
  GPU 2: [Layer0_full, Layer1_full, Layer2_full]

FSDP (each GPU holds 1/N of each layer):
  GPU 0: [Layer0_shard0, Layer1_shard0, Layer2_shard0]  ← 1x memory
  GPU 1: [Layer0_shard1, Layer1_shard1, Layer2_shard1]
  GPU 2: [Layer0_shard2, Layer1_shard2, Layer2_shard2]
```

This reduces per-GPU memory by N (number of GPUs), enabling training of much larger models.

### The all_gather / reduce_scatter Dance

Since each GPU only holds a shard, it must **reconstruct the full parameter** before it can compute. This is done per-layer:

**Forward pass (layer by layer):**

```
For each layer L (0, 1, 2, ...):
  1. all_gather(my_shard_L) → full_param_L      ← reconstruct full parameter
  2. GEMM(input, full_param_L) → output          ← compute with full parameter
  3. (optionally free full_param_L to save memory)
```

**Backward pass (reverse layer order):**

```
For each layer L (reversed: ..., 2, 1, 0):
  1. all_gather(my_shard_L) → full_param_L      ← reconstruct (was freed)
  2. GEMM_backward(grad, full_param_L) → grad_L  ← compute gradient
  3. reduce_scatter(grad_L) → my_grad_shard_L    ← shard gradient back
```

### Many Small Collectives vs. Few Large Ones

This is the fundamental difference between FSDP and the other modes:

```
Default (TorchRec):   [──── one big all_to_all ────] [── one big all_reduce ──]

DDP:                  [──────────── one big all_reduce ────────────]

FSDP:                 [ag₀][ag₁][ag₂]...[rs₂][rs₁][rs₀]
                       ↑    ↑    ↑       ↑    ↑    ↑
                       many small collectives interleaved with compute
```

Where `ag` = all_gather, `rs` = reduce_scatter.

FSDP has `2 × num_layers` collective operations per iteration (one all_gather + one reduce_scatter per layer, plus all_gathers in forward). This creates a very different load on the NCCL runtime -- many small kernel launches instead of a few large ones.

### Per-Layer Timeline

```
default_stream:  [all_gather L0] [GEMM L0] [all_gather L1] [GEMM L1] [all_gather L2] [GEMM L2]
                  ─── forward ──────────────────────────────────────────────────────────────────→

default_stream:  [all_gather L2] [GEMM bwd L2] [reduce_scatter L2]
                 [all_gather L1] [GEMM bwd L1] [reduce_scatter L1]
                 [all_gather L0] [GEMM bwd L0] [reduce_scatter L0]
                  ─── backward (reversed) ─────────────────────────────────────────────────────→
```

All collectives run on the **default stream** -- there's no separate communication stream in FSDP. Overlap comes from:
1. NCCL internal pipelining (NCCL can overlap its own network transfers with GPU compute)
2. H2D on `memcpy_stream` (same as all modes)

### Buffer Reuse

In real FSDP, the full parameter buffer is reused across layers (to save memory -- that's the whole point of FSDP). The reproducer mirrors this:

```python
# Same full_param buffer reused for every layer's all_gather
self.full_param = torch.empty(shard_size * world_size, ...)

for layer in range(num_layers):
    dist.all_gather_into_tensor(self.full_param, self.param_shards[layer])
    # full_param now contains layer's full parameter
    # ... compute ...
    # full_param will be overwritten by next layer's all_gather
```

Only the **last layer's** all_gather and reduce_scatter results are verified. If the runtime corrupts any collective in the chain, the corruption propagates through the reused buffer and will be caught at verification time.

### What the Reproducer Simulates

| Real FSDP | Reproducer |
|-----------|-----------|
| Sharded model parameters | `param_shards[L]` filled with `float(rank)` |
| all_gather to reconstruct full param | `dist.all_gather_into_tensor(full_param, param_shards[L])` |
| Forward GEMM per layer | `torch.mm(weight_matrices[L], activation)` |
| Backward GEMM per layer | `torch.mm(weight_matrices[L].T, grad_buffer)` |
| reduce_scatter to shard gradients | `dist.reduce_scatter_tensor(grad_shard, full_grad)` |

---

## 6. H2D Strategies

All three modes support two H2D (Host-to-Device) buffering strategies. This controls how training data moves from CPU to GPU.

### Single-Buffered (default)

One buffer pair. Copy, wait, use. Simple and sequential.

```
Iteration N:
  memcpy_stream:   [fill CPU buf N] [copy to GPU ...]
                                                      │ wait_stream()
                                                      ▼
  default_stream:                                    [use GPU buf N for forward...]

Iteration N+1:
  memcpy_stream:   [fill CPU buf N+1] [copy to GPU ...]
                                                        │ wait_stream()
                                                        ▼
  default_stream:                                      [use GPU buf N+1 for forward...]
```

**Gap:** Between iterations, the GPU is idle waiting for the next H2D copy. No overlap.

### Double-Buffered / Prefetch (`--prefetch`)

Two buffer pairs (`current` and `next`). While the GPU computes with the current batch, the next batch is being copied in the background:

```
Iteration N:
  memcpy_stream:                              [copy batch N+1 to GPU_next ...]
                                               ↑ prefetch starts after forward
  default_stream:  [use GPU_current (batch N)] [backward ...] [collectives ...]
                                                                              │
                   ─── swap: GPU_current ↔ GPU_next ──────────────────────────┘

Iteration N+1:
  memcpy_stream:                              [copy batch N+2 to GPU_next ...]
  default_stream:  [use GPU_current (batch N+1)] [backward ...] [collectives ...]
                                                                              │
                   ─── swap ──────────────────────────────────────────────────┘
```

**No gap:** H2D for the next batch overlaps with the current iteration's backward pass. This is what real training pipelines (DDP, FSDP) use.

The swap is just a pointer exchange -- no data copy:

```python
self.batch_gpu, self._batch_gpu_next = self._batch_gpu_next, self.batch_gpu
```

### Why Both Matter for Bug Detection

The two strategies create different **timing profiles** on the `memcpy_stream`:

- **Single-buffered:** H2D happens at the start, then memcpy_stream is idle during compute. Short burst of DMA activity.
- **Double-buffered:** H2D overlaps with backward/collectives. DMA engine active during compute. More concurrent resource usage.

Since the RCCL bug is timing-sensitive, these different profiles can trigger (or mask) the bug differently. Testing both gives broader coverage.

---

## 7. The Race Condition Bug

### What Goes Wrong

The bug occurs in the RCCL/HIP runtime's internal synchronization between hardware dispatch queues. When `GPU_MAX_HW_QUEUES=4`:

```
Normal (correct):
  HW Queue 0: [H2D copy completes] ──── signal ────→ [Forward reads correct data]
  HW Queue 1:                         [all_to_all]

Bug (broken):
  HW Queue 0: [H2D copy in progress...]
  HW Queue 1:                         [Forward reads STALE data]  ← signal lost/delayed
```

The `wait_stream()` call in the application code is correct. The application did everything right. But the runtime's internal mechanism for propagating the synchronization signal across HW queues has a bug, so the GPU starts the forward pass before the H2D copy actually finishes.

### What Corruption Looks Like

The data is not random garbage -- it's **stale data** from a previous iteration:

```
Expected (iteration 42): batch_gpu = [42.0, 42.0, 42.0, ...]
Actual (stale):           batch_gpu = [41.0, 41.0, 41.0, ...]  ← previous iteration's data
```

This is particularly insidious because:
1. The model still trains (the data is valid, just wrong)
2. Loss may look normal (slightly noisier, but not obviously broken)
3. No errors, no crashes, no NaN -- just silently wrong results

### Why This Reproducer Catches It

The reproducer fills every buffer with **predictable, verifiable patterns** every iteration. After the full pipeline runs with proper synchronization, it checks every buffer:

| Buffer | Expected | Corruption Means |
|--------|----------|-----------------|
| `batch_gpu` | `iteration % 1000` | H2D was read before copy finished |
| `recv_buf[j]` | `float(j)` | all_to_all data was corrupted |
| `reduce_buf` | `sum(1..world_size)` | all_reduce result was corrupted |
| `full_param` chunk j | `float(j)` | all_gather data was corrupted |
| `grad_shard` | `sum(1..world_size)` | reduce_scatter result was corrupted |

If any value doesn't match, it's proof of a runtime bug -- the application's synchronization was correct.

### Factors That Affect Bug Reproduction

| Factor | Exposes Bug | Masks Bug |
|--------|------------|-----------|
| `GPU_MAX_HW_QUEUES` | 4 (more parallelism) | 2 (less parallelism) |
| Compute duration | ~500ms/step (realistic timing) | Very fast steps (no overlap window) |
| Number of streams | 3 (default mode) | 1 (everything serial) |
| `NCCL_LAUNCH_ORDER_IMPLICIT` | 0 (default, concurrent) | 1 (serialized, slow but safe) |
| Warmup iterations | Many (runtime in hot state) | Few (runtime still cold) |
| Number of GPUs | More (more cross-GPU traffic) | Fewer |

---

## Further Reading

- [PyTorch DDP](https://pytorch.org/docs/stable/notes/ddp.html) -- Distributed Data Parallel internals
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) -- Fully Sharded Data Parallel
- [TorchRec](https://pytorch.org/torchrec/) -- Recommendation model training framework
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) -- Collective communication library (RCCL is AMD's fork)
- [GPU Streams (HIP)](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html) -- Asynchronous GPU execution model
