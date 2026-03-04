# Developer Guide - RCCL Race Condition Reproducer

This guide explains the modular architecture and how to extend the reproducer with new modes and compute types.

## Validating Changes (Multi-Node)

After modifying the reproducer code, run these tests on a multi-node cluster to verify correctness. All commands below are run from the **master node** (first node in `node_ip_list.txt`).

### Prerequisites

1. Two or more compute nodes listed in `scripts/multi_node/node_ip_list.txt` (master first)
2. Docker containers running on all nodes (use `start_docker_all_nodes.sh`)
3. All nodes on the same git branch

```bash
# Start containers on all nodes (one-time, persists across runs)
./scripts/multi_node/start_docker_all_nodes.sh \
    docker/docker-compose.rocm70_9-1-shampoo.yaml \
    training-overlap-bugs-rocm70_9-1-shampoo
```

### Quick Smoke Tests

Run these first -- they complete in ~1-2 minutes each and catch import errors, buffer allocation issues, and basic communication failures.

```bash
# 1. Default mode smoke test (TorchRec-like: H2D + all_to_all + all_reduce)
./scripts/multi_node/launch_reproducer.sh \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --hw-queues 4 --warmup 5 --verify 20 --no-compute

# 2. DDP mode smoke test (gradient all_reduce + H2D prefetch)
./scripts/multi_node/launch_reproducer.sh \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --mode ddp --hw-queues 4 --warmup 5 --verify 20 --no-compute --deterministic

# 3. DDP bucketed mode smoke test (per-layer backward + all_reduce overlap)
./scripts/multi_node/launch_reproducer.sh \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --mode ddp --bucketed --hw-queues 4 --warmup 5 --verify 20 --no-compute --deterministic

# 4. FSDP mode smoke test (per-layer all_gather + reduce_scatter)
./scripts/multi_node/launch_reproducer.sh \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --mode fsdp --hw-queues 4 --warmup 5 --verify 20 --no-compute
```

**Expected output** (from `experiments/reproducer_*/logs/node_0.txt`):

```
PASSED: No corruption in 25 iterations with proper synchronization
VERDICT: No runtime bug detected with current settings.
```

Both nodes should report PASSED on all ranks. Check for errors:

```bash
grep -i 'CORRUPTION\|RUNTIME BUG\|Error\|Traceback' experiments/reproducer_*/logs/node_*.txt
```

### Comprehensive Validation

Run these for thorough testing (~80-90 min each with compute, ~15 min with `--no-compute`):

```bash
# 1. Default mode — baseline (HW_QUEUES=4, exposes timing-sensitive bugs)
./scripts/multi_node/launch_reproducer.sh \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --hw-queues 4 --warmup 100 --verify 10000

# 2. Default mode — serialized comparison (HW_QUEUES=2, masks parallelism bugs)
./scripts/multi_node/launch_reproducer.sh \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --hw-queues 2 --warmup 100 --verify 10000

# 3. Default mode — same-stream (definitive runtime bug test)
./scripts/multi_node/launch_reproducer.sh \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --hw-queues 4 --same-stream --warmup 100 --verify 10000

# 4. DDP mode — gradient all_reduce + H2D prefetch
./scripts/multi_node/launch_reproducer.sh \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --mode ddp --deterministic --warmup 100 --verify 10000

# 5. FSDP mode — per-layer all_gather + reduce_scatter
./scripts/multi_node/launch_reproducer.sh \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --mode fsdp --hw-queues 4 --warmup 100 --verify 10000

# 6. FSDP mode — with H2D prefetch
./scripts/multi_node/launch_reproducer.sh \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --mode fsdp --prefetch --hw-queues 4 --warmup 100 --verify 10000

# 7. Default mode — NCCL implicit order workaround
./scripts/multi_node/launch_reproducer.sh \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --hw-queues 4 --nccl-implicit --warmup 100 --verify 10000
```

### Monitoring and Checking Results

```bash
# Follow live output (master node)
tail -f experiments/reproducer_*/logs/node_0.txt

# Follow all nodes
tail -f experiments/reproducer_*/logs/node_*.txt

# Check final verdict
grep -i 'VERDICT\|PASSED\|FAILED' experiments/reproducer_*/logs/node_0.txt

# Check for any corruption across all nodes
grep -i 'CORRUPTION\|RUNTIME BUG' experiments/reproducer_*/logs/*.txt
```

### What to Verify After Code Changes

| Change Type | Minimum Validation |
|-------------|-------------------|
| New mode added | Smoke test the new mode + smoke test existing modes (no regression) |
| Modified `base.py` | Smoke test **all** modes (default + ddp + fsdp) |
| Modified `compute.py` | Smoke test with compute enabled (remove `--no-compute`) |
| Modified `config.py` | Smoke test both modes |
| Modified a single mode | Smoke test that mode + one other mode |
| Launch script changes | Smoke test any mode via the launch script |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        __main__.py                              │
│                    (CLI entry point)                            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     modes/__init__.py                           │
│                  create_reproducer(config)                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┬───────────────────┐
              ▼               ▼               ▼                   ▼
┌───────────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ modes/default.py  │ │ modes/ddp.py  │ │ modes/fsdp.py │ │    [future]   │
│ DefaultMode       │ │ DDPMode       │ │ FSDPMode      │ │  PipelineMode │
│ (all_to_all +     │ │ (gradient     │ │ (sharded      │ │  etc.         │
│  all_reduce)      │ │  all_reduce)  │ │  gradients)   │ │               │
└────────┬──────────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
         │                    │                 │                 │
         └────────────────────┴─────────────────┴─────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                         base.py                                 │
│                     BaseReproducer                              │
│  (streams, compute, optimizer, run loop, verification)         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        compute.py                               │
│              BaseCompute → GEMMCompute → [AttentionCompute]     │
│         (forward/backward simulation)      [future]             │
└─────────────────────────────────────────────────────────────────┘
```

### Future Work

| Component | Description | Status |
|-----------|-------------|--------|
| `modes/fsdp.py` | FSDP mode with per-layer all_gather + reduce_scatter | **Implemented** |
| `modes/pipeline.py` | Pipeline parallel mode | Planned |
| `AttentionCompute` | Transformer attention compute | Planned |
| `EmbeddingCompute` | Sparse embedding lookup | Planned |

## File Structure

```
src/aorta/race/
├── __init__.py              # Public API exports
├── __main__.py              # CLI entry point
├── config.py                # ReproducerConfig, ReproducerResult, RaceConfig
├── base.py                  # BaseReproducer abstract class
├── compute.py               # Pluggable compute simulation
├── modes/                   # Mode implementations
│   ├── __init__.py          # Factory function + MODE_REGISTRY
│   ├── default.py           # TorchRec-like mode (all_to_all + all_reduce)
│   ├── ddp.py               # DDP mode (gradient all_reduce)
│   └── fsdp.py              # FSDP mode (per-layer all_gather + reduce_scatter)
├── README.md                # Usage documentation
└── developer_guide.md       # This file
```

## Available Modes

| Mode | Description | Data Flow |
|------|-------------|-----------|
| `default` | TorchRec-like pattern | H2D → Forward → Backward + all_to_all → all_reduce |
| `ddp` | DDP gradient sync | H2D (single/double-buffered) → Forward → Backward → gradient all_reduce |
| `fsdp` | FSDP sharded params | H2D → per-layer [all_gather → GEMM] → per-layer [GEMM bwd → reduce_scatter] |

---

## Adding a New Mode

To add a new reproducer mode (e.g., FSDP, Pipeline Parallel):

### Step 1: Create the mode file

Create `src/aorta/race/modes/your_mode.py`:

```python
"""Your mode description."""

import torch
import torch.distributed as dist

from aorta.race.base import BaseReproducer


class YourModeReproducer(BaseReproducer):
    """Your mode reproducer implementation."""

    def setup_buffers(self) -> None:
        """Allocate mode-specific buffers."""
        # Required: Create your GPU buffers
        self.my_buffer = torch.zeros(
            self.config.tensor_size,
            dtype=self.dtype,
            device="cuda",
        )
        # ... more buffers as needed

    def run_iteration(self, iteration: int) -> bool:
        """
        Run one iteration of the mode-specific data flow.

        Args:
            iteration: Current iteration number

        Returns:
            True if verification passed, False if corruption detected
        """
        # 1. Prepare data with known pattern
        expected_value = float(iteration % 1000)
        self.batch_cpu.fill_(expected_value)

        # 2. H2D transfer on memcpy_stream
        with torch.cuda.stream(self.memcpy_stream):
            self.batch_gpu.copy_(self.batch_cpu, non_blocking=True)

        # 3. Record event for synchronization
        self.h2d_complete.record(self.memcpy_stream)

        # 4. Wait for H2D before compute
        self.default_stream.wait_event(self.h2d_complete)

        # 5. Run your compute/communication pattern
        # ... your mode-specific logic here

        # 6. Verify data integrity (during verification phase)
        if iteration >= self.config.warmup_iterations:
            return self._verify_my_data(expected_value)

        return True

    def _verify_my_data(self, expected: float) -> bool:
        """Verify your mode-specific data."""
        # Check for corruption
        if not torch.allclose(self.my_buffer, expected_tensor):
            self.corruption_count += 1
            return False
        return True


__all__ = ["YourModeReproducer"]
```

### Step 2: Register the mode

Edit `src/aorta/race/modes/__init__.py`:

```python
from aorta.race.modes.your_mode import YourModeReproducer

MODE_REGISTRY = {
    "default": DefaultModeReproducer,
    "ddp": DDPModeReproducer,
    "your_mode": YourModeReproducer,  # Add your mode
}
```

### Step 3: Use your mode

```bash
torchrun --nproc_per_node=8 -m aorta.race -o mode=your_mode
```

---

## Adding a New Compute Type

To add a new compute pattern (e.g., Attention, Embedding Lookup):

### Step 1: Create the compute class

Edit `src/aorta/race/compute.py`:

```python
class AttentionCompute(BaseCompute):
    """Attention-based compute simulation."""

    def setup(self, requires_grad: bool = False) -> None:
        """Initialize attention layers."""
        self.query = torch.nn.Linear(
            self.config.gemm_size, self.config.gemm_size
        ).to("cuda", dtype=self.dtype)
        self.key = torch.nn.Linear(
            self.config.gemm_size, self.config.gemm_size
        ).to("cuda", dtype=self.dtype)
        self.value = torch.nn.Linear(
            self.config.gemm_size, self.config.gemm_size
        ).to("cuda", dtype=self.dtype)

        if requires_grad:
            for p in self.parameters():
                p.requires_grad_(True)

    def forward(self, batch_gpu: torch.Tensor) -> torch.Tensor:
        """Run attention forward pass."""
        # Reshape input for attention
        x = batch_gpu.view(-1, self.config.gemm_size)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.config.gemm_size ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        return output.sum()

    def backward(self, output: torch.Tensor, use_autograd: bool = False) -> None:
        """Run attention backward pass."""
        if use_autograd:
            output.backward()
        else:
            # Manual backward simulation
            for _ in range(self.config.gemm_layers):
                dummy = torch.matmul(
                    torch.randn(self.config.gemm_size, self.config.gemm_size,
                               device="cuda", dtype=self.dtype),
                    torch.randn(self.config.gemm_size, self.config.gemm_size,
                               device="cuda", dtype=self.dtype)
                )

    def parameters(self):
        """Return trainable parameters."""
        return list(self.query.parameters()) + \
               list(self.key.parameters()) + \
               list(self.value.parameters())
```

### Step 2: Register the compute type

At the bottom of `compute.py`:

```python
register_compute("attention", AttentionCompute)
```

### Step 3: Use your compute type

Update config yaml:

```yaml
compute_type: attention  # instead of "gemm"
```

---

## Key Design Patterns

### 1. Template Method Pattern

`BaseReproducer.run()` defines the algorithm structure:

```python
def run(self):
    self.setup_buffers()      # Subclass implements
    for i in range(iterations):
        self.run_iteration(i)  # Subclass implements
    return self.result
```

Subclasses implement `setup_buffers()` and `run_iteration()`.

### 2. Pluggable Compute

Compute is separate from the reproducer so different compute patterns can be tested without modifying mode logic:

```python
# In base.py
self.compute = create_compute(config.compute_type, config, dtype)
self.compute.setup(requires_grad=True)

# In run_iteration
output = self.compute.forward(batch_gpu)
self.compute.backward(output, use_autograd=True)
```

### 3. Registry Pattern

`MODE_REGISTRY` and `COMPUTE_REGISTRY` allow adding new types without modifying existing code:

```python
# modes/__init__.py
MODE_REGISTRY = {
    "default": DefaultModeReproducer,
    "ddp": DDPModeReproducer,
}

def create_reproducer(config, rank, world_size):
    cls = MODE_REGISTRY[config.mode]
    return cls(config, rank, world_size)
```

### 4. Verification in Base Class

`_verify_h2d()` is shared since all modes need H2D verification. Mode-specific verification (all_to_all, gradient consistency) is in subclasses.

---

## BaseReproducer Provided Utilities

The `BaseReproducer` class provides these utilities for subclasses:

| Attribute | Type | Description |
|-----------|------|-------------|
| `self.config` | `ReproducerConfig` | Configuration object |
| `self.rank` | `int` | Current process rank |
| `self.world_size` | `int` | Total number of processes |
| `self.dtype` | `torch.dtype` | Data type (e.g., bfloat16) |
| `self.memcpy_stream` | `torch.cuda.Stream` | GPU stream for H2D transfers |
| `self.default_stream` | `torch.cuda.Stream` | Default GPU stream for compute |
| `self.batch_cpu` | `torch.Tensor` | Pinned CPU tensor (H2D source) |
| `self.batch_gpu` | `torch.Tensor` | GPU tensor (H2D destination) |
| `self.compute` | `BaseCompute` | Compute simulator instance |
| `self.optimizer` | `torch.optim.Optimizer` | Optimizer (if configured) |
| `self.in_verification_phase` | `bool` | Whether currently in verification phase |
| `self.corruption_details` | `list` | Details of each corruption |

| Method | Description |
|--------|-------------|
| `_verify_h2d(expected)` | Verify H2D data integrity, returns bool |
| `_log_progress(iteration)` | Log iteration progress if at log interval |
| `_setup_deterministic()` | Set seeds for reproducibility |

---

## Data Flow Examples

### Default Mode (TorchRec-like)

```
memcpy_stream:  [H2D] → batch_gpu
                          ↓ (Forward READS batch_gpu)
default_stream:          [Forward] → [Backward] → [all_reduce]

datadist_stream:         [all_to_all]
                          (overlaps with backward)
```

**Verification checks:**
- H2D: `batch_gpu == iteration % 1000`
- all_to_all: `recv_buf[j] == j` (data from rank j)
- all_reduce: `reduce_buf == sum(1..world_size)`

### DDP Mode

```
Iteration N:
    memcpy_stream:  [H2D batch_N+1] ────────────────────────┐
                                                            │ (prefetch)
    default_stream: [Forward(batch_N)] → [Backward] → [all_reduce grads]
                                                            │
                    ← swap buffers ─────────────────────────┘
```

**Verification checks:**
- H2D: `batch_gpu == iteration % 1000`
- Gradient consistency: All ranks have identical gradient checksums

### DDP Mode (Bucketed, `--bucketed`)

```
memcpy_stream:   [H2D] ──────────────────────────────────────────────────────────────┐
                                                                                      │ wait
default_stream:  [Forward all layers]                                                 │
                 [Bwd L2 + all_reduce L2] → [Bwd L1 + all_reduce L1] → [Bwd L0 + AR L0]
                 [optimizer step]
```

Per-layer backward GEMM followed by immediate all_reduce. Each layer's all_reduce can overlap with the next layer's backward via NCCL internal pipelining.

**Verification checks:**
- H2D: `batch_gpu == iteration % 1000`
- Gradient consistency: All ranks have identical gradient checksums

### FSDP Mode

```
memcpy_stream:   [H2D] ─────────────────────────────────────────────────────┐
                                                                             │ wait
default_stream:  [all_gather L0 → GEMM L0 → all_gather L1 → GEMM L1 → ...]│
                 [... → GEMM bwd L1 → reduce_scatter L1 →                   │
                        GEMM bwd L0 → reduce_scatter L0]                    │
                 [optimizer step]
```

**Verification checks:**
- H2D: `batch_gpu == iteration % 1000`
- all_gather: after gathering rank-filled shards, chunk j == `float(j)`
- reduce_scatter: after scattering, output == `sum(1..world_size)`
