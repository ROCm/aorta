# Hardware Queue Evaluation

A framework for stress-testing GPU hardware queue scheduling with workloads requiring high stream concurrency (8, 16, 32+ concurrent streams).

## Purpose

Modern GPU workloads increasingly rely on multiple concurrent streams for overlapping compute, communication, and data movement. This module measures:

- Queue switch latency overhead
- Throughput scaling with stream count
- Scheduling behavior under high concurrency
- Performance bottlenecks in queue dispatch

## Quick Start

```bash
# List available workloads
python -m aorta.hw_queue_eval list

# Run a single workload
python -m aorta.hw_queue_eval run hetero_kernels --streams 8

# Sweep across stream counts
python -m aorta.hw_queue_eval sweep hetero_kernels --streams 1,2,4,8,16

# Run all P0 (critical) workloads
python -m aorta.hw_queue_eval run-priority P0

# Compare results
python -m aorta.hw_queue_eval compare --baseline results_a.json --test results_b.json

# Capture rocprof trace
python -m aorta.hw_queue_eval profile hetero_kernels --streams 8
```

## Workloads

### Priority P0 (Critical)

| Workload | Description |
| --- | --- |
| `hetero_kernels` | Mixed tiny (~10us) and large (~10ms) GEMMs - tests convoy effect |
| `tiny_kernel_stress` | Extreme small kernel dispatch stress test |
| `large_gemm_only` | Pure GEMM baseline for throughput reference |

### Distributed Training

| Workload | Description |
| --- | --- |
| `comms_compute_overlap` | Configurable comms-compute overlap with GEMM and collectives |
| `fsdp_tp` | FSDP + Tensor Parallelism (3D parallelism) with actual collectives |
| `moe` | Mixture of Experts with 8-16 parallel expert streams |
| `activation_ckpt` | Activation checkpointing with recomputation patterns |
| `grad_accum` | Gradient accumulation with early reduction |

### Inference

| Workload | Description |
| --- | --- |
| `speculative_decode` | Draft + verify decoding with tight latency requirements |
| `continuous_batch` | Prefill/decode overlap with memory-bound operations |
| `rag_pipeline` | Multi-model RAG pipelines |

### Pipeline / System-Level

| Workload | Description |
| --- | --- |
| `simple_transformer` | GPT-2-style transformer training with pipelined forward pass |
| `async_dataload` | Async data loading with GPU preprocessing |
| `zero_offload` | ZeRO-style memory offload patterns |
| `torch_compile` | Multi-region execution under torch.compile |

### Latency-Sensitive

| Workload | Description |
| --- | --- |
| `graph_subgraphs` | Independent subgraph execution patterns |

## Metrics

Each run collects:

- **Latency**: Mean, P50, P95, P99 per-iteration
- **Throughput**: Operations/sec, samples/sec
- **Queue Switch Overhead**: Inter-stream vs intra-stream gaps
- **Memory**: Peak, allocated, reserved
- **Scaling Analysis**: Throughput per stream, efficiency curves

## Configuration Options

```bash
# Stream count
--streams 8

# Synchronization mode
--sync-mode per_iteration  # or: end_only, none

# Iterations
--iterations 100
--warmup 10

# Output
--output results.json

# Profiling
--profile --profile-dir traces/
```

### Comms-Compute Overlap Options

These options apply to the `comms_compute_overlap` workload:

```bash
# Workload mode
--mode comms_compute        # or: compute_only, comms_only

# GEMM configuration
--mm-dim 2048,2048,2048     # M,N,K dimensions (single value for square)
--num-compute 10            # GEMMs per iteration per compute stream
--comp-dtype bfloat16       # float32, float16, bfloat16

# Communication configuration
--comm-size 128M            # supports K/M/G suffix
--num-coll 1                # collectives per iteration
--comm-dtype bfloat16       # float32, float16, bfloat16

# Stream control
--compute-streams 2         # independent of --streams

# Distributed mode (requires torchrun)
--real-collectives          # use real NCCL/RCCL collectives
--async-op                  # non-blocking collectives
--backend nccl              # nccl or gloo
--process-groups "[0,1,2,3],[4,5,6,7]"
```

Example with all options:

```bash
torchrun --nproc_per_node=8 -m aorta.hw_queue_eval run comms_compute_overlap \
    --streams 4 --real-collectives --async-op --backend nccl \
    --process-groups "[0,1,2,3,4,5,6,7]" \
    --compute-streams 2 --comp-dtype bfloat16 --comm-dtype bfloat16 \
    --mm-dim 4096,4096,4096 --num-compute 10 --comm-size 128M \
    --profile --profile-dir traces/
```

## Running the Simple Transformer Workload

The `simple_transformer` workload implements a GPT-2-style decoder-only transformer with multi-stream pipelined training. Layers are split into groups, each assigned to a different CUDA stream. The forward pass is pipelined (stream K+1 waits on stream K), loss and backward run on a dedicated stream, and the optimizer step overlaps with the next iteration's data prep.

### Running locally

```bash
# Single run with 4 streams (recommended)
python -m aorta.hw_queue_eval run simple_transformer --streams 4

# Sweep across stream counts
python -m aorta.hw_queue_eval sweep simple_transformer --streams 2,4,8,16

# With profiling enabled
python -m aorta.hw_queue_eval run simple_transformer --streams 4 --profile --profile-dir traces/

# Save results to JSON
python -m aorta.hw_queue_eval run simple_transformer --streams 4 -o results/simple_transformer.json
```

### Running on a SLURM cluster node with Docker

When running on a cluster node (e.g. via an interactive SLURM allocation), use Docker with GPU passthrough:

```bash
srun --jobid=<JOB_ID> --nodelist=<NODE> --nodes=1 --ntasks=1 bash -c \
  "docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /path/to/aorta-main:/workspace/aorta \
    -w /workspace/aorta \
    -e PYTHONPATH=/workspace/aorta/src \
    <docker-image> \
    python -m aorta.hw_queue_eval run simple_transformer --streams 4"
```

To run a stream count sweep:

```bash
srun --jobid=<JOB_ID> --nodelist=<NODE> --nodes=1 --ntasks=1 bash -c \
  "docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /path/to/aorta-main:/workspace/aorta \
    -w /workspace/aorta \
    -e PYTHONPATH=/workspace/aorta/src \
    <docker-image> \
    python -m aorta.hw_queue_eval sweep simple_transformer --streams 2,4,8,16"
```

### Stream assignment

With `--streams 4` (the default/recommended):

| Stream | Role |
| --- | --- |
| 0 | Data preparation, optimizer step |
| 1 | Forward layers 0–1 |
| 2 | Forward layers 2–3 |
| 3 | Forward layers 4–5, loss, backward |

### Configurable parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `hidden_size` | 512 | Transformer hidden dimension |
| `num_layers` | 6 | Number of transformer layers |
| `num_heads` | 8 | Attention heads |
| `batch_size` | 8 | Training batch size |
| `seq_length` | 128 | Sequence length |
| `vocab_size` | 32000 | Vocabulary size |

### What to look for

- **Throughput scaling**: Should remain flat or slightly degrade with more streams, since forward layers are sequentially dependent.
- **Queue switch overhead**: At high stream counts (16+), watch for increased latency indicating hardware queue contention.
- **P99/P50 ratio**: Values close to 1.0 indicate stable scheduling; values above 2.0 suggest queue contention.

## Analysis

Compare sweep results:

```bash
python scripts/analyze_sweep_results.py \
  --baseline sweep_v1.json \
  --test sweep_v2.json
```

Profile with rocprofv3:

```bash
bash scripts/hw_queue/profile_queues.sh hetero_kernels 8
```

## Architecture

```
src/aorta/
├── utils/
│   ├── distributed.py       # torch.distributed init, process groups
│   └── ...
└── hw_queue_eval/
    ├── core/
    │   ├── harness.py        # Execution harness
    │   ├── metrics.py        # Metrics collection
    │   └── torch_profiler.py # Profiler integration (rank-aware traces)
    ├── workloads/
    │   ├── base.py           # Base workload classes
    │   ├── registry.py       # Workload discovery
    │   ├── distributed/      # FSDP, MoE, comm-compute overlap, etc.
    │   ├── inference/        # Speculative decode, RAG, etc.
    │   ├── pipeline/         # Async dataload, ZeRO, etc.
    │   └── latency_sensitive/ # Hetero kernels, tiny kernels
    └── cli.py                # Command-line interface
```
