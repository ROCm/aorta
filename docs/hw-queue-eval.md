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
```

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
src/aorta/hw_queue_eval/
├── core/
│   ├── harness.py        # Execution harness
│   ├── metrics.py        # Metrics collection
│   └── torch_profiler.py # Profiler integration
├── workloads/
│   ├── base.py           # Base workload classes
│   ├── registry.py       # Workload discovery
│   ├── distributed/      # FSDP, MoE, etc.
│   ├── inference/        # Speculative decode, RAG, etc.
│   ├── pipeline/         # Async dataload, ZeRO, etc.
│   └── latency_sensitive/ # Hetero kernels, tiny kernels
└── cli.py                # Command-line interface
```
