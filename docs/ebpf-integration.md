# eBPF Integration for Aorta

Aorta integrates eBPF-based kernel-level tracing to complement its existing
user-space GPU profiling (PyTorch profiler, CUDA events, rocprof).  This
provides ground-truth driver-level visibility into hardware queue dispatch,
memory management, and scheduling behaviour on AMD ROCm GPUs.

## Overview

| Layer          | Tool/Mechanism                   | What It Captures                          |
|----------------|----------------------------------|-------------------------------------------|
| User-space     | PyTorch profiler, CUDA events    | Kernel launch latency, inter-stream gaps  |
| **Kernel-level** | **bpftrace + amdgpu tracepoints** | **Command submission, HW dispatch, BO map** |
| Device-side    | rocprof --att (interim)          | Per-instruction CU utilization            |

## Prerequisites

- Linux kernel ≥ 5.x with `amdgpu` / `amdkfd` drivers loaded
- `bpftrace` installed (`apt-get install bpftrace` or `dnf install bpftrace`)
- Root or `CAP_BPF` capability for attaching tracepoints
- `debugfs` mounted at `/sys/kernel/debug` (usually automatic)

### Verify Your System

```bash
# Check kernel version
uname -r

# Check amdgpu tracepoints
sudo ls /sys/kernel/debug/tracing/events/amdgpu/

# Check amdkfd tracepoints
sudo ls /sys/kernel/debug/tracing/events/amdkfd/

# Check bpftrace
which bpftrace && bpftrace --version

# Quick BPF sanity test
sudo bpftrace -e 'BEGIN { printf("BPF works\n"); exit(); }'

# Or use aorta's built-in check
python -m aorta.hw_queue_eval ebpf-info
```

## Quick Start

### Queue Tracing

Trace driver-level command submission and dispatch alongside your benchmark:

```bash
# Run with eBPF queue tracing
sudo python -m aorta.hw_queue_eval run hetero_kernels --streams 8 --ebpf-trace

# Sweep with eBPF tracing
sudo python -m aorta.hw_queue_eval sweep hetero_kernels \
    --streams 1,2,4,8,16 --ebpf-trace
```

The output includes a new **eBPF DRIVER-LEVEL QUEUE METRICS** section:

```
eBPF DRIVER-LEVEL QUEUE METRICS:
  Total submissions:  800
  Total dispatches:   800
  HW rings used:      [0, 1, 2, 3]
  Submit→dispatch avg:  12.3 us
  Submit→dispatch P99:  45.7 us

eBPF vs CUDA COMPARISON:
  eBPF submit→dispatch:  0.012 ms
  CUDA switch overhead:  0.015 ms
  Measurement accuracy:  80.0%
```

### Memory Tracing

Trace GPU memory events (buffer mapping, process evictions):

```bash
sudo python -m aorta.hw_queue_eval run hetero_kernels \
    --streams 8 --ebpf-memory-trace
```

### Policy Sweep

Compare workload performance under different scheduling/memory policies:

```bash
# Default policies: baseline, priority_lc, priority_be
python -m aorta.hw_queue_eval policy-sweep hetero_kernels --streams 8

# Custom policy selection
python -m aorta.hw_queue_eval policy-sweep moe --streams 16 \
    --policies baseline,priority_lc,high_queue -o policy_results.json
```

Available built-in policies:

| Policy             | Type       | Description                                    |
|--------------------|------------|------------------------------------------------|
| `baseline`         | scheduling | Default round-robin, no hardware constraints   |
| `priority_lc`      | scheduling | Latency-critical: max clocks (level 7)         |
| `priority_be`      | scheduling | Best-effort: reduced clocks (level 2), 150W    |
| `multi_tenant_fair`| scheduling | Fair sharing via GPU_MAX_HW_QUEUES=2            |
| `high_queue`       | scheduling | Maximum HW queues (GPU_MAX_HW_QUEUES=8)         |
| `default_uvm`      | memory     | HSA_XNACK=1 (retryable page faults)            |
| `xnack_off`        | memory     | HSA_XNACK=0 (no retryable page faults)         |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    aorta CLI                                │
│   run --ebpf-trace   sweep --ebpf-memory-trace             │
│   policy-sweep       ebpf-info                             │
├──────────────┬──────────────┬───────────────────────────────┤
│  StreamHarness              │  PolicyEvaluator              │
│  ├─ BPFQueueTracer          │  ├─ PolicyConfig presets      │
│  ├─ BPFMemoryTracer         │  ├─ env + GPU control knobs  │
│  └─ MetricsCollector        │  └─ comparison reports        │
├──────────────┴──────────────┴───────────────────────────────┤
│  Kernel: amdgpu tracepoints    amdkfd tracepoints           │
│  ├─ amdgpu_cs_ioctl            ├─ kfd_evict_process_worker_start  │
│  ├─ amdgpu_sched_run_job       ├─ kfd_restore_process_worker_start│
│  ├─ amdgpu_vm_bo_map           ├─ kfd_map_memory_to_gpu_start     │
│  └─ amdgpu_vm_bo_unmap         └─ kfd_map_memory_to_gpu_end       │
└─────────────────────────────────────────────────────────────┘
```

> On older kernels the eviction/restore tracepoints are exposed as
> ``kfd_evict_process`` / ``kfd_restore_process`` (without the
> ``_worker_start`` suffix).  The tracer auto-detects which variant is
> available at startup.

## Module Reference

### `ebpf_tracer.py`

- `check_ebpf_capabilities()` -- detect bpftrace, tracepoints, kernel version
- `BPFQueueTracer` -- attach to `amdgpu_cs_ioctl` and `amdgpu_sched_run_job`
- `DriverQueueMetrics` -- aggregated submission/dispatch counts and latencies
- `EBPFCapabilities` -- system capability snapshot

### `ebpf_memory_tracer.py`

- `BPFMemoryTracer` -- attach to `amdgpu_vm_bo_map/unmap`, `amdgpu_bo_move`,
  and the KFD `kfd_evict_process_worker_start` /
  `kfd_restore_process_worker_start` /
  `kfd_map_memory_to_gpu_start`/`_end` tracepoints.  Older kernels expose
  the eviction probes without the `_worker_start` suffix; the tracer
  auto-detects which is available.
- `MemoryTraceMetrics` -- BO move/map/unmap counts, evict/restore rates,
  migration bytes.  Note: ``total_faults`` and friends are deprecated
  aliases for ``total_eviction_restore_pairs``; they count process
  eviction cycles, *not* GPU UVM page faults.

### `policy_evaluator.py`

- `PolicyConfig` -- describes a scheduling or memory policy
- `PolicyEvaluator` -- runs a workload under multiple policies
- `PolicyComparison` -- comparison table and JSON export
- `BUILTIN_POLICIES` -- preset policy configurations

### `device_ebpf.py` (stub)

- `DeviceEBPFProfiler` -- placeholder for future bpftime SPIR-V support
- Currently raises `NotImplementedError`
- Interim: use `rocprof --att` for per-instruction profiling

### Metrics Integration

- `compare_ebpf_vs_cuda()` in `metrics.py` -- compare driver-level vs
  user-space switch latency measurements
- `MetricsCollector.export_to_json()` accepts optional eBPF metric dicts
- `HarnessResult` carries `ebpf_queue_metrics`, `ebpf_memory_metrics`,
  and `ebpf_vs_cuda` fields

## Standalone Testing (No Code Changes)

You can test eBPF tracing independently before running through aorta:

```bash
# Trace GPU command submissions (all processes)
sudo bpftrace -e '
  tracepoint:amdgpu:amdgpu_cs_ioctl {
    printf("%s [%d] ring=%d\n", comm, pid, args->ring);
  }
'

# Trace job dispatch
sudo bpftrace -e '
  tracepoint:amdgpu:amdgpu_sched_run_job {
    printf("dispatch [%d] ring=%d seqno=%d\n", pid, args->ring, args->seqno);
  }
'

# Trace memory evictions
sudo bpftrace -e '
  tracepoint:amdkfd:kfd_evict_process {
    printf("EVICT pid=%d\n", pid);
  }
  tracepoint:amdkfd:kfd_restore_process {
    printf("RESTORE pid=%d\n", pid);
  }
'
```

## Future Work

- **Device-side eBPF**: When bpftime adds SPIR-V/GCN backend for AMD,
  `DeviceEBPFProfiler` will enable per-CU, per-warp profiling
- **gpu_ext struct_ops**: When AMD's amdgpu driver exposes eBPF struct_ops
  hooks (following the gpu_ext design), aorta's policy evaluator can test
  programmable scheduling/eviction policies directly
- **Chrome timeline**: Merge eBPF kernel events with user-space CUDA events
  into a unified Chrome tracing timeline
