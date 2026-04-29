# Using eBPF with AORTA

This guide walks through setting up and using eBPF-based GPU tracing in AORTA.
eBPF gives you kernel-level visibility into AMD GPU behaviour -- hardware queue
dispatch, memory management, and scheduling -- that user-space tools like the
PyTorch profiler and CUDA events cannot see.

For module-level API details and architecture diagrams, see
[ebpf-integration.md](ebpf-integration.md).

## When to Use eBPF Tracing

| Scenario | Recommended Tool | Why |
|----------|-----------------|-----|
| Measure kernel launch latency | PyTorch profiler / CUDA events | Low overhead, no root needed |
| **Diagnose hardware queue contention** | **eBPF queue tracing** | Sees real HW ring assignments |
| **Investigate memory pressure / evictions** | **eBPF memory tracing** | Catches driver-level evicts invisible to PyTorch |
| Compare scheduling policies | Policy sweep | Automates rocm-smi + env knobs |
| Per-instruction CU utilization | `rocprof --att` | Device-level, not yet eBPF-based |

Use eBPF when:
- Throughput stops scaling with stream count and you need to confirm whether
  hardware queues are the bottleneck.
- You see unexplained latency spikes that don't show up in CUDA event timings.
- Multi-tenant GPU workloads experience intermittent slowdowns from memory
  evictions.

## Environment Setup

### Option A: Docker (Recommended)

The eBPF Docker image comes with `bpftrace`, BCC tools, and all ROCm
dependencies pre-installed.

```bash
cd docker
bash setup-env.sh          # select option 5: Dockerfile.rocm-ubuntu-ebpf
docker compose -f docker-compose.build.yaml up -d
docker exec -it <your-container-name> bash
```

The container runs with `privileged: true` and `SYS_PTRACE`, which are
required for eBPF tracepoint attachment.

### Option B: Bare-Metal / Existing Container

Install the prerequisites manually:

```bash
# Ubuntu / Debian
sudo apt-get install -y bpftrace bpfcc-tools linux-tools-common

# RHEL / Fedora
sudo dnf install -y bpftrace bcc-tools

# Verify
bpftrace --version
```

Ensure `debugfs` is mounted (usually automatic):

```bash
mount | grep debugfs
# If not mounted:
sudo mount -t debugfs none /sys/kernel/debug
```

### Verify eBPF Readiness

Run AORTA's built-in diagnostic:

```bash
python -m aorta.hw_queue_eval ebpf-info
```

Expected output on a properly configured system:

```
eBPF Capabilities
==================================================

Kernel version:    6.8.0-90-generic
bpftrace:          bpftrace v0.21.0
Root/CAP_BPF:      yes
Overall available: yes

amdgpu tracepoints:
  - amdgpu_cs_ioctl
  - amdgpu_sched_run_job
  - amdgpu_vm_bo_map
  - amdgpu_vm_bo_unmap

amdkfd tracepoints:
  - kfd_evict_process_worker_start
  - kfd_restore_process_worker_start
  - kfd_map_memory_to_gpu_start
  - kfd_map_memory_to_gpu_end
```

> Older kernels exposed these as ``kfd_evict_process`` /
> ``kfd_restore_process``; aorta's tracer auto-detects which variant is
> available at startup.  The standalone bpftrace scripts under
> ``ebpfaultline/bpftrace/`` use the same naming convention.

If `bpftrace` shows "not installed" or tracepoints show "(not accessible)",
see [Troubleshooting](#troubleshooting) below.

## Queue Tracing

Queue tracing attaches to `amdgpu_cs_ioctl` (command submission) and
`amdgpu_sched_run_job` (hardware dispatch) to measure the real
submit-to-dispatch latency per hardware ring.

### Single Run

```bash
sudo python -m aorta.hw_queue_eval run hetero_kernels \
    --streams 8 --ebpf-trace
```

This produces normal benchmark output plus an additional section:

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

The **eBPF vs CUDA Comparison** section cross-references driver-level dispatch
timing with user-space CUDA event measurements, helping you understand how much
of the observed switch overhead comes from actual hardware queue dispatch versus
API/driver overhead.

### Stream Sweep with eBPF

Run across multiple stream counts to find where hardware queue saturation
begins:

```bash
sudo python -m aorta.hw_queue_eval sweep hetero_kernels \
    --streams 1,2,4,8,16,32 --ebpf-trace
```

Look for the point where `Submit→dispatch P99` increases sharply -- that
indicates the hardware queue limit.

### Interpreting Queue Metrics

| Metric | Good | Potential Problem |
|--------|------|-------------------|
| Submit→dispatch avg | < 20 us | > 100 us indicates queue contention |
| Submit→dispatch P99 | < 3x avg | > 10x avg suggests scheduling stalls |
| HW rings used | Matches stream count (up to HW limit) | Fewer rings than streams = multiplexing |
| eBPF vs CUDA accuracy | > 70% | < 50% suggests events are being missed |

## Memory Tracing

Memory tracing captures buffer object (BO) map/unmap events and process
eviction/restore events that indicate GPU memory pressure.

```bash
sudo python -m aorta.hw_queue_eval run hetero_kernels \
    --streams 8 --ebpf-memory-trace
```

Output includes:

```
eBPF MEMORY METRICS:
  BO moves (migrations): 42  (350 /sec)
  Migration volume:      64.0 MB
  BO maps / unmaps:      120 / 100
  Evictions / restores:  4 / 4
  Eviction rate:         0.5 /sec
  Avg evict latency:     8.2 us
```

> Note: the historical "Page faults" terminology refers to KFD
> eviction/restore cycles, *not* GPU UVM page faults.  This module does
> not currently attach to UVM fault tracepoints; the eviction/restore
> counts here signal driver-level memory pressure (the GPU OOM'd, the
> process was paged out, then brought back).

### Combining Queue and Memory Tracing

Both flags can be used together:

```bash
sudo python -m aorta.hw_queue_eval run moe \
    --streams 16 --ebpf-trace --ebpf-memory-trace
```

This is useful when investigating whether memory migrations are contributing
to queue dispatch delays.

### Interpreting Memory Metrics

| Metric | What It Means |
|--------|---------------|
| BO moves | Buffer-object migrations between memory domains (`amdgpu_bo_move`) |
| Migration volume | Total bytes moved by `amdgpu_bo_move` events |
| BO maps / unmaps | Counts of `amdgpu_vm_bo_map` / `amdgpu_vm_bo_unmap` |
| Evictions / restores | Process-level KFD eviction/restore worker invocations |
| Eviction rate | Evictions per second -- if non-zero, GPU memory is oversubscribed |
| Avg evict latency | Time between matched evict -> restore pairs (high values = thrash) |

> The JSON output also exposes a legacy ``total_faults`` /
> ``fault_rate_per_sec`` / ``avg_fault_latency_us`` aliases for the
> eviction/restore-pair metric.  These are *not* GPU UVM page faults.

## Policy Sweep

The policy sweep command evaluates workload performance under different
scheduling and memory configurations. Each policy adjusts GPU clock levels,
power limits, and/or environment variables, then runs the workload and
compares results.

### Basic Usage

```bash
# Default policies: baseline, priority_lc, priority_be
python -m aorta.hw_queue_eval policy-sweep hetero_kernels --streams 8
```

### Selecting Specific Policies

```bash
python -m aorta.hw_queue_eval policy-sweep moe \
    --streams 16 \
    --policies baseline,priority_lc,high_queue,xnack_off \
    -o policy_results.json
```

### Available Built-in Policies

**Scheduling policies** (adjust clock/power/queue configuration):

| Policy | Clock Level | Power Limit | HW Queues | Description |
|--------|------------|-------------|-----------|-------------|
| `baseline` | default | default | default | No constraints, round-robin scheduling |
| `priority_lc` | 7 (max) | -- | -- | Latency-critical: maximum clock speed |
| `priority_be` | 2 (low) | 150 W | -- | Best-effort: reduced resources |
| `multi_tenant_fair` | -- | -- | 2 | Fair sharing via `GPU_MAX_HW_QUEUES=2` |
| `high_queue` | -- | -- | 8 | Maximum HW queues via `GPU_MAX_HW_QUEUES=8` |

**Memory policies** (adjust XNACK / page fault behaviour):

| Policy | HSA_XNACK | Description |
|--------|-----------|-------------|
| `default_uvm` | 1 | Retryable page faults enabled (unified memory) |
| `xnack_off` | 0 | No retryable page faults |

### Reading Policy Sweep Output

The output is a comparison table showing throughput, latency, and switch
overhead for each policy. Look for:

- **Throughput delta**: policies that improve throughput indicate the default
  configuration is suboptimal for your workload.
- **Latency P99 changes**: `priority_lc` should reduce tail latency;
  `multi_tenant_fair` may increase it.
- **Clock level vs throughput trade-off**: `priority_be` uses less power --
  check if the throughput drop is acceptable.

## Saving and Exporting Results

### JSON Output

All commands accept `-o <file.json>` to save results. When eBPF tracing is
enabled, the JSON output includes the eBPF metrics alongside standard
benchmark data:

```bash
sudo python -m aorta.hw_queue_eval run hetero_kernels \
    --streams 8 --ebpf-trace --ebpf-memory-trace \
    -o results_ebpf.json
```

The JSON file will contain `ebpf_queue_metrics`, `ebpf_memory_metrics`, and
`ebpf_vs_cuda_comparison` sections in addition to the standard throughput and
latency data.

### Comparing Runs

Use the `compare` command to diff two result files:

```bash
python -m aorta.hw_queue_eval compare \
    -b results_baseline.json -t results_ebpf.json
```

## Standalone bpftrace (Without AORTA)

You can test eBPF tracing independently to validate your system or
investigate GPU behaviour outside of AORTA benchmarks.

### Trace GPU Command Submissions

```bash
sudo bpftrace -e '
  tracepoint:amdgpu:amdgpu_cs_ioctl {
    printf("%s [%d] ring=%d\n", comm, pid, args->ring);
  }
'
```

Run a GPU workload in another terminal and watch submissions appear in
real time.

### Trace Job Dispatch

```bash
sudo bpftrace -e '
  tracepoint:amdgpu:amdgpu_sched_run_job {
    printf("dispatch [%d] ring=%d seqno=%d\n", pid, args->ring, args->seqno);
  }
'
```

### Trace Memory Evictions

```bash
sudo bpftrace -e '
  tracepoint:amdkfd:kfd_evict_process {
    printf("EVICT pid=%d\n", pid);
  }
  tracepoint:amdkfd:kfd_restore_process {
    printf("RESTORE pid=%d\n", pid);
  }
'
```

### Measure Submit-to-Dispatch Latency (Histogram)

```bash
sudo bpftrace -e '
  tracepoint:amdgpu:amdgpu_cs_ioctl {
    @submit[tid] = nsecs;
  }
  tracepoint:amdgpu:amdgpu_sched_run_job {
    if (@submit[tid]) {
      @latency_us = hist((nsecs - @submit[tid]) / 1000);
      delete(@submit[tid]);
    }
  }
  END { clear(@submit); }
'
```

Press Ctrl+C to stop and see the histogram.

## End-to-End Workflow Example

A typical investigation workflow combining AORTA's profiling layers:

```bash
# 1. Establish baseline without eBPF
python -m aorta.hw_queue_eval run hetero_kernels --streams 8 -o baseline.json

# 2. Run with eBPF to get driver-level data
sudo python -m aorta.hw_queue_eval run hetero_kernels \
    --streams 8 --ebpf-trace --ebpf-memory-trace -o ebpf_run.json

# 3. Compare to see if eBPF overhead affects results
python -m aorta.hw_queue_eval compare -b baseline.json -t ebpf_run.json

# 4. Sweep stream counts with eBPF to find HW queue saturation
sudo python -m aorta.hw_queue_eval sweep hetero_kernels \
    --streams 1,2,4,8,16,32 --ebpf-trace -o sweep_ebpf.json

# 5. Try different scheduling policies
python -m aorta.hw_queue_eval policy-sweep hetero_kernels \
    --streams 8 --policies baseline,priority_lc,high_queue \
    -o policy_comparison.json

# 6. Lock clocks for deterministic measurements
sudo python -m aorta.hw_queue_eval run hetero_kernels \
    --streams 8 --lock-clocks 7 --ebpf-trace -o locked_clocks.json
```

## Troubleshooting

### "bpftrace: not installed"

Install bpftrace for your distribution:

```bash
# Ubuntu/Debian
sudo apt-get install -y bpftrace

# RHEL/Fedora
sudo dnf install -y bpftrace
```

Or use the eBPF Docker image (option 5 in `setup-env.sh`).

### "tracepoints not accessible"

Tracepoints require `debugfs` to be mounted and root access:

```bash
# Mount debugfs if not already mounted
sudo mount -t debugfs none /sys/kernel/debug

# Verify tracepoints exist
sudo ls /sys/kernel/debug/tracing/events/amdgpu/
sudo ls /sys/kernel/debug/tracing/events/amdkfd/
```

If the directories don't exist, the `amdgpu` / `amdkfd` kernel modules may
not be loaded:

```bash
lsmod | grep amdgpu
# If empty, load the driver:
sudo modprobe amdgpu
```

### "Permission denied" or "CAP_BPF required"

eBPF tracepoint attachment requires root or `CAP_BPF`:

```bash
# Run with sudo
sudo python -m aorta.hw_queue_eval run hetero_kernels --streams 8 --ebpf-trace

# Or grant CAP_BPF to bpftrace (persistent)
sudo setcap cap_bpf,cap_perfmon+ep $(which bpftrace)
```

Inside Docker, ensure the container runs with `--privileged` or the
equivalent compose setting (the eBPF Dockerfile's compose config already
includes this).

### eBPF Tracing Shows Zero Events

- Confirm a GPU workload is actually running during the trace window.
- Check that your workload uses the AMD GPU (not a CPU fallback):
  `python -c "import torch; print(torch.cuda.is_available())"`
- The tracer targets the PID of the Python process by default. If running
  under `torchrun`, use the `--ebpf-trace` flag (which passes the correct
  PID automatically) rather than standalone `bpftrace`.

### High eBPF Overhead

eBPF tracing adds minimal overhead (typically < 2%), but on very
high-throughput micro-benchmarks with tiny kernels, you may see measurable
impact. Mitigations:

- Use `--ebpf-trace` or `--ebpf-memory-trace` separately rather than both
  at once.
- Increase `--iterations` to amortize startup/teardown cost.
- Run a baseline without eBPF and use `compare` to quantify the delta.

## Next Steps

- [Profiling Guide](profiling.md) -- PyTorch profiler and rocprof workflows
- [eBPF Integration Reference](ebpf-integration.md) -- Module API and
  architecture details
- [Running the Benchmark](running-benchmark.md) -- General benchmark usage
- [Troubleshooting](troubleshooting.md) -- Common issues and solutions
