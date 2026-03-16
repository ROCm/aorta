# ROCm GPU Memory Access Debugger

Standalone tool for diagnosing intermittent **illegal memory access** (`hipErrorIllegalAddress`) errors on AMD GPUs using eBPF kernel tracing + dmesg correlation.

**Zero dependencies** beyond Python 3.8+ stdlib and `bpftrace`.
**No AORTA imports** -- this is a self-contained, customer-deployable script.

---

## Quick Start

```bash
# 1. System check (no tracing, safe to run first)
sudo python3 rocm_mem_debug.py --check-only

# 2. Trace all GPU memory activity for 2 minutes
sudo python3 rocm_mem_debug.py --duration 120

# 3. Trace a specific process and save report
sudo python3 rocm_mem_debug.py --pid <PID> --duration 300 --output report.json
```

---

## Prerequisites

| Requirement      | How to check                | Install                                      |
|------------------|-----------------------------|----------------------------------------------|
| Linux kernel     | `uname -r`                  | (already installed)                           |
| amdgpu driver    | `lsmod \| grep amdgpu`     | Comes with ROCm                              |
| bpftrace         | `bpftrace --version`        | `apt install bpftrace` / `dnf install bpftrace` |
| Root privileges  | `whoami`                    | Use `sudo`                                   |
| Python 3.8+      | `python3 --version`         | (already installed on most distros)           |
| debugfs mounted  | `ls /sys/kernel/debug/tracing/events/amdgpu/` | `mount -t debugfs debugfs /sys/kernel/debug` |

Optional (for integration tests only):

| Requirement      | How to check                | Install                                      |
|------------------|-----------------------------|----------------------------------------------|
| hipcc            | `hipcc --version`           | Comes with ROCm                              |
| AMD GPU hardware | `rocm-smi`                  | (physical GPU required)                       |

---

## How It Works

The script runs **two concurrent monitors** and correlates their output:

1. **eBPF (bpftrace)** -- Traces kernel-level GPU memory operations via tracepoints:
   - `amdgpu:amdgpu_bo_move` -- Buffer Object migrations (VRAM <-> GTT <-> System)
   - `amdgpu:amdgpu_vm_bo_map` / `amdgpu_vm_bo_unmap` -- Virtual memory mappings
   - `amdgpu:amdgpu_vm_set_ptes` -- Page table entry updates
   - `amdgpu:amdgpu_iv` -- GPU interrupt vectors (includes fault interrupts)
   - `amdkfd:kfd_evict_process_worker_start` -- Process eviction (memory pressure)
   - `amdkfd:kfd_restore_process_worker_start` -- Process restore after eviction
   - `amdkfd:kfd_map_memory_to_gpu_start/end` -- KFD memory mapping operations

2. **dmesg** -- Watches the kernel log for GPU fault messages:
   - `GPU fault`, `VM fault`, `page fault`
   - `illegal` (illegal memory access)
   - `Xnack` / `xnack`
   - `ECC`, `RAS error`, `MCE`
   - `ring * timeout`, `gpu_recover`

GPU page faults and illegal memory access errors do **not** appear as eBPF tracepoints -- they are only visible in `dmesg`. However, the *precursor events* (BO migrations, evictions, VM mappings, interrupt bursts) **are** traceable. The script correlates both sources: when a dmesg fault appears, it looks at what kernel-level GPU events happened in the preceding 500ms to identify the root cause.

---

## Verifying eBPF Tracing Capabilities

Before deploying at a customer site, verify that the script can actually capture events on your system.

### Step 1: System Check

```bash
sudo python3 rocm_mem_debug.py --check-only
```

Expected output (on a properly configured system):

```
============================================================
AMD GPU MEMORY ACCESS DEBUGGER -- SYSTEM CHECK
============================================================

  Kernel:           6.8.0-90-generic
  bpftrace:         bpftrace v0.20.1
  Root/CAP_BPF:     yes
  ROCm (rocm-smi):  rocm-smi 6.3.0
  hipcc:            HIP version: 6.3.42133-cacfa6654
  HSA_XNACK:        not set

  GPU:
    GPU[0] : AMD Instinct MI300X

  VRAM:
    GPU[0] : VRAM Total Memory (B): 196592746496
    GPU[0] : VRAM Total Used Memory (B): 3145728

  amdgpu tracepoints: 15 found
    - amdgpu_bo_move
    - amdgpu_cs_ioctl
    - amdgpu_iv
    - amdgpu_sched_run_job
    - amdgpu_vm_bo_map
    - amdgpu_vm_bo_unmap
    - amdgpu_vm_set_ptes
    ...
  amdkfd tracepoints: 8 found
    - kfd_evict_process_worker_start
    - kfd_map_memory_to_gpu_end
    - kfd_map_memory_to_gpu_start
    - kfd_restore_process_worker_start
    ...

  System is ready for GPU memory debugging.
```

**What to check:**

- **bpftrace** is installed and shows a version
- **Root/CAP_BPF** shows `yes`
- **amdgpu tracepoints** shows > 0 found (especially `amdgpu_bo_move`, `amdgpu_vm_bo_map`, `amdgpu_iv`)
- **amdkfd tracepoints** shows > 0 found (especially `kfd_evict_process_worker_start`)
- Status line says "System is ready for GPU memory debugging"

**If tracepoints show 0:**

```bash
# Mount debugfs if not already mounted
sudo mount -t debugfs debugfs /sys/kernel/debug

# Verify tracepoints exist
ls /sys/kernel/debug/tracing/events/amdgpu/
ls /sys/kernel/debug/tracing/events/amdkfd/
```

### Step 2: Verify Event Capture with a Short Trace

Run a short capture while a GPU workload is active:

```bash
# In terminal 1: Start any GPU workload (e.g., a PyTorch training script)
python3 train.py &
TRAIN_PID=$!

# In terminal 2: Capture 10 seconds of GPU events
sudo python3 rocm_mem_debug.py --pid $TRAIN_PID --duration 10 --output verify.json --verbose
```

With `--verbose`, you should see individual events scrolling:

```
  [BO_MOVE] pid=12345 comm=python3
  [VM_MAP] pid=12345 comm=python3
  [GPU_IRQ] pid=12345 comm=python3
  ...
```

After 10 seconds, the report should show non-zero event counts:

```
  EVENT COUNTS:
    BO_MOVE                    47
    GPU_IRQ                   312
    VM_MAP                     23
    VM_UNMAP                   19
```

**If event counts are all zero:**

1. Confirm amdgpu driver is loaded: `lsmod | grep amdgpu`
2. Confirm GPU activity is happening: `rocm-smi` should show non-zero GPU utilization
3. Check if bpftrace can attach: `sudo bpftrace -e 'tracepoint:amdgpu:amdgpu_bo_move { printf("ok\n"); exit(); }'`
4. Try without `--pid` filter (trace all processes)

### Step 3: Verify dmesg Monitoring

The dmesg monitor watches for GPU fault keywords. To verify it works:

```bash
# In one terminal, run the debugger
sudo python3 rocm_mem_debug.py --duration 30

# In another terminal, inject a test message (requires root)
echo "amdgpu: GPU fault detected on ring 0" | sudo tee /dev/kmsg
```

You should see the fault highlighted in the debugger output:

```
  FAULT: amdgpu: GPU fault detected on ring 0
```

And the final report should show `FAULTS DETECTED: 1`.

### Step 4: Verify JSON Report Output

```bash
sudo python3 rocm_mem_debug.py --duration 5 --output test_report.json
python3 -c "import json; r = json.load(open('test_report.json')); print(json.dumps(r, indent=2))"
```

The JSON report should contain:
- `system` -- kernel, ROCm version, GPU info, XNACK status
- `event_counts` -- per-event-type counters
- `faults_detected` -- number of dmesg faults seen
- `patterns` -- detected anomaly patterns (eviction storm, IRQ spike, etc.)
- `recommendations` -- actionable suggestions

### Step 5: Run Integration Tests (requires hipcc + GPU)

The `tests/` directory contains HIP C++ programs that deliberately trigger (and avoid) illegal memory access patterns.

```bash
cd tests/

# Build all test programs
make all

# Run the full test suite (requires root + GPU)
sudo ./run_tests.sh

# Run only positive tests (fault-triggering)
sudo ./run_tests.sh --skip-negative

# Run only negative tests (clean programs)
sudo ./run_tests.sh --skip-positive

# Skip build step (if already compiled)
sudo ./run_tests.sh --no-build
```

**Positive tests** (should trigger detection):

| Test | What it does | Expected detection |
|------|-------------|-------------------|
| `01_use_after_free` | `hipFree` then kernel reads freed ptr | dmesg fault + IRQ burst |
| `02_oob_write` | Kernel writes beyond allocation | dmesg fault |
| `03_null_ptr_deref` | Kernel writes to nullptr | dmesg fault |
| `04_eviction_storm` | Fills 95% VRAM then allocates more | `eviction_storm` pattern |
| `05_unsynced_streams` | Two streams, shared buffer, no sync | Concurrent access (race-dependent) |
| `06_large_alloc_copy` | 5 GB copy (tests >4GB ROCm bug) | Large migration volume |

**Negative tests** (should produce zero false positives):

| Test | What it does | Expected |
|------|-------------|----------|
| `01_normal_alloc` | Standard alloc/compute/free | Clean, 0 faults |
| `02_synced_streams` | Two streams with proper event sync | Clean, 0 faults |
| `03_correct_h2d_d2h` | Standard H2D/D2H copy pattern | Clean, 0 faults |
| `04_large_clean_alloc` | 2 GB alloc (under 4GB boundary) | Clean, 0 faults |

---

## Deployment at Customer Site

### Minimal deployment (just the debugger)

Copy `rocm_mem_debug.py` to the customer machine. That's it -- single file, no dependencies.

```bash
scp rocm_mem_debug.py customer-machine:~/

# On the customer machine:
sudo python3 rocm_mem_debug.py --check-only          # verify setup
sudo python3 rocm_mem_debug.py --duration 600         # 10 min capture
sudo python3 rocm_mem_debug.py --pid $(pgrep python3) --duration 300 -o report.json
```

### Attaching to a running training job

The script can attach to an already-running process without restart:

```bash
# Find the training process PID
pgrep -f "python.*train"

# Attach and capture 5 minutes
sudo python3 rocm_mem_debug.py --pid <PID> --duration 300 --output diag.json
```

This works because bpftrace attaches to kernel tracepoints for any PID at any time -- no process restart or code changes needed.

### Recommended environment variables to try

If faults are detected, the report's `RECOMMENDATIONS` section will suggest environment variable changes. Common ones:

```bash
# Toggle XNACK to isolate page fault behavior
HSA_XNACK=0 python3 train.py   # disable retryable page faults
HSA_XNACK=1 python3 train.py   # enable retryable page faults

# Serialize GPU operations to narrow down races
AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3 python3 train.py

# Disable SDMA (use shader copies instead)
HSA_ENABLE_SDMA=0 python3 train.py
```

---

## Directory Structure

```
scripts/rocm_mem_debug/
├── README.md                 # This file
├── rocm_mem_debug.py         # Main debugger script (standalone, single file)
└── tests/                    # Integration tests (require hipcc + GPU)
    ├── Makefile              # hipcc build rules
    ├── run_tests.sh          # Test orchestrator
    ├── positive/             # Tests that trigger faults / anomalies
    │   ├── 01_use_after_free.cpp
    │   ├── 02_oob_write.cpp
    │   ├── 03_null_ptr_deref.cpp
    │   ├── 04_eviction_storm.cpp
    │   ├── 05_unsynced_streams.cpp
    │   └── 06_large_alloc_copy.cpp
    ├── negative/             # Tests that should run cleanly
    │   ├── 01_normal_alloc.cpp
    │   ├── 02_synced_streams.cpp
    │   ├── 03_correct_h2d_d2h.cpp
    │   └── 04_large_clean_alloc.cpp
    ├── build/                # Compiled test binaries (gitignored)
    └── results/              # Test output and reports (gitignored)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `bpftrace is not installed` | `apt install bpftrace` or `dnf install bpftrace` |
| `not running as root` | Use `sudo python3 rocm_mem_debug.py ...` |
| `no amdgpu tracepoints found` | `sudo mount -t debugfs debugfs /sys/kernel/debug` |
| `bpftrace failed to start` | Check `bpftrace --version`; may need kernel >= 4.18 |
| Event counts all zero | Verify GPU workload is active (`rocm-smi`); try without `--pid` |
| dmesg monitor fails | Some containers restrict `dmesg --follow`; script continues without it |
| `hipcc not found` (tests only) | Ensure ROCm is installed and `hipcc` is on PATH |
