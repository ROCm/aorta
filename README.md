# AORTA

GPU performance benchmarking and debugging toolkit for PyTorch workloads on AMD ROCm.

![Training Overlap Issue](analysis/figures/training_bad_overlap.png)

## What It Does

**FSDP2 Compute-Communication Overlap Analysis**
Debug why distributed training isn't overlapping compute with communication. Runs a synthetic transformer workload with explicit multi-stream execution, captures per-iteration timing, and generates overlap efficiency reports.

![param_sweep](docs/param_sweep.png)

**Hardware Queue Evaluation**
Stress-test GPU queue scheduling with 8-64+ concurrent streams. Includes 15 workloads covering distributed training patterns (FSDP, MoE, activation checkpointing), inference (speculative decoding, continuous batching), and latency-sensitive scenarios (heterogeneous kernels, tiny kernel dispatch).

![hw_queue_cmds](docs/hw_queue_cmds.png)

**Environment Snapshot for Reproducibility**
Capture a versioned, schema-stable snapshot of the trial environment — ROCm / HIP / hipBLASLt / rocBLAS / MIOpen / RCCL identities, GPU arch, PyTorch build flags + cmake cache + per-target HIPCC defines, runtime SDPA backend state, ~30 numerics-relevant env vars — so cross-environment regressions become a `jq` diff instead of a multi-day investigation. Used standalone (`aorta env probe`) and embedded automatically into every trial result.

**LLM Determinism Probe**
Catch kernel-level nondeterminism / silent data corruption in a transformer training step. Runs the same forward+backward twice on the same inputs (params + RNG restored between runs) and compares bit-exact checksums of every per-block boundary activation, loss, logits, every grad, and every param. FSDP2-aware, RCCL-safe, optional MoE. See [`docs/llm-determinism.md`](docs/llm-determinism.md) for the quick start.


## Quick Start

```bash
# FSDP2 overlap benchmark
bash scripts/launch_rocm.sh config/default.yaml

# Hardware queue evaluation
python -m aorta.hw_queue_eval list                          # List workloads
python -m aorta.hw_queue_eval run hetero_kernels --streams 8
python -m aorta.hw_queue_eval sweep hetero_kernels --streams 1,2,4,8,16

# Comm-compute overlap (simulated collectives)
python -m aorta.hw_queue_eval run comms_compute_overlap --streams 4 --profile

# Comm-compute overlap (real NCCL collectives via torchrun)
torchrun --nproc_per_node=8 -m aorta.hw_queue_eval run comms_compute_overlap \
    --streams 4 --real-collectives --async-op --backend nccl \
    --process-groups "[0,1,2,3,4,5,6,7]" --profile --profile-dir traces/

# Environment snapshot for reproducibility
aorta env probe -o env.json                               # full snapshot to disk
aorta env probe --summary                                 # one-screen brief, no file write
aorta env probe --field pytorch_build.git_commit          # one field, JSON-typed
diff <(jq -S . env_a.json) <(jq -S . env_b.json)          # diff two snapshots
```

## Example Analysis

AORTA generates comprehensive performance reports comparing ROCm versions across multiple configurations. See a [full example report](docs/comprehensive_report.html) comparing rocm-7.0.8-meta vs rocm-7.0.10-meta:

- **8 configurations tested**: 256/512 threads × 28/42/56/70 RCCL channels
- **96 visualizations**: Overlap ratios, GEMM throughput, NCCL metrics, timeline comparisons
- **Side-by-side diffs**: Identify regressions or improvements between driver/library versions

![Overlap Breakdown](analysis/figures/overlap_breakdown.png)

## Documentation

| Guide | Description |
| --- | --- |
| [Getting Started](docs/getting-started.md) | Prerequisites, Docker setup, installation |
| [Running the Benchmark](docs/running-benchmark.md) | Launch scripts, torch.compile, direct invocation |
| [Hardware Queue Eval](docs/hw-queue-eval.md) | Workloads, CLI usage, metrics |
| [Configuration](docs/configuration.md) | FSDP tuning, RCCL variables, profiler settings |
| [Profiling](docs/profiling.md) | Torch profiler, rocprofv3, overlap reports |
| [Environment Probe](docs/env-probe.md) | Capture / diff / query a versioned environment snapshot; jq cookbook |
| [`aorta probe`](docs/probe-188/usage.md) | Wrap-and-collect opaque launch commands; matrix + classifier |
| [`aorta agent`](docs/agent/agentic-testing-guide.md) | Closed-loop mitigation search (optional LLM proposer) |
| [`aorta bundle`](docs/probe-188/bundle.md) | Package probe artifacts with recipe-driven redaction |
| [Buck2](docs/buck2.md) | Build / run the AORTA CLI via Buck2; hermetic Python, `env probe` + `triage` walkthroughs |
| [Releasing](docs/releasing.md) | Cut a customer-installable release; maintainer + customer install flow |
| [Troubleshooting](docs/troubleshooting.md) | Common issues |

## Repository Layout

```
src/aorta/
├── training/          # FSDP2 trainer with multi-stream overlap instrumentation
├── hw_queue_eval/     # Hardware queue evaluation framework
├── models/            # Synthetic ranking transformer
├── profiling/         # Stream profiler for overlap measurement
├── instrumentation/   # Environment probe (versioned env.json schema + capture)
├── registry/          # Mitigations + environments registry (extension points)
├── cli/               # `aorta` CLI command groups (run, env probe, ...)
└── utils/             # Config loading, timing, device detection

config/                # YAML configurations for different scenarios
scripts/               # Launch scripts, profiling, analysis tools
analysis/              # Overlap report generation
```

## Installation

### From a release (recommended for users)

Tagged [releases](https://github.com/ROCm/aorta/releases) ship a prebuilt wheel
you can `pip install` directly. PyTorch is installed separately from the ROCm
index (it is not bundled in the wheel). Replace `X.Y.Z` below with the release
version you want — the newest is tagged **Latest** on the
[releases page](https://github.com/ROCm/aorta/releases):

```bash
# Install PyTorch for your ROCm version (adjust the index URL accordingly)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.1/

# Install AORTA from the release (pin the version you want)
pip install "amd-aorta @ https://github.com/ROCm/aorta/releases/download/vX.Y.Z/amd_aorta-X.Y.Z-py3-none-any.whl"

# With optional extras, e.g. the hardware-queue tools
pip install "amd-aorta[hw-queue] @ https://github.com/ROCm/aorta/releases/download/vX.Y.Z/amd_aorta-X.Y.Z-py3-none-any.whl"
```

Maintainers: see [docs/releasing.md](docs/releasing.md) for how releases are cut.

### From source (for development)

We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable Python environment management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv && source .venv/bin/activate

# Install PyTorch nightly for ROCm 7.1
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.1/

# Install remaining dependencies
uv pip install -r requirements.txt

# For full installation including hw_queue_eval
uv pip install -e ".[hw-queue]"
```

## Development

```bash
uv pip install -r requirements-dev.txt
pre-commit install
pytest tests/
```

---

*The FSDP2 overlap workloads also run on NVIDIA CUDA for side-by-side comparison with ROCm.*
