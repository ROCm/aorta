# Getting Started

This guide covers prerequisites, installation, and initial setup for AORTA.

## Prerequisites

- PyTorch >= 2.2 with FSDP2 APIs (ROCm 7/RCCL)
- ROCm tooling (`rocm-smi`, `rocminfo`)
- PyYAML, matplotlib
- GPU nodes with RCCL capable interconnects
- Sufficient GPU memory for the configured model (see `config/default.yaml`)

## Key Assumptions

- TorchTitan components required by your wider stack are pre-installed (the synthetic workload does not import TorchTitan directly).
- The code gracefully degrades when optional dependencies are absent.
- All processes run under a job launcher that sets `LOCAL_RANK` (e.g., `torchrun`, Slurm, or similar).
- The synthetic dataset is intended for profiling and does not reflect production data distributions.

## Docker Setup (Recommended for Training)

Training runs in Docker containers with all dependencies pre-installed.

### Quick Start

#### Option 1: Using the Setup Script (Recommended for First-Time Users)

The interactive setup script guides you through creating your personal `.env` configuration:

```bash
cd docker
bash setup-env.sh
docker compose -f docker-compose.build.yaml up -d
```

The script will prompt you to:
- Select a Dockerfile (ROCm version, with/without Shampoo optimizer, etc.)
- Choose a container name (defaults to `${USER}-${variant}-${date}`)
- Configure workspace and RCCL paths
- Set up optional volume mounts

#### Option 2: Manual .env Configuration

For more control, manually create your `.env` file:

```bash
cd docker
cp .env.example .env
# Edit .env with your preferred editor
nano .env  # or vim, code, etc.
docker compose -f docker-compose.build.yaml up -d
```

**Available Dockerfiles:**
- `Dockerfile.rocm70_9-1` - Standard ROCm 7.0.9.1 build
- `Dockerfile.rocm70_9-1-shampoo` - ROCm 7.0.9.1 with Shampoo optimizer
- `Dockerfile.rocm70_2-ubuntu-pytorch` - ROCm 7.0.2 Ubuntu PyTorch build
- `Dockerfile.rocm70_2-ubuntu-nan` - ROCm 7.0.2 with NaN debugging tools
- `Dockerfile.rocm-ubuntu-ebpf` - ROCm 7.2 with eBPF tracing tools (bpftrace, bcc)

**Example `.env` configurations:**

For standard ROCm development:
```bash
DOCKERFILE=Dockerfile.rocm70_9-1
CONTAINER_NAME=myuser-rocm70-dev
AORTA_WORKSPACE=..
RCCL_PATH=/tmp/rccl_placeholder
```

For Shampoo optimizer testing with custom RCCL:
```bash
DOCKERFILE=Dockerfile.rocm70_9-1-shampoo
CONTAINER_NAME=myuser-shampoo-exp1
AORTA_WORKSPACE=/apps/username/aorta_work/aorta_1
RCCL_PATH=/apps/username/rccl
```

#### Option 3: Pre-built Image (Alternative)

If you prefer using a pre-built image instead of building from a Dockerfile:

```bash
cd docker
docker compose up -d
```

This uses the default `docker-compose.yaml` with a pre-configured image.

### Connecting to Your Container

Connect to the running container via CLI or VSCode:

```bash
# Via Docker CLI
docker exec -it <your-container-name> bash

# Or use VSCode's "Attach to Running Container" feature
```

### Running TorchRec Benchmark

```bash
python -m torchrec.distributed.benchmark.benchmark_train_pipeline \
  --yaml_config=$ROOT/config/torchrec_dist/sparse_data_dist_base.yaml \
  --name="sparse_data_dist_q_contend$(git rev-parse --short HEAD || echo $USER)"
```

This captures a profiler trace file locally.

**What runs in Docker:**
- `train.py` - Model training
- Distributed workloads
- GPU profiling

## Local Installation (Analysis & Processing)

For running analysis scripts and processing traces locally.

We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable Python environment management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/ROCm/aorta.git
cd aorta

# Create and activate a virtual environment
uv venv && source .venv/bin/activate

# Install PyTorch nightly for ROCm 7.1
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.1/

# Install dependencies for analysis scripts
uv pip install -r requirements.txt

# For contributors: install development tools (pytest, pre-commit, etc.)
uv pip install -r requirements-dev.txt
pre-commit install
```

**What runs locally:**
- `scripts/utils/merge_gpu_trace_ranks.py` - Merge distributed traces
- `analysis/overlap_report.py` - Generate analysis reports
- `scripts/analyze_*.py` - Analysis utilities
- Test suite (`pytest tests/`)

## Additional Notes

- On ROCm systems, verify `rocm-smi` and `rocminfo` are in `$PATH`.
- Run scripts from the repository root so path bootstrapping works correctly.

## Next Steps

- [Running the Benchmark](running-benchmark.md) - Launch your first training run
- [Configuration Guide](configuration.md) - Customize model and training parameters
- [eBPF Usage Guide](ebpf-usage-guide.md) - Kernel-level GPU queue and memory tracing
