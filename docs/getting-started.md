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

```bash
cd docker
docker compose up -d
```

Connect to the running container via CLI or VSCode.

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

For running analysis scripts and processing traces locally:

```bash
# Clone the repository
git clone https://github.com/ROCm/aorta.git
cd aorta

# Install dependencies for analysis scripts
pip install -r requirements.txt

# For contributors: install development tools (pytest, pre-commit, etc.)
pip install -r requirements-dev.txt
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
