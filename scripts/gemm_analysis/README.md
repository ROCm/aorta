# GEMM Sweep Profiling

Profile GEMM kernel performance across multiple NCCL configurations.

## Prerequisites

- Docker with ROCm support
- TraceLens installed

## Pipeline Steps

### 1. Build Docker Container

```bash
cd /home/oyazdanb/aorta/docker
docker-compose -f docker-compose.rocm70_9-1.yaml build
docker-compose -f docker-compose.rocm70_9-1.yaml up -d
docker exec -it training-overlap-bugs-rocm70_9-1 bash
```

### 2. Run Training Sweep

```bash
bash scripts/gemm_analysis/run_train_various_channels.sh \
  -c 28,42,56,70 \
  -t 256,512 \
  -f config/gemm_overlap/gemm_test_1.yaml
```

### 3. Generate TraceLens Reports

```bash
bash scripts/gemm_analysis/run_tracelens_analysis.sh /home/oyazdanb/aorta/experiments/sweep_20251124_222204
```

### 4. Extract Top GEMM Kernels

```bash
python scripts/gemm_analysis/analyze_gemm_reports.py \
  --base-path /home/oyazdanb/aorta/experiments/sweep_20251124_222204/tracelens_analysis \
  --threads 256 512 \
  --channels 28 42 56 70 \
  --ranks 0 1 2 3 4 5 6 7 \
  --top-k 5
```

This generates `top5_gemm_kernels_time_variance.csv` with the kernels showing highest time variance across runs.

## Output Structure

```
experiments/sweep_YYYYMMDD_HHMMSS/
├── 256thread/
│   └── nccl_XXchannels/
│       ├── torch_profiler/rank*/
│       └── run_output.log
├── 512thread/
│   └── nccl_XXchannels/
└── tracelens_analysis/
    ├── 256thread/
    │   ├── individual_reports/perf_*ch_rank*.xlsx
    │   └── collective_reports/collective_*ch.xlsx
    ├── 512thread/
    └── top5_gemm_kernels_time_variance.csv
```

## Quick Reference

```bash
# Run complete sweep
bash scripts/gemm_analysis/run_train_various_channels.sh \
  -c 28,42,56,70 -t 256,512 -f config/gemm_overlap/gemm_test_1.yaml

# Generate TraceLens reports
bash scripts/gemm_analysis/run_tracelens_analysis.sh experiments/sweep_YYYYMMDD_HHMMSS

# Extract top GEMM kernels
python scripts/gemm_analysis/analyze_gemm_reports.py \
  --base-path experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis \
  --threads 256 512 --channels 28 42 56 70 --top-k 5
```
