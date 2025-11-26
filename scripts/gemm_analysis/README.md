# GEMM Analysis Pipeline

Analyze GEMM kernel performance across multiple NCCL configurations.

> Single config? Use [../tracelens_single_config/README.md](../tracelens_single_config/README.md)

## Prerequisites

- Docker with ROCm support
- TraceLens installed

## Pipeline Steps

### 1. Build Docker Container

```bash
cd /home/oyazdanb/aorta/docker
docker-compose -f docker-compose.rocm70.yaml build
docker-compose -f docker-compose.rocm70.yaml up -d
docker exec -it training-overlap-bugs-rocm70 bash
```

### 2. Run Training Sweep

```bash
bash scripts/run_train_various_channels.sh \
  -c 28,42,56,70 \
  -t 256,512 \
  -f config/gemm_overlap/gemm_test_1.yaml
```

### 3. Generate TraceLens Reports

```bash
bash scripts/run_tracelens_analysis.sh /home/oyazdanb/aorta/experiments/sweep_20251124_222204
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

### 5. Generate Plots

```bash
python scripts/gemm_analysis/plot_gemm_variance.py \
  --csv-path /home/oyazdanb/aorta/experiments/sweep_20251124_222204/tracelens_analysis/top5_gemm_kernels_time_variance.csv \
  --output-dir /home/oyazdanb/aorta/experiments/sweep_20251124_222204/tracelens_analysis/plots
```

### 6. Add Timestamps

```bash
python scripts/gemm_analysis/enhance_gemm_variance_with_timestamps.py \
  --input-csv experiments/sweep_20251124_222204/tracelens_analysis/top5_gemm_kernels_time_variance.csv \
  --base-path experiments/sweep_20251124_222204
```

### 7. Analyze Overlap

```bash
python scripts/gemm_analysis/gemm_report_with_collective_overlap.py \
  --input-csv experiments/sweep_20251124_222204/tracelens_analysis/top5_gemm_kernels_time_variance_with_timestamps.csv \
  --tracelens-path experiments/sweep_20251124_222204/tracelens_analysis
```

### 8. Merge Traces

```bash
bash scripts/merge_all_traces.sh experiments/sweep_20251124_222204
```

### 9. Create HTML Report

```bash
python scripts/gemm_analysis/create_embeded_html_report.py \
  --sweep1 experiments/sweep_20251121_155219 \
  --sweep2 experiments/sweep_20251124_222204 \
  --label1 "Base ROCm" \
  --label2 "ROCm 7.0" \
  --output sweep_comparison.html
```

## Output Structure

```
experiments/sweep_YYYYMMDD_HHMMSS/
├── 256thread/
│   └── nccl_XXchannels/
│       ├── torch_profiler/rank*/
│       └── run_output.log
├── 512thread/
│   └── nccl_XXchannels/
├── tracelens_analysis/
│   ├── 256thread/
│   │   ├── individual_reports/
│   │   └── collective_reports/
│   ├── 512thread/
│   ├── comparisons/
│   ├── plots/
│   └── top5_gemm_kernels_time_variance.csv
└── merged_traces/
```

## Quick Commands

```bash
# Complete analysis
bash scripts/run_tracelens_analysis.sh experiments/sweep_20251124_222204

# Extract GEMM variance
python scripts/gemm_analysis/analyze_gemm_reports.py \
  --base-path experiments/sweep_20251124_222204/tracelens_analysis

# Create plots
python scripts/gemm_analysis/plot_gemm_variance.py \
  --csv-path experiments/sweep_20251124_222204/tracelens_analysis/top5_gemm_kernels_time_variance.csv

# Merge traces
bash scripts/merge_all_traces.sh experiments/sweep_20251124_222204
```