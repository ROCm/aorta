# GEMM Sweep Profiling

Profile GEMM kernel performance across multiple NCCL configurations.

## Prerequisites

- Docker with ROCm support
- TraceLens installed

## Pipeline Steps

### 1. Build Docker Container

```bash
cd ~/aorta/docker
docker compose -f docker-compose.rocm70_9-1.yaml build
docker compose -f docker-compose.rocm70_9-1.yaml up -d
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
bash scripts/gemm_analysis/run_tracelens_analysis.sh ~/aorta/experiments/sweep_20251124_222204
```

### 4. Extract Top GEMM Kernels

```bash
python scripts/gemm_analysis/analyze_gemm_reports.py \
  --base-path ~/aorta/experiments/sweep_20251124_222204/tracelens_analysis \
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
# GEMM Visualization and Reporting

Visualization, overlap analysis, and reporting tools for GEMM kernel performance data.

## Prerequisites

- Python packages: pandas, openpyxl, matplotlib, seaborn
- Completed GEMM sweep profiling with generated `top5_gemm_kernels_time_variance.csv`

## Pipeline Steps

### 1. Generate Variance Plots

Create comprehensive visualization of GEMM kernel variance:

```bash
python scripts/gemm_analysis/plot_gemm_variance.py \
  --csv-path experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/top5_gemm_kernels_time_variance.csv \
  --output-dir experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/plots
```

Generates:
- Box plots by thread count, channel count, and rank
- Violin plots showing distribution
- Thread-channel interaction plots

### 2. Add Timestamp Information

Enhance variance data with kernel execution timestamps:

```bash
python scripts/gemm_analysis/enhance_gemm_variance_with_timestamps.py \
  --input-csv experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/top5_gemm_kernels_time_variance.csv \
  --base-path experiments/sweep_YYYYMMDD_HHMMSS
```

Output: `top5_gemm_kernels_time_variance_with_timestamps.csv`

### 3. Analyze Collective Overlap

Identify NCCL collective operations overlapping with GEMM kernels:

```bash
python scripts/gemm_analysis/gemm_report_with_collective_overlap.py \
  --input-csv experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/top5_gemm_kernels_time_variance_with_timestamps.csv \
  --tracelens-path experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis
```

Output: `top5_gemm_kernels_time_variance_with_collective_overlap.csv`

### 4. Create Comparison HTML Report

Generate side-by-side comparison of two experiment sweeps:

```bash
python scripts/gemm_analysis/create_embeded_html_report.py \
  --sweep1 experiments/sweep_20251121_155219 \
  --sweep2 experiments/sweep_20251124_222204 \
  --label1 "Base ROCm" \
  --label2 "ROCm 7.0" \
  --output sweep_comparison.html
```

Creates self-contained HTML with embedded images.

## Additional Analysis Tools

### Process GPU Timeline

Aggregate GPU timeline data across all ranks and configurations:

```bash
python scripts/gemm_analysis/process_gpu_timeline.py \
  --sweep-dir experiments/sweep_YYYYMMDD_HHMMSS
```

Output: `gpu_timeline_all_configs_mean.xlsx` with multiple sheets:
- All_Data - Complete dataset
- Pivot_Time_ms - Matrix view of time
- Pivot_Percent - Matrix view of percentages
- Summary_By_Config - Key metrics per configuration

### Process NCCL Communication Data

Extract and aggregate NCCL collective operation data:

```bash
python scripts/gemm_analysis/process_comms.py \
  --sweep-dir experiments/sweep_YYYYMMDD_HHMMSS
```

Output: `nccl_master_all_configs.xlsx` and `.csv` with:
- Communication latency statistics
- Bandwidth metrics
- Time skew analysis

## Output Structure

```
experiments/sweep_YYYYMMDD_HHMMSS/
└── tracelens_analysis/
    ├── top5_gemm_kernels_time_variance.csv
    ├── top5_gemm_kernels_time_variance_with_timestamps.csv
    ├── top5_gemm_kernels_time_variance_with_collective_overlap.csv
    ├── gpu_timeline_all_configs_mean.xlsx
    ├── nccl_master_all_configs.xlsx
    └── plots/
        ├── variance_by_threads_boxplot.png
        ├── variance_by_channels_boxplot.png
        ├── variance_by_ranks_boxplot.png
        ├── variance_violin_combined.png
        └── variance_thread_channel_interaction.png
```

## Quick Reference

```bash
# Generate all visualizations
python scripts/gemm_analysis/plot_gemm_variance.py \
  --csv-path experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/top5_gemm_kernels_time_variance.csv

# Add timestamps and analyze overlap
python scripts/gemm_analysis/enhance_gemm_variance_with_timestamps.py \
  --input-csv experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/top5_gemm_kernels_time_variance.csv \
  --base-path experiments/sweep_YYYYMMDD_HHMMSS

python scripts/gemm_analysis/gemm_report_with_collective_overlap.py \
  --input-csv experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/top5_gemm_kernels_time_variance_with_timestamps.csv \
  --tracelens-path experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis

# Process Excel reports
python scripts/gemm_analysis/process_gpu_timeline.py --sweep-dir experiments/sweep_YYYYMMDD_HHMMSS
python scripts/gemm_analysis/process_comms.py --sweep-dir experiments/sweep_YYYYMMDD_HHMMSS

# Create comparison report
python scripts/gemm_analysis/create_embeded_html_report.py \
  --sweep1 experiments/sweep1 --sweep2 experiments/sweep2 \
  --label1 "Baseline" --label2 "Optimized" --output comparison.html
```
