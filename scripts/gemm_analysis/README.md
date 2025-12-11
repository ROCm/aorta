# GEMM Sweep Profiling and Analysis

Profile GEMM kernel performance across multiple NCCL configurations.

## Prerequisites

- Docker with ROCm support
- TraceLens installed (optional for some scripts)
- Python packages: pandas, openpyxl, matplotlib, seaborn

## Workflow

### 1. Build Docker Container

```bash
cd docker
docker compose -f docker-compose.rocm70_9-1.yaml build
docker compose -f docker-compose.rocm70_9-1.yaml up -d
docker exec -it training-overlap-bugs-rocm70_9-1 bash
```

### 2. Run Training Sweep

Basic sweep:
```bash
bash scripts/gemm_analysis/run_train_various_channels.sh \
  --channels 28,42,56,70 \
  --threads 256,512 \
  --config config/gemm_overlap/gemm_test_1.yaml
```

With rocprof tracing:
```bash
bash scripts/gemm_analysis/run_train_various_channels.sh \
  --rocprof \
  --rocprof-input scripts/gemm_analysis/rocprof_cu_only.yaml \
  --channels 28,42,56,70 \
  --threads 256,512 \
  --config config/gemm_overlap/gemm_test_1.yaml
```

For rocprof configuration details, see [rocprof_guide.md](rocprof_guide.md).

### 3. Generate TraceLens Reports

```bash
bash scripts/gemm_analysis/run_tracelens_analysis.sh experiments/sweep_YYYYMMDD_HHMMSS
```

### 4. Extract Top GEMM Kernels

```bash
python scripts/gemm_analysis/analyze_gemm_reports.py \
  --base-path experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis \
  --threads 256 512 \
  --channels 28 42 56 70 \
  --ranks 0 1 2 3 4 5 6 7 \
  --top-k 5
```

Output: `top5_gemm_kernels_time_variance.csv`

### 5. Generate Variance Plots

```bash
python scripts/gemm_analysis/plot_gemm_variance.py \
  --csv-path experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/top5_gemm_kernels_time_variance.csv \
  --output-dir experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/plots
```

Generates box plots, violin plots, and interaction plots.

### 6. Add Timestamp Information

```bash
python scripts/gemm_analysis/enhance_gemm_variance_with_timestamps.py \
  --input-csv experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/top5_gemm_kernels_time_variance.csv \
  --base-path experiments/sweep_YYYYMMDD_HHMMSS
```

Output: `top5_gemm_kernels_time_variance_with_timestamps.csv`

### 7. Analyze Collective Overlap

```bash
python scripts/gemm_analysis/gemm_report_with_collective_overlap.py \
  --input-csv experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/top5_gemm_kernels_time_variance_with_timestamps.csv \
  --tracelens-path experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis
```

Output: `top5_gemm_kernels_time_variance_with_collective_overlap.csv`

### 8. Process GPU Timeline

```bash
python scripts/gemm_analysis/process_gpu_timeline.py \
  --sweep-dir experiments/sweep_YYYYMMDD_HHMMSS
```

Output: `gpu_timeline_all_configs_mean.xlsx`

### 9. Process NCCL Communication Data

```bash
python scripts/gemm_analysis/process_comms.py \
  --sweep-dir experiments/sweep_YYYYMMDD_HHMMSS
```

Output: `nccl_master_all_configs.xlsx`

### 10. Create Comparison HTML Report

```bash
python scripts/gemm_analysis/create_embeded_html_report.py \
  --sweep1 experiments/sweep_20251121_155219 \
  --sweep2 experiments/sweep_20251124_222204 \
  --label1 "Base ROCm" \
  --label2 "ROCm 7.0" \
  --output sweep_comparison.html
```

Creates self-contained HTML with embedded images.

## Output Structure

```
experiments/sweep_YYYYMMDD_HHMMSS/
├── 256thread/
│   └── nccl_XXchannels/
│       ├── torch_profiler/rank*/
│       ├── rocprof_traces/           # if --rocprof used
│       └── run_output.log
├── 512thread/
│   └── nccl_XXchannels/
└── tracelens_analysis/
    ├── 256thread/
    │   ├── individual_reports/perf_*ch_rank*.xlsx
    │   └── collective_reports/collective_*ch.xlsx
    ├── 512thread/
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

## Script Reference

### Core Pipeline Scripts

- `run_train_various_channels.sh` - Execute training sweep across configurations
- `run_tracelens_analysis.sh` - Generate TraceLens Excel reports
- `analyze_gemm_reports.py` - Extract top GEMM kernels by variance
- `plot_gemm_variance.py` - Generate visualization plots

### Enhancement Scripts

- `enhance_gemm_variance_with_timestamps.py` - Add kernel execution timestamps
- `gemm_report_with_collective_overlap.py` - Identify NCCL overlap with GEMM
- `process_gpu_timeline.py` - Aggregate GPU timeline across configurations
- `process_comms.py` - Extract NCCL communication statistics

### Reporting Scripts

- `create_embeded_html_report.py` - Generate HTML comparison reports (pairwise)

## Regression Testing

The pipeline includes automated regression tests to ensure script changes don't break functionality.

Setup and run tests:
```bash
source ~/venvs/aorta/bin/activate
pytest tests/gemm_analysis/test_gemm_regression.py -v
```

For details on test architecture and adding new tests, see [tests/gemm_analysis/README.md](../../tests/gemm_analysis/README.md).

## Troubleshooting

### TraceLens Not Installed

If TraceLens is not available, analysis scripts will skip TraceLens-dependent processing.

### Missing Dependencies

```bash
pip install pandas openpyxl matplotlib seaborn
```

### Import Errors

Ensure virtual environment is activated:
```bash
source ~/venvs/aorta/bin/activate
```

## Additional Documentation

- [rocprof_guide.md](rocprof_guide.md) - Detailed rocprof configuration and performance counters
- [tests/gemm_analysis/README.md](../../tests/gemm_analysis/README.md) - Test architecture and development
