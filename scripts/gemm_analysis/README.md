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
  --channels 28,42,56,70 \
  --threads 256,512 \
  --config config/gemm_overlap/gemm_test_1.yaml
```

#### rocprof Tracing Options

Add rocprofv3 kernel tracing to capture detailed GEMM performance:

Simple YAML-based tracing (recommended)
---------------------------------------

1) Create a minimal rocprof input file, e.g. `scripts/gemm_analysis/rocprof_input.yaml`:
```yaml
jobs:
  - kernel_include_regex: "(gemm|Cijk_.*)"  # pattern for kernels to trace
    kernel_trace: true                      # enable kernel tracing
    stats: true                             # set false if you don’t want stats
    output_format: [json, csv]              # add perfetto/otf2 if desired
    sys_trace: false
    advanced_thread_trace: false            # leave false unless ATT decoder is installed
```
2) Run the sweep with the YAML:
```bash
bash scripts/gemm_analysis/run_train_various_channels.sh \
  --rocprof \
  --rocprof-input scripts/gemm_analysis/rocprof_input.yaml \
  --channels 28,42,56 --threads 256,512 \
  --config config/gemm_overlap/gemm_test_1.yaml
```
Notes:
- Kernel filtering/stats come from the YAML. The current rocprofv3 build ignores CLI kernel filters, so use the YAML to include/exclude kernels.
- Remove `advanced_thread_trace` or keep it `false` unless the ATT decoder debs are installed.

**Key Options:**
- `--rocprof` : Enable rocprofv3 tracing
- `--stats` : Include CU utilization/occupancy metrics (remove if not needed)
- `--channels VALUES` : Comma-separated NCCL channel values
- `--threads VALUES` : Comma-separated thread values

**Output:** Traces saved to `rocprof_traces/` in each run directory.

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
│       ├── rocprof_traces/           # if --rocprof flag used
│       │   ├── *.json                # Chrome trace files
│       │   ├── *.csv                 # Kernel stats
│       │   └── stats.txt             # CU metrics (if --stats used)
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
  --channels 28,42,56,70 \
  --threads 256,512 \
  --config config/gemm_overlap/gemm_test_1.yaml

# Run with rocprof tracing (all kernels with stats)
bash scripts/gemm_analysis/run_train_various_channels.sh \
  --rocprof --stats \
  --channels 28,42,56,70 \
  --threads 256,512

# Run with rocprof using a YAML input (recommended for filtering)
bash scripts/gemm_analysis/run_train_various_channels.sh \
  --rocprof --stats \
  --rocprof-input scripts/gemm_analysis/rocprof_input.yaml \
  --channels 28,42,56,70 \
  --threads 256,512

# Generate TraceLens reports
bash scripts/gemm_analysis/run_tracelens_analysis.sh experiments/sweep_YYYYMMDD_HHMMSS

# Extract top GEMM kernels
python scripts/gemm_analysis/analyze_gemm_reports.py \
  --base-path experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis \
  --threads 256 512 --channels 28 42 56 70 --top-k 5
```
