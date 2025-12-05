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

Use the `rocprof_cu_only.yaml` configuration file for CU utilization metrics:

```yaml
jobs:
  - kernel_include_regex: "(gemm|Cijk_.*)"  # pattern for kernels to trace
    kernel_trace: true                      # enable kernel tracing
    stats: true                             # timing statistics only (not CU utilization)
    output_format: [json, csv]              # add perfetto for Chrome tracing
    sys_trace: false
    advanced_thread_trace: false            # leave false unless ATT decoder is installed
```

Run the sweep with the CU-only YAML:
```bash
bash scripts/gemm_analysis/run_train_various_channels.sh \
  --rocprof \
  --rocprof-input scripts/gemm_analysis/rocprof_cu_only.yaml \
  --channels 28,42,56 --threads 256,512 \
  --config config/gemm_overlap/gemm_test_1.yaml
```
Notes:
- Kernel filtering/stats come from the YAML. The current rocprofv3 build ignores CLI kernel filters, so use the YAML to include/exclude kernels.
- Remove `advanced_thread_trace` or keep it `false` unless the ATT decoder debs are installed.
- **Important**: `stats: true` only collects timing statistics, NOT CU utilization metrics.
- **Output Files**: rocprof generates 5 files per rank/process:
  - `PID_agent_info.csv`: Hardware information about CPUs and GPUs
  - `PID_counter_collection.csv`: **Main file with CU utilization metrics** (focus on this)
  - `PID_kernel_trace.csv`: Kernel execution timeline data
  - `PID_results.json`: Chrome trace format for visualization
  - `PID_results.csv`: Summary statistics

**Analyzing Unique GEMM Kernels (counter_collection.csv columns):**
- `Grid_Size`: Total number of workgroups in the kernel launch
- `Kernel_Name`: Name of the GEMM kernel (e.g., Cijk_Alik_Bljk_SB_MT128x128x32_MI32x32x1x2)
- `Workgroup_Size`: Number of work-items per workgroup
- `LDS_Block_Size`: Local Data Share memory allocation per workgroup
- `Scratch_Size`: Private memory allocation per work-item
- `VGPR_Count`: Vector General Purpose Registers used
- `Accum_VGPR_Count`: Accumulator VGPRs (for matrix operations)
- `SGPR_Count`: Scalar General Purpose Registers used
- `Counter_Name`: Performance counter being measured (e.g., SQ_BUSY_CU_CYCLES)
- `Counter_Value`: Value of the performance counter
- `Start_Timestamp` / `End_Timestamp`: Kernel execution timing

**Key Options:**
- `--rocprof` : Enable rocprofv3 tracing
- `--stats` : Include timing statistics (not CU utilization)
- `--channels VALUES` : Comma-separated NCCL channel values
- `--threads VALUES` : Comma-separated thread values

**Output:** Traces saved to `rocprof_traces/` in each run directory.

**Key Performance Counters (found in counter_collection.csv files):**
- `SQ_BUSY_CU_CYCLES`: Percentage of time CUs are active (CU utilization)
- `SQ_WAVES`: Number of active wavefronts (occupancy indicator)
- `SQ_INSTS_MFMA`: Matrix FMA instructions (critical for GEMM performance)
- `SQ_INSTS_VALU`: Vector ALU instructions (general compute)

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
│       │   ├── PID_agent_info.csv    # Hardware info for each rank
│       │   ├── PID_counter_collection.csv  # CU utilization metrics (main focus)
│       │   ├── PID_kernel_trace.csv  # Kernel execution timeline
│       │   ├── PID_results.json      # Chrome trace format
│       │   └── PID_results.csv       # Summary statistics
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

# Run with rocprof using CU-only YAML (recommended)
bash scripts/gemm_analysis/run_train_various_channels.sh \
  --rocprof --stats \
  --rocprof-input scripts/gemm_analysis/rocprof_cu_only.yaml \
  --channels 28,42,56,70 \
  --threads 256,512

# Generate TraceLens reports
bash scripts/gemm_analysis/run_tracelens_analysis.sh experiments/sweep_YYYYMMDD_HHMMSS

# Extract top GEMM kernels
python scripts/gemm_analysis/analyze_gemm_reports.py \
  --base-path experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis \
  --threads 256 512 --channels 28 42 56 70 --top-k 5
```
