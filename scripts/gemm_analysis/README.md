# GEMM Sweep Profiling

Profile GEMM kernel performance across multiple NCCL configurations.

## Prerequisites

- Docker with ROCm support
- TraceLens installed

## Pipeline Steps

### 1. Build Docker Container

```bash
cd docker
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
bash scripts/gemm_analysis/run_tracelens_analysis.sh experiments/sweep_20251124_222204
```

### 4. Extract Top GEMM Kernels

```bash
python scripts/gemm_analysis/analyze_gemm_reports.py \
  --base-path experiments/sweep_20251124_222204/tracelens_analysis \
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

Note: Currently supports pairwise (2-sweep) comparison. For comparing multiple sweeps,
run multiple pairwise comparisons or aggregate data using the process_gpu_timeline.py
and process_comms.py scripts.

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

## Regression Testing

Automated regression test suite to ensure script changes don't produce incorrect reports.

### First Time Setup

**Important**: Activate virtual environment and generate test data before running tests (not tracked in git):

```bash
# Activate virtual environment
source ~/venvs/aorta/bin/activate

# Step 1: Generate synthetic test data (first time only)
python tests/gemm_analysis/generate_synthetic_data.py --output-dir tests/gemm_analysis/testdata

# Step 2: Generate baseline expected outputs (first time only)
# Run this on a known-good branch (e.g., origin/main)
# NOTE: This will automatically run TraceLens analysis as part of the pipeline
# If TraceLens is not installed, the test will skip with instructions
pytest tests/gemm_analysis/test_gemm_regression.py --generate-baseline
```

**Note on TraceLens**: The end-to-end test automatically runs TraceLens analysis as part of the pipeline. If TraceLens is not installed, you have two options:
1. Install TraceLens
2. Manually run TraceLens once and reuse the outputs:
   ```bash
   bash scripts/gemm_analysis/run_tracelens_analysis.sh tests/gemm_analysis/testdata/test_sweep
   ```

### Running Regression Tests

```bash
# Ensure virtual environment is activated
source ~/venvs/aorta/bin/activate

# Run full regression test suite
pytest tests/gemm_analysis/test_gemm_regression.py -v

# Run specific test classes
pytest tests/gemm_analysis/test_gemm_regression.py::TestFullPipeline -v
pytest tests/gemm_analysis/test_gemm_regression.py::TestIndividualSteps -v
pytest tests/gemm_analysis/test_gemm_regression.py::TestCustomConfigurations -v
```

### Test Architecture

The test suite is organized into modular components:

#### PipelineRunner Class
Central class that encapsulates pipeline execution logic with methods for each step:
- `run_tracelens()` - Run TraceLens analysis
- `run_analyze_gemm()` - Analyze GEMM reports
- `run_plot_variance()` - Generate variance plots
- `run_enhancement_scripts()` - Run additional analysis scripts
- `run_full_pipeline()` - Execute complete pipeline

#### Test Classes

1. **TestFullPipeline** - Full end-to-end regression test
2. **TestIndividualSteps** - Test individual pipeline components in isolation
3. **TestCustomConfigurations** - Test with different thread/channel configurations
4. **TestErrorHandling** - Test error conditions and edge cases

### What Gets Tested

The regression suite validates:
- TraceLens analysis execution (skips if not installed)
- GEMM report analysis across configurations
- Variance plot generation
- Enhancement script outputs
- GPU timeline processing
- Numeric output consistency (within tolerance)
- Script integration and data flow

### Test Data

- Uses synthetic PyTorch profiler traces (not real traces)
- Fixed configurations: 2 threads × 2 channels × 7 ranks
- 2 batches (ProfilerSteps) per trace
- Based on actual MI350X trace structure
- Located in `tests/gemm_analysis/testdata/` (not tracked in git)

### Baseline Comparison

- Baseline outputs stored in `tests/gemm_analysis/expected_outputs/`
- Generated from a known-good branch (typically origin/main)
- Current branch outputs compared against baseline
- Numeric values checked within small tolerance
- Cosmetic differences (whitespace, ordering) ignored

## Troubleshooting

### TraceLens Not Found
If TraceLens is not installed, the test suite will skip TraceLens-dependent tests. To install TraceLens, contact your AMD representative or use pre-generated TraceLens outputs.

### Import Errors
Ensure all required packages are installed in your virtual environment:
```bash
pip install pandas openpyxl matplotlib
```

### Test Data Not Found
Generate synthetic test data first:
```bash
python tests/gemm_analysis/generate_synthetic_data.py --output-dir tests/gemm_analysis/testdata
```

### Tests Failing After Generating Synthetic Data
If tests fail with errors related to missing TraceLens outputs, you need to run TraceLens analysis on the synthetic data:
```bash
bash scripts/gemm_analysis/run_tracelens_analysis.sh tests/gemm_analysis/testdata/test_sweep
```
This creates the TraceLens Excel reports that the analysis scripts need to process.

### Baseline Not Found
Generate baseline outputs on a known-good branch:
```bash
git checkout origin/main
pytest tests/gemm_analysis/test_gemm_regression.py --generate-baseline
git checkout your-feature-branch
```

## References

- Test suite: `tests/gemm_analysis/`
- Test plan: `scripts/gemm_analysis/test_plan.md`
- Implementation logs: `scripts/gemm_analysis/implementation_logs.md`
