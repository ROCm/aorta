# GEMM Analysis Pipeline

Complete workflow for analyzing GEMM kernel performance variance across different NCCL configurations using ROCm 7.0.

## Overview

This pipeline performs distributed training sweeps across different RCCL thread and NCCL channel configurations, then analyzes GEMM kernel performance variance to identify computation-communication overlap opportunities.

## Prerequisites

- Docker and docker-compose installed
- AMD GPU with ROCm support
- Access to AMD internal repositories for ROCm 7.0 builds

## Pipeline Steps

### 1. Build and Start Docker Container with ROCm 7.0

Navigate to the docker directory and build the custom ROCm 7.0 image:

```bash
cd /home/oyazdanb/aorta/docker

# Build the Docker image with ROCm 7.0 (specific builds: amdgpu-2247890, rocm-compute-rocm-rel-7.0-meta/7)
docker-compose -f docker-compose.rocm70.yaml build

# Start the container
docker-compose -f docker-compose.rocm70.yaml up -d

# Enter the container
docker exec -it training-overlap-bugs-rocm70 bash
```

### 2. Run Training Sweep

Inside the container, run the training sweep across different RCCL thread and NCCL channel configurations:

```bash
bash scripts/run_train_various_channels.sh \
  -c 28,42,56,70 \
  -t 256,512 \
  -f config/gemm_overlap/gemm_test_1.yaml
```

**Parameters:**
- `-c`: NCCL channel configurations to test (28, 42, 56, 70)
- `-t`: RCCL threads per block configurations (256, 512)
- `-f`: Configuration file to use for training

**Output:** Creates a timestamped directory under `experiments/sweep_YYYYMMDD_HHMMSS/` with:
- Subdirectories for each thread/channel combination
- Training logs and profiler traces
- Summary log and results

### 3. Generate TraceLens Analysis Reports

Analyze all profiler traces and generate comprehensive Excel reports:

```bash
bash scripts/run_tracelens_analysis.sh /home/oyazdanb/aorta/experiments/sweep_20251124_222204
```

**What it does:**
- Auto-discovers all thread and channel configurations
- Processes traces for all 8 ranks (model parallelism)
- Generates individual performance reports per rank/config
- Creates collective reports combining all ranks
- Produces comparison reports across configurations

**Output:** Creates `tracelens_analysis/` directory with:
- `256thread/individual_reports/` - Per-rank GEMM analysis
- `512thread/individual_reports/` - Per-rank GEMM analysis
- `256thread/collective_reports/` - Multi-rank collective analysis
- `512thread/collective_reports/` - Multi-rank collective analysis
- `comparisons/` - Cross-configuration comparisons

### 4. Extract Top GEMM Kernels with Variance

Analyze GEMM reports to find kernels with highest time variance (max - min duration):

```bash
python scripts/gemm_analysis/analyze_gemm_reports.py \
  --base-path /home/oyazdanb/aorta/experiments/sweep_20251124_222204/tracelens_analysis \
  --threads 256 512 \
  --channels 28 42 56 70 \
  --ranks 0 1 2 3 4 5 6 7 \
  --top-k 5
```

**Parameters:**
- `--base-path`: TraceLens analysis directory
- `--threads`: Thread configurations to analyze
- `--channels`: Channel configurations to analyze
- `--ranks`: GPU ranks to analyze (0-7 for 8 GPUs)
- `--top-k`: Number of top variance kernels to extract per file

**Output:** `top5_gemm_kernels_time_variance.csv` containing:
- Kernel name and configuration (threads, channels, rank)
- Min/max kernel execution times
- Time difference (variance)
- Additional kernel metadata

### 5. Generate Variance Distribution Plots

Create visualizations showing how GEMM variance distributes across configurations:

```bash
python scripts/gemm_analysis/plot_gemm_variance.py \
  --csv-path /home/oyazdanb/aorta/experiments/sweep_20251124_222204/tracelens_analysis/top5_gemm_kernels_time_variance.csv \
  --output-dir /home/oyazdanb/aorta/experiments/sweep_20251124_222204/tracelens_analysis/plots
```

**Output:** Creates 5 visualization plots in `plots/` directory:
1. `variance_by_threads_boxplot.png` - Box plot by thread count
2. `variance_by_channels_boxplot.png` - Box plot by channel count
3. `variance_by_ranks_boxplot.png` - Box plot by rank
4. `variance_violin_combined.png` - Combined violin plots
5. `variance_thread_channel_interaction.png` - Thread-channel interaction plot

### 6. Add Timestamps to Variance Data

Enhance the variance CSV with timestamp information for min/max kernel occurrences:

```bash
python scripts/gemm_analysis/enhance_gemm_variance_with_timestamps.py \
  --input-csv experiments/sweep_20251124_222204/tracelens_analysis/top5_gemm_kernels_time_variance.csv \
  --base-path experiments/sweep_20251124_222204
```

**What it does:**
- Finds exact timestamps when min and max duration kernels occurred
- Calculates time between min/max occurrences
- Verifies durations match expected values

**Output:** `top5_gemm_kernels_time_variance_with_timestamps.csv` with additional columns:
- `min_duration_timestamp_ms` - When shortest kernel occurred
- `max_duration_timestamp_ms` - When longest kernel occurred
- `time_between_min_max_ms` - Time gap between min/max
- Verification columns for data quality

### 7. Analyze GEMM-Collective Overlap

Identify which NCCL collective operations overlapped with high-variance GEMM kernels:

```bash
python scripts/gemm_analysis/gemm_report_with_collective_overlap.py \
  --input-csv experiments/sweep_20251124_222204/tracelens_analysis/top5_gemm_kernels_time_variance_with_timestamps.csv \
  --tracelens-path experiments/sweep_20251124_222204/tracelens_analysis
```

**What it does:**
- Loads collective operation data from TraceLens collective reports
- Checks if any NCCL collectives (allreduce, reducescatter, etc.) overlapped with max duration GEMM kernels
- Calculates overlap percentages and unique overlap durations
- Handles multiple concurrent collectives correctly (no double-counting)

**Output:** `top5_gemm_kernels_time_variance_with_collective_overlap.csv` with additional columns:
- `overlapping_collective_count` - Number of collectives that overlapped
- `overlapping_collective_names` - Names of overlapping collectives (semicolon-separated)
- `max_overlap_percentage` - Highest overlap percentage among all collectives
- `max_overlap_collective` - Name of collective with highest overlap
- `total_overlap_duration_ms` - Total unique time spent overlapping (merged intervals)

**Analysis insights:**
- High overlap (>50%) suggests GEMM was blocked by communication
- Multiple overlapping collectives indicate communication-heavy periods
- Zero overlap suggests pure computation without communication interference

### 8. Merge Multi-Rank Traces for Visualization

Combine traces from all 8 ranks into a single timeline for visualization in Perfetto.

#### Option A: Merge All Configurations (Recommended)

Use the batch script to automatically merge all thread/channel combinations:

```bash
bash scripts/merge_all_traces.sh experiments/sweep_20251124_222204
```

This will:
- Auto-discover all thread and channel configurations
- Merge traces for each configuration
- Save to `experiments/sweep_20251124_222204/merged_traces/`
- Show progress and summary statistics

#### Option B: Merge Individual Configuration

For merging a specific configuration manually:

```bash
# Create output directory
mkdir -p experiments/sweep_20251124_222204/merged_traces

# Merge specific configuration (e.g., 256 threads, 28 channels)
python scripts/merge_gpu_trace_ranks.py \
  experiments/sweep_20251124_222204/256thread/nccl_28channels/torch_profiler \
  -o experiments/sweep_20251124_222204/merged_traces/256thread_28ch_merged.json \
  -n 8
```

**Parameters:**
- `trace_dir`: Directory containing rank0/, rank1/, ..., rank7/ subdirectories
- `-o, --output`: Output filename for merged trace
- `-n, --num-ranks`: Number of ranks to process (default: 8)
- `--trace-name`: Trace filename in each rank directory (default: `trace_step19.json`)

**What it does:**
- Reads trace files from all ranks
- Reorganizes processes for clear visualization:
  - **PID N000**: Rank N GPU streams (kernels, memcpy)
  - **PID N500**: Rank N CPU/CUDA runtime
- Filters to keep GPU and important CPU events
- Prefixes event names with rank: `[R0]`, `[R1]`, etc.
- Creates single merged JSON file

**Viewing merged traces:**
1. Open https://ui.perfetto.dev
2. Click "Open trace file"
3. Select the merged `.json` file
4. View all 8 ranks' GPU activity on a single timeline

**Use cases:**
- Identify cross-rank synchronization issues
- Visualize communication patterns across all GPUs
- Debug rank imbalance and stragglers
- Verify GEMM-collective overlap visually

### 9. Create Embedded HTML Comparison Report

Generate a self-contained HTML report comparing two experiment sweeps side-by-side:

```bash
python scripts/gemm_analysis/create_embeded_html_report.py \
  --sweep1 experiments/sweep_20251121_155219 \
  --sweep2 experiments/sweep_20251124_222204 \
  --label1 "Base ROCm" \
  --label2 "ROCm 7.0" \
  --output sweep_comparison.html
```

**Parameters:**
- `--sweep1`: Path to first sweep directory (required)
- `--sweep2`: Path to second sweep directory (required)
- `--label1`: Custom label for first sweep (optional, defaults to directory name)
- `--label2`: Custom label for second sweep (optional, defaults to directory name)
- `--output`: Output HTML file path (optional, defaults to `sweep_comparison_report.html`)

**What it does:**
- Automatically finds all plot images in both sweep directories
- Converts images to base64 and embeds them directly in the HTML
- Creates a self-contained HTML file (typically 2-3 MB) that can be shared easily
- Displays plots side-by-side for easy comparison
- Validates that sweep directories exist before processing

**Output:** Single HTML file with embedded images showing:
- Variance by thread count comparison
- Variance by channel count comparison
- Variance by rank comparison
- Violin plot distributions
- Thread-channel interaction plots

**Benefits:**
- No external dependencies - all images embedded
- Easy to share via email or file transfer
- Opens in any web browser
- Professional styling with responsive layout
- Perfect for presentations and reports

## Quick Reference

### Default Configurations

- **Threads**: 256, 512
- **Channels**: 28, 42, 56, 70
- **Ranks**: 0-7 (8 GPUs)
- **Top K**: 5 kernels per configuration

### Key Output Files

1. **`top5_gemm_kernels_time_variance.csv`** - GEMM kernels with highest variance
2. **`top5_gemm_kernels_time_variance_with_timestamps.csv`** - Enhanced with temporal data
3. **`top5_gemm_kernels_time_variance_with_collective_overlap.csv`** - Final analysis with overlap info
4. **`plots/*.png`** - Visualization plots
5. **`merged_traces/*.json`** - Multi-rank traces for Perfetto
6. **`sweep_comparison.html`** - Self-contained HTML report comparing two sweeps

### Output Structure

```
experiments/sweep_YYYYMMDD_HHMMSS/
├── 256thread/
│   └── nccl_XXchannels/
│       ├── torch_profiler/rank*/
│       └── run_output.log
├── 512thread/
│   └── nccl_XXchannels/
│       ├── torch_profiler/rank*/
│       └── run_output.log
├── tracelens_analysis/
│   ├── 256thread/
│   │   ├── individual_reports/perf_XXch_rankY.xlsx
│   │   └── collective_reports/collective_XXch.xlsx
│   ├── 512thread/
│   │   ├── individual_reports/perf_XXch_rankY.xlsx
│   │   └── collective_reports/collective_XXch.xlsx
│   ├── comparisons/compare_XXch_rankY_across_threads.xlsx
│   ├── plots/
│   │   ├── variance_by_threads_boxplot.png
│   │   ├── variance_by_channels_boxplot.png
│   │   ├── variance_by_ranks_boxplot.png
│   │   ├── variance_violin_combined.png
│   │   └── variance_thread_channel_interaction.png
│   ├── top5_gemm_kernels_time_variance.csv
│   ├── top5_gemm_kernels_time_variance_with_timestamps.csv
│   └── top5_gemm_kernels_time_variance_with_collective_overlap.csv
├── merged_traces/
│   ├── 256thread_28ch_merged.json
│   ├── 256thread_42ch_merged.json
│   ├── 512thread_28ch_merged.json
│   └── ... (one per configuration)
├── nccl_thread_sweep_YYYYMMDD_HHMMSS.log
└── nccl_thread_sweep_summary_YYYYMMDD_HHMMSS.txt
```


## Quick Start Commands

For a complete analysis of a sweep, run these commands in order:

```bash
# 1. Run TraceLens analysis
bash scripts/run_tracelens_analysis.sh experiments/sweep_20251124_222204

# 2. Extract GEMM variance
python scripts/gemm_analysis/analyze_gemm_reports.py \
  --base-path experiments/sweep_20251124_222204/tracelens_analysis

# 3. Create plots
python scripts/gemm_analysis/plot_gemm_variance.py \
  --csv-path experiments/sweep_20251124_222204/tracelens_analysis/top5_gemm_kernels_time_variance.csv

# 4. Add timestamps
python scripts/gemm_analysis/enhance_gemm_variance_with_timestamps.py \
  --input-csv experiments/sweep_20251124_222204/tracelens_analysis/top5_gemm_kernels_time_variance.csv \
  --base-path experiments/sweep_20251124_222204

# 5. Analyze collective overlap
python scripts/gemm_analysis/gemm_report_with_collective_overlap.py \
  --input-csv experiments/sweep_20251124_222204/tracelens_analysis/top5_gemm_kernels_time_variance_with_timestamps.csv \
  --tracelens-path experiments/sweep_20251124_222204/tracelens_analysis

# 6. Merge all traces for visualization
bash scripts/merge_all_traces.sh experiments/sweep_20251124_222204

# 7. (Optional) Create HTML comparison report between two sweeps
python scripts/gemm_analysis/create_embeded_html_report.py \
  --sweep1 experiments/sweep_20251121_155219 \
  --sweep2 experiments/sweep_20251124_222204 \
  --label1 "Base ROCm" \
  --label2 "ROCm 7.0" \
  --output sweep_comparison.html
```
