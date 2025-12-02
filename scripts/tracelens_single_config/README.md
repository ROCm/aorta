# RCCL Warp Speed Performance Testing

Test RCCL warp_speed_v1 branch from https://github.com/mustafabar/rccl.git

## Prerequisites

```bash
pip install pandas openpyxl matplotlib seaborn numpy
```

## Run Tests

### Step 1: Start Container and Build RCCL

```bash
cd docker
docker compose -f docker-compose.rocm70_9-1.yaml build
docker compose -f docker-compose.rocm70_9-1.yaml up -d
docker compose -f docker-compose.rocm70_9-1.yaml exec torchenv-rocm70 bash

# Inside container - build warp_speed_v1 (first time only)
if [ ! -f /opt/rccl/build/release/librccl.so ]; then
    cd /opt
    git clone --recursive https://github.com/mustafabar/rccl.git
    cd rccl
    git checkout warp_speed_v1
    ./install.sh -l --amdgpu_targets=gfx942
fi

cd /workspace/aorta
```

### Step 2: Run RCCL Tests

```bash
# Default 3 configurations
./scripts/tracelens_single_config/run_rccl_warp_speed_comparison.sh

# Custom configurations (CU_count,threads pairs)
./scripts/tracelens_single_config/run_rccl_warp_speed_comparison.sh -p "56,256 37,384 32,512"
```

Output structure:
```
experiments/
  rccl_warp_speed_YYYYMMDD_HHMMSS/
    56cu_256threads/
      torch_profiler/       # Raw profiler traces
      run_output.log       # Training output log
    37cu_384threads/
    32cu_512threads/
    rccl_warp_speed_summary_YYYYMMDD_HHMMSS.txt
```

### Step 3: Generate Reports (Outside Container)

```bash
# Exit container
exit

# Run complete analysis
python scripts/tracelens_single_config/run_full_analysis.py \
  --baseline experiments/rccl_warp_speed_YYYYMMDD/56cu_256threads \
  --test experiments/rccl_warp_speed_YYYYMMDD/37cu_384threads \
  --output comparison_results \
  --all

# Or skip TraceLens if already done
python scripts/tracelens_single_config/run_full_analysis.py \
  --baseline experiments/rccl_warp_speed_YYYYMMDD/56cu_256threads \
  --test experiments/rccl_warp_speed_YYYYMMDD/37cu_384threads \
  --output comparison_results \
  --all --skip-tracelens
```

## Generated Excel Reports

### Individual TraceLens Reports (per configuration)
Each configuration generates:
- `tracelens_analysis/individual_reports/perf_rank*.xlsx` - Per-rank performance breakdown
- `tracelens_analysis/collective_reports/collective_all_ranks.xlsx` - Collective operations summary
- `tracelens_analysis/gpu_timeline_summary_mean.xlsx` - GPU timeline averages

### Final Analysis Report (`final_analysis_report.xlsx`)

Contains multiple sheets:

**Summary Sheets:**
- `Summary_Dashboard` - High-level comparison metrics with percentage changes
- `Summary_Comparison` - Side-by-side summary comparison
- `GPU_ByRank_Comparison` - Detailed per-rank performance comparison
- `Comparison_By_Rank` - Rank-wise metric comparison with differences

**GPU Timeline Sheets:**
- `All_Ranks_Combined` - Combined GPU timeline data from all ranks
- `Summary` - Aggregated GPU timeline summary
- `Rank_*` - Individual rank GPU timelines

**Collective/NCCL Sheets:**
- `nccl_summary_implicit_sync` - NCCL operations with implicit synchronization
- `nccl_summary_long` - Long-running NCCL operations
- `nccl_summary_implicit_sync_comparison` - Comparison of implicit sync operations
- `nccl_summary_long_comparison` - Comparison of long operations

**Raw Data Sheets (hidden by default):**
- `gpu_timeline_combined` - Raw combined GPU timeline data
- `gpu_timeline_comparison` - Raw GPU timeline comparison data
- `collective_combined` - Raw collective operations data
- `collective_comparison` - Raw collective comparison data

### Comparison Reports

- `gpu_timeline_combined.xlsx` - Baseline and test GPU metrics combined
- `gpu_timeline_comparison.xlsx` - GPU metrics with comparison analysis
- `collective_combined.xlsx` - Baseline and test collective operations combined
- `collective_comparison.xlsx` - Collective operations with comparison analysis

## Generated Visualizations

### HTML Report
- `performance_analysis_report.html` - Complete report with all embedded plots

### Individual Plot Files (12 Total)
1. `plot1_percentage_change_overview.png` - Horizontal bar chart showing performance changes
2. `plot2_absolute_time_comparison.png` - Bar chart comparing absolute times
3. `plot3_performance_heatmap.png` - Heatmap of performance by rank
4. `plot4_total_execution_time.png` - Line plot of total execution time per rank
5. `plot5_computation_time.png` - Line plot of computation time across ranks
6. `plot6_communication_time.png` - Line plot of communication time across ranks
7. `plot7_idle_time.png` - Line plot of idle time across ranks
8. `plot8_percentage_difference_all_metrics.png` - Bar plot showing percentage differences for all metrics
9. `plot9_nccl_latency.png` - Line plot of latency vs message size
10. `plot10_algorithm_bandwidth.png` - Line plot of algorithm bandwidth vs message size
11. `plot11_bus_bandwidth.png` - Line plot of bus bandwidth vs message size
12. `plot12_nccl_summary.png` - Combined percentage summary and total latency

## Key Metrics Analyzed

**GPU Metrics:**
- `computation_time` - Time spent in computation
- `total_comm_time` - Total communication time
- `exposed_comm_time` - Non-overlapped communication time
- `idle_time` - GPU idle time
- `total_memcpy_time` - Memory copy time
- `exposed_memcpy_time` - Non-overlapped memory copy time
- `busy_time` - Total GPU busy time
- `total_time` - Total execution time

**NCCL Metrics:**
- `comm_latency_mean` - Average communication latency
- `algo bw (GB/s)_mean` - Algorithm bandwidth
- `bus bw (GB/s)_mean` - Bus bandwidth
- `Total comm latency (ms)` - Total communication latency
- `count` - Number of operations

## Convert to PDF

1. Open `performance_analysis_report.html` in browser
2. Print to PDF (Ctrl+P or Cmd+P)
3. Choose landscape orientation for better plot visibility
