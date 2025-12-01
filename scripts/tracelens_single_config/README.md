# TraceLens Single Configuration

Compare PyTorch profiler traces across RCCL configurations.

## Prerequisites

### For Step 1 (Training/Trace Generation) - Docker Required

Training scripts require RCCL warp_speed_v1, so Step 1 must run inside the Docker container:

```bash
# Start the RCCL warp_speed_v1 Docker container
cd /home/oyazdanb/aorta/docker
docker compose -f docker-compose.rccl-warpspeed.yaml up -d

# Enter the container
docker compose -f docker-compose.rccl-warpspeed.yaml exec rccl-warpspeed bash

# IMPORTANT: Verify you're using RCCL warp_speed_v1 (3 quick checks)
echo "1. RCCL Path: $RCCL_ROOT"  # Should show: /opt/rccl-warpspeed
ls -la /opt/rccl-warpspeed/lib/librccl.so 2>/dev/null && echo "   [OK] RCCL library found"
echo "2. Branch:" && git -C /opt/rccl branch --show-current 2>/dev/null  # Should show: warp_speed_v1
echo "3. Library priority:" && echo $LD_LIBRARY_PATH | grep -o '^[^:]*'  # Should start with /opt/rccl-warpspeed/lib

# For detailed verification, run:
# ./scripts/tracelens_single_config/verify_rccl.sh

# Now move to workspace
cd /workspace/aorta
```

### For Steps 2-3 (Analysis) - Python venv

Analysis scripts only process trace files and don't need RCCL. Run in your Python virtual environment (e.g., `~/venvs/aorta`).

## Quick Start

### Step 1: Generate traces (MUST run inside Docker container)

The script automatically sets RCCL warp_speed environment variables for each configuration:

**Configuration 1: 56 CUs, 256 threads**
- `RCCL_WARP_SPEED_ENABLE=1`
- `RCCL_UNROLL_FACTOR=1`
- `RCCL_WARP_SPEED_CU_COUNT=56`
- `RCCL_THREADS_PER_BLOCK=256`

**Configuration 2: 37 CUs, 384 threads**
- `RCCL_WARP_SPEED_ENABLE=1`
- `RCCL_UNROLL_FACTOR=1`
- `RCCL_WARP_SPEED_CU_COUNT=37`
- `RCCL_THREADS_PER_BLOCK=384`

**Configuration 3: 32 CUs, 512 threads**
- `RCCL_WARP_SPEED_ENABLE=1`
- `RCCL_UNROLL_FACTOR=1`
- `RCCL_WARP_SPEED_CU_COUNT=32`
- `RCCL_THREADS_PER_BLOCK=512`

```bash
# Run with default 3 configurations above
./scripts/tracelens_single_config/run_rccl_warp_speed_comparison.sh

# Or specify custom CU,thread pairs
./scripts/tracelens_single_config/run_rccl_warp_speed_comparison.sh -p "56,256 37,384 32,512"
```

Output: `experiments/rccl_warp_speed_YYYYMMDD_HHMMSS/`
- `56cu_256threads/`
- `37cu_384threads/`
- `32cu_512threads/`

After traces are generated, exit the Docker container:
```bash
exit
# Optionally stop the container
docker compose -f docker-compose.rccl-warpspeed.yaml down
```

### Step 2: Generate individual and collective reports (in Python venv)
```bash
# Activate your Python virtual environment (e.g., ~/venvs/aorta)
source ~/venvs/aorta/bin/activate

# Run for each configuration to generate needed reports
./scripts/tracelens_single_config/run_tracelens_single_config.sh experiments/rccl_warp_speed_20251120_212921/56cu_256threads
./scripts/tracelens_single_config/run_tracelens_single_config.sh experiments/rccl_warp_speed_20251120_212921/37cu_384threads
./scripts/tracelens_single_config/run_tracelens_single_config.sh experiments/rccl_warp_speed_20251120_212921/32cu_512threads
```
Replace the timestamp with your actual output directory from Step 1.

This generates individual and collective reports needed for comparison.

### Step 3: Compare the 3 configurations (in venv)

**Compare 56cu (baseline) vs 37cu:**
```bash
python scripts/tracelens_single_config/run_full_analysis.py \
  --baseline experiments/rccl_warp_speed_20251120_212921/56cu_256threads \
  --test experiments/rccl_warp_speed_20251120_212921/37cu_384threads \
  --output comparison_56cu_vs_37cu \
  --all --skip-tracelens
```

**Compare 56cu (baseline) vs 32cu:**
```bash
python scripts/tracelens_single_config/run_full_analysis.py \
  --baseline experiments/rccl_warp_speed_20251120_212921/56cu_256threads \
  --test experiments/rccl_warp_speed_20251120_212921/32cu_512threads \
  --output comparison_56cu_vs_32cu \
  --all --skip-tracelens
```

**Compare 37cu (baseline) vs 32cu:**
```bash
python scripts/tracelens_single_config/run_full_analysis.py \
  --baseline experiments/rccl_warp_speed_20251120_212921/37cu_384threads \
  --test experiments/rccl_warp_speed_20251120_212921/32cu_512threads \
  --output comparison_37cu_vs_32cu \
  --all --skip-tracelens
```

Each comparison generates multiple Excel files:

**Individual Comparison Files:**
- `gpu_timeline_combined.xlsx` - Raw GPU timeline data (baseline + test)
- `gpu_timeline_comparison.xlsx` - GPU kernel analysis with deltas and color coding
- `collective_combined.xlsx` - Raw collective operations data (baseline + test)
- `collective_comparison.xlsx` - RCCL collective operations analysis with deltas

**Comprehensive Report (ALL-IN-ONE):**
- `final_analysis_report.xlsx` - **Complete analysis with:**
  - Summary Dashboard (first sheet with key metrics)
  - All comparison sheets from above files
  - Color coding (green=better, red=worse)
  - Excel tables with filters
  - Raw data sheets (hidden but accessible)

**Recommended:** Use `final_analysis_report.xlsx` as it contains everything in one file.

### Tabs in final_analysis_report.xlsx:

**1. Summary_Dashboard** (First Sheet)
- Key metrics at a glance
- Baseline vs Test comparison
- Improvement percentages
- Status indicators (Better/Worse/Similar)

**2. GPU_Summary_Cmp**
- GPU kernel summary comparison
- Overall GPU timeline metrics with deltas

**3. GPU_ByRank_Cmp**
- GPU kernel comparison broken down by rank
- Per-rank GPU performance analysis

**4. NCCL_ImplSync_Cmp**
- NCCL implicit synchronization comparison
- Communication overhead analysis

**5. NCCL_Long_Cmp**
- Detailed NCCL operations comparison
- Long-form collective operations analysis

Note: Additional raw data sheets are hidden but can be unhidden in Excel (right-click any tab → Unhide).

## Output Structure

After Step 1:
```
experiments/rccl_warp_speed_*/
├── 56cu_256threads/
│   └── torch_profiler/rank*/trace.json
├── 37cu_384threads/
│   └── torch_profiler/rank*/trace.json
└── 32cu_512threads/
    └── torch_profiler/rank*/trace.json
```

After Step 2:
```
experiments/rccl_warp_speed_*/tracelens_analysis/
├── individual_reports/
│   ├── rank0_individual_report.xlsx
│   └── ...
├── collective_reports/
│   └── collective_all_ranks.xlsx
└── comparison_report.xlsx
```

## Notes

- **Step 1** (training) must be run inside the Docker container
- **Steps 2-3** (analysis) should be run outside Docker in your Python venv
  - Optional: You can also run analysis inside Docker if you prefer
- Each run takes approximately 5 minutes per configuration
- Results are saved as timestamped directories
- All reports are in Excel format with color-coded comparisons
- After Step 1, stop the Docker container (unless running analysis inside):
  ```bash
  docker compose -f docker-compose.rccl-warpspeed.yaml down
  ```
