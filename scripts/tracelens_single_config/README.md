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

# Verify you're using RCCL warp_speed_v1
echo "[INFO] RCCL installation path: $RCCL_ROOT"
ls -la /opt/rccl-warpspeed/lib/librccl.so
ldd $(which python) | grep rccl  # Check which RCCL library Python will use
echo "[INFO] LD_LIBRARY_PATH: $LD_LIBRARY_PATH" | grep rccl-warpspeed

# Optional: Check RCCL build info
cat /opt/rccl/.git/HEAD  # Should show warp_speed_v1
git -C /opt/rccl log --oneline -1  # Shows the latest commit

# Now move to workspace
cd /workspace/aorta
```

### For Steps 2-3 (Analysis) - Python venv

Analysis scripts only process trace files and don't need RCCL. Run in your Python virtual environment (e.g., `~/venvs/aorta`).

## Quick Start

### Step 1: Generate traces (MUST run inside Docker container)
```bash
# Default configurations (56cu/256threads, 37cu/384threads, 32cu/512threads)
./scripts/tracelens_single_config/run_rccl_warp_speed_comparison.sh

# Or specify custom configurations
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

Each comparison generates:
- `gpu_timeline_comparison.xlsx` - GPU kernel analysis with deltas
- `collective_comparison.xlsx` - RCCL operation analysis with deltas
- `final_analysis_report.xlsx` - Comprehensive report with dashboard

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
