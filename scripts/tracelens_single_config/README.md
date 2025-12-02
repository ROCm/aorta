# RCCL Warp Speed Performance Testing

Test RCCL warp_speed_v1 branch performance with distributed training.

## Goal

Test the specific `warp_speed_v1` branch of RCCL from https://github.com/mustafabar/rccl.git

## Setup

### Step 1: Start Container
```bash
cd docker
docker compose -f docker-compose.rocm70_9-1.yaml build  # First time only
docker compose -f docker-compose.rocm70_9-1.yaml up -d
docker compose -f docker-compose.rocm70_9-1.yaml exec torchenv-rocm70 bash
```

### Step 2: Build RCCL warp_speed_v1 Branch (First Time Only)
```bash
# Inside container - check if already built
if [ -f /opt/rccl/build/release/librccl.so ]; then
    echo "RCCL already built"
    cd /opt/rccl && git log --oneline -1
else
    # Build warp_speed_v1 branch
    cd /opt
    git clone --recursive https://github.com/mustafabar/rccl.git
    cd rccl
    git checkout warp_speed_v1
    ./install.sh -l --amdgpu_targets=gfx942
fi
```

This builds RCCL at: `/opt/rccl/build/release/`

### Step 3: Verify Using warp_speed_v1
```bash
# Check we have the right branch
cd /opt/rccl && git branch --show-current  # Should show: warp_speed_v1

# Verify library exists
ls -lh /opt/rccl/build/release/librccl.so*
```

### Step 4: Run Tests (Inside Container)
```bash
cd /workspace/aorta

# The script sets LD_LIBRARY_PATH to use custom RCCL
# and exports warp_speed environment variables

# Run with default 3 configurations
./scripts/tracelens_single_config/run_rccl_warp_speed_comparison.sh

# Or custom configurations
./scripts/tracelens_single_config/run_rccl_warp_speed_comparison.sh -c ./config/single_node/gemm_overlap_comm.yaml -p "56,256 37,384 32,512"
```

The script automatically:
- Sets `LD_LIBRARY_PATH=/opt/rccl/build/release:$LD_LIBRARY_PATH`
- Exports RCCL_WARP_SPEED_ENABLE=1
- Sets RCCL_WARP_SPEED_CU_COUNT and RCCL_THREADS_PER_BLOCK

### Step 5: Generate Reports (Outside Container)
```bash
# Exit container
exit

# Activate Python environment
source ~/venvs/aorta/bin/activate
cd /path/to/aorta

# Replace with your test directory
TESTDIR=rccl_warp_speed_20251202_151333

# Generate individual reports
./scripts/tracelens_single_config/run_tracelens_single_config.sh experiments/${TESTDIR}/56cu_256threads
./scripts/tracelens_single_config/run_tracelens_single_config.sh experiments/${TESTDIR}/37cu_384threads

# Compare configurations
python scripts/tracelens_single_config/run_full_analysis.py \
  --baseline experiments/${TESTDIR}/56cu_256threads \
  --test experiments/${TESTDIR}/37cu_384threads \
  --output comparison_results \
  --all --skip-tracelens
```

## Important Notes

- **We ARE using the custom-built warp_speed_v1 branch**, not PyTorch's bundled RCCL
- The custom RCCL is built inside the container to avoid library compatibility issues
- LD_LIBRARY_PATH prioritizes the custom build over PyTorch's bundled version
- Even though PyTorch reports "NCCL 2.27.7", the warp_speed features come from our custom build

## Test Different RCCL Branches

Inside container:
```bash
cd /opt/rccl
git fetch origin
git checkout develop  # or another branch
git pull
./install.sh -l --amdgpu_targets=gfx942
# Run tests again
```
