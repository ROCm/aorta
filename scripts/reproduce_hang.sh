#!/bin/bash
# Script to reproduce the multi-stream hang issue
# Uses settings from Optuna Trial 9 which achieved minimal overlap (1.16%) with multiple streams

set -e

# RCCL settings - EXTREME to trigger hang
export RCCL_ENABLE_SDMA=1
export RCCL_NUM_CHANNELS=256    # Extreme channel count
export ROCM_MAX_HW_QUEUES=1     # Absolute minimum - force maximum contention (for hangs)
export GPU_MAX_HW_QUEUES=8      # Set to 4, 8, or 16 to reproduce compute latencies
export RCCL_SDMA_WORKERS_PER_CHANNEL=8  # Way beyond reasonable
export RCCL_BUFFER_SIZE=262144  # 256KB - very small buffers
export RCCL_MIN_NCHANNELS=256   # Force minimum channels
export RCCL_MAX_NCHANNELS=256   # Force maximum channels

# Additional RCCL stress settings
export RCCL_ALGO=Tree           # Force tree algorithm (more complex)
export RCCL_PROTO=Simple        # Simple protocol
export RCCL_IGNORE_CPU_AFFINITY=1
export RCCL_FORCE_ENABLE_DMABUF=1  # Force DMA buffer usage

# Disable RCCL optimizations that might prevent the hang
export RCCL_GRAPH_REGISTER=0    # Disable graph registration
export RCCL_ENABLE_DIRECT_PEER_ACCESS=0  # Disable direct peer access

# Enable RCCL debug logging to verify stream usage
export RCCL_DEBUG=INFO
export RCCL_DEBUG_SUBSYS=INIT,COLL

# Number of GPUs (default 8 for maximum multi-GPU stress)
NUM_GPUS=${1:-8}  # More GPUs = more RCCL traffic and contention

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Set Python path
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"

echo "=========================================="
echo "Reproducing Multi-Stream Hang"
echo "=========================================="
echo "RCCL Settings:"
echo "  RCCL_ENABLE_SDMA: ${RCCL_ENABLE_SDMA}"
echo "  RCCL_NUM_CHANNELS: ${RCCL_NUM_CHANNELS}"
echo "  RCCL_SDMA_WORKERS_PER_CHANNEL: ${RCCL_SDMA_WORKERS_PER_CHANNEL}"
echo "  RCCL_BUFFER_SIZE: ${RCCL_BUFFER_SIZE}"
echo ""
echo "Config: config/reproduce_hang.yaml"
echo "Output: artifacts_hang_repro/"
echo "=========================================="
echo ""

# Run training with aggressive overlap settings
torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    "${REPO_ROOT}/train.py" \
    --config "${REPO_ROOT}/config/reproduce_hang.yaml"

echo ""
echo "=========================================="
echo "Training completed (or hung)"
echo "Check artifacts_hang_repro/ for results"
echo "=========================================="
