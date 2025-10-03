#!/bin/bash
# Script to reproduce the multi-stream hang with rocprof tracing
# rocprof generates chrome-compatible traces that work better on ROCm

set -e

# RCCL settings from Trial 9 (best trial for minimal overlap)
export RCCL_ENABLE_SDMA=1
export RCCL_NUM_CHANNELS=8
export RCCL_SDMA_WORKERS_PER_CHANNEL=2
export RCCL_BUFFER_SIZE=3407872

# Enable RCCL debug logging
export RCCL_DEBUG=INFO
export RCCL_DEBUG_SUBSYS=INIT,COLL

# Number of GPUs (default 2)
NUM_GPUS=${1:-2}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Set Python path
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"

# Create output directory
OUTPUT_DIR="${REPO_ROOT}/artifacts_hang_repro_rocprof"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Reproducing Multi-Stream Hang with rocprof"
echo "=========================================="
echo "RCCL Settings:"
echo "  RCCL_ENABLE_SDMA: ${RCCL_ENABLE_SDMA}"
echo "  RCCL_NUM_CHANNELS: ${RCCL_NUM_CHANNELS}"
echo "  RCCL_SDMA_WORKERS_PER_CHANNEL: ${RCCL_SDMA_WORKERS_PER_CHANNEL}"
echo "  RCCL_BUFFER_SIZE: ${RCCL_BUFFER_SIZE}"
echo ""
echo "Config: config/reproduce_hang_no_torch_prof.yaml"
echo "Output: ${OUTPUT_DIR}/"
echo "=========================================="
echo ""

# Check if rocprof is available
if ! command -v rocprof &> /dev/null; then
    echo "Warning: rocprof not found. Running without profiling."
    echo "Install ROCm profiler tools for chrome traces."
    echo ""

    # Run without rocprof
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        "${REPO_ROOT}/train.py" \
        --config "${REPO_ROOT}/config/reproduce_hang_no_torch_prof.yaml"
else
    # Run with rocprof
    # Note: rocprof traces each rank separately
    echo "Running with rocprof (this will generate chrome traces)..."

    rocprof \
        --hip-trace \
        --hsa-trace \
        --sys-trace \
        --output-file "${OUTPUT_DIR}/rocprof_trace.csv" \
        torchrun \
            --standalone \
            --nproc_per_node="${NUM_GPUS}" \
            "${REPO_ROOT}/train.py" \
            --config "${REPO_ROOT}/config/reproduce_hang_no_torch_prof.yaml"

    echo ""
    echo "rocprof traces saved to: ${OUTPUT_DIR}/"
    echo "Convert to chrome trace with:"
    echo "  /opt/rocm/bin/rpl2pftrace ${OUTPUT_DIR}/rocprof_trace.csv -o ${OUTPUT_DIR}/trace.json"
fi

echo ""
echo "=========================================="
echo "Training completed (or hung)"
echo "Check ${OUTPUT_DIR}/ for results"
echo "=========================================="
