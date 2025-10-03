#!/bin/bash
# Run training with hang detection and automatic debugger attachment

set -e

# RCCL settings - EXTREME to trigger hang (matching reproduce_hang.sh)
export RCCL_ENABLE_SDMA=1
export RCCL_NUM_CHANNELS=256
export ROCM_MAX_HW_QUEUES=1
export GPU_MAX_HW_QUEUES=8      # Set to 4, 8, or 16 to reproduce compute latencies
export RCCL_SDMA_WORKERS_PER_CHANNEL=8
export RCCL_BUFFER_SIZE=262144
export RCCL_MIN_NCHANNELS=256
export RCCL_MAX_NCHANNELS=256
export RCCL_ALGO=Tree
export RCCL_PROTO=Simple
export RCCL_IGNORE_CPU_AFFINITY=1
export RCCL_FORCE_ENABLE_DMABUF=1
export RCCL_GRAPH_REGISTER=0
export RCCL_ENABLE_DIRECT_PEER_ACCESS=0

# Enable RCCL debug logging
export RCCL_DEBUG=INFO
export RCCL_DEBUG_SUBSYS=INIT,COLL

# Hang detection settings
HANG_TIMEOUT=${HANG_TIMEOUT:-120}  # Seconds without output before considering it hung
ITERATION_TIMEOUT=${ITERATION_TIMEOUT:-30}  # Expected max time per iteration

NUM_GPUS=${1:-8}  # Default to 8 GPUs for maximum stress

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"

OUTPUT_DIR="${REPO_ROOT}/artifacts_hang_repro"
mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${OUTPUT_DIR}/training.log"

echo "=========================================="
echo "Running with Hang Detection"
echo "=========================================="
echo "RCCL Settings:"
echo "  RCCL_NUM_CHANNELS: ${RCCL_NUM_CHANNELS}"
echo "  ROCM_MAX_HW_QUEUES: ${ROCM_MAX_HW_QUEUES}"
echo "  RCCL_ENABLE_SDMA: ${RCCL_ENABLE_SDMA}"
echo ""
echo "Hang Detection:"
echo "  Timeout: ${HANG_TIMEOUT}s without output"
echo "  Log file: ${LOG_FILE}"
echo "=========================================="
echo ""

# Run training in background
torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    "${REPO_ROOT}/train.py" \
    --config "${REPO_ROOT}/config/reproduce_hang.yaml" \
    2>&1 | tee "${LOG_FILE}" &

TRAIN_PID=$!

echo "Training started with PID: ${TRAIN_PID}"
echo "Monitoring for hangs..."

# Monitor for hangs
LAST_CHANGE=$(date +%s)
LAST_SIZE=0

while kill -0 ${TRAIN_PID} 2>/dev/null; do
    sleep 5

    # Check if log file is still growing
    CURRENT_SIZE=$(stat -f%z "${LOG_FILE}" 2>/dev/null || stat -c%s "${LOG_FILE}" 2>/dev/null || echo 0)
    CURRENT_TIME=$(date +%s)

    if [ "${CURRENT_SIZE}" != "${LAST_SIZE}" ]; then
        LAST_CHANGE=${CURRENT_TIME}
        LAST_SIZE=${CURRENT_SIZE}
    else
        # No change in log size
        TIME_SINCE_CHANGE=$((CURRENT_TIME - LAST_CHANGE))

        if [ ${TIME_SINCE_CHANGE} -gt ${HANG_TIMEOUT} ]; then
            echo ""
            echo "=========================================="
            echo "HANG DETECTED!"
            echo "=========================================="
            echo "No output for ${TIME_SINCE_CHANGE} seconds"
            echo "Training PID: ${TRAIN_PID}"
            echo ""

            # Find all child processes (torchrun spawns multiple processes)
            echo "Finding all training processes..."
            CHILD_PIDS=$(pgrep -P ${TRAIN_PID} || echo "")

            if [ -n "${CHILD_PIDS}" ]; then
                echo "Child processes: ${CHILD_PIDS}"

                # Save process info
                echo "Saving process information..."
                ps aux | grep -E "(${TRAIN_PID}|$(echo ${CHILD_PIDS} | tr ' ' '|'))" > "${OUTPUT_DIR}/hung_processes.txt"

                # Check GPU status
                echo "Checking GPU status..."
                if command -v rocm-smi &> /dev/null; then
                    rocm-smi > "${OUTPUT_DIR}/gpu_status_during_hang.txt" 2>&1
                fi

                echo ""
                echo "To debug manually, attach rocgdb to any worker process:"
                echo ""
                for PID in ${CHILD_PIDS}; do
                    echo "  rocgdb -p ${PID}"
                done
                echo ""
                echo "Or run the debug script:"
                echo "  ${SCRIPT_DIR}/debug_hang.sh ${TRAIN_PID}"
                echo ""
                echo "Processes will remain hung for debugging."
                echo "Press Ctrl+C to kill all processes."

                # Wait for user intervention
                wait ${TRAIN_PID}
                exit 1
            else
                echo "Warning: Could not find child processes"
            fi

            break
        fi
    fi
done

# Check exit status
wait ${TRAIN_PID}
EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed with exit code: ${EXIT_CODE}"
    echo "=========================================="
fi

exit ${EXIT_CODE}
