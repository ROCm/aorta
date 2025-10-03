#!/bin/bash
# Test impact of GPU_MAX_HW_QUEUES on compute latencies
# Run with different queue settings to measure performance impact

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Test with different GPU_MAX_HW_QUEUES values
QUEUE_VALUES=(4 8 16)
NUM_GPUS=${1:-8}
NUM_STEPS=${2:-50}  # Short runs to measure latency

echo "=========================================="
echo "GPU_MAX_HW_QUEUES Impact Test"
echo "=========================================="
echo "Testing queue values: ${QUEUE_VALUES[@]}"
echo "GPUs: ${NUM_GPUS}"
echo "Steps per test: ${NUM_STEPS}"
echo "=========================================="
echo ""

# Base RCCL settings (matching reproduce_hang.sh but less extreme for measurement)
export RCCL_ENABLE_SDMA=1
export RCCL_NUM_CHANNELS=128  # Moderate channel count for testing
export ROCM_MAX_HW_QUEUES=2   # Keep moderate for stability
export RCCL_SDMA_WORKERS_PER_CHANNEL=4
export RCCL_BUFFER_SIZE=512000
export RCCL_DEBUG=WARN  # Less verbose for cleaner output

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"

# Create results directory
RESULTS_DIR="${REPO_ROOT}/gpu_queue_impact_results"
mkdir -p "${RESULTS_DIR}"

echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

for QUEUE_VAL in "${QUEUE_VALUES[@]}"; do
    echo "=========================================="
    echo "Testing GPU_MAX_HW_QUEUES=${QUEUE_VAL}"
    echo "=========================================="

    export GPU_MAX_HW_QUEUES=${QUEUE_VAL}

    OUTPUT_DIR="${RESULTS_DIR}/queues_${QUEUE_VAL}"
    mkdir -p "${OUTPUT_DIR}"

    # Update config to use this output dir and short run
    CONFIG_FILE="${OUTPUT_DIR}/config.yaml"
    cp "${REPO_ROOT}/config/reproduce_hang.yaml" "${CONFIG_FILE}"

    # Modify config for short test run
    sed -i "s|max_steps: .*|max_steps: ${NUM_STEPS}|" "${CONFIG_FILE}"
    sed -i "s|output_dir: .*|output_dir: ${OUTPUT_DIR}|" "${CONFIG_FILE}"

    echo "Running with GPU_MAX_HW_QUEUES=${QUEUE_VAL}..."
    START_TIME=$(date +%s)

    if torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        "${REPO_ROOT}/train.py" \
        --config "${CONFIG_FILE}" \
        > "${OUTPUT_DIR}/training.log" 2>&1; then

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        echo "✓ Completed in ${ELAPSED} seconds"

        # Extract timing info from metrics
        if [ -f "${OUTPUT_DIR}/rank_00_metrics.jsonl" ]; then
            echo "Analyzing metrics..."
            python3 "${SCRIPT_DIR}/analyze_metrics.py" "${OUTPUT_DIR}" > "${OUTPUT_DIR}/analysis.txt" 2>&1 || true

            # Extract average iteration time
            AVG_ITER=$(grep "Average total iteration time:" "${OUTPUT_DIR}/analysis.txt" 2>/dev/null | awk '{print $5}' || echo "N/A")
            AVG_COMPUTE=$(grep "Average compute time:" "${OUTPUT_DIR}/analysis.txt" 2>/dev/null | awk '{print $4}' || echo "N/A")
            AVG_OVERLAP=$(grep "Average overlap ratio:" "${OUTPUT_DIR}/analysis.txt" 2>/dev/null | awk '{print $4}' || echo "N/A")

            echo "  Average iteration time: ${AVG_ITER} ms"
            echo "  Average compute time: ${AVG_COMPUTE} ms"
            echo "  Average overlap ratio: ${AVG_OVERLAP}"

            # Save summary
            echo "GPU_MAX_HW_QUEUES=${QUEUE_VAL}" > "${OUTPUT_DIR}/summary.txt"
            echo "Total time: ${ELAPSED}s" >> "${OUTPUT_DIR}/summary.txt"
            echo "Avg iteration: ${AVG_ITER} ms" >> "${OUTPUT_DIR}/summary.txt"
            echo "Avg compute: ${AVG_COMPUTE} ms" >> "${OUTPUT_DIR}/summary.txt"
            echo "Overlap ratio: ${AVG_OVERLAP}" >> "${OUTPUT_DIR}/summary.txt"
        fi
    else
        EXIT_CODE=$?
        echo "✗ Failed with exit code ${EXIT_CODE}"
        echo "See log: ${OUTPUT_DIR}/training.log"
    fi

    echo ""
    sleep 2  # Small delay between tests
done

echo "=========================================="
echo "Test Complete - Results Summary"
echo "=========================================="
echo ""

# Print comparison table
printf "%-20s %-15s %-20s %-20s %-15s\n" "GPU_MAX_HW_QUEUES" "Total Time" "Avg Iteration (ms)" "Avg Compute (ms)" "Overlap Ratio"
printf "%-20s %-15s %-20s %-20s %-15s\n" "─────────────────" "──────────" "──────────────────" "──────────────────" "─────────────"

for QUEUE_VAL in "${QUEUE_VALUES[@]}"; do
    SUMMARY_FILE="${RESULTS_DIR}/queues_${QUEUE_VAL}/summary.txt"
    if [ -f "${SUMMARY_FILE}" ]; then
        TOTAL_TIME=$(grep "Total time:" "${SUMMARY_FILE}" | awk '{print $3}')
        AVG_ITER=$(grep "Avg iteration:" "${SUMMARY_FILE}" | awk '{print $3, $4}')
        AVG_COMPUTE=$(grep "Avg compute:" "${SUMMARY_FILE}" | awk '{print $3, $4}')
        OVERLAP=$(grep "Overlap ratio:" "${SUMMARY_FILE}" | awk '{print $3}')

        printf "%-20s %-15s %-20s %-20s %-15s\n" "${QUEUE_VAL}" "${TOTAL_TIME}" "${AVG_ITER}" "${AVG_COMPUTE}" "${OVERLAP}"
    fi
done

echo ""
echo "Full results in: ${RESULTS_DIR}"
echo ""
echo "To visualize timelines:"
for QUEUE_VAL in "${QUEUE_VALUES[@]}"; do
    echo "  python3 scripts/visualize_timeline.py ${RESULTS_DIR}/queues_${QUEUE_VAL} --output timeline_queues_${QUEUE_VAL}.png"
done
echo ""
echo "=========================================="
