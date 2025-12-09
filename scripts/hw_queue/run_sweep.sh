#!/bin/bash
# Run stream count sweep for hardware queue evaluation
#
# Usage:
#   ./run_sweep.sh [workload] [stream_counts] [output_dir]
#
# Examples:
#   ./run_sweep.sh hetero_kernels "1,2,4,8,16,32" results/
#   ./run_sweep.sh moe "4,8,16" results/moe/

set -e

WORKLOAD="${1:-hetero_kernels}"
STREAM_COUNTS="${2:-1,2,4,8,16,32}"
OUTPUT_DIR="${3:-results}"
ITERATIONS="${ITERATIONS:-100}"
WARMUP="${WARMUP:-10}"
DEVICE="${DEVICE:-cuda:0}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get timestamp for output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/${WORKLOAD}_sweep_${TIMESTAMP}.json"

echo "=========================================="
echo "Hardware Queue Evaluation - Stream Sweep"
echo "=========================================="
echo "Workload:      $WORKLOAD"
echo "Stream counts: $STREAM_COUNTS"
echo "Iterations:    $ITERATIONS (warmup: $WARMUP)"
echo "Device:        $DEVICE"
echo "Output:        $OUTPUT_FILE"
echo "=========================================="
echo

# Run sweep
python -m aorta.hw_queue_eval sweep "$WORKLOAD" \
    --streams "$STREAM_COUNTS" \
    --iterations "$ITERATIONS" \
    --warmup "$WARMUP" \
    --device "$DEVICE" \
    --output "$OUTPUT_FILE"

echo
echo "Results saved to: $OUTPUT_FILE"
