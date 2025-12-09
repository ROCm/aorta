#!/bin/bash
# Profile hardware queue usage with ROCm tools
#
# Usage:
#   ./profile_queues.sh [workload] [stream_count] [output_dir]
#
# Examples:
#   ./profile_queues.sh hetero_kernels 8 traces/
#   ./profile_queues.sh moe 16 traces/moe/
#
# Environment variables:
#   AMD_LOG_LEVEL: Set to 4 for detailed queue logging
#   GPU_MAX_HW_QUEUES: Override max hardware queues

set -e

WORKLOAD="${1:-hetero_kernels}"
STREAM_COUNT="${2:-8}"
OUTPUT_DIR="${3:-traces}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PREFIX="${OUTPUT_DIR}/${WORKLOAD}_${STREAM_COUNT}s_${TIMESTAMP}"

echo "=========================================="
echo "Hardware Queue Profiling"
echo "=========================================="
echo "Workload:     $WORKLOAD"
echo "Streams:      $STREAM_COUNT"
echo "Output:       $OUTPUT_PREFIX.*"
echo "=========================================="
echo

# Check if rocprof is available
if ! command -v rocprof &> /dev/null; then
    echo "Warning: rocprof not found. Creating standalone profiling script instead."

    # Create standalone script
    cat > "${OUTPUT_PREFIX}_profile.py" << 'SCRIPT'
#!/usr/bin/env python3
"""Auto-generated profiling script."""

import sys
sys.path.insert(0, '.')

from aorta.hw_queue_eval.workloads.registry import get_workload
from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

def main():
    workload_name = "${WORKLOAD}"
    stream_count = ${STREAM_COUNT}

    workload = get_workload(workload_name)
    config = HarnessConfig(
        stream_count=stream_count,
        warmup_iterations=5,
        measurement_iterations=50,
    )

    harness = StreamHarness(config)
    result = harness.run_workload(workload)

    print(f"Throughput: {result.throughput:.2f} {result.throughput_unit}")
    print(f"P99 Latency: {result.latency_ms['p99']:.3f} ms")

if __name__ == "__main__":
    main()
SCRIPT

    # Substitute variables
    sed -i "s/\${WORKLOAD}/${WORKLOAD}/g" "${OUTPUT_PREFIX}_profile.py"
    sed -i "s/\${STREAM_COUNT}/${STREAM_COUNT}/g" "${OUTPUT_PREFIX}_profile.py"
    chmod +x "${OUTPUT_PREFIX}_profile.py"

    echo "Created: ${OUTPUT_PREFIX}_profile.py"
    echo
    echo "Run with rocprof:"
    echo "  rocprof --hip-trace -o ${OUTPUT_PREFIX}.csv python ${OUTPUT_PREFIX}_profile.py"
    exit 0
fi

# Set ROCm environment for better tracing
export AMD_LOG_LEVEL=4
export HSA_TOOLS_LIB=""

# Run with rocprof
echo "Running profiler..."
rocprof --hip-trace -o "${OUTPUT_PREFIX}.csv" \
    python -m aorta.hw_queue_eval run "$WORKLOAD" \
        --streams "$STREAM_COUNT" \
        --iterations 50 \
        --warmup 5

echo
echo "Profiling complete!"
echo "Output files:"
echo "  CSV:  ${OUTPUT_PREFIX}.csv"

# Generate timeline if possible
if command -v python3 &> /dev/null; then
    echo
    echo "Generating timeline..."
    python3 -c "
from aorta.hw_queue_eval.core.profiler import ROCmProfiler
from pathlib import Path

profiler = ROCmProfiler('${OUTPUT_DIR}')
trace_file = Path('${OUTPUT_PREFIX}.csv')

if trace_file.exists():
    queue_info = profiler.parse_queue_info(trace_file)
    print(f'  Queues used: {queue_info.num_queues}')
    print(f'  Kernels per queue: {queue_info.kernels_per_queue}')

    timeline = profiler.generate_timeline(trace_file)
    print(f'  Timeline: {timeline}')
" 2>/dev/null || echo "  (Timeline generation skipped)"
fi
