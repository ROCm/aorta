#!/bin/bash
# Stress test to repeatedly trigger hang attempts
# Hangs may be non-deterministic, so run multiple trials

set -e

NUM_TRIALS=${1:-10}
NUM_GPUS=${2:-8}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Hang Stress Test"
echo "=========================================="
echo "Trials: ${NUM_TRIALS}"
echo "GPUs per trial: ${NUM_GPUS}"
echo "=========================================="
echo ""

for trial in $(seq 1 ${NUM_TRIALS}); do
    echo ""
    echo "=========================================="
    echo "Trial ${trial}/${NUM_TRIALS}"
    echo "=========================================="

    # Run with short max_steps to cycle through quickly
    export OVERRIDE_MAX_STEPS=$((50 + trial * 10))  # Vary steps each trial

    OUTPUT_DIR="${REPO_ROOT}/artifacts_hang_stress_trial${trial}"
    mkdir -p "${OUTPUT_DIR}"

    echo "Running trial ${trial} with ${OVERRIDE_MAX_STEPS} steps..."
    echo "Output: ${OUTPUT_DIR}"

    # Run with timeout
    if timeout 600 "${SCRIPT_DIR}/reproduce_hang.sh" ${NUM_GPUS} > "${OUTPUT_DIR}/output.log" 2>&1; then
        echo "✓ Trial ${trial} completed successfully"
    else
        EXIT_CODE=$?
        if [ ${EXIT_CODE} -eq 124 ]; then
            echo "⚠ Trial ${trial} TIMED OUT (likely hung!)"
            echo ""
            echo "=========================================="
            echo "POTENTIAL HANG DETECTED IN TRIAL ${trial}"
            echo "=========================================="
            echo ""

            # Try to capture debug info
            echo "Attempting to capture debug information..."

            # Find hung processes
            HUNG_PIDS=$(pgrep -f "train.py.*reproduce_hang" || echo "")
            if [ -n "${HUNG_PIDS}" ]; then
                echo "Found potentially hung processes: ${HUNG_PIDS}"

                PARENT_PID=$(echo "${HUNG_PIDS}" | head -1)
                "${SCRIPT_DIR}/debug_hang.sh" "${PARENT_PID}" || true

                # Kill hung processes
                echo "Killing hung processes..."
                for PID in ${HUNG_PIDS}; do
                    kill -9 ${PID} 2>/dev/null || true
                done
            fi

            echo ""
            echo "Trial ${trial} output saved to: ${OUTPUT_DIR}/output.log"
            echo "Debug info saved to: ${REPO_ROOT}/artifacts_hang_repro/hang_debug/"
            echo ""
            echo "Hang successfully reproduced! Stopping stress test."
            exit 0
        else
            echo "✗ Trial ${trial} failed with exit code ${EXIT_CODE}"
        fi
    fi

    # Clean up between trials
    rm -rf "${REPO_ROOT}/artifacts_hang_repro" 2>/dev/null || true

    # Small delay between trials
    sleep 2
done

echo ""
echo "=========================================="
echo "Stress Test Complete"
echo "=========================================="
echo "Ran ${NUM_TRIALS} trials without reproducing hang"
echo ""
echo "Possible reasons:"
echo "  - Hang may require specific hardware (MI350?)"
echo "  - ROCm version may have fix already"
echo "  - Need even more extreme settings"
echo "  - Hang may be very rare/timing-dependent"
echo ""
echo "Try:"
echo "  - Run on customer's exact hardware/ROCm version"
echo "  - Increase NUM_TRIALS: $0 100 8"
echo "  - Check customer's exact RCCL settings"
echo "=========================================="
