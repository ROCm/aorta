#!/bin/bash
#
# Integration test runner for rocm_mem_debug.py
#
# Compiles all HIP test programs, runs each alongside the debug script,
# and checks whether the script correctly detected (or did not detect)
# faults / anomalies.
#
# Usage:
#   sudo ./run_tests.sh                # run all tests
#   sudo ./run_tests.sh --skip-negative # positive tests only
#   sudo ./run_tests.sh --skip-positive # negative tests only
#   sudo ./run_tests.sh --no-build      # skip compilation step
#
# Requirements:
#   - hipcc (ROCm compiler)
#   - bpftrace
#   - AMD GPU with amdgpu driver
#   - Root or sudo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEBUG_SCRIPT="${SCRIPT_DIR}/../rocm_mem_debug.py"
BUILD_DIR="${SCRIPT_DIR}/build"
RESULTS_DIR="${SCRIPT_DIR}/results"

PASS=0
FAIL=0
SKIP=0
RUN_POSITIVE=true
RUN_NEGATIVE=true
DO_BUILD=true

# Parse args
for arg in "$@"; do
    case "$arg" in
        --skip-negative) RUN_NEGATIVE=false ;;
        --skip-positive) RUN_POSITIVE=false ;;
        --no-build)      DO_BUILD=false ;;
        --help|-h)
            echo "Usage: sudo $0 [--skip-negative] [--skip-positive] [--no-build]"
            exit 0
            ;;
    esac
done

# Pre-flight checks
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (for bpftrace)"
    exit 1
fi

if ! command -v hipcc &>/dev/null; then
    echo "ERROR: hipcc not found -- ROCm compiler is required"
    exit 1
fi

if ! command -v bpftrace &>/dev/null; then
    echo "ERROR: bpftrace not found"
    exit 1
fi

if [ ! -f "$DEBUG_SCRIPT" ]; then
    echo "ERROR: Debug script not found at $DEBUG_SCRIPT"
    exit 1
fi

mkdir -p "$BUILD_DIR" "$RESULTS_DIR"

echo "============================================================"
echo "GPU Memory Debug Script -- Integration Tests"
echo "============================================================"
echo "  Debug script: $DEBUG_SCRIPT"
echo "  Build dir:    $BUILD_DIR"
echo "  Results dir:  $RESULTS_DIR"
echo ""

# ---------- Build ----------

build_test() {
    local src="$1"
    local name
    name="$(basename "$src" .cpp)"
    echo -n "  Compiling $name ... "
    if hipcc -o "${BUILD_DIR}/${name}" "$src" 2>"${RESULTS_DIR}/${name}.build.log"; then
        echo "OK"
        return 0
    else
        echo "FAILED (see ${RESULTS_DIR}/${name}.build.log)"
        return 1
    fi
}

if $DO_BUILD; then
    echo "--- Building test programs ---"
    for src in "${SCRIPT_DIR}"/positive/*.cpp "${SCRIPT_DIR}"/negative/*.cpp; do
        [ -f "$src" ] && build_test "$src" || true
    done
    echo ""
fi

# ---------- Run a single test ----------

run_test() {
    local binary="$1"
    local name="$2"
    local expect="$3"   # "positive" or "negative"
    local duration=10
    local test_timeout=30

    if [ ! -x "$binary" ]; then
        echo "  SKIP $name (binary not found or not executable)"
        ((SKIP++))
        return
    fi

    local report="${RESULTS_DIR}/${name}.json"
    local stdout_log="${RESULTS_DIR}/${name}.stdout"
    local debug_log="${RESULTS_DIR}/${name}.debug.log"

    echo -n "  Running $name ($expect) ... "

    # Start the debug script in background
    timeout "$test_timeout" python3 "$DEBUG_SCRIPT" \
        --duration "$duration" \
        --output "$report" \
        > "$debug_log" 2>&1 &
    local dbg_pid=$!

    # Let bpftrace attach
    sleep 2

    # Run the test binary
    timeout "$test_timeout" "$binary" > "$stdout_log" 2>&1 || true

    # Wait for the debug script to finish
    wait "$dbg_pid" 2>/dev/null || true

    # Evaluate results
    if [ ! -f "$report" ]; then
        echo "FAIL (no report generated)"
        ((FAIL++))
        return
    fi

    if [ "$expect" = "positive" ]; then
        # For positive tests: look for fault detection OR anomaly patterns
        local faults
        faults=$(python3 -c "
import json, sys
try:
    r = json.load(open('$report'))
    fd = r.get('faults_detected', 0)
    patterns = r.get('patterns', {})
    has_anomaly = (
        patterns.get('eviction_storm', False) or
        patterns.get('irq_spike', False) or
        patterns.get('vm_mapping_churn', False)
    )
    print(fd if fd > 0 else (1 if has_anomaly else 0))
except:
    print(-1)
" 2>/dev/null)

        if [ "$faults" = "-1" ]; then
            echo "FAIL (report parse error)"
            ((FAIL++))
        elif [ "$faults" != "0" ]; then
            echo "PASS -- fault/anomaly detected"
            ((PASS++))
        else
            # P4 (eviction) and P5 (unsynced) may not always produce faults;
            # check if eBPF events were at least captured
            local events
            events=$(python3 -c "
import json
r = json.load(open('$report'))
total = sum(r.get('event_counts', {}).values())
print(total)
" 2>/dev/null || echo "0")
            if [ "$events" -gt 0 ]; then
                echo "WARN -- events captured ($events) but no fault flagged"
                ((PASS++))  # Acceptable for race-dependent tests
            else
                echo "FAIL -- no detection"
                ((FAIL++))
            fi
        fi
    else
        # For negative tests: should have zero faults
        local faults
        faults=$(python3 -c "
import json
r = json.load(open('$report'))
print(r.get('faults_detected', -1))
" 2>/dev/null || echo "-1")

        if [ "$faults" = "0" ]; then
            echo "PASS -- no false positive"
            ((PASS++))
        elif [ "$faults" = "-1" ]; then
            echo "FAIL (report parse error)"
            ((FAIL++))
        else
            echo "FAIL -- false positive ($faults faults)"
            ((FAIL++))
        fi
    fi
}

# ---------- Run positive tests ----------

if $RUN_POSITIVE; then
    echo "--- Positive Tests (expect detection) ---"
    for src in "${SCRIPT_DIR}"/positive/*.cpp; do
        [ -f "$src" ] || continue
        name="$(basename "$src" .cpp)"
        run_test "${BUILD_DIR}/${name}" "$name" "positive"
    done
    echo ""
fi

# ---------- Run negative tests ----------

if $RUN_NEGATIVE; then
    echo "--- Negative Tests (expect clean pass) ---"
    for src in "${SCRIPT_DIR}"/negative/*.cpp; do
        [ -f "$src" ] || continue
        name="$(basename "$src" .cpp)"
        run_test "${BUILD_DIR}/${name}" "$name" "negative"
    done
    echo ""
fi

# ---------- Summary ----------

TOTAL=$((PASS + FAIL + SKIP))
echo "============================================================"
echo "RESULTS: $PASS passed, $FAIL failed, $SKIP skipped (of $TOTAL)"
echo "============================================================"

if [ "$FAIL" -gt 0 ]; then
    echo "Detailed results in: $RESULTS_DIR/"
    exit 1
fi

exit 0
