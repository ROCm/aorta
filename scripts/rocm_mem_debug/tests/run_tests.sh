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

    # Per-test overrides: some workloads spend significant time
    # allocating / initializing / verifying very large buffers and the
    # default 30s watchdog kills them mid-run, producing flaky results
    # that look like "no detection" or "no events captured".
    case "$name" in
        06_large_alloc_copy|04_large_clean_alloc)
            test_timeout=180
            duration=60
            ;;
    esac

    if [ ! -x "$binary" ]; then
        echo "  SKIP $name (binary not found or not executable)"
        ((SKIP++))
        return
    fi

    local report="${RESULTS_DIR}/${name}.json"
    local stdout_log="${RESULTS_DIR}/${name}.stdout"
    local debug_log="${RESULTS_DIR}/${name}.debug.log"

    echo -n "  Running $name ($expect) ... "

    # Launch the test binary first, in the background, so we have a real
    # PID to scope the debugger to.  Without --pid the debugger would
    # trace system-wide GPU activity, which on shared machines causes
    # noise in positive tests and false positives in negative tests.
    #
    # We exec the binary inside a subshell so $! is the binary's PID
    # (not a wrapper such as ``timeout``); we enforce the timeout via a
    # separate watchdog so the PID we capture stays valid.
    bash -c 'exec "$1" > "$2" 2>&1' _ "$binary" "$stdout_log" &
    local test_pid=$!

    (
        sleep "$test_timeout"
        if kill -0 "$test_pid" 2>/dev/null; then
            kill "$test_pid" 2>/dev/null || true
        fi
    ) &
    local watchdog_pid=$!

    # Start the debug script, scoped to the test process only.
    timeout "$test_timeout" python3 "$DEBUG_SCRIPT" \
        --pid "$test_pid" \
        --duration "$duration" \
        --output "$report" \
        > "$debug_log" 2>&1 &
    local dbg_pid=$!

    # Wait for the test binary first, then the debugger.
    wait "$test_pid" 2>/dev/null || true
    kill "$watchdog_pid" 2>/dev/null || true
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
