#!/usr/bin/env bash
# run_all_tests.sh — Run all ASAN verification tests.
#
# Usage (inside the Docker container):
#   /workspace/asan_tests/run_all_tests.sh           # all tests
#   /workspace/asan_tests/run_all_tests.sh therock    # TheRock ASAN verification
#   /workspace/asan_tests/run_all_tests.sh pytorch    # PyTorch ASAN verification
#   /workspace/asan_tests/run_all_tests.sh sanity     # TheRock official sanity tests
#   /workspace/asan_tests/run_all_tests.sh asan       # pytest ASAN feature tests
#
# ASAN reports are written to /workspace/asan_tests/results/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUITE="${1:-all}"

echo "============================================================"
echo " ASAN Verification Test Suite"
echo " Time:    $(date)"
echo " Suite:   $SUITE"
echo " Results: $RESULTS_DIR"
echo "============================================================"
echo ""

rc=0

if [ "$SUITE" = "all" ] || [ "$SUITE" = "therock" ]; then
    echo ">>> Running TheRock ASAN verification..."
    echo ""
    bash "$SCRIPT_DIR/verify_asan_therock.sh" \
        2>&1 | tee "$RESULTS_DIR/therock_${TIMESTAMP}.log" || rc=1
    echo ""
fi

if [ "$SUITE" = "all" ] || [ "$SUITE" = "pytorch" ]; then
    echo ">>> Running PyTorch ASAN verification..."
    echo ""
    bash "$SCRIPT_DIR/verify_asan_pytorch.sh" \
        2>&1 | tee "$RESULTS_DIR/pytorch_${TIMESTAMP}.log" || rc=1
    echo ""
fi

if [ "$SUITE" = "all" ] || [ "$SUITE" = "asan" ]; then
    echo ">>> Running pytest ASAN feature verification (test_therock_asan.py)..."
    echo ""
    python3 -m pytest "$SCRIPT_DIR/test_therock_asan.py" -v \
        2>&1 | tee "$RESULTS_DIR/asan_features_${TIMESTAMP}.log" || rc=1
    echo ""
fi

if [ "$SUITE" = "all" ] || [ "$SUITE" = "sanity" ]; then
    echo ">>> Running TheRock official sanity tests (test_rocm_sanity.py)..."
    echo ""
    bash "$SCRIPT_DIR/run_therock_sanity.sh" \
        2>&1 | tee "$RESULTS_DIR/sanity_${TIMESTAMP}.log" || rc=1
    echo ""
fi

echo "============================================================"
echo " Logs saved to: $RESULTS_DIR/"
ls -la "$RESULTS_DIR"/*.log 2>/dev/null || true
echo "============================================================"

exit $rc
