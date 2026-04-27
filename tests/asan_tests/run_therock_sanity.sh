#!/usr/bin/env bash
# run_therock_sanity.sh — Run TheRock's official test_rocm_sanity.py
#
# This wraps https://github.com/ROCm/TheRock/blob/main/tests/test_rocm_sanity.py
# with the right environment variables for our ASAN Docker container.
#
# Usage (inside the Docker container, with GPU access):
#   /workspace/asan_tests/run_therock_sanity.sh
#   /workspace/asan_tests/run_therock_sanity.sh -v              # verbose
#   /workspace/asan_tests/run_therock_sanity.sh -k test_hip     # run specific test
#   /workspace/asan_tests/run_therock_sanity.sh --co             # list tests only
#
# Requirements:
#   - GPU access (--device=/dev/kfd --device=/dev/dri)
#   - TheRock source tree at /build/TheRock (present in the Docker image)
#
# Note: Some tests are skipped for ASAN builds (rocminfo, hipcc) due to
#   known issues: https://github.com/ROCm/TheRock/issues/3312
#                 https://github.com/ROCm/TheRock/issues/3313

set -euo pipefail

ROCM=${ROCM_HOME:-/opt/rocm}
THEROCK_SRC=${THEROCK_SRC:-/build/TheRock}

# --- Validate environment ---
if [ ! -d "$THEROCK_SRC/tests" ]; then
    echo "ERROR: TheRock source tree not found at $THEROCK_SRC/tests"
    echo "       Expected from the Docker build. Set THEROCK_SRC if different."
    exit 1
fi

if [ ! -f "$THEROCK_SRC/tests/test_rocm_sanity.py" ]; then
    echo "ERROR: test_rocm_sanity.py not found in $THEROCK_SRC/tests/"
    exit 1
fi

if [ ! -d "$ROCM/bin" ]; then
    echo "ERROR: ROCm bin directory not found at $ROCM/bin"
    exit 1
fi

# --- Install pytest dependencies (if not already installed) ---
if ! python3 -c "import pytest" 2>/dev/null; then
    echo "Installing pytest and pytest-check..."
    pip install --quiet pytest pytest-check
fi

if ! python3 -c "import pytest_check" 2>/dev/null; then
    echo "Installing pytest-check..."
    pip install --quiet pytest-check
fi

# --- Set environment variables for test_rocm_sanity.py ---

# THEROCK_BIN_DIR: where rocminfo, hipcc, offload-arch live
export THEROCK_BIN_DIR="$ROCM/bin"

# AMDGPU_FAMILIES: GPU family being tested (from build arg or detected)
export AMDGPU_FAMILIES="${AMDGPU_FAMILIES:-${PYTORCH_ROCM_ARCH:-gfx950}}"

# ARTIFACT_GROUP: must contain "asan" for is_asan() to return True,
# which skips tests known to fail under ASAN (rocminfo, hipcc).
export ARTIFACT_GROUP="therock-asan"

# ASAN is activated via LD_LIBRARY_PATH (baked into the Docker image ENV).
# Do NOT use LD_PRELOAD — it breaks HIP's internal dlopen/dlsym and causes
# SEGVs in amd::Device::init(). The ASAN runtime loads as a NEEDED dependency
# of the ASAN-built libraries; verify_asan_link_order=0 suppresses ASAN's
# complaint about not being loaded first.
export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:halt_on_error=0:symbolize=1:verify_asan_link_order=0}"

# --- Print config ---
echo "============================================================"
echo " TheRock Sanity Tests (test_rocm_sanity.py)"
echo "============================================================"
echo "  THEROCK_BIN_DIR:   $THEROCK_BIN_DIR"
echo "  AMDGPU_FAMILIES:   $AMDGPU_FAMILIES"
echo "  ARTIFACT_GROUP:    $ARTIFACT_GROUP"
echo "  ASAN_OPTIONS:      $ASAN_OPTIONS"
echo "  Test source:       $THEROCK_SRC/tests/test_rocm_sanity.py"
echo "============================================================"
echo ""

# --- Detect which tests to skip based on what was actually built ---
SKIP_FILTERS=""

AMDSMI_TEST="$ROCM/share/amd_smi/tests/amdsmitst"
if [ ! -f "$AMDSMI_TEST" ]; then
    echo "  NOTE: amdsmitst not found — skipping test_amdsmi_suite"
    echo "        (expected with THEROCK_ENABLE_ALL=OFF minimal builds)"
    SKIP_FILTERS="not test_amdsmi_suite"
fi

# --- Run the tests ---
cd "$THEROCK_SRC/tests"

# Pass through any extra arguments (e.g., -v, -k, --co)
if [ -n "$SKIP_FILTERS" ]; then
    exec python3 -m pytest test_rocm_sanity.py -v -k "$SKIP_FILTERS" "$@"
else
    exec python3 -m pytest test_rocm_sanity.py -v "$@"
fi
