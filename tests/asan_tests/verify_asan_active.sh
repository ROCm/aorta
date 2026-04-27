#!/usr/bin/env bash
# verify_asan_active.sh — Verify ASAN overlay is active and catching bugs.
#
# Run inside the ASAN Docker container:
#   bash tests/asan_tests/verify_asan_active.sh
#
# The ASAN overlay is baked into the image via ENV directives. This script
# auto-activates it if ASAN_OVERLAY_ACTIVE is not set (e.g. if the env was
# manually cleared).

set -uo pipefail

ROCM=${ROCM_HOME:-/opt/rocm}
ASAN_OVERLAY_DIR=${ASAN_LIB_DIR:-/opt/rocm-asan/lib}
TMPDIR=$(mktemp -d /tmp/asan_verify.XXXXXX)
trap "rm -rf $TMPDIR" EXIT

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

pass=0
fail=0
skip=0

check() {
    local name="$1"
    local result="$2"
    if [ "$result" = "pass" ]; then
        echo -e "  ${GREEN}PASS${NC}  $name"
        pass=$((pass + 1))
    elif [ "$result" = "skip" ]; then
        echo -e "  ${YELLOW}SKIP${NC}  $name"
        skip=$((skip + 1))
    else
        echo -e "  ${RED}FAIL${NC}  $name"
        fail=$((fail + 1))
    fi
}

# =========================================================================
# Auto-activate ASAN overlay if entrypoint hasn't run
# =========================================================================
ASAN_RT=$(find "$ROCM/llvm/lib/clang" -name "libclang_rt.asan-x86_64.so" 2>/dev/null | head -1)
ASAN_RT_DIR=""
[ -n "$ASAN_RT" ] && ASAN_RT_DIR=$(dirname "$ASAN_RT")

if [ "${ASAN_OVERLAY_ACTIVE:-0}" != "1" ]; then
    echo -e "${YELLOW}ASAN overlay not active (ENV may have been cleared). Activating...${NC}"
    echo ""
    if [ -d "$ASAN_OVERLAY_DIR" ] && [ -n "$ASAN_RT_DIR" ]; then
        export LD_LIBRARY_PATH="${ASAN_OVERLAY_DIR}:${ASAN_RT_DIR}:${ROCM}/lib:${ROCM}/llvm/lib:${ROCM}/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:halt_on_error=0:symbolize=1:verify_asan_link_order=0}"
        export HSA_TOOLS_LIB=""
        SYMBOLIZER="$ROCM/llvm/bin/llvm-symbolizer"
        [ -x "$SYMBOLIZER" ] && export ASAN_SYMBOLIZER_PATH="$SYMBOLIZER"
        export ASAN_OVERLAY_ACTIVE=1
    else
        echo -e "${RED}Cannot activate: overlay dir or ASAN runtime not found${NC}"
    fi
fi

echo ""
echo -e "${BOLD}=== ASAN Overlay Verification ===${NC}"
echo ""

# =========================================================================
# Section 1: Static checks (no GPU required)
# =========================================================================
echo -e "${BOLD}--- 1. ASAN overlay files ---${NC}"

if [ -d "$ASAN_OVERLAY_DIR" ]; then
    check "ASAN overlay directory exists ($ASAN_OVERLAY_DIR)" "pass"
else
    check "ASAN overlay directory exists ($ASAN_OVERLAY_DIR)" "fail"
fi

ASAN_HIP="$ASAN_OVERLAY_DIR/libamdhip64.so"
if [ -f "$ASAN_HIP" ]; then
    # Check for ASAN symbols (U = undefined, resolved at runtime by ASAN runtime).
    # Use /usr/bin/nm to avoid ROCm's llvm-nm which may behave differently.
    NM_BIN=$({ which /usr/bin/nm || which nm; } 2>/dev/null)
    if "$NM_BIN" -D "$ASAN_HIP" 2>/dev/null | grep -q '__asan_'; then
        check "Overlay libamdhip64.so has ASAN symbols" "pass"
    elif readelf -d "$ASAN_HIP" 2>/dev/null | grep -q "libclang_rt.asan"; then
        check "Overlay libamdhip64.so links ASAN runtime (NEEDED)" "pass"
    elif strings "$ASAN_HIP" 2>/dev/null | grep -q '__asan_'; then
        check "Overlay libamdhip64.so has ASAN symbols (via strings)" "pass"
    else
        check "Overlay libamdhip64.so has ASAN symbols" "fail"
        echo "         Debug: nm=$NM_BIN, file=$(file -b "$ASAN_HIP" | head -c 80)"
    fi
else
    check "Overlay libamdhip64.so exists" "fail"
fi

NORMAL_HIP="$ROCM/lib/libamdhip64.so"
if [ -f "$NORMAL_HIP" ]; then
    if nm -D "$NORMAL_HIP" 2>/dev/null | grep -q '__asan_'; then
        check "Normal libamdhip64.so is NOT ASAN-instrumented" "fail"
    else
        check "Normal libamdhip64.so is NOT ASAN-instrumented" "pass"
    fi
fi

if [ -n "$ASAN_RT" ]; then
    check "ASAN runtime found: $ASAN_RT" "pass"
else
    check "ASAN runtime found" "fail"
fi

# Check LD_LIBRARY_PATH ordering
if echo "${LD_LIBRARY_PATH:-}" | grep -q "rocm-asan"; then
    check "LD_LIBRARY_PATH includes ASAN overlay dir" "pass"
else
    check "LD_LIBRARY_PATH includes ASAN overlay dir" "fail"
    echo "         LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
fi

# =========================================================================
# Section 2: Runtime library loading (requires Python + torch)
# =========================================================================
echo ""
echo -e "${BOLD}--- 2. Runtime library loading ---${NC}"

if command -v python3 >/dev/null 2>&1 && python3 -c "import torch" 2>/dev/null; then
    MAPS_OUTPUT=$(python3 -c "
import torch, os
torch.cuda.device_count()
maps = open(f'/proc/{os.getpid()}/maps').read()
found_asan_hip = False
found_asan_rt = False
for line in maps.splitlines():
    if 'rocm-asan' in line and 'libamdhip64' in line:
        found_asan_hip = True
    if 'libclang_rt.asan' in line:
        found_asan_rt = True
print(f'asan_hip={found_asan_hip}')
print(f'asan_rt={found_asan_rt}')
" 2>/dev/null || echo "error=true")

    if echo "$MAPS_OUTPUT" | grep -q "asan_hip=True"; then
        check "ASAN libamdhip64.so loaded from /opt/rocm-asan/lib" "pass"
    else
        check "ASAN libamdhip64.so loaded from /opt/rocm-asan/lib" "fail"
        echo "         Normal lib may be loaded instead."
    fi

    if echo "$MAPS_OUTPUT" | grep -q "asan_rt=True"; then
        check "ASAN runtime (libclang_rt.asan) loaded in process" "pass"
    else
        check "ASAN runtime (libclang_rt.asan) loaded in process" "fail"
    fi
else
    check "Python + torch import" "skip"
    check "ASAN runtime loaded" "skip"
fi

# =========================================================================
# Section 3: ASAN detects intentional bugs
# =========================================================================
echo ""
echo -e "${BOLD}--- 3. ASAN bug detection (compile + run C++ test) ---${NC}"

CLANG="$ROCM/llvm/bin/clang++"

if [ ! -x "$CLANG" ]; then
    check "clang++ available" "skip"
    echo "         $CLANG not found"
else
    # Write a test program. No HIP needed — pure host-side ASAN tests.
    # Compiled with the SAME clang that ships the ASAN runtime to avoid
    # "incompatible ASan runtimes" errors.
    cat > "$TMPDIR/asan_test.cpp" << 'CPPEOF'
#include <cstdio>
#include <cstdlib>
#include <cstring>

int test_heap_overflow() {
    printf("[BUG] heap-buffer-overflow: writing 1 past end of buffer\n");
    float* buf = (float*)malloc(10 * sizeof(float));
    buf[10] = 42.0f;  // OOB write
    printf("wrote buf[10]=%f\n", buf[10]);
    free(buf);
    return 0;
}

int test_use_after_free() {
    printf("[BUG] use-after-free: reading freed buffer\n");
    float* buf = (float*)malloc(32 * sizeof(float));
    buf[0] = 42.0f;
    free(buf);
    volatile float val = buf[0];  // UAF read
    printf("read=%f\n", (float)val);
    return 0;
}

int test_clean() {
    printf("[CLEAN] Correct code — no ASAN errors expected\n");
    float* buf = (float*)malloc(10 * sizeof(float));
    for (int i = 0; i < 10; i++) buf[i] = (float)i;
    float sum = 0;
    for (int i = 0; i < 10; i++) sum += buf[i];
    free(buf);
    printf("sum=%f CORRECT\n", sum);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <test>\n", argv[0]); return 1; }
    if (strcmp(argv[1], "heap_overflow") == 0) return test_heap_overflow();
    if (strcmp(argv[1], "use_after_free") == 0) return test_use_after_free();
    if (strcmp(argv[1], "clean") == 0) return test_clean();
    fprintf(stderr, "Unknown: %s\n", argv[1]); return 1;
}
CPPEOF

    # Compile with ASAN using TheRock's clang (same compiler = same ASAN runtime)
    if "$CLANG" -fsanitize=address -o "$TMPDIR/asan_test" "$TMPDIR/asan_test.cpp" 2>"$TMPDIR/compile.err"; then
        check "Compiled ASAN test binary (host C++)" "pass"
    else
        check "Compiled ASAN test binary" "fail"
        cat "$TMPDIR/compile.err" | head -3 | sed 's/^/         /'
    fi

    if [ -x "$TMPDIR/asan_test" ]; then
        # The binary links its own ASAN runtime. We just need the runtime dir
        # on LD_LIBRARY_PATH so it can find it. No overlay conflict because
        # this is a plain C++ binary (no HIP, no libamdhip64).
        TEST_ENV="ASAN_OPTIONS=detect_leaks=0:halt_on_error=0:symbolize=1"
        [ -n "$ASAN_RT_DIR" ] && TEST_ENV="$TEST_ENV LD_LIBRARY_PATH=$ASAN_RT_DIR:${LD_LIBRARY_PATH:-}"

        # Test: heap-buffer-overflow
        OUTPUT=$(env $TEST_ENV "$TMPDIR/asan_test" heap_overflow 2>&1 || true)
        if echo "$OUTPUT" | grep -q "heap-buffer-overflow"; then
            check "ASAN detects heap-buffer-overflow" "pass"
        else
            check "ASAN detects heap-buffer-overflow" "fail"
            echo "$OUTPUT" | grep -i "asan\|error\|sanitizer" | head -2 | sed 's/^/         /'
        fi

        # Test: use-after-free
        OUTPUT=$(env $TEST_ENV "$TMPDIR/asan_test" use_after_free 2>&1 || true)
        if echo "$OUTPUT" | grep -qi "use-after-free\|heap-use-after-free"; then
            check "ASAN detects use-after-free" "pass"
        else
            check "ASAN detects use-after-free" "fail"
            echo "$OUTPUT" | grep -i "asan\|error\|sanitizer" | head -2 | sed 's/^/         /'
        fi

        # Test: clean code — no ASAN reports
        OUTPUT=$(env $TEST_ENV "$TMPDIR/asan_test" clean 2>&1 || true)
        if echo "$OUTPUT" | grep -q "ERROR: AddressSanitizer"; then
            check "Clean code produces NO false positives" "fail"
            echo "         Unexpected ASAN report on clean code"
        elif echo "$OUTPUT" | grep -q "CORRECT"; then
            check "Clean code produces NO false positives" "pass"
        else
            check "Clean code" "fail"
            echo "$OUTPUT" | tail -2 | sed 's/^/         /'
        fi
    fi
fi

# =========================================================================
# Section 4: PyTorch under ASAN overlay (requires GPU)
# =========================================================================
echo ""
echo -e "${BOLD}--- 4. PyTorch under ASAN overlay ---${NC}"

if command -v python3 >/dev/null 2>&1 && python3 -c "import torch" 2>/dev/null; then
    DEV_OUTPUT=$(python3 -c "
import torch
n = torch.cuda.device_count()
print(f'devices={n}')
" 2>&1 || true)

    DEV_COUNT=$(echo "$DEV_OUTPUT" | grep -oP 'devices=\K\d+' || echo "0")
    if [ "$DEV_COUNT" -gt 0 ]; then
        check "torch.cuda.device_count() = $DEV_COUNT" "pass"
    else
        check "torch.cuda.device_count()" "skip"
    fi

    if [ "$DEV_COUNT" -gt 0 ]; then
        # GPU matmul
        GPU_OUTPUT=$(python3 -c "
import torch
a = torch.randn(512, 512, device='cuda')
b = torch.randn(512, 512, device='cuda')
c = torch.mm(a, b)
torch.cuda.synchronize()
print(f'matmul_ok shape={c.shape}')
" 2>&1 || true)

        if echo "$GPU_OUTPUT" | grep -q "matmul_ok"; then
            if echo "$GPU_OUTPUT" | grep -q "ERROR: AddressSanitizer"; then
                check "GPU matmul (ASAN report detected — memory bug found!)" "pass"
                echo -e "         ${YELLOW}ASAN found a memory issue during GPU matmul${NC}"
            else
                check "GPU matmul completes cleanly under ASAN" "pass"
            fi
        else
            check "GPU matmul under ASAN" "fail"
            echo "$GPU_OUTPUT" | tail -3 | sed 's/^/         /'
        fi

        # hipEventQuery stress test
        EVENT_OUTPUT=$(python3 -c "
import torch, time
x = torch.randn(1024, 1024, device='cuda')
stream = torch.cuda.Stream()
event = torch.cuda.Event()
polls = 0
for _ in range(5):
    with torch.cuda.stream(stream):
        y = torch.mm(x, x)
    event.record(stream)
    while not event.query():
        polls += 1
        time.sleep(0.00001)
torch.cuda.synchronize()
print(f'event_query_ok polls={polls}')
" 2>&1 || true)

        if echo "$EVENT_OUTPUT" | grep -q "event_query_ok"; then
            POLLS=$(echo "$EVENT_OUTPUT" | grep -oP 'polls=\K\d+' || echo "?")
            if echo "$EVENT_OUTPUT" | grep -q "ERROR: AddressSanitizer"; then
                check "hipEventQuery stress ($POLLS polls, ASAN report detected!)" "pass"
                echo -e "         ${YELLOW}ASAN found a memory issue in hipEventQuery path${NC}"
            else
                check "hipEventQuery stress ($POLLS polls, clean)" "pass"
            fi
        else
            check "hipEventQuery stress test" "fail"
        fi
    fi
else
    check "PyTorch available" "skip"
fi

# =========================================================================
# Summary
# =========================================================================
echo ""
echo -e "${BOLD}=== Summary ===${NC}"
echo -e "  ${GREEN}Passed: $pass${NC}"
[ $fail -gt 0 ] && echo -e "  ${RED}Failed: $fail${NC}"
[ $skip -gt 0 ] && echo -e "  ${YELLOW}Skipped: $skip${NC}"
echo ""

if [ $fail -gt 0 ]; then
    echo -e "${RED}ASAN verification FAILED — check output above${NC}"
    exit 1
else
    echo -e "${GREEN}ASAN verification PASSED${NC}"
    exit 0
fi
