#!/usr/bin/env bash
# verify_asan_pytorch.sh — Verify PyTorch works with ASAN-instrumented ROCm.
#
# Run inside the ASAN Docker container:
#   docker run --device=/dev/kfd --device=/dev/dri \
#     --ipc=host --group-add=video --group-add=render \
#     therock-host-asan-pytorch  bash tests/asan_tests/verify_asan_pytorch.sh
#
# ASAN is activated via LD_LIBRARY_PATH (set by Dockerfile ENV).
# No LD_PRELOAD needed — the ASAN runtime loads as a NEEDED dependency.

set -uo pipefail

ROCM=${ROCM_HOME:-/opt/rocm}

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass=0
fail=0
asan_reports=0

run_python() {
    local name="$1"
    local code="$2"

    echo -n "  [$name] "
    set +e
    OUTPUT=$(python3 -c "$code" 2>&1)
    RC=$?
    set -e

    if grep -q "ERROR: AddressSanitizer" <<< "$OUTPUT"; then
        asan_reports=$((asan_reports + 1))
    fi

    if [ "$RC" -eq 0 ]; then
        echo -e "${GREEN}OK${NC} (rc=0)"
        pass=$((pass + 1))
    elif [ "$RC" -eq 139 ]; then
        echo -e "${RED}SEGFAULT${NC} (rc=139)"
        fail=$((fail + 1))
    else
        echo -e "${RED}FAILED${NC} (rc=$RC)"
        echo "$OUTPUT" | tail -5 | sed 's/^/    /'
        fail=$((fail + 1))
    fi
}

echo ""
echo "=========================================="
echo " PyTorch + ASAN Verification"
echo "=========================================="
echo "  ASAN_OPTIONS: ${ASAN_OPTIONS:-not set}"
echo ""

# ===== Section 1: Basic import =====
echo "--- 1. Basic import and version ---"

run_python "import torch" "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  Built with ROCm: {torch.version.hip is not None}')
"

run_python "torch.version details" "
import torch
print(f'  torch.version.hip: {torch.version.hip}')
print(f'  torch.__file__: {torch.__file__}')
"

# ===== Section 2: CPU operations =====
echo ""
echo "--- 2. CPU tensor operations ---"

run_python "tensor creation" "
import torch
x = torch.randn(100, 100)
print(f'  Shape: {x.shape}, dtype: {x.dtype}')
"

run_python "matmul CPU" "
import torch
a = torch.randn(64, 128)
b = torch.randn(128, 64)
c = torch.mm(a, b)
print(f'  Result shape: {c.shape}, mean: {c.mean().item():.4f}')
"

run_python "autograd CPU" "
import torch
x = torch.randn(10, requires_grad=True)
y = (x * x).sum()
y.backward()
print(f'  grad shape: {x.grad.shape}, grad[0]: {x.grad[0].item():.4f}')
"

# ===== Section 3: GPU operations =====
echo ""
echo "--- 3. GPU operations (requires GPU) ---"

run_python "GPU availability" "
import torch
count = torch.cuda.device_count()
print(f'  GPU count: {count}')
if count > 0:
    print(f'  GPU 0: {torch.cuda.get_device_name(0)}')
"

set +e
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
set -e

if [ "$GPU_COUNT" -gt 0 ]; then
    run_python "tensor to GPU" "
import torch
x = torch.randn(100, 100, device='cuda')
print(f'  Shape: {x.shape}, device: {x.device}')
"

    run_python "matmul GPU" "
import torch
a = torch.randn(256, 512, device='cuda')
b = torch.randn(512, 256, device='cuda')
c = torch.mm(a, b)
torch.cuda.synchronize()
print(f'  Result shape: {c.shape}, mean: {c.cpu().mean().item():.4f}')
"

    run_python "GPU autograd" "
import torch
x = torch.randn(100, device='cuda', requires_grad=True)
y = (x * x).sum()
y.backward()
torch.cuda.synchronize()
print(f'  grad mean: {x.grad.mean().item():.6f}')
"

    run_python "GPU streams + events" "
import torch
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
e = torch.cuda.Event()

with torch.cuda.stream(s1):
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.mm(a, a)
e.record(s1)
s2.wait_event(e)
with torch.cuda.stream(s2):
    c = b + 1.0
torch.cuda.synchronize()
print(f'  Streams + events: OK, result shape: {c.shape}')
"

    run_python "hipEventQuery loop (ASAN target)" "
import torch, time

x = torch.randn(2048, 2048, device='cuda')
stream = torch.cuda.Stream()
event = torch.cuda.Event(enable_timing=True)
start_event = torch.cuda.Event(enable_timing=True)

polls = 0
for i in range(10):
    start_event.record(stream)
    with torch.cuda.stream(stream):
        y = torch.mm(x, x)
    event.record(stream)
    while not event.query():
        polls += 1
        time.sleep(0.0001)

torch.cuda.synchronize()
elapsed = start_event.elapsed_time(event)
print(f'  10 iterations, {polls} event polls, last kernel: {elapsed:.2f}ms')
"

    run_python "simple training step" "
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

x = torch.randn(32, 784, device='cuda')
target = torch.randint(0, 10, (32,), device='cuda')

for step in range(5):
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()
print(f'  5 training steps OK, final loss: {loss.item():.4f}')
"
else
    echo -e "  ${YELLOW}[SKIP]${NC} No GPU available — skipping GPU tests"
    echo "         Run with --device=/dev/kfd --device=/dev/dri for GPU tests"
fi

# ===== Section 4: Distributed sanity =====
echo ""
echo "--- 4. Distributed module availability ---"

run_python "torch.distributed importable" "
import torch.distributed as dist
print(f'  Available backends: {dist.Backend.backend_list}')
print(f'  NCCL available: {dist.is_nccl_available()}')
"

# ===== Summary =====
echo ""
echo "=========================================="
echo " Summary"
echo "=========================================="
echo -e "  ${GREEN}Passed: $pass${NC}"
echo -e "  ${RED}Failed: $fail${NC}"
echo -e "  ${YELLOW}ASAN reports observed: $asan_reports${NC}"
echo ""

if [ "$asan_reports" -gt 0 ]; then
    echo -e "${YELLOW}NOTE: ASAN reports were detected. This is expected if there are"
    echo -e "real memory bugs in the ROCm stack. Check stderr for details.${NC}"
fi

if [ "$fail" -gt 0 ]; then
    echo -e "${RED}SOME CHECKS FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}ALL CHECKS PASSED${NC}"
    exit 0
fi
