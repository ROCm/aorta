#!/bin/bash
# Debug a hung training process - inspect pending GPU dispatches

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <parent_pid>"
    echo "  parent_pid: The torchrun parent process ID"
    exit 1
fi

PARENT_PID=$1
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${REPO_ROOT}/artifacts_hang_repro"

mkdir -p "${OUTPUT_DIR}/hang_debug"

echo "=========================================="
echo "Debugging Hung Process: ${PARENT_PID}"
echo "=========================================="

# Find all child processes
echo "Finding worker processes..."
CHILD_PIDS=$(pgrep -P ${PARENT_PID} | grep -v $$ || echo "")

if [ -z "${CHILD_PIDS}" ]; then
    echo "Error: No child processes found for PID ${PARENT_PID}"
    exit 1
fi

echo "Found worker PIDs: ${CHILD_PIDS}"

# Save full process tree
echo ""
echo "Saving process tree..."
pstree -p ${PARENT_PID} > "${OUTPUT_DIR}/hang_debug/process_tree.txt" 2>&1 || \
    ps -ef | grep ${PARENT_PID} > "${OUTPUT_DIR}/hang_debug/process_tree.txt"

# Check GPU utilization
echo "Checking GPU status..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi > "${OUTPUT_DIR}/hang_debug/rocm_smi.txt" 2>&1
    rocm-smi --showpids > "${OUTPUT_DIR}/hang_debug/rocm_smi_pids.txt" 2>&1 || true
    rocm-smi --showmeminfo vram > "${OUTPUT_DIR}/hang_debug/rocm_smi_memory.txt" 2>&1 || true
fi

# Get stack traces from all workers
echo ""
echo "Capturing stack traces from worker processes..."

for WORKER_PID in ${CHILD_PIDS}; do
    echo "  Worker PID ${WORKER_PID}..."

    # Get basic stack trace
    if command -v gdb &> /dev/null; then
        timeout 30 gdb -batch -ex "thread apply all bt" -p ${WORKER_PID} \
            > "${OUTPUT_DIR}/hang_debug/stacktrace_${WORKER_PID}.txt" 2>&1 || \
            echo "GDB failed or timed out for PID ${WORKER_PID}"
    fi
done

# Try to get ROCm-specific debug info
echo ""
echo "Attempting to capture ROCm GPU queue state..."

# Check if rocgdb is available
if command -v rocgdb &> /dev/null; then
    echo "rocgdb found - attaching to first worker for GPU state..."
    FIRST_WORKER=$(echo ${CHILD_PIDS} | awk '{print $1}')

    # Create rocgdb command file
    cat > "${OUTPUT_DIR}/hang_debug/rocgdb_commands.txt" << 'EOF'
# Get info about GPU queues and dispatches
info rocm queues
info rocm kernel-exec-info
info threads
thread apply all bt
# Try to show pending dispatches
monitor info dispatches
quit
EOF

    timeout 60 rocgdb -batch -x "${OUTPUT_DIR}/hang_debug/rocgdb_commands.txt" -p ${FIRST_WORKER} \
        > "${OUTPUT_DIR}/hang_debug/rocgdb_gpu_state.txt" 2>&1 || \
        echo "rocgdb failed or timed out"
else
    echo "rocgdb not found - install ROCm debugger for GPU queue inspection"
fi

# Check for pending HIP operations
echo ""
echo "Checking HIP runtime state..."

# Look for HIP/HSA environment info
env | grep -E "(HIP|HSA|ROCM|RCCL)" > "${OUTPUT_DIR}/hang_debug/environment.txt"

# Try to dump HIP API trace if available
if [ -f "/tmp/hip_api_trace_${FIRST_WORKER}.log" ]; then
    cp "/tmp/hip_api_trace_${FIRST_WORKER}.log" "${OUTPUT_DIR}/hang_debug/" || true
fi

# Summary
echo ""
echo "=========================================="
echo "Debug Information Collected"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}/hang_debug/"
echo ""
echo "Files generated:"
ls -lh "${OUTPUT_DIR}/hang_debug/" | tail -n +2
echo ""
echo "Key files to check:"
echo "  - rocgdb_gpu_state.txt : GPU queues and pending dispatches"
echo "  - stacktrace_*.txt     : CPU stack traces for each worker"
echo "  - rocm_smi*.txt        : GPU utilization and memory"
echo ""

# Check for common hang patterns
echo "Analyzing stack traces for common hang patterns..."
echo ""

HANG_PATTERNS_FOUND=0

# Check for hipMemcpy
if grep -q "hipMemcpy" "${OUTPUT_DIR}/hang_debug"/stacktrace_*.txt 2>/dev/null; then
    echo "✓ Found hipMemcpy in stack trace (matches customer issue pattern)"
    HANG_PATTERNS_FOUND=$((HANG_PATTERNS_FOUND + 1))
fi

# Check for RCCL collective operations
if grep -q "ncclAllGather\|ncclReduceScatter\|ncclAllReduce" "${OUTPUT_DIR}/hang_debug"/stacktrace_*.txt 2>/dev/null; then
    echo "✓ Found RCCL collective operations in stack trace"
    HANG_PATTERNS_FOUND=$((HANG_PATTERNS_FOUND + 1))
fi

# Check for stream synchronization
if grep -q "streamSynchronize\|hipStreamWaitEvent" "${OUTPUT_DIR}/hang_debug"/stacktrace_*.txt 2>/dev/null; then
    echo "✓ Found stream synchronization in stack trace"
    HANG_PATTERNS_FOUND=$((HANG_PATTERNS_FOUND + 1))
fi

# Check GPU queue info
if [ -f "${OUTPUT_DIR}/hang_debug/rocgdb_gpu_state.txt" ]; then
    if grep -q "rocprim" "${OUTPUT_DIR}/hang_debug/rocgdb_gpu_state.txt"; then
        echo "✓ Found rocprim kernel (matches customer issue description!)"
        HANG_PATTERNS_FOUND=$((HANG_PATTERNS_FOUND + 1))
    fi

    PENDING_DISPATCHES=$(grep -c "pending" "${OUTPUT_DIR}/hang_debug/rocgdb_gpu_state.txt" 2>/dev/null || echo 0)
    if [ ${PENDING_DISPATCHES} -gt 0 ]; then
        echo "✓ Found ${PENDING_DISPATCHES} pending GPU dispatches"
        HANG_PATTERNS_FOUND=$((HANG_PATTERNS_FOUND + 1))
    fi
fi

echo ""
if [ ${HANG_PATTERNS_FOUND} -gt 0 ]; then
    echo "Detected ${HANG_PATTERNS_FOUND} hang pattern(s)"
    echo ""
    echo "Next steps:"
    echo "  1. Review rocgdb_gpu_state.txt for pending dispatches"
    echo "  2. Check if rocprim kernels are deadlocked"
    echo "  3. Share hang_debug/ directory with AMD/ROCm support"
else
    echo "No obvious hang patterns detected - manual analysis required"
fi

echo ""
echo "=========================================="
echo ""
echo "Processes are still running. To kill them:"
echo "  kill ${PARENT_PID}"
echo "  # or force kill:"
echo "  kill -9 ${PARENT_PID}"
echo ""
