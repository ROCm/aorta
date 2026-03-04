#!/bin/bash
# Multi-node launch script for the minimal RCCL race condition reproducer
#
# Usage:
#   ./scripts/multi_node/launch_reproducer.sh \
#       --docker training-overlap-bugs-rocm70_9-1-shampoo \
#       --hw-queues 4 \
#       --warmup 100 \
#       --verify 10000
#
# This script launches the minimal reproducer on all nodes defined in
# scripts/multi_node/node_ip_list.txt

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Launch the minimal RCCL race condition reproducer on multi-node setup."
    echo ""
    echo "Basic Options:"
    echo "  -d, --docker CONTAINER    Docker container name (required)"
    echo "  -m, --mode MODE           Reproducer mode: default, ddp, fsdp (default: default)"
    echo "      --fsdp-shard-size N   FSDP shard size per rank (default: 100000)"
    echo "  -p, --nproc NPROC         Processes per node (default: 8)"
    echo "  -w, --warmup N            Warmup iterations (default: 100)"
    echo "  -v, --verify N            Verification iterations (default: 10000)"
    echo "      --prefetch             Use double-buffered H2D prefetch"
    echo "  -s, --same-stream         Use same stream for H2D and datadist"
    echo "  -c, --no-compute          Disable compute simulation"
    echo "      --deterministic       Enable deterministic mode (for DDP gradient verification)"
    echo "      --bucketed            Use bucketed per-layer gradient all_reduce (DDP mode)"
    echo "      --optimizer OPT       Optimizer: none, adamw, sgd, shampoo (default: none)"
    echo "  -l, --label LABEL         Experiment label"
    echo "      --master-port PORT    Master port (default: auto-select)"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Tested Environment Variables:"
    echo "  -q, --hw-queues N         GPU_MAX_HW_QUEUES (4=expose bug, 2=mask)"
    echo "      --signal-pool N       ROC_SIGNAL_POOL_SIZE (tried 16384 - still NaN)"
    echo "      --disable-sdma        HSA_ENABLE_SDMA=0 (tried - still NaN)"
    echo "      --blit-copy N         GPU_FORCE_BLIT_COPY_SIZE (tried 128 - still NaN)"
    echo "      --nccl-implicit       NCCL_LAUNCH_ORDER_IMPLICIT=1 (no NaN but SLOW)"
    echo "      --disable-cheap-fence RCCL_GFX9_CHEAP_FENCE_OFF=1 (tried - still NaN)"
    echo "      --disable-clr-batch   DEBUG_CLR_BATCH_CPU_SYNC_SIZE=0 (tried - still NaN)"
    echo ""
    echo "Examples:"
    echo "  # Default mode (TorchRec-like) - most common test"
    echo "  $0 --docker training-overlap-bugs-rocm70_9-1-shampoo --hw-queues 4"
    echo ""
    echo "  # DDP mode (gradient all_reduce + H2D prefetch)"
    echo "  $0 --docker training-overlap-bugs-rocm70_9-1-shampoo --mode ddp --deterministic"
    echo ""
    echo "  # FSDP mode (per-layer all_gather + reduce_scatter)"
    echo "  $0 --docker training-overlap-bugs-rocm70_9-1-shampoo --mode fsdp --hw-queues 4"
    echo ""
    echo "  # Test with settings that MASK the bug (comparison)"
    echo "  $0 --docker training-overlap-bugs-rocm70_9-1-shampoo --hw-queues 2"
    echo ""
    echo "  # Same-stream mode (definitive runtime bug test)"
    echo "  $0 --docker training-overlap-bugs-rocm70_9-1-shampoo --hw-queues 4 --same-stream"
    echo ""
    echo "  # Test NCCL_LAUNCH_ORDER_IMPLICIT workaround"
    echo "  $0 --docker training-overlap-bugs-rocm70_9-1-shampoo --hw-queues 4 --nccl-implicit"
    echo ""
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AORTA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MACHINE_IP_FILE="$SCRIPT_DIR/node_ip_list.txt"

# Source NCCL/RCCL environment variables for multi-node
if [[ -f "$SCRIPT_DIR/set_env_variables.sh" ]]; then
    source "$SCRIPT_DIR/set_env_variables.sh"
fi

# Default values
DOCKER_CONTAINER=""
MODE="default"
NPROC_PER_NODE=8
HW_QUEUES=4
WARMUP=100
VERIFY=10000
SAME_STREAM=""
NO_COMPUTE=""
PREFETCH=""
DETERMINISTIC=""
BUCKETED=""
OPTIMIZER=""
FSDP_SHARD_SIZE=""
LABEL=""
MASTER_PORT=""

# Tested env var flags
SIGNAL_POOL=""
DISABLE_SDMA=""
BLIT_COPY=""
NCCL_IMPLICIT=""
DISABLE_CHEAP_FENCE=""
DISABLE_CLR_BATCH=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--docker)
            DOCKER_CONTAINER="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        --deterministic)
            DETERMINISTIC="--deterministic"
            shift
            ;;
        --bucketed)
            BUCKETED="--bucketed"
            shift
            ;;
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --fsdp-shard-size)
            FSDP_SHARD_SIZE="$2"
            shift 2
            ;;
        -p|--nproc)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        -q|--hw-queues)
            HW_QUEUES="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP="$2"
            shift 2
            ;;
        -v|--verify)
            VERIFY="$2"
            shift 2
            ;;
        --prefetch)
            PREFETCH="--prefetch"
            shift
            ;;
        -s|--same-stream)
            SAME_STREAM="--same-stream"
            shift
            ;;
        -c|--no-compute)
            NO_COMPUTE="--no-compute"
            shift
            ;;
        -l|--label)
            LABEL="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        # Tested env var flags
        --signal-pool)
            SIGNAL_POOL="$2"
            shift 2
            ;;
        --disable-sdma)
            DISABLE_SDMA="--disable-sdma"
            shift
            ;;
        --blit-copy)
            BLIT_COPY="$2"
            shift 2
            ;;
        --nccl-implicit)
            NCCL_IMPLICIT="--nccl-implicit-order"
            shift
            ;;
        --disable-cheap-fence)
            DISABLE_CHEAP_FENCE="--disable-cheap-fence"
            shift
            ;;
        --disable-clr-batch)
            DISABLE_CLR_BATCH="--disable-clr-batch"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DOCKER_CONTAINER" ]]; then
    echo "ERROR: --docker is required"
    usage
fi

# Auto-select master port if not specified
if [[ -z "$MASTER_PORT" ]]; then
    if ! MASTER_PORT=$(python3 - <<'PY'
import socket
s=socket.socket()
s.bind(('',0))
print(s.getsockname()[1])
s.close()
PY
    ); then
        echo "Error: Failed to auto-select master port. Set --master-port manually."
        exit 1
    fi
fi

# Count nodes and calculate world size
NUM_NODES=$(awk 'NF' "$MACHINE_IP_FILE" | wc -l)
WORLD_SIZE=$((NPROC_PER_NODE * NUM_NODES))

# Create experiment directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="$AORTA_ROOT/experiments/reproducer_${MODE}_hw${HW_QUEUES}_${TIMESTAMP}${LABEL:+_$LABEL}"
mkdir -p "$EXPERIMENT_DIR/logs"

# Build extra flags for Python CLI
EXTRA_FLAGS=""
[[ -n "$SIGNAL_POOL" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --signal-pool-size $SIGNAL_POOL"
[[ -n "$DISABLE_SDMA" ]] && EXTRA_FLAGS="$EXTRA_FLAGS $DISABLE_SDMA"
[[ -n "$BLIT_COPY" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --blit-copy-size $BLIT_COPY"
[[ -n "$NCCL_IMPLICIT" ]] && EXTRA_FLAGS="$EXTRA_FLAGS $NCCL_IMPLICIT"
[[ -n "$DISABLE_CHEAP_FENCE" ]] && EXTRA_FLAGS="$EXTRA_FLAGS $DISABLE_CHEAP_FENCE"
[[ -n "$DISABLE_CLR_BATCH" ]] && EXTRA_FLAGS="$EXTRA_FLAGS $DISABLE_CLR_BATCH"

# Save experiment info
cat > "$EXPERIMENT_DIR/experiment_info.txt" << EOF
Experiment: Minimal RCCL Race Condition Reproducer
Timestamp: $TIMESTAMP
Label: ${LABEL:-unlabeled}
Mode: $MODE
Docker: $DOCKER_CONTAINER
GPU_MAX_HW_QUEUES: $HW_QUEUES
Warmup iterations: $WARMUP
Verify iterations: $VERIFY
H2D prefetch: ${PREFETCH:-no}
Same stream mode: ${SAME_STREAM:-no}
Compute simulation: ${NO_COMPUTE:-enabled}
Deterministic: ${DETERMINISTIC:-no}
Bucketed: ${BUCKETED:-no}
Optimizer: ${OPTIMIZER:-none}
Signal pool size: ${SIGNAL_POOL:-default}
Disable SDMA: ${DISABLE_SDMA:-no}
Blit copy size: ${BLIT_COPY:-default}
NCCL implicit order: ${NCCL_IMPLICIT:-no}
Disable cheap fence: ${DISABLE_CHEAP_FENCE:-no}
Disable CLR batch: ${DISABLE_CLR_BATCH:-no}
Nodes: $NUM_NODES
World size: $WORLD_SIZE
Git: $(git rev-parse --short HEAD 2>/dev/null || echo "N/A")
EOF

# Save config files to experiment directory
CONFIG_DIR="$EXPERIMENT_DIR/config"
mkdir -p "$CONFIG_DIR"

# Copy the main reproducer config (reference)
if [[ -f "$AORTA_ROOT/config/race/minimal_reproducer.yaml" ]]; then
    cp "$AORTA_ROOT/config/race/minimal_reproducer.yaml" "$CONFIG_DIR/"
fi

# Save the ACTUAL config used for this test (CLI args override defaults)
cat > "$CONFIG_DIR/run_config.yaml" << EOF
# Actual configuration used for this test run
# Generated by launch_reproducer.sh on $(date)
#
# Note: CLI arguments override the defaults in minimal_reproducer.yaml

# Iteration settings
warmup_iterations: $WARMUP
verify_iterations: $VERIFY
stop_on_first_corruption: true
log_interval: 100

# Compute simulation
simulate_compute: $([ -z "$NO_COMPUTE" ] && echo "true" || echo "false")
gemm_size: 5120
gemm_layers: 26
include_backward_compute: true

# H2D buffering
h2d_prefetch: $([ -z "$PREFETCH" ] && echo "false" || echo "true")

# Stream configuration
same_stream_mode: $([ -z "$SAME_STREAM" ] && echo "false" || echo "true")

# Hardware settings
gpu_max_hw_queues: $HW_QUEUES

# Environment variables
env_vars:
  GPU_MAX_HW_QUEUES: $HW_QUEUES
  ROC_SIGNAL_POOL_SIZE: ${SIGNAL_POOL:-default}
  HSA_ENABLE_SDMA: ${DISABLE_SDMA:+0}${DISABLE_SDMA:-default}
  GPU_FORCE_BLIT_COPY_SIZE: ${BLIT_COPY:-default}
  NCCL_LAUNCH_ORDER_IMPLICIT: ${NCCL_IMPLICIT:+1}${NCCL_IMPLICIT:-default}
  RCCL_GFX9_CHEAP_FENCE_OFF: ${DISABLE_CHEAP_FENCE:+1}${DISABLE_CHEAP_FENCE:-default}
  DEBUG_CLR_BATCH_CPU_SYNC_SIZE: ${DISABLE_CLR_BATCH:+0}${DISABLE_CLR_BATCH:-default}

# Cluster settings
nodes: $NUM_NODES
world_size: $WORLD_SIZE
nproc_per_node: $NPROC_PER_NODE
EOF

# Copy NCCL/RCCL environment settings for reference
if [[ -f "$SCRIPT_DIR/set_env_variables.sh" ]]; then
    cp "$SCRIPT_DIR/set_env_variables.sh" "$CONFIG_DIR/"
fi

echo "Config saved to: $CONFIG_DIR/run_config.yaml"

echo "=========================================="
echo "Minimal RCCL Race Condition Reproducer"
echo "=========================================="
echo ""
echo "This reproducer uses PROPER SYNCHRONIZATION everywhere."
echo "If corruption occurs, it indicates a RUNTIME BUG in RCCL/HIP."
echo ""
echo "Configuration:"
echo "  Docker container: $DOCKER_CONTAINER"
echo "  Mode: $MODE"
echo "  Nodes: $NUM_NODES | World size: $WORLD_SIZE GPUs"
echo "  Warmup: $WARMUP | Verify: $VERIFY iterations"
echo "  H2D prefetch: ${PREFETCH:-no}"
echo "  Same stream mode: ${SAME_STREAM:-no}"
echo "  Compute simulation: ${NO_COMPUTE:-enabled}"
echo "  Deterministic: ${DETERMINISTIC:-no}"
echo "  Bucketed: ${BUCKETED:-no}"
echo "  Optimizer: ${OPTIMIZER:-none}"
echo ""
echo "Tested env vars:"
echo "  GPU_MAX_HW_QUEUES: $HW_QUEUES"
[[ -n "$SIGNAL_POOL" ]] && echo "  ROC_SIGNAL_POOL_SIZE: $SIGNAL_POOL"
[[ -n "$DISABLE_SDMA" ]] && echo "  HSA_ENABLE_SDMA: 0"
[[ -n "$BLIT_COPY" ]] && echo "  GPU_FORCE_BLIT_COPY_SIZE: $BLIT_COPY"
[[ -n "$NCCL_IMPLICIT" ]] && echo "  NCCL_LAUNCH_ORDER_IMPLICIT: 1"
[[ -n "$DISABLE_CHEAP_FENCE" ]] && echo "  RCCL_GFX9_CHEAP_FENCE_OFF: 1"
[[ -n "$DISABLE_CLR_BATCH" ]] && echo "  DEBUG_CLR_BATCH_CPU_SYNC_SIZE: 0"
echo ""
echo "Master port: $MASTER_PORT"
echo "Experiment directory: $EXPERIMENT_DIR"
echo "=========================================="
echo ""

# Build the reproducer command
REPRODUCER_CMD="python -m aorta.race --warmup $WARMUP --verify $VERIFY --hw-queues $HW_QUEUES $SAME_STREAM $NO_COMPUTE"

# Launch on each node
node=0
while IFS= read -r HOST || [[ -n "$HOST" ]]; do
    if [[ -z "$HOST" ]]; then
        continue
    fi

    echo "Launching on Node $node: $HOST"

    LOG_FILE="$EXPERIMENT_DIR/logs/node_${node}.txt"

    if [[ "$node" -eq 0 ]]; then
        MASTER_ADDR="$HOST"
        echo "Master node: $MASTER_ADDR"
    fi

    # Build torchrun command
    TORCHRUN_CMD="torchrun \
        --nnodes $NUM_NODES \
        --node_rank $node \
        --nproc_per_node $NPROC_PER_NODE \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        -m aorta.race \
        --mode $MODE \
        --warmup $WARMUP \
        --verify $VERIFY \
        --hw-queues $HW_QUEUES \
        $PREFETCH $SAME_STREAM $NO_COMPUTE $DETERMINISTIC $BUCKETED \
        ${OPTIMIZER:+--optimizer $OPTIMIZER} \
        ${FSDP_SHARD_SIZE:+--fsdp-shard-size $FSDP_SHARD_SIZE} \
        $EXTRA_FLAGS"

    # Build docker exec command with environment variables from set_env_variables.sh
    DOCKER_ENV_FLAGS=$(build_docker_env_flags)
    DOCKER_CMD="docker exec \
        -e GPU_MAX_HW_QUEUES=$HW_QUEUES \
        -e PYTHONPATH=/workspace/aorta/src \
        $DOCKER_ENV_FLAGS \
        $DOCKER_CONTAINER \
        bash -c 'cd /workspace/aorta && $TORCHRUN_CMD'"

    if [[ "$node" -eq 0 ]]; then
        # Master node - run locally
        eval "$DOCKER_CMD" > "$LOG_FILE" 2>&1 &
    else
        # Worker nodes - run via SSH
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            "$USER@$HOST" "$DOCKER_CMD" > "$LOG_FILE" 2>&1 &
    fi

    ((node++))
done < "$MACHINE_IP_FILE"

echo ""
echo "=========================================="
echo "All nodes launched"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f $EXPERIMENT_DIR/logs/node_0.txt"
echo ""
echo "Monitor all nodes:"
echo "  tail -f $EXPERIMENT_DIR/logs/node_*.txt"
echo ""
echo "Waiting for reproducer to complete..."
echo ""

wait

echo ""
echo "=========================================="
echo "Reproducer completed"
echo "=========================================="
echo ""
echo "Results saved to: $EXPERIMENT_DIR"
echo ""
echo "Check for corruption:"
echo "  grep -i 'CORRUPTION\|RUNTIME BUG' $EXPERIMENT_DIR/logs/*.txt"
echo ""
echo "Summary:"
echo "  grep -i 'VERDICT\|PASSED\|FAILED' $EXPERIMENT_DIR/logs/node_0.txt"
