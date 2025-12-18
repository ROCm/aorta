#!/bin/bash
# Multi-node local launch script for GEMM training
# Runs on each node with single channel/thread configuration

if [[ $# -lt 11 ]]; then
  echo "Usage: $0 <NODE_RANK> <NODE_IP> <MASTER_IP> <MASTER_PORT> <NNODES> <WORLD_SIZE> <EXPERIMENT_DIR> <CONFIG_FILE> <NPROC_PER_NODE> <CHANNELS> <THREADS> [ENABLE_ROCPROF] [ROCPROF_STATS] [ROCPROF_INPUT] [DOCKER_CONTAINER]"
  exit 1
fi

NODE_RANK="$1"
NODE_IP="$2"
MASTER_IP="$3"
MASTER_PORT="$4"
NNODES="$5"
WORLD_SIZE="$6"
EXPERIMENT_DIR="$7"
CONFIG_FILE="$8"
NPROC_PER_NODE="$9"
CHANNELS="${10}"
THREADS="${11}"
ENABLE_ROCPROF="${12:-false}"
ROCPROF_STATS="${13:-false}"
ROCPROF_INPUT="${14:-}"
DOCKER_CONTAINER="${15:-training-overlap-bugs-rocm70_9-1}"

echo "=========================================="
echo "Local Launch Configuration"
echo "=========================================="
echo "Node Rank: $NODE_RANK"
echo "Node IP: $NODE_IP"
echo "Master IP: $MASTER_IP"
echo "Master Port: $MASTER_PORT"
echo "Number of Nodes: $NNODES"
echo "World Size: $WORLD_SIZE"
echo "Processes per node: $NPROC_PER_NODE"
echo "Experiment Dir: $EXPERIMENT_DIR"
echo "Config File: $CONFIG_FILE"
echo "Channels: $CHANNELS"
echo "Threads: $THREADS"
echo "rocprof enabled: $ENABLE_ROCPROF"
echo "=========================================="
echo ""

# Output directory for this configuration
OUTPUT_DIR="${EXPERIMENT_DIR}/${THREADS}thread_${CHANNELS}channels"
mkdir -p "${OUTPUT_DIR}"

# Convert host path to Docker path for use inside container
# Docker mounts host aorta directory -> /workspace/aorta
# Extract aorta root from EXPERIMENT_DIR (e.g., /home/user/aorta/experiments/... -> /home/user/aorta)
AORTA_ROOT_FROM_EXP=$(echo "$EXPERIMENT_DIR" | sed 's|/experiments/.*||')
# Replace the aorta root with /workspace/aorta
OUTPUT_DIR_DOCKER=$(echo "$OUTPUT_DIR" | sed "s|^${AORTA_ROOT_FROM_EXP}|/workspace/aorta|")

# Also convert CONFIG_FILE to Docker path if it's an absolute path
if [[ "$CONFIG_FILE" =~ ^/ ]]; then
    CONFIG_FILE_DOCKER=$(echo "$CONFIG_FILE" | sed "s|^${AORTA_ROOT_FROM_EXP}|/workspace/aorta|")
else
    CONFIG_FILE_DOCKER="$CONFIG_FILE"
fi

# Function to log with timestamp
log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [Node ${NODE_RANK}] ${message}"
}

# Cleanup function
cleanup() {
    echo ""
    echo "=== Caught interrupt signal ==="
    log "Cleaning up training processes on node ${NODE_RANK}..."

    # Try to kill processes inside Docker container
    docker exec training-overlap-bugs-rocm70_9-1 pkill -9 -f "train.py" 2>/dev/null || true
    docker exec training-overlap-bugs-rocm70_9-1 pkill -9 -f "torchrun" 2>/dev/null || true

    # Also try on host (in case anything leaked)
    sudo pkill -9 -f "train.py" 2>/dev/null || true
    sudo pkill -9 -f "torchrun" 2>/dev/null || true

    log "Cleanup complete. Exiting."
    exit 130
}

trap cleanup SIGINT SIGTERM

log "Starting multi-node training with RCCL_THREADS_PER_BLOCK=${THREADS}, NCCL_MAX_NCHANNELS=${CHANNELS}"
log "Output directory: ${OUTPUT_DIR}"

START_TIME=$(date +%s)

# Check if Docker container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER}$"; then
    log "ERROR: Docker container '${DOCKER_CONTAINER}' is not running"
    log "Start it with: cd /path/to/aorta/docker && docker compose -f docker-compose.rocm70_9-1.yaml up -d"
    exit 1
fi

log "Docker container '${DOCKER_CONTAINER}' is running"

# Base command for torchrun with multi-node parameters
BASE_CMD="torchrun --nnodes ${NNODES} --node_rank ${NODE_RANK} --nproc_per_node ${NPROC_PER_NODE} --master_addr ${MASTER_IP} --master_port ${MASTER_PORT} train.py --config ${CONFIG_FILE_DOCKER}"
BASE_OVERRIDES="--override profiling.tensorboard=false"

# Build docker exec prefix with environment variables
# Pass through all NCCL/RCCL configuration from set_env_variables.sh
DOCKER_EXEC="docker exec \
    -e RCCL_THREADS_PER_BLOCK=${THREADS} \
    -e NCCL_MAX_NCHANNELS=${CHANNELS} \
    -e HSA_ENABLE_SDMA=0 \
    -e PYTORCH_ROCM_PROFILER_ENABLE_TRACING=1 \
    -e NCCL_DEBUG=${NCCL_DEBUG:-} \
    -e NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-} \
    -e NCCL_IB_HCA=${NCCL_IB_HCA:-} \
    -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} \
    -e NCCL_NCHANNELS_PER_NET_PEER=${NCCL_NCHANNELS_PER_NET_PEER:-8} \
    -e HSA_ENABLE_IPC_MODE_LEGACY=${HSA_ENABLE_IPC_MODE_LEGACY:-1} \
    -e NCCL_PROTO=${NCCL_PROTO:-Simple} \
    -e NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-40} \
    -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-} \
    -e NCCL_TIMEOUT_MS=${NCCL_TIMEOUT_MS:-60000} \
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1} \
    -e TORCH_NCCL_TRACE_BUFFER_SIZE=10000 \
    -e TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
    -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120 \
    -e TORCH_NCCL_BLOCKING_WAIT=0 \
    ${DOCKER_CONTAINER}"

# Run with or without rocprofv3
if [ "${ENABLE_ROCPROF}" = "true" ]; then
    ROCPROF_DIR="${OUTPUT_DIR}/rocprof_traces/node_${NODE_RANK}"
    mkdir -p "${ROCPROF_DIR}"

    if [ -n "${ROCPROF_INPUT}" ]; then
        log "Using rocprofv3 input file: ${ROCPROF_INPUT}"
        ${DOCKER_EXEC} bash -c "rocprofv3 -i ${ROCPROF_INPUT} -d ${ROCPROF_DIR} -- \
            ${BASE_CMD} ${BASE_OVERRIDES} \
            --override training.output_dir=${OUTPUT_DIR_DOCKER}" \
            2>&1 | tee -a "${LOG_FILE}"
    else
        ROCPROF_ARGS="--kernel-trace"
        if [ "${ROCPROF_STATS}" = "true" ]; then
            ROCPROF_ARGS="${ROCPROF_ARGS} --stats"
        fi

        log "Running with rocprofv3 kernel tracing inside Docker"
        ${DOCKER_EXEC} bash -c "rocprofv3 ${ROCPROF_ARGS} -d ${ROCPROF_DIR} -- \
            ${BASE_CMD} ${BASE_OVERRIDES} \
            --override training.output_dir=${OUTPUT_DIR_DOCKER}" \
            2>&1 | tee -a "${LOG_FILE}"
    fi
else
    log "Running inside Docker container"
    log "Command: ${BASE_CMD} ${BASE_OVERRIDES} --override training.output_dir=${OUTPUT_DIR_DOCKER}"
    ${DOCKER_EXEC} bash -c "${BASE_CMD} ${BASE_OVERRIDES} \
        --override training.output_dir=${OUTPUT_DIR_DOCKER}" \
        2>&1 | tee -a "${LOG_FILE}"
fi

EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $EXIT_CODE -eq 0 ]; then
    log "Training completed successfully (duration: ${DURATION}s)"
else
    log "Training failed with exit code: $EXIT_CODE (duration: ${DURATION}s)"
fi

echo ""
log "Node ${NODE_RANK} finished"
