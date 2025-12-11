#!/bin/bash
# Per-node configuration and launch script for Aorta GEMM training
# This script runs on each node (via SSH or locally)

NODE_RANK=$(echo "$1" | sed 's/"//g')
NODE_IP=$(echo "$2" | sed 's/"//g')
MASTER_IP=$(echo "$3" | sed 's/"//g')
MASTER_PORT=$(echo "$4" | sed 's/"//g')
NNODES=$(echo "$5" | sed 's/"//g')
WORLD_SIZE=$(echo "$6" | sed 's/"//g')
WORKDIR=$(echo "$7" | sed 's/"//g')
EXPERIMENT_DIR=$(echo "$8" | sed 's/"//g')
CONFIG_FILE=$(echo "$9" | sed 's/"//g')
NPROC_PER_NODE=$(echo "${10}" | sed 's/"//g')
CHANNELS=$(echo "${11}" | sed 's/"//g')
THREADS=$(echo "${12}" | sed 's/"//g')
ENABLE_ROCPROF=$(echo "${13}" | sed 's/"//g')
ROCPROF_STATS=$(echo "${14}" | sed 's/"//g')
ROCPROF_INPUT=$(echo "${15}" | sed 's/"//g')

echo "============================================"
echo "Node Configuration"
echo "============================================"
echo "Node Rank: $NODE_RANK"
echo "Node IP: $NODE_IP"
echo "Master IP: $MASTER_IP"
echo "Master Port: $MASTER_PORT"
echo "Number of Nodes: $NNODES"
echo "World Size: $WORLD_SIZE GPUs"
echo "Processes per node: $NPROC_PER_NODE"
echo "Work Directory: $WORKDIR"
echo "Experiment Directory: $EXPERIMENT_DIR"
echo "Config File: $CONFIG_FILE"
echo "Channels: $CHANNELS"
echo "Threads: $THREADS"
echo "============================================"
echo ""

# Change to working directory
cd "$WORKDIR" || exit 1

# Activate virtual environment if it exists
VENV_PATH="$WORKDIR/.venv"
if [[ -d "$VENV_PATH" ]]; then
    echo "Activating virtual environment at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi

# Source common environment variables
if [[ -f "$WORKDIR/scripts/multi_node/set_env_variables.sh" ]]; then
    echo "Sourcing set_env_variables.sh"
    source "$WORKDIR/scripts/multi_node/set_env_variables.sh"
else
    echo "Warning: set_env_variables.sh not found, using default NCCL settings"
    export NCCL_DEBUG=WARN
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME=eth0
fi

echo ""
echo "Environment configured. Starting GEMM training..."
echo ""

# Launch local_launch.sh with all parameters
"$WORKDIR/scripts/multi_node/local_launch.sh" \
    "$NODE_RANK" "$NODE_IP" "$MASTER_IP" "$MASTER_PORT" "$NNODES" "$WORLD_SIZE" \
    "$EXPERIMENT_DIR" "$CONFIG_FILE" "$NPROC_PER_NODE" "$CHANNELS" "$THREADS" \
    "$ENABLE_ROCPROF" "$ROCPROF_STATS" "$ROCPROF_INPUT"

echo ""
echo "============================================"
echo "Node $NODE_RANK training completed"
echo "============================================"



