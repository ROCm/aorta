#!/bin/bash
# Per-node configuration and launch script for Aorta
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

echo "============================================"
echo "Node Configuration"
echo "============================================"
echo "Node Rank: $NODE_RANK"
echo "Node IP: $NODE_IP"
echo "Master IP: $MASTER_IP"
echo "Master Port: $MASTER_PORT"
echo "Number of Nodes: $NNODES"
echo "World Size: $WORLD_SIZE GPUs"
echo "Work Directory: $WORKDIR"
echo "Experiment Directory: $EXPERIMENT_DIR"
echo "Config File: $CONFIG_FILE"
echo "============================================"
echo ""

# Change to working directory
cd "$WORKDIR" || exit 1

# Set up environment variables (customize for your system)
# Example: Conda/virtualenv activation
# source /path/to/your/venv/bin/activate
# or
# conda activate your_env

# Optional: Set NCCL/RCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=eth0  # Change to your network interface (e.g., ib0, eth0)

# ROCm-specific (if using AMD GPUs)
# export HSA_ENABLE_SDMA=0
# export RCCL_THREADS_PER_BLOCK=256

# CUDA-specific (if using NVIDIA GPUs)
# export NCCL_NSOCKS_PERTHREAD=4
# export NCCL_SOCKET_NTHREADS=2

echo "Environment configured. Starting training..."
echo ""

# Launch torchrun
# Note: torchrun will spawn 8 processes (one per GPU) on this node
torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node=8 \
  --master_addr="$MASTER_IP" \
  --master_port="$MASTER_PORT" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_IP}:${MASTER_PORT}" \
  train.py \
  --config "$CONFIG_FILE" \
  --override training.output_dir="$EXPERIMENT_DIR/outputs" \
  --override profiling.enabled=true \
  --override profiling.active=5

echo ""
echo "============================================"
echo "Node $NODE_RANK training completed"
echo "============================================"


