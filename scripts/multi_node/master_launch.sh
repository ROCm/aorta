#!/bin/bash
# Multi-node orchestration script for Aorta GEMM training
# Adapted from DLRM master_launch.sh pattern

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -c, --channels CHANNELS     NCCL_MAX_NCHANNELS value (default: 28)"
    echo "  -t, --threads THREADS       RCCL_THREADS_PER_BLOCK value (default: 256)"
    echo "  -f, --config CONFIG         Config file path (default: config/multi_node/distributed_multinode.yaml)"
    echo "  -p, --nproc NPROC           Number of processes per node (default: 8)"
    echo "  -r, --rocprof               Enable rocprofv3 tracing"
    echo "  -m, --stats                 Enable rocprof stats (CU utilization, occupancy)"
    echo "      --rocprof-input FILE    Use rocprofv3 input yaml/json"
    echo "      --master-port PORT      Master port (default: auto-select)"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --channels 28 --threads 256"
    echo "  $0 -c 28 -t 256 --rocprof"
    echo "  $0 --channels 28 --config config/my_custom.yaml"
    echo ""
    echo "Or use environment variables:"
    echo "  CHANNELS=28 THREADS=256 $0"
    exit 1
}

MACHINE_IP_FILE="node_ip_list.txt"

# Default values (can be overridden by env vars or command-line args)
CONFIG_FILE="${CONFIG_FILE:-config/multi_node/distributed_multinode.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CHANNELS="${CHANNELS:-28}"
THREADS="${THREADS:-256}"
ENABLE_ROCPROF="${ENABLE_ROCPROF:-false}"
ROCPROF_STATS="${ROCPROF_STATS:-false}"
ROCPROF_INPUT="${ROCPROF_INPUT:-}"
MASTER_PORT="${MASTER_PORT:-}"

# Parse command-line arguments (override env vars)
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--channels)
            CHANNELS="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -f|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -p|--nproc)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        -r|--rocprof)
            ENABLE_ROCPROF="true"
            shift
            ;;
        -m|--stats)
            ROCPROF_STATS="true"
            shift
            ;;
        --rocprof-input)
            ROCPROF_INPUT="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
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


if [[ -z "$MASTER_PORT" ]]; then
  if ! MASTER_PORT=$(python3 - <<'PY'
import socket
s=socket.socket()
s.bind(('',0))
print(s.getsockname()[1])
s.close()
PY
  ); then
    echo "Error: Failed to auto-select master port. Set MASTER_PORT manually."
    exit 1
  fi
fi

TRACE_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="/apps/$USER/aorta/experiments/multinode_${CHANNELS}ch_${THREADS}th_${TRACE_TIMESTAMP}"
mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR/logs"

echo "=== Aorta Multi-Node GEMM Training ==="
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Config file: $CONFIG_FILE"
echo "NCCL Channels: $CHANNELS"
echo "RCCL Threads per block: $THREADS"
echo "Processes per node: $NPROC_PER_NODE"
echo "rocprof enabled: $ENABLE_ROCPROF"

NUM_NODES=$(awk 'NF' "$MACHINE_IP_FILE" | wc -l)
WORLD_SIZE=$((NPROC_PER_NODE * NUM_NODES))
NNODES=$NUM_NODES

echo "Number of nodes: $NUM_NODES"
echo "World size: $WORLD_SIZE (GPUs)"
echo "Using MASTER_PORT: $MASTER_PORT"
echo ""

node=0
while IFS= read -r IP || [[ -n "$IP" ]]; do
  if [[ -z "$IP" ]]; then
    continue
  fi

  echo "Setting up Node: $node, IP: $IP"

  TIME=$(date +"%Y%m%d_%H%M%S")
  LOG_FILE="$EXPERIMENT_DIR/logs/node_${node}_${TIME}.txt"

  if [[ "$node" -eq 0 ]]; then
      MASTER_ADDR="$IP"
      echo "Master node: $MASTER_ADDR"
      echo ""
  
      ./scripts/multi_node/config_node.sh "$node" "$IP" "$MASTER_ADDR" "$MASTER_PORT" "$NNODES" "$WORLD_SIZE" "$PWD" "$EXPERIMENT_DIR" \
        "$CONFIG_FILE" "$NPROC_PER_NODE" "$CHANNELS" "$THREADS" "$ENABLE_ROCPROF" "$ROCPROF_STATS" "$ROCPROF_INPUT" \
        > "$LOG_FILE" 2>&1 &
                                                                  
  else
      ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
          "$USER"@"$IP" bash -s -- "$node" "$IP" "$MASTER_ADDR" "$MASTER_PORT" "$NNODES" "$WORLD_SIZE" "$PWD" "$EXPERIMENT_DIR" \
          "$CONFIG_FILE" "$NPROC_PER_NODE" "$CHANNELS" "$THREADS" "$ENABLE_ROCPROF" "$ROCPROF_STATS" "$ROCPROF_INPUT" \
        < ./scripts/multi_node/config_node.sh \
        > "$LOG_FILE" 2>&1 &    
  fi

  ((node++))

done < "$MACHINE_IP_FILE"

echo ""
echo "=== All nodes launched ==="
echo "Monitor logs in: $EXPERIMENT_DIR/logs/"
echo ""
echo "To monitor progress:"
echo "  tail -f $EXPERIMENT_DIR/logs/node_0_*.txt"
echo ""
echo "To check all nodes:"
echo "  tail -f $EXPERIMENT_DIR/logs/node_*.txt"
echo ""
echo "Waiting for training to complete..."
echo "Press Ctrl+C to stop monitoring (training will continue in background)"

wait

echo ""
echo "=== Training completed ==="
echo "Results saved to: $EXPERIMENT_DIR"



