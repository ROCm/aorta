#!/bin/bash
# Multi-node orchestration script for Aorta training
# Adapted from DLRM master_launch.sh pattern

MACHINE_IP_FILE="node_ip_list.txt"

if [[ -z "${MASTER_PORT:-}" ]]; then
  # Choose a free high port on the master host; fall back to 29500
  MASTER_PORT=$(python3 - <<'PY'
import socket
s=socket.socket()
s.bind(('',0))
print(s.getsockname()[1])
s.close()
PY
  ) || MASTER_PORT=29500
fi

# Configuration file path
CONFIG_FILE="${CONFIG_FILE:-config/multi_node/distributed_two_nodes.yaml}"

# Generate single timestamp for all nodes
TRACE_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create experiment directory with all subdirectories
EXPERIMENT_DIR="/apps/$USER/aorta/experiments/multinode_${TRACE_TIMESTAMP}"
mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR/logs"
mkdir -p "$EXPERIMENT_DIR/traces"

echo "=== Aorta Multi-Node Training ==="
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Config file: $CONFIG_FILE"

# Calculate world size (8 GPUs per node)
NUM_NODES=$(awk 'NF' "$MACHINE_IP_FILE" | wc -l)
WORLD_SIZE=$((8 * NUM_NODES))
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
  
      # Run on local master node
      ./scripts/multi_node/config_node.sh "$node" "$IP" "$MASTER_ADDR" "$MASTER_PORT" "$NNODES" "$WORLD_SIZE" "$PWD" "$EXPERIMENT_DIR" "$CONFIG_FILE" \
        > "$LOG_FILE" 2>&1 &
                                                                  
  else
      # SSH to remote nodes
      ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
          "$USER"@"$IP" bash -s -- "$node" "$IP" "$MASTER_ADDR" "$MASTER_PORT" "$NNODES" "$WORLD_SIZE" "$PWD" "$EXPERIMENT_DIR" "$CONFIG_FILE" \
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

# Wait for all background processes
wait

echo ""
echo "=== Training completed ==="
echo "Results saved to: $EXPERIMENT_DIR"


