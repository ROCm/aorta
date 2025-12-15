#!/bin/bash
# Monitor GPU usage across all nodes in multi-node training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_LIST="${SCRIPT_DIR}/node_ip_list.txt"

if [[ ! -f "$NODE_LIST" ]]; then
    echo "Error: node_ip_list.txt not found at $NODE_LIST"
    exit 1
fi

clear
echo "==================================================================="
echo "Multi-Node GPU Monitoring - 3 nodes x 8 GPUs = 24 GPUs total"
echo "==================================================================="
echo ""

NODE_NUM=0
for HOST in $(cat "$NODE_LIST"); do
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│ NODE $NODE_NUM: $HOST"
    echo "└─────────────────────────────────────────────────────────────┘"
    ssh "$HOST" "rocm-smi -u 2>/dev/null" | grep -E "GPU\[|GPU use" | sed 's/GPU\[/  GPU[/g'
    echo ""
    NODE_NUM=$((NODE_NUM + 1))
done

echo "==================================================================="
echo "Legend: Each node has 8 GPUs (GPU[0] through GPU[7])"
echo "  Node 0 GPUs → Ranks 0-7"
echo "  Node 1 GPUs → Ranks 8-15"
echo "  Node 2 GPUs → Ranks 16-23"
echo "==================================================================="
