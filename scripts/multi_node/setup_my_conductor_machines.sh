#!/bin/bash
# Quick setup for oyazdanb's Conductor MI350 pool machines
# Creates node_ip_list.txt for multi-node training

set -e

AORTA_PATH="/apps/oyazdanb/aorta"

echo "================================================"
echo "  Conductor MI350 Pool Multi-Node Setup"
echo "================================================"
echo ""
echo "Available machines:"
echo "  1. b10  - smci350-zts-gtu-b10-15.zts-gtu.dcgpu"
echo "  2. f8   - smci350-zts-gtu-f8-25.zts-gtu.dcgpu"
echo "  3. b20  - smci350-zts-gtu-b20-05.zts-gtu.dcgpu"
echo "  4. c6   - smci350-zts-gtu-c6-25.zts-gtu.dcgpu"
echo ""

# Default: b10 as master, f8 as worker
read -p "Enter master machine [b10]: " MASTER_INPUT
MASTER_INPUT=${MASTER_INPUT:-b10}

read -p "Enter worker machine [f8]: " WORKER_INPUT
WORKER_INPUT=${WORKER_INPUT:-f8}

# Map short names to hostnames
case $MASTER_INPUT in
    b10) MASTER_HOST="smci350-zts-gtu-b10-15.zts-gtu.dcgpu" ;;
    f8)  MASTER_HOST="smci350-zts-gtu-f8-25.zts-gtu.dcgpu" ;;
    b20) MASTER_HOST="smci350-zts-gtu-b20-05.zts-gtu.dcgpu" ;;
    c6)  MASTER_HOST="smci350-zts-gtu-c6-25.zts-gtu.dcgpu" ;;
    *)   MASTER_HOST="$MASTER_INPUT" ;;
esac

case $WORKER_INPUT in
    b10) WORKER_HOST="smci350-zts-gtu-b10-15.zts-gtu.dcgpu" ;;
    f8)  WORKER_HOST="smci350-zts-gtu-f8-25.zts-gtu.dcgpu" ;;
    b20) WORKER_HOST="smci350-zts-gtu-b20-05.zts-gtu.dcgpu" ;;
    c6)  WORKER_HOST="smci350-zts-gtu-c6-25.zts-gtu.dcgpu" ;;
    *)   WORKER_HOST="$WORKER_INPUT" ;;
esac

echo ""
echo "Selected configuration:"
echo "  Master: $MASTER_HOST"
echo "  Worker: $WORKER_HOST"
echo ""

# Check if we're on the master machine
CURRENT_HOST=$(hostname)
if [[ "$CURRENT_HOST" != "$MASTER_HOST" ]]; then
    echo "[WARN] Current host ($CURRENT_HOST) is not master ($MASTER_HOST)"
    echo "       Run this script on the master machine"
    echo ""
    echo "SSH to master and run:"
    echo "  ssh $MASTER_HOST"
    echo "  cd $AORTA_PATH"
    echo "  ./scripts/multi_node/setup_my_conductor_machines.sh"
    exit 1
fi

# Test SSH to worker
echo "Testing SSH to worker..."
if ! ssh -o ConnectTimeout=5 "$WORKER_HOST" "hostname" >/dev/null 2>&1; then
    echo "[FAIL] Cannot SSH to worker: $WORKER_HOST"
    echo ""
    echo "Setup passwordless SSH:"
    echo "  ssh-keygen -t rsa -b 4096"
    echo "  ssh-copy-id oyazdanb@$WORKER_HOST"
    exit 1
fi
echo "[OK] SSH to worker successful"
echo ""

# Get IPs
echo "Retrieving IP addresses..."
MASTER_IP=$(hostname -I | awk '{print $1}')
WORKER_IP=$(ssh "$WORKER_HOST" "hostname -I | awk '{print \$1}'")

echo "  Master IP: $MASTER_IP"
echo "  Worker IP: $WORKER_IP"
echo ""

# Create node_ip_list.txt
cd "$AORTA_PATH"

cat > node_ip_list.txt << EOF
$MASTER_IP
$WORKER_IP
EOF

echo "[OK] Created node_ip_list.txt:"
cat node_ip_list.txt
echo ""

# Detect network interface
echo "Detecting network interface..."
INTERFACE=$(ifconfig 2>/dev/null | grep -E "^(ib|enp|eth)" | head -1 | cut -d: -f1 || echo "unknown")
if [[ "$INTERFACE" == "unknown" ]]; then
    INTERFACE=$(ip addr show 2>/dev/null | grep -E "^[0-9]+: (ib|enp|eth)" | head -1 | awk '{print $2}' | tr -d ':' || echo "eth0")
fi

echo "  Detected: $INTERFACE"
echo ""

# Check current interface in set_env_variables.sh
if [[ -f "scripts/multi_node/set_env_variables.sh" ]]; then
    CURRENT_INTERFACE=$(grep "export NCCL_SOCKET_IFNAME=" scripts/multi_node/set_env_variables.sh | cut -d= -f2 | tr -d '"' | tr -d "'")
    echo "Current interface in set_env_variables.sh: $CURRENT_INTERFACE"
    
    if [[ "$CURRENT_INTERFACE" != "$INTERFACE" ]]; then
        read -p "Update to $INTERFACE? [Y/n]: " UPDATE_INTERFACE
        if [[ ! "$UPDATE_INTERFACE" =~ ^[Nn]$ ]]; then
            cp scripts/multi_node/set_env_variables.sh scripts/multi_node/set_env_variables.sh.bak
            sed -i "s/export NCCL_SOCKET_IFNAME=.*/export NCCL_SOCKET_IFNAME=$INTERFACE/" scripts/multi_node/set_env_variables.sh
            echo "[OK] Updated NCCL_SOCKET_IFNAME=$INTERFACE"
        fi
    else
        echo "[OK] Interface already configured correctly"
    fi
fi
echo ""

# Check GPUs
MASTER_GPUS=$(rocm-smi --showid 2>/dev/null | grep -c "GPU" || echo "unknown")
WORKER_GPUS=$(ssh "$WORKER_HOST" "rocm-smi --showid 2>/dev/null | grep -c GPU" || echo "unknown")

echo "GPU count:"
echo "  Master: $MASTER_GPUS"
echo "  Worker: $WORKER_GPUS"
echo ""

echo "================================================"
echo "  Setup Complete"
echo "================================================"
echo ""
echo "Launch training:"
echo "  ./scripts/multi_node/master_launch.sh"
echo ""
echo "With custom settings:"
echo "  ./scripts/multi_node/master_launch.sh --channels 28 --threads 256"
echo ""
echo "Monitor:"
echo "  tail -f experiments/multinode_*/logs/node_*.txt"
echo ""
