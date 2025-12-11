#!/bin/bash
# Interactive setup script for multi-node training with 2 Conductor machines
# Run this on Machine 1 (master node)

set -e

echo "================================================"
echo "  Multi-Node Setup for 2 Conductor Machines"
echo "================================================"
echo ""
echo "Prerequisites:"
echo "  - You have reserved 2 machines on conductor.amd.com"
echo "  - You can SSH into both machines"
echo "  - You have the hostnames of both machines"
echo ""
read -p "Press Enter to continue..."
echo ""

# Get current machine info
CURRENT_HOST=$(hostname)
CURRENT_IP=$(hostname -I | awk '{print $1}')

echo "Current machine (Master Node):"
echo "  Hostname: $CURRENT_HOST"
echo "  IP: $CURRENT_IP"
echo ""

# Ask for worker machine
read -p "Enter the hostname of your second machine (worker): " WORKER_HOST

if [[ -z "$WORKER_HOST" ]]; then
    echo "Error: Worker hostname cannot be empty"
    exit 1
fi

# Test SSH to worker
echo ""
echo "Testing SSH connection to worker..."
if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$USER@$WORKER_HOST" "hostname" >/dev/null 2>&1; then
    echo "[OK] SSH to worker successful"
else
    echo "[FAIL] SSH to worker failed"
    echo ""
    echo "Fixes:"
    echo "  1. Ensure your SSH key is added to Conductor:"
    echo "     https://conductor.amd.com/docs/ssh_keys/"
    echo ""
    echo "  2. Generate and copy SSH key:"
    echo "     ssh-keygen -t rsa -b 4096"
    echo "     ssh-copy-id $USER@$WORKER_HOST"
    echo ""
    echo "  3. Check access permissions:"
    echo "     http://sut-auth.aus.dcgpu-infra.amd.com/api/v1/ssh_access/authorize?email=<YOUR_NTID>&hostname=$WORKER_HOST"
    exit 1
fi

# Get worker IP
WORKER_IP=$(ssh "$USER@$WORKER_HOST" "hostname -I | awk '{print \$1}'")
echo "  Worker Hostname: $WORKER_HOST"
echo "  Worker IP: $WORKER_IP"
echo ""

# Test connectivity
echo "Testing connectivity between machines..."
if ping -c 2 "$WORKER_IP" >/dev/null 2>&1; then
    echo "[OK] Ping to worker successful"
else
    echo "[WARN] Ping to worker failed"
    echo "  This might be okay if ICMP is blocked, but check network config"
fi
echo ""

# Test SSH from worker to master (for NFS/shared filesystem checks)
echo "Testing reverse SSH (worker to master)..."
if ssh "$USER@$WORKER_HOST" "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$CURRENT_HOST hostname" >/dev/null 2>&1; then
    echo "[OK] Reverse SSH successful"
else
    echo "[WARN] Reverse SSH failed - setting up passwordless SSH"
    echo "  Copying SSH key from worker to master..."
    
    # Generate key on worker if needed
    ssh "$USER@$WORKER_HOST" "test -f ~/.ssh/id_rsa || ssh-keygen -t rsa -b 4096 -N '' -f ~/.ssh/id_rsa" >/dev/null 2>&1
    
    # Copy worker's public key to master
    WORKER_PUBKEY=$(ssh "$USER@$WORKER_HOST" "cat ~/.ssh/id_rsa.pub")
    mkdir -p ~/.ssh
    echo "$WORKER_PUBKEY" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    
    # Test again
    if ssh "$USER@$WORKER_HOST" "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$CURRENT_HOST hostname" >/dev/null 2>&1; then
        echo "[OK] Reverse SSH now working"
    else
        echo "[FAIL] Reverse SSH still failing - manual setup needed"
    fi
fi
echo ""

# Check if code exists on worker
AORTA_PATH="/apps/$USER/aorta"
echo "Checking code availability on worker..."

if ssh "$USER@$WORKER_HOST" "test -d $AORTA_PATH" 2>/dev/null; then
    echo "[OK] Code found on worker: $AORTA_PATH"
    
    # Check if it's the same filesystem
    MASTER_INODE=$(stat -c %i "$AORTA_PATH" 2>/dev/null || echo "0")
    WORKER_INODE=$(ssh "$USER@$WORKER_HOST" "stat -c %i $AORTA_PATH" 2>/dev/null || echo "0")
    
    if [[ "$MASTER_INODE" == "$WORKER_INODE" ]] && [[ "$MASTER_INODE" != "0" ]]; then
        echo "[OK] Shared filesystem detected - code is automatically synced"
        SHARED_FS=true
    else
        echo "[WARN] Separate filesystems - you'll need to sync code manually"
        SHARED_FS=false
    fi
else
    echo "[FAIL] Code not found on worker"
    echo "  You'll need to clone or rsync the code to: $AORTA_PATH"
    SHARED_FS=false
fi
echo ""

# Check GPUs
echo "Checking GPUs..."
MASTER_GPUS=$(rocm-smi --showid 2>/dev/null | grep -c "GPU" || echo "unknown")
WORKER_GPUS=$(ssh "$USER@$WORKER_HOST" "rocm-smi --showid 2>/dev/null | grep -c GPU" || echo "unknown")

echo "  Master GPUs: $MASTER_GPUS"
echo "  Worker GPUs: $WORKER_GPUS"

if [[ "$MASTER_GPUS" != "$WORKER_GPUS" ]]; then
    echo "[WARN] GPU count mismatch"
    echo "  Use --nproc flag with master_launch.sh to specify GPU count"
fi
echo ""

# Detect network interface
echo "Detecting network interface..."
INTERFACE=$(ifconfig 2>/dev/null | grep -E "^(ib|enp|eth)" | head -1 | cut -d: -f1 || echo "unknown")
if [[ "$INTERFACE" == "unknown" ]]; then
    INTERFACE=$(ip addr show 2>/dev/null | grep -E "^[0-9]+: (ib|enp|eth)" | head -1 | awk '{print $2}' | tr -d ':' || echo "eth0")
fi

echo "  Detected interface: $INTERFACE"

# Ask user to confirm or change
read -p "Use this interface for NCCL? (or enter different name) [$INTERFACE]: " USER_INTERFACE
if [[ -n "$USER_INTERFACE" ]]; then
    INTERFACE="$USER_INTERFACE"
fi
echo ""

# Create node_ip_list.txt
echo "Creating node_ip_list.txt..."
cd "$AORTA_PATH"

cat > node_ip_list.txt << EOF
$CURRENT_IP
$WORKER_IP
EOF

echo "[OK] Created node_ip_list.txt:"
cat node_ip_list.txt
echo ""

# Update network interface in set_env_variables.sh
echo "Updating network interface in set_env_variables.sh..."
if [[ -f "scripts/multi_node/set_env_variables.sh" ]]; then
    # Backup original
    cp scripts/multi_node/set_env_variables.sh scripts/multi_node/set_env_variables.sh.bak
    
    # Update interface
    sed -i "s/export NCCL_SOCKET_IFNAME=.*/export NCCL_SOCKET_IFNAME=$INTERFACE/" scripts/multi_node/set_env_variables.sh
    
    echo "[OK] Updated NCCL_SOCKET_IFNAME=$INTERFACE"
else
    echo "[WARN] set_env_variables.sh not found - manual configuration needed"
fi
echo ""

# Summary
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "Configuration Summary:"
echo "  Master: $CURRENT_HOST ($CURRENT_IP)"
echo "  Worker: $WORKER_HOST ($WORKER_IP)"
echo "  Network Interface: $INTERFACE"
echo "  Master GPUs: $MASTER_GPUS"
echo "  Worker GPUs: $WORKER_GPUS"
echo "  Shared Filesystem: ${SHARED_FS:-false}"
echo ""
echo "Node IP list created at:"
echo "  $AORTA_PATH/node_ip_list.txt"
echo ""

if [[ "${SHARED_FS:-false}" == "false" ]]; then
    echo "[IMPORTANT] Sync code to worker before running:"
    echo "  ssh $WORKER_HOST 'cd /apps/$USER && git clone <repo> aorta'"
    echo "  OR"
    echo "  rsync -avz $AORTA_PATH/ $WORKER_HOST:$AORTA_PATH/"
    echo ""
fi

echo "Ready to launch training!"
echo ""
echo "Basic usage:"
echo "  cd $AORTA_PATH"
echo "  ./scripts/multi_node/master_launch.sh"
echo ""
echo "With custom settings:"
echo "  ./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --nproc 8"
echo ""
echo "With custom config:"
echo "  ./scripts/multi_node/master_launch.sh --config config/distributed_two_nodes.yaml"
echo ""
echo "Monitor training:"
echo "  tail -f experiments/multinode_*/logs/node_*.txt"
echo ""
echo "Stop training:"
echo "  ssh $CURRENT_HOST 'pkill -9 -f train.py'"
echo "  ssh $WORKER_HOST 'pkill -9 -f train.py'"
echo ""
