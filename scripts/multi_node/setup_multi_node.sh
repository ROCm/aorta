#!/bin/bash
# Interactive setup script for multi-node training
# Run this on Machine 1 (master node)

set -e

echo "================================================"
echo "  Multi-Node Training Setup"
echo "================================================"
echo ""
echo "Prerequisites:"
echo "  - You have access to multiple machines"
echo "  - You can SSH into all machines from the master node"
echo "  - You have the hostnames/IPs of all machines"
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

# Ask for number of worker nodes
read -p "How many worker nodes? (default: 1): " NUM_WORKERS
NUM_WORKERS=${NUM_WORKERS:-1}

if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || [[ "$NUM_WORKERS" -lt 1 ]]; then
    echo "Error: Number of workers must be a positive integer"
    exit 1
fi

echo "Setting up 1 master + $NUM_WORKERS worker(s) = $((NUM_WORKERS + 1)) total nodes"
echo ""

# Collect worker hostnames
WORKER_HOSTS=()
for i in $(seq 1 $NUM_WORKERS); do
    read -p "Enter hostname for worker $i: " WORKER_HOST

    if [[ -z "$WORKER_HOST" ]]; then
        echo "Error: Worker hostname cannot be empty"
        exit 1
    fi

    WORKER_HOSTS+=("$WORKER_HOST")
done
echo ""

# Test SSH to all workers and collect IPs
echo "Testing SSH connections to workers..."
WORKER_IPS=()
for i in "${!WORKER_HOSTS[@]}"; do
    WORKER_HOST="${WORKER_HOSTS[$i]}"
    WORKER_NUM=$((i + 1))

    echo "Worker $WORKER_NUM: $WORKER_HOST"

    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$USER@$WORKER_HOST" "hostname" >/dev/null 2>&1; then
        echo "  [OK] SSH successful"
        WORKER_IP=$(ssh "$USER@$WORKER_HOST" "hostname -I | awk '{print \$1}'")
        echo "  [OK] IP: $WORKER_IP"
        WORKER_IPS+=("$WORKER_IP")
    else
        echo "  [FAIL] SSH failed"
        echo ""
        echo "Fixes:"
        echo "  1. Ensure your SSH key is registered with your cluster management system"
        echo "  2. Generate and copy SSH key:"
        echo "     ssh-keygen -t rsa -b 4096 -C 'multi-node' -f ~/.ssh/id_rsa_cluster -N ''"
        echo "     # Then register the public key (~/.ssh/id_rsa_cluster.pub) with your cluster"
        echo "  3. Or use ssh-copy-id if direct access is available:"
        echo "     ssh-copy-id -i ~/.ssh/id_rsa_cluster.pub $USER@$WORKER_HOST"
        exit 1
    fi
    echo ""
done

# Test connectivity and reverse SSH
echo "Testing connectivity and reverse SSH..."
for i in "${!WORKER_HOSTS[@]}"; do
    WORKER_HOST="${WORKER_HOSTS[$i]}"
    WORKER_IP="${WORKER_IPS[$i]}"
    WORKER_NUM=$((i + 1))

    echo "Worker $WORKER_NUM ($WORKER_HOST):"

    # Test ping
    if ping -c 2 "$WORKER_IP" >/dev/null 2>&1; then
        echo "  [OK] Ping successful"
    else
        echo "  [WARN] Ping failed (might be okay if ICMP is blocked)"
    fi

    # Test reverse SSH
    if ssh "$USER@$WORKER_HOST" "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$CURRENT_HOST hostname" >/dev/null 2>&1; then
        echo "  [OK] Reverse SSH successful"
    else
        echo "  [WARN] Reverse SSH failed - setting up passwordless SSH"

        # Generate key on worker if needed
        ssh "$USER@$WORKER_HOST" "test -f ~/.ssh/id_rsa || ssh-keygen -t rsa -b 4096 -N '' -f ~/.ssh/id_rsa" >/dev/null 2>&1

        # Copy worker's public key to master
        WORKER_PUBKEY=$(ssh "$USER@$WORKER_HOST" "cat ~/.ssh/id_rsa.pub")
        mkdir -p ~/.ssh
        echo "$WORKER_PUBKEY" >> ~/.ssh/authorized_keys
        chmod 600 ~/.ssh/authorized_keys

        # Test again
        if ssh "$USER@$WORKER_HOST" "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$CURRENT_HOST hostname" >/dev/null 2>&1; then
            echo "  [OK] Reverse SSH now working"
        else
            echo "  [FAIL] Reverse SSH still failing - manual setup needed"
        fi
    fi
    echo ""
done

# Check if code exists on workers
AORTA_PATH="$HOME/aorta"
echo "Checking code availability on workers..."

MASTER_INODE=$(stat -c %i "$AORTA_PATH" 2>/dev/null || echo "0")
MASTER_BRANCH=$(cd "$AORTA_PATH" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "not-a-git-repo")
SHARED_FS=true

for i in "${!WORKER_HOSTS[@]}"; do
    WORKER_HOST="${WORKER_HOSTS[$i]}"
    WORKER_NUM=$((i + 1))

    echo "Worker $WORKER_NUM ($WORKER_HOST):"

    if ssh "$USER@$WORKER_HOST" "test -d $AORTA_PATH" 2>/dev/null; then
        echo "  [OK] Code found: $AORTA_PATH"

        # Check if it's the same filesystem
        WORKER_INODE=$(ssh "$USER@$WORKER_HOST" "stat -c %i $AORTA_PATH" 2>/dev/null || echo "0")

        if [[ "$MASTER_INODE" == "$WORKER_INODE" ]] && [[ "$MASTER_INODE" != "0" ]]; then
            echo "  [OK] Shared filesystem detected"
        else
            echo "  [WARN] Separate filesystem - manual sync needed"
            SHARED_FS=false
        fi

        # Check git branch
        WORKER_BRANCH=$(ssh "$USER@$WORKER_HOST" "cd $AORTA_PATH && git rev-parse --abbrev-ref HEAD 2>/dev/null" || echo "not-a-git-repo")

        if [[ "$MASTER_BRANCH" == "not-a-git-repo" ]] || [[ "$WORKER_BRANCH" == "not-a-git-repo" ]]; then
            echo "  [WARN] Not a git repository - cannot verify branch"
        elif [[ "$MASTER_BRANCH" != "$WORKER_BRANCH" ]]; then
            echo "  [ERROR] Branch mismatch!"
            echo "    Master: $MASTER_BRANCH"
            echo "    Worker: $WORKER_BRANCH"
            echo ""
            echo "  Fix: ssh $WORKER_HOST 'cd $AORTA_PATH && git checkout $MASTER_BRANCH && git pull'"
            exit 1
        else
            echo "  [OK] Branch: $MASTER_BRANCH"
        fi
    else
        echo "  [FAIL] Code not found"
        echo "  You'll need to clone or rsync the code to: $AORTA_PATH"
        SHARED_FS=false
    fi
    echo ""
done

# Check GPUs
echo "Checking GPUs on all nodes..."
MASTER_GPUS=$(rocm-smi --showid 2>/dev/null | grep -c "GPU" || echo "unknown")
echo "Master: $MASTER_GPUS GPUs"

GPU_MISMATCH=false
for i in "${!WORKER_HOSTS[@]}"; do
    WORKER_HOST="${WORKER_HOSTS[$i]}"
    WORKER_NUM=$((i + 1))
    WORKER_GPUS=$(ssh "$USER@$WORKER_HOST" "rocm-smi --showid 2>/dev/null | grep -c GPU" || echo "unknown")

    echo "Worker $WORKER_NUM: $WORKER_GPUS GPUs"

    if [[ "$MASTER_GPUS" != "$WORKER_GPUS" ]]; then
        GPU_MISMATCH=true
    fi
done

if [[ "$GPU_MISMATCH" == "true" ]]; then
    echo "[WARN] GPU count mismatch detected"
    echo "  Use --nproc flag with master_launch.sh to specify GPU count per node"
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
read -p "Network interface for NCCL (press Enter to use detected, or enter different name): " USER_INTERFACE
if [[ -n "$USER_INTERFACE" ]]; then
    INTERFACE="$USER_INTERFACE"
    echo "  Using: $INTERFACE"
else
    echo "  Using detected interface: $INTERFACE"
fi
echo ""

# Create node_ip_list.txt
echo "Creating node_ip_list.txt..."
cd "$AORTA_PATH"

# Write master IP first
echo "$CURRENT_IP" > node_ip_list.txt

# Add all worker IPs
for WORKER_IP in "${WORKER_IPS[@]}"; do
    echo "$WORKER_IP" >> node_ip_list.txt
done

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
echo "  Total Nodes: $((NUM_WORKERS + 1)) (1 master + $NUM_WORKERS workers)"
echo "  Network Interface: $INTERFACE"
echo "  Shared Filesystem: ${SHARED_FS:-false}"
echo ""
echo "  Master: $CURRENT_HOST ($CURRENT_IP) - $MASTER_GPUS GPUs"
for i in "${!WORKER_HOSTS[@]}"; do
    WORKER_HOST="${WORKER_HOSTS[$i]}"
    WORKER_IP="${WORKER_IPS[$i]}"
    WORKER_NUM=$((i + 1))
    WORKER_GPUS=$(ssh "$USER@$WORKER_HOST" "rocm-smi --showid 2>/dev/null | grep -c GPU" || echo "unknown")
    echo "  Worker $WORKER_NUM: $WORKER_HOST ($WORKER_IP) - $WORKER_GPUS GPUs"
done
echo ""
echo "Node IP list created at:"
echo "  $AORTA_PATH/node_ip_list.txt"
echo ""

if [[ "${SHARED_FS:-false}" == "false" ]]; then
    echo "[IMPORTANT] Sync code to all workers before running:"
    for WORKER_HOST in "${WORKER_HOSTS[@]}"; do
        echo "  ssh $WORKER_HOST 'cd ~/ && git clone <repo> aorta'"
    done
    echo "  OR use rsync:"
    for WORKER_HOST in "${WORKER_HOSTS[@]}"; do
        echo "  rsync -avz $AORTA_PATH/ $WORKER_HOST:$AORTA_PATH/"
    done
    echo ""
fi

echo "Next Steps:"
echo ""
echo "1. Start Docker on all nodes (run once, containers persist):"
echo "  cd $AORTA_PATH"
echo "  ./scripts/multi_node/start_docker_all_nodes.sh"
echo ""
echo "2. Launch training (run as many times as you want):"
echo "  ./scripts/multi_node/master_launch.sh --channels 28 --threads 256"
echo ""
echo "3. Monitor training:"
echo "  tail -f experiments/multinode_*/logs/node_*.txt"
echo ""
echo "Additional Options:"
echo ""
echo "  Different parameters:"
echo "    ./scripts/multi_node/master_launch.sh --channels 42 --threads 512 --nproc 8"
echo ""
echo "  With profiling:"
echo "    ./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --rocprof --stats"
echo ""
echo "  Custom config:"
echo "    ./scripts/multi_node/master_launch.sh --config config/distributed_two_nodes.yaml"
echo ""
echo "Stop training on all nodes:"
for WORKER_HOST in "${WORKER_HOSTS[@]}"; do
    echo "  ssh $WORKER_HOST 'pkill -9 -f train.py'"
done
echo "  pkill -9 -f train.py  # On master"
echo ""
echo "Stop Docker when completely done:"
echo "  for IP in \$(cat node_ip_list.txt); do"
echo "    ssh \$USER@\$IP 'cd $AORTA_PATH/docker && docker compose -f docker-compose.rocm70_9-1.yaml down'"
echo "  done"
echo ""
