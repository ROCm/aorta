# Multi-Node Training

Scripts for multi-node distributed GEMM training with custom NCCL channel and thread configurations.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Common Workflows](#common-workflows)
- [Monitoring](#monitoring)
- [Setup](#setup) (one-time configuration)
  - [Conductor Setup (Docker)](#conductor-setup-automated)
  - [Slurm Cluster Setup](#slurm-cluster-setup)
  - [Manual Setup (Non-Docker)](#manual-setup-general-cluster--non-docker)
- [Troubleshooting](#troubleshooting)
- [NCCL Configuration](#nccl-configuration)

## Prerequisites

- 2+ machines with ROCm GPUs
- Docker installed on all nodes
- Network connectivity between nodes (with `network_mode: host`)
- Shared codebase (NFS mount) or manual sync via git/rsync
- For Conductor: Active reservations on all machines

## File Structure

```
aorta/
├── scripts/multi_node/
│   ├── master_launch.sh                # Main entrypoint
│   ├── start_docker_all_nodes.sh       # Start Docker on all nodes
│   ├── setup_multi_node.sh             # Automated setup (2+ nodes)
│   ├── config_node.sh                  # Per-node setup
│   ├── local_launch.sh                 # Per-node training (runs in Docker)
│   └── set_env_variables.sh            # NCCL/RCCL config
├── docker/
│   └── docker-compose.rocm70_9-1.yaml  # Docker config
├── node_ip_list.txt                    # Node IPs (create this)
└── config/multi_node/
    └── distributed_multinode.yaml      # Default config
```

## World Size

```
WORLD_SIZE = NPROC_PER_NODE × NUMBER_OF_NODES
```

| Nodes | GPUs/Node | World Size | Command |
|-------|-----------|------------|---------|
| 2 | 8 | 16 | `./scripts/multi_node/master_launch.sh` |
| 2 | 4 | 8 | `./scripts/multi_node/master_launch.sh -p 4` |
| 4 | 8 | 32 | `./scripts/multi_node/master_launch.sh -p 8` |
| 8 | 8 | 64 | `./scripts/multi_node/master_launch.sh -p 8` |

## Quick Start

**Complete workflow (first time):**

```bash
# 1. Setup SSH keys (once ever) - see Conductor Setup
# 2. Run setup script (once per machine allocation)
./scripts/multi_node/setup_multi_node.sh

# 3. Start Docker (once after setup, containers persist)
./scripts/multi_node/start_docker_all_nodes.sh

# 4. Run training (as many times as you want!)
./scripts/multi_node/master_launch.sh --channels 28 --threads 256
```

**Subsequent experiments (same machines, Docker still running):**

```bash
# Just launch training - no setup or Docker restart needed!
./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --nproc 4

# Different parameters
./scripts/multi_node/master_launch.sh --channels 42 --threads 512 --rocprof

# Run as many experiments as you want!
```



**Prerequisites:**
- [ ] Passwordless SSH between nodes
- [ ] `node_ip_list.txt` with node IPs (master first)
- [ ] Docker containers running on all nodes
- [ ] Code synced across all nodes (mounted in Docker)
- [ ] **All nodes on the same git branch** (checked automatically)

**First-time setup?** See [Setup](#setup) section below.

---

## Usage

**After completing setup once, use these commands for all subsequent training runs.**

### Basic Launch

```bash
./scripts/multi_node/master_launch.sh
```

Default: 28 channels, 256 threads, 8 GPUs/node, config/multi_node/distributed_multinode.yaml

### Parameters

```bash
./scripts/multi_node/master_launch.sh -c 28 -t 256 -p 4 -f config/custom.yaml --rocprof
```

| Flag | Option | Default | Description |
|------|--------|---------|-------------|
| -c | --channels | 28 | NCCL_MAX_NCHANNELS |
| -t | --threads | 256 | RCCL_THREADS_PER_BLOCK |
| -p | --nproc | 4 | GPUs per node |
| -f | --config | config/multi_node/distributed_multinode.yaml | Config file |
| -r | --rocprof | false | Enable rocprofv3 |
| -m | --stats | false | rocprof stats |
|  | --rocprof-input | none | rocprof yaml |
|  | --master-port | auto | Master port |
| -h | --help | - | Show help |

Environment variables also supported:
```bash
CHANNELS=42 THREADS=512 ./scripts/multi_node/master_launch.sh
```

### GPU Subset

```bash
# 4 GPUs per node
./scripts/multi_node/master_launch.sh -p 4

# Specific GPUs
export CUDA_VISIBLE_DEVICES=0,2,4,6
./scripts/multi_node/master_launch.sh -p 4
```

Note: Set CUDA_VISIBLE_DEVICES on all nodes via set_env_variables.sh

### Examples

```bash
# Default run
./scripts/multi_node/master_launch.sh

# Custom parameters
./scripts/multi_node/master_launch.sh -c 28 -t 256 -p 4

# With profiling
./scripts/multi_node/master_launch.sh -c 28 -t 256 --rocprof --stats

# Custom config
./scripts/multi_node/master_launch.sh -f config/my_config.yaml -c 28
```

## Common Workflows

### Quick Test Run (2 nodes, 4 GPUs each)

```bash
./scripts/multi_node/master_launch.sh -p 4 -c 28 -t 256
```

### Parameter Sweep (Multiple Configurations)

To sweep channels and threads like single-node `run_train_various_channels.sh`:

```bash
# Run multiple configurations sequentially
for CHANNELS in 28 42 56 70; do
  for THREADS in 256 512; do
    echo "Running CHANNELS=$CHANNELS THREADS=$THREADS"
    ./scripts/multi_node/master_launch.sh -c $CHANNELS -t $THREADS -p 8
    sleep 10  # Wait between runs
  done
done
```

### With Profiling

```bash
# Basic profiling
./scripts/multi_node/master_launch.sh -c 28 -t 256 --rocprof

# With stats (CU utilization, occupancy)
./scripts/multi_node/master_launch.sh -c 28 -t 256 --rocprof --stats
```

### Custom Config

```bash
./scripts/multi_node/master_launch.sh \
  -f config/my_custom_model.yaml \
  -c 28 -t 256 -p 8
```

### Complete Workflow (First Time Setup + Multiple Experiments)

```bash
# === ONE-TIME SETUP (per machine allocation) ===

# 1. Run setup script (SSH keys should already be configured)
./scripts/multi_node/setup_multi_node.sh

# 2. Start Docker on all nodes (run once, containers persist)
./scripts/multi_node/start_docker_all_nodes.sh

# === RUN EXPERIMENTS (as many times as you want!) ===

# 3. Launch first experiment
./scripts/multi_node/master_launch.sh -c 28 -t 256

# 4. Monitor training
tail -f experiments/multinode_*/logs/node_*.txt

# 5. Launch more experiments (NO Docker restart needed!)
./scripts/multi_node/master_launch.sh -c 42 -t 512 --rocprof
./scripts/multi_node/master_launch.sh -c 56 -t 256
./scripts/multi_node/master_launch.sh -c 70 -t 512 --rocprof --stats

# === CLEANUP (only when completely done) ===

# 6. Stop Docker when done with machine allocation
for IP in $(cat node_ip_list.txt); do
  ssh $USER@$IP "cd /home/$USER/aorta/docker && docker compose -f docker-compose.rocm70_9-1.yaml down"
done
```

**Important:**
- Docker containers **stay running** between experiments
- Only stop Docker when you're **completely done** with the machines
- Each training run uses the same Docker containers

## What Happens During Launch

When you run `master_launch.sh`, it:

1. **Verifies git branch consistency** across all nodes (fails if mismatch)
2. Reads `node_ip_list.txt` to find worker nodes
3. SSHs to each node and runs `local_launch.sh` (background process)
4. Each node sources `set_env_variables.sh` (NCCL/RCCL settings)
5. Launches `torchrun` inside Docker containers with your specified parameters
6. Creates output directory: `experiments/multinode_<channels>ch_<threads>th_<timestamp>/`
7. Logs for each node saved to `logs/node_<rank>_<timestamp>.txt`

## Monitoring

```bash
# All nodes
tail -f experiments/multinode_*/logs/node_*.txt

# Master only
tail -f experiments/multinode_*/logs/node_0_*.txt

# Training metrics
cat experiments/multinode_*/outputs/rank_00_metrics.jsonl | tail -n 5
```

---

## Setup

Complete one of these setup paths based on your environment (one-time only).

---

## Conductor Setup (Automated)

**One-time setup (run when you first get machines, or when machines change):**

### Step 1: SSH Key Setup Between Nodes

Conductor nodes cannot SSH to each other by default. Generate a key and register it with Conductor.

**On your master node:**

```bash
# Generate SSH key
ssh-keygen -t rsa -b 4096 -C "conductor-multi-node" -f ~/.ssh/id_rsa_conductor -N ''

# Display public key
cat ~/.ssh/id_rsa_conductor.pub
```

**Register key with Conductor:**
1. Copy the public key output above
2. Go to https://conductor.amd.com/user/dashboard/key-management
3. Click "Add SSH Key" and paste the public key
4. Wait 2-3 minutes for propagation

**Configure SSH to use the key:**

```bash
cat >> ~/.ssh/config << 'EOF'
Host *.dcgpu smci350-* *.zts-gtu.dcgpu
    IdentityFile ~/.ssh/id_rsa_conductor
    StrictHostKeyChecking no
EOF

chmod 600 ~/.ssh/config
```

**Test SSH connectivity:**

```bash
# Replace with your worker node hostname
ssh your-worker-node.zts-gtu.dcgpu hostname
```

If successful, you'll see the worker hostname. If you see "Permission denied", wait a bit longer or verify the key was added correctly.

### Step 2: Run Automated Setup Script

**Run once per machine allocation (or when nodes change):**

```bash
cd ~/aorta
./scripts/multi_node/setup_multi_node.sh
```

This interactive script will:
- Ask how many worker nodes you have (default: 1)
- Prompt for each worker node hostname
- Verify SSH connectivity between all nodes
- Check git branch consistency across all nodes
- Detect network interfaces and GPU counts
- Create `node_ip_list.txt` with all node IPs
- Configure `NCCL_SOCKET_IFNAME` in `set_env_variables.sh`

**Supports any number of nodes:** 2, 3, 4, 8, 16+

**When to run again:**
- New machine allocation (different machines)
- Number of nodes changed
- Node IPs changed
- Need to reconfigure network interface

**Verify setup:**

```bash
# Check node_ip_list.txt was created
cat node_ip_list.txt

# Test SSH to all nodes
while read IP; do ssh "$USER@$IP" hostname; done < node_ip_list.txt
```

### Step 3: Start Docker Containers

**Run once after setup** (containers stay running for multiple experiments):

```bash
# Start Docker on all nodes
./scripts/multi_node/start_docker_all_nodes.sh

# Verify containers are running
docker ps  # Check master
ssh $USER@<worker-hostname> "docker ps"  # Check worker
```

**Check if Docker is already running:**

```bash
# Quick check on all nodes
docker ps  # Master
for IP in $(cat node_ip_list.txt | tail -n +2); do
  echo "Node $IP:"
  ssh $USER@$IP "docker ps | grep training-overlap-bugs"
done
```

**When to run start_docker_all_nodes.sh:**
- **After initial setup** (Step 2) - Run once
- **After machine reboot** - If containers stopped
- **After `docker compose down`** - If you manually stopped containers
- **If containers not running** - Check with `docker ps`
- **NOT between experiments** - Containers stay running!

**Important:**
- Multi-node scripts automatically run commands inside Docker container (`training-overlap-bugs-rocm70_9-1`)
- Container uses `network_mode: host` for node-to-node communication
- Code directory mounted: `/home/$USER/aorta:/workspace/aorta`
- Container must be running on ALL nodes before launching training
- **Containers persist** - No need to restart between training runs

### Step 4: Launch Training

```bash
./scripts/multi_node/master_launch.sh --channels 28 --threads 256
```

**Subsequent runs** (same machines):
```bash
# No setup needed! Just launch training again
./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --nproc 8

# Or with different parameters
./scripts/multi_node/master_launch.sh --channels 42 --threads 512 --rocprof
```

See [Usage](#usage) section for all launch options.

---

## Slurm Cluster Setup

For Slurm-managed HPC clusters with consistent environments (Docker optional if ROCm/PyTorch already installed).

```bash
# 1. Allocate nodes (adjust partition and time as needed)
salloc -N 2 -p gpu_partition -t 4:00:00

# 2. Verify allocation and GPUs
srun -N 2 hostname
srun -N 2 rocm-smi --showid

# 3. SSH to master node
srun --pty bash

# 4. Inside master node, create node_ip_list.txt
cd /path/to/aorta
hostname -I | awk '{print $1}' > node_ip_list.txt
ssh <worker-hostname> hostname -I | awk '{print $1}' >> node_ip_list.txt

# 5. Launch training
./scripts/multi_node/master_launch.sh --channels 28 --threads 256
```

**Note:** Slurm allocations typically have SSH pre-configured between nodes.

**To cancel:**
```bash
scancel <job_id>
```

---

## Manual Setup (General Cluster / Non-Docker)

For bare metal, cloud VMs, or non-Slurm clusters. **This path is for advanced users running without Docker.**

**Note:** The recommended approach is to use Docker (see Conductor setup above). Non-Docker setups require manual environment management across all nodes.

### Step 1: Setup Passwordless SSH

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t rsa -b 4096

# Copy to all worker nodes
for IP in <worker1_ip> <worker2_ip>; do
  ssh-copy-id $USER@$IP
done
```

### Step 2: Create node_ip_list.txt

```bash
cd /path/to/aorta

# Find your IP on each node
hostname -I | awk '{print $1}'

# Create node list (master first, then workers)
cat > node_ip_list.txt << EOF
192.168.1.10
192.168.1.11
192.168.1.12
EOF
```

### Step 3: Configure Network Interface

Find your network interface:

```bash
ifconfig | grep -E "^(eth|ib|ens|enp)"
# Example output: enp49s0f0np0, ib0, eth0
```

Edit `scripts/multi_node/set_env_variables.sh`:

```bash
# Line 17 - change to your interface
export NCCL_SOCKET_IFNAME=enp49s0f0np0  # Your interface here
```

### Step 4: Sync Code to All Nodes

**Option A: NFS/shared filesystem** (recommended)
- Code automatically synced if `/path/to/aorta` is on shared storage

**Option B: Manual sync**
```bash
for IP in $(cat node_ip_list.txt | tail -n +2); do
  rsync -avz /path/to/aorta/ $USER@$IP:/path/to/aorta/
done
```

**Option C: Git pull on each node**
```bash
for IP in $(cat node_ip_list.txt); do
  ssh $USER@$IP "cd /path/to/aorta && git pull"
done
```

### Step 5: Verify Setup

```bash
# Test SSH to all nodes
while read IP; do ssh "$USER@$IP" hostname; done < node_ip_list.txt

# Check GPU count on all nodes
while read IP; do
  echo "Node $IP:"
  ssh "$USER@$IP" "rocm-smi --showid | wc -l"
done < node_ip_list.txt
```

### Step 6: Launch Training

```bash
./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --nproc 8
```

---

## Troubleshooting

### SSH fails

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t rsa -b 4096

# Copy to all nodes
for IP in $(cat node_ip_list.txt); do
  ssh-copy-id $USER@$IP
done

# Test connectivity
while read IP; do
  echo "Testing $IP..."
  ssh "$USER@$IP" hostname
done < node_ip_list.txt
```

**For managed clusters (Conductor, Slurm):** Consult your cluster documentation for SSH key management.

### NCCL timeout
```bash
# Check interface
ifconfig | grep -E "^(eth|ib|ens)"

# Update set_env_variables.sh
export NCCL_SOCKET_IFNAME=<your_interface>

# Check firewall on MASTER_PORT (default 29500)
```

### World size mismatch
```bash
# Check GPUs per node
rocm-smi --showid | wc -l

# Adjust nproc
./scripts/multi_node/master_launch.sh --nproc <num_gpus>
```

### Code sync
```bash
# Pull on all nodes
for IP in $(cat node_ip_list.txt); do
  ssh $USER@$IP "cd /path/to/aorta && git pull"
done
```

### Branch mismatch

If you see "[ERROR] Branch mismatch", all nodes must be on the same git branch:

```bash
# Check current branches
git branch  # Master
ssh $USER@<worker-ip> "cd ~/aorta && git branch"  # Worker

# Fix: checkout same branch on all nodes
TARGET_BRANCH=$(git rev-parse --abbrev-ref HEAD)
for IP in $(cat node_ip_list.txt | tail -n +2); do
  echo "Updating node $IP to branch $TARGET_BRANCH..."
  ssh $USER@$IP "cd ~/aorta && git checkout $TARGET_BRANCH && git pull"
done
```

**Automatic checks:** Both `start_docker_all_nodes.sh` and `master_launch.sh` verify branch consistency before running.

## NCCL Configuration

Edit `set_env_variables.sh` for network-specific tuning:

### InfiniBand
```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0  # Check: ibstat
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0
```

### Ethernet
```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
```

### Debug
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

## Emergency Stop

```bash
for IP in $(cat node_ip_list.txt); do
  ssh $USER@$IP "pkill -9 -f 'train.py|torchrun'"
done
```

## Output Structure

```
experiments/multinode_28ch_256th_20251211_143052/
├── logs/
│   ├── node_0_20251211_143052.txt
│   └── node_1_20251211_143052.txt
└── 256thread_28channels/
    ├── node_0_output.log
    ├── node_1_output.log
    └── torch_profiler/
```
