# Multi-Node Training

Scripts for multi-node distributed GEMM training with custom NCCL channel and thread configurations.

## Table of Contents

- [Quick Start](#quick-start)
- [Usage](#usage)
- [Setup](#setup) (one-time)
  - [Conductor Setup](#conductor-setup-automated)
  - [Slurm Cluster](#slurm-cluster-setup)
- [Stopping Training](#stopping-training)
- [Troubleshooting](#troubleshooting)
- [NCCL Configuration](#nccl-configuration)

## Prerequisites

- 2+ machines with ROCm GPUs, Docker, network connectivity (host mode)
- Passwordless SSH between nodes
- `scripts/multi_node/node_ip_list.txt` with node hostnames (Conductor) or IPs (Slurm) - master first
- All nodes on same git branch

## File Structure

```
aorta/
├── scripts/multi_node/
│   ├── master_launch.sh                # Main entrypoint
│   ├── start_docker_all_nodes.sh       # Start Docker on all nodes
│   ├── setup_multi_node.sh             # Automated setup (2+ nodes)
│   ├── config_node.sh                  # Per-node setup
│   ├── local_launch.sh                 # Per-node training (runs in Docker)
│   ├── set_env_variables.sh            # NCCL/RCCL config
│   └── node_ip_list.txt                # Hostnames (Conductor) or IPs (Slurm)
├── docker/
│   └── docker-compose.rocm70_9-1.yaml  # Docker config
└── config/multi_node/
    └── distributed_multinode.yaml      # Default config
```

## Quick Start

```bash
# First time setup (once per machine allocation)
./scripts/multi_node/setup_multi_node.sh
./scripts/multi_node/start_docker_all_nodes.sh

# Run training on nodes with 4 GPUs (repeat as needed)
./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --nproc 4

# Subsequent runs (no Docker restart needed)
./scripts/multi_node/master_launch.sh --channels 42 --threads 512 --nproc 2 --config config/multi_node/distributed_multinode.yaml
```

World size: `NPROC_PER_NODE × NUM_NODES` (e.g., 8 GPUs/node × 2 nodes = 16)

---

## Usage

```bash
# Basic launch (defaults: 28 channels, 256 threads, 8 GPUs/node)
./scripts/multi_node/master_launch.sh

# Custom parameters
./scripts/multi_node/master_launch.sh -c 28 -t 256 -p 4 -f config/custom.yaml
```

### Parameters

| Flag | Option | Default | Description |
|------|--------|---------|-------------|
| -c | --channels | 28 | NCCL_MAX_NCHANNELS |
| -t | --threads | 256 | RCCL_THREADS_PER_BLOCK |
| -p | --nproc | 8 | GPUs per node |
| -f | --config | config/multi_node/distributed_multinode.yaml | Config file |
| -r | --rocprof | false | Enable rocprofv3 |
| -m | --stats | false | rocprof stats |
|  | --rocprof-input | none | rocprof yaml |
|  | --master-port | auto | Master port |

Environment variables: `CHANNELS=42 THREADS=512 ./scripts/multi_node/master_launch.sh`

GPU subset: Use `-p 4` or `export CUDA_VISIBLE_DEVICES=0,2,4,6`

### Monitoring

```bash
tail -f experiments/multinode_*/logs/node_*.txt                        # All nodes
tail -f experiments/multinode_*/logs/node_0_*.txt                      # Master only
cat experiments/multinode_*/outputs/rank_00_metrics.jsonl | tail -n 5  # Metrics
```

---

## Setup

One-time setup per machine allocation. Choose your environment:

---

## Conductor Setup (Automated)

### Step 1: SSH Key Setup

```bash
# Generate and display key
ssh-keygen -t rsa -b 4096 -C "conductor-multi-node" -f ~/.ssh/id_rsa_conductor -N ''
cat ~/.ssh/id_rsa_conductor.pub
```

Register public key with your cluster's SSH key management system (wait 2-3 min for propagation)

```bash
# Configure SSH
cat >> ~/.ssh/config << 'EOF'
Host *.dcgpu smci350-* *.zts-gtu.dcgpu
    IdentityFile ~/.ssh/id_rsa_conductor
    StrictHostKeyChecking no
EOF
chmod 600 ~/.ssh/config

# Test
ssh your-worker-node.zts-gtu.dcgpu hostname
```

### Step 2: Run Setup Script

```bash
./scripts/multi_node/setup_multi_node.sh
```

Creates `scripts/multi_node/node_ip_list.txt` with hostnames, detects network interfaces, verifies SSH and git branches.

### Step 3: Start Docker

```bash
./scripts/multi_node/start_docker_all_nodes.sh  # Run once, containers persist
```

Script checks: git branches, Docker versions, SSH connectivity. Then starts containers on all nodes.

Only restart Docker after reboot or manual stop. Containers stay running between experiments.

**Debugging:** If script hangs, check the last [STAGE] message to identify where it's stuck (SSH connection, docker compose, etc).

---

## Slurm Cluster Setup

**Slurm typically uses hostnames for SSH:**

```bash
salloc -N 2 -p gpu_partition -t 4:00:00
srun --pty bash  # SSH to master
cd /path/to/aorta
hostname > scripts/multi_node/node_ip_list.txt
ssh <worker-hostname> hostname >> scripts/multi_node/node_ip_list.txt
./scripts/multi_node/master_launch.sh --channels 28 --threads 256
```

**If your Slurm cluster requires IPs for SSH instead:**

```bash
hostname -I | awk '{print $1}' > scripts/multi_node/node_ip_list.txt
ssh <worker-hostname> hostname -I | awk '{print $1}' >> scripts/multi_node/node_ip_list.txt
```

Then you may need to adjust SSH options in the multi-node scripts.

---



---

## Stopping Training

**Press `Ctrl+C`** - Stops monitoring but **training continues in background**

**Find and kill by PID:**

```bash
# Find PIDs
ps aux | grep -E 'config_node.sh|torchrun.*train.py' | grep -v grep

# Kill on all nodes (replace 12345 with your PID)
for HOST in $(cat scripts/multi_node/node_ip_list.txt); do
  ssh $USER@$HOST "kill -9 12345"
done
```

## Troubleshooting

**Script hangs:** Check last [STAGE] message to see where it's stuck. Common: SSH timeout, docker compose pulling images, container startup

**SSH fails:** `ssh-copy-id $USER@<host>` for each node, test with `ssh $USER@<host> hostname`

**SSH works manually but fails in scripts:** Check if `node_ip_list.txt` contains hostnames or IPs. Conductor requires hostnames (for SSH config matching), Slurm typically uses IPs. Match your environment.

**Docker version mismatch:** Install matching Docker version on all nodes. Check with `docker compose version` on each node.

**NCCL timeout:** Check `ifconfig | grep -E "^(eth|ib|ens)"` and update `NCCL_SOCKET_IFNAME` in `set_env_variables.sh`

**World size mismatch:** Check `rocm-smi --showid | wc -l` and adjust `--nproc`

**Code sync:** `for HOST in $(cat scripts/multi_node/node_ip_list.txt); do ssh $USER@$HOST "cd ~/aorta && git pull"; done`

**Branch mismatch:** Checkout same branch on all nodes:
```bash
TARGET_BRANCH=$(git rev-parse --abbrev-ref HEAD)
for HOST in $(cat scripts/multi_node/node_ip_list.txt | tail -n +2); do
  ssh $USER@$HOST "cd ~/aorta && git checkout $TARGET_BRANCH && git pull"
done
```

## NCCL Configuration

Edit `set_env_variables.sh`:

**InfiniBand:**
```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0  # Check: ibstat
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0
```

**Ethernet:**
```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
```

**Debug:** `export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL`
