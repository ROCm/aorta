# Multi-Node Training

Scripts for multi-node distributed training with custom NCCL channel and thread configurations.

## Table of Contents

- [Quick Start](#quick-start)
- [Slurm Setup](#slurm-setup)
- [Usage](#usage)
- [Stopping Training](#stopping-training)
- [Troubleshooting](#troubleshooting)
- [NCCL Configuration](#nccl-configuration)
- [Conductor Setup](#conductor-setup)

## Prerequisites

- 2+ machines with ROCm GPUs, Docker, network connectivity (host mode)
- Passwordless SSH between nodes
- `scripts/multi_node/node_ip_list.txt` with node hostnames - master first
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
│   └── node_ip_list.txt                # Node hostnames
├── docker/
│   ├── docker-compose.rocm70_9-1.yaml         # Base Docker config
│   └── docker-compose.rocm70_9-1-shampoo.yaml # Docker with Shampoo optimizer
├── config/
│   ├── multi_node/
│   │   └── distributed_multinode.yaml  # Default config
│   └── shampoo_opt.yaml                # Shampoo optimizer config
```

## Quick Start

```bash
# First time setup (once per allocation)
scontrol show hostnames $SLURM_NODELIST > scripts/multi_node/node_ip_list.txt
./scripts/multi_node/start_docker_all_nodes.sh

# Run training
./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --nproc 8

# With custom Docker container and experiment label
./scripts/multi_node/master_launch.sh --docker my-container --label experiment_v1
```

World size: `NPROC_PER_NODE × NUM_NODES` (e.g., 8 GPUs/node × 2 nodes = 16)

---

## Slurm Setup

### Step 1: Pull Base Docker Image

```bash
docker pull rocm/pytorch-private:20251030_rocm_e2e_phantom_mi350_genai_nightly

# If authentication required
docker login
```

### Step 2: Allocate Nodes

```bash
# From head node
salloc -N 3 -p gpu_partition -t 4:00:00
squeue -u $USER
```

### Step 3: Create node_ip_list.txt

```bash
cd /path/to/aorta/scripts/multi_node
scontrol show hostnames $SLURM_NODELIST > node_ip_list.txt
cat node_ip_list.txt
```

### Step 4: SSH to Master and Test Connectivity

```bash
ssh node1-hostname
cd /path/to/aorta

# Test worker connectivity
ssh node2-hostname hostname
ssh node3-hostname hostname
```

### Step 5: Pull Image on All Nodes

```bash
for HOST in $(cat scripts/multi_node/node_ip_list.txt); do
  ssh $HOST "docker pull rocm/pytorch-private:20251030_rocm_e2e_phantom_mi350_genai_nightly"
done
```

### Step 6: Start Docker and Run Training

```bash
./scripts/multi_node/start_docker_all_nodes.sh

./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --nproc 8
```

---

## Usage

```bash
# Basic launch (defaults: 28 channels, 256 threads, 8 GPUs/node)
./scripts/multi_node/master_launch.sh

# Custom parameters
./scripts/multi_node/master_launch.sh -c 28 -t 256 -p 4 -f config/custom.yaml

# With Shampoo optimizer container and experiment label
./scripts/multi_node/master_launch.sh \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --label shampoo_test \
    --config config/multi_node/shampoo_opt_multi_node.yaml
```

### Parameters

| Flag | Option | Default | Description |
|------|--------|---------|-------------|
| -c | --channels | 28 | NCCL_MAX_NCHANNELS |
| -t | --threads | 256 | RCCL_THREADS_PER_BLOCK |
| -p | --nproc | 8 | GPUs per node |
| -f | --config | config/multi_node/distributed_multinode.yaml | Config file |
| -d | --docker | training-overlap-bugs-rocm70_9-1 | Docker container name |
| -l | --label | none | Experiment label (appended to directory name) |
| -r | --rocprof | false | Enable rocprofv3 |
| -m | --stats | false | rocprof stats |
|  | --rocprof-input | none | rocprof yaml |
|  | --master-port | auto | Master port |

Supports `--option=value` syntax: `./scripts/multi_node/master_launch.sh --docker=my-container --label=test`

Environment variables: `CHANNELS=42 THREADS=512 ./scripts/multi_node/master_launch.sh`

GPU subset: Use `-p 4` or `export CUDA_VISIBLE_DEVICES=0,2,4,6`

### Custom Config

Select a config file from `config/` or `config/multi_node/`:

```bash
./scripts/multi_node/master_launch.sh \
    --channels 28 --threads 256 \
    --config config/multi_node/distributed_multinode.yaml
```

### Experiment Output

Each run creates an experiment directory with:
```
experiments/multinode_28ch_256th_20260119_171958_mylabel/
├── config_used.yaml       # Copy of config file used
├── experiment_info.txt    # Experiment metadata
├── logs/                  # Per-node launch logs
│   ├── node_0_*.txt
│   └── node_1_*.txt
└── 256thread_28channels/  # Training outputs
    ├── rank_00_metrics.jsonl
    └── checkpoints/
```

### Monitoring

```bash
tail -f experiments/multinode_*/logs/node_*.txt                        # All nodes
tail -f experiments/multinode_*/logs/node_0_*.txt                      # Master only
cat experiments/multinode_*/256thread_*/rank_00_metrics.jsonl | tail -5  # Metrics
cat experiments/multinode_*/experiment_info.txt                        # Experiment info
```

---

## Stopping Training

`Ctrl+C` stops monitoring but training continues in background.

To stop training (replace container name if using `--docker`):
```bash
CONTAINER="training-overlap-bugs-rocm70_9-1"  # or your custom container
for HOST in $(cat scripts/multi_node/node_ip_list.txt); do
  ssh $HOST "docker exec $CONTAINER pkill -9 -f 'train.py|torchrun'"
done
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Script hangs | Check last [STAGE] message |
| SSH fails | `ssh-copy-id $USER@<host>` |
| Docker version mismatch | `docker compose version` on each node |
| NCCL timeout | Update `NCCL_SOCKET_IFNAME` in `set_env_variables.sh` |
| World size mismatch | Check `rocm-smi --showid \| wc -l`, adjust `--nproc` |

### Training Hangs at RCCL Initialization

If training hangs at "Warming up global world group..." or during FSDP initialization:

1. **Ensure NCCL environment variables are set** in `local_launch.sh`:
   - `NCCL_SOCKET_IFNAME` and `TORCH_NCCL_DUMP_ON_TIMEOUT=1` are critical
   - See the full set in `local_launch.sh` DOCKER_EXEC section

2. **Enable warmup settings** in your config YAML:

```yaml
warmup:
  # RCCL communicator warmup - runs all_reduce before FSDP init
  enable_rccl_warmup: true
  rccl_warmup_iterations: 5
  # Training warmup - runs forward/backward/optimizer before main loop
  enable_training_warmup: true
  training_warmup_steps: 1
```

3. **Debug with NCCL logging**:
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

The warmup settings exercise RCCL communicators before the main training loop starts, preventing race conditions during inter-node RDMA setup with HYBRID_SHARD strategy.

---

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

---

## Conductor Setup

For Conductor environments with SSH key management:

### SSH Key Setup

```bash
ssh-keygen -t rsa -b 4096 -C "conductor-multi-node" -f ~/.ssh/id_rsa_conductor -N ''
cat ~/.ssh/id_rsa_conductor.pub
```

Register public key with your cluster's SSH key management system.

```bash
cat >> ~/.ssh/config << 'EOF'
Host *.dcgpu smci350-* *.zts-gtu.dcgpu
    IdentityFile ~/.ssh/id_rsa_conductor
    StrictHostKeyChecking no
EOF
chmod 600 ~/.ssh/config
```

### Run Setup and Start Docker

```bash
./scripts/multi_node/setup_multi_node.sh
./scripts/multi_node/start_docker_all_nodes.sh
```

Creates `node_ip_list.txt` with hostnames, detects network interfaces, verifies SSH and git branches.
