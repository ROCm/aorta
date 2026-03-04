# Multi-Node Training

Scripts for multi-node distributed training with custom NCCL channel and thread configurations.

## Table of Contents

- [Quick Start](#quick-start)
- [Slurm Setup](#slurm-setup)
- [Usage](#usage)
- [Race Experiments](#race-experiments)
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
│   └── docker-compose.rocm70_9-1-shampoo.yaml # Docker with Shampoo (supports both Adam and Shampoo)
├── config/
│   ├── multi_node/
│   │   ├── distributed_multinode.yaml  # Default config
│   │   └── shampoo_opt_multi_node.yaml # Shampoo + race experiments config
│   └── shampoo_opt.yaml                # Shampoo optimizer config
├── src/aorta/
│   ├── training/
│   │   └── fsdp_trainer.py             # Main FSDP trainer
│   └── race/                           # Standalone race condition reproducer
│       ├── __init__.py                 # Public API exports
│       ├── __main__.py                 # CLI entry point
│       ├── config.py                   # ReproducerConfig, RaceConfig
│       ├── base.py                     # BaseReproducer abstract class
│       ├── compute.py                  # Pluggable compute simulation
│       └── modes/                      # Mode implementations
│           ├── __init__.py             # Factory + registry
│           ├── default.py              # TorchRec-like (all_to_all + all_reduce)
│           ├── ddp.py                  # DDP (gradient all_reduce)
│           └── fsdp.py                 # FSDP (all_gather + reduce_scatter)
```

## Quick Start

```bash
# First time setup (once per allocation)
scontrol show hostnames $SLURM_NODELIST > scripts/multi_node/node_ip_list.txt
./scripts/multi_node/start_docker_all_nodes.sh \
    docker/docker-compose.rocm70_9-1-shampoo.yaml \
    training-overlap-bugs-rocm70_9-1-shampoo

# Run training
./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --nproc 8 \
    --config ./config/multi_node/shampoo_opt_multi_node.yaml \
    --docker training-overlap-bugs-rocm70_9-1-shampoo
```

Change optimizer by editing the config file (`optimizer.name: adamw` or `optimizer.name: shampoo`).

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
./scripts/multi_node/start_docker_all_nodes.sh \
    docker/docker-compose.rocm70_9-1-shampoo.yaml \
    training-overlap-bugs-rocm70_9-1-shampoo

./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --nproc 8 \
    --config ./config/multi_node/shampoo_opt_multi_node.yaml \
    --docker training-overlap-bugs-rocm70_9-1-shampoo
```

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
| -d | --docker | training-overlap-bugs-rocm70_9-1 | Docker container name |
| -l | --label | (none) | Experiment label suffix |
| -r | --rocprof | false | Enable rocprofv3 |
| -m | --stats | false | rocprof stats |
|  | --rocprof-input | none | rocprof yaml |
|  | --master-port | auto | Master port |

### Experiment Labels

Use `--label` to add a descriptive suffix to experiment directories for easy identification:

```bash
# Without label - creates: experiments/multinode_28ch_256th_20260116_123456/
./scripts/multi_node/master_launch.sh

# With label - creates: experiments/multinode_28ch_256th_20260116_123456_shampoo_test/
./scripts/multi_node/master_launch.sh --label shampoo_test

# Short form
./scripts/multi_node/master_launch.sh -l my_experiment
```

This is useful when running multiple experiments to track different configurations, optimizers, or hyperparameters.

Environment variables: `CHANNELS=42 THREADS=512 ./scripts/multi_node/master_launch.sh`

GPU subset: Use `-p 4` or `export ROCR_VISIBLE_DEVICES=0,2,4,6`

### Monitoring

```bash
tail -f experiments/multinode_*/logs/node_*.txt                        # All nodes
tail -f experiments/multinode_*/logs/node_0_*.txt                      # Master only
cat experiments/multinode_*/outputs/rank_00_metrics.jsonl | tail -n 5  # Metrics
```

---

## Race Experiments

The `src/aorta/race/` module provides tools to inject controlled race conditions for testing distributed training robustness.

### Configuration

Race settings are configured in the `race_experiment:` section of your config file:

```yaml
race_experiment:
  # Warmup control
  skip_training_warmup: false         # Skip forward/backward warmup
  training_warmup_steps: 1            # Steps if warmup is enabled
  skip_rccl_warmup: false             # Skip RCCL communicator warmup
  rccl_warmup_iterations: 10          # Iterations if warmup is enabled

  # H2D memcpy racing (realistic pattern)
  h2d_memcpy_racing: false           # Use separate stream for H2D batch copy
  h2d_skip_sync_before_forward: false # Skip sync before forward (causes race!)
  h2d_racing_start_step: 0           # Step to start racing

  # Supporting options
  nan_check_collectives: false       # NaN check around RCCL collectives
  gpu_max_hw_queues: null            # Set to 4+ to expose races
```

### Race Types

| Type | Description |
|------|-------------|
| **H2D Race** | Races H2D memcpy with forward pass. Enable `h2d_memcpy_racing` + `h2d_skip_sync_before_forward` |
| **Datadist Race** | Races all_to_all with FSDP collectives. Enable `datadist_racing` + `datadist_skip_sync_before_collective` |

### Hardware Queues

**Critical:** Set `gpu_max_hw_queues: 4` to expose races. With 1-2 queues, streams share hardware queues causing implicit serialization that masks races.

```yaml
race_experiment:
  gpu_max_hw_queues: 4  # Required to expose race conditions
```

Or via environment: `export GPU_MAX_HW_QUEUES=4`

---

## Stopping Training

`Ctrl+C` stops monitoring but training continues in background.

To stop training:
```bash
for HOST in $(cat scripts/multi_node/node_ip_list.txt); do
  ssh $HOST "docker exec training-overlap-bugs-rocm70_9-1 pkill -9 -f 'train.py|torchrun'"
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
