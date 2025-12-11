# Multi-Node Training

Scripts for multi-node distributed GEMM training with custom NCCL channel and thread configurations.

## Prerequisites

- Passwordless SSH between nodes
- Same network, Python environment, PyTorch/NCCL version across nodes
- Shared codebase (NFS or git sync)

## File Structure

```
aorta/
├── scripts/multi_node/
│   ├── master_launch.sh         # Main entrypoint
│   ├── config_node.sh           # Per-node setup
│   ├── local_launch.sh          # Per-node training
│   └── set_env_variables.sh     # NCCL/RCCL config
├── node_ip_list.txt             # Node IPs (create this)
└── config/multi_node/
    └── distributed_multinode.yaml  # Default config
```

## World Size

```
WORLD_SIZE = NPROC_PER_NODE × NUMBER_OF_NODES
```

| Nodes | GPUs/Node | World Size | Command |
|-------|-----------|------------|---------|
| 2 | 8 | 16 | `./scripts/multi_node/master_launch.sh` |
| 2 | 4 | 8 | `./scripts/multi_node/master_launch.sh -p 4` |
| 2 | 2 | 4 | `./scripts/multi_node/master_launch.sh -p 2` |

## Setup

Quick setup for Conductor machines: `./scripts/multi_node/setup_my_conductor_machines.sh`

### 1. Create node_ip_list.txt

```bash
cd /apps/oyazdanb/aorta
cat > node_ip_list.txt << EOF
192.168.1.10
192.168.1.11
EOF
```

First line is master node. Find IPs: `hostname -I | awk '{print $1}'`

### 2. Configure Network Interface

Edit `set_env_variables.sh` line 17:
```bash
export NCCL_SOCKET_IFNAME=enp49s0f0np0  # Change to your interface
```

Find interface: `ifconfig | grep -E "^(eth|ib|ens)"`

### 3. Test SSH

```bash
while read IP; do ssh "$USER@$IP" hostname; done < node_ip_list.txt
```

Setup keys if needed: `ssh-copy-id $USER@<node_ip>`

## Usage

### Basic

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

## Monitoring

```bash
# All nodes
tail -f experiments/multinode_*/logs/node_*.txt

# Master only
tail -f experiments/multinode_*/logs/node_0_*.txt

# Training metrics
cat experiments/multinode_*/outputs/rank_00_metrics.jsonl | tail -n 5
```

## Troubleshooting

### SSH fails
```bash
ssh-keygen -t rsa -b 4096
for IP in $(cat node_ip_list.txt); do ssh-copy-id $USER@$IP; done
```

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
  ssh $USER@$IP "cd /apps/$USER/aorta && git pull"
done
```

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

