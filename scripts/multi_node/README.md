# Multi-Node Training Setup for Aorta

This directory contains scripts for orchestrating multi-node distributed training, adapted from the DLRM pattern.

## Prerequisites

1. **SSH Key Setup**: Ensure passwordless SSH access from head node to all worker nodes:
   ```bash
   ssh-copy-id user@worker_node_ip
   ```

2. **Network Configuration**: All nodes must be on the same network and able to communicate via TCP.

3. **Environment Consistency**: All nodes should have:
   - Same Python environment (conda/virtualenv)
   - Same PyTorch version
   - Same NCCL/RCCL version
   - Same Aorta codebase (synced via NFS or git)

## File Structure

```
aorta/
├── scripts/
│   └── multi_node/
│       ├── master_launch.sh        # Orchestrator (run this on head node)
│       ├── config_node.sh          # Per-node setup (called by master_launch)
│       └── README.md               # This file
├── node_ip_list.txt                # List of node IPs (YOU MUST CREATE THIS)
└── config/
    └── multi_node/
        └── distributed_two_nodes.yaml  # Multi-node training config
```

## Setup Steps

### 1. Create `node_ip_list.txt` in Aorta Root

Create a file in the Aorta root directory with your node IP addresses, one per line. First line is the master node:

```bash
cd /apps/oyazdanb/aorta
cat > node_ip_list.txt << EOF
192.168.1.10
192.168.1.11
EOF
```

To find your node IPs:
```bash
# On each node, run:
hostname -I | awk '{print $1}'
```

### 2. Configure Network Interface

Edit `config_node.sh` line 48 to set your network interface:

```bash
export NCCL_SOCKET_IFNAME=eth0  # Change to: ib0 (InfiniBand) or eth0 (Ethernet)
```

To find your network interface:
```bash
# On each node, run:
ifconfig  # or: ip addr show
```

Look for the interface connected to your high-speed network (InfiniBand: `ib0`, Ethernet: `eth0`, `ens0`, etc.)

### 3. Test SSH Connectivity

From the head node, verify SSH access to all worker nodes:

```bash
while read IP; do
  echo "Testing SSH to $IP..."
  ssh -o StrictHostKeyChecking=no "$USER@$IP" "hostname && nvidia-smi -L" || echo "FAILED: $IP"
done < node_ip_list.txt
```

### 4. (Optional) Customize Config

Edit `config/multi_node/distributed_two_nodes.yaml` to adjust:
- Model size (`model.num_layers`, `model.model_dim`)
- Batch size (`training.batch_size`)
- Number of training steps (`training.max_steps`)
- FSDP sharding strategy (`fsdp.sharding_strategy`)

For multi-node, `hybrid_shard` is recommended (shards across nodes, replicates within nodes).

## Running Multi-Node Training

### Basic Run

```bash
cd /apps/oyazdanb/aorta
./scripts/multi_node/master_launch.sh
```

### With Custom Config

```bash
CONFIG_FILE=config/multi_node/your_custom_config.yaml ./scripts/multi_node/master_launch.sh
```

### With Custom Port

```bash
MASTER_PORT=29600 ./scripts/multi_node/master_launch.sh
```

## Monitoring

### Real-time Logs (All Nodes)

```bash
tail -f experiments/multinode_*/logs/node_*.txt
```

### Master Node Only

```bash
tail -f experiments/multinode_*/logs/node_0_*.txt
```

### Check Training Progress

```bash
# On master node, check metrics:
cat experiments/multinode_*/outputs/rank_00_metrics.jsonl | tail -n 5
```

## Troubleshooting

### Issue: SSH Connection Failed

**Symptom**: `Permission denied (publickey)` or `Connection refused`

**Solution**:
```bash
# Generate SSH key if you don't have one:
ssh-keygen -t rsa -b 4096

# Copy to all nodes:
for IP in $(cat node_ip_list.txt); do
  ssh-copy-id $USER@$IP
done
```

### Issue: NCCL Initialization Timeout

**Symptom**: `NCCL WARN Bootstrap : no socket interface found` or timeout

**Solution**:
1. Check network interface name:
   ```bash
   ifconfig | grep -E "^(eth|ib|ens)"
   ```
2. Update `NCCL_SOCKET_IFNAME` in `config_node.sh`
3. Ensure firewall allows traffic on `MASTER_PORT` (default: 29500)

### Issue: Mismatched World Size

**Symptom**: `Expected world size X but got Y`

**Solution**:
- Verify all nodes in `node_ip_list.txt` are reachable
- Check that each node has 8 GPUs: `nvidia-smi -L` or `rocm-smi`

### Issue: Different Code Versions on Nodes

**Symptom**: Import errors or unexpected behavior on some nodes

**Solution**:
- Use NFS shared filesystem for codebase, OR
- Git pull on all nodes:
  ```bash
  for IP in $(cat node_ip_list.txt); do
    ssh $USER@$IP "cd /apps/$USER/aorta && git pull"
  done
  ```

## Example Output

```
=== Aorta Multi-Node Training ===
Experiment directory: /apps/oyazdanb/aorta/experiments/multinode_20251124_143052
Config file: config/multi_node/distributed_two_nodes.yaml
Number of nodes: 2
World size: 16 (GPUs)
Using MASTER_PORT: 29500

Setting up Node: 0, IP: 192.168.1.10
Master node: 192.168.1.10

Setting up Node: 1, IP: 192.168.1.11

=== All nodes launched ===
Monitor logs in: /apps/oyazdanb/aorta/experiments/multinode_20251124_143052/logs/
```

## Advanced: NCCL Environment Variables

For optimal performance, you may need to tune NCCL settings in `config_node.sh`:

### InfiniBand Networks
```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0  # Check with: ibstat
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0
```

### Ethernet Networks (slower)
```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
```

### Debug NCCL Issues
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

## Performance Tips

1. **Use InfiniBand** if available (100x faster than Ethernet for all-reduce)
2. **Tune batch size**: Multi-node works best with large batches
3. **Use `hybrid_shard`** FSDP strategy for multi-node (reduces cross-node communication)
4. **Profile first iteration** to verify overlap is working:
   ```yaml
   profiling:
     enabled: true
     active: 2  # Profile first 2 iterations
   ```

## Cleaning Up

### Stop All Training (Emergency)

```bash
# On all nodes:
for IP in $(cat node_ip_list.txt); do
  ssh $USER@$IP "pkill -9 -f 'train.py|torchrun'"
done
```

### Remove Old Experiments

```bash
# Keep last 5 experiments, delete older:
cd experiments/
ls -t | tail -n +6 | xargs rm -rf
```


