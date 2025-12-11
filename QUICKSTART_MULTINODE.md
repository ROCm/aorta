# Quick Start: Multi-Node Training

## 🚀 Fast Setup (5 minutes)

### 1. Create Node IP List

```bash
cd /apps/oyazdanb/aorta

# Edit and add your node IPs:
nano node_ip_list.txt
```

Add your node IPs (one per line, master node first):
```
192.168.1.10
192.168.1.11
```

### 2. Test SSH Access

```bash
# Test connectivity to all nodes:
while read IP; do
  echo "Testing $IP..."
  ssh -o ConnectTimeout=5 "$USER@$IP" "hostname" || echo "❌ FAILED: $IP"
done < node_ip_list.txt
```

If SSH fails, set up passwordless SSH:
```bash
ssh-keygen -t rsa -b 4096  # Press Enter for all prompts
for IP in $(cat node_ip_list.txt); do ssh-copy-id $USER@$IP; done
```

### 3. Configure Network Interface

Find your network interface:
```bash
# On your nodes, run:
ifconfig | grep -E "^(ib|eth|ens)" | cut -d: -f1
```

Edit `scripts/multi_node/config_node.sh` line 48:
```bash
export NCCL_SOCKET_IFNAME=ib0  # Change to your interface (ib0, eth0, etc.)
```

### 4. Launch Training

```bash
./scripts/multi_node/master_launch.sh
```

That's it! 🎉

---

## 📊 Monitor Training

### Watch All Nodes
```bash
tail -f experiments/multinode_*/logs/node_*.txt
```

### Watch Master Only
```bash
tail -f experiments/multinode_*/logs/node_0_*.txt
```

### Check Metrics
```bash
cat experiments/multinode_*/outputs/rank_00_metrics.jsonl | jq -r '.loss' | tail -5
```

---

## 🛑 Emergency Stop

```bash
for IP in $(cat node_ip_list.txt); do
  ssh $USER@$IP "pkill -9 -f 'train.py|torchrun'"
done
```

---

## ⚙️ Advanced Options

### Custom Config
```bash
CONFIG_FILE=config/multi_node/your_config.yaml ./scripts/multi_node/master_launch.sh
```

### Custom Port
```bash
MASTER_PORT=29600 ./scripts/multi_node/master_launch.sh
```

### Change Training Steps
```bash
# In your config file:
training:
  max_steps: 100  # Change this
```

---

## 📖 Full Documentation

See `scripts/multi_node/README.md` for detailed troubleshooting and advanced configuration.

---

## 🔍 Common Issues

| Issue | Solution |
|-------|----------|
| **SSH timeout** | Run: `ssh-copy-id user@node_ip` |
| **NCCL timeout** | Check `NCCL_SOCKET_IFNAME` matches your interface |
| **World size mismatch** | Ensure all nodes have 8 GPUs (`nvidia-smi -L`) |
| **Import errors** | Sync code on all nodes or use NFS |

---

## 📁 File Structure

```
aorta/
├── scripts/multi_node/
│   ├── master_launch.sh      ← Run this on head node
│   ├── config_node.sh         ← Per-node setup script
│   └── README.md              ← Full documentation
├── node_ip_list.txt           ← YOU CREATE THIS
├── config/multi_node/
│   └── distributed_two_nodes.yaml
└── experiments/
    └── multinode_TIMESTAMP/   ← Results saved here
        ├── logs/
        └── outputs/
```


