#!/bin/bash
# Start Docker containers on all nodes for multi-node training

set -e

MACHINE_IP_FILE="node_ip_list.txt"
DOCKER_COMPOSE_FILE="docker/docker-compose.rocm70_9-1.yaml"
DOCKER_CONTAINER="training-overlap-bugs-rocm70_9-1"

if [[ ! -f "$MACHINE_IP_FILE" ]]; then
    echo "Error: $MACHINE_IP_FILE not found"
    echo "Run setup_multi_node.sh first"
    exit 1
fi

# Check git branch consistency before starting Docker
echo "=== Checking git branch consistency ==="
MASTER_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "not-a-git-repo")

if [[ "$MASTER_BRANCH" != "not-a-git-repo" ]]; then
    echo "Master node branch: $MASTER_BRANCH"

    node=0
    while IFS= read -r IP || [[ -n "$IP" ]]; do
        if [[ -z "$IP" ]]; then continue; fi

        if [[ "$node" -gt 0 ]]; then
            WORKER_BRANCH=$(ssh "$USER@$IP" "cd ~/aorta && git rev-parse --abbrev-ref HEAD 2>/dev/null" || echo "not-a-git-repo")

            if [[ "$WORKER_BRANCH" == "not-a-git-repo" ]]; then
                echo "[WARN] Worker node $IP: Not a git repository"
            elif [[ "$MASTER_BRANCH" != "$WORKER_BRANCH" ]]; then
                echo "[ERROR] Branch mismatch on node $IP!"
                echo "  Master: $MASTER_BRANCH"
                echo "  Worker: $WORKER_BRANCH"
                echo ""
                echo "Fix: ssh $USER@$IP 'cd ~/aorta && git checkout $MASTER_BRANCH && git pull'"
                exit 1
            else
                echo "Worker node $IP: $WORKER_BRANCH [OK]"
            fi
        fi
        ((node++))
    done < "$MACHINE_IP_FILE"
    echo ""
else
    echo "[WARN] Not a git repository - skipping branch check"
    echo ""
fi

echo "=== Starting Docker containers on all nodes ==="
echo ""

node=0
while IFS= read -r IP || [[ -n "$IP" ]]; do
  if [[ -z "$IP" ]]; then
    continue
  fi

  echo "Node $node (IP: $IP):"

  if [[ "$node" -eq 0 ]]; then
    # Master node (local)
    echo "  Starting Docker on master node..."
    cd docker && docker compose -f docker-compose.rocm70_9-1.yaml up -d && cd ..

    if docker ps --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER}$"; then
      echo "  [OK] Docker container '${DOCKER_CONTAINER}' is running"
    else
      echo "  [FAIL] Failed to start Docker container"
      exit 1
    fi
  else
    # Worker nodes (via SSH)
    echo "  Starting Docker on worker node via SSH..."
    ssh -o StrictHostKeyChecking=no "$USER@$IP" \
      "cd /home/$USER/aorta/docker && docker compose -f docker-compose.rocm70_9-1.yaml up -d"

    # Verify
    if ssh "$USER@$IP" "docker ps --format '{{.Names}}'" | grep -q "^${DOCKER_CONTAINER}$"; then
      echo "  [OK] Docker container '${DOCKER_CONTAINER}' is running on worker"
    else
      echo "  [FAIL] Failed to start Docker container on worker"
      exit 1
    fi
  fi

  echo ""
  ((node++))
done < "$MACHINE_IP_FILE"

echo "=== All Docker containers started successfully ==="
echo ""
echo "Verify with:"
echo "  docker ps  # Check master"
while IFS= read -r IP || [[ -n "$IP" ]]; do
  if [[ -z "$IP" ]]; then continue; fi
  if [[ "$node" -gt 1 ]]; then
    echo "  ssh $USER@$IP 'docker ps'  # Check worker"
  fi
  ((node++))
done < "$MACHINE_IP_FILE"
echo ""
echo "Ready to launch training:"
echo "  ./scripts/multi_node/master_launch.sh --channels 28 --threads 256"
