#!/bin/bash
# Start Docker containers on all nodes for multi-node training
# Usage: ./start_docker_all_nodes.sh [docker-compose-file] [container-name]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AORTA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MACHINE_IP_FILE="$SCRIPT_DIR/node_ip_list.txt"  # Contains hostnames or IPs

# Allow custom docker-compose file and container name via arguments
DOCKER_COMPOSE_FILE="${1:-docker/docker-compose.rocm70_9-1.yaml}"
DOCKER_CONTAINER="${2:-training-overlap-bugs-rocm70_9-1}"

if [[ ! -f "$MACHINE_IP_FILE" ]]; then
    echo "Error: $MACHINE_IP_FILE not found"
    echo "Run setup_multi_node.sh first"
    exit 1
fi

cd "$AORTA_ROOT"

# Check git branch consistency before starting Docker
echo "=== Checking git branch consistency ==="
MASTER_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "not-a-git-repo")

if [[ "$MASTER_BRANCH" != "not-a-git-repo" ]]; then
    echo "Master node branch: $MASTER_BRANCH"
    TOTAL_NODES=$(grep -c . "$MACHINE_IP_FILE" || echo "0")
    echo "Found $TOTAL_NODES nodes in $MACHINE_IP_FILE"
    echo ""

    node=0
    while IFS= read -r HOST || [[ -n "$HOST" ]]; do  # HOST can be hostname or IP
        # Skip empty lines
        if [[ -z "$HOST" ]]; then
            continue
        fi

        if [[ "$node" -gt 0 ]]; then
            echo "[STAGE] Checking worker node $node ($HOST)..."
            WORKER_BRANCH=$(ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$HOST" "cd ~/aorta && git rev-parse --abbrev-ref HEAD 2>/dev/null" || echo "not-a-git-repo")

            if [[ "$WORKER_BRANCH" == "not-a-git-repo" ]]; then
                echo "  [WARN] Worker node $HOST: Not a git repository"
            elif [[ "$MASTER_BRANCH" != "$WORKER_BRANCH" ]]; then
                echo "  [ERROR] Branch mismatch on node $HOST!"
                echo "  Master: $MASTER_BRANCH"
                echo "  Worker: $WORKER_BRANCH"
                echo ""
                echo "Fix: ssh $USER@$HOST 'cd ~/aorta && git checkout $MASTER_BRANCH && git pull'"
                exit 1
            else
                echo "  Worker node $HOST: $WORKER_BRANCH [OK]"
            fi
        fi
        ((node++)) || true
    done < "$MACHINE_IP_FILE"
    echo ""
    echo "Branch check complete [OK]"
    echo ""
else
    echo "[WARN] Not a git repository - skipping branch check"
    echo ""
fi

echo "=== Starting Docker containers on all nodes ==="
TOTAL_NODES=$(wc -l < "$MACHINE_IP_FILE")
echo "Total nodes to process: $TOTAL_NODES"
echo ""

node=0
while IFS= read -r HOST || [[ -n "$HOST" ]]; do  # HOST can be hostname or IP
  if [[ -z "$HOST" ]]; then
    continue
  fi

  echo "Node $node (Host: $HOST):"

  if [[ "$node" -eq 0 ]]; then
    # Master node (local)
    echo "  [STAGE] Checking existing containers on master..."
    if docker ps --format '{{.Names}}' < /dev/null | grep -q "^${DOCKER_CONTAINER}$"; then
      echo "  [INFO] Container already running, restarting..."
    fi

    echo "  [STAGE] Running docker compose up -d on master..."
    COMPOSE_FILE_PATH="${DOCKER_COMPOSE_FILE#docker/}"
    cd docker && docker compose -f "$COMPOSE_FILE_PATH" up -d < /dev/null && cd ..

    echo "  [STAGE] Verifying master container..."
    if docker ps --format '{{.Names}}' < /dev/null | grep -q "^${DOCKER_CONTAINER}$"; then
      echo "  [OK] Docker container '${DOCKER_CONTAINER}' is running"
    else
      echo "  [FAIL] Failed to start Docker container"
      exit 1
    fi
  else
    # Worker nodes (via SSH)
    echo "  [STAGE] Connecting to worker via SSH..."
    if ! ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$HOST" "echo 'SSH connection successful'" > /dev/null 2>&1; then
      echo "  [FAIL] Cannot SSH to worker node $HOST"
      exit 1
    fi
    echo "  [OK] SSH connection successful"

    echo "  [STAGE] Checking existing containers on worker..."
    if ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$HOST" "docker ps --format '{{.Names}}'" | grep -q "^${DOCKER_CONTAINER}$"; then
      echo "  [INFO] Container already running, restarting..."
    fi

    echo "  [STAGE] Running docker compose up -d on worker..."
    COMPOSE_FILE_PATH="${DOCKER_COMPOSE_FILE#docker/}"
    ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$HOST" \
      "cd /home/$USER/aorta/docker && docker compose -f $COMPOSE_FILE_PATH up -d"

    echo "  [STAGE] Verifying worker container..."
    if ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER@$HOST" "docker ps --format '{{.Names}}'" | grep -q "^${DOCKER_CONTAINER}$"; then
      echo "  [OK] Docker container '${DOCKER_CONTAINER}' is running on worker"
    else
      echo "  [FAIL] Failed to start Docker container on worker"
      exit 1
    fi
  fi

  echo ""
  ((node++)) || true
done < "$MACHINE_IP_FILE"

echo "=== All Docker containers started successfully ==="
echo "Docker container: $DOCKER_CONTAINER"
echo ""
echo "Verify with:"
echo "  docker ps  # Check master"
while IFS= read -r HOST || [[ -n "$HOST" ]]; do
  if [[ -z "$HOST" ]]; then continue; fi
  if [[ "$node" -gt 1 ]]; then
    echo "  ssh $USER@$HOST 'docker ps'  # Check worker"
  fi
  ((node++))
done < "$MACHINE_IP_FILE"
echo ""
echo "Ready to launch training:"
echo "  ./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --config <your-config>.yaml"
