#!/bin/bash
#
# Interactive script to generate .env file for Docker Compose
#
# Usage:
#   bash setup-env.sh
#
# This script will guide you through creating a .env file with your
# preferred Dockerfile, container name, and volume mount configurations.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Docker Environment Setup for Aorta  ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if .env already exists
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Warning: .env file already exists at: $ENV_FILE${NC}"
    read -p "Do you want to overwrite it? (y/N): " overwrite
    if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted. Existing .env file kept.${NC}"
        exit 0
    fi
    echo ""
fi

# Step 1: Select Dockerfile
echo -e "${GREEN}Step 1: Select Dockerfile${NC}"
echo "Available Dockerfiles:"
echo "  1) Dockerfile.rocm70_9-1              - Standard ROCm 7.0.9.1 build"
echo "  2) Dockerfile.rocm70_9-1-shampoo      - ROCm 7.0.9.1 with Shampoo optimizer"
echo "  3) Dockerfile.rocm70_2-ubuntu-pytorch - ROCm 7.0.2 Ubuntu PyTorch build"
echo "  4) Dockerfile.rocm70_2-ubuntu-nan     - ROCm 7.0.2 with NaN debugging tools"
echo "  5) Dockerfile.rocm-ubuntu-ebpf        - ROCm 7.2 with eBPF tracing tools (bpftrace, bcc)"
echo ""

while true; do
    read -p "Enter choice [1-5]: " dockerfile_choice
    case $dockerfile_choice in
        1)
            DOCKERFILE="Dockerfile.rocm70_9-1"
            VARIANT="rocm70_9-1"
            break
            ;;
        2)
            DOCKERFILE="Dockerfile.rocm70_9-1-shampoo"
            VARIANT="rocm70_9-1-shampoo"
            break
            ;;
        3)
            DOCKERFILE="Dockerfile.rocm70_2-ubuntu-pytorch"
            VARIANT="rocm70_2-ubuntu-pytorch"
            break
            ;;
        4)
            DOCKERFILE="Dockerfile.rocm70_2-ubuntu-nan"
            VARIANT="rocm70_2-ubuntu-nan"
            break
            ;;
        5)
            DOCKERFILE="Dockerfile.rocm-ubuntu-ebpf"
            VARIANT="rocm-ubuntu-ebpf"
            break
            ;;
        *)
            echo -e "${RED}Invalid choice. Please enter 1-5.${NC}"
            ;;
    esac
done

echo -e "${GREEN}Selected: $DOCKERFILE${NC}"
echo ""

# Step 2: Image and Container Names
echo -e "${GREEN}Step 2: Image Name${NC}"
DEFAULT_IMAGE_NAME="aorta:${VARIANT}"
echo "Suggested default: $DEFAULT_IMAGE_NAME"
read -p "Enter image name (or press Enter to use default): " IMAGE_NAME
IMAGE_NAME=${IMAGE_NAME:-$DEFAULT_IMAGE_NAME}
echo -e "${GREEN}Image name: $IMAGE_NAME${NC}"
echo ""

echo -e "${GREEN}Step 3: Container Name${NC}"
DEFAULT_CONTAINER_NAME="${USER:-default}-${VARIANT}-$(date +%Y%m%d)"
echo "Suggested default: $DEFAULT_CONTAINER_NAME"
read -p "Enter container name (or press Enter to use default): " CONTAINER_NAME
CONTAINER_NAME=${CONTAINER_NAME:-$DEFAULT_CONTAINER_NAME}
echo -e "${GREEN}Container name: $CONTAINER_NAME${NC}"
echo ""

# Step 4: Aorta Workspace Path
echo -e "${GREEN}Step 4: Aorta Workspace Path${NC}"
echo "This is the path to your aorta workspace directory."
echo "Default: .. (parent directory - assumes you're running from docker/ subdirectory)"
read -p "Enter path (or press Enter for default '..'): " AORTA_WORKSPACE
AORTA_WORKSPACE=${AORTA_WORKSPACE:-..}
echo -e "${GREEN}Aorta workspace: $AORTA_WORKSPACE${NC}"
echo ""

# Step 5: RCCL Path (optional)
echo -e "${GREEN}Step 5: Custom RCCL Build (Optional)${NC}"
echo "Path to your custom RCCL build directory."
echo "Press Enter to use the RCCL bundled in the image (no custom build)."
read -p "Enter path (or press Enter to skip): " RCCL_PATH
if [[ -n "$RCCL_PATH" ]]; then
    echo -e "${GREEN}RCCL path: $RCCL_PATH (use -f docker-compose.rccl.yaml when starting)${NC}"
else
    echo -e "${GREEN}Using image RCCL (no custom path)${NC}"
fi
echo ""

# Step 6: AMD GPU Driver Variant (optional)
echo -e "${GREEN}Step 6: AMD GPU Driver Variant (Optional)${NC}"
echo "Set driver variant for environment_info.json"
echo "Possible values: patched, base, mqd_vram, default, or leave empty"
read -p "Enter variant (or press Enter to skip): " AMDGPU_DRIVER_VARIANT
echo ""

# Step 7: Additional mounts (optional)
echo -e "${GREEN}Step 7: Additional Volume Mounts (Optional)${NC}"
echo "Do you need any additional volume mounts?"
read -p "Add extra mounts? (y/N): " add_mounts

EXTRA_MOUNTS=""
if [[ "$add_mounts" =~ ^[Yy]$ ]]; then
    mount_count=1
    while true; do
        echo ""
        echo "Extra mount #$mount_count:"
        read -p "  Source path (host): " mount_src
        if [ -z "$mount_src" ]; then
            break
        fi
        read -p "  Destination path (container): " mount_dst
        if [ -z "$mount_dst" ]; then
            break
        fi

        EXTRA_MOUNTS+="EXTRA_MOUNT_SRC_${mount_count}=${mount_src}"$'\n'
        EXTRA_MOUNTS+="EXTRA_MOUNT_DST_${mount_count}=${mount_dst}"$'\n'

        mount_count=$((mount_count + 1))
        read -p "Add another mount? (y/N): " add_another
        if [[ ! "$add_another" =~ ^[Yy]$ ]]; then
            break
        fi
    done
fi

# Generate .env file
echo ""
echo -e "${BLUE}Generating .env file...${NC}"

cat > "$ENV_FILE" << EOF
# Docker Compose Environment Configuration
# Generated by setup-env.sh on $(date)

# ==============================================================================
# DOCKERFILE SELECTION
# ==============================================================================
DOCKERFILE=$DOCKERFILE

# ==============================================================================
# IMAGE NAME
# ==============================================================================
IMAGE_NAME=$IMAGE_NAME

# ==============================================================================
# CONTAINER NAME
# ==============================================================================
CONTAINER_NAME=$CONTAINER_NAME

# ==============================================================================
# VOLUME MOUNTS
# ==============================================================================
AORTA_WORKSPACE=$AORTA_WORKSPACE
# RCCL: leave unset to use image RCCL. To use custom RCCL, set RCCL_PATH and run with -f docker-compose.rccl.yaml

EOF
if [[ -n "$RCCL_PATH" ]]; then
    echo "RCCL_PATH=$RCCL_PATH" >> "$ENV_FILE"
fi
echo "" >> "$ENV_FILE"

# Add extra mounts if any
if [ -n "$EXTRA_MOUNTS" ]; then
    echo "# Additional volume mounts" >> "$ENV_FILE"
    echo "$EXTRA_MOUNTS" >> "$ENV_FILE"
fi

# Add driver variant if set
cat >> "$ENV_FILE" << EOF
# ==============================================================================
# ENVIRONMENT VARIABLES
# ==============================================================================
AMDGPU_DRIVER_VARIANT=$AMDGPU_DRIVER_VARIANT
EOF

echo -e "${GREEN}✓ Successfully created .env file at: $ENV_FILE${NC}"
echo ""
if [[ -n "$EXTRA_MOUNTS" ]]; then
    echo -e "${YELLOW}You configured extra volume mounts.${NC}"
    echo -e "${YELLOW}docker-compose.build.yaml currently supports up to two extra mounts via${NC}"
    echo -e "${YELLOW}EXTRA_MOUNT_SRC_1 / EXTRA_MOUNT_DST_1 and OPTIONAL EXTRA_MOUNT_SRC_2 / EXTRA_MOUNT_DST_2.${NC}"
    echo -e "${YELLOW}If you configured more than two extra mounts, you will need to manually${NC}"
    echo -e "${YELLOW}extend docker-compose.build.yaml to use EXTRA_MOUNT_SRC_3 / DST_3, etc.${NC}"
    echo ""
fi
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "You can now run Docker Compose with:"
echo -e "  ${YELLOW}cd ${SCRIPT_DIR}${NC}"
echo -e "  ${YELLOW}docker compose -f docker-compose.build.yaml up${NC}"
echo ""
echo "To view your configuration:"
echo -e "  ${YELLOW}cat ${ENV_FILE}${NC}"
echo ""
echo "To modify settings later, edit the .env file directly or re-run this script."
