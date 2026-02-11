#!/bin/bash
# =============================================================================
# Global NCCL/RCCL environment variables for multi-node training
# Configured for MI350X cluster
#
# This file is the SINGLE SOURCE OF TRUTH for all NCCL/RCCL configuration.
# Edit variables here - local_launch.sh will automatically pick them up.
#
# NOTE: When adding a new environment variable, you MUST also add its name
#       to the DOCKER_ENV_VARS array below, otherwise it won't be passed
#       to the Docker container.
# =============================================================================

# -----------------------------------------------------------------------------
# NCCL Debug Settings
# -----------------------------------------------------------------------------
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=                    # Options: COLL,INIT,NET (empty = none)

# -----------------------------------------------------------------------------
# RCCL-Specific Settings (ROCm)
# -----------------------------------------------------------------------------
export RCCL_DIRECT_ALLGATHER_DISABLE=1       # Disable direct allgather
export RCCL_MSCCL_ENABLE=0                   # Disable MSCCL
export RCCL_THREADS_PER_BLOCK=256            # Threads per block (override via --threads)

# -----------------------------------------------------------------------------
# IB/RNIC Configuration for MI350X
# -----------------------------------------------------------------------------
export NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7
export NCCL_IB_GID_INDEX=3
export NCCL_NCHANNELS_PER_NET_PEER=8

# -----------------------------------------------------------------------------
# HSA Settings for ROCm
# -----------------------------------------------------------------------------
export HSA_ENABLE_IPC_MODE_LEGACY=1
export HSA_ENABLE_SDMA=0                     # Disable SDMA for stability

# -----------------------------------------------------------------------------
# NCCL Protocol and Channels
# -----------------------------------------------------------------------------
export NCCL_PROTO=Simple
#export NCCL_MIN_NCHANNELS=40
export NCCL_MAX_NCHANNELS=56                 # Override via --channels

# -----------------------------------------------------------------------------
# Network Interface for MI350X cluster
# -----------------------------------------------------------------------------
export NCCL_SOCKET_IFNAME=enp49s0f0np0,fenic0

# -----------------------------------------------------------------------------
# Timeout and Error Handling
# -----------------------------------------------------------------------------
export NCCL_TIMEOUT_MS=12000                 # 12 second timeout (legacy, not used by PyTorch)
export NCCL_TIMEOUT=100                      # 100 second timeout - first backward can be slow due to JIT/init
export TORCH_DIST_INIT_TIMEOUT=150           # Match collective timeout for consistency
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=10000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1          # Critical for hang debugging
#export AMD_LOG_LEVEL=5
# AMD_LOG_LEVEL_FILE is set dynamically in local_launch.sh to point to experiment directory
# Default fallback (will be overridden):
#export AMD_LOG_LEVEL_FILE=trace_amd.log
# -----------------------------------------------------------------------------
# PyTorch ROCm Profiler
# -----------------------------------------------------------------------------
export PYTORCH_ROCM_PROFILER_ENABLE_TRACING=1

# -----------------------------------------------------------------------------
# List of environment variables to pass to Docker container
# Add/remove variables here to control what gets passed through
# -----------------------------------------------------------------------------
DOCKER_ENV_VARS=(
    # NCCL Debug
    NCCL_DEBUG
    NCCL_DEBUG_SUBSYS
    # RCCL
    RCCL_DIRECT_ALLGATHER_DISABLE
    RCCL_MSCCL_ENABLE
    RCCL_THREADS_PER_BLOCK
    # IB/RNIC
    NCCL_IB_HCA
    NCCL_IB_GID_INDEX
    NCCL_NCHANNELS_PER_NET_PEER
    # HSA
    HSA_ENABLE_IPC_MODE_LEGACY
    HSA_ENABLE_SDMA
    # Protocol/Channels
    NCCL_PROTO
    NCCL_MIN_NCHANNELS
    NCCL_MAX_NCHANNELS
    # Network
    NCCL_SOCKET_IFNAME
    # Timeout/Error Handling
    NCCL_TIMEOUT_MS
    NCCL_TIMEOUT
    TORCH_DIST_INIT_TIMEOUT
    TORCH_NCCL_ASYNC_ERROR_HANDLING
    TORCH_NCCL_TRACE_BUFFER_SIZE
    TORCH_NCCL_DUMP_ON_TIMEOUT
    # AMD Logging
    AMD_LOG_LEVEL
    AMD_LOG_LEVEL_FILE
    # Profiler
    PYTORCH_ROCM_PROFILER_ENABLE_TRACING
)
export DOCKER_ENV_VARS

# -----------------------------------------------------------------------------
# Helper function: Build docker -e flags from DOCKER_ENV_VARS
# Usage: DOCKER_ENV_FLAGS=$(build_docker_env_flags)
# -----------------------------------------------------------------------------
build_docker_env_flags() {
    local flags=""
    for var in "${DOCKER_ENV_VARS[@]}"; do
        local value="${!var}"
        flags+=" -e ${var}=${value}"
    done
    echo "$flags"
}
export -f build_docker_env_flags

# =============================================================================
# Optional settings (uncomment to enable)
# =============================================================================

# Force non-overlap for debugging (single HW queue)
# export GPU_MAX_HW_QUEUES=1
# unset TORCH_NCCL_HIGH_PRIORITY

# Disable IB for Ethernet-only testing
# export NCCL_IB_DISABLE=1
