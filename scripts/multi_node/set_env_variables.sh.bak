#!/bin/bash
# Global NCCL/RCCL environment variables for multi-node training
# Based on DLRM_set_env_variables.sh

# NCCL Debug Settings (use INFO for debugging network issues)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
# Try disabling IB if InfiniBand is not properly configured
export NCCL_IB_DISABLE=1

# IB/RNIC Configuration (commented out when IB is disabled)
# export NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7
# export NCCL_IB_GID_INDEX=3
export NCCL_NCHANNELS_PER_NET_PEER=8

# HSA Settings for ROCm
export HSA_ENABLE_IPC_MODE_LEGACY=1

# NCCL Protocol
export NCCL_PROTO=Simple

# Channel Configuration (can be overridden by sweep parameters)
export NCCL_MIN_NCHANNELS=40
export NCCL_MAX_NCHANNELS=40

# Network Interface
# Change this to match your network interface: eth0, ib0, enp49s0f0np0, etc.
# Temporarily commented out for auto-detection:
# export NCCL_SOCKET_IFNAME=enp193s0f0

# PyTorch ROCm Profiler
export PYTORCH_ROCM_PROFILER_ENABLE_TRACING=1

# Optional: Force non-overlap for debugging
# export GPU_MAX_HW_QUEUES=1
# unset TORCH_NCCL_HIGH_PRIORITY

# Optional: Disable SDMA for testing
# export HSA_ENABLE_SDMA=0

# Optional: Disable IB for Ethernet-only testing
# export NCCL_IB_DISABLE=1
