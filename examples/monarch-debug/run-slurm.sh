#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Attach Monarch to the Slurm allocation already in this shell and run the
# GPU all-reduce debug demo. Does not call salloc or sbatch.
#
#   bash examples/monarch-debug/run-slurm.sh 1   # step 2: one node, 8 GPUs
#   bash examples/monarch-debug/run-slurm.sh 2   # step 3: two nodes, 16 GPUs (default)
#
# Run it from a shell on one of the allocated nodes, not the login node. Use
# an allocation you already hold: find it with `squeue -u $USER` and
# export MONARCH_SLURM_JOB=<id>. Or hold one:
#
#   salloc -N2 --gpus-per-node=8     # covers steps 2 and 3; -N1 is enough for 2
#
# With 1 inside a two-node allocation, the demo uses the node you are on. A
# one-shot `srun hostname` does not leave SLURM_JOB_ID set. salloc does.
# Extra arguments go to gpu_reduce.py.
#
# MONARCH_PYTHON is the Python of the environment that has Monarch (default:
# `python` on PATH); see docs/monarch-install.md.
#
# A two-node all-reduce can stall in NCCL setup without the cluster's RCCL or
# NCCL network settings (NCCL_SOCKET_IFNAME, NCCL_IB_HCA, and so on). Put them
# in a file and set DEBUG_DEMO_NET_ENV to its path; this script sources it.
# Unset or empty skips it.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${MONARCH_PYTHON:-python}
NET_ENV=${DEBUG_DEMO_NET_ENV:-}
if [[ -n $NET_ENV ]]; then
  if [[ ! -f $NET_ENV ]]; then
    echo "DEBUG_DEMO_NET_ENV is '$NET_ENV', which is not a file." >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$NET_ENV"
fi
if [[ -z "${MONARCH_SLURM_JOB:-${SLURM_JOB_ID:-}}" ]]; then
  echo "No Slurm allocation in this shell. Set SLURM_JOB_ID or MONARCH_SLURM_JOB." >&2
  echo "If you already hold one, find it with \`squeue -u \$USER\` and run" >&2
  echo "  export MONARCH_SLURM_JOB=<id>" >&2
  echo "A shell opened by ssh into an allocated node has no SLURM_JOB_ID." >&2
  echo "A one-shot \`srun hostname\` does not leave SLURM_JOB_ID set either." >&2
  echo "Otherwise hold one with: salloc -N2 --gpus-per-node=8   (or -N1 for step 2)" >&2
  exit 2
fi
N=2
if [[ $# -gt 0 && $1 != -* ]]; then
  N=$1
  shift
fi
if [[ $N != 1 && $N != 2 ]]; then
  echo "Node count must be 1 or 2, got '$N'." >&2
  exit 2
fi
exec "$PYTHON" "$HERE/gpu_reduce.py" --nodes "$N" "$@"
