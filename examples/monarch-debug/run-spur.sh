#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Attach Monarch to the Spur allocation already in this shell and run the
# GPU all-reduce debug demo. Does not call `spur salloc`.
#
#   bash examples/monarch-debug/run-spur.sh 1   # step 2: one node, 8 GPUs
#   bash examples/monarch-debug/run-spur.sh 2   # step 3: two nodes, 16 GPUs (default)
#
# On one of the machines Spur gave you:
#
#   export SPUR_CONTROLLER_ADDR=http://<controller IPv4>:6817
#   spur salloc -N2 --gpus-per-node=8 --exclusive -t 02:00:00
#   bash examples/monarch-debug/run-spur.sh 2
#
# SPUR_JOB_ID is set by that shell. DEBUG_DEMO_SHARED_DIR (or
# AORTA_SHARED_DIR) must be visible on every node; the default below is a
# `shared` directory next to this script, which works when the checkout is on
# a shared filesystem. SPUR_BIN may name the spur binary or its directory; see
# docs/spur-install.md. Extra arguments go to gpu_reduce.py. MONARCH_PYTHON
# and DEBUG_DEMO_NET_ENV work as in run-slurm.sh.
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
if [[ -z "${SPUR_JOB_ID:-}" ]]; then
  echo "No Spur allocation in this shell. SPUR_JOB_ID is not set." >&2
  echo "\`spur salloc\` opens that shell; this script does not request an allocation." >&2
  echo "If you already hold one, find it with \`spur squeue --me\` and run" >&2
  echo "  export SPUR_JOB_ID=<id>" >&2
  echo "Otherwise start one with: spur salloc -N2 --gpus-per-node=8 --exclusive   (or -N1 for step 2)" >&2
  exit 2
fi
export DEBUG_DEMO_SHARED_DIR="${DEBUG_DEMO_SHARED_DIR:-${AORTA_SHARED_DIR:-$HERE/shared}}"
mkdir -p "$DEBUG_DEMO_SHARED_DIR"
SPUR_CLI=spur
for candidate in "${SPUR_BIN:-}" "${SPUR_BIN:-}/spur" "${SPUR_HOME:-}/bin/spur"; do
  if [[ -f $candidate && -x $candidate ]]; then
    SPUR_CLI=$candidate
    break
  fi
done
N=2
if [[ $# -gt 0 && $1 != -* ]]; then
  N=$1
  shift
fi
if [[ $N != 1 && $N != 2 ]]; then
  echo "Node count must be 1 or 2, got '$N'." >&2
  exit 2
fi
exec "$PYTHON" "$HERE/gpu_reduce.py" \
  --scheduler spur --nodes "$N" --spur-bin "$SPUR_CLI" "$@"
