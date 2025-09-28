#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG=${1:-${REPO_ROOT}/config/default.yaml}

GPU_COUNT=$(python - <<'PY'
import os
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    visible = os.environ.get('HIP_VISIBLE_DEVICES')
    if visible:
        print(len([v for v in visible.split(',') if v.strip()]))
    else:
        print(0)
PY
)

GPU_COUNT=${GPU_COUNT:-0}
if [ "${GPU_COUNT}" -le 0 ]; then
  if command -v rocminfo >/dev/null 2>&1; then
    GPU_COUNT=$(rocminfo | grep -c "Name: " || echo 1)
  else
    GPU_COUNT=${GPU_COUNT:-1}
  fi
fi

if [ "${GPU_COUNT}" -lt 1 ]; then
  GPU_COUNT=1
fi

NPROC=${NPROC:-${GPU_COUNT}}
if [ "${NPROC}" -lt 1 ]; then
  NPROC=1
fi

if [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
  HIP_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC - 1)))
fi
export HIP_VISIBLE_DEVICES
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

torchrun \
  --nproc_per_node "${NPROC}" \
  --standalone \
  "${REPO_ROOT}/train.py" \
  --config "${CONFIG}" \
  --enable-rocm-metrics
