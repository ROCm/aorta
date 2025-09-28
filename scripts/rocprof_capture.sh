#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG=${1:-${REPO_ROOT}/config/default.yaml}
shift || true

OUT_ROOT=${ROCPROF_OUTPUT_DIR:-${REPO_ROOT}/rocprof_traces}
STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR=${OUT_ROOT}/run_${STAMP}
mkdir -p "${RUN_DIR}"

DEFAULT_ROCPROF_ARGS="${ROCPROF_ARGS:---att --kernel-trace}"
IFS=' ' read -r -a ROC_PROF_ARGS <<< "${DEFAULT_ROCPROF_ARGS}"

GPU_COUNT=$(python - <<'PY'
import os
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    visible = os.environ.get('HIP_VISIBLE_DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        print(len([v for v in visible.split(',') if v.strip()]))
    else:
        print(1)
PY
)

if [ "${GPU_COUNT}" -lt 1 ]; then
  GPU_COUNT=1
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

rocprofv3 "${ROC_PROF_ARGS[@]}" -d "${RUN_DIR}" -- \
  torchrun \
    --standalone \
    --nproc_per_node "${GPU_COUNT}" \
    "${REPO_ROOT}/train.py" \
    --config "${CONFIG}" \
    "$@"
