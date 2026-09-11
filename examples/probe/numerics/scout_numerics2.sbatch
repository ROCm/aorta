#!/bin/bash
# Round 2. Round 1 found DISABLE_TF32 inert, so the question is no longer
# "which reproducer" but "does the mitigation registry contain any live lever
# over a non-finite outcome on this stack at all". Answering that needs a real
# non-finite mechanism and the whole registry run against it.
#SBATCH -p meta64
#SBATCH -N 1
#SBATCH -t 00:45:00
#SBATCH -J numerics-scout2
#SBATCH -o /apps/vikhande/probe-gpu/logs/numscout2-%j.out
#SBATCH -e /apps/vikhande/probe-gpu/logs/numscout2-%j.err

set -uo pipefail
export TMPDIR=/apps/vikhande/tmp
export HF_HOME=/apps/vikhande/cache/hf
export PIP_CACHE_DIR=/apps/vikhande/cache/pip

ROOT=/apps/vikhande/probe-gpu
IMAGE=rocm/primus:v26.3

echo "=== node $(hostname) job ${SLURM_JOB_ID} nodelist=${SLURM_JOB_NODELIST} $(date -u +%FT%TZ) ==="
scontrol show hostnames "${SLURM_JOB_NODELIST}"
if scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -qi smci350; then
  echo "FATAL: allocation contains an smci350 node -- refusing to run" >&2
  exit 90
fi

docker run --rm --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size=32g --security-opt seccomp=unconfined \
  -v "${ROOT}/wt:/work:ro" -v "${ROOT}:/out" -w /out \
  -e PYTHONPATH=/work/src \
  "${IMAGE}" bash -lc '
set -uo pipefail
P=/out/scripts/numerics_probe2.py

echo "############ C  is the tf32 setting engaging a reduced mantissa? ############"
python "$P" calib 2>&1 | grep -E "PROBE_JSON|Error"
echo "-- and the PyTorch-ROCm override env, as a control for DISABLE_TF32 --"
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 python "$P" calib 2>&1 | grep -E "PROBE_JSON|Error"

echo
echo "############ M  do the three mechanisms actually go non-finite? ############"
for m in fp16_overflow onepass_var uninit_read; do
  echo "---- ${m} (no mitigation) ----"
  python "$P" "$m" 2>&1 | grep -E "PROBE_JSON|loss=nan"; echo "exit=$?"
done

echo
echo "############ R  every registered mitigation against every failing mechanism ############"
# name:VAR=VAL, transcribed from src/aorta/registry/mitigations.py BUILTIN_MITIGATIONS.
MITS=(
  "none:AORTA_NOOP=1"
  "tf32_off:DISABLE_TF32=1"
  "xnack:HSA_XNACK=1"
  "gpu_max_hw_queues_2:GPU_MAX_HW_QUEUES=2"
  "debug_hip_dynamic_queues_1:DEBUG_HIP_DYNAMIC_QUEUES=1"
  "debug_hip_dynamic_queues_2:DEBUG_HIP_DYNAMIC_QUEUES=2"
  "roc_aql_queue_size_1024:ROC_AQL_QUEUE_SIZE=1024"
  "hsa_no_sdma:HSA_ENABLE_SDMA=0"
  "hsa_no_scratch_reclaim:HSA_NO_SCRATCH_RECLAIM=1"
  "hsa_disable_cache:HSA_DISABLE_CACHE=1"
  "hsa_enable_cache:HSA_DISABLE_CACHE=0"
  "roc_signal_pool_16k:ROC_SIGNAL_POOL_SIZE=16384"
  "gpu_force_blit_copy_128:GPU_FORCE_BLIT_COPY_SIZE=128"
  "debug_clr_no_batch_cpu_sync:DEBUG_CLR_BATCH_CPU_SYNC_SIZE=0"
  "nccl_launch_order_implicit:NCCL_LAUNCH_ORDER_IMPLICIT=1"
  "rccl_gfx942_cheap_fence_off:RCCL_GFX942_CHEAP_FENCE_OFF=1"
  "pytorch_no_cuda_memory_caching:PYTORCH_NO_CUDA_MEMORY_CACHING=1"
  "pytorch_alloc_expandable_segments:PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  "hip_launch_blocking:HIP_LAUNCH_BLOCKING=1"
  "fa_prefer_ck:TORCH_ROCM_FA_PREFER_CK=1"
  "fa_prefer_aotriton:TORCH_ROCM_FA_PREFER_CK=0"
)
# amd_log_level_4 omitted from the sweep: it only changes log volume and costs
# minutes of output. Recorded as untested-by-choice rather than silently.

for m in fp16_overflow onepass_var uninit_read; do
  echo
  echo "======== mechanism ${m} ========"
  for entry in "${MITS[@]}"; do
    name="${entry%%:*}"; kv="${entry#*:}"
    out=$(env "${kv}" python "$P" "$m" 2>&1)
    rc=$?
    verdict=$(echo "$out" | grep -o "\"finite\": [a-z]*" | head -1)
    printf "%-38s rc=%s  %s\n" "${name}" "${rc}" "${verdict:-<no json>}"
  done
done
'
echo "=== done $(date -u +%FT%TZ) ==="
