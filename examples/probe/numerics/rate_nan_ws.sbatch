#!/bin/bash
# Characterise the reproducer before wiring it into aorta: is it deterministic
# or rate-based, on both sides of the mitigation? A resolve signal measured
# once is an anecdote; the fix-half reward treats "this mitigation resolves the
# scenario" as ground truth, so the rate is what makes the ground truth mean
# anything.
#SBATCH -p meta64
#SBATCH -N 1
#SBATCH -t 00:40:00
#SBATCH -J nanws-rate
#SBATCH -o /apps/vikhande/probe-gpu/logs/nanws-rate-%j.out
#SBATCH -e /apps/vikhande/probe-gpu/logs/nanws-rate-%j.err

set -uo pipefail
export TMPDIR=/apps/vikhande/tmp HF_HOME=/apps/vikhande/cache/hf PIP_CACHE_DIR=/apps/vikhande/cache/pip
ROOT=/apps/vikhande/probe-gpu
LOG="${TMPDIR}/nanws-rate-${SLURM_JOB_ID}.txt"

echo "=== node $(hostname) job ${SLURM_JOB_ID} nodelist=${SLURM_JOB_NODELIST} $(date -u +%FT%TZ) ==="
scontrol show hostnames "${SLURM_JOB_NODELIST}"
if scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -qi smci350; then
  echo "FATAL: allocation contains an smci350 node -- refusing to run" >&2
  exit 90
fi

run_one () {  # <trial> <extra -e args...>
  local trial="$1"; shift
  docker run --rm --network=host --device=/dev/kfd --device=/dev/dri \
    --group-add video --ipc=host --shm-size=32g --security-opt seccomp=unconfined \
    -v "${ROOT}:/out" -w /out -e AORTA_TRIAL="${trial}" "$@" \
    rocm/primus:v26.3 python /out/scripts/nan_workspace.py
}

N=${N:-20}
for cond in baseline resolver; do
  echo
  echo "======== ${cond}: ${N} trials ========"
  fail=0; ok=0
  for i in $(seq 1 "${N}"); do
    if [ "${cond}" = "resolver" ]; then
      run_one "${i}" -e PYTORCH_NO_CUDA_MEMORY_CACHING=1 > "${LOG}" 2>&1
    else
      run_one "${i}" > "${LOG}" 2>&1
    fi
    rc=$?
    if [ "${rc}" -eq 0 ]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    grep -h "NAN_WS_JSON" "${LOG}" | sed "s/^/[${cond} ${i} rc=${rc}] /"
    if [ "${i}" -eq 1 ]; then grep -h "phase 1: fp16 peaks\|phase 1: freed" "${LOG}"; fi
  done
  echo "---- ${cond}: pass=${ok} fail=${fail} of ${N} ----"
done
rm -f "${LOG}"
echo "=== done $(date -u +%FT%TZ) ==="
