#!/bin/bash
# The archiving run for the uninitialised-workspace NaN scenario.
#
#   A  agent  -- `aorta agent mitigate`, fake backend. The baseline FAILS here,
#                which is the condition the previous run lacked: the loop
#                actually searches, so agent_log.jsonl carries real llm_step
#                events instead of a single baseline-pass short circuit.
#   B  matrix -- `aorta sweep run`, full 8-cell axis x 4 trials. This is the
#                artifact the fix-half reward reads: resolution_from_matrix
#                recovers the resolvers from archived cell verdicts, so the
#                matrix has to contain the decoys too, not just the winner.
#   C  fp16   -- the companion scenario nothing resolves, for the withhold path.
#
# Not the CI runner: the ten-night perf baseline window is live through 17 Sep
# and meta64 contains zero smci350 nodes.
#SBATCH -p meta64
#SBATCH -N 1
#SBATCH -t 00:55:00
#SBATCH -J nan-probe
#SBATCH -o /apps/vikhande/probe-gpu/logs/nan-probe-%j.out
#SBATCH -e /apps/vikhande/probe-gpu/logs/nan-probe-%j.err

set -uo pipefail
export TMPDIR=/apps/vikhande/tmp HF_HOME=/apps/vikhande/cache/hf PIP_CACHE_DIR=/apps/vikhande/cache/pip

ROOT=/apps/vikhande/probe-gpu
WT=/apps/vikhande/probe-gpu/wt-num
VENV=/apps/vikhande/probe-gpu/venv-num
R_WS="${WT}/recipes/probe/probe-numerics-uninit-workspace.yaml"
R_FP16="${WT}/recipes/probe/probe-numerics-fp16-overflow.yaml"

step () { echo; echo "[$(date -u +%T)] ======== $* ========"; }

step "node $(hostname) job ${SLURM_JOB_ID} nodelist=${SLURM_JOB_NODELIST}"
scontrol show hostnames "${SLURM_JOB_NODELIST}"
if scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -qi smci350; then
  echo "FATAL: allocation contains an smci350 node -- refusing to run" >&2
  exit 90
fi

step "python env for the aorta CLI (CPU only; the GPU work is in the container)"
if [ ! -x "${VENV}/bin/aorta" ]; then
  python3.11 -m venv "${VENV}" || exit 1
  "${VENV}/bin/python" -m pip install -q --upgrade pip || exit 1
  "${VENV}/bin/python" -m pip install -q -e "${WT}" || exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
aorta --version 2>/dev/null || true
cd "${WT}" || exit 1

# ---------------------------------------------------------------------------
step "PHASE A -- aorta agent mitigate (baseline fails, so the loop searches)"
# ---------------------------------------------------------------------------
rm -rf "${ROOT}/nan_agent"
unset NAN_WS_MODE
aorta agent mitigate \
  --recipe "${R_WS}" \
  --output "${ROOT}/nan_agent" \
  --ticket PROBE-NAN-WS \
  --symptom "loss goes NaN in a microbatch accumulation workspace; an earlier unscaled fp16 phase overflowed, and the NaN appears in a tensor no numeric op touched" \
  --llm-backend fake \
  --max-iterations 8 \
  -vv \
  -- "${ROOT}/scripts/launch_nan_ws.sh"
echo "agent exit=$?"

step "PHASE A -- agent_log.jsonl"
cat "${ROOT}/nan_agent/PROBE-NAN-WS/agent_log.jsonl" 2>/dev/null

# ---------------------------------------------------------------------------
step "PHASE B -- aorta sweep run, full 8-cell mitigation axis x 4 trials"
# ---------------------------------------------------------------------------
rm -rf "${ROOT}/nan_matrix"
aorta sweep run \
  --recipe "${R_WS}" \
  --output-dir "${ROOT}/nan_matrix" \
  --ticket PROBE-NAN-WS-MATRIX \
  -v \
  -- "${ROOT}/scripts/launch_nan_ws.sh"
echo "sweep exit=$?"

# ---------------------------------------------------------------------------
step "PHASE C -- the companion scenario nothing resolves (fp16 overflow)"
# ---------------------------------------------------------------------------
rm -rf "${ROOT}/nan_fp16"
export NAN_WS_MODE=overflow_only
aorta sweep run \
  --recipe "${R_FP16}" \
  --output-dir "${ROOT}/nan_fp16" \
  --ticket PROBE-NAN-FP16 \
  -v \
  -- "${ROOT}/scripts/launch_nan_ws.sh"
echo "sweep(fp16) exit=$?"
unset NAN_WS_MODE

# ---------------------------------------------------------------------------
step "VERDICTS -- per cell, with the mitigation env each trial actually recorded"
# ---------------------------------------------------------------------------
python - <<'PY'
import json, pathlib
for tree in ("nan_agent", "nan_matrix", "nan_fp16"):
    root = pathlib.Path("/apps/vikhande/probe-gpu") / tree
    print(f"\n---- {tree} ----")
    for res in sorted(root.rglob("trial_*/result.json")):
        d = json.loads(res.read_text())
        env = d.get("env") or {}
        # Only the keys a mitigation could have set; the rest is image noise.
        keys = {k: v for k, v in env.items() if k in {
            "DISABLE_TF32", "PYTORCH_NO_CUDA_MEMORY_CACHING", "HSA_ENABLE_SDMA",
            "HIP_LAUNCH_BLOCKING", "HSA_XNACK", "PYTORCH_CUDA_ALLOC_CONF",
            "GPU_MAX_HW_QUEUES"}}
        print(f"{res.parent.parent.name:44s} {res.parent.name:9s} "
              f"verdict={d.get('verdict'):8s} det={d.get('failure_detectors_fired')} env={keys}")
PY

echo "=== done $(date -u +%FT%TZ) ==="
