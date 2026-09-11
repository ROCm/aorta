#!/bin/bash
# Scout for a PUBLIC numerics reproducer: is there a configuration that
# genuinely goes non-finite, where a registered mitigation genuinely fixes it?
#
# Question zero is whether DISABLE_TF32 is even observed on this stack. If it
# is not, "tf32_off is the mitigation" is a claim about a knob that does
# nothing here, and every later cell would be two baselines with different
# labels. Nothing else in this file is worth reading until wiring answers.
#
# Deliberately NOT the CI runner smci350-rck-g03-f16-12: the ten-night perf
# baseline window is live through 17 Sep. meta64 contains zero smci350 nodes.
#SBATCH -p meta64
#SBATCH -N 1
#SBATCH -t 00:45:00
#SBATCH -J numerics-scout
#SBATCH -o /apps/vikhande/probe-gpu/logs/numscout-%j.out
#SBATCH -e /apps/vikhande/probe-gpu/logs/numscout-%j.err

set -uo pipefail

export TMPDIR=/apps/vikhande/tmp
export HF_HOME=/apps/vikhande/cache/hf
export PIP_CACHE_DIR=/apps/vikhande/cache/pip

ROOT=/apps/vikhande/probe-gpu
IMAGE=rocm/primus:v26.3
OUT="${ROOT}/scout-numerics"

echo "=== node $(hostname) job ${SLURM_JOB_ID} nodelist=${SLURM_JOB_NODELIST} $(date -u +%FT%TZ) ==="
scontrol show hostnames "${SLURM_JOB_NODELIST}"
if scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -qi smci350; then
  echo "FATAL: allocation contains an smci350 node -- refusing to run" >&2
  exit 90
fi

mkdir -p "${OUT}"
rm -f "${OUT}"/*.json

docker run --rm --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size=32g --security-opt seccomp=unconfined \
  -v "${ROOT}/wt:/work:ro" -v "${ROOT}:/out" -w /out \
  -e PYTHONPATH=/work/src \
  "${IMAGE}" bash -lc '
set -uo pipefail
python -c "import torch;print(\"torch\",torch.__version__,\"hip\",torch.version.hip,\"gpu\",torch.cuda.get_device_properties(0).gcnArchName)"

run () {  # <label> <mode> <env...>
  local label="$1"; shift
  local mode="$1"; shift
  echo
  echo "-------- ${label}  (mode=${mode})  env: $* --------"
  env "$@" python /out/scripts/numerics_probe.py "${mode}" 2>&1 | tail -12
}

echo "############ P1  is DISABLE_TF32 wired at all? ############"
# Four corners. If the two DISABLE_TF32 values give the same error at
# ALLOW_TF32=1, the env var is dead on this stack and tf32_off cannot be the
# resolving mitigation for anything.
run "allow=0 disable=unset" wiring ALLOW_TF32=0
run "allow=1 disable=unset" wiring ALLOW_TF32=1
run "allow=1 disable=1"     wiring ALLOW_TF32=1 DISABLE_TF32=1
run "allow=0 disable=1"     wiring ALLOW_TF32=0 DISABLE_TF32=1
# The knob PyTorch-ROCm documents, as a control for the one aorta registers.
run "allow=1 hipblaslt_tf32=0" wiring ALLOW_TF32=1 HIPBLASLT_ALLOW_TF32=0
run "allow=1 hipblaslt_tf32=1" wiring ALLOW_TF32=1 HIPBLASLT_ALLOW_TF32=1

echo
echo "############ P2  one-pass variance: where does precision break it? ############"
run "cancel allow=0"           cancel ALLOW_TF32=0
run "cancel allow=1"           cancel ALLOW_TF32=1
run "cancel allow=1 disable=1" cancel ALLOW_TF32=1 DISABLE_TF32=1

echo
echo "############ P3  fp16 attention: is the SDPA backend pin a numerics axis? ############"
run "attn default"  attn
run "attn prefer_ck"       attn TORCH_ROCM_FA_PREFER_CK=1
run "attn prefer_aotriton" attn TORCH_ROCM_FA_PREFER_CK=0

echo
echo "############ P4  aggressive-lr training: does precision decide divergence? ############"
run "train allow=0 lr=0.5"           train ALLOW_TF32=0 PROBE_LR=0.5
run "train allow=1 lr=0.5"           train ALLOW_TF32=1 PROBE_LR=0.5
run "train allow=1 lr=0.5 disable=1" train ALLOW_TF32=1 PROBE_LR=0.5 DISABLE_TF32=1
'

echo "=== done $(date -u +%FT%TZ) ==="
