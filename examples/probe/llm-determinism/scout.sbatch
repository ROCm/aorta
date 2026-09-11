#!/bin/bash
# Scout: does any SHIPPED llm_determinism configuration diverge on this node,
# and does DISABLE_TF32=1 change the answer?
#
# This job answers one question and writes no corpus: is there a real
# execution-checked resolve signal available here, or only a control? Nothing
# is perturbed or injected -- every cell below is a configuration the committed
# recipe or the documented knob table already supports.
#
# Deliberately NOT the CI runner smci350-rck-g03-f16-12: the ten-night perf
# baseline window is live through 17 Sep. meta64 contains zero smci350 nodes.
#SBATCH -p meta64
#SBATCH -N 1
#SBATCH -w cv350-rck-g03-c16-18
#SBATCH -t 00:40:00
#SBATCH -J probe-gpu-scout
#SBATCH -o /apps/vikhande/probe-gpu/logs/scout-%j.out
#SBATCH -e /apps/vikhande/probe-gpu/logs/scout-%j.err

set -uo pipefail

export TMPDIR=/apps/vikhande/tmp
export HF_HOME=/apps/vikhande/cache/hf
export PIP_CACHE_DIR=/apps/vikhande/cache/pip

# The docker daemon on the compute node cannot resolve /home/vikhande (autofs
# NFS), so the repo is mounted from an /apps copy kept in sync by rsync. The
# git worktree stays the source of truth; this copy is execution-only.
WT=/apps/vikhande/probe-gpu/wt
ROOT=/apps/vikhande/probe-gpu
IMAGE=rocm/primus:v26.3

echo "=== node $(hostname) job ${SLURM_JOB_ID} nodelist=${SLURM_JOB_NODELIST} started $(date -u +%FT%TZ) ==="
echo "=== guardrail check: allocation must contain no smci350 node ==="
scontrol show hostnames "${SLURM_JOB_NODELIST}"
if scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -qi smci350; then
  echo "FATAL: allocation contains an smci350 node -- refusing to run" >&2
  exit 90
fi

mkdir -p "${ROOT}/scout" "${ROOT}/logs"
rm -rf "${ROOT}/scout"/*

docker run --rm --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size=32g --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "${WT}:/work:ro" -v "${ROOT}:/out" \
  -w /out \
  -e PYTHONPATH=/work/src \
  "${IMAGE}" bash -lc '
set -uo pipefail
echo "--- torch ---"
python -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.device_count())"
python -c "import torch;print(torch.cuda.get_device_properties(0).gcnArchName)"

run_cell () {   # <name> <json-cfg> <extra-env...>
  local name="$1"; shift
  local cfg="$1"; shift
  echo
  echo "======== CELL ${name} ========  extra_env: $* "
  mkdir -p "/out/scout/${name}"
  local t0 t1
  t0=$(date +%s)
  env "$@" DET_CFG="${cfg}" \
    torchrun --standalone --nproc_per_node=8 /out/scripts/det_launch.py \
    > "/out/scout/${name}/stdout.log" 2> "/out/scout/${name}/stderr.log"
  local rc=$?
  t1=$(date +%s)
  echo "exit=${rc} wall=$((t1-t0))s"
  grep -h "^DET_JSON " "/out/scout/${name}/stdout.log" | tail -1 | tee "/out/scout/${name}/det.json"
  grep -h "passed=" "/out/scout/${name}/stdout.log" | sort -u | head -3
  grep -h "reason:" "/out/scout/${name}/stdout.log" | head -6
  echo "${name} ${rc}" >> /out/scout/verdicts.txt
}

BASE_KEYS="\"hidden_size\": 2048, \"ffn_size\": 5632, \"num_heads\": 16, \"seq_len\": 512, \"batch_size\": 1, \"seed\": 1234, \"steps\": 3, \"checksum_mode\": \"per_rank\""

# 1-2. The committed recipe baseline cell, with and without the tf32_off bundle.
run_cell baseline-bf16-24L \
  "{${BASE_KEYS}, \"dtype\": \"bf16\", \"num_layers\": 24, \"capture_dir\": \"/out/scout/baseline-bf16-24L/cap\"}"
run_cell tf32off-bf16-24L \
  "{${BASE_KEYS}, \"dtype\": \"bf16\", \"num_layers\": 24, \"capture_dir\": \"/out/scout/tf32off-bf16-24L/cap\"}" \
  DISABLE_TF32=1

# 3-4. fp32 is where TF32/xf32 reduction ordering can actually bite; bf16 never
# takes the TF32 path at all, so a bf16-only scout could not distinguish
# "tf32_off does nothing here" from "tf32_off is inapplicable here".
run_cell fp32-24L \
  "{${BASE_KEYS}, \"dtype\": \"fp32\", \"num_layers\": 24, \"capture_dir\": \"/out/scout/fp32-24L/cap\"}"
run_cell tf32off-fp32-24L \
  "{${BASE_KEYS}, \"dtype\": \"fp32\", \"num_layers\": 24, \"capture_dir\": \"/out/scout/tf32off-fp32-24L/cap\"}" \
  DISABLE_TF32=1

# 5-6. The recipe MoE cell: top-1 routing is a scatter/atomic path and the most
# plausible in-tree source of genuine run-to-run nondeterminism.
run_cell moe4-bf16-12L \
  "{${BASE_KEYS}, \"dtype\": \"bf16\", \"num_layers\": 12, \"num_experts\": 4, \"capture_dir\": \"/out/scout/moe4-bf16-12L/cap\"}"
run_cell moe4-fp32-12L \
  "{${BASE_KEYS}, \"dtype\": \"fp32\", \"num_layers\": 12, \"num_experts\": 4, \"capture_dir\": \"/out/scout/moe4-fp32-12L/cap\"}"

# 7. global checksum mode additionally compares an all-reduced fingerprint,
# which is the only cell able to see collective-ordering drift.
run_cell global-fp32-24L \
  "{\"hidden_size\": 2048, \"ffn_size\": 5632, \"num_heads\": 16, \"seq_len\": 512, \"batch_size\": 1, \"seed\": 1234, \"steps\": 3, \"checksum_mode\": \"global\", \"dtype\": \"fp32\", \"num_layers\": 24, \"capture_dir\": \"/out/scout/global-fp32-24L/cap\"}"

echo
echo "======== SCOUT SUMMARY (0 = passed, non-zero = diverged/failed) ========"
cat /out/scout/verdicts.txt
'

echo "=== done $(date -u +%FT%TZ) ==="
