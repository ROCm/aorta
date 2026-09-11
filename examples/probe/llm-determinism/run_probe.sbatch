#!/bin/bash
# The first archived probe-mode run in this project against a real GPU workload.
#
# Three phases, in decreasing order of what they are allowed to claim:
#
#   A  agent   -- `aorta agent mitigate` on the UNMODIFIED llm_determinism
#                 replay. This is the deliverable. Whatever it says is what the
#                 hardware said.
#   B  sweep   -- the same recipe and the same launch command as a full probe
#                 matrix. Exists because the agent loop short-circuits on a
#                 passing baseline (loop.py:399) and therefore never executes
#                 the tf32_off arm; scouting says the baseline passes, so
#                 without this the mitigation axis would go unmeasured.
#   C  harness -- the fault-injection variant docs/llm-determinism.md ships for
#                 detector validation. Its verdict is a property of the
#                 injection, not of the stack. It is here so that "no
#                 divergence" in A/B can be distinguished from "the detector
#                 never fires here", and it is excluded from the corpus.
#
# Deliberately NOT the CI runner smci350-rck-g03-f16-12: the ten-night perf
# baseline window is live through 17 Sep. meta64 contains zero smci350 nodes.
#SBATCH -p meta64
#SBATCH -N 1
#SBATCH -w cv350-rck-g03-c07-18
#SBATCH -t 00:50:00
#SBATCH -J probe-gpu-archive
#SBATCH -o /apps/vikhande/probe-gpu/logs/probe-%j.out
#SBATCH -e /apps/vikhande/probe-gpu/logs/probe-%j.err

set -uo pipefail

export TMPDIR=/apps/vikhande/tmp
export HF_HOME=/apps/vikhande/cache/hf
export PIP_CACHE_DIR=/apps/vikhande/cache/pip

ROOT=/apps/vikhande/probe-gpu
WT="${ROOT}/wt"
VENV="${ROOT}/venv"
RECIPE=recipes/probe/probe-llm-determinism-replay.yaml

step () { echo; echo "[$(date -u +%T)] ======== $* ========"; }

step "node $(hostname) job ${SLURM_JOB_ID} nodelist=${SLURM_JOB_NODELIST}"
scontrol show hostnames "${SLURM_JOB_NODELIST}"
if scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -qi smci350; then
  echo "FATAL: allocation contains an smci350 node -- refusing to run" >&2
  exit 90
fi
cat /opt/rocm/.info/version 2>/dev/null

step "python env for the aorta CLI (CPU only; the GPU work is in the container)"
if [ ! -x "${VENV}/bin/aorta" ]; then
  python3.11 -m venv "${VENV}" || exit 1
  "${VENV}/bin/python" -m pip install -q --upgrade pip || exit 1
  "${VENV}/bin/python" -m pip install -q -e "${WT}" || exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -V
aorta --version 2>/dev/null || true

cd "${WT}" || exit 1
mkdir -p "${ROOT}/captures"

# Five replays per trial x two trials per cell. `steps` is a first-class knob:
# each step is a self-contained snapshot -> r1 -> restore -> r2 -> compare, so
# it is the cheap way to buy replay comparisons without paying FSDP/RCCL
# startup again.
export DET_CFG='{"hidden_size": 2048, "ffn_size": 5632, "num_heads": 16, "seq_len": 512, "batch_size": 1, "seed": 1234, "steps": 5, "checksum_mode": "per_rank", "dtype": "bf16", "num_layers": 24}'

# ---------------------------------------------------------------------------
step "PHASE A -- aorta agent mitigate, unmodified workload"
# ---------------------------------------------------------------------------
rm -rf "${ROOT}/agent_results"
aorta agent mitigate \
  --recipe "${RECIPE}" \
  --output "${ROOT}/agent_results" \
  --ticket PROBE-LLMDET-GPU \
  --symptom "bit-exact replay of a single transformer block; checking for run-to-run nondeterminism in matmul / collective reduction ordering on gfx950" \
  --llm-backend fake \
  --max-iterations 4 \
  -vv \
  -- "${ROOT}/scripts/launch_det.sh"
echo "agent exit=$?"

step "PHASE A -- agent_log.jsonl"
cat "${ROOT}/agent_results/PROBE-LLMDET-GPU/agent_log.jsonl" 2>/dev/null

# ---------------------------------------------------------------------------
step "PHASE B -- aorta sweep run, same recipe, full mitigation axis"
# ---------------------------------------------------------------------------
rm -rf "${ROOT}/sweep_results"
aorta sweep run \
  --recipe "${RECIPE}" \
  --output-dir "${ROOT}/sweep_results" \
  --ticket PROBE-LLMDET-GPU-MATRIX \
  -v \
  -- "${ROOT}/scripts/launch_det.sh"
echo "sweep exit=$?"

# ---------------------------------------------------------------------------
step "PHASE C -- harness validation (INJECTED FAULT, not a scenario)"
# ---------------------------------------------------------------------------
# steps: 1 -- this phase is testing plumbing, not soaking for a rate.
export DET_CFG='{"hidden_size": 2048, "ffn_size": 5632, "num_heads": 16, "seq_len": 512, "batch_size": 1, "seed": 1234, "steps": 1, "checksum_mode": "per_rank", "dtype": "bf16", "num_layers": 24}'
rm -rf "${ROOT}/harness_validation"
aorta agent mitigate \
  --recipe "${RECIPE}" \
  --output "${ROOT}/harness_validation" \
  --ticket HARNESS-VALIDATION-INJECTED \
  --symptom "injected divergence: detector validation only" \
  --llm-backend fake \
  --max-iterations 4 \
  -vv \
  -- "${ROOT}/scripts/launch_det.sh" /out/scripts/det_launch_injected.py
echo "agent(injected) exit=$?"

step "PHASE C -- agent_log.jsonl"
cat "${ROOT}/harness_validation/HARNESS-VALIDATION-INJECTED/agent_log.jsonl" 2>/dev/null

# ---------------------------------------------------------------------------
step "VERDICT TABLE"
# ---------------------------------------------------------------------------
python - "${ROOT}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for tree in ("agent_results", "sweep_results", "harness_validation"):
    base = root / tree
    if not base.is_dir():
        continue
    print(f"\n--- {tree} ---")
    for result in sorted(base.rglob("trial_*/result.json")):
        doc = json.loads(result.read_text())
        rel = result.relative_to(base)
        print(f"  {str(rel):<52} verdict={doc.get('verdict'):<6} "
              f"exit={doc.get('exit_code')} "
              f"failures={doc.get('failure_detectors_fired')} "
              f"env={ {k: v for k, v in (doc.get('env') or {}).items()} }")
PY

step "done $(date -u +%FT%TZ)"
