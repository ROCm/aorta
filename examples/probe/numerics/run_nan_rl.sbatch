#!/bin/bash
# Score the fix half against a REAL resolver instead of an enumerated
# hypothesis.
#
# `a247b27` could only report the fix half under hypothesised resolution sets,
# because every archived scenario it had was either sanitizer-mode (no
# mitigation cells at all) or the synthetic bash-exit probe. This job supplies
# the missing input: a matrix in which the baseline genuinely fails, exactly
# one registered mitigation genuinely resolves it, and six registered decoys
# genuinely do not.
#
# `--full-candidates` is deliberate. Offering the corpus's four-name slice
# would hand the model a list the resolver is nearly certain to be in; offering
# the whole registry makes naming it a choice, which is the only version of the
# measurement worth reporting.
#
# meta64 only; the ten-night perf window on the MI350 CI runner is live.
#SBATCH -p meta64
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -J nan-rl
#SBATCH -o /apps/vikhande/probe-gpu/logs/nan-rl-%j.out
#SBATCH -e /apps/vikhande/probe-gpu/logs/nan-rl-%j.err

set -uo pipefail
export TMPDIR=/apps/vikhande/tmp HF_HOME=/apps/vikhande/cache/hf PIP_CACHE_DIR=/apps/vikhande/cache/pip

ROOT=/apps/vikhande/probe-gpu
WT=${ROOT}/wt-num
VENV=${ROOT}/venv-num
RL=${ROOT}/rl-nan
MATRIX=${ROOT}/nan_matrix/PROBE-NAN-WS-MATRIX

step () { echo; echo "[$(date -u +%T)] ======== $* ========"; }

step "node $(hostname) job ${SLURM_JOB_ID} nodelist=${SLURM_JOB_NODELIST}"
scontrol show hostnames "${SLURM_JOB_NODELIST}"
if scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -qi smci350; then
  echo "FATAL: allocation contains an smci350 node -- refusing to run" >&2
  exit 90
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "${WT}" || exit 1
mkdir -p "${RL}"

step "corpus from the archived matrix"
rm -rf "${RL}/corpus"
python examples/rl/build_corpus.py \
  --results "${MATRIX}" \
  --baselines recipes/sanitizers/fixtures/expected/verdict_baselines.json \
  --out "${RL}/corpus" \
  --run-meta "${RL}/run-meta.json" \
  --families "${RL}/families.json"
echo "build_corpus exit=$?"
wc -l "${RL}"/corpus/*.jsonl 2>/dev/null

step "resolvers recovered from the matrix (the fix half's ground truth)"
python - <<PY
import sys; sys.path.insert(0, "examples/rl")
from pathlib import Path
from fix_reward import resolution_from_matrix
r = resolution_from_matrix(Path("${MATRIX}"))
print("cells        :", r.cells)
print("baseline_failed:", r.baseline_failed)
print("resolvers    :", sorted(r.resolvers))
PY

step "serve Qwen3-8B for rollouts (hold in its own step; see the script header)"
export TS_HF_HOME="${HF_HOME}"
export TS_OUT_DIR="${TMPDIR}/aorta-rl-nan/ts-out"
export TS_LOG_DIR="${TMPDIR}/aorta-rl-nan/logs"
mkdir -p "${TS_OUT_DIR}" "${TS_LOG_DIR}"
srun --overlap -N1 -n1 examples/rl/serve_for_rollouts.sh hold &
HOLD_PID=$!

for i in $(seq 1 240); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "endpoint healthy after ${i}0s"; break
  fi
  sleep 10
done
curl -s http://127.0.0.1:8000/v1/models | head -c 400; echo
if ! curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "FATAL: endpoint never became healthy; not pretending to measure" >&2
  examples/rl/serve_for_rollouts.sh logs | tail -40
  examples/rl/serve_for_rollouts.sh down
  exit 91
fi

step "rollouts: 8 samples per scenario, t=0.7, whole registry offered"
srun --overlap -N1 -n1 python examples/rl/run_e2e.py \
  --corpus "${RL}/corpus/triage.jsonl" \
  --base-url http://127.0.0.1:8000/v1 \
  --model openai/Qwen/Qwen3-8B \
  --samples 8 \
  --temperature 0.7 \
  --seed 1234 \
  --full-candidates \
  --out "${RL}/results"
echo "run_e2e exit=$?"

step "tear the endpoint down"
examples/rl/serve_for_rollouts.sh down
kill "${HOLD_PID}" 2>/dev/null
wait "${HOLD_PID}" 2>/dev/null

step "rescore -- form half alone, then with the fix half against the real matrix"
python examples/rl/rescore_e2e.py "${RL}"/results/*.json 2>&1 | tail -60
echo
echo "######## with --matrix ${MATRIX} ########"
python examples/rl/rescore_e2e.py "${RL}"/results/*.json --matrix "${MATRIX}" 2>&1 | tail -80

echo "=== done $(date -u +%FT%TZ) ==="
