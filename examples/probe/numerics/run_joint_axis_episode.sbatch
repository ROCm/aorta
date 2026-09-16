#!/bin/bash
# One LIVE joint-axis episode: a real Qwen3-8B driving `aorta agent mitigate`
# across BOTH probe axes against the uninitialised-workspace NaN reproducer.
#
# Why live rather than a replay against the archived matrix. The archive
# (nan_matrix/PROBE-NAN-WS-MATRIX) has a diagnostic axis of exactly `[none]`,
# so it contains no cell a *diagnostic* action could be observed at. Replaying
# the joint axis against it would mean inventing the observation, which is the
# one thing a demo of this must not do. Running it live costs about three
# minutes of GPU: a trial of this workload is ~3.9s measured, so a 10-cell
# budget at 4 trials is ~160s of workload plus the engine bring-up.
#
# The model half is real too. `serve_for_rollouts.sh` exists for this exact
# consumer and defaults --grammar-backend xgrammar on, because
# LiteLLMProposer sends response_format={"type":"json_object"} on every call
# and a default-configured engine answers that with a 500.
#
# meta64 only, and the allocation is checked: the ten-night perf window on the
# MI350 CI runner is live until 17 Sep and nothing here may touch it.
#SBATCH -p meta64
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -J demo-joint
#SBATCH -o /apps/vikhande/demo/logs/joint-%j.out
#SBATCH -e /apps/vikhande/demo/logs/joint-%j.err

set -uo pipefail
export TMPDIR=/apps/vikhande/tmp HF_HOME=/apps/vikhande/cache/hf PIP_CACHE_DIR=/apps/vikhande/cache/pip

DEMO=/apps/vikhande/demo
PROBE=/apps/vikhande/probe-gpu
WT=${DEMO}/wt
# Built on the COMPUTE node with python3.11, not on the head node. The head
# node's default python3 is 3.10 and these nodes ship only 3.9 and 3.11, so a
# head-built venv resolves its own bin/python to a version whose site-packages
# is not there -- which surfaces as `No module named 'aorta'` from a script
# that ran, and reads like a broken install rather than a wrong interpreter.
VENV=${DEMO}/venv-node311
EP=${DEMO}/episode

step () { echo; echo "[$(date -u +%T)] ======== $* ========"; }

step "node $(hostname) job ${SLURM_JOB_ID} nodelist=${SLURM_JOB_NODELIST}"
scontrol show hostnames "${SLURM_JOB_NODELIST}"
if scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -qi smci350; then
  echo "FATAL: allocation contains an smci350 node -- refusing to run" >&2
  exit 90
fi

step "python env (on the node, python3.11)"
if [ ! -x "${VENV}/bin/aorta" ]; then
  python3.11 -m venv "${VENV}" || exit 1
  "${VENV}/bin/python" -m pip install -q --upgrade pip || exit 1
  "${VENV}/bin/python" -m pip install -q -e "${WT}" litellm || exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "${WT}" || exit 1
mkdir -p "${EP}"
python -c "import aorta, litellm; print('aorta at', aorta.__file__)" || exit 92
aorta --version 2>/dev/null || true

# ---------------------------------------------------------------------------
step "serve Qwen3-8B (own step; a detached container dies with its step cgroup)"
# ---------------------------------------------------------------------------
export TS_HF_HOME="${HF_HOME}"
export TS_OUT_DIR="${TMPDIR}/aorta-demo-joint/ts-out"
export TS_LOG_DIR="${TMPDIR}/aorta-demo-joint/logs"
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

# LiteLLMProposer routes through litellm's openai provider; these two are all
# `aorta agent` needs to reach a local engine.
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local-not-a-secret

step "prove the exact call the proposer makes, before spending the episode on it"
# LiteLLMProposer sends response_format={"type":"json_object"} on every call.
# An engine launched with --grammar-backend none answers that with a 500, and
# finding out after a 7-minute bring-up costs the whole job. This is that one
# call, through litellm, with the same model string the agent will use.
python - <<'PY' || { echo "FATAL: the proposer's own call shape does not work" >&2; \
  examples/rl/serve_for_rollouts.sh logs | tail -30; \
  examples/rl/serve_for_rollouts.sh down; exit 93; }
import json, sys
import litellm
r = litellm.completion(
    model="openai/Qwen/Qwen3-8B",
    messages=[
        {"role": "system", "content": "Return strict JSON with key ok (bool)."},
        {"role": "user", "content": "{}"},
    ],
    response_format={"type": "json_object"},
    max_tokens=64,
)
content = r.choices[0].message.content
print("probe reply:", (content or "")[:200])
sys.exit(0 if content and content.strip() else 1)
PY

# ---------------------------------------------------------------------------
step "LIVE episode -- real model, both axes, cell budget 10"
# ---------------------------------------------------------------------------
# --diagnostic names the two registry entries whose own section header calls
# them diagnostics (AMD_LOG_LEVEL=4, HIP_LAUNCH_BLOCKING=1): they buy
# visibility rather than testing a cause. --max-cells is the budget the new
# cost model is denominated in; --max-iterations still bounds proposal cycles.
rm -rf "${EP}/live"
unset NAN_WS_MODE
aorta agent mitigate \
  --recipe recipes/probe/probe-numerics-uninit-workspace.yaml \
  --output "${EP}/live" \
  --ticket DEMO-JOINT-LIVE \
  --symptom "loss goes NaN in a microbatch accumulation workspace; an earlier unscaled fp16 phase overflowed, and the NaN appears in a tensor no numeric op touched" \
  --llm-backend litellm \
  --llm-model openai/Qwen/Qwen3-8B \
  --diagnostic amd_log_level_4 \
  --diagnostic hip_launch_blocking \
  --max-cells 10 \
  --max-iterations 4 \
  -vv \
  -- "${PROBE}/scripts/launch_nan_ws.sh"
echo "agent(live) exit=$?"

step "LIVE episode -- agent_log.jsonl"
cat "${EP}/live/DEMO-JOINT-LIVE/agent_log.jsonl" 2>/dev/null

step "LIVE episode -- cells on disk"
ls -1 "${EP}/live/DEMO-JOINT-LIVE" 2>/dev/null

# ---------------------------------------------------------------------------
step "CONTROL -- same recipe, same budget, fake backend"
# ---------------------------------------------------------------------------
# Not decoration: it separates "the loop can grow both axes" from "the model
# chose to". The fake proposer walks the axes in registry order, so a live
# episode that looks identical to this one has demonstrated the plumbing and
# nothing about the policy.
rm -rf "${EP}/fake"
aorta agent mitigate \
  --recipe recipes/probe/probe-numerics-uninit-workspace.yaml \
  --output "${EP}/fake" \
  --ticket DEMO-JOINT-FAKE \
  --symptom "loss goes NaN in a microbatch accumulation workspace; an earlier unscaled fp16 phase overflowed, and the NaN appears in a tensor no numeric op touched" \
  --llm-backend fake \
  --diagnostic amd_log_level_4 \
  --diagnostic hip_launch_blocking \
  --max-cells 10 \
  --max-iterations 4 \
  -v \
  -- "${PROBE}/scripts/launch_nan_ws.sh"
echo "agent(fake) exit=$?"
cat "${EP}/fake/DEMO-JOINT-FAKE/agent_log.jsonl" 2>/dev/null

step "tear the endpoint down"
examples/rl/serve_for_rollouts.sh down
kill "${HOLD_PID}" 2>/dev/null
wait "${HOLD_PID}" 2>/dev/null

step "record provenance"
python - <<PY
import json, os, socket, pathlib
out = pathlib.Path("${EP}") / "provenance.json"
out.write_text(json.dumps({
    "slurm_job": os.environ.get("SLURM_JOB_ID"),
    "node": socket.gethostname(),
    "model": "openai/Qwen/Qwen3-8B",
    "base_url": "http://127.0.0.1:8000/v1",
    "episode_backend": "litellm",
    "live_run_dir": "${EP}/live/DEMO-JOINT-LIVE",
    "fake_run_dir": "${EP}/fake/DEMO-JOINT-FAKE",
    "recipe": "recipes/probe/probe-numerics-uninit-workspace.yaml",
    "diagnostics_offered": ["amd_log_level_4", "hip_launch_blocking"],
    "max_probe_cells": 10,
}, indent=2) + "\n", encoding="utf-8")
print(out.read_text())
PY

echo "=== done $(date -u +%FT%TZ) ==="
