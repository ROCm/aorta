#!/usr/bin/env bash
#
# Run a matrix of AORTA CLI commands under mirage (rocjitsu + rocjitsu-dbt).
#
# Prerequisites: MIRAGE_BIN set, docker, vLLM image pulled.
#
# Usage:
#   export MIRAGE_BIN=/path/to/mirage
#   ./scripts/emulation/run_mirage_matrix.sh
#
# Optional env knobs:
#   TIMEOUT_SEC       per-case timeout (default 900)
#   INCLUDE_LLM_DET   set to 1 to also run the (slow) llm_determinism case;
#                     pair with a larger TIMEOUT_SEC (e.g. 1800) under rocjitsu
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AORTA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/aorta_cli_runner.py"

MIRAGE_BIN="${MIRAGE_BIN:?set MIRAGE_BIN}"
AORTA_SRC="${AORTA_SRC:-$AORTA_ROOT}"
IMAGE="${MIRAGE_AORTA_IMAGE:-docker.io/vllm/vllm-openai-rocm:v0.23.0-patched-v2}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/aorta-mirage-matrix-$(date +%Y%m%d-%H%M%S)}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

[[ -x "$MIRAGE_BIN" ]] || { echo "mirage missing: $MIRAGE_BIN" >&2; exit 1; }
[[ -f "$RUNNER" ]] || { echo "runner missing: $RUNNER" >&2; exit 1; }

mkdir -p "$RESULT_ROOT"
SUMMARY="$RESULT_ROOT/summary.tsv"
printf 'emulator\tprofile\tcommand\texit\tnotes\n' > "$SUMMARY"

BOOT='
rm -rf /tmp/aorta-build/src
cp -a /aorta-src /tmp/aorta-build/src
cp /runner.py /tmp/aorta_cli_runner.py
python3 -m pip install -q /tmp/aorta-build/src --no-deps
python3 -m pip install -q click pyyaml
cd /out
python3 /tmp/aorta_cli_runner.py
'

run_case() {
  local emulator="$1" profile="$2" tag="$3" json_args="$4"
  local outdir="$RESULT_ROOT/${emulator}/${tag}"
  mkdir -p "$outdir"
  local xdg_cfg="/tmp/mirage-matrix-${emulator}-config"
  local xdg_rt="/tmp/mirage-matrix-${emulator}-runtime"
  mkdir -p "$xdg_cfg" "$xdg_rt"

  log "[$emulator/$tag]"
  if ! XDG_CONFIG_HOME="$xdg_cfg" XDG_RUNTIME_DIR="$xdg_rt" \
      "$MIRAGE_BIN" profile show "$profile" >/dev/null 2>&1; then
    if [[ "$emulator" == "rocjitsu-dbt" ]]; then
      XDG_CONFIG_HOME="$xdg_cfg" XDG_RUNTIME_DIR="$xdg_rt" \
        "$MIRAGE_BIN" profile create "$profile" --emulator rocjitsu-dbt --agent MI350X \
          --num-nodes 1 --gpus-per-node 1 --no-input
    else
      XDG_CONFIG_HOME="$xdg_cfg" XDG_RUNTIME_DIR="$xdg_rt" \
        "$MIRAGE_BIN" profile create "$profile" --emulator rocjitsu --agent MI350X \
          --num-nodes 1 --gpus-per-node 1 --no-input
    fi
  fi

  # torch.distributed workloads (llm_determinism) use env:// rendezvous and
  # require RANK/WORLD_SIZE/etc even for a world_size=1 singleton. Inject them
  # only for those cases; other cases (help, env probe, gpu_smoke, ...) don't
  # need them.
  local extra_env=()
  if [[ "$tag" == *llm-determinism* ]]; then
    extra_env=(--env RANK=0 --env WORLD_SIZE=1 --env LOCAL_RANK=0
               --env MASTER_ADDR=127.0.0.1 --env MASTER_PORT=29500)
  fi

  local logfile="$outdir/run.log"
  local ec=0
  set +e
  timeout "$TIMEOUT_SEC" env XDG_CONFIG_HOME="$xdg_cfg" XDG_RUNTIME_DIR="$xdg_rt" \
    "$MIRAGE_BIN" run --in-process --profile "$profile" \
      --image "$IMAGE" \
      --env "AORTA_CLI_JSON=$json_args" \
      ${extra_env[@]+"${extra_env[@]}"} \
      --mount "$AORTA_SRC:/aorta-src:ro" \
      --mount "$RUNNER:/runner.py:ro" \
      --mount "$outdir:/out" \
      --mount "$outdir:/tmp/aorta-build" \
      -- sh -c "$BOOT" > "$logfile" 2>&1
  ec=$?
  set -e
  local note=""
  [[ $ec -eq 124 ]] && note="TIMEOUT"
  printf '%s\t%s\t%s\t%s\t%s\n' "$emulator" "$profile" "$tag" "$ec" "$note" >> "$SUMMARY"
  log "[$emulator/$tag] exit=$ec"
}

CASES=(
  'help|["--help"]'
  'environments-list|["environments","list"]'
  'triage-list-mitigations|["triage","list-mitigations"]'
  'triage-list-environments|["triage","list-environments"]'
  'env-probe-summary|["env","probe","--summary"]'
  'env-probe-json|["env","probe","-o","/out/env.json"]'
  'env-probe-field-gpu|["env","probe","--field","gpu_arch"]'
  'run-gpu-smoke|["run","--workload","gpu_smoke","--environment","local","--trials","1"]'
  'triage-dry-run|["triage","run","--dry-run","--recipe","/tmp/aorta-build/src/recipes/emulated/gpu-smoke-emulated.yaml"]'
  'triage-gpu-smoke|["triage","run","--verbose","--recipe","/tmp/aorta-build/src/recipes/emulated/gpu-smoke-emulated.yaml","--output-dir","/out/triage_results"]'
  'triage-inference-smoke|["triage","run","--verbose","--recipe","/tmp/aorta-build/src/recipes/emulated/inference-smoke-emulated.yaml","--output-dir","/out/triage_results"]'
  'triage-training-ddp-smoke|["triage","run","--verbose","--recipe","/tmp/aorta-build/src/recipes/emulated/training-ddp-smoke-emulated.yaml","--output-dir","/out/triage_results"]'
  'triage-training-fsdp-smoke|["triage","run","--verbose","--recipe","/tmp/aorta-build/src/recipes/emulated/training-fsdp-smoke-emulated.yaml","--output-dir","/out/triage_results"]'
  'probe-smoke|["probe","--recipe","/tmp/aorta-build/src/recipes/probe/example-probe-smoke.yaml","--output","/out/probe_results","--ticket","PROBE-MIRAGE-MATRIX","--","bash","-c","echo hi from mirage probe"]'
  'probe-list-patterns|["probe","--list-patterns"]'
)

# Torch-heavy cases kept out of the default matrix: llm_determinism is very slow
# under rocjitsu CPU emulation and would routinely trip TIMEOUT_SEC. Opt in with
# INCLUDE_LLM_DET=1 (and consider a larger TIMEOUT_SEC, e.g. 1800).
if [[ "${INCLUDE_LLM_DET:-0}" == "1" ]]; then
  CASES+=(
    'triage-llm-determinism|["triage","run","--verbose","--recipe","/tmp/aorta-build/src/recipes/emulated/llm-determinism-emulated.yaml","--output-dir","/out/triage_results"]'
  )
fi

for pair in "rocjitsu:mi350x" "rocjitsu-dbt:dbt-mi350x"; do
  emulator="${pair%%:*}"
  profile="${pair##*:}"
  for entry in "${CASES[@]}"; do
    tag="${entry%%|*}"
    json="${entry#*|}"
    run_case "$emulator" "$profile" "$tag" "$json"
  done
done

log "Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
