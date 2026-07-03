#!/usr/bin/env bash
#
# Run AORTA CLI commands inside a mirage GPU session (vLLM/ROCm container).
#
# This is the supported path for torch-based AORTA workloads: host-side
# `mirage run -- aorta …` under rocjitsu CPU emulation is impractically slow.
#
# Usage:
#   ./scripts/emulation/run_mirage_container.sh gpu-smoke
#   ./scripts/emulation/run_mirage_container.sh probe
#   EMULATOR=rocjitsu-dbt ./scripts/emulation/run_mirage_container.sh gpu-smoke
#
# Env knobs:
#   MIRAGE_BIN          mirage CLI (required)
#   MIRAGE_AORTA_IMAGE  container image (default: vllm/vllm-openai-rocm:v0.23.0-patched-v2)
#   AORTA_SRC           AORTA checkout (default: repo root)
#   EMULATOR            rocjitsu (default) or rocjitsu-dbt
#   PROFILE             mirage profile (default: mi350x or dbt-mi350x)
#   OUT                 host output directory
#   XDG_CONFIG_HOME     mirage config dir
#   XDG_RUNTIME_DIR     mirage runtime dir
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AORTA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/aorta_cli_runner.py"

WORKLOAD="${1:-gpu-smoke}"
MIRAGE_BIN="${MIRAGE_BIN:?set MIRAGE_BIN to the mirage binary}"
IMAGE="${MIRAGE_AORTA_IMAGE:-docker.io/vllm/vllm-openai-rocm:v0.23.0-patched-v2}"
AORTA_SRC="${AORTA_SRC:-$AORTA_ROOT}"
EMULATOR="${EMULATOR:-rocjitsu}"
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/tmp/mirage-aorta-config}"
XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/mirage-aorta-runtime}"

case "$EMULATOR" in
  rocjitsu)     PROFILE="${PROFILE:-mi350x}" ;;
  rocjitsu-dbt) PROFILE="${PROFILE:-dbt-mi350x}" ;;
  *) echo "unknown EMULATOR=$EMULATOR (use rocjitsu or rocjitsu-dbt)" >&2; exit 1 ;;
esac

log()  { printf '==> %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

[[ -x "$MIRAGE_BIN" ]] || fail "mirage not found at $MIRAGE_BIN"
[[ -d "$AORTA_SRC" ]] || fail "AORTA checkout not found at $AORTA_SRC"
[[ -f "$RUNNER" ]] || fail "runner not found at $RUNNER"

mkdir -p "$XDG_CONFIG_HOME" "$XDG_RUNTIME_DIR"
export XDG_CONFIG_HOME XDG_RUNTIME_DIR MIRAGE_BIN

ensure_profile() {
  "$MIRAGE_BIN" profile show "$PROFILE" >/dev/null 2>&1 && return 0
  case "$EMULATOR" in
    rocjitsu)
      "$MIRAGE_BIN" profile create "$PROFILE" --emulator rocjitsu --agent MI350X \
        --num-nodes 1 --gpus-per-node 1 --no-input ;;
    rocjitsu-dbt)
      "$MIRAGE_BIN" profile create "$PROFILE" --emulator rocjitsu-dbt --agent MI350X \
        --num-nodes 1 --gpus-per-node 1 --no-input ;;
  esac
}
ensure_profile

CONTAINER_BOOT='
      rm -rf /tmp/aorta-build/src
      cp -a /aorta-src /tmp/aorta-build/src
      cp /runner.py /tmp/aorta_cli_runner.py
      python3 -m pip install -q /tmp/aorta-build/src --no-deps
      python3 -m pip install -q click pyyaml
      cd /out
      python3 /tmp/aorta_cli_runner.py
    '

run_in_container() {
  local json_args="$1"
  "$MIRAGE_BIN" run --in-process --profile "$PROFILE" \
    --image "$IMAGE" \
    --env "AORTA_CLI_JSON=$json_args" \
    --mount "$AORTA_SRC:/aorta-src:ro" \
    --mount "$RUNNER:/runner.py:ro" \
    --mount "$OUT:/out" \
    --mount "$OUT:/tmp/aorta-build" \
    -- sh -c "$CONTAINER_BOOT"
}

case "$WORKLOAD" in
  gpu-smoke)
    OUT="${OUT:-/tmp/aorta-gpu-smoke-out}"
    mkdir -p "$OUT"
    log "triage gpu-smoke (emulator=$EMULATOR profile=$PROFILE) -> $OUT"
    run_in_container '["triage","run","--verbose","--recipe","/tmp/aorta-build/src/recipes/gpu-smoke-emulated.yaml","--output-dir","/out/triage_results"]'
    ;;
  probe)
    OUT="${OUT:-/tmp/aorta-probe-out}"
    mkdir -p "$OUT"
    log "probe smoke (emulator=$EMULATOR profile=$PROFILE) -> $OUT"
    run_in_container '["probe","--recipe","/tmp/aorta-build/src/recipes/example-probe-smoke.yaml","--output","/out/probe_results","--ticket","PROBE-MIRAGE","--","bash","-c","echo hi from mirage probe"]'
    ;;
  llm-determinism)
    OUT="${OUT:-/tmp/aorta-llm-out}"
    RECIPE="${LLM_RECIPE:?set LLM_RECIPE to a llm_determinism recipe YAML on the host}"
    [[ -f "$RECIPE" ]] || fail "recipe not found: $RECIPE"
    mkdir -p "$OUT"
    log "llm_determinism (slow under rocjitsu; faster with rocjitsu-dbt) -> $OUT"
    "$MIRAGE_BIN" run --in-process --profile "$PROFILE" \
      --image "$IMAGE" \
      --env 'AORTA_CLI_JSON=["triage","run","--verbose","--recipe","/recipe.yaml","--output-dir","/out/triage_results"]' \
      --env RANK=0 --env WORLD_SIZE=1 --env LOCAL_RANK=0 \
      --env MASTER_ADDR=127.0.0.1 --env MASTER_PORT=29500 \
      --mount "$AORTA_SRC:/aorta-src:ro" \
      --mount "$RUNNER:/runner.py:ro" \
      --mount "$RECIPE:/recipe.yaml:ro" \
      --mount "$OUT:/out" \
      --mount "$OUT:/tmp/aorta-build" \
      -- sh -c "$CONTAINER_BOOT"
    ;;
  *)
    fail "unknown workload: $WORKLOAD (try: gpu-smoke, probe, llm-determinism)"
    ;;
esac

log "done (workload=$WORKLOAD emulator=$EMULATOR output=$OUT)"
