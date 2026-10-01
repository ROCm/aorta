#!/usr/bin/env bash
#
# Start or stop the Ruby-cluster Qwen vLLM service used by AORTA Chat.
#
# The driver runs on a login host. It submits this same file to Slurm in
# --worker mode, waits for the model to answer, and leaves the allocation
# running. The selected endpoint is written to a shared, sourceable file.

set -euo pipefail

LOGIN_HOST="${AORTA_QWEN_LOGIN_HOST:-ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com}"
PARTITION="${AORTA_QWEN_PARTITION:-interactive}"
TIME_LIMIT="${AORTA_QWEN_TIME_LIMIT:-03:00:00}"
RUNTIME_DIR="${AORTA_CHAT_RUNTIME_DIR:-/apps/avsharma/aorta-chat-runtime}"
HF_CACHE="${AORTA_QWEN_HF_CACHE:-/apps/avsharma/hf-cache}"
IMAGE="${AORTA_QWEN_IMAGE:-rocm/vllm-dev:nightly}"
MODEL="${AORTA_QWEN_MODEL:-Qwen/Qwen3.8-27B}"
PORT_START="${AORTA_QWEN_PORT_START:-8001}"
PORT_END="${AORTA_QWEN_PORT_END:-8100}"
STARTUP_TIMEOUT="${AORTA_QWEN_STARTUP_TIMEOUT:-900}"

MODE="start"
WORKER=false

usage() {
  cat <<'EOF'
Usage:
  scripts/chat/start_qwen_vllm.sh [options]
  scripts/chat/start_qwen_vllm.sh --stop

Starts Qwen/Qwen3.8-27B in an exclusive Slurm allocation, choosing the first
free compute-node port from 8001 upward. Port 8000 is never inspected or used.
An already-running healthy service recorded by this script is reused.

Options:
  --stop                 Cancel the recorded vLLM job and wait for cleanup.
  --login-host HOST      Slurm login host.
  --partition NAME       Slurm partition.
  --time LIMIT           Slurm time limit (default: 03:00:00).
  --runtime-dir PATH     Shared state/log directory.
  --hf-cache PATH        Shared Hugging Face cache.
  --image IMAGE          vLLM container image.
  --model MODEL          Served model ID.
  --port-start PORT      First candidate model port (minimum: 8001).
  --port-end PORT        Last candidate model port.
  --startup-timeout SEC  Maximum model startup wait.
  -h, --help             Show this help.

Environment variables with the AORTA_QWEN_* names shown by the defaults near
the top of this script provide the same configuration non-interactively.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

remote_command() {
  local rendered
  printf -v rendered '%q ' "$@"
  printf '%s\n' "${rendered% }"
}

job_status() {
  local job_id="$1"
  [[ "$job_id" =~ ^[0-9]+$ ]] || return 1
  ssh -o BatchMode=yes "$LOGIN_HOST" \
    "squeue -h -j ${job_id} -o '%i %T %R'"
}

wait_for_job_exit() {
  local job_id="$1"
  local status
  while status="$(job_status "$job_id")" && [[ -n "$status" ]]; do
    printf '%s\n' "$status"
    sleep 2
  done
}

cancel_job() {
  local job_id="$1"
  [[ "$job_id" =~ ^[0-9]+$ ]] || die "invalid recorded Slurm job ID: $job_id"
  if [[ -n "$(job_status "$job_id" || true)" ]]; then
    printf 'Cancelling vLLM job %s\n' "$job_id"
    ssh -o BatchMode=yes "$LOGIN_HOST" "scancel ${job_id}"
    wait_for_job_exit "$job_id"
  fi
}

while (($#)); do
  case "$1" in
    --stop)
      MODE="stop"
      shift
      ;;
    --worker)
      WORKER=true
      shift
      ;;
    --login-host)
      LOGIN_HOST="${2:?--login-host requires a value}"
      shift 2
      ;;
    --partition)
      PARTITION="${2:?--partition requires a value}"
      shift 2
      ;;
    --time)
      TIME_LIMIT="${2:?--time requires a value}"
      shift 2
      ;;
    --runtime-dir)
      RUNTIME_DIR="${2:?--runtime-dir requires a value}"
      shift 2
      ;;
    --hf-cache)
      HF_CACHE="${2:?--hf-cache requires a value}"
      shift 2
      ;;
    --image)
      IMAGE="${2:?--image requires a value}"
      shift 2
      ;;
    --model)
      MODEL="${2:?--model requires a value}"
      shift 2
      ;;
    --port-start)
      PORT_START="${2:?--port-start requires a value}"
      shift 2
      ;;
    --port-end)
      PORT_END="${2:?--port-end requires a value}"
      shift 2
      ;;
    --startup-timeout)
      STARTUP_TIMEOUT="${2:?--startup-timeout requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

is_uint "$PORT_START" || die "--port-start must be an integer"
is_uint "$PORT_END" || die "--port-end must be an integer"
is_uint "$STARTUP_TIMEOUT" || die "--startup-timeout must be an integer"
((PORT_START >= 8001)) || die "--port-start must be at least 8001"
((PORT_END >= PORT_START)) || die "--port-end must be at least --port-start"

ENDPOINT_FILE="${RUNTIME_DIR}/qwen38-endpoint.env"

run_worker() {
  require_command docker
  require_command python3

  mkdir -p "$RUNTIME_DIR" "$HF_CACHE"

  local port host tmp_endpoint
  port="$(
    AORTA_PORT_START="$PORT_START" AORTA_PORT_END="$PORT_END" python3 - <<'PY'
import os
import socket

start = int(os.environ["AORTA_PORT_START"])
end = int(os.environ["AORTA_PORT_END"])
for candidate in range(start, end + 1):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", candidate))
    except OSError:
        sock.close()
        continue
    sock.close()
    print(candidate)
    break
else:
    raise SystemExit(f"no free model port in the range {start}-{end}")
PY
  )"
  host="$(hostname -f)"
  tmp_endpoint="${ENDPOINT_FILE}.${SLURM_JOB_ID}.tmp"

  umask 077
  {
    printf 'export QWEN_VLLM_JOB_ID=%q\n' "$SLURM_JOB_ID"
    printf 'export AORTA_CHAT_VLLM_BASE_URL=%q\n' "http://${host}:${port}/v1"
    printf 'export AORTA_CHAT_VLLM_MODEL=%q\n' "$MODEL"
  } >"$tmp_endpoint"
  mv "$tmp_endpoint" "$ENDPOINT_FILE"

  cleanup_worker() {
    if [[ -f "$ENDPOINT_FILE" ]] &&
       grep -Fq "QWEN_VLLM_JOB_ID=${SLURM_JOB_ID}" "$ENDPOINT_FILE"; then
      rm -f "$ENDPOINT_FILE"
    fi
  }
  trap cleanup_worker EXIT

  printf 'vLLM endpoint: http://%s:%s/v1\n' "$host" "$port"
  docker pull "$IMAGE"
  docker run --rm \
    --name "aorta-qwen38-${SLURM_JOB_ID}" \
    --network host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 32g \
    -e HIP_VISIBLE_DEVICES=0 \
    -e HF_HOME=/root/.cache/huggingface \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    "$IMAGE" \
    vllm serve "$MODEL" \
      --served-model-name "$MODEL" \
      --host 0.0.0.0 \
      --port "$port" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.85 \
      --max-model-len 32768 \
      --trust-remote-code \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_xml \
      --default-chat-template-kwargs '{"enable_thinking":false}'
}

if [[ "$WORKER" == true ]]; then
  run_worker
  exit $?
fi

require_command curl
require_command python3
require_command ssh

mkdir -p "$RUNTIME_DIR" "$HF_CACHE"

if [[ "$MODE" == "stop" ]]; then
  if [[ ! -s "$ENDPOINT_FILE" ]]; then
    printf 'No recorded Qwen vLLM job is running.\n'
    exit 0
  fi
  # Generated by this script with mode 0600 and shell-escaped values.
  # shellcheck disable=SC1090
  source "$ENDPOINT_FILE"
  cancel_job "${QWEN_VLLM_JOB_ID:?endpoint file has no job ID}"
  rm -f "$ENDPOINT_FILE"
  printf 'Qwen vLLM is stopped.\n'
  exit 0
fi

active_job=""
if [[ -s "$ENDPOINT_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENDPOINT_FILE"
  active_job="${QWEN_VLLM_JOB_ID:-}"
  if [[ -z "$active_job" || -z "$(job_status "$active_job" || true)" ]]; then
    rm -f "$ENDPOINT_FILE"
    active_job=""
  fi
fi

new_job=""
keep_job=false
cleanup_driver() {
  if [[ -n "$new_job" && "$keep_job" != true ]]; then
    cancel_job "$new_job" || true
  fi
}
trap cleanup_driver EXIT
trap 'exit 130' INT TERM

if [[ -z "$active_job" ]]; then
  script_path="$(readlink -f "${BASH_SOURCE[0]}")"
  log_pattern="${RUNTIME_DIR}/qwen38-vllm-%j.log"
  submit=(
    sbatch
    --parsable
    --partition "$PARTITION"
    --nodes 1
    --exclusive
    --time "$TIME_LIMIT"
    --job-name aorta-qwen38
    --output "$log_pattern"
    "$script_path"
    --worker
    --runtime-dir "$RUNTIME_DIR"
    --hf-cache "$HF_CACHE"
    --image "$IMAGE"
    --model "$MODEL"
    --port-start "$PORT_START"
    --port-end "$PORT_END"
  )
  new_job="$(
    ssh -o BatchMode=yes "$LOGIN_HOST" "$(remote_command "${submit[@]}")"
  )"
  [[ "$new_job" =~ ^[0-9]+$ ]] || die "sbatch returned an invalid job ID: $new_job"
  active_job="$new_job"
  printf 'Submitted Qwen vLLM job %s\n' "$active_job"
else
  printf 'Reusing recorded Qwen vLLM job %s\n' "$active_job"
fi

deadline=$((SECONDS + STARTUP_TIMEOUT))
while [[ ! -s "$ENDPOINT_FILE" ]]; do
  status="$(job_status "$active_job" || true)"
  [[ -n "$status" ]] || {
    ssh -o BatchMode=yes "$LOGIN_HOST" \
      "sacct -j ${active_job} -o JobID,State,ExitCode" || true
    die "vLLM job $active_job exited before publishing an endpoint"
  }
  printf '%s\n' "$status"
  ((SECONDS < deadline)) || die "timed out waiting for the vLLM endpoint file"
  sleep 5
done

# shellcheck disable=SC1090
source "$ENDPOINT_FILE"
[[ "${QWEN_VLLM_JOB_ID:-}" == "$active_job" ]] ||
  die "endpoint file belongs to job ${QWEN_VLLM_JOB_ID:-unknown}, expected $active_job"
[[ "$AORTA_CHAT_VLLM_BASE_URL" != *"localhost:8000"* ]] ||
  die "refusing forbidden fallback endpoint: $AORTA_CHAT_VLLM_BASE_URL"

health_url="${AORTA_CHAT_VLLM_BASE_URL%/v1}/health"
while ! curl --fail --silent --max-time 5 "$health_url" >/dev/null 2>&1; do
  status="$(job_status "$active_job" || true)"
  [[ -n "$status" ]] || die "vLLM job $active_job exited before becoming ready"
  ((SECONDS < deadline)) || die "timed out waiting for $health_url"
  printf 'Waiting for Qwen at %s ...\n' "$health_url"
  sleep 5
done

models="$(
  curl --fail --silent --show-error "${AORTA_CHAT_VLLM_BASE_URL}/models"
)"
AORTA_EXPECTED_MODEL="$MODEL" python3 -c '
import json
import os
import sys

payload = json.load(sys.stdin)
models = {item.get("id") for item in payload.get("data", [])}
expected = os.environ["AORTA_EXPECTED_MODEL"]
if expected not in models:
    raise SystemExit(f"{expected!r} not served; got {sorted(models)!r}")
' <<<"$models"

completion="$(
  curl --fail --silent --show-error \
    "${AORTA_CHAT_VLLM_BASE_URL}/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(
      AORTA_EXPECTED_MODEL="$MODEL" python3 -c '
import json
import os

print(json.dumps({
    "model": os.environ["AORTA_EXPECTED_MODEL"],
    "messages": [{"role": "user", "content": "Reply with READY only."}],
    "temperature": 0,
    "max_tokens": 16,
}))
'
    )"
)"
python3 -c '
import json
import sys

message = json.load(sys.stdin)["choices"][0]["message"]
if message.get("content", "").strip() != "READY":
    raise SystemExit(f"unexpected smoke response: {message!r}")
if message.get("reasoning"):
    raise SystemExit(f"thinking was not disabled: {message!r}")
' <<<"$completion"

tool_completion="$(
  curl --fail --silent --show-error \
    "${AORTA_CHAT_VLLM_BASE_URL}/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(
      AORTA_EXPECTED_MODEL="$MODEL" python3 -c '
import json
import os

name = "aorta_tool_protocol_probe"
print(json.dumps({
    "model": os.environ["AORTA_EXPECTED_MODEL"],
    "messages": [{
        "role": "user",
        "content": "Call the provided tool with source set to READY. Do not answer in text.",
    }],
    "tools": [{
        "type": "function",
        "function": {
            "name": name,
            "description": "Verify native function calling.",
            "parameters": {
                "type": "object",
                "properties": {"source": {"type": "string"}},
                "required": ["source"],
            },
        },
    }],
    "tool_choice": {"type": "function", "function": {"name": name}},
    "temperature": 0,
    "max_tokens": 128,
}))
'
    )"
)"
python3 -c '
import json
import sys

message = json.load(sys.stdin)["choices"][0]["message"]
calls = message.get("tool_calls") or []
if len(calls) != 1:
    raise SystemExit(f"native tool-call smoke produced {len(calls)} calls: {message!r}")
function = calls[0].get("function") or {}
if function.get("name") != "aorta_tool_protocol_probe":
    raise SystemExit(f"native tool-call smoke used the wrong function: {message!r}")
arguments = function.get("arguments") or {}
if isinstance(arguments, str):
    arguments = json.loads(arguments)
if arguments.get("source") != "READY":
    raise SystemExit(f"native tool-call smoke used the wrong arguments: {message!r}")
' <<<"$tool_completion"

keep_job=true
trap - EXIT INT TERM

printf '\nQwen is ready.\n'
printf '  job:      %s\n' "$active_job"
printf '  endpoint: %s\n' "$AORTA_CHAT_VLLM_BASE_URL"
printf '  env file: %s\n' "$ENDPOINT_FILE"
printf '\nNext: scripts/chat/start_aorta_chat.sh\n'
