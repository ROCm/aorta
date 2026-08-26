#!/usr/bin/env bash
# Bring-up probe for a TokenSpeed serving endpoint. Runs INSIDE the TokenSpeed
# container; `host_launch.sh` is what puts it there.
#
# `tokenspeed serve` never exits, so handing it straight to `aorta sweep run`
# would always hit `timeout_per_trial` and classify as tier1:timeout -> "error"
# rather than a pass or a fail. This script gives the probe something that
# terminates: it starts the server, waits for readiness, generates once, tears
# the server down, and exits 0 only if every step worked.
#
# Verdict channel is the exit code (Tier 1). The TS_PROBE_* markers on stdout
# exist so the recipe's `custom_patterns` can name *which* step failed, and so a
# human reading stdout.log sees the same story.
#
# Exit codes:
#   0   server came up and generated
#   20  server process died during startup
#   21  readiness deadline expired
#   22  /health_generate never went healthy
#   23  completion request failed or returned no content
#   64  usage / environment error (missing tokenspeed CLI, bad config)
#
# Two ports, because `tokenspeed serve` is an orchestrator, not one server: it
# spawns an smg gateway (the OpenAI-compatible /v1 surface, on --port) in front
# of a gRPC engine, plus a TokenSpeed control server that owns /health and
# /health_generate. The control server defaults to --port + 1; we pass
# --control-port explicitly so readiness never depends on that implicit offset.
#
# Readiness is therefore checked on the control port and the completion is sent
# to the gateway port -- the gateway is the path a real client would take, so a
# gateway that is up but not wired to the engine still fails the probe.
#
# Env:
#   TS_MODEL           HF model id to serve         (default Qwen/Qwen3-0.6B)
#   TS_PORT            gateway port, OpenAI /v1     (default 8000)
#   TS_CONTROL_PORT    control port, /health        (default TS_PORT + 1)
#   TS_READY_TIMEOUT   seconds to wait for /health  (default 900)
#   TS_GEN_TIMEOUT     seconds for the completion   (default 120)
#   TS_SERVE_ARGS      extra args for `tokenspeed serve` (word-split)
#   TS_TEARDOWN_GRACE  seconds to wait for SIGTERM  (default 45)
#   TS_OUT_DIR         where the log is written     (default /ts-out)
#   TS_RUN_TOKEN       tag qualifying this trial's filenames; set by
#                      host_launch.sh. Falls back to $$, which is only distinct
#                      when running outside a fresh PID namespace.

set -uo pipefail

MODEL="${TS_MODEL:-Qwen/Qwen3-0.6B}"
PORT="${TS_PORT:-8000}"
CONTROL_PORT="${TS_CONTROL_PORT:-$(( PORT + 1 ))}"
READY_TIMEOUT="${TS_READY_TIMEOUT:-900}"
GEN_TIMEOUT="${TS_GEN_TIMEOUT:-120}"
OUT_DIR="${TS_OUT_DIR:-/ts-out}"
GATEWAY="http://127.0.0.1:${PORT}"
CONTROL="http://127.0.0.1:${CONTROL_PORT}"

mkdir -p "${OUT_DIR}" || {
  echo "TS_PROBE_FAIL: cannot create out dir ${OUT_DIR}"
  exit 64
}
# Token-qualified because TS_OUT_DIR is one host directory shared by every trial
# in the matrix: a fixed name means the next trial overwrites the log of the one
# that just failed, which is precisely the log worth keeping. TS_RUN_TOKEN is
# minted by host_launch.sh; an in-container $$ is always 1 under a fresh PID
# namespace and so would not distinguish trials at all. The failure paths below
# also tail this file into stdout, so aorta captures the tail per trial anyway.
SERVER_LOG="${OUT_DIR}/server.${TS_RUN_TOKEN:-$$}.log"

command -v tokenspeed >/dev/null 2>&1 || {
  echo "TS_PROBE_FAIL: 'tokenspeed' not on PATH inside the container"
  exit 64
}

echo "TS_PROBE_INFO: model=${MODEL} gateway_port=${PORT} control_port=${CONTROL_PORT}"
echo "TS_PROBE_INFO: ready_timeout=${READY_TIMEOUT}s gen_timeout=${GEN_TIMEOUT}s"
echo "TS_PROBE_INFO: tokenspeed=$(tokenspeed version 2>&1 | tr '\n' ' ')"
echo "TS_PROBE_INFO: rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"

SRV_PID=""
# Kill the server's whole process group: `tokenspeed serve` forks scheduler and
# detokenizer children that survive a bare kill on the parent and keep the GPUs
# pinned for the next trial in the matrix.
#
# The grace period must exceed TokenSpeed's own gateway drain (drain_timeout
# defaults to 30s in OrchestratorOpts). At 20s every teardown escalated to
# SIGKILL, which leaves the next cell in the matrix racing the kernel to reclaim
# the KV cache and the pinned host memory behind it -- an amount that scales with
# the model, so the failure shows up on big models first.
TEARDOWN_GRACE="${TS_TEARDOWN_GRACE:-45}"

# Liveness of the process *group*, not of its leader. `tokenspeed serve` forks
# scheduler and detokenizer children into the group, and they can outlive the
# leader: a leader-only check then reports "already gone" and returns without
# signalling anything, leaving those children holding the GPU for the next
# trial. `kill -0 -PGID` succeeds while any member survives.
group_alive() {
  kill -0 "-${1}" 2>/dev/null
}

teardown() {
  [ -n "${SRV_PID}" ] || return 0
  group_alive "${SRV_PID}" || return 0
  echo "TS_PROBE_INFO: tearing down server pgid ${SRV_PID} (grace ${TEARDOWN_GRACE}s)"
  kill -TERM "-${SRV_PID}" 2>/dev/null
  for _ in $(seq 1 "${TEARDOWN_GRACE}"); do
    group_alive "${SRV_PID}" || {
      echo "TS_PROBE_INFO: server exited cleanly on SIGTERM"
      return 0
    }
    sleep 1
  done
  echo "TS_PROBE_INFO: server ignored SIGTERM after ${TEARDOWN_GRACE}s; sending SIGKILL"
  kill -KILL "-${SRV_PID}" 2>/dev/null
  wait "${SRV_PID}" 2>/dev/null
}

# A bash trap handler returns to where the signal interrupted it, so sharing one
# handler between EXIT and the signals would tear the server down and then carry
# on probing a server that is gone -- and on the runner's SIGTERM the probe would
# keep working until the escalation to SIGKILL. Signals therefore exit
# explicitly, which re-enters teardown through EXIT; the guard above makes the
# second call a no-op.
on_signal() {
  echo "TS_PROBE_INFO: received ${1}; aborting probe"
  exit 143
}
trap teardown EXIT
trap 'on_signal SIGINT' INT
trap 'on_signal SIGTERM' TERM

# setsid so the server leads its own process group and `teardown` can signal the
# whole tree. Word-splitting TS_SERVE_ARGS is deliberate -- it carries multiple
# flags -- so the shellcheck warning is suppressed rather than fixed.
# shellcheck disable=SC2086
setsid tokenspeed serve "${MODEL}" \
  --host 127.0.0.1 --port "${PORT}" --control-port "${CONTROL_PORT}" \
  ${TS_SERVE_ARGS:-} >"${SERVER_LOG}" 2>&1 &
SRV_PID=$!
echo "TS_PROBE_INFO: server pid=${SRV_PID}, log=${SERVER_LOG}"

http_code() {
  curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1" 2>/dev/null || echo 000
}

# Phase 1: /health. Poll on a deadline, but re-check liveness every iteration --
# a crash during weight load must surface as its own failure instead of burning
# the full readiness timeout and reporting the wrong cause.
t0=$(date +%s)
deadline=$(( t0 + READY_TIMEOUT ))
ready=0
while [ "$(date +%s)" -lt "${deadline}" ]; do
  if ! kill -0 "${SRV_PID}" 2>/dev/null; then
    wait "${SRV_PID}"; rc=$?
    echo "TS_PROBE_FAIL: server_exited_during_startup rc=${rc}"
    echo "--- last 40 lines of server.log ---"
    tail -n 40 "${SERVER_LOG}" 2>/dev/null
    exit 20
  fi
  if [ "$(http_code "${CONTROL}/health")" = "200" ]; then
    ready=1
    break
  fi
  sleep 3
done
startup_sec=$(( $(date +%s) - t0 ))

if [ "${ready}" -ne 1 ]; then
  echo "TS_PROBE_FAIL: readiness_timeout after ${startup_sec}s"
  echo "--- last 40 lines of server.log ---"
  tail -n 40 "${SERVER_LOG}" 2>/dev/null
  exit 21
fi
echo "TS_PROBE_OK: health after ${startup_sec}s"
echo "TS_PROBE_METRIC: server_startup_sec=${startup_sec}"

# Phase 2: /health_generate actually pushes a token through the engine, so it is
# the real "can serve" signal. /health only proves the HTTP port is bound.
gen_health=0
for _ in $(seq 1 10); do
  if [ "$(http_code "${CONTROL}/health_generate")" = "200" ]; then
    gen_health=1
    break
  fi
  sleep 3
done
if [ "${gen_health}" -ne 1 ]; then
  echo "TS_PROBE_FAIL: health_generate_unhealthy"
  tail -n 40 "${SERVER_LOG}" 2>/dev/null
  exit 22
fi
echo "TS_PROBE_OK: health_generate"

# Phase 3: one real completion over the OpenAI-compatible surface. Greedy, so
# the same prompt is reproducible across cells in the matrix.
echo "TS_PROBE_INFO: issuing completion"
resp_file="${OUT_DIR}/completion.${TS_RUN_TOKEN:-$$}.json"
code=$(curl -s -o "${resp_file}" -w '%{http_code}' --max-time "${GEN_TIMEOUT}" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Name three primary colors.\"}],\"max_tokens\":32,\"temperature\":0}" \
  "${GATEWAY}/v1/chat/completions" 2>/dev/null || echo 000)

if [ "${code}" != "200" ]; then
  echo "TS_PROBE_FAIL: completion_http_${code}"
  head -c 2000 "${resp_file}" 2>/dev/null
  echo
  exit 23
fi

content=$(python3 -c '
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
    print(doc["choices"][0]["message"]["content"] or "")
except Exception as exc:
    print(f"__PARSE_ERROR__ {exc}")
' "${resp_file}" 2>/dev/null)

case "${content}" in
  __PARSE_ERROR__*)
    echo "TS_PROBE_FAIL: completion_unparseable ${content}"
    exit 23
    ;;
  "")
    echo "TS_PROBE_FAIL: completion_empty"
    exit 23
    ;;
esac

echo "TS_PROBE_OK: completion (${#content} chars)"
echo "TS_PROBE_INFO: completion_text=${content}"
echo "TS_PROBE_RESULT: pass"
exit 0
