#!/usr/bin/env bash
# Serving-throughput benchmark for TokenSpeed. Runs INSIDE the TokenSpeed
# container; the `tokenspeed_serve` workload class is what puts it there.
#
# Where `ts_serve_probe.sh` answers "does the server come up and generate?",
# this script answers "how fast does it serve?" -- it drives TokenSpeed's own
# `tokenspeed bench serve` load generator and leaves the per-step JSON on disk
# for the workload class to turn into `WorkloadResult.metrics`.
#
# One server, N bench steps. Weight load and KV-cache warmup cost far more than
# a bench step, so re-serving per step would mostly measure model load. Starting
# the server once and benchmarking it TS_BENCH_STEPS times also gives the
# workload class real step-to-step variance to report, which is the signal that
# distinguishes a stable cell from a flaky one.
#
# THE SILENT-PASS TRAP: `tokenspeed bench serve` exits 0 no matter how many
# requests failed. `metrics.failed` is printed and written to the JSON, never
# consulted for an exit code, so a run where the engine refused every single
# request looks exactly like a clean one to any caller reading `$?`. This is the
# same trap `tokenspeed_kernel.benchmark --verify` sets for `ts_kernel_probe.sh`.
# Phase 4 below therefore re-reads each exported JSON and fails the step unless
# `failed == 0` and `completed == num_prompts`.
#
# Exit codes (a distinct band from ts_serve_probe.sh's 20-23, so a triage log
# never leaves it ambiguous which script produced the verdict):
#   0   every bench step completed with every request served
#   50  server process died during startup
#   51  readiness deadline expired
#   52  /health_generate never went healthy
#   53  a bench step failed or timed out
#   54  a bench step exported no parseable result JSON
#   55  a bench step served fewer requests than asked (the silent-pass guard)
#   64  usage / environment error (missing tokenspeed CLI, bad config)
#
# Two ports, for the reason `ts_serve_probe.sh` documents at length: `tokenspeed
# serve` is an orchestrator, so readiness lives on the control port while the
# OpenAI-compatible /v1 surface the bench drives lives on the gateway port.
#
# Env:
#   TS_MODEL              HF model id to serve            (default Qwen/Qwen3-0.6B)
#   TS_SERVED_MODEL_NAME  name the bench asks for         (default TS_MODEL)
#   TS_PORT               gateway port, OpenAI /v1        (default 8000)
#   TS_CONTROL_PORT       control port, /health           (default TS_PORT + 1)
#   TS_READY_TIMEOUT      seconds to wait for /health     (default 900)
#   TS_SERVE_ARGS         extra args for `tokenspeed serve` (word-split)
#   TS_BENCH_STEPS        measured bench repetitions      (default 1)
#   TS_BENCH_WARMUP_STEPS discarded bench repetitions run
#                         first, to absorb Triton JIT     (default 0)
#   TS_GATEWAY_STARTUP_TIMEOUT  orchestrator gateway budget (default READY_TIMEOUT)
#   TS_DRAIN_TIMEOUT      gateway drain on shutdown       (default TEARDOWN_GRACE - 5)
#   TS_NUM_PROMPTS        requests per bench step         (default 64)
#   TS_INPUT_LEN          random-dataset ISL, tokens      (default 1024)
#   TS_OUTPUT_LEN         random-dataset OSL, tokens      (default 128)
#   TS_MAX_CONCURRENCY    in-flight request cap           (default unset = unbounded)
#   TS_REQUEST_RATE       arrival rate, req/s             (default inf = all at once)
#   TS_NUM_WARMUPS        untimed warmup requests         (default 1)
#   TS_IGNORE_EOS         1 to hold OSL fixed             (default 1)
#   TS_SEED               dataset/sampling seed           (default 0)
#   TS_PERCENTILE_METRICS which metrics get percentiles   (default ttft,tpot,itl,e2el)
#   TS_METRIC_PERCENTILES which percentiles               (default 50,90,99)
#   TS_BENCH_ARGS         extra args for `tokenspeed bench serve` (word-split)
#   TS_BENCH_TIMEOUT      seconds per bench step          (default 1800)
#   TS_TEARDOWN_GRACE     seconds to wait for SIGTERM     (default 45)
#   TS_OUT_DIR            where JSON + logs are written   (default /ts-out)
#   TS_RUN_TOKEN          tag qualifying this trial's filenames. Falls back to
#                         $$, which is always 1 under a fresh PID namespace and
#                         so does not distinguish trials -- the workload class
#                         always sets it.

set -uo pipefail

MODEL="${TS_MODEL:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${TS_SERVED_MODEL_NAME:-${MODEL}}"
TOKENIZER="${TS_TOKENIZER:-${MODEL}}"
PORT="${TS_PORT:-8000}"
CONTROL_PORT="${TS_CONTROL_PORT:-$(( PORT + 1 ))}"
READY_TIMEOUT="${TS_READY_TIMEOUT:-900}"
BENCH_STEPS="${TS_BENCH_STEPS:-1}"
WARMUP_STEPS="${TS_BENCH_WARMUP_STEPS:-0}"
NUM_PROMPTS="${TS_NUM_PROMPTS:-64}"
INPUT_LEN="${TS_INPUT_LEN:-1024}"
OUTPUT_LEN="${TS_OUTPUT_LEN:-128}"
REQUEST_RATE="${TS_REQUEST_RATE:-inf}"
NUM_WARMUPS="${TS_NUM_WARMUPS:-1}"
IGNORE_EOS="${TS_IGNORE_EOS:-1}"
SEED="${TS_SEED:-0}"
PERCENTILE_METRICS="${TS_PERCENTILE_METRICS:-ttft,tpot,itl,e2el}"
METRIC_PERCENTILES="${TS_METRIC_PERCENTILES:-50,90,99}"
BENCH_TIMEOUT="${TS_BENCH_TIMEOUT:-1800}"
TEARDOWN_GRACE="${TS_TEARDOWN_GRACE:-45}"
OUT_DIR="${TS_OUT_DIR:-/ts-out}"
TOKEN="${TS_RUN_TOKEN:-$$}"
GATEWAY="http://127.0.0.1:${PORT}"
CONTROL="http://127.0.0.1:${CONTROL_PORT}"

mkdir -p "${OUT_DIR}" || {
  echo "TS_BENCH_FAIL: cannot create out dir ${OUT_DIR}"
  exit 64
}
SERVER_LOG="${OUT_DIR}/bench-server.${TOKEN}.log"

command -v tokenspeed >/dev/null 2>&1 || {
  echo "TS_BENCH_FAIL: 'tokenspeed' not on PATH inside the container"
  exit 64
}

# Validate here rather than letting a bad value reach the CLI: `--num-prompts 0`
# is accepted by the bench and reports completed=0, which Phase 4 would then
# call a served-request shortfall. A config error must not masquerade as a
# serving failure.
for pair in "BENCH_STEPS=${BENCH_STEPS}" "NUM_PROMPTS=${NUM_PROMPTS}" \
            "INPUT_LEN=${INPUT_LEN}" "OUTPUT_LEN=${OUTPUT_LEN}" \
            "BENCH_TIMEOUT=${BENCH_TIMEOUT}"; do
  name="${pair%%=*}"
  val="${pair#*=}"
  case "${val}" in
    ''|*[!0-9]*)
      echo "TS_BENCH_FAIL: ${name} must be a positive integer, got '${val}'"
      exit 64
      ;;
  esac
  if [ "${val}" -lt 1 ]; then
    echo "TS_BENCH_FAIL: ${name} must be >= 1, got '${val}'"
    exit 64
  fi
done
for pair in "NUM_WARMUPS=${NUM_WARMUPS}" "WARMUP_STEPS=${WARMUP_STEPS}"; do
  name="${pair%%=*}"
  val="${pair#*=}"
  case "${val}" in
    ''|*[!0-9]*)
      echo "TS_BENCH_FAIL: ${name} must be a non-negative integer, got '${val}'"
      exit 64
      ;;
  esac
done

echo "TS_BENCH_INFO: model=${MODEL} gateway_port=${PORT} control_port=${CONTROL_PORT}"
echo "TS_BENCH_INFO: steps=${BENCH_STEPS} num_prompts=${NUM_PROMPTS} isl=${INPUT_LEN} osl=${OUTPUT_LEN}"
echo "TS_BENCH_INFO: max_concurrency=${TS_MAX_CONCURRENCY:-unbounded} request_rate=${REQUEST_RATE} warmups=${NUM_WARMUPS}"
echo "TS_BENCH_INFO: tokenspeed=$(tokenspeed version 2>&1 | tr '\n' ' ')"
echo "TS_BENCH_INFO: rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"

SRV_PID=""
# Kill the server's whole process group: `tokenspeed serve` forks scheduler and
# detokenizer children that survive a bare kill on the parent and keep the GPUs
# pinned for the next trial in the matrix. The grace period must exceed
# TokenSpeed's own gateway drain (30s by default in OrchestratorOpts) or every
# teardown escalates to SIGKILL and leaves the next cell racing the kernel to
# reclaim the KV cache -- an amount that scales with the model, so it bites on
# big models first.

# Liveness of the process *group*, not of its leader. `tokenspeed serve` is an
# orchestrator: it forks the engine, the scheduler and the smg gateway into its
# group, and they can outlive it. Checking the leader alone means the exact case
# this teardown exists for -- children surviving the orchestrator -- reports
# "already gone" and returns without signalling anything, leaving them holding
# the GPU and the ports for the next cell in the matrix. `kill -0 -PGID`
# succeeds while any member of the group survives.
group_alive() {
  kill -0 "-${1}" 2>/dev/null
}

teardown() {
  [ -n "${SRV_PID}" ] || return 0
  group_alive "${SRV_PID}" || return 0
  echo "TS_BENCH_INFO: tearing down server pgid ${SRV_PID} (grace ${TEARDOWN_GRACE}s)"
  kill -TERM "-${SRV_PID}" 2>/dev/null
  for _ in $(seq 1 "${TEARDOWN_GRACE}"); do
    group_alive "${SRV_PID}" || {
      echo "TS_BENCH_INFO: server exited cleanly on SIGTERM"
      return 0
    }
    sleep 1
  done
  echo "TS_BENCH_INFO: server ignored SIGTERM after ${TEARDOWN_GRACE}s; sending SIGKILL"
  kill -KILL "-${SRV_PID}" 2>/dev/null
  wait "${SRV_PID}" 2>/dev/null
}

# A bash trap handler returns to where the signal interrupted it, so sharing one
# handler between EXIT and the signals would tear the server down and then carry
# on benchmarking against a server that no longer exists -- writing exports the
# host would then aggregate as if they were measurements. On the runner's SIGTERM
# that continues until the escalation to SIGKILL. Signals therefore exit
# explicitly, which re-enters teardown through EXIT; the guard above makes the
# second call a no-op.
on_signal() {
  echo "TS_BENCH_INFO: received ${1}; aborting bench"
  exit 143
}
trap teardown EXIT
trap 'on_signal SIGINT' INT
trap 'on_signal SIGTERM' TERM

# `tokenspeed serve` is an orchestrator that starts the engine and the smg
# gateway, then waits for the gateway to reach /readiness -- and that wait
# defaults to only 60s (OrchestratorOpts.gateway_startup_timeout). A cold start
# blows straight through it: downloading weights and JIT-compiling Gluon kernels
# takes minutes on a fresh node, the gateway's gRPC health check to the engine
# keeps timing out, and the orchestrator tears everything down. The wreckage
# reads as a wall of 503s and "gRPC not reachable (tried sglang, vllm, trtllm,
# mlx, tokenspeed)", which looks like a broken engine rather than an expired
# budget. Raise it to our own readiness deadline so OUR timeout is the binding
# one and a slow start is reported as a slow start.
GATEWAY_STARTUP_TIMEOUT="${TS_GATEWAY_STARTUP_TIMEOUT:-${READY_TIMEOUT}}"
# Keep the gateway drain just inside the teardown grace, or `teardown` escalates
# to SIGKILL while the gateway is still draining.
DRAIN_TIMEOUT="${TS_DRAIN_TIMEOUT:-$(( TEARDOWN_GRACE > 15 ? TEARDOWN_GRACE - 5 : 10 ))}"

serve_args=( --host 127.0.0.1 --port "${PORT}" --control-port "${CONTROL_PORT}" )
# Only supply these when the caller has not: `tokenspeed serve` takes the last
# occurrence of a repeated flag, so appending ours unconditionally would silently
# override an explicit operator choice.
case " ${TS_SERVE_ARGS:-} " in
  *" --gateway-startup-timeout"*) ;;
  *) serve_args+=( --gateway-startup-timeout "${GATEWAY_STARTUP_TIMEOUT}" ) ;;
esac
case " ${TS_SERVE_ARGS:-} " in
  *" --drain-timeout"*) ;;
  *) serve_args+=( --drain-timeout "${DRAIN_TIMEOUT}" ) ;;
esac

# Phase 1: bring the server up. setsid so it leads its own process group and
# `teardown` can signal the whole tree. Word-splitting TS_SERVE_ARGS is
# deliberate -- it carries multiple flags.
# shellcheck disable=SC2086
setsid tokenspeed serve "${MODEL}" \
  "${serve_args[@]}" \
  ${TS_SERVE_ARGS:-} >"${SERVER_LOG}" 2>&1 &
SRV_PID=$!
echo "TS_BENCH_INFO: server pid=${SRV_PID}, log=${SERVER_LOG}"

http_code() {
  curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1" 2>/dev/null || echo 000
}

# Poll on a deadline but re-check liveness every iteration: a crash during weight
# load must surface as its own failure instead of burning the whole readiness
# timeout and then reporting the wrong cause.
t0=$(date +%s)
deadline=$(( t0 + READY_TIMEOUT ))
ready=0
while [ "$(date +%s)" -lt "${deadline}" ]; do
  if ! kill -0 "${SRV_PID}" 2>/dev/null; then
    wait "${SRV_PID}"; rc=$?
    echo "TS_BENCH_FAIL: server_exited_during_startup rc=${rc}"
    echo "--- last 40 lines of server log ---"
    tail -n 40 "${SERVER_LOG}" 2>/dev/null
    exit 50
  fi
  if [ "$(http_code "${CONTROL}/health")" = "200" ]; then
    ready=1
    break
  fi
  sleep 3
done
startup_sec=$(( $(date +%s) - t0 ))

if [ "${ready}" -ne 1 ]; then
  echo "TS_BENCH_FAIL: readiness_timeout after ${startup_sec}s"
  echo "--- last 40 lines of server log ---"
  tail -n 40 "${SERVER_LOG}" 2>/dev/null
  exit 51
fi
echo "TS_BENCH_OK: health after ${startup_sec}s"
echo "TS_BENCH_METRIC: server_startup_sec=${startup_sec}"

# Phase 2: /health_generate pushes a real token through the engine, so it is the
# "can actually serve" signal. /health only proves the HTTP port is bound, and
# benchmarking a bound-but-unwired gateway would report a wall of failed
# requests instead of a clear bring-up failure.
gen_health=0
for _ in $(seq 1 10); do
  if [ "$(http_code "${CONTROL}/health_generate")" = "200" ]; then
    gen_health=1
    break
  fi
  sleep 3
done
if [ "${gen_health}" -ne 1 ]; then
  echo "TS_BENCH_FAIL: health_generate_unhealthy"
  tail -n 40 "${SERVER_LOG}" 2>/dev/null
  exit 52
fi
echo "TS_BENCH_OK: health_generate"

# Phase 3+4: bench, then audit the JSON it exported.
bench_args=(
  --backend openai
  --base-url "${GATEWAY}"
  --endpoint /v1/completions
  --model "${SERVED_MODEL_NAME}"
  --tokenizer "${TOKENIZER}"
  --dataset-name random
  --random-input-len "${INPUT_LEN}"
  --random-output-len "${OUTPUT_LEN}"
  --num-prompts "${NUM_PROMPTS}"
  --num-warmups "${NUM_WARMUPS}"
  --request-rate "${REQUEST_RATE}"
  --seed "${SEED}"
  --percentile-metrics "${PERCENTILE_METRICS}"
  --metric-percentiles "${METRIC_PERCENTILES}"
  --disable-tqdm
)
# Unbounded unless asked: `--max-concurrency` defaults to None, and passing an
# empty string would be parsed as an int and fail.
if [ -n "${TS_MAX_CONCURRENCY:-}" ]; then
  bench_args+=( --max-concurrency "${TS_MAX_CONCURRENCY}" )
fi
# Hold OSL fixed. Without this a model that emits EOS early produces short
# outputs, so TPOT/throughput describe a different amount of work per cell and
# the matrix compares cells that did not run the same benchmark.
if [ "${IGNORE_EOS}" = "1" ]; then
  bench_args+=( --ignore-eos )
fi

overall=0

# One bench invocation. $1 is the filename prefix, $2 the step index, $3 the
# label for logs. Warmup steps use a different prefix so the workload class,
# which globs "bench.<token>.step*.json", never folds them into the reported
# metrics -- while still auditing them here, because a warmup that fails to
# serve is a real failure and not something to shrug off.
run_bench_step() {
  local prefix="$1" step="$2" what="$3"
  # Deliberately NOT local: both loops read it after the call to audit what
  # this invocation exported.
  result_json="${OUT_DIR}/${prefix}.${TOKEN}.step${step}.json"
  rm -f "${result_json}"
  echo "TS_BENCH_INFO: ${what} -> ${result_json}"

  # shellcheck disable=SC2086
  timeout --signal=TERM --kill-after=60 "${BENCH_TIMEOUT}" \
    tokenspeed bench serve \
      "${bench_args[@]}" \
      --label "${prefix}-step${step}" \
      --request-id-prefix "aorta-${TOKEN}-${prefix}-s${step}" \
      --output-file "${result_json}" \
      ${TS_BENCH_ARGS:-}
  return $?
}

# The silent-pass guard, factored out because warmup and measured steps both
# need it: `tokenspeed bench serve` returns 0 even when every request errored,
# so a step's exit code proves only that the harness ran. Re-read what it wrote
# and require that it actually served what we asked for. Echoes one of
# OK / SHORTFALL / UNPARSEABLE for the caller to classify.
audit_result_json() {
  python3 - "$1" "${NUM_PROMPTS}" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
try:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
except Exception as exc:  # noqa: BLE001 - any parse failure is a hard fail
    print(f"UNPARSEABLE {type(exc).__name__}: {exc}")
    raise SystemExit(0)

if not isinstance(doc, dict):
    print(f"UNPARSEABLE top-level {type(doc).__name__}, expected object")
    raise SystemExit(0)

completed = doc.get("completed")
failed = doc.get("failed")
if not isinstance(completed, int) or not isinstance(failed, int):
    print(f"UNPARSEABLE completed={completed!r} failed={failed!r}")
    raise SystemExit(0)

if failed > 0 or completed != expected:
    print(f"SHORTFALL completed={completed} failed={failed} expected={expected}")
    raise SystemExit(0)

thr = doc.get("output_throughput")
ttft = doc.get("median_ttft_ms")
print(f"OK completed={completed} output_throughput={thr} median_ttft_ms={ttft}")
PY
}

# The first bench invocation against a fresh server pays for Triton JIT
# compilation of every kernel shape it touches, which in practice makes it
# several times slower than every later one (6.2s vs 1.1s on Qwen3-0.6B). Rolled
# into the reported metrics that single outlier dominates the mean step time and
# inflates TTFT, so a cell looks like a regression purely because it went first.
# `--num-warmups` does not help: it warms requests *within* an invocation, not
# the compile cache. These discarded invocations do.
for step in $(seq 1 "${WARMUP_STEPS}"); do
  run_bench_step "bench-warmup" "${step}" "warmup step ${step}/${WARMUP_STEPS}"
  rc=$?
  if [ "${rc}" -ne 0 ]; then
    if [ "${rc}" -eq 124 ]; then
      echo "TS_BENCH_FAIL: warmup step ${step} timed out after ${BENCH_TIMEOUT}s"
    else
      echo "TS_BENCH_FAIL: warmup step ${step} bench exited rc=${rc}"
    fi
    tail -n 40 "${SERVER_LOG}" 2>/dev/null
    exit 53
  fi

  # Audited like a measured step, and fatal for the same reason: a warmup that
  # served nothing did not warm anything, so the measured steps that follow
  # would still be paying for JIT while reporting themselves as steady-state.
  # Failing here also names the real cause instead of leaving it to look like a
  # slow first measured step.
  if [ ! -s "${result_json}" ]; then
    echo "TS_BENCH_FAIL: warmup step ${step} exported no result JSON"
    exit 54
  fi
  audit=$(audit_result_json "${result_json}")
  case "${audit}" in
    OK*)
      echo "TS_BENCH_OK: warmup step ${step} ${audit#OK }"
      ;;
    SHORTFALL*)
      echo "TS_BENCH_FAIL: warmup step ${step} served_request_shortfall ${audit#SHORTFALL }"
      tail -n 40 "${SERVER_LOG}" 2>/dev/null
      exit 55
      ;;
    *)
      echo "TS_BENCH_FAIL: warmup step ${step} result_json_unusable ${audit}"
      exit 54
      ;;
  esac
done

for step in $(seq 1 "${BENCH_STEPS}"); do
  run_bench_step "bench" "${step}" "step ${step}/${BENCH_STEPS}"
  rc=$?
  if [ "${rc}" -ne 0 ]; then
    # 124 is timeout(1)'s own "deadline expired" code.
    if [ "${rc}" -eq 124 ]; then
      echo "TS_BENCH_FAIL: step ${step} timed out after ${BENCH_TIMEOUT}s"
    else
      echo "TS_BENCH_FAIL: step ${step} bench exited rc=${rc}"
    fi
    tail -n 40 "${SERVER_LOG}" 2>/dev/null
    exit 53
  fi

  if [ ! -s "${result_json}" ]; then
    echo "TS_BENCH_FAIL: step ${step} exported no result JSON at ${result_json}"
    exit 54
  fi

  # Unlike a warmup shortfall this records the verdict and keeps going: every
  # measured step is a data point, and which steps degraded is the signal.
  audit=$(audit_result_json "${result_json}")
  case "${audit}" in
    OK*)
      echo "TS_BENCH_OK: step ${step} ${audit#OK }"
      ;;
    SHORTFALL*)
      echo "TS_BENCH_FAIL: step ${step} served_request_shortfall ${audit#SHORTFALL }"
      tail -n 40 "${SERVER_LOG}" 2>/dev/null
      overall=55
      ;;
    *)
      echo "TS_BENCH_FAIL: step ${step} result_json_unusable ${audit}"
      overall=54
      ;;
  esac
done

if [ "${overall}" -ne 0 ]; then
  echo "TS_BENCH_RESULT: fail"
  exit "${overall}"
fi

echo "TS_BENCH_RESULT: pass"
exit 0
