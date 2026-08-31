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
#   TS_SERVED_MODEL_NAME  model id both sides use: the bench asks for it and,
#                         when it differs from TS_MODEL, the server is told to
#                         register it                     (default TS_MODEL)
#   TS_PORT               gateway port, OpenAI /v1        (default 8000)
#   TS_CONTROL_PORT       control port, /health           (default TS_PORT + 1)
#   TS_READY_TIMEOUT      seconds to wait for /health     (default 900)
#   TS_SERVE_ARGS         extra args for `tokenspeed serve`, as a JSON array
#                         of strings, e.g. ["--foo","bar baz"]
#   TS_BENCH_STEPS        measured bench repetitions      (default 1)
#   TS_BENCH_WARMUP_STEPS discarded bench repetitions run
#                         first, to absorb Triton JIT     (default 0)
#   TS_GATEWAY_STARTUP_TIMEOUT  orchestrator gateway budget (default READY_TIMEOUT)
#   TS_DRAIN_TIMEOUT      gateway drain on shutdown       (default TEARDOWN_GRACE - 5)
#   TS_NUM_PROMPTS        requests per bench step         (default 64)
#   TS_INPUT_LEN          random-dataset ISL, tokens      (default 1024;
#                         unset for sharegpt, which takes its lengths from the
#                         conversations)
#   TS_OUTPUT_LEN         random-dataset OSL, tokens      (default 128; likewise)
#   TS_MAX_CONCURRENCY    in-flight request cap           (default unset = unbounded)
#   TS_REQUEST_RATE       arrival rate, req/s             (default inf = all at once)
#   TS_NUM_WARMUPS        untimed warmup requests         (default 1)
#   TS_IGNORE_EOS         1 to hold OSL fixed             (default 1)
#   TS_SEED               dataset/sampling seed           (default 0)
#   TS_PERCENTILE_METRICS which metrics get percentiles   (default ttft,tpot,itl,e2el)
#   TS_METRIC_PERCENTILES which percentiles               (default 50,90,99)
#   TS_BENCH_ARGS         extra args for `tokenspeed bench serve`, as a JSON
#                         array of strings
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

# Ports and timeouts are checked before anything computes with them, unlike the
# counts below, which are checked once the run area exists. The difference is
# that these are arithmetic and `seq` operands: TS_PORT=abc would abort inside
# `$(( PORT + 1 ))` with a bash arithmetic error instead of the documented usage
# exit, and a zero timeout produces an empty wait loop, so the bench would report
# a server that never became ready without having waited at all.
#
# require_uint <label> <value> <min> <max>
require_uint() {
  case "${2}" in
    ''|*[!0-9]*)
      echo "TS_BENCH_FAIL: usage ${1} must be a positive integer, got '${2}'"
      exit 64
      ;;
    0|[1-9]*) ;;
    *)
      # All digits, but zero-padded -- and every comparison and derivation below
      # is bash arithmetic, which reads a leading zero as octal. `TS_PORT=08000`
      # passed this check and then aborted inside `$(( PORT + 1 ))` with status
      # 1, which is neither the documented usage exit nor a message about the
      # port. Rejected here rather than normalised, because a recipe writing
      # 08000 more likely means 8000 than 4096 and should say which.
      echo "TS_BENCH_FAIL: usage ${1} must not be zero-padded, got '${2}'"
      echo "  Leading zeros are read as octal by the arithmetic below; write"
      echo "  ${1}=$(printf '%s' "${2}" | sed 's/^0*//') instead."
      exit 64
      ;;
  esac
  if [ "${2}" -lt "${3}" ] || [ "${2}" -gt "${4}" ]; then
    echo "TS_BENCH_FAIL: usage ${1} must be between ${3} and ${4}, got '${2}'"
    exit 64
  fi
}

# 1-1023 are privileged and this does not run as root. 65535 is excluded when
# TS_CONTROL_PORT is unset, because the default derives from PORT + 1.
if [ -n "${TS_CONTROL_PORT:-}" ]; then
  require_uint TS_PORT "${PORT}" 1024 65535
  require_uint TS_CONTROL_PORT "${TS_CONTROL_PORT}" 1024 65535
  CONTROL_PORT="${TS_CONTROL_PORT}"
else
  require_uint TS_PORT "${PORT}" 1024 65534
  CONTROL_PORT="$(( PORT + 1 ))"
fi
if [ "${PORT}" -eq "${CONTROL_PORT}" ]; then
  echo "TS_BENCH_FAIL: usage TS_PORT and TS_CONTROL_PORT must differ, both '${PORT}'"
  echo "  Readiness is checked on the control port and the load is sent to the"
  echo "  gateway; sharing one port would let a half-wired server be benchmarked."
  exit 64
fi

# `tokenspeed serve` and `tokenspeed bench serve` both take the *last* occurrence
# of a repeated flag, so a caller's extra argument silently outranks the value
# this script and the host agreed on -- and the resulting failure never names the
# cause. `--port` in TS_SERVE_ARGS starts the gateway somewhere the readiness
# poll is not looking, so a healthy server reads as one that never came up;
# `--output-file` in TS_BENCH_ARGS sends the export somewhere neither the audit
# below nor the host's glob inspects, so a completed benchmark reads as a missing
# result; `--num-prompts` runs a different request count than the one the host
# audits against. Reordering would fix the precedence but leave the override
# silently ignored, which is its own trap, so these are rejected by name.
#
# TS_SERVE_ARGS and TS_BENCH_ARGS arrive as a JSON array of strings, not as a
# space-joined string. Joining and re-splitting destroys argument boundaries:
# a single recipe item like `--foo=a b` became two arguments, and a value
# containing `*` or `?` was glob-expanded against the container's filesystem
# before TokenSpeed ever saw it. The recipe documents these as lists, so they
# have to arrive as lists.
#
# Decoded through a NUL-separated stream, which is the one delimiter that
# cannot appear inside an argument. `mapfile` reads it straight into an array;
# a command substitution could not, since `$()` discards NUL bytes.
#
# decode_json_args <array-name> <label> <json>
decode_json_args() {
  local -n _decoded="$1"
  local label="$2" raw="${3:-}"
  _decoded=()
  [ -z "${raw}" ] && return 0

  local err
  if ! err="$(TS_ARGS_RAW="${raw}" python3 -c '
import json, os, sys
try:
    items = json.loads(os.environ["TS_ARGS_RAW"])
except ValueError as exc:
    sys.exit(f"is not valid JSON ({exc})")
if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
    sys.exit("must be a JSON array of strings")
# JSON can spell a NUL (\u0000) inside a string, and the decoder below streams
# NUL-separated. An entry carrying one would be split there into two arguments
# -- a silent change to the argv, which is what this decoding exists to prevent.
# An exec argument cannot contain a NUL in any case, so there is nothing to
# preserve: it is rejected here instead.
for index, item in enumerate(items):
    if "\0" in item:
        sys.exit(f"item {index} contains a NUL byte, which no argument can carry")
' 2>&1)"; then
    echo "TS_BENCH_FAIL: usage ${label} ${err}"
    echo "  It is serialized by the workload from the recipe's list form; set it"
    echo "  by hand only as a JSON array, e.g. '[\"--foo\",\"bar baz\"]'."
    exit 64
  fi

  mapfile -t -d '' _decoded < <(TS_ARGS_RAW="${raw}" python3 -c '
import json, os, sys
for item in json.loads(os.environ["TS_ARGS_RAW"]):
    sys.stdout.write(item + "\0")
')
}

# reject_owned_flags <label> <arg>... -- <owned>...
# Operates on the decoded argv rather than on a joined string, so a flag can be
# recognised exactly rather than by substring.
#
# Matches abbreviations as well as exact spellings. Python's argparse resolves
# any unambiguous prefix by default (allow_abbrev=True), so if the upstream CLI
# is argparse-based then `--max-conc 1` sets --max-concurrency without ever
# matching an exact-only denylist -- and that is the mislabeled pass this guard
# exists to stop: the applied load changes while the host still publishes the
# configured cap, every request completes, and the cell goes green describing a
# run that did not happen. The count-changing flags fail closed via the
# shortfall audit; the load-shape ones do not.
#
# Failing closed on prefixes costs nothing if the CLI disables abbreviation. The
# rejected spelling would then simply be an unrecognised flag, and the only
# extra argument this refuses is one that is a strict prefix of a flag the
# workload owns -- which a CLI with abbreviation enabled could not offer as a
# distinct option anyway, since it would be ambiguous with the owned one.
reject_owned_flags() {
  local label="$1"
  shift
  local -a args=() owned=()
  local seen_separator=0
  for token in "$@"; do
    if [ "${token}" = "--" ] && [ "${seen_separator}" -eq 0 ]; then
      seen_separator=1
      continue
    fi
    if [ "${seen_separator}" -eq 0 ]; then
      args+=( "${token}" )
    else
      owned+=( "${token}" )
    fi
  done
  for word in "${args[@]+"${args[@]}"}"; do
    # Split a `--flag=value` form so the flag half can be prefix-matched.
    local candidate="${word%%=*}"
    case "${candidate}" in
      --?*) ;;
      *) continue ;;
    esac
    for flag in "${owned[@]}"; do
      # `--` alone never abbreviates anything, and a bare `--x` is too short to
      # be worth guessing at; argparse needs at least one character past the
      # dashes, which "${flag}" always has.
      if [ "${candidate}" = "${flag}" ] || [ "${flag#"${candidate}"}" != "${flag}" ]; then
        echo "TS_BENCH_FAIL: usage ${label} may not set ${flag}"
        if [ "${candidate}" != "${flag}" ]; then
          echo "  '${candidate}' is a prefix of it, and argparse resolves an"
          echo "  unambiguous prefix to the full option."
        fi
        echo "  This script and the host agree on it; overriding it here would"
        echo "  desynchronise them and the run would fail for an unrelated-"
        echo "  looking reason. Use the corresponding workload_config field."
        exit 64
      fi
    done
  done
}

decode_json_args SERVE_EXTRA_ARGS TS_SERVE_ARGS "${TS_SERVE_ARGS:-}"
decode_json_args BENCH_EXTRA_ARGS TS_BENCH_ARGS "${TS_BENCH_ARGS:-}"

reject_owned_flags TS_SERVE_ARGS "${SERVE_EXTRA_ARGS[@]+"${SERVE_EXTRA_ARGS[@]}"}" -- \
  --host --port --control-port --served-model-name
# The load controls belong on this list for a sharper reason than the export
# plumbing above. `--max-concurrency 1` against a configured cap of 8 changes the
# load that was actually applied while the host still publishes 8, and both
# request-count audits pass -- 32 of 32 completed, none failed -- so the cell goes
# green carrying a number that describes a run that did not happen. Nothing
# detects it, which is the same mislabelled pass the mitigation guard exists to
# prevent, arriving through the one field that was still appended unchecked.
#
# Included whether or not this script sets the flag for a given configuration:
# `--max-concurrency` and `--request-rate` are omitted entirely at their
# defaults, so reserving only what is currently appended would leave the default
# unguarded -- and the default is what most cells run.
reject_owned_flags TS_BENCH_ARGS "${BENCH_EXTRA_ARGS[@]+"${BENCH_EXTRA_ARGS[@]}"}" -- \
  --base-url --output-file --num-prompts --label --request-id-prefix \
  --random-input-len --random-output-len --model --tokenizer --backend \
  --endpoint --dataset-name --dataset-path --percentile-metrics \
  --metric-percentiles \
  --max-concurrency --request-rate --num-warmups --ignore-eos --seed

READY_TIMEOUT="${TS_READY_TIMEOUT:-900}"
BENCH_STEPS="${TS_BENCH_STEPS:-1}"
WARMUP_STEPS="${TS_BENCH_WARMUP_STEPS:-0}"
DATASET="${TS_DATASET:-random}"
case "${DATASET}" in
  random)
    if [ -n "${TS_DATASET_PATH:-}" ]; then
      echo "TS_BENCH_FAIL: usage TS_DATASET_PATH is meaningless for TS_DATASET=random"
      echo "  The bench CLI rejects the combination; random generates its prompts"
      echo "  from TS_INPUT_LEN/TS_OUTPUT_LEN."
      exit 64
    fi
    ;;
  sharegpt)
    # Checked here, before a server is started, rather than where bench_args are
    # assembled: a dataset problem found after the model has loaded costs minutes
    # and reports as a bench failure instead of a usage error.
    if [ -z "${TS_DATASET_PATH:-}" ]; then
      echo "TS_BENCH_FAIL: usage TS_DATASET=sharegpt requires TS_DATASET_PATH"
      echo "  Without it the bench CLI would download the dataset mid-run: not"
      echo "  reproducible, and it would be measured as serving time."
      exit 64
    fi
    if [ ! -r "${TS_DATASET_PATH}" ]; then
      echo "TS_BENCH_FAIL: usage TS_DATASET_PATH (${TS_DATASET_PATH}) is not readable"
      echo "  The host is expected to mount the dataset read-only at this path."
      exit 64
    fi
    ;;
  *)
    echo "TS_BENCH_FAIL: usage TS_DATASET (${DATASET}) must be random or sharegpt"
    exit 64
    ;;
esac
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

# Day-long ceilings are sanity rails, not policy: they catch a millisecond value
# passed as seconds. The grace period bounds a `seq` loop, and at 0 teardown
# would escalate straight to SIGKILL, handing the next cell a KV cache the kernel
# is still reclaiming -- and it also derives TS_DRAIN_TIMEOUT below.
#
# The grace floor is 5 rather than 1 because it has to *contain* a gateway
# drain, and the drain is derived from it. Below 5 there is no positive drain
# that fits inside the window, so the two settings would contradict each other
# and teardown would SIGKILL a gateway that was still draining. A grace that
# small cannot do the job it exists for either way: the teardown poll loop ticks
# once per second.
require_uint TS_READY_TIMEOUT "${READY_TIMEOUT}" 1 86400
require_uint TS_TEARDOWN_GRACE "${TEARDOWN_GRACE}" 5 3600

# Whether one word is that flag, in any spelling argparse would accept for it:
# exact, `=`-attached, or an abbreviation. `reject_owned_flags` already fails
# closed on prefixes for the bench flags, and the drain guard is the same
# situation -- `serve_args: ["--drain", "600"]` is resolved by argparse to
# `--drain-timeout` while an exact-match test saw nothing, so the derived value
# was appended, the caller's won as the last occurrence, and the invariant this
# guard exists for was bypassed by a shorter spelling of the same flag.
#
# Ambiguity cannot be judged from here without the server's full option list, so
# any prefix counts. The cost of being wrong is a rejected recipe that could
# have run; the cost of the other direction is a SIGKILL mid-drain.
flag_matches() {  # flag_matches <flag> <word>
  local flag="$1" word="$2" name="${2%%=*}"
  [ "${word}" = "${flag}" ] && return 0
  case "${word}" in
    "${flag}"=*) return 0 ;;
  esac
  # An abbreviation is a long option that is a proper prefix of the flag.
  case "${name}" in
    --?*)
      case "${flag}" in
        "${name}"*) return 0 ;;
      esac
      ;;
  esac
  return 1
}
supplies_flag() {  # supplies_flag <flag> <arg>...
  local flag="$1"
  shift
  for word in "$@"; do
    flag_matches "${flag}" "${word}" && return 0
  done
  return 1
}
# The value behind such a flag, so an explicit choice can be validated rather
# than merely detected.
#
# The *last* occurrence, because that is the one `tokenspeed serve` honours.
# Returning the first meant `["--drain-timeout", "30", "--drain-timeout", "60"]`
# validated the 30 and ran the 60 -- the check passing on a value the server
# never used, which is worse than not checking at all.
#
# Non-zero when the flag is the last word with nothing after it, which
# `tokenspeed serve` would reject anyway, but which would otherwise read here as
# an empty value and fail a numeric check with a message about the wrong thing.
flag_value() {  # flag_value <flag> <arg>...
  local flag="$1" expect_value="" found="" value=""
  shift
  for word in "$@"; do
    if [ -n "${expect_value}" ]; then
      # Another option where a value should be means the earlier occurrence
      # never got one. Taking it as the value validated something the server
      # would reject at startup, reported as a broken engine rather than as the
      # malformed argument it is.
      case "${word}" in
        -*) return 2 ;;
      esac
      value="${word}"
      found=1
      expect_value=""
      continue
    fi
    if flag_matches "${flag}" "${word}"; then
      case "${word}" in
        *=*) value="${word#*=}"; found=1 ;;
        *) expect_value=1 ;;
      esac
    fi
  done
  # A trailing bare flag overrides anything before it, and has no value.
  if [ -n "${expect_value}" ] || [ -z "${found}" ]; then
    return 1
  fi
  printf '%s' "${value}"
}

# Keep the gateway drain just inside the teardown grace, or `teardown` escalates
# to SIGKILL while the gateway is still draining -- the delayed-VRAM-release
# failure this exists to avoid, arriving by the route meant to prevent it.
#
# The small-grace branch used to be a flat 10, which broke that invariant for
# every grace at or below 10. Derived from the grace in both branches now, so
# the relationship holds across the whole accepted range; the floor on
# TEARDOWN_GRACE above is what keeps it positive.
DRAIN_TIMEOUT="${TS_DRAIN_TIMEOUT:-$(( TEARDOWN_GRACE > 15 ? TEARDOWN_GRACE - 5 : TEARDOWN_GRACE - 2 ))}"
# The derivation cannot break the invariant. The two ways of setting the drain
# explicitly both could, and both bypassed the bench-flag guard: TS_DRAIN_TIMEOUT
# arrives from a mitigation, and `--drain-timeout` is a *serve* flag, so
# `reject_owned_flags` never sees it. `serve_args: ["--drain-timeout", "60"]`
# against the default 45s grace put every teardown back in the failure this
# derivation exists to prevent. Checked before the environment is probed, so a
# recipe that cannot work says so as a usage error.
require_uint TS_DRAIN_TIMEOUT "${DRAIN_TIMEOUT}" 1 "$(( TEARDOWN_GRACE - 1 ))"
if supplies_flag --drain-timeout "${SERVE_EXTRA_ARGS[@]+"${SERVE_EXTRA_ARGS[@]}"}"; then
  caller_drain="$(flag_value --drain-timeout "${SERVE_EXTRA_ARGS[@]+"${SERVE_EXTRA_ARGS[@]}"}")" || {
    echo "TS_BENCH_FAIL: usage serve_args --drain-timeout given without a value"
    echo "  (it is followed by another option, or ends the argument list)"
    exit 64
  }
  require_uint "serve_args --drain-timeout" "${caller_drain}" 1 "$(( TEARDOWN_GRACE - 1 ))"
fi

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
# ISL/OSL only where they were sent. They are `random`-only knobs, so for
# sharegpt -- which takes its lengths from the conversations -- this line was
# reporting `isl=1024 osl=128` for a request shape that never ran, in the log the
# audit is read from. The dataset is named instead, which is the shape fact that
# is true on that path.
if [ "${DATASET}" = "random" ]; then
  echo "TS_BENCH_INFO: steps=${BENCH_STEPS} num_prompts=${NUM_PROMPTS} dataset=random isl=${INPUT_LEN} osl=${OUTPUT_LEN}"
else
  echo "TS_BENCH_INFO: steps=${BENCH_STEPS} num_prompts=${NUM_PROMPTS} dataset=${DATASET} isl=from-dataset osl=from-dataset"
fi
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
# to SIGKILL while the gateway is still draining -- which is the delayed-VRAM
# -release failure this exists to avoid, arriving by the route meant to prevent
# it.
#
# The small-grace branch used to be a flat 10, which broke that invariant for
# every grace at or below 10: the drain was then equal to or longer than the
# window it was supposed to fit inside. Derived from the grace in both branches
# now, so the relationship holds across the whole accepted range. The floor on
# TEARDOWN_GRACE below is what keeps this positive.
serve_args=( --host 127.0.0.1 --port "${PORT}" --control-port "${CONTROL_PORT}" )
# The bench asks for ${SERVED_MODEL_NAME}; the server has to be the thing that
# answers to it. Only the bench half was wired, so a non-default
# served_model_name benchmarked a model id the server had never registered --
# every request 404s and the step reports as a serving failure, naming the
# gateway rather than the setting that broke it.
#
# Passed only when it differs from the model path, which is the configuration
# that was broken: at the default the server already registers TS_MODEL, so the
# flag would be a no-op on a CLI that has it and a hard failure on one that does
# not. --served-model-name is reserved in TS_SERVE_ARGS for the same reason the
# ports are: the two sides have to name one model, and spelling it there instead
# would set the server's half while the bench kept asking for the other.
if [ "${SERVED_MODEL_NAME}" != "${MODEL}" ]; then
  serve_args+=( --served-model-name "${SERVED_MODEL_NAME}" )
fi
if ! supplies_flag --gateway-startup-timeout "${SERVE_EXTRA_ARGS[@]+"${SERVE_EXTRA_ARGS[@]}"}"; then
  serve_args+=( --gateway-startup-timeout "${GATEWAY_STARTUP_TIMEOUT}" )
fi
# Validated above, next to the grace it has to fit inside; only supplied when
# the caller has not, since `tokenspeed serve` takes the last occurrence.
if ! supplies_flag --drain-timeout "${SERVE_EXTRA_ARGS[@]+"${SERVE_EXTRA_ARGS[@]}"}"; then
  serve_args+=( --drain-timeout "${DRAIN_TIMEOUT}" )
fi

# Phase 1: bring the server up. setsid so it leads its own process group and
# `teardown` can signal the whole tree.
setsid tokenspeed serve "${MODEL}" \
  "${serve_args[@]}" \
  "${SERVE_EXTRA_ARGS[@]+"${SERVE_EXTRA_ARGS[@]}"}" >"${SERVER_LOG}" 2>&1 &
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
  --dataset-name "${DATASET}"
  --num-prompts "${NUM_PROMPTS}"
  --num-warmups "${NUM_WARMUPS}"
  --request-rate "${REQUEST_RATE}"
  --seed "${SEED}"
  --percentile-metrics "${PERCENTILE_METRICS}"
  --metric-percentiles "${METRIC_PERCENTILES}"
  --disable-tqdm
)
# ISL/OSL are `random`-only knobs: the bench CLI maps them onto
# --random-input-len/--random-output-len, and sharegpt takes its lengths from the
# conversations themselves. Passing them for sharegpt would advertise a shape the
# run did not have.
if [ "${DATASET}" = "random" ]; then
  bench_args+=( --random-input-len "${INPUT_LEN}" --random-output-len "${OUTPUT_LEN}" )
else
  # Presence and readability were established during validation above, before any
  # server was started.
  bench_args+=( --dataset-path "${TS_DATASET_PATH}" )
fi
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

  timeout --signal=TERM --kill-after=60 "${BENCH_TIMEOUT}" \
    tokenspeed bench serve \
      "${bench_args[@]}" \
      --label "${prefix}-step${step}" \
      --request-id-prefix "aorta-${TOKEN}-${prefix}-s${step}" \
      --output-file "${result_json}" \
      "${BENCH_EXTRA_ARGS[@]+"${BENCH_EXTRA_ARGS[@]}"}"
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
# `type(...) is not int`, not `isinstance`: `bool` subclasses `int` in Python and
# `json` decodes `true`/`false` into it, so with the legitimate `num_prompts: 1`
# an export carrying `completed: true, failed: false` would compare equal to 1
# and 0 and satisfy this audit. Neither audit may be the lenient one.
if type(completed) is not int or type(failed) is not int:
    print(f"UNPARSEABLE completed={completed!r} failed={failed!r}")
    raise SystemExit(0)

# `!= 0` rather than `> 0`: the requirement is that none failed, and a negative
# count is not better than that, it is an export that cannot be trusted. Both
# this audit and the host's must fail closed on it, or the one that does not
# becomes the one a malformed export is read through.
if failed != 0 or completed != expected:
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
  # Announced before the step runs, so the host can tell "the measured phase
  # began" from "the measured phase produced a parseable export". Those are
  # different facts and main_work_started is defined as the first one: a step
  # that ran and wrote corrupt JSON is a benchmark failure, not a trial that
  # never got started, and classifying it as the latter hides it from the
  # triage matrix.
  echo "TS_BENCH_STEP_START: ${step}"
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
