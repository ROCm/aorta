#!/usr/bin/env bash
# Bring up one long-lived TokenSpeed server for a rollout-shaped driver.
#
# `tokenspeed_serve` (the workload) starts a server, benchmarks it and tears it
# down inside one container lifetime, which is right for a measurement and
# wrong for driving `aorta agent` against it: the agent needs an endpoint that
# outlives the process talking to it. So this reuses the workload's serving
# configuration rather than inventing one -- the same pinned image digest, the
# same device/group/shm/network flags from `_docker_argv`, the same
# `HF_HOME=/hf-cache` mount contract, and the same `tokenspeed serve` argv
# `ts_bench_serve.sh` builds -- and only changes the container's lifetime.
#
# Anything measured through this endpoint is therefore comparable with the
# recipes' numbers in configuration, but it is NOT a benchmark: nothing here
# audits served-request counts or guards against a silent pass, which is
# exactly what the workload adds on top. Use the recipes to measure serving,
# and this to drive a consumer.
#
#   examples/rl/serve_for_rollouts.sh up      # start, wait for health_generate
#   examples/rl/serve_for_rollouts.sh hold    # up, then block until signalled
#   examples/rl/serve_for_rollouts.sh models  # what the engine advertises
#   examples/rl/serve_for_rollouts.sh logs
#   examples/rl/serve_for_rollouts.sh down
#
# Under Slurm, use `hold` in its own step and drive from a separate `--overlap`
# step. A detached container started inside an `srun` step belongs to that
# step's cgroup, so Slurm SIGKILLs it when the step ends -- the server dies as
# soon as the bring-up command returns, and what the driver sees is a connection
# refused against an endpoint that was healthy a moment earlier. `hold` keeps
# the step alive for as long as the endpoint is meant to exist.
#
# Environment:
#   TS_MODEL      HF model id to serve   (default Qwen/Qwen3-8B)
#   TS_PORT       gateway port           (default 8000)
#   TS_CONTROL    control port           (default 8001)
#   TS_HF_HOME    host HF cache          (default $HF_HOME, else ~/.cache/huggingface)
#   TS_OUT_DIR    host scratch (/ts-out) (default $TMPDIR/aorta-rl-e2e/ts-out)
#   TS_LOG_DIR    host log directory     (default $TMPDIR/aorta-rl-e2e/logs)
#   TS_GPU        HIP_VISIBLE_DEVICES    (default 0)
#   TS_READY_SEC  readiness deadline     (default 2400)
#   TS_GRAMMAR    --grammar-backend      (default xgrammar; see below)
#   TS_SAMPLING   --sampling-backend     (default triton; see below)
set -euo pipefail

# The digest from recipes/tokenspeed/tokenspeed-serve-rollout.yaml, not the tag
# it resolved from: a moving tag would let the engine change underneath a result
# while this script still read as pinned.
IMAGE="${TS_IMAGE:-lightseekorg/tokenspeed-amd@sha256:60c12e37c01496891053b9c30c4204e5d1cf9b4b641859d3aadcbd95bccc7c78}"
MODEL="${TS_MODEL:-Qwen/Qwen3-8B}"
PORT="${TS_PORT:-8000}"
CONTROL="${TS_CONTROL:-8001}"
# Defaults are deliberately machine-neutral: the model download is large and
# slow, so an HF cache the caller already populated should be reused rather than
# refilled, and everything else is scratch. On a node whose root filesystem is
# small or full, point TMPDIR (or TS_OUT_DIR/TS_LOG_DIR) at the big mount --
# `/tmp` is the fallback, not a recommendation.
HF_HOME_HOST="${TS_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"
OUT_DIR="${TS_OUT_DIR:-${TMPDIR:-/tmp}/aorta-rl-e2e/ts-out}"
LOG_DIR="${TS_LOG_DIR:-${TMPDIR:-/tmp}/aorta-rl-e2e/logs}"
GPU="${TS_GPU:-0}"
READY_SEC="${TS_READY_SEC:-2400}"
# Defaulted ON here, against the engine's own default of `none`, because
# `LiteLLMProposer.propose` sends `response_format={"type": "json_object"}` on
# every call and TokenSpeed answers that with a 500 --
#   "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not
#    supported when the server is launched with --grammar-backend none"
# -- so a default-configured server cannot serve `aorta agent` at all. The
# failure is total rather than degraded, and it does not appear in any existing
# `tokenspeed-serve-*` recipe because `tokenspeed bench serve` drives
# /v1/completions and never asks for a response format. The two choices the
# engine offers are `xgrammar` and `none`; there is no third.
GRAMMAR="${TS_GRAMMAR:-xgrammar}"
# Defaulted to `triton` here, against the engine's own default, for the same
# class of reason as the grammar backend above -- and this one is worse, because
# it fails silently.
#
#   tokenspeed/runtime/sampling/registry.py
#     def _get_default_backend_name() -> str:
#         if current_platform().is_nvidia:
#             return "flashinfer"
#         return "greedy"
#
# On this hardware `is_nvidia` is false, so the default resolves to `greedy`.
# The greedy backend ignores `temperature`, `top_p`, `top_k` and `seed`
# entirely: the request is accepted, HTTP 200 comes back, and every completion
# is the argmax. Asking for temperature 1.2 and asking for temperature 0 return
# the same tokens, and `n=8` in one request returns eight identical choices.
#
# Nothing warns. That makes it invisible to any check that reads status codes or
# eyeballs one completion, and it is fatal to a GRPO rollout specifically: the
# group's samples are all the same string, so the advantage is identically zero
# whatever the reward says. The first end-to-end run's "zero within-group
# spread on 9 of 9 groups" had this as a cause, not just a saturated reward.
#
# `--sampling-backend` accepts greedy | triton | triton_full | flashinfer |
# flashinfer_full. flashinfer is CUDA-only, so `triton` is the portable choice
# that honours sampling parameters on ROCm.
#
# Leave this at `greedy` only if you *want* argmax decoding -- which the serving
# recipes reasonably do, since a benchmark wants reproducible output.
SAMPLING="${TS_SAMPLING:-triton}"
NAME="${TS_NAME:-ts-rollout-serve}"

http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1" 2>/dev/null || echo 000; }

# Whether this invocation currently owns the container -- created by its
# `docker run` and not yet handed over. One fact, read by every trap, because
# this leak class has now been found four times and each fix guarded one more
# code location:
#
#   1. `up` exited on a failure path with the container still named
#   2. `hold` armed its traps before `up` checked for a name collision, so a
#      collision removed *another* invocation's server
#   3. plain `up` had no interrupt guard during the readiness wait
#   4. the guard was cleared straight after `health_generate`, leaving
#      `docker logs` / `models` / `backends` unguarded
#
# Every one of those was a window, and guarding a window means the next line
# added outside it opens a fifth. So the guard is keyed on *ownership* instead:
# it is armed once, at the moment the container comes into existence, and
# released at exactly one point, where `up` decides to hand it over. Anything
# that exits in between -- a signal, `teardown_failed`, or an `exit` some later
# change adds -- goes through the same release.
_OWNED=0

_release_owned() {
  [ "${_OWNED}" = "1" ] || return 0
  echo "bring-up did not complete; removing ${NAME}" >&2
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  _OWNED=0
}

# Give up on a container this invocation started, after saving what it said.
#
# Every failure path in `up` below is reached *after* `docker run -d` has
# succeeded, and each used to exit with the container still named. Two things
# followed. The next `up` died with "container already exists; run 'down'
# first", so the script needed a manual `down` before it would work again --
# and on the readiness and health_generate paths the process is still
# *running*, so an engine nobody is driving kept its GPU. On a shared node that
# is the expensive half: the allocation looks busy and the next job cannot have
# the device.
#
# Logs are captured before removal, because `docker rm -f` takes them with it
# and a bring-up failure with no logs is not diagnosable.
teardown_failed() {  # teardown_failed <exit-code> <message>
  local code="$1" msg="$2"
  echo "FAIL: ${msg}" >&2
  docker logs --tail 60 "${NAME}" > "${LOG_DIR}/server-failure.log" 2>&1 || true
  tail -n 60 "${LOG_DIR}/server-failure.log" >&2 2>/dev/null || true
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  # Ownership ends here, so the EXIT trap does not attempt a second removal.
  _OWNED=0
  exit "${code}"
}

up() {
  local rc
  mkdir -p "${LOG_DIR}" "${HF_HOME_HOST}/hub" "${OUT_DIR}/triton-cache" "${OUT_DIR}/home"
  if [ -n "$(docker ps -aq -f "name=^${NAME}$")" ]; then
    echo "container ${NAME} already exists; run 'down' first" >&2
    exit 1
  fi

  # The cache/identity env block is `tokenspeed_serve`'s, copied whole rather
  # than reconstructed. Every entry is load-bearing under `--user`, and each one
  # fails in a way that names something other than itself:
  #
  #   HOME, TRITON_CACHE_DIR, XDG_CACHE_HOME
  #       the image's HOME is `/`, unwritable by an ordinary uid, and Triton
  #       masks the EACCES as "Triton is not supported on the current
  #       platform" -- a broken-GPU-stack message on a node whose GPUs are fine
  #   TORCHINDUCTOR_CACHE_DIR, USER, LOGNAME
  #       torch's `cache_dir()` only computes a default when the first is
  #       unset, and that default calls `getpass.getuser()`, which has no
  #       passwd entry for the uid and raises `KeyError: getpwuid()` at import
  #       of `torch._dynamo`
  #
  # Rediscovering this list costs two failed bring-ups, which is what it cost
  # here before the workload's own copy was read.
  #
  # The engine's arguments are passed as positional parameters to `bash -c`,
  # not interpolated into its script text. Every one of them --  MODEL, PORT,
  # CONTROL, GRAMMAR, SAMPLING, READY_SEC -- comes from a caller-settable
  # environment variable, and interpolating them meant the container's shell
  # reparsed their contents: a quote or a `;` in any of them ran a different
  # command inside a `--network host` container with /dev/kfd and
  # seccomp=unconfined. As positional `$1`..`$6` the shell never reparses them,
  # so the values stay data whatever they contain. `_` fills `$0`.
  #
  # --gateway-startup-timeout for the reason ts_bench_serve.sh raises it: the
  # orchestrator's own default is 60s, and a cold start blows through it while
  # the engine is still loading weights and compiling kernels. Left at our
  # readiness deadline so a slow start reports as slow rather than as a dead
  # engine.
  docker run -d \
    --name "${NAME}" \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --security-opt seccomp=unconfined \
    --network host \
    --shm-size 16g \
    -v "${HF_HOME_HOST}:/hf-cache" \
    -v "${OUT_DIR}:/ts-out" \
    --user "$(id -u):$(id -g)" \
    -e HF_HOME=/hf-cache \
    -e HF_HUB_CACHE=/hf-cache/hub \
    -e TRITON_CACHE_DIR=/ts-out/triton-cache \
    -e HOME=/ts-out/home \
    -e XDG_CACHE_HOME=/ts-out/home/.cache \
    -e TORCHINDUCTOR_CACHE_DIR=/ts-out/torchinductor \
    -e USER="$(id -un)" \
    -e LOGNAME="$(id -un)" \
    -e HIP_VISIBLE_DEVICES="${GPU}" \
    --entrypoint bash \
    "${IMAGE}" -c 'exec tokenspeed serve "$1" \
      --host 127.0.0.1 --port "$2" --control-port "$3" \
      --grammar-backend "$4" \
      --sampling-backend "$5" \
      --gateway-startup-timeout "$6"' \
    _ "${MODEL}" "${PORT}" "${CONTROL}" "${GRAMMAR}" "${SAMPLING}" "${READY_SEC}" \
    > "${LOG_DIR}/container-id.txt"
  echo "started ${NAME} ($(cut -c1-12 "${LOG_DIR}/container-id.txt"))"

  # The container now exists, so this invocation owns it until it says
  # otherwise. Armed here rather than at any later checkpoint: after
  # `docker run` is the earliest moment there is something to clean up, and it
  # is also the only moment that cannot drift as checks are added below.
  #
  # EXIT as well as INT/TERM, so an `exit` from anywhere in the bring-up path
  # releases too rather than only a signal.
  _OWNED=1
  trap '_release_owned; exit 130' INT TERM
  trap '_release_owned' EXIT

  # Poll health, but re-check liveness every iteration: a crash during weight
  # load has to surface as its own failure instead of burning the whole
  # readiness budget and then being reported as a timeout.
  local t0 deadline
  t0=$(date +%s)
  deadline=$(( t0 + READY_SEC ))
  while [ "$(date +%s)" -lt "${deadline}" ]; do
    if [ -z "$(docker ps -q -f "name=^${NAME}$")" ]; then
      teardown_failed 50 "container exited during startup after $(( $(date +%s) - t0 ))s"
    fi
    if [ "$(http_code "http://127.0.0.1:${CONTROL}/health")" = "200" ]; then
      echo "OK: /health after $(( $(date +%s) - t0 ))s"
      break
    fi
    sleep 5
  done
  if [ "$(http_code "http://127.0.0.1:${CONTROL}/health")" != "200" ]; then
    # Still running, so this is the path where the leak cost a GPU rather than
    # just an awkward name collision.
    teardown_failed 51 "readiness timeout after $(( $(date +%s) - t0 ))s"
  fi

  # /health_generate pushes a real token through the engine. /health only proves
  # the port is bound, and driving a bound-but-unwired gateway would report a
  # wall of failed requests instead of a bring-up failure.
  for _ in $(seq 1 20); do
    if [ "$(http_code "http://127.0.0.1:${CONTROL}/health_generate")" = "200" ]; then
      echo "OK: /health_generate after $(( $(date +%s) - t0 ))s"
      # Bring-up is done, so the interrupt guard stops applying: a successful
      # `up` leaves the engine running on purpose, and `hold` installs its own
      docker logs "${NAME}" > "${LOG_DIR}/server.log" 2>&1 || true
      models
      # An engine that ignores the sampling parameters, or that cannot answer a
      # `response_format` request at all, is useless for the one job this
      # server exists to do -- so bring-up has failed even though every health
      # check passed. Torn down rather than left running, for the reason the
      # other failure paths are: an engine nobody can use still holds a GPU.
      #
      # `backends`'s own status is propagated rather than a constant. It was
      # `teardown_failed 57 "...sampling_backend=greedy..."`, which meant a
      # grammar failure would have been reported as a greedy sampler under the
      # wrong exit code -- one verdict name for two defects, which is the
      # mistake this script has already been corrected for once. The specific
      # cause is on the FAIL line `backends` printed.
      backends
      rc=$?
      if [ "${rc}" -ne 0 ]; then
        teardown_failed "${rc}" "engine is not usable for rollouts; see the FAIL line above"
      fi
      # Handover: bring-up is complete and the container is meant to outlive
      # this call, so ownership ends. The single place it does -- `hold`
      # installs its own traps after this returns.
      _OWNED=0
      trap - INT TERM EXIT
      return 0
    fi
    sleep 5
  done
  teardown_failed 52 "health_generate unhealthy"
}

# The name the engine advertises, which is what has to match the litellm model
# id minus its `openai/` routing prefix. Checked rather than assumed: the prefix
# is stripped on the wire, so a mismatch here is a 404 on every request and
# reads as a broken gateway rather than a naming error.
models() {
  echo "--- advertised models (${PORT}) ---"
  curl -s --max-time 10 "http://127.0.0.1:${PORT}/v1/models" || echo "(no response)"
  echo
}

# The two backends this script overrides, read back from the engine rather than
# assumed to have applied. Both defaults are wrong for driving `aorta agent`,
# and `sampling_backend` is the one that cannot be caught any other way: a
# greedy engine answers every sampled request with HTTP 200 and the argmax, so
# there is no failure to notice downstream -- only completions that are all
# identical, which reads as a model property rather than a server setting.
# Returns non-zero when the engine reports greedy and greedy was not asked for.
# It only *reports*; the caller decides what that means, because the two callers
# want different things -- `up` owns a container and must tear it down, while
# the standalone `backends` command does not own one and must not touch it.
# One field out of /get_server_info, whitespace-tolerant.
#
# Factored out so the two backends are read the same way. The sampling check
# originally matched `'"sampling_backend":"greedy"'` with no space, which a
# pretty-printed response defeats -- `json.dumps` writes `": "` by default --
# and the grammar check added next would otherwise have been a second place to
# get that wrong.
_backend_field() {  # _backend_field <name> <json>
  printf '%s' "$2" \
    | tr ',{}' '\n\n\n' \
    | sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p" \
    | head -n 1
}

backends() {
  local info reported grammar
  # No `|| echo '{}'`. Defaulting a failed fetch to an empty object made
  # `greedy` count 0, so a curl failure, a timeout or an empty body read
  # exactly like a healthy non-greedy engine -- the absence of evidence
  # encoded as favourable evidence, which is the one thing this check exists
  # to prevent. A fetch that did not work is now a fetch that did not work.
  if ! info=$(curl -s --max-time 10 "http://127.0.0.1:${CONTROL}/get_server_info"); then
    echo "FAIL: could not read ${CONTROL}/get_server_info, so the sampling" \
         "backend is unverified; refusing to report the engine as usable" >&2
    return 57
  fi
  echo "--- backends in effect (${CONTROL}) ---"
  echo "${info}" | tr ',' '\n' | grep -E '"(sampling_backend|grammar_backend)"' \
    || echo "(no backend fields in the response)"

  # Whitespace-tolerant, and extracting the value rather than testing for one
  # spelling of it. The old `grep -c '"sampling_backend":"greedy"'` required
  # compact JSON: `json.dumps` writes `": "` by default and any pretty-printed
  # response has a space, so a genuinely greedy engine could answer in a shape
  # this check read as non-greedy. The test stub happened to emit the compact
  # form, which is why the narrow match looked fine.
  reported=$(_backend_field sampling_backend "${info}")
  if [ -z "${reported}" ]; then
    # Present-but-unreadable is the same class as unfetchable: we asked and
    # cannot say, so we must not say "fine".
    echo "FAIL: ${CONTROL}/get_server_info reported no sampling_backend, so it" \
         "is unverified; refusing to report the engine as usable" >&2
    return 57
  fi
  if [ "${reported}" = "greedy" ] && [ "${SAMPLING}" != "greedy" ]; then
    # Fatal to the caller, not a warning, and the aorta side already treats it
    # that way: `tokenspeed_serve` fails the step with exit 57,
    # `rollout_sampling_ignored`, on exactly this reading. Warning here while
    # failing there meant two halves of one defect behaved differently, which
    # is invisible unless someone reads both -- and is how it survived.
    #
    # The severity is the point rather than the tidiness. A greedy engine
    # answers every sampled request with HTTP 200 and the argmax, so a rollout
    # driven against it yields a group of identical completions and an
    # identically zero advantage: the whole run is wasted and nothing in it
    # looks wrong. Leaving the server up to be driven anyway is the failure.
    echo "FAIL: asked for --sampling-backend ${SAMPLING} but the engine reports" \
         "'greedy'; sampling parameters are silently ignored, so every" \
         "completion in a group would be the argmax and the advantage zero" >&2
    echo
    return 57
  fi

  # The grammar backend, checked for the same reason and with a larger blast
  # radius than the sampling one. `LiteLLMProposer.propose` sends
  # `response_format={"type": "json_object"}` on *every* call, and an engine on
  # `--grammar-backend none` answers that with a 500 -- so the failure is
  # total rather than degraded, and `up` was reporting such a server as ready.
  # A greedy sampler at least returns text; this returns nothing usable at all.
  #
  # Exit 59 rather than reusing 57. 57 means "the engine ignored the sampling
  # parameters", which is a different statement, and this script has already
  # been corrected once for labelling two defects with one verdict name.
  # Deliberately not 58 either: that is `rollout_sampling_backend_mismatch` on
  # the aorta side, and a reader comparing the two should not meet one number
  # with two meanings.
  grammar=$(_backend_field grammar_backend "${info}")
  if [ -z "${grammar}" ]; then
    echo "FAIL: ${CONTROL}/get_server_info reported no grammar_backend, so it" \
         "is unverified; refusing to report the engine as usable" >&2
    return 59
  fi
  if [ "${grammar}" = "none" ] && [ "${GRAMMAR}" != "none" ]; then
    echo "FAIL: asked for --grammar-backend ${GRAMMAR} but the engine reports" \
         "'none'; every request carrying a response_format returns 500, so" \
         "aorta agent cannot be served at all" >&2
    echo
    return 59
  fi
  echo
}

hold() {
  # Tear the container down on the way out, so a cancelled step does not leave
  # a live engine holding a GPU that the next allocation cannot use.
  # Armed only *after* `up` returns, and the ordering is the whole point.
  #
  # Arming first looked safer and was dangerous: `up` refuses to start when the
  # name is already taken, and it refuses by exiting -- so on a collision with
  # another invocation's live server, the trap fired on the way out and removed
  # *their* container. That trades a leak for destroying someone else's running
  # engine, which is the worse failure by a distance.
  #
  # Nothing is lost by waiting. `up`'s own failure paths go through
  # `teardown_failed`, which removes the container it created, so the window
  # this trap used to cover is already covered by the code that owns it.
  up
  trap 'echo "signalled; tearing down"; docker rm -f "${NAME}" >/dev/null 2>&1 || true; exit 0' INT TERM
  trap 'docker rm -f "${NAME}" >/dev/null 2>&1 || true' EXIT
  echo "holding ${NAME}; endpoint http://127.0.0.1:${PORT}/v1"
  while [ -n "$(docker ps -q -f "name=^${NAME}$")" ]; do
    sleep 10
  done
  echo "container ${NAME} is no longer running; releasing hold" >&2
  exit 1
}

case "${1:-up}" in
  up) up ;;
  hold) hold ;;
  models) models ;;
  # Exits non-zero on a greedy engine, so this is usable as a precondition
  # check -- but removes nothing: this invocation did not create the container
  # and must not delete one it does not own.
  backends) backends ;;
  logs) docker logs "${@:2}" "${NAME}" ;;
  down)
    docker rm -f "${NAME}" >/dev/null 2>&1 && echo "removed ${NAME}" || echo "no ${NAME}"
    ;;
  *) echo "usage: $0 {up|hold|models|backends|logs|down}" >&2; exit 64 ;;
esac
