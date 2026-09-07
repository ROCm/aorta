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
#   TS_HF_HOME    host HF cache          (default /apps/vikhande/cache/hf)
#   TS_OUT_DIR    host scratch (/ts-out) (default /apps/vikhande/rl-e2e/ts-out)
#   TS_LOG_DIR    host log directory     (default /apps/vikhande/rl-e2e/logs)
#   TS_GPU        HIP_VISIBLE_DEVICES    (default 0)
#   TS_READY_SEC  readiness deadline     (default 2400)
#   TS_GRAMMAR    --grammar-backend      (default xgrammar; see below)
set -euo pipefail

# The digest from recipes/tokenspeed/tokenspeed-serve-rollout.yaml, not the tag
# it resolved from: a moving tag would let the engine change underneath a result
# while this script still read as pinned.
IMAGE="${TS_IMAGE:-lightseekorg/tokenspeed-amd@sha256:60c12e37c01496891053b9c30c4204e5d1cf9b4b641859d3aadcbd95bccc7c78}"
MODEL="${TS_MODEL:-Qwen/Qwen3-8B}"
PORT="${TS_PORT:-8000}"
CONTROL="${TS_CONTROL:-8001}"
HF_HOME_HOST="${TS_HF_HOME:-/apps/vikhande/cache/hf}"
OUT_DIR="${TS_OUT_DIR:-/apps/vikhande/rl-e2e/ts-out}"
LOG_DIR="${TS_LOG_DIR:-/apps/vikhande/rl-e2e/logs}"
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
NAME="${TS_NAME:-ts-rollout-serve}"

http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1" 2>/dev/null || echo 000; }

up() {
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
    "${IMAGE}" -c "exec tokenspeed serve '${MODEL}' \
      --host 127.0.0.1 --port ${PORT} --control-port ${CONTROL} \
      --grammar-backend ${GRAMMAR} \
      --gateway-startup-timeout ${READY_SEC}" \
    > "${LOG_DIR}/container-id.txt"
  echo "started ${NAME} ($(cut -c1-12 "${LOG_DIR}/container-id.txt"))"

  # Poll health, but re-check liveness every iteration: a crash during weight
  # load has to surface as its own failure instead of burning the whole
  # readiness budget and then being reported as a timeout.
  local t0 deadline
  t0=$(date +%s)
  deadline=$(( t0 + READY_SEC ))
  while [ "$(date +%s)" -lt "${deadline}" ]; do
    if [ -z "$(docker ps -q -f "name=^${NAME}$")" ]; then
      echo "FAIL: container exited during startup after $(( $(date +%s) - t0 ))s" >&2
      docker logs --tail 60 "${NAME}" 2>&1 | tee "${LOG_DIR}/server-crash.log" >&2
      exit 50
    fi
    if [ "$(http_code "http://127.0.0.1:${CONTROL}/health")" = "200" ]; then
      echo "OK: /health after $(( $(date +%s) - t0 ))s"
      break
    fi
    sleep 5
  done
  if [ "$(http_code "http://127.0.0.1:${CONTROL}/health")" != "200" ]; then
    echo "FAIL: readiness timeout after $(( $(date +%s) - t0 ))s" >&2
    docker logs --tail 60 "${NAME}" 2>&1 >&2
    exit 51
  fi

  # /health_generate pushes a real token through the engine. /health only proves
  # the port is bound, and driving a bound-but-unwired gateway would report a
  # wall of failed requests instead of a bring-up failure.
  for _ in $(seq 1 20); do
    if [ "$(http_code "http://127.0.0.1:${CONTROL}/health_generate")" = "200" ]; then
      echo "OK: /health_generate after $(( $(date +%s) - t0 ))s"
      docker logs "${NAME}" > "${LOG_DIR}/server.log" 2>&1 || true
      models
      return 0
    fi
    sleep 5
  done
  echo "FAIL: health_generate unhealthy" >&2
  docker logs --tail 60 "${NAME}" 2>&1 >&2
  exit 52
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

hold() {
  # Tear the container down on the way out, so a cancelled step does not leave
  # a live engine holding a GPU that the next allocation cannot use.
  trap 'echo "signalled; tearing down"; docker rm -f "${NAME}" >/dev/null 2>&1 || true; exit 0' INT TERM
  up
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
  logs) docker logs "${@:2}" "${NAME}" ;;
  down)
    docker rm -f "${NAME}" >/dev/null 2>&1 && echo "removed ${NAME}" || echo "no ${NAME}"
    ;;
  *) echo "usage: $0 {up|hold|models|logs|down}" >&2; exit 64 ;;
esac
