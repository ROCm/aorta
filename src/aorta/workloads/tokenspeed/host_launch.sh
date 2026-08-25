#!/usr/bin/env bash
# Host-side launcher: runs one TokenSpeed probe trial inside the TokenSpeed
# container. This is the opaque command handed to `aorta sweep run -- ...`;
# aorta forwards it byte-for-byte and never parses it.
#
# Why a host wrapper instead of pointing aorta straight at `docker run`: the
# per-cell mitigation env vars arrive as a file whose path is only known at run
# time (AORTA_ENV_FILE, written by `env_passthrough_mode: file`). Expanding that
# into `--env-file` needs a shell, and aorta deliberately does not provide one.
#
# The mount paths must be node-local. The docker daemon runs as root, so on a
# root-squashed NFS home export `-v /home/...` fails with "mkdir /home/<user>:
# permission denied". Stage scripts, the HF cache, and the output dir under /tmp
# on the compute node.
#
# Env (all optional except TS_IMAGE):
#   TS_IMAGE        container image (required)
#   TS_SCRIPTS_DIR  host dir holding ts_*.sh          (default alongside this file)
#   TS_HF_DIR       host dir for the HF weights cache (default /tmp/ts-work/hf)
#   TS_OUT_DIR      host dir for this trial's output  (default /tmp/ts-work/out)
#   TS_GPUS         HIP_VISIBLE_DEVICES for the trial (default 0)
#   TS_ENTRY        script to run inside the container
#                                                     (default ts_serve_probe.sh)
#   TS_RUN_TOKEN    tag for this trial's output files (default: minted below;
#                                                     set it to correlate
#                                                     artifacts with a caller-
#                                                     side run id)
#   plus the TS_* knobs read by the entry script itself, forwarded below.

set -euo pipefail

: "${TS_IMAGE:?TS_IMAGE must be set to the TokenSpeed container image}"

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${TS_SCRIPTS_DIR:-${_here}}"
HF_DIR="${TS_HF_DIR:-/tmp/ts-work/hf}"
OUT_DIR="${TS_OUT_DIR:-/tmp/ts-work/out}"
GPUS="${TS_GPUS:-0}"
ENTRY="${TS_ENTRY:-ts_serve_probe.sh}"

# Token that makes this trial's output files unique. It has to be minted here,
# on the host: every `docker run` gets a fresh PID namespace, so the entry
# script is always PID 1 and an in-container `$$` would give every trial in the
# matrix the same filename -- each one overwriting the last, leaving only the
# final trial's evidence behind. The host PID is genuinely per-trial, and the
# nanosecond stamp keeps two trials distinct even if the PID is recycled.
RUN_TOKEN="${TS_RUN_TOKEN:-$$-$(date +%s%N)}"

for d in "${SCRIPTS_DIR}" "${HF_DIR}" "${OUT_DIR}"; do
  case "${d}" in
    /home/*|/nfs/*)
      echo "host_launch: refusing to bind-mount ${d}: the docker daemon cannot" >&2
      echo "  traverse the root-squashed NFS export. Stage it under /tmp." >&2
      exit 64
      ;;
  esac
done
mkdir -p "${HF_DIR}" "${OUT_DIR}"

# Pre-flight the entry script on the host. Without this, a stale or incomplete
# staging directory surfaces only as `bash: /ts-scripts/<x>: No such file or
# directory` on the container's stderr -- every cell in the matrix fails
# identically and the recipe looks broken rather than unstaged.
if [ ! -r "${SCRIPTS_DIR}/${ENTRY}" ]; then
  echo "host_launch: entry script ${ENTRY} not found in ${SCRIPTS_DIR}" >&2
  echo "  Present: $(ls "${SCRIPTS_DIR}" 2>/dev/null | tr '\n' ' ')" >&2
  echo "  Re-stage with stage_scripts.sh (it copies the current source tree)." >&2
  exit 64
fi

# Fail loudly rather than silently pulling: a pull inside a probe trial would be
# charged to the trial's walltime, and where ~/.docker/config.json holds an
# expired credential it needs a DOCKER_CONFIG override to succeed at all.
if ! docker image inspect "${TS_IMAGE}" >/dev/null 2>&1; then
  echo "host_launch: image ${TS_IMAGE} is not present locally." >&2
  echo "  Pull it first (note the credential override):" >&2
  echo "    mkdir -p /tmp/dockercfg-anon && echo '{}' > /tmp/dockercfg-anon/config.json" >&2
  echo "    DOCKER_CONFIG=/tmp/dockercfg-anon docker pull ${TS_IMAGE}" >&2
  exit 64
fi

env_file_args=()
if [ -n "${AORTA_ENV_FILE:-}" ] && [ -r "${AORTA_ENV_FILE}" ]; then
  env_file_args+=(--env-file "${AORTA_ENV_FILE}")
  echo "host_launch: forwarding cell env from ${AORTA_ENV_FILE}"
  # Echoed so the trial's stdout.log records which mitigation the cell applied;
  # the keys are aorta mitigation names, not secrets.
  sed 's/^/host_launch:   /' "${AORTA_ENV_FILE}"
else
  echo "host_launch: no AORTA_ENV_FILE; running with container defaults"
fi

# Forward only the TS_* knobs that are actually set, so the entry script's own
# defaults stay authoritative for everything else.
ts_env_args=()
for var in TS_MODEL TS_PORT TS_CONTROL_PORT TS_READY_TIMEOUT TS_GEN_TIMEOUT \
           TS_SERVE_ARGS TS_TEARDOWN_GRACE \
           TS_KERNEL_OP TS_KERNEL_NAME TS_KERNEL_DTYPE TS_KERNEL_DTYPE_ROLE \
           TS_KERNEL_ARGS TS_KERNEL_MODE TS_KERNEL_WARMUP TS_KERNEL_ITERS \
           TS_PYTEST_SUITE TS_PYTEST_K TS_PYTEST_ARGS TS_WORKSPACE TS_MIN_PASSED; do
  if [ -n "${!var:-}" ]; then
    ts_env_args+=(-e "${var}=${!var}")
  fi
done

echo "host_launch: image=${TS_IMAGE} entry=${ENTRY} gpus=${GPUS} token=${RUN_TOKEN}"

# --ipc=host + a large shm are needed because tokenspeed's scheduler and
# detokenizer talk over shared memory; the 64MB default makes the server die
# during startup with an opaque bus error.
exec docker run --rm \
  --network host \
  --ipc=host --shm-size=16g \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -e "HIP_VISIBLE_DEVICES=${GPUS}" \
  -e "HF_HOME=/hf-cache" \
  -e "TS_OUT_DIR=/ts-out" \
  -e "TS_RUN_TOKEN=${RUN_TOKEN}" \
  "${env_file_args[@]+"${env_file_args[@]}"}" \
  "${ts_env_args[@]+"${ts_env_args[@]}"}" \
  -v "${SCRIPTS_DIR}:/ts-scripts:ro" \
  -v "${HF_DIR}:/hf-cache" \
  -v "${OUT_DIR}:/ts-out" \
  "${TS_IMAGE}" \
  bash "/ts-scripts/${ENTRY}"
