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
#   TS_NETWORK      docker network for the trial       (default: bridge, except
#                                                     `host` for the serving
#                                                     entries, which resolve the
#                                                     model through the HF hub)
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
#
# A caller-supplied value is kept as a *prefix*, never as the whole token. Used
# as-is, a `TS_RUN_TOKEN` set once for a sweep -- which is the natural way to
# label a run for correlation -- gave every trial in that sweep the same
# filenames again, so they overwrote each other and concurrent trials could read
# each other's verdict files. That is the collision this token exists to
# prevent, reintroduced by the one knob meant to label it. Correlation only
# needs the prefix to be searchable, so both properties fit in one name.
#
# The prefix is sanitized to the same filename-safe alphabet the container name
# uses, because the token *is* a filename component in the container: a label
# like `feature/foo` put a path separator in the middle of every export and log
# path, so the redirection failed on a directory that does not exist -- a
# failure that reads as a broken script rather than as a label with a slash in
# it. Everything outside the alphabet becomes `-`, so the prefix stays
# recognisable in `docker ps` and in the artifact names.
RUN_TOKEN="$$-$(date +%s%N)"
if [ -n "${TS_RUN_TOKEN:-}" ]; then
  _prefix="$(printf '%s' "${TS_RUN_TOKEN}" | tr -c 'a-zA-Z0-9_.-' '-')"
  RUN_TOKEN="${_prefix}-${RUN_TOKEN}"
fi

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

# Record what actually ran, not what was asked for. A date tag is mutable: it can
# be retargeted upstream, and the check above only proves *a* image with that tag
# is present locally, so two nodes can run these same commands against different
# content and nothing in either run would say so. Printing the digest puts it in
# the trial's stdout.log, which is what makes a result attributable afterwards.
TS_IMAGE_DIGEST="$(docker image inspect "${TS_IMAGE}" \
  --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}' 2>/dev/null || echo unknown)"
echo "host_launch: image_digest=${TS_IMAGE_DIGEST}"

env_file_args=()
if [ -n "${AORTA_ENV_FILE:-}" ]; then
  # Set but unreadable is a hard error, not a fallback. The dispatcher writes
  # this file to carry the cell's mitigations; silently running container
  # defaults instead would make every cell in the matrix identical while the
  # run still reported them as distinct mitigations -- a green matrix that
  # measured one configuration four times. The no-file path below is only for
  # manual runs, where the variable is absent altogether.
  if [ ! -r "${AORTA_ENV_FILE}" ]; then
    echo "host_launch: AORTA_ENV_FILE=${AORTA_ENV_FILE} is not readable." >&2
    echo "  The cell's mitigations would be silently dropped." >&2
    exit 64
  fi
  env_file_args+=(--env-file "${AORTA_ENV_FILE}")
  echo "host_launch: forwarding cell env from ${AORTA_ENV_FILE}"
  # Names only, never values. Recording which variables a cell set is what makes
  # a result attributable, and it is enough for that -- but a mitigation may come
  # from a plugin or sidecar and is only required to resolve to dict[str, str],
  # so a value may legitimately be a credential. aorta's own registry declines to
  # repr these for exactly this reason (see src/aorta/registry/mitigations.py),
  # and stdout.log is a broadly retained artifact, so printing them here would
  # undo that. Anything that needs the values can read the env file itself.
  sed -n 's/^[[:space:]]*\([A-Za-z_][A-Za-z0-9_]*\)=.*/host_launch:   \1/p' \
    "${AORTA_ENV_FILE}"
else
  echo "host_launch: no AORTA_ENV_FILE; running with container defaults"
fi

# Forward only the TS_* knobs that are actually set, so the entry script's own
# defaults stay authoritative for everything else.
#
# The per-cell file wins over the invoking shell, because `docker run -e VAR=...`
# overrides the same name from `--env-file`. Without the check below, a
# TS_PYTEST_SUITE or TS_KERNEL_NAME left exported in the caller's shell would
# replace the selector in *every* cell -- the matrix would run one identical
# target while still labelling the cells as different, which is the failure this
# whole file exists to avoid, arriving from the one direction nothing else
# guards. Exporting a knob is still how a manual run sets it; it just cannot
# outrank a cell that names the same knob.
in_env_file() {
  [ -n "${AORTA_ENV_FILE:-}" ] || return 1
  grep -qE "^[[:space:]]*${1}=" "${AORTA_ENV_FILE}"
}

# Matching on the name alone is not enough, because some knobs are alternative
# spellings of one decision rather than independent settings. ts_kernel_probe.sh
# resolves TS_KERNEL_NAME first and only falls back to TS_KERNEL_OP, so a cell
# that selects an operator family is still displaced by a TS_KERNEL_NAME left
# exported in the caller's shell: that name appears nowhere in the cell's env
# file, the per-name check above sees no conflict and forwards it, and the
# container then runs one pinned kernel for every cell while each cell keeps its
# own label -- the exact silent substitution this block exists to stop, just
# reached through the other half of the pair. So the unit of the check is the
# group: name a cell's selector in either spelling and the shell loses both.
selector_group() {
  case "$1" in
    TS_KERNEL_OP|TS_KERNEL_NAME) echo "TS_KERNEL_OP TS_KERNEL_NAME" ;;
    *) echo "$1" ;;
  esac
}

ts_env_args=()
for var in TS_MODEL TS_PORT TS_CONTROL_PORT TS_READY_TIMEOUT TS_GEN_TIMEOUT \
           TS_SERVE_ARGS TS_TEARDOWN_GRACE \
           TS_KERNEL_OP TS_KERNEL_NAME TS_KERNEL_DTYPE TS_KERNEL_DTYPE_ROLE \
           TS_KERNEL_ARGS TS_KERNEL_MODE TS_KERNEL_WARMUP TS_KERNEL_ITERS \
           TS_PYTEST_SUITE TS_PYTEST_K TS_PYTEST_ARGS TS_WORKSPACE TS_MIN_PASSED; do
  if [ -z "${!var:-}" ]; then
    continue
  fi
  claimed=""
  for peer in $(selector_group "${var}"); do
    if in_env_file "${peer}"; then
      claimed="${peer}"
      break
    fi
  done
  if [ -n "${claimed}" ]; then
    if [ "${claimed}" = "${var}" ]; then
      echo "host_launch: ${var} is set in the environment and in the cell's env" \
        "file; keeping the cell's value"
    else
      echo "host_launch: ${var} is set in the environment but the cell's env" \
        "file selects ${claimed}; dropping ${var} so the cell's selector stands"
    fi
    continue
  fi
  ts_env_args+=(-e "${var}=${!var}")
done

# Network defaults per route, because the routes differ in what they need.
#
# The kernel and pytest probes run TokenSpeed's own code against the source tree
# already in the image, reach nothing off-node, and talk to no server -- so they
# get Docker's default bridge. Host networking would buy them nothing while
# publishing container ports on a node that may be shared.
#
# The serving routes are the documented exception. They resolve the model
# through huggingface_hub, which contacts the Hub for the revision even when the
# weights are already cached, so on a node with IPv4 forwarding disabled a
# bridged container dies during startup with "Temporary failure in name
# resolution". The probe still calls the gateway on 127.0.0.1 from inside its own
# container; it is the weight resolution that needs egress.
#
# TS_NETWORK overrides either way -- set it to `bridge` for a serving run on a
# node whose bridge does have egress, or to `host` if a kernel run ever needs it.
case "${ENTRY}" in
  *serve*) _default_network=host ;;
  *) _default_network=bridge ;;
esac
NETWORK="${TS_NETWORK:-${_default_network}}"

echo "host_launch: image=${TS_IMAGE} entry=${ENTRY} gpus=${GPUS} token=${RUN_TOKEN} network=${NETWORK}"

# Named and supervised rather than exec'd. aorta escalates to SIGKILL 10s after
# SIGTERM while container teardown is allowed up to 45s, and killing the
# foreground docker *client* does not stop the daemon-managed container: a
# timed-out trial could otherwise leave a server holding the GPUs and the
# gateway port for every later cell. The name gives us a handle to stop it.
#
# Minted per launcher process rather than taken from TS_RUN_TOKEN. The token is
# caller-controlled and may deliberately be reused to correlate artifacts across
# trials, but the name is what cleanup falls back to when the daemon never wrote
# the cidfile -- and `docker run` not writing one is exactly what happens when
# the name is already taken. A reused token would then have this trial's EXIT
# trap force-remove the *other* trial's container. The token still leads the
# name so a human reading `docker ps` can tie the two together.
#
# Sanitized: a caller-supplied TS_RUN_TOKEN is free-form, and docker only accepts
# [a-zA-Z0-9][a-zA-Z0-9_.-]* for a name.
CONTAINER="aorta-ts-$(printf '%s' "${RUN_TOKEN}" | tr -c 'a-zA-Z0-9_.-' '-').$$.${RANDOM}"

# The daemon writes the container id here, so cleanup can target the exact
# container this trial started rather than trusting that a name still refers to
# it.
#
# The file has to be absent for docker to accept it, so it lives inside a
# private directory rather than being a reserved name in the shared /tmp
# namespace. `mktemp -u` only promises the name was unused at the time it
# looked: another local user could then create that path as a symlink to a file
# holding some other container's id, and while docker would refuse to start
# against an existing cidfile, the EXIT trap below would go on to read the
# planted id and force-remove *that* container. `mktemp -d` is atomic and
# 0700, so the directory cannot be pre-created or its contents substituted.
CID_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aorta-ts-cid.XXXXXXXX")"
CID_FILE="${CID_DIR}/cid"

# Idempotent, because it runs from both a signal trap and the EXIT trap below.
cleanup_container() {
  local cid=""
  if [ -s "${CID_FILE}" ]; then
    cid="$(cat "${CID_FILE}" 2>/dev/null || true)"
  fi
  # `docker rm -f` covers both the running and already-exited cases, so no
  # separate stop is needed. Falling back to the name matters when the client
  # died before the daemon wrote the cidfile.
  docker rm -f "${cid:-${CONTAINER}}" >/dev/null 2>&1 || true
  rm -rf "${CID_DIR}" 2>/dev/null || true
}

on_signal() {
  echo "host_launch: received ${1}; stopping container ${CONTAINER}" >&2
  cleanup_container
  exit 143
}
trap 'on_signal SIGINT' INT
trap 'on_signal SIGTERM' TERM
trap 'on_signal SIGHUP' HUP

# EXIT covers what the signal traps cannot: a docker *client* that dies on its
# own -- an API disconnect, a daemon restart, an OOM-killed client -- returns
# from `wait` with the daemon still running the container, and `--rm` only
# fires once the container itself exits. On the ordinary pass/fail path the
# container is already reaped, so this costs one no-op `docker rm`, which is
# worth paying to never strand a GPU container.
#
# A SIGKILL cannot be trapped, so nothing here helps in that case; aorta sends
# SIGTERM first and escalates 10s later, which is the window this uses.
trap 'cleanup_container' EXIT

# A large shm is needed because tokenspeed's scheduler and detokenizer talk over
# shared memory and the 64MB default makes the server die during startup with an
# opaque bus error. --shm-size alone supplies that: processes inside one
# container already share a private IPC namespace, so --ipc=host would only add
# node-wide shared memory and semaphore visibility for a third-party image.
docker run --rm \
  --cidfile "${CID_FILE}" \
  --name "${CONTAINER}" \
  --network "${NETWORK}" \
  --shm-size=16g \
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
  bash "/ts-scripts/${ENTRY}" &
DOCKER_PID=$!
wait "${DOCKER_PID}"
exit $?
