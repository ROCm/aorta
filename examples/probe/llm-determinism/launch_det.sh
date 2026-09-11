#!/bin/bash
# The opaque user launch command handed to `aorta agent mitigate -- ...`.
#
# aorta never parses argv; the only boundary is probe.env, which the probe
# writes per trial under `env_passthrough_mode: file` and points at with
# AORTA_ENV_FILE. That matters here because the workload runs inside a
# container: `inherit` mode stamps the mitigation bundle on the wrapper's
# environment, and `docker run` does not forward the host environment, so a
# mitigation would be silently dropped and every cell would secretly be the
# baseline. `--env-file` is the documented way across that boundary.
#
# $1 selects the in-container entry point (det_launch.py by default).
set -uo pipefail

ROOT=/apps/vikhande/probe-gpu
IMAGE=rocm/primus:v26.3
ENTRY="${1:-/out/scripts/det_launch.py}"
NPROC="${DET_NPROC:-8}"

ENV_ARGS=()
CAPTURE=""
if [ -n "${AORTA_ENV_FILE:-}" ] && [ -f "${AORTA_ENV_FILE}" ]; then
  ENV_ARGS=(--env-file "${AORTA_ENV_FILE}")
  # Capture goes to a sibling tree rather than into the trial directory: the
  # container writes as uid 0 and the probe owns the trial dir as the invoking
  # user, so keeping the two apart avoids a root-owned artifact appearing
  # inside a directory aorta may rewrite on resume.
  trial_dir="$(dirname "${AORTA_ENV_FILE}")"
  CAPTURE="/out/captures/$(echo "${trial_dir#"${ROOT}/"}" | tr '/' '_')"
fi

echo "[launch_det] entry=${ENTRY} env_file=${AORTA_ENV_FILE:-<unset>} capture=${CAPTURE:-<none>}"
if [ -n "${AORTA_ENV_FILE:-}" ] && [ -f "${AORTA_ENV_FILE}" ]; then
  echo "[launch_det] probe.env contents:"
  sed 's/^/[launch_det]   /' "${AORTA_ENV_FILE}"
fi

exec docker run --rm --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size=32g --security-opt seccomp=unconfined \
  -v "${ROOT}/wt:/work:ro" -v "${ROOT}:/out" -w /out \
  "${ENV_ARGS[@]}" \
  -e PYTHONPATH=/work/src \
  -e DET_CFG="${DET_CFG}" \
  -e DET_CAPTURE_DIR="${CAPTURE}" \
  "${IMAGE}" \
  torchrun --standalone --nproc_per_node="${NPROC}" "${ENTRY}"
