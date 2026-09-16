#!/bin/bash
# The opaque command a probe cell wraps.
#
# The load-bearing line is --env-file. Probe mode with
# `env_passthrough_mode: file` writes the cell's mitigation env to a 0600
# probe.env per trial and exports AORTA_ENV_FILE; `docker run` does not
# forward the host environment, so without this the mitigation would be set
# on the host process and absent inside the container -- every cell would be
# the baseline under a different name, and the run would look like a clean
# negative result rather than a broken harness.
set -uo pipefail

ROOT=/apps/vikhande/probe-gpu
IMAGE=rocm/primus:v26.3

ENV_ARGS=()
if [ -n "${AORTA_ENV_FILE:-}" ] && [ -f "${AORTA_ENV_FILE}" ]; then
  ENV_ARGS=(--env-file "${AORTA_ENV_FILE}")
fi

exec docker run --rm --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size=32g --security-opt seccomp=unconfined \
  -v "${ROOT}:/out" -w /out \
  -e NAN_WS_MODE="${NAN_WS_MODE:-}" \
  "${ENV_ARGS[@]}" \
  "${IMAGE}" python /out/scripts/nan_workspace.py
