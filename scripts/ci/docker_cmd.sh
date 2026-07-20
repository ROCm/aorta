#!/usr/bin/env bash
# Run docker with the resolved invocation (direct or sudo -n).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/ci/docker_env.sh
source "${ROOT}/scripts/ci/docker_env.sh"

# shellcheck disable=SC2086
exec ${AORTA_DOCKER} "$@"
