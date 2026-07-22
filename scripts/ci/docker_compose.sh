#!/usr/bin/env bash
# Wrapper for docker compose on self-hosted GPU runners.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/ci/docker_env.sh
source "${ROOT}/scripts/ci/docker_env.sh"

# shellcheck disable=SC2086
exec ${AORTA_COMPOSE} "$@"
