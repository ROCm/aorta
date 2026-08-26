#!/usr/bin/env bash
# Copy the TokenSpeed probe scripts to a node-local staging directory.
#
# Needed because the docker daemon runs as root, so on a root-squashed NFS home
# `-v /home/<user>/...` fails with "permission denied" and the scripts have to
# live on the compute node's own disk. Editing the source tree and forgetting to
# re-copy is the easiest way to waste a matrix run, so this always mirrors the
# full set rather than copying files one at a time.
#
# Run this ON the compute node. /tmp is per-node, so staging from a login node
# writes a directory the trials never see -- and if the compute node already has
# an older copy, the run silently succeeds against stale scripts.
#
# The .py helpers are staged too, so that list_harness_coverage.py (which runs
# inside the container) can be bind-mounted from here without a second copy.
#
# Usage:
#   bash stage_scripts.sh [dest]     # dest defaults to /tmp/ts-work/scripts

set -euo pipefail

dest="${1:-/tmp/ts-work/scripts}"
src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${dest}" in
  /home/*|/nfs/*)
    echo "stage_scripts: ${dest} is on NFS; the docker daemon cannot read it." >&2
    echo "  Pick a node-local path such as /tmp/ts-work/scripts." >&2
    exit 64
    ;;
esac

mkdir -p "${dest}"

# Clear the set this script owns before copying, so the destination really is a
# mirror. A plain `cp` only ever adds: a probe that was renamed or deleted
# upstream stays behind in the staging directory, and because the recipes name
# their entry script by filename, a stale copy is not just dead weight -- it is
# still executable, and a run pointed at the old name succeeds against code that
# no longer exists in the tree. Scoped to *.sh / *.py rather than wiping the
# directory, which may also hold caller-owned files (an env file, an out dir).
rm -f "${dest}"/*.sh "${dest}"/*.py
rm -rf "${dest}/__pycache__"

cp "${src}"/*.sh "${src}"/*.py "${dest}/"
chmod +x "${dest}"/*.sh

echo "stage_scripts: staged to ${dest}"
for f in "${dest}"/*.sh "${dest}"/*.py; do
  # Syntax-check here rather than discovering a typo inside a container, where
  # the only symptom is a failed trial.
  case "${f}" in
    *.sh) checker=(bash -n "${f}") ;;
    *.py) checker=(python3 -m py_compile "${f}") ;;
  esac
  "${checker[@]}" || {
    echo "stage_scripts: syntax error in ${f}" >&2
    exit 65
  }
  printf '  %s (%s bytes)\n' "$(basename "${f}")" "$(stat -c%s "${f}")"
done
# py_compile drops caches next to the source; they would otherwise be copied
# into the container on the next run for no reason.
rm -rf "${dest}/__pycache__"
echo "stage_scripts: syntax OK"
