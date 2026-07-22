#!/usr/bin/env bash
# Run GPU workload regression smokes declared in config/ci/gpu_regression_smokes.yaml.
#
# Runs inside the pinned ROCm docker container. Extend coverage by adding entries
# to the manifest -- no workflow edits required.
#
# Tier selection (env AORTA_CI_TIER, default "full"):
#   pr    -> only entries marked ``pr: true`` (fast, single-GPU PR gate).
#   full  -> every entry (nightly / workflow_dispatch).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${ROOT}/config/ci/gpu_regression_smokes.yaml"
TIER="${AORTA_CI_TIER:-full}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing regression manifest: ${MANIFEST}" >&2
  exit 1
fi

if [[ "${TIER}" != "pr" && "${TIER}" != "full" ]]; then
  echo "Invalid AORTA_CI_TIER='${TIER}' (expected 'pr' or 'full')" >&2
  exit 1
fi

cd "${ROOT}"

GPU_COUNT="$(
  python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

echo "GPU regression smokes: tier=${TIER}, detected ${GPU_COUNT} GPU(s)"

# These are GPU regression smokes: zero visible GPUs means a broken runner /
# container device mapping (e.g. missing /dev/kfd, /dev/dri). Fail loudly rather
# than silently "passing" by skipping every entry.
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "ERROR: no GPUs detected in the CI container; check device passthrough (/dev/kfd, /dev/dri)." >&2
  exit 1
fi

python - "${MANIFEST}" "${GPU_COUNT}" "${TIER}" <<'PY'
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml

manifest_path = Path(sys.argv[1])
gpu_count = int(sys.argv[2])
tier = sys.argv[3]

manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
if not isinstance(manifest, dict):
    raise SystemExit(
        f"Manifest {manifest_path} must be a YAML mapping, got {type(manifest).__name__}"
    )

smokes = manifest.get("smokes") or []
if not smokes:
    raise SystemExit(f"No smokes declared in {manifest_path}")

ran = 0
for idx, entry in enumerate(smokes):
    if not isinstance(entry, dict):
        raise SystemExit(f"smoke entry #{idx} is not a mapping: {entry!r}")
    if "name" not in entry or "command" not in entry:
        raise SystemExit(f"smoke entry #{idx} must have 'name' and 'command': {entry!r}")

    name = str(entry["name"])
    raw_command = entry["command"]
    if not isinstance(raw_command, list) or not raw_command:
        raise SystemExit(f"smoke '{name}': 'command' must be a non-empty list")

    # YAML may parse argv tokens as ints/floats/bools (e.g. an unquoted 1);
    # coerce every token to str so subprocess.run / ' '.join never raise.
    command = [str(token) for token in raw_command]

    # Resolve the `aorta` console-script token to its absolute path. We run argv
    # with no shell, so `$(which aorta)` can't be used in the manifest; more
    # importantly, `torchrun ... aorta sweep run ...` passes `aorta` as
    # torchrun's training_script, which must be a real file path (torchrun would
    # otherwise look for ./aorta). Resolving here fixes both argv[0] and the
    # torchrun training_script cases.
    aorta_path = shutil.which("aorta")
    resolved = []
    for token in command:
        if token == "aorta":
            if not aorta_path:
                raise SystemExit(f"smoke '{name}': 'aorta' not found on PATH")
            resolved.append(aorta_path)
        else:
            resolved.append(token)
    command = resolved

    # Validate types strictly: YAML `pr: "false"` would be a truthy string under
    # bool(), silently running an excluded smoke; `min_gpus: "2"` would compare
    # wrong. Require a real bool / int (bool is a subclass of int, so exclude it
    # explicitly for min_gpus).
    min_gpus_raw = entry.get("min_gpus", 1)
    if isinstance(min_gpus_raw, bool) or not isinstance(min_gpus_raw, int):
        raise SystemExit(
            f"smoke '{name}': 'min_gpus' must be an integer, got {min_gpus_raw!r}"
        )
    min_gpus = min_gpus_raw

    is_pr = entry.get("pr", False)
    if not isinstance(is_pr, bool):
        raise SystemExit(f"smoke '{name}': 'pr' must be a boolean, got {is_pr!r}")

    if tier == "pr" and not is_pr:
        print(f"SKIP {name}: not in the PR tier (pr: false)")
        continue

    if gpu_count < min_gpus:
        print(f"SKIP {name}: requires {min_gpus} GPU(s), have {gpu_count}")
        continue

    print(f"RUN  {name}: {' '.join(command)}")
    subprocess.run(command, check=True)
    ran += 1

if tier == "pr" and ran == 0:
    raise SystemExit(
        "No PR-tier smokes ran; mark at least one manifest entry with 'pr: true'."
    )

print(f"All eligible GPU regression smokes passed (tier={tier}, ran={ran}).")
PY
