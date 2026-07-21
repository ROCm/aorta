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

python - "${MANIFEST}" "${GPU_COUNT}" "${TIER}" <<'PY'
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

manifest_path = Path(sys.argv[1])
gpu_count = int(sys.argv[2])
tier = sys.argv[3]

manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
smokes = manifest.get("smokes") or []

if not smokes:
    raise SystemExit(f"No smokes declared in {manifest_path}")

ran = 0
for entry in smokes:
    name = entry["name"]
    command = entry["command"]
    min_gpus = int(entry.get("min_gpus", 1))
    is_pr = bool(entry.get("pr", False))

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
