#!/usr/bin/env bash
# Run GPU workload regression smokes declared in config/ci/gpu_regression_smokes.yaml.
#
# Intended for the nightly GPU CI job inside the pinned ROCm docker container.
# Extend coverage by adding entries to the manifest -- no workflow edits required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${ROOT}/config/ci/gpu_regression_smokes.yaml"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing regression manifest: ${MANIFEST}" >&2
  exit 1
fi

cd "${ROOT}"

GPU_COUNT="$(
  python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

echo "GPU regression smokes: detected ${GPU_COUNT} GPU(s)"

python - "${MANIFEST}" "${GPU_COUNT}" <<'PY'
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

manifest_path = Path(sys.argv[1])
gpu_count = int(sys.argv[2])

manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
smokes = manifest.get("smokes") or []

if not smokes:
    raise SystemExit(f"No smokes declared in {manifest_path}")

for entry in smokes:
    name = entry["name"]
    command = entry["command"]
    min_gpus = int(entry.get("min_gpus", 1))

    if gpu_count < min_gpus:
        print(f"SKIP {name}: requires {min_gpus} GPU(s), have {gpu_count}")
        continue

    print(f"RUN  {name}: {' '.join(command)}")
    subprocess.run(command, check=True)

print("All eligible GPU regression smokes passed.")
PY
