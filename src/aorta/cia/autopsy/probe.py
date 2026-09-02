from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from aorta.cia.launch.job import JobRecord

# How long to wait for the production sweep (4 h)
PROBE_TIMEOUT_SEC = 4 * 3600
POLL_INTERVAL_SEC = 30


def _ssh(node: str, cmd: str, background: bool = False) -> subprocess.CompletedProcess | None:
    full_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
                f"root@{node}", cmd + (" &" if background else "")]
    if background:
        subprocess.Popen(full_cmd)
        return None
    return subprocess.run(full_cmd, capture_output=True, text=True, timeout=60)


def run_aorta_probe(bundle_root: Path, job: JobRecord, head_node: str = "149.28.124.225") -> Path | None:
    """SSH to job node, run production Aorta sweep, wait for matrix.json.

    Returns path to matrix.json in the bundle on success, None on timeout.
    """
    aorta_output = job.aorta_output or str(bundle_root / "aorta_run")
    matrix_remote = Path(aorta_output) / "matrix.json"

    # Find recipe and sidecar on the node
    find = _ssh(
        head_node,
        f"ssh -o ConnectTimeout=10 {job.node} "
        f"'find /root /mnt -name Residual-NaN-Repro.yaml 2>/dev/null | head -1; "
        f"find /root /mnt -name Residual-NaN-Repro-sidecar.json 2>/dev/null | head -1'"
    )
    lines = (find.stdout + find.stderr).strip().splitlines() if find else []
    recipe = next((l.strip() for l in lines if "Residual-NaN-Repro.yaml" in l and "smoke" not in l), "")
    sidecar = next((l.strip() for l in lines if "sidecar.json" in l), "")

    if not recipe or not sidecar:
        print(f"[probe] could not locate recipe/sidecar on {job.node}: {lines}")
        return None

    cmd = (
        f"ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no {job.node} "
        f"'nohup aorta sweep run "
        f"--recipe {recipe} "
        f"--mitigations-file {sidecar} "
        f"--output {aorta_output} "
        f"> {aorta_output}/aorta_sweep.log 2>&1 &'"
    )
    print(f"[probe] launching production sweep on {job.node}")
    print(f"[probe]   recipe:  {recipe}")
    print(f"[probe]   output:  {aorta_output}")
    _ssh(head_node, cmd, background=False)

    # Wait for matrix.json to appear
    deadline = time.time() + PROBE_TIMEOUT_SEC
    while time.time() < deadline:
        check = _ssh(
            head_node,
            f"ssh -o ConnectTimeout=5 {job.node} 'test -f {matrix_remote} && echo EXISTS || echo WAITING'"
        )
        if check and "EXISTS" in (check.stdout + check.stderr):
            print(f"[probe] matrix.json ready on {job.node}")
            break
        elapsed = int(time.time() - (deadline - PROBE_TIMEOUT_SEC))
        print(f"[probe] waiting for matrix.json... ({elapsed}s elapsed)")
        time.sleep(POLL_INTERVAL_SEC)
    else:
        print(f"[probe] timed out after {PROBE_TIMEOUT_SEC}s waiting for matrix.json")
        return None

    # Copy matrix.json into bundle
    dest = bundle_root / "aorta" / "matrix.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    copy_cmd = (
        f"scp -o StrictHostKeyChecking=no "
        f"root@{job.node}:{matrix_remote} {dest}"
    )
    r = subprocess.run(copy_cmd, shell=True, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print(f"[probe] scp failed: {r.stderr}")
        return None

    # Also update manifest to point at the new matrix
    manifest_path = bundle_root / "manifest.yaml"
    if manifest_path.is_file():
        import yaml
        manifest = yaml.safe_load(manifest_path.read_text()) or {}
        manifest.setdefault("paths", {})["aorta_matrix"] = "aorta/matrix.json"
        manifest_path.write_text(yaml.dump(manifest, default_flow_style=False))

    print(f"[probe] matrix.json copied to {dest}")
    return dest
