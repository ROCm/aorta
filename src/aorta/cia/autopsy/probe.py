from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from aorta.cia.launch.cluster import ssh_user
from aorta.cia.launch.planner import check_recipe_mode
from aorta.cia.launch.job import JobRecord

# How long to wait for the production sweep (4 h)
PROBE_TIMEOUT_SEC = 4 * 3600
POLL_INTERVAL_SEC = 30


def default_head_node() -> str:
    """The host scheduler queries are issued from, or "" when none is set.

    No address is shipped. One site's head node was the default here, which
    made every other site's installation quietly wrong -- and put that site's
    address in a public repository.
    """
    return os.environ.get("CIA_SSH_HOST", "")


def _ssh(node: str, cmd: str, background: bool = False) -> subprocess.CompletedProcess | None:
    full_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
                f"{ssh_user()}@{node}", cmd + (" &" if background else "")]
    if background:
        subprocess.Popen(full_cmd)
        return None
    return subprocess.run(full_cmd, capture_output=True, text=True, timeout=60)


def resolve_recipe(bundle_root: Path, job: JobRecord) -> tuple[str, str]:
    """The recipe this job ran, and its sidecar, or ("", "") if unrecorded.

    This used to run ``find /root /mnt -name Residual-NaN-Repro.yaml`` on the
    node and sweep whatever came back, whichever job was under autopsy. On a
    machine where that demo happened to be installed, a wait-hazard escalation
    swept the NaN demo and folded its results into the verdict -- and the
    escalation triggers below 0.85 confidence, which is where a static finding
    lands. A canned recipe is worse than no recipe, because nothing about the
    verdict says the wrong thing was run.

    ``job.recipe`` is a label a person reads ("rms_norm NaN at step 5"), so it
    is no use here; ``recipe_path`` is the file. The manifest is consulted
    second, for a bundle assembled by something that did not write the job
    record.
    """
    if job.recipe_path:
        return job.recipe_path, job.sidecar_path

    manifest_path = bundle_root / "manifest.yaml"
    if manifest_path.is_file():
        import yaml

        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        paths = manifest.get("paths") or {}
        recipe = str(paths.get("recipe") or "")
        if recipe:
            return recipe, str(paths.get("mitigations") or "")

    return "", ""


def run_aorta_probe(bundle_root: Path, job: JobRecord, head_node: str = "") -> Path | None:
    """SSH to job node, run production Aorta sweep, wait for matrix.json.

    *head_node* falls back to the job's own and then to CIA_SSH_HOST. With
    none of the three set there is nowhere to send the query, so this
    returns None rather than guessing at an address.

    Returns path to matrix.json in the bundle on success, None on timeout.
    """
    head_node = head_node or job.head_node or default_head_node()
    if not head_node:
        print("[probe] no head node configured (set CIA_SSH_HOST); skipping probe")
        return None
    aorta_output = job.aorta_output or str(bundle_root / "aorta_run")
    matrix_remote = Path(aorta_output) / "matrix.json"

    recipe, sidecar = resolve_recipe(bundle_root, job)
    if not recipe:
        print(
            "[probe] this job records no recipe file, so there is nothing to "
            "re-run; not escalating"
        )
        return None

    sweep = ["nohup aorta sweep run", f"--recipe {shlex.quote(recipe)}"]
    if sidecar and check_recipe_mode(recipe).get("mode") != "sanitizer":
        sweep.append(f"--mitigations-file {shlex.quote(sidecar)}")
    sweep.append(f"--output {shlex.quote(aorta_output)}")

    cmd = (
        f"ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no {shlex.quote(job.node)} "
        + shlex.quote(" ".join(sweep) + f" > {aorta_output}/aorta_sweep.log 2>&1 &")
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
        f"{ssh_user()}@{job.node}:{matrix_remote} {dest}"
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
