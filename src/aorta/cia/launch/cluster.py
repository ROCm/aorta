from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# Hosts that mean "run here" rather than "SSH somewhere".
LOCAL_HOSTS = {"", "local", "localhost", "127.0.0.1"}


def slurm_available() -> bool:
    """True when Slurm client commands are reachable from this host."""
    return shutil.which("sinfo") is not None


def sbatch_available() -> bool:
    return shutil.which("sbatch") is not None


def _ssh_user() -> str:
    return os.environ.get("CIA_SSH_USER") or os.environ.get("USER") or "root"


def default_jobs_root() -> str:
    """Rendezvous root for job.json/bundles. Must be readable from every node."""
    return os.environ.get("CIA_JOBS_ROOT") or os.path.expanduser("~/cia-jobs")


def search_roots() -> list[str]:
    """Directories to search for recipes, sidecars and existing launch scripts.

    Defaults to the home directory only. Sites that keep work on a shared
    filesystem elsewhere name it in ``CIA_SEARCH_ROOTS`` (colon-separated)
    rather than having their layout guessed here.
    """
    raw = os.environ.get("CIA_SEARCH_ROOTS")
    if raw:
        return [r for r in raw.split(":") if r]
    home = os.path.expanduser("~")
    return [home] if Path(home).is_dir() else []


def run_probe(host: str, cmd: str, timeout: int = 15) -> str:
    """Run a read-only probe command and return stdout+stderr. Never raises.

    Probes run locally whenever this host has Slurm or no remote host was given.
    The SSH branch remains for clusters where the agents run off-cluster and a
    head node is the only way in; many sites also forbid SSH to compute nodes,
    so prefer scheduler queries over per-node SSH in callers.
    """
    try:
        if host in LOCAL_HOSTS or slurm_available():
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout + 5
            )
        else:
            r = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-o", f"ConnectTimeout={timeout}",
                 f"{_ssh_user()}@{host}", cmd],
                capture_output=True, text=True, timeout=timeout + 5,
            )
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"ERROR: {e}"


def node_exists(node: str) -> bool:
    """True when Slurm knows about this node name."""
    if not node or not slurm_available():
        return False
    try:
        r = subprocess.run(
            ["scontrol", "show", "node", node],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0
    except Exception:
        return False


def venv_activate_path() -> str:
    """Path to the activate script of the virtualenv this agent runs in, if any."""
    venv = os.environ.get("VIRTUAL_ENV") or (sys.prefix if sys.prefix != sys.base_prefix else "")
    if not venv:
        return ""
    activate = Path(venv) / "bin" / "activate"
    return str(activate) if activate.is_file() else ""


# Environment the workload needs but the planner does not know about. sbatch's
# default --export=ALL already forwards these, but naming them in the script keeps
# a submitted job reproducible on its own and lets them cross into a container.
# LD_PRELOAD matters for ConSan: the HSA runtime dlopens librocjitsu_dbi_hooks.so
# into a process that has already loaded the host libstdc++, and the hook needs
# GLIBCXX_3.4.31 which the host only provides up to 3.4.29, so the newer library
# has to be preloaded or the tool lib silently fails to load.
FORWARDED_ENV = (
    "ROCJITSU_BUILD",
    "ROCM_PATH",
    "HSA_TOOLS_LIB",
    "HIP_VISIBLE_DEVICES",
    "LD_PRELOAD",
)


def forwarded_env() -> dict[str, str]:
    """Values of FORWARDED_ENV that are actually set in this process."""
    return {k: os.environ[k] for k in FORWARDED_ENV if os.environ.get(k)}


def containerize(command: str, working_dir: str, env_vars: dict[str, str]) -> str:
    """Wrap command in `docker run` when CIA_CONTAINER_IMAGE is set, else return it.

    Sanitizer and triage recipes can require a ROCm image that differs from the host
    stack; the GPU devices, the shared filesystem and the resolved environment all
    have to cross the container boundary or the workload cannot see its inputs.
    """
    image = os.environ.get("CIA_CONTAINER_IMAGE", "")
    if not image:
        return command

    parts = [
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri",
        "--group-add", "video",
        "--security-opt", "seccomp=unconfined",
        "--ipc=host",
    ]
    for mount in ("/apps", os.path.expanduser("~")):
        if mount and Path(mount).is_dir():
            parts += ["-v", f"{mount}:{mount}"]
    if working_dir:
        parts += ["-w", working_dir]
    for key in sorted(env_vars):
        parts += ["-e", key]
    parts += shlex.split(os.environ.get("CIA_CONTAINER_EXTRA", ""))
    parts += [image, "bash", "-lc", command]
    return shlex.join(parts)


def build_sbatch_script(
    *,
    command: str,
    job_name: str,
    log_path: str,
    working_dir: str = "",
    env_vars: dict[str, str] | None = None,
    node: str = "",
) -> str:
    """Render an sbatch script that runs command and writes all output to log_path.

    stdout and stderr both go to log_path so the watchdog has a single stream to
    tail, and scontrol reports it as StdOut for scheduler-native log discovery.
    """
    directives = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_path}",
        f"#SBATCH --error={log_path}",
        "#SBATCH --nodes=1",
        f"#SBATCH --time={os.environ.get('CIA_TIME_LIMIT', '04:00:00')}",
    ]
    partition = os.environ.get("CIA_PARTITION", "")
    if partition:
        directives.append(f"#SBATCH --partition={partition}")
    if node and node_exists(node):
        directives.append(f"#SBATCH --nodelist={node}")
    for token in shlex.split(os.environ.get("CIA_SBATCH_EXTRA", "")):
        directives.append(f"#SBATCH {token}")

    body = ["set -uo pipefail", 'echo "[cia] node=$(hostname) slurm_job=$SLURM_JOB_ID"']
    # sbatch does propagate PATH, but only whatever the submitting shell had. Sourcing
    # the venv makes the script resolve the same CLIs no matter how it was submitted.
    activate = venv_activate_path()
    if activate:
        body.append(f"source {shlex.quote(activate)}")
        body.append('echo "[cia] venv=$VIRTUAL_ENV aorta=$(command -v aorta || echo MISSING)"')
    if working_dir:
        body.append(f"cd {shlex.quote(working_dir)}")

    resolved_env: dict[str, str] = {**forwarded_env(), **{k: str(v) for k, v in (env_vars or {}).items()}}
    for key, value in resolved_env.items():
        body.append(f"export {key}={shlex.quote(value)}")

    body.append(containerize(command, working_dir, resolved_env))

    # Sanitizer positive controls exit non-zero on purpose ("guardrail not clean"),
    # so record the code for the watchdog and only fail the job when the caller has
    # not declared a non-zero exit expected.
    body.append("rc=$?")
    body.append('echo "[cia] workload exit=$rc"')
    if os.environ.get("CIA_TOLERATE_NONZERO"):
        body.append('[ "$rc" -ne 0 ] && echo "[cia] non-zero tolerated (CIA_TOLERATE_NONZERO)"')
        body.append("exit 0")
    else:
        body.append("exit $rc")

    return "\n".join(["#!/bin/bash", *directives, "", *body, ""])


def submit_sbatch(
    *,
    command: str,
    job_name: str,
    log_path: str,
    script_path: Path,
    working_dir: str = "",
    env_vars: dict[str, str] | None = None,
    node: str = "",
) -> tuple[str, str]:
    """Submit command as a batch job. Returns (slurm_job_id, error_message).

    On success error_message is empty; on failure the job id is empty. Callers
    must surface the error instead of reporting a launch that never happened.
    """
    if not sbatch_available():
        return "", "sbatch not found on PATH — is this a Slurm cluster?"

    script = build_sbatch_script(
        command=command,
        job_name=job_name,
        log_path=log_path,
        working_dir=working_dir,
        env_vars=env_vars,
        node=node,
    )
    try:
        script_path.parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o755)
    except Exception as e:
        return "", f"could not write sbatch script {script_path}: {e}"

    try:
        r = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as e:
        return "", f"sbatch invocation failed: {e}"

    if r.returncode != 0:
        return "", (r.stderr or r.stdout).strip() or f"sbatch exited {r.returncode}"

    # --parsable yields "jobid" or "jobid;cluster"
    job_id = r.stdout.strip().split(";")[0].strip()
    if not job_id:
        return "", f"could not parse job id from sbatch output: {r.stdout!r}"
    return job_id, ""
