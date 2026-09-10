from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Hosts that mean "run here" rather than "SSH somewhere".
LOCAL_HOSTS = {"", "local", "localhost", "127.0.0.1"}


def slurm_available() -> bool:
    """True when Slurm client commands are reachable from this host."""
    return shutil.which("sinfo") is not None


def sbatch_available() -> bool:
    return shutil.which("sbatch") is not None


def ssh_user() -> str:
    """The account cluster SSH runs as: CIA_SSH_USER, else the caller's own."""
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


def quoted_search_roots() -> str:
    """The search roots as one shell-safe argument list.

    These are interpolated into ``find`` commands that run through a shell, so
    a root containing a space arrives as two paths and a root containing a
    semicolon arrives as a second command. They come from CIA_SEARCH_ROOTS,
    which is the operator's to set -- but a probe should not be the thing that
    decides whether their directory name is safe to write.
    """
    return " ".join(shlex.quote(root) for root in search_roots()) or "~"


#: Assignments whose *name* says the value is a credential. Matched on the name
#: rather than the value, because a token looks like any other opaque string.
_SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    \b[A-Z0-9_]*
    (TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API[_-]?KEY|ACCESS[_-]?KEY|
     PRIVATE[_-]?KEY|SESSION|COOKIE|BEARER|AUTH)
    [A-Z0-9_]*
    \s*=
    """,
)


def launch_script_summary(roots: str, limit: int) -> str:
    """A probe that reports how jobs are submitted, not what the scripts contain.

    This used to be ``find ... | xargs head -25``, which read the first 25 lines
    of up to eight shell scripts under the user's home directory and put them in
    an LLM prompt bound for whatever LITELLM_API_BASE points at. The first lines
    of a personal ``.sh`` are where ``export ..._API_KEY=`` lives, so the part
    that got sent was the part worth keeping.

    What the planner actually needs from these files is the #SBATCH directives:
    the partition, the time limit, the gres. So that is all this returns, with
    the filename for context.
    """
    return (
        rf"find {roots} -maxdepth 4 \( -name '*.sbatch' -o -name '*.slurm' -o -name '*.sh' \) "
        rf"2>/dev/null | head -{limit} | while IFS= read -r f; do "
        r'echo "== $f"; '
        r"""grep -hE '^[[:space:]]*#(SBATCH|PBS)' "$f" 2>/dev/null | head -20; """
        "done"
    )


def scrub_secrets(text: str) -> str:
    """Drop lines that assign something named like a credential.

    The probe above should not produce any, being limited to directive lines.
    This is the second gate: a directive line is not supposed to carry a
    secret, but "supposed to" is not a property of somebody else's file, and
    the cost of being wrong is a credential in a third party's logs.
    """
    kept = [
        line for line in text.splitlines() if not _SECRET_ASSIGNMENT.search(line)
    ]
    return "\n".join(kept)


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
                 f"{ssh_user()}@{host}", cmd],
                capture_output=True, text=True, timeout=timeout + 5,
            )
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"ERROR: {e}"


def node_placement(node: str) -> tuple[bool, str]:
    """Whether *node* can be pinned, and why not when it cannot.

    Three different things used to arrive here as one ``False``: a node Slurm
    has never heard of, a Slurm that is not reachable at all, and an scontrol
    that timed out. They mean different things to whoever asked for that node,
    and only the first is about the node.

    Returns (can_pin, reason); *reason* is empty when there is nothing to say.
    """
    if not node:
        return False, ""
    if not slurm_available():
        return False, "Slurm is not reachable from here"
    try:
        r = subprocess.run(
            ["scontrol", "show", "node", node],
            capture_output=True, text=True, timeout=15,
        )
    except Exception as exc:
        return False, f"scontrol could not be asked ({type(exc).__name__})"
    if r.returncode != 0:
        return False, "Slurm does not know that node"
    return True, ""


def node_exists(node: str) -> bool:
    """True when Slurm knows about this node name."""
    return node_placement(node)[0]


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


#: What a shell will accept as a variable name, and nothing else.
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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
    can_pin, why_not = node_placement(node)
    if can_pin:
        directives.append(f"#SBATCH --nodelist={node}")
    for token in shlex.split(os.environ.get("CIA_SBATCH_EXTRA", "")):
        directives.append(f"#SBATCH {token}")

    body = ["set -uo pipefail", 'echo "[cia] node=$(hostname) slurm_job=$SLURM_JOB_ID"']
    if node and not can_pin:
        # A repro pinned to one machine that quietly runs on another reports on
        # the wrong machine, and the report reads exactly the same either way.
        # The echo matters more than the warning: the job log is in the bundle
        # when somebody reads the verdict, and a warning is long gone by then.
        log.warning(
            "Requested node %s was not pinned: %s. The job will run wherever the "
            "scheduler puts it.", node, why_not,
        )
        body.append(
            "echo "
            + shlex.quote(
                f"[cia] requested node {node} not pinned: {why_not}; "
                "the scheduler chose this one"
            )
        )
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
        # The name is rendered verbatim, so quoting the value is not enough:
        # these come from LaunchPlan.env_vars, which is model output, and a key
        # like ``X; curl ... #`` renders as `export X; curl ... #=value` --
        # an export, then a command, then a comment eating the rest of the line.
        # A name that is not a shell identifier means the plan is malformed, so
        # this refuses rather than dropping it: a job that runs without a
        # variable it was told to set fails later and somewhere else.
        if _ENV_NAME.fullmatch(key) is None:
            raise ValueError(f"invalid environment variable name: {key!r}")
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

    try:
        script = build_sbatch_script(
            command=command,
            job_name=job_name,
            log_path=log_path,
            working_dir=working_dir,
            env_vars=env_vars,
            node=node,
        )
    except ValueError as e:
        # Rendering rejects a malformed plan. This function promises
        # (job_id, error), and callers surface that error -- a traceback out of
        # here would instead read to them as a launch that never happened.
        return "", f"could not render the sbatch script: {e}"

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
