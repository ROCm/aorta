from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import dspy
import yaml

from aorta.cia.launch.cluster import run_probe, search_roots, venv_activate_path
from aorta.cia.llm import ensure_configured


# ---------------------------------------------------------------------------
# Planner tools
# ---------------------------------------------------------------------------

def check_recipe_mode(recipe_path: str) -> dict[str, Any]:
    """Read a recipe's `mode` and report which sweep flags it accepts.

    A `mode: sanitizer` recipe rejects --mitigations-file and every matrix axis;
    passing one makes `aorta sweep run` abort before it does any work. Sanitizer
    recipes also carry their own environments, so they need no sidecar at all.
    """
    if not recipe_path:
        return {"mode": "", "error": "no recipe path given"}
    try:
        data = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8")) or {}
    except Exception as e:
        return {"mode": "", "error": f"could not read {recipe_path}: {e}"}

    mode = str(data.get("mode") or "triage")
    if mode == "sanitizer":
        plan = data.get("sanitizer_plan") or {}
        return {
            "mode": mode,
            "allowed_flags": ["--recipe", "--output", "--dry-run", "--verbose"],
            "rejected_flags": ["--mitigations-file", "--workload", "--trials", "--steps", "--ticket"],
            "needs_sidecar": False,
            "target": plan.get("target", ""),
            "sanitizers": plan.get("sanitizers", []),
            "ticket": data.get("ticket", ""),
        }
    return {
        "mode": mode,
        "allowed_flags": ["--recipe", "--mitigations-file", "--output"],
        "rejected_flags": [],
        "needs_sidecar": True,
        "ticket": data.get("ticket", ""),
        "workload": data.get("workload", ""),
    }


def check_torchrun(host: str, node: str) -> dict[str, Any]:
    """Check torchrun availability and version in the current environment."""
    out = run_probe(host, "which torchrun 2>/dev/null && torchrun --version 2>/dev/null || echo NOT_FOUND")
    return {"available": "NOT_FOUND" not in out and "ERROR" not in out, "output": out[:200]}


def check_aorta_install(host: str, recipe: str = "") -> dict[str, Any]:
    """Check for the aorta CLI and locate a recipe plus the sidecar that belongs to it.

    Availability comes from `command -v aorta` alone. Recipe paths contain the
    substring "aorta", so testing the combined output reports the CLI as present
    whenever the repos are merely checked out.

    A sidecar defines the environments and mitigations its recipe references, so the
    sidecar sitting beside the recipe wins over any generic example — pairing a
    recipe with the wrong sidecar fails at load time with UnknownEnvironmentError.
    """
    roots = " ".join(search_roots()) or "~"
    cli = run_probe(host, "command -v aorta 2>/dev/null || echo NOT_FOUND")
    available = bool(cli.strip()) and "NOT_FOUND" not in cli and "ERROR" not in cli
    if not available:
        # The batch script sources this venv, so an aorta living there is what the
        # job will resolve even when the submitting shell has no venv on PATH.
        activate = venv_activate_path()
        if activate and (Path(activate).parent / "aorta").is_file():
            cli, available = str(Path(activate).parent / "aorta"), True
    version = run_probe(host, f"{shlex.quote(cli.strip())} --version 2>/dev/null | head -2") if available else ""

    stem = Path(recipe or "").name
    for suffix in (".yaml", ".yml"):
        stem = stem.removesuffix(suffix)
    pattern = f"{stem}.yaml" if stem else "*.yaml"
    found = run_probe(host, rf"find {roots} -maxdepth 6 -name {shlex.quote(pattern)} 2>/dev/null | head -3")
    recipe_path = next((l.strip() for l in found.splitlines() if l.strip().endswith((".yaml", ".yml"))), "")

    candidates: list[str] = []
    if recipe_path:
        beside = run_probe(host, rf"ls {shlex.quote(str(Path(recipe_path).parent))}/*sidecar*.json 2>/dev/null")
        candidates = [l.strip() for l in beside.splitlines() if l.strip().endswith(".json")]
    if not candidates:
        anywhere = run_probe(host, rf"find {roots} -maxdepth 6 -name '*sidecar*.json' 2>/dev/null | head -5")
        candidates = [l.strip() for l in anywhere.splitlines() if l.strip().endswith(".json")]

    ticket = stem.removesuffix("-smoke")
    sidecar_path = next((c for c in candidates if ticket and Path(c).name.startswith(ticket)), "")
    if not sidecar_path and candidates:
        sidecar_path = candidates[0]

    return {
        "available": available,
        "cli_path": cli.strip() if available else "",
        "version": version.strip(),
        "recipe_path": recipe_path,
        "sidecar_path": sidecar_path,
        "sidecar_candidates": candidates,
    }


def check_partitions(host: str) -> dict[str, Any]:
    """List schedulable partitions/queues with their time limits."""
    out = run_probe(host, (
        "sinfo --noheader -o '%P avail=%a timelimit=%l nodes=%D' 2>/dev/null | head -10 || "
        "primus queue list 2>/dev/null"
    ))
    return {"output": out[:400]}


def read_existing_launch_scripts(host: str, node: str) -> str:
    """Read existing launch scripts on the shared filesystem to learn local patterns."""
    roots = " ".join(search_roots()) or "~"
    out = run_probe(host, (
        rf"find {roots} -maxdepth 4 \( -name '*.sbatch' -o -name '*.slurm' -o -name '*.sh' \) "
        "2>/dev/null | head -5 | xargs head -30 2>/dev/null"
    ))
    return out[:1500]


def estimate_runtime(recipe: str, gpu_count: int) -> str:
    """Estimate job duration from known recipe characteristics."""
    estimates = {
        "daily-consan-clean": "1-2 minutes (single ConSan record/replay repro)",
        "daily-consan-racy": "1-2 minutes (single ConSan record/replay repro)",
        "daily-waitcheck-gemm": "2-5 minutes (static waitcheck scan of 3 GEMM kernels)",
        "Residual-NaN-Repro-smoke": "10-15 minutes (1 trial × 5 steps)",
        "Residual-NaN-Repro": "3-4 hours (16 trials × 1000 steps)",
        "Residual-NaN-Repro.yaml": "3-4 hours (16 trials × 1000 steps)",
        "Residual-NaN-Repro-smoke.yaml": "10-15 minutes (1 trial × 5 steps)",
    }
    for key, val in estimates.items():
        if key in recipe:
            return val
    return "unknown — check recipe step count"


# ---------------------------------------------------------------------------
# DSPy signature + module
# ---------------------------------------------------------------------------

class LaunchPlan(dspy.Signature):
    """
    You are planning how to launch a GPU training job on a specific cluster.
    Given the cluster profile, construct the exact command to run.

    Rules:
    - Emit only the bare workload command. Do NOT wrap it in sbatch, srun, ssh,
      nohup or '&' — the deploy agent submits it to the scheduler itself and
      redirects output. A command containing sbatch/srun/ssh is wrong.
    - Call check_recipe_mode on the recipe path before composing the command; the
      accepted flags differ per mode and the wrong flag aborts the run.
    - For mode 'triage' (matrix sweep) the command is
      'aorta sweep run --recipe <path> --mitigations-file <sidecar> --output <output_dir>'
    - For mode 'sanitizer' the command is exactly
      'aorta sweep run --recipe <path> --output <output_dir>'
      with NO --mitigations-file and no matrix axes: those flags are rejected.
      Sanitizer recipes define their own environments, so they need no sidecar.
    - Always use absolute paths for recipe, sidecar, and output.
    - log_path must be '<jobs_root>/<job_id>/watch.log' using the jobs_root input.
    - aorta_output must be '<jobs_root>/<job_id>/bundle/aorta' using the jobs_root input.
    - Use check_aorta_install to find exact recipe and sidecar paths before
      constructing the command; never invent a path that was not found. Pass the
      recipe name to it, and use the sidecar_path it returns — a sidecar from some
      other directory will not define this recipe's environments and the run dies
      with UnknownEnvironmentError. Ignore sidecar_path for sanitizer recipes.
    - If check_aorta_install reports available=false the aorta CLI is not
      installed. Do not emit an 'aorta ...' command in that case: leave command
      empty, set needs_confirmation=true, and ask how the workload should be
      launched. Finding recipe files on disk does not mean the CLI exists.
    - If a field cannot be determined, set it to empty string and add a question
      to questions[].
    """
    cluster_profile_json: str = dspy.InputField(desc="JSON ClusterProfile from discovery")
    recipe: str = dspy.InputField(desc="Recipe name e.g. Residual-NaN-Repro-smoke")
    job_id: str = dspy.InputField(desc="Job ID for path construction")
    jobs_root: str = dspy.InputField(desc="Rendezvous root that all nodes can read")
    user_hints: str = dspy.InputField(desc="Additional operator hints")

    command: str = dspy.OutputField(desc="Bare workload command, no scheduler or ssh wrapper")
    working_dir: str = dspy.OutputField(desc="Directory to run from on the target node")
    env_vars: dict = dspy.OutputField(desc="Environment variables dict to set before launch")
    log_path: str = dspy.OutputField(desc="Absolute path where job stdout+stderr will be written")
    aorta_output: str = dspy.OutputField(desc="Absolute path where Aorta writes matrix.json")
    estimated_runtime: str = dspy.OutputField(desc="Human-readable estimate e.g. '10-15 minutes'")
    needs_confirmation: bool = dspy.OutputField(desc="True if any required field is uncertain")
    questions: list[str] = dspy.OutputField(desc="Specific questions to ask the user if needs_confirmation")


class LaunchPlanner(dspy.Module):
    def __init__(self):
        ensure_configured()
        self.react = dspy.ReAct(
            LaunchPlan,
            tools=[check_recipe_mode, check_torchrun, check_aorta_install, check_partitions,
                   read_existing_launch_scripts, estimate_runtime],
            max_iters=6,
        )

    def forward(self, cluster_profile_json: str, recipe: str, job_id: str,
                jobs_root: str, user_hints: str = "") -> dspy.Prediction:
        return self.react(
            cluster_profile_json=cluster_profile_json,
            recipe=recipe,
            job_id=job_id,
            jobs_root=jobs_root,
            user_hints=user_hints,
        )
