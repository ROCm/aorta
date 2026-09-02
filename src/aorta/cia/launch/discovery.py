from __future__ import annotations

import re
from typing import Any

import dspy

from aorta.cia.launch.cluster import run_probe, search_roots
from aorta.cia.llm import ensure_configured


# ---------------------------------------------------------------------------
# Discovery tools (called by the ReAct loop)
# ---------------------------------------------------------------------------

def check_slurm(host: str) -> dict[str, Any]:
    """Check if Slurm is available. Returns scheduler info and node states."""
    out = run_probe(host, (
        "sinfo --noheader -o '%n %T %G' 2>/dev/null | head -30; "
        "squeue --noheader 2>/dev/null | wc -l"
    ))
    has_slurm = "ERROR" not in out and out.strip() not in ("", "0")
    return {"available": has_slurm, "output": out[:800]}


def check_kubernetes(host: str) -> dict[str, Any]:
    """Check if Kubernetes is available."""
    out = run_probe(host, "kubectl get nodes --no-headers 2>/dev/null | head -10")
    available = "ERROR" not in out and "No resources" not in out and out.strip() != ""
    return {"available": available, "output": out[:500]}


def check_primus(host: str) -> dict[str, Any]:
    """Check if Primus CLI is available."""
    out = run_probe(host, "which primus 2>/dev/null && primus --version 2>/dev/null | head -3 || echo NOT_FOUND")
    available = "NOT_FOUND" not in out and "ERROR" not in out
    return {"available": available, "output": out[:300]}


def check_spur(host: str) -> dict[str, Any]:
    """Check if spur reservation system is available."""
    out = run_probe(host, "which spur 2>/dev/null && spur --help 2>/dev/null | head -3 || echo NOT_FOUND")
    available = "NOT_FOUND" not in out and "ERROR" not in out
    return {"available": available, "output": out[:300]}


def list_nodes(host: str) -> dict[str, Any]:
    """List compute nodes with state and GPU gres, plus partition time limits."""
    out = run_probe(host, (
        "sinfo --noheader -o '%n %T %G' 2>/dev/null | head -25; "
        "echo '--partitions(name avail timelimit nodes)--'; "
        "sinfo --noheader -o '%P %a %l %D' 2>/dev/null | head -10"
    ))
    return {"output": out[:1200]}


def check_gpu_arch(host: str, node: str) -> dict[str, Any]:
    """Get GPU arch and count for node, preferring scheduler gres over rocm-smi.

    Many clusters refuse direct SSH to compute nodes, so Slurm's gres/features
    are the only reliable source; rocm-smi is a fallback for when this process
    is already running on the target node.
    """
    out = run_probe(host, f"sinfo -N -n {node} --noheader -o '%G %f' 2>/dev/null | head -3")
    if not out.strip() or "ERROR" in out:
        out = run_probe(host, "rocm-smi --showproductname 2>/dev/null | grep -E 'Card Series|Card Model'")

    count = 0
    gres = re.search(r"gpu:[^:\s]*:(\d+)", out)
    if gres:
        count = int(gres.group(1))
    elif "Card Series" in out:
        count = out.count("Card Series")

    arch = "gfx90a" if re.search(r"MI25\d", out, re.IGNORECASE) else "gfx942"
    return {"arch": arch, "count": count, "output": out[:400]}


def read_cluster_configs(host: str) -> str:
    """Read existing job scripts and scheduler config to learn how jobs run here."""
    roots = " ".join(search_roots()) or "~"
    out = run_probe(host, (
        rf"find {roots} -maxdepth 4 \( -name '*.sbatch' -o -name '*.slurm' -o -name '*.sh' \) "
        "2>/dev/null | head -8 | xargs head -25 2>/dev/null; "
        "echo '--scheduler config--'; "
        "scontrol show config 2>/dev/null | grep -iE 'ClusterName|SchedulerType|MaxJobCount'"
    ))
    return out[:1500]


# ---------------------------------------------------------------------------
# DSPy signature + module
# ---------------------------------------------------------------------------

class ClusterProfile(dspy.Signature):
    """
    You are discovering the topology of a GPU compute cluster to decide how to
    launch a training job. Use the tools to probe what scheduler and launcher
    infrastructure is available. Read existing job scripts — they are the most
    reliable signal for how jobs are actually run here.

    Key rules:
    - If check_slurm reports Slurm is available, set scheduler='slurm' and
      launcher='sbatch'. The deploy agent submits the job with sbatch; compute
      nodes on Slurm clusters are usually not reachable by direct SSH.
    - Only choose launcher='aorta_direct' when there is no scheduler and the
      nodes are reachable over SSH, so aorta can manage Docker itself.
    - Pick target_node from list_nodes output, preferring a node in state 'idle'.
      Leave target_node empty to let the scheduler choose — that is better than
      guessing a name that may not exist.
    - Report confidence honestly. If uncertain about launcher, set it to 'unknown'
      and list it in uncertain_fields so the user can be asked.
    """
    head_node: str = dspy.InputField(desc="Cluster head/login node, or empty when probing locally")
    user_hints: str = dspy.InputField(desc="Operator hints from config (may be empty)")
    job_requirements: str = dspy.InputField(desc="What needs to run: recipe, GPU count, framework")

    scheduler: str = dspy.OutputField(desc="slurm | kubernetes | bare_metal | spur | unknown")
    launcher: str = dspy.OutputField(desc="sbatch | aorta_direct | torchrun | primus | kubectl | unknown")
    target_node: str = dspy.OutputField(desc="Best node to run the job on, or empty to let the scheduler pick")
    gpu_arch: str = dspy.OutputField(desc="ROCm GPU arch string e.g. gfx942")
    gpu_count: int = dspy.OutputField(desc="Number of GPUs on target node")
    confidence: float = dspy.OutputField(desc="0.0-1.0 overall confidence in the plan")
    uncertain_fields: list[str] = dspy.OutputField(desc="Fields the user should confirm if confidence < 0.8")
    reasoning: str = dspy.OutputField(desc="What evidence led to these conclusions")


class ClusterDiscovery(dspy.Module):
    def __init__(self):
        ensure_configured()
        self.react = dspy.ReAct(
            ClusterProfile,
            tools=[check_slurm, check_kubernetes, check_primus, check_spur,
                   list_nodes, check_gpu_arch, read_cluster_configs],
            max_iters=8,
        )

    def forward(self, head_node: str, user_hints: str = "", job_requirements: str = "") -> dspy.Prediction:
        return self.react(
            head_node=head_node,
            user_hints=user_hints,
            job_requirements=job_requirements,
        )
