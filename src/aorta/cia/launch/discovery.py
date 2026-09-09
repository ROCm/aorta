from __future__ import annotations

import re
import shlex
from typing import Any

import dspy

from aorta.cia.launch.cluster import quoted_search_roots, run_probe
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


#: Board names that name an architecture unambiguously. Only boards whose
#: mapping is settled belong here: a board absent from this table reads as
#: unknown, which the planner can act on, rather than as the nearest guess.
_BOARD_ARCH = (
    (re.compile(r"\bMI2[15]\d", re.IGNORECASE), "gfx90a"),   # MI210, MI250, MI250X
    (re.compile(r"\bMI3[02]\d", re.IGNORECASE), "gfx942"),   # MI300X/A, MI325X
    (re.compile(r"\bMI3[56]\d", re.IGNORECASE), "gfx950"),   # MI350X, MI355X
)


def parse_gpu_arch(probe_output: str) -> str:
    """The architecture *probe_output* states, or "" when it states none.

    This used to answer ``gfx90a`` for an MI25x and ``gfx942`` for everything
    else -- including an empty probe, an ERROR, a Navi card, and the MI355X
    this is developed against, which is ``gfx950``. Being wrong was half the
    problem; the other half was that the planner could not tell a reading from
    a default, so a guess arrived with the same authority as a fact and
    uncertain_fields had nothing to carry.

    An explicit gfx token wins, because a cluster that publishes one is stating
    the answer. A board name is read only through the table above. Anything
    else is "": on the cluster this was written against, sinfo reports gres
    ``(null)`` and features ``xgmi36,pod1``, which names no architecture at all.
    """
    explicit = re.search(r"\bgfx[0-9a-f]{3,}\b", probe_output, re.IGNORECASE)
    if explicit:
        return explicit.group(0).lower()

    for pattern, arch in _BOARD_ARCH:
        if pattern.search(probe_output):
            return arch

    return ""


def check_gpu_arch(host: str, node: str) -> dict[str, Any]:
    """Get GPU arch and count for node, preferring scheduler gres over rocm-smi.

    Many clusters refuse direct SSH to compute nodes, so Slurm's gres/features
    are the only reliable source; rocm-smi is a fallback for when this process
    is already running on the target node.
    """
    # node is a ReAct tool argument, so the model chooses it and it reaches a
    # shell: "n1; curl ..." would otherwise run. run_probe composes pipelines,
    # so it cannot take an argv list; quoting is what keeps this one argument.
    out = run_probe(host, f"sinfo -N -n {shlex.quote(node)} --noheader -o '%G %f' 2>/dev/null | head -3")
    if not out.strip() or "ERROR" in out:
        out = run_probe(host, "rocm-smi --showproductname 2>/dev/null | grep -E 'Card Series|Card Model'")

    count = 0
    gres = re.search(r"gpu:[^:\s]*:(\d+)", out)
    if gres:
        count = int(gres.group(1))
    elif "Card Series" in out:
        count = out.count("Card Series")

    return {"arch": parse_gpu_arch(out), "count": count, "output": out[:400]}


def read_cluster_configs(host: str) -> str:
    """Read existing job scripts and scheduler config to learn how jobs run here."""
    roots = quoted_search_roots()
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
    gpu_arch: str = dspy.OutputField(desc="ROCm GPU arch string e.g. gfx942, or empty if the probe did not report one -- then list gpu_arch in uncertain_fields")
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
