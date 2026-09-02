from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import dspy

from aorta.cia.llm import ensure_configured

# Extensions considered log files by default
DEFAULT_LOG_EXTENSIONS = {".log", ".txt", ".out", ".err"}
DEFAULT_EXCLUDE_PATTERNS = {".ckpt", ".bin", ".pt", ".safetensors", ".pkl", ".npz", ".npy"}
DEFAULT_MAX_FILES = 8
# Directories that never hold training logs but do hold thousands of .txt files.
SKIP_DIR_PARTS = {".git", ".venv", "site-packages", "node_modules", "__pycache__"}


# ---------------------------------------------------------------------------
# Scheduler-native log path discovery (highest priority, most reliable)
# ---------------------------------------------------------------------------

def query_scheduler_logs(scheduler: str, scheduler_job_id: str, head_node: str = "") -> list[str]:
    """Ask the scheduler for the exact stdout/stderr paths of a running job.

    Returns a list of absolute file paths. Empty list if not available.
    This is the most reliable source — no guessing needed.

    Slurm:  scontrol show job <id>  →  StdOut=, StdErr= fields
    K8s:    kubectl get pod <name> -o json  →  volumeMounts + log paths
    """
    paths: list[str] = []
    if not scheduler_job_id:
        return paths

    def _run(cmd: str) -> str:
        try:
            if head_node:
                user = os.environ.get("CIA_SSH_USER") or os.environ.get("USER") or "root"
                r = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                     f"{user}@{head_node}", cmd],
                    capture_output=True, text=True, timeout=15,
                )
            else:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
            return r.stdout + r.stderr
        except Exception:
            return ""

    if scheduler == "slurm":
        out = _run(f"scontrol show job {scheduler_job_id}")
        # Parse StdOut=... StdErr=... (may be on same line or separate)
        for token in out.split():
            for key in ("StdOut=", "StdErr="):
                if token.startswith(key):
                    p = token[len(key):].strip()
                    if p and p != "/dev/null" and p not in paths:
                        paths.append(p)
        # Also grab WorkDir and Command for context
        for token in out.split():
            if token.startswith("WorkDir="):
                wd = token[len("WorkDir="):].strip()
                if wd:
                    # Look for log files in the workdir too
                    paths.append(f"__workdir__:{wd}")

    elif scheduler == "kubernetes":
        # Get pod spec to find volume mounts and log paths
        out = _run(f"kubectl get pod {scheduler_job_id} -o json")
        try:
            pod = json.loads(out)
            containers = pod.get("spec", {}).get("containers", [])
            for c in containers:
                for vm in c.get("volumeMounts", []):
                    mount = vm.get("mountPath", "")
                    # Common training log mount paths
                    if any(k in mount.lower() for k in ("log", "output", "scratch", "work")):
                        paths.append(f"__dir__:{mount}")
            # Also try kubectl logs path pattern
            paths.append(f"__kubectl_logs__:{scheduler_job_id}")
        except Exception:
            pass

    return paths


def _dir_listing(job_dir: Path) -> str:
    """Return a find-style listing of job_dir with sizes and mtimes."""
    try:
        r = subprocess.run(
            ["find", str(job_dir), "-maxdepth", "4", "-type", "f",
             "-printf", "%T@ %s %p\n"],
            capture_output=True, text=True, timeout=10,
        )
        lines = []
        for line in r.stdout.strip().splitlines():
            parts = line.split(" ", 2)
            if len(parts) == 3:
                mtime, size, path = parts
                lines.append(f"{path}  size={size}  mtime={mtime}")
        return "\n".join(lines[:60])
    except Exception as e:
        return f"listing error: {e}"


# ---------------------------------------------------------------------------
# DSPy signature for auto-discovery
# ---------------------------------------------------------------------------

class LogDiscovery(dspy.Signature):
    """
    You are discovering which files in a job directory contain training logs
    worth monitoring for a GPU cluster job. Given a file listing, identify the
    files most likely to contain training progress output such as loss values,
    throughput, step counts, GPU errors, or stack traces.

    Rules:
    - Prefer recently modified files (high mtime).
    - Prefer files with 'log', 'stderr', 'stdout', 'out', 'err' in their name.
    - Skip checkpoints (.ckpt, .bin, .pt, .safetensors), weights, and binaries.
    - Skip files smaller than 100 bytes — likely empty placeholders.
    - Return at most 8 files, ranked by relevance (most relevant first).
    """
    job_dir_listing: str = dspy.InputField(desc="File listing with size and mtime")
    job_context: str = dspy.InputField(desc="Recipe name, framework, node, what the job does")

    relevant_files: list[str] = dspy.OutputField(desc="Ranked list of absolute file paths to monitor")
    reasoning: str = dspy.OutputField(desc="Brief explanation of why each file was selected")


class LogFinder(dspy.Module):
    """Discover which log files to monitor for a job.

    Config hints (all optional):
      paths      — explicit glob patterns (highest priority)
      extensions — file extensions to include (fallback if paths absent)
      exclude    — patterns to never watch
      max_files  — cap on number of files returned
    """

    def __init__(self, config: dict[str, Any] | None = None):
        ensure_configured()
        cfg = config or {}
        self.paths: list[str] = cfg.get("paths", [])
        self.extensions: set[str] = set(cfg.get("extensions", DEFAULT_LOG_EXTENSIONS))
        self.exclude: set[str] = set(cfg.get("exclude", DEFAULT_EXCLUDE_PATTERNS))
        self.max_files: int = int(cfg.get("max_files", DEFAULT_MAX_FILES))
        self._discovery = dspy.Predict(LogDiscovery)

    def find(
        self,
        job_dir: Path,
        job_context: str = "",
        scheduler: str = "",
        scheduler_job_id: str = "",
        head_node: str = "",
    ) -> list[Path]:
        """Return ordered list of paths to monitor. Caches nothing — caller caches.

        Priority:
          0. Scheduler-native query (scontrol / kubectl) — most reliable, no guessing
          1. Explicit config path hints
          2. Extension-based scan
          3. LLM auto-discovery from dir listing
        """
        job_dir = Path(job_dir)

        # 0. Scheduler-native: ask Slurm/K8s for StdOut/StdErr paths directly
        if scheduler and scheduler_job_id:
            sched_paths = query_scheduler_logs(scheduler, scheduler_job_id, head_node)
            resolved: list[Path] = []
            extra_dirs: list[Path] = []
            for p_str in sched_paths:
                if p_str.startswith("__workdir__:") or p_str.startswith("__dir__:"):
                    extra_dirs.append(Path(p_str.split(":", 1)[1]))
                elif p_str.startswith("__kubectl_logs__:"):
                    pass  # handled separately if needed
                else:
                    p = Path(p_str)
                    if p.is_file() and not self._excluded(p):
                        resolved.append(p)
            # StdOut/StdErr from the scheduler are authoritative. Only trawl the
            # work dir when it named no usable file, or a WorkDir that happens to
            # be a source checkout drags in unrelated .txt files.
            if not resolved:
                for d in extra_dirs:
                    resolved.extend(self._scan_by_extension(d))
            if resolved:
                return resolved[: self.max_files]

        # 1. Explicit path hints — expand globs
        if self.paths:
            found: list[Path] = []
            for pattern in self.paths:
                pattern = pattern.replace("{job_id}", job_dir.name)
                for p in sorted(Path("/").glob(pattern.lstrip("/"))):
                    if p.is_file() and not self._excluded(p):
                        found.append(p)
            if found:
                return found[: self.max_files]

        # 2. Extension-based scan — fast, no LLM
        by_ext = self._scan_by_extension(job_dir)
        if by_ext:
            # If we found files with known extensions, return them directly
            # without burning an LLM call on the obvious case
            if len(by_ext) <= 3:
                return by_ext[: self.max_files]

        # 3. LLM auto-discovery — for ambiguous or large directories
        listing = _dir_listing(job_dir)
        if not listing.strip():
            return by_ext[: self.max_files]

        try:
            pred = self._discovery(
                job_dir_listing=listing,
                job_context=job_context or f"job_dir={job_dir}",
            )
            result = []
            for p_str in (pred.relevant_files or []):
                p = Path(p_str.strip())
                if p.is_file() and not self._excluded(p):
                    result.append(p)
            if result:
                return result[: self.max_files]
        except Exception:
            pass

        return by_ext[: self.max_files]

    def _scan_by_extension(self, job_dir: Path) -> list[Path]:
        candidates: list[tuple[float, Path]] = []
        try:
            for p in job_dir.rglob("*"):
                if not p.is_file():
                    continue
                if SKIP_DIR_PARTS & set(p.parts):
                    continue
                if self._excluded(p):
                    continue
                if p.suffix.lower() not in self.extensions:
                    continue
                if p.stat().st_size < 100:
                    continue
                candidates.append((p.stat().st_mtime, p))
        except Exception:
            pass
        # Sort newest-first
        candidates.sort(reverse=True)
        return [p for _, p in candidates]

    def _excluded(self, p: Path) -> bool:
        name = p.name.lower()
        return any(name.endswith(ext) for ext in self.exclude)
