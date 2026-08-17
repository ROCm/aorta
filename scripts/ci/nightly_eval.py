#!/usr/bin/env python3
"""Nightly evaluation harness.

Runs each entry in config/ci/nightly_eval_matrix.yaml against the installed
aorta (the released nightly wheel in CI), harvests matrix.json, compares each
cell to config/ci/regression_baselines.yaml (record-only when no baseline
exists), and writes a results JSON.

Exit is non-zero if any cell FAILs (a failed/errored cell, a missing matrix.json,
a per-entry timeout, a blessed-baseline breach), OR if zero entries actually ran
(e.g. no GPUs) -- a green run that did no work is itself a failure.

Usage:
    python scripts/ci/nightly_eval.py --out results.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_lib  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX = REPO_ROOT / "config" / "ci" / "nightly_eval_matrix.yaml"
BASELINES = REPO_ROOT / "config" / "ci" / "regression_baselines.yaml"

_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from aorta.instrumentation.rocm_paths import resolve_rocm_roots  # noqa: E402

_ROCM_ROOTS = resolve_rocm_roots()


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must be a YAML mapping, got {type(data).__name__}")
    return data


def gpu_count() -> int:
    # Only a genuine missing-torch (ImportError) means "0 GPUs". A present-but-
    # broken torch (bad libs, HIP/CUDA init error) must surface, not be swallowed
    # into 0 -- otherwise every entry would skip, the nightly would go green while
    # running nothing, and alerting would close open regression issues.
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def build_metadata() -> dict[str, Any]:
    import os

    meta: dict[str, Any] = {"amd_aorta_version": None, "torch": None, "hip": None, "rocm": None}
    try:
        import importlib.metadata as im

        meta["amd_aorta_version"] = im.version("amd-aorta")
    except Exception:
        pass
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["hip"] = getattr(torch.version, "hip", None)
    except Exception:
        pass
    # Resolved rather than hardcoded to /opt/rocm (issue #381): a TheRock
    # wheel install keeps .info/version under site-packages, and reading the
    # literal path there left the dashboard's `rocm` column silently null
    # while `torch` and `hip` still populated -- the worst kind of gap,
    # because the row still looked complete.
    for candidate in (_ROCM_ROOTS.version_file, _ROCM_ROOTS.version_dev_file):
        if candidate.exists():
            meta["rocm"] = candidate.read_text(encoding="utf-8").strip()
            break
    # Provenance of the exact triggering wheel/commit (set by the workflow) so a
    # dashboard result is attributable to a specific source, not "whatever was
    # latest". See nightly-eval.yml.
    meta["upstream_run_id"] = os.environ.get("UPSTREAM_RUN_ID")
    meta["head_sha"] = os.environ.get("UPSTREAM_HEAD_SHA")
    meta["wheel_file"] = os.environ.get("WHEEL_FILE")
    return meta


def _aorta() -> str:
    path = shutil.which("aorta")
    if not path:
        raise SystemExit("aorta CLI not found on PATH (is the wheel installed?)")
    return path


def run_entry(entry: dict[str, Any], out_dir: Path) -> tuple[int, Path | None, bool]:
    """Run one recipe; return (exit_code, matrix_json_path_or_None, timed_out)."""
    recipe = entry["recipe"]
    nproc = entry.get("nproc")
    timeout_sec = int(entry.get("timeout_sec", 1800))
    aorta = _aorta()
    run_out = out_dir / entry["name"]
    # Start from a clean per-entry dir so a stale matrix.json from a prior run
    # can't be picked up if this invocation fails before producing a fresh one.
    if run_out.exists():
        shutil.rmtree(run_out)
    run_out.mkdir(parents=True, exist_ok=True)

    argv: list[str] = []
    if nproc:
        argv += ["torchrun", "--standalone", f"--nproc_per_node={int(nproc)}"]
    argv += [aorta, "sweep", "run", "--recipe", recipe, "--output-dir", str(run_out), "--strict"]

    print(f"RUN {entry['name']} (timeout {timeout_sec}s): {' '.join(argv)}", flush=True)
    timed_out = False
    # Run in its own process group/session so a timeout can kill the ENTIRE tree
    # (torchrun + all rank workers), not just the immediate torchrun process --
    # orphaned GPU workers would otherwise retain memory/collectives and
    # contaminate later matrix entries.
    proc = subprocess.Popen(argv, cwd=REPO_ROOT, start_new_session=True)
    try:
        rc = proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT {entry['name']} after {timeout_sec}s; killing process group", flush=True)
        timed_out = True
        rc = 124
        _kill_process_group(proc)

    matrices = sorted(run_out.rglob("matrix.json"), key=lambda p: p.stat().st_mtime)
    return rc, (matrices[-1] if matrices else None), timed_out


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Terminate the whole process group with bounded TERM then KILL escalation."""
    import os
    import signal

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            continue


def evaluate(matrix_doc: dict[str, Any], baselines: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    ngpu = gpu_count()
    print(f"nightly eval: {ngpu} GPU(s) visible", flush=True)
    baseline_map = baselines.get("baselines") or {}
    results: list[dict[str, Any]] = []

    for entry in matrix_doc.get("entries") or []:
        name = entry["name"]
        min_gpus = int(entry.get("min_gpus", entry.get("nproc", 1)))
        if ngpu < min_gpus:
            results.append(
                {"entry": name, "recipe": entry["recipe"], "cell": None,
                 "verdict": "skip", "reasons": [f"needs {min_gpus} GPU(s), have {ngpu}"],
                 "metrics": {}, "deltas": {}, "duration_sec": 0.0, "error": None}
            )
            continue

        start = _dt.datetime.now()
        rc, matrix_path, timed_out = run_entry(entry, out_dir)
        dur = (_dt.datetime.now() - start).total_seconds()

        # A timeout is authoritative: even if rank 0 wrote a matrix.json before
        # another worker hung, the entry is a failure -- never harvest it green.
        if timed_out:
            results.append(
                {"entry": name, "recipe": entry["recipe"], "cell": None,
                 "verdict": "fail",
                 "reasons": [f"timed out ({entry.get('timeout_sec', 1800)}s)"],
                 "metrics": {}, "deltas": {}, "duration_sec": dur,
                 "error": "timeout", "matrix_path": None}
            )
            continue

        if matrix_path is None:
            results.append(
                {"entry": name, "recipe": entry["recipe"], "cell": None,
                 "verdict": "fail", "reasons": [f"no matrix.json produced (exit {rc})"],
                 "metrics": {}, "deltas": {}, "duration_sec": dur,
                 "error": f"exit {rc}", "matrix_path": None}
            )
            continue

        # A truncated/corrupt matrix.json (e.g. a crash mid-write) must not abort
        # the whole run before results JSON is produced -- otherwise the job goes
        # red but alerting/dashboard silently no-op (no artifact). Record a
        # per-entry fail so the regression is still surfaced.
        try:
            harvested_cells = eval_lib.harvest_matrix_json(matrix_path)
        except (ValueError, OSError) as exc:  # JSONDecodeError is a ValueError
            results.append(
                {"entry": name, "recipe": entry["recipe"], "cell": None,
                 "verdict": "fail", "reasons": [f"unreadable matrix.json: {exc}"],
                 "metrics": {}, "deltas": {}, "duration_sec": dur,
                 "error": "corrupt_matrix",
                 "matrix_path": str(matrix_path.relative_to(REPO_ROOT))
                 if matrix_path.is_relative_to(REPO_ROOT) else str(matrix_path)}
            )
            continue

        if not harvested_cells:
            # matrix.json with zero cells: the entry produced no observation.
            # Fail it explicitly so a broken entry can't vanish from the report.
            results.append(
                {"entry": name, "recipe": entry["recipe"], "cell": None,
                 "verdict": "fail", "reasons": ["matrix.json produced no cells"],
                 "metrics": {}, "deltas": {}, "duration_sec": dur,
                 "error": "empty_matrix",
                 "matrix_path": str(matrix_path.relative_to(REPO_ROOT))
                 if matrix_path.is_relative_to(REPO_ROOT) else str(matrix_path)}
            )
            continue

        for harvested in harvested_cells:
            key = eval_lib.cell_key(name, harvested["cell"])
            cmp = eval_lib.compare_to_baseline(harvested, baseline_map.get(key))
            results.append(
                {"entry": name, "recipe": entry["recipe"], "cell": harvested["cell"],
                 "verdict": cmp["verdict"], "reasons": cmp["reasons"],
                 "metrics": harvested["metrics"], "deltas": cmp["deltas"],
                 # Retain the raw counts + artifact path for debugging.
                 "passed_count": harvested.get("passed_count"),
                 "failed_count": harvested.get("failed_count"),
                 "error_count": harvested.get("error_count"),
                 "trials": harvested.get("trials"),
                 "matrix_path": str(matrix_path.relative_to(REPO_ROOT))
                 if matrix_path.is_relative_to(REPO_ROOT) else str(matrix_path),
                 "duration_sec": dur, "error": harvested["error"]}
            )

    # BLOCKER fix: a run where nothing actually executed (no entries, or every
    # entry skipped -- e.g. zero GPUs / broken passthrough) must FAIL, not go
    # green having done no work.
    if not results or all(item["verdict"] == "skip" for item in results):
        results.append(
            {"entry": "_nightly_eval", "recipe": None, "cell": None, "verdict": "fail",
             "reasons": ["no matrix entry ran (empty matrix or all skipped -- check GPUs)"],
             "metrics": {}, "deltas": {}, "duration_sec": 0.0, "error": "zero_work",
             "matrix_path": None}
        )

    return {
        "schema_version": 1,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "build": build_metadata(),
        "summary": eval_lib.summarize(results),
        "entries": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "nightly-results.json")
    ap.add_argument("--work-dir", type=Path, default=REPO_ROOT / ".nightly-eval")
    args = ap.parse_args()

    matrix_doc = _load_yaml(MATRIX)
    baselines = _load_yaml(BASELINES) if BASELINES.exists() else {"baselines": {}}

    args.work_dir.mkdir(parents=True, exist_ok=True)
    doc = evaluate(matrix_doc, baselines, args.work_dir)
    args.out.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    s = doc["summary"]
    print(
        f"nightly eval summary: total={s['total']} pass={s['pass']} "
        f"fail={s['fail']} record={s['record']} skip={s['skip']}",
        flush=True,
    )
    # Fail closed: any `fail` verdict fails the job. That includes a failed or
    # errored cell (even without a baseline), a missing/empty matrix.json, a
    # per-entry timeout, a blessed-baseline breach, and a run that did zero work.
    return 1 if s["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
