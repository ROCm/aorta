"""Adapter for importing Magpie benchmark results into aorta's report pipeline.

Converts Magpie benchmark workspace layout into the format expected by
aorta-report commands (analyze, compare, generate).

A Magpie benchmark workspace has the structure::

    benchmark_{framework}_{timestamp}/
        config.yaml
        benchmark_report.json
        inferencemax_result.json
        torch_trace/
        tracelens_rank0_csvs/
        tracelens_collective_csvs/

This adapter:
  1. Reads Magpie's ``benchmark_report.json``
  2. Locates TraceLens CSV/Excel output (if present)
  3. Produces a normalised summary that ``aorta-report compare`` can consume
  4. Can run TraceLens analysis on the torch traces (using aorta's GEMM-patched wrapper)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def locate_magpie_workspaces(results_dir: str | Path) -> List[Path]:
    """Find Magpie benchmark workspace directories under *results_dir*.

    A workspace is identified by having a ``benchmark_report.json`` file.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return []

    workspaces = []
    for child in sorted(results_path.iterdir(), reverse=True):
        if child.is_dir() and (child / "benchmark_report.json").exists():
            workspaces.append(child)
    return workspaces


def read_magpie_report(workspace: str | Path) -> Dict[str, Any]:
    """Read and normalise a Magpie benchmark_report.json.

    Returns a dict with keys:
      - source: "magpie"
      - workspace, framework, model, success
      - throughput: {request_throughput, output_throughput, ...}
      - latency: {ttft: {mean_ms, p99_ms, ...}, tpot: ..., itl: ..., e2el: ...}
      - kernel_summary: [{name, time_ms, percent, calls}, ...]
      - top_bottlenecks: [str, ...]
      - has_torch_trace, has_tracelens
    """
    ws = Path(workspace)
    report_file = ws / "benchmark_report.json"
    if not report_file.exists():
        return {"error": f"benchmark_report.json not found in {workspace}"}

    with open(report_file) as f:
        report = json.load(f)

    normalised: Dict[str, Any] = {
        "source": "magpie",
        "workspace": str(ws),
        "framework": report.get("framework", ""),
        "model": report.get("model", ""),
        "success": report.get("success", False),
        "execution_time": report.get("execution_time", 0.0),
        "throughput": report.get("throughput"),
        "latency": report.get("latency"),
        "kernel_summary": report.get("kernel_summary", []),
        "top_bottlenecks": report.get("top_bottlenecks", []),
        "errors": report.get("errors", []),
        "has_torch_trace": (ws / "torch_trace").is_dir()
        and any((ws / "torch_trace").iterdir()),
        "has_tracelens": (ws / "tracelens_rank0_csvs").is_dir()
        or (ws / "tracelens_collective_csvs").is_dir(),
    }

    config_file = ws / "config.yaml"
    if config_file.exists():
        try:
            import yaml

            with open(config_file) as f:
                normalised["config"] = yaml.safe_load(f)
        except Exception:
            pass

    return normalised


def import_magpie_workspace(
    workspace: str | Path,
    output_dir: str | Path,
    run_tracelens: bool = False,
    num_ranks: int = 8,
) -> Dict[str, Any]:
    """Import a Magpie workspace into an aorta-compatible layout.

    Copies or symlinks the TraceLens output and torch traces so that
    ``aorta-report`` commands can operate on them directly.

    If *run_tracelens* is True and TraceLens output is missing, runs
    aorta's GEMM-patched TraceLens analysis on the torch traces.

    Args:
        workspace: Path to Magpie benchmark workspace.
        output_dir: Destination directory for aorta-format output.
        run_tracelens: Whether to run TraceLens analysis if not already present.
        num_ranks: Number of ranks for multi-rank collective analysis.

    Returns:
        Dict with imported paths and status.
    """
    ws = Path(workspace)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "source_workspace": str(ws),
        "output_dir": str(out),
        "imported_files": [],
    }

    # Copy benchmark report
    report_src = ws / "benchmark_report.json"
    if report_src.exists():
        report_dst = out / "benchmark_report.json"
        shutil.copy2(report_src, report_dst)
        result["imported_files"].append(str(report_dst))

    # Copy config
    config_src = ws / "config.yaml"
    if config_src.exists():
        config_dst = out / "config.yaml"
        shutil.copy2(config_src, config_dst)
        result["imported_files"].append(str(config_dst))

    # Link or copy TraceLens output
    for tl_dir_name in ("tracelens_rank0_csvs", "tracelens_collective_csvs"):
        tl_src = ws / tl_dir_name
        if tl_src.is_dir():
            tl_dst = out / tl_dir_name
            if not tl_dst.exists():
                shutil.copytree(tl_src, tl_dst)
            result["imported_files"].append(str(tl_dst))

    # Link torch traces
    trace_src = ws / "torch_trace"
    if trace_src.is_dir():
        trace_dst = out / "torch_profiler" / "rank0"
        trace_dst.mkdir(parents=True, exist_ok=True)
        for trace_file in trace_src.iterdir():
            dst_file = trace_dst / trace_file.name
            if not dst_file.exists():
                shutil.copy2(trace_file, dst_file)
        result["imported_files"].append(str(trace_dst))

    # Optionally run TraceLens analysis on torch traces
    has_tracelens = (out / "tracelens_rank0_csvs").is_dir()
    if run_tracelens and not has_tracelens:
        trace_dir = out / "torch_profiler" / "rank0"
        if trace_dir.is_dir() and any(trace_dir.iterdir()):
            try:
                from aorta.report.analysis.tracelens_wrapper import TraceLensWrapper

                wrapper = TraceLensWrapper()
                traces = sorted(
                    list(trace_dir.glob("*.json.gz")) + list(trace_dir.glob("*.json"))
                )
                if traces:
                    tl_output = out / "tracelens_analysis"
                    tl_output.mkdir(parents=True, exist_ok=True)
                    perf_xlsx = tl_output / "perf_rank0.xlsx"
                    wrapper.generate_perf_report(
                        trace_path=traces[0], output_path=perf_xlsx
                    )
                    result["imported_files"].append(str(perf_xlsx))
                    result["tracelens_ran"] = True
            except ImportError:
                result["tracelens_ran"] = False
                result["tracelens_error"] = "TraceLens not installed"
            except Exception as e:
                result["tracelens_ran"] = False
                result["tracelens_error"] = str(e)

    return result


def compare_magpie_reports(
    baseline_workspace: str | Path,
    test_workspace: str | Path,
) -> Dict[str, Any]:
    """Quick comparison of two Magpie benchmark reports.

    Computes throughput and latency deltas without requiring TraceLens
    or the full aorta-report pipeline.

    Returns:
        Dict with throughput/latency percent changes and a status summary.
    """
    baseline = read_magpie_report(baseline_workspace)
    test = read_magpie_report(test_workspace)

    if "error" in baseline:
        return {"error": f"Baseline: {baseline['error']}"}
    if "error" in test:
        return {"error": f"Test: {test['error']}"}

    comparison: Dict[str, Any] = {
        "baseline": {
            "workspace": baseline["workspace"],
            "framework": baseline["framework"],
            "model": baseline["model"],
        },
        "test": {
            "workspace": test["workspace"],
            "framework": test["framework"],
            "model": test["model"],
        },
        "throughput": {},
        "latency": {},
    }

    # Throughput comparison (higher is better)
    b_tp = baseline.get("throughput") or {}
    t_tp = test.get("throughput") or {}
    for key in ("request_throughput", "output_throughput", "total_token_throughput"):
        bv = b_tp.get(key, 0)
        tv = t_tp.get(key, 0)
        pct = ((tv - bv) / bv * 100) if bv else 0.0
        comparison["throughput"][key] = {
            "baseline": bv,
            "test": tv,
            "percent_change": round(pct, 2),
            "status": "better" if pct > 1 else ("worse" if pct < -1 else "similar"),
        }

    # Latency comparison (lower is better)
    b_lat = baseline.get("latency") or {}
    t_lat = test.get("latency") or {}
    for metric in ("ttft", "tpot", "itl", "e2el"):
        b_sub = b_lat.get(metric, {})
        t_sub = t_lat.get(metric, {})
        for stat in ("mean_ms", "p99_ms"):
            bv = b_sub.get(stat, 0)
            tv = t_sub.get(stat, 0)
            # Positive = test is faster (lower latency)
            pct = ((bv - tv) / bv * 100) if bv else 0.0
            comparison["latency"][f"{metric}_{stat}"] = {
                "baseline": bv,
                "test": tv,
                "percent_change": round(pct, 2),
                "status": "better" if pct > 1 else ("worse" if pct < -1 else "similar"),
            }

    # Summary
    tp_results = [v["status"] for v in comparison["throughput"].values()]
    lat_results = [v["status"] for v in comparison["latency"].values()]
    has_regressions = "worse" in tp_results or "worse" in lat_results
    has_improvements = "better" in tp_results or "better" in lat_results

    comparison["summary"] = {
        "has_regressions": has_regressions,
        "has_improvements": has_improvements,
        "overall": "regression" if has_regressions else (
            "improvement" if has_improvements else "neutral"
        ),
    }

    return comparison
