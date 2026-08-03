#!/usr/bin/env python3
"""Select the "top kernels" a workload actually launches, for sanitizing.

This is the front of the ``rocjitsu_sanitizers`` pipeline: before we can
run waitcheck (static) or consan (dynamic) on anything, we need a ranked
worklist of *which* kernels matter. Two complementary sources feed it:

  1. **Magpie kernel summary** -- a Magpie benchmark workspace's
     ``benchmark_report.json`` carries ``kernel_summary`` (one entry per
     kernel: ``name``, ``time_ms``, ``percent``, ``calls``) and
     ``top_bottlenecks`` (surfaced via ``aorta.report.magpie_adapter``).
     Ranked by wall time, it answers "which kernels dominate this model's
     runtime".

  2. **hipBLASLt GEMM dispatch CSV** -- ``gemm_shapes_unique.csv``
     (columns ``rank,count,...,transA,transB,M,N,K,...``) exported from a
     GEMM dispatch trace. Ranked by dispatch count, it answers "which GEMM
     shapes are launched the most".

Any producer that emits the ``rocjitsu_sanitizers.kernels/1`` worklist schema
below can feed the sanitizers directly (see ``runner.py --kernels``); the two
loaders here are conveniences, not the only entry points.

The output is a JSON worklist consumed by ``runner.py``. Kernel selection is
host-only, needs no GPU, and is deterministic, so it is the fully
unit-testable core of the tool.

Usage::

    python -m aorta.tools.rocjitsu_sanitizers.select \
        --magpie-report benchmark_report.json \
        --gemm-csv gemm_shapes_unique.csv \
        --top-n 20 \
        --output kernels.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _load_magpie_kernels(report_path: Path, top_n: int) -> list[dict[str, Any]]:
    """Read ``kernel_summary`` from a Magpie ``benchmark_report.json``.

    Accepts either the raw Magpie report or the normalised dict emitted by
    ``aorta.report.magpie_adapter.read_magpie_report`` -- both expose the
    same ``kernel_summary`` list. Ranks by ``percent`` (falling back to
    ``time_ms``) so the hottest kernels come first.
    """
    with open(report_path) as fh:
        report = json.load(fh)
    summary = report.get("kernel_summary") or []
    bottlenecks = set(report.get("top_bottlenecks") or [])

    def sort_key(entry: dict[str, Any]) -> tuple[float, float]:
        return (float(entry.get("percent") or 0.0), float(entry.get("time_ms") or 0.0))

    ranked = sorted(summary, key=sort_key, reverse=True)[:top_n]
    kernels: list[dict[str, Any]] = []
    for rank, entry in enumerate(ranked, start=1):
        name = entry.get("name", "")
        kernels.append(
            {
                "source": "magpie",
                "rank": rank,
                "name": name,
                "percent": entry.get("percent"),
                "time_ms": entry.get("time_ms"),
                "calls": entry.get("calls"),
                "is_bottleneck": name in bottlenecks,
            }
        )
    return kernels


def _load_gemm_kernels(csv_path: Path, top_n: int) -> list[dict[str, Any]]:
    """Read the top GEMM shapes from a ``gemm_shapes_unique.csv``.

    The CSV is pre-sorted by dispatch ``count`` (rank 1 = most-launched). We
    read the shape tuple and carry the Tensile ``top_solution_idx`` so a later
    stage can map a shape back to the concrete ``.co`` variant it selects.
    """
    kernels: list[dict[str, Any]] = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if len(kernels) >= top_n:
                break
            m, n, k = row.get("M"), row.get("N"), row.get("K")
            trans = f"{row.get('transA', '?')}{row.get('transB', '?')}"
            kernels.append(
                {
                    "source": "gemm_csv",
                    "rank": int(row.get("rank", len(kernels) + 1)),
                    "name": f"gemm_{trans}_M{m}_N{n}_K{k}",
                    "shape": {
                        "transA": row.get("transA"),
                        "transB": row.get("transB"),
                        "M": _int_or_none(m),
                        "N": _int_or_none(n),
                        "K": _int_or_none(k),
                        "batch_count": _int_or_none(row.get("batch_count")),
                    },
                    "count": _int_or_none(row.get("count")),
                    "compute_type": row.get("compute_type"),
                    "top_solution_idx": _int_or_none(row.get("top_solution_idx")),
                }
            )
    return kernels


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def select_kernels(
    *,
    magpie_report: Path | None,
    gemm_csv: Path | None,
    top_n: int,
) -> dict[str, Any]:
    """Build the ranked kernel worklist from whichever sources are given."""
    if magpie_report is None and gemm_csv is None:
        raise ValueError("provide at least one of --magpie-report / --gemm-csv")

    kernels: list[dict[str, Any]] = []
    sources: list[str] = []
    if magpie_report is not None:
        kernels += _load_magpie_kernels(magpie_report, top_n)
        sources.append(f"magpie:{magpie_report.name}")
    if gemm_csv is not None:
        kernels += _load_gemm_kernels(gemm_csv, top_n)
        sources.append(f"gemm_csv:{gemm_csv.name}")

    return {
        "schema": "rocjitsu_sanitizers.kernels/1",
        "top_n": top_n,
        "sources": sources,
        "kernel_count": len(kernels),
        "kernels": kernels,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magpie-report", type=Path, default=None,
                        help="Magpie benchmark_report.json with kernel_summary")
    parser.add_argument("--gemm-csv", type=Path, default=None,
                        help="gemm_shapes_unique.csv (ranked by dispatch count)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="keep the top N kernels from each source")
    parser.add_argument("--output", type=Path, default=None,
                        help="write worklist JSON here (default: stdout)")
    args = parser.parse_args(argv)

    worklist = select_kernels(
        magpie_report=args.magpie_report,
        gemm_csv=args.gemm_csv,
        top_n=args.top_n,
    )
    text = json.dumps(worklist, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
        print(f"wrote {worklist['kernel_count']} kernels -> {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
