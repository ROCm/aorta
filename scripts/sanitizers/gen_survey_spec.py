#!/usr/bin/env python3
"""Emit the workload-survey ``--survey`` spec for the sanitizer dashboard (Tab 2).

The survey tab (``gen_sanitizer_dashboard.py --survey``) renders observed-only
kernels drawn from multiple workloads with *no* baseline/expected comparison: a
``warn``/``error``/``not_checked`` here is an observation, never a regression.

This generator is a thin, deterministic adapter: it enumerates the committed,
already-scrubbed ``sanitizer_report.json`` fixtures under ``reports/<case>/`` and
writes a ``{"cases": [...]}`` spec that ``survey_cases_from_spec`` consumes. It
does **not** run GPUs or fabricate results -- the committed reports are the
recorded outputs; this only maps them into spec cases. Re-running reproduces the
committed spec byte-for-byte (see ``tests/sanitizers/test_survey_generic_gemm.py``).

Public-safe policy (CLAUDE.md rule #4): every case is generically named. The
GEMM lane scans the *generic public* hipBLASLt gfx950 Tensile object; the
synthetic ``tiny_vecadd`` / ``lds_reduce`` controls are ordinary repros. No
customer, NDA, or ticket identifiers appear here or in the fixtures. The scrub
guard that enforces this (a forbidden-token denylist over the committed spec,
fixtures, and rendered output) lives in
``tests/sanitizers/test_survey_generic_gemm.py``.

Regenerate the committed spec with:

    python scripts/sanitizers/gen_survey_spec.py \
        --reports-dir recipes/sanitizers/survey/reports \
        --out recipes/sanitizers/survey/generic_gemm_survey.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Ordered survey cases: (report case dir, stable name, display label, workload, backend).
# Both sanitizers (waitcheck static + ConSan dynamic) are represented for every
# kernel, so the survey shows each kernel under each tool it was observed with.
CASES: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "gemm_f32_waitcheck",
        "hipblaslt-gemm-f32-nt-128x128-waitcheck",
        "hipBLASLt GEMM f32 nt 128x128 \u00b7 waitcheck (static)",
        "hipblaslt:gemm_f32",
        "waitcheck (static)",
    ),
    (
        "gemm_f32_consan",
        "hipblaslt-gemm-f32-nt-128x128-consan",
        "hipBLASLt GEMM f32 nt 128x128 \u00b7 ConSan (dynamic)",
        "hipblaslt:gemm_f32",
        "consan (dynamic)",
    ),
    (
        "tiny_vecadd_waitcheck",
        "tiny-vecadd-waitcheck",
        "tiny_vecadd \u00b7 waitcheck (static)",
        "synthetic:vecadd",
        "waitcheck (static)",
    ),
    (
        "tiny_vecadd_consan",
        "tiny-vecadd-consan",
        "tiny_vecadd \u00b7 ConSan (dynamic)",
        "synthetic:vecadd",
        "consan (dynamic)",
    ),
    (
        "lds_reduce_waitcheck",
        "lds-reduce-waitcheck",
        "lds_reduce \u00b7 waitcheck (static)",
        "synthetic:lds_reduce",
        "waitcheck (static)",
    ),
    (
        "lds_reduce_consan",
        "lds-reduce-consan",
        "lds_reduce \u00b7 ConSan (dynamic)",
        "synthetic:lds_reduce",
        "consan (dynamic)",
    ),
)

def build_spec(reports_dir: Path, *, report_root: Path | None = None) -> dict:
    """Build the ``{"cases": [...]}`` survey spec from the committed report tree.

    ``report_path`` is emitted relative to ``report_root`` (the spec's own
    directory, so the dashboard resolves fixtures next to the spec at render
    time). ``report_rel`` is the *published* drill-down link (``survey/<case>/``,
    co-located next to ``index.html`` by the publish step), a safe same-origin
    relative path.
    """
    root = report_root if report_root is not None else reports_dir
    cases: list[dict] = []
    for case_dir, name, label, workload, backend in CASES:
        report_file = reports_dir / case_dir / "sanitizer_report.json"
        report = json.loads(report_file.read_text(encoding="utf-8"))
        _assert_public_safe(report, case_dir)
        rel_path = (reports_dir / case_dir / "sanitizer_report.json").relative_to(root)
        cases.append(
            {
                "name": name,
                "label": label,
                "backend": backend,
                "workload": workload,
                "report_path": rel_path.as_posix(),
                "report_rel": f"survey/{case_dir}/sanitizer_report.json",
            }
        )
    return {"cases": cases}


def _assert_public_safe(report: dict, case_dir: str) -> None:
    # Schema check only; the forbidden-token scrub guard lives in the test module
    # so this generator carries no denylist literals of its own.
    if report.get("schema") != "aorta.sanitizer_report/0.1":
        raise SystemExit(f"{case_dir}: unexpected schema {report.get('schema')!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("recipes/sanitizers/survey/reports"),
        help="directory of <case>/sanitizer_report.json survey fixtures",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("recipes/sanitizers/survey/generic_gemm_survey.json"),
        help="survey spec JSON to write",
    )
    args = parser.parse_args()
    # report_path is resolved by the dashboard relative to the spec's own dir, so
    # compute it relative to the spec's parent directory.
    spec = build_spec(args.reports_dir, report_root=args.out.parent)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(spec['cases'])} survey cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
