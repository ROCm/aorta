#!/usr/bin/env python3
"""Compare sanitizer nightly reports against committed verdict baselines.

The comparator is the Phase-1 gate, so it must fail closed rather than trust a
single top-level field. Each report is *strictly* reloaded through
``read_report`` (schema + internal-consistency validation, e.g. that
``overall_verdict``/``execution_status`` actually agree with the checks), and
then compared against the committed baseline's ``overall_verdict``,
``execution_status`` (when declared), per-check verdicts, and finding shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aorta.instrumentation.rocjitsu_sanitizers.models import SanitizerReport
from aorta.instrumentation.rocjitsu_sanitizers.report import read_report

_BASELINES = Path("recipes/sanitizers/fixtures/expected/verdict_baselines.json")
_CASES = {
    "waitcheck_gemm": Path("waitcheck") / "sanitizer_report.json",
    "consan_clean": Path("consan-clean") / "sanitizer_report.json",
    "consan_racy": Path("consan-racy") / "sanitizer_report.json",
}


def _load_baselines(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"baselines file is empty or malformed: {path}")
    return data


def _check_by_name(report: SanitizerReport, sanitizer: str):
    for check in report.checks:
        if check.sanitizer == sanitizer:
            return check
    return None


def _compare_case(name: str, report: SanitizerReport, expected: dict[str, object]) -> list[str]:
    problems: list[str] = []
    want_verdict = expected.get("overall_verdict")
    if report.overall_verdict.value != want_verdict:
        problems.append(
            f"overall_verdict={report.overall_verdict.value!r}, expected {want_verdict!r}"
        )
    want_execution = expected.get("execution_status")
    if want_execution is not None and report.execution_status.value != want_execution:
        problems.append(
            f"execution_status={report.execution_status.value!r}, expected {want_execution!r}"
        )
    for sanitizer, want_check in (expected.get("checks") or {}).items():
        check = _check_by_name(report, sanitizer)
        if check is None:
            problems.append(f"missing check {sanitizer!r}")
        elif check.verdict.value != want_check:
            problems.append(
                f"check {sanitizer!r} verdict={check.verdict.value!r}, expected {want_check!r}"
            )
    for sanitizer, want_shape in (expected.get("finding_shape") or {}).items():
        check = _check_by_name(report, sanitizer)
        messages = [] if check is None else [finding.message for finding in check.findings]
        if not any(want_shape in message for message in messages):
            problems.append(
                f"check {sanitizer!r} findings do not contain expected shape {want_shape!r}"
            )
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: compare_verdict_baselines.py <results-root>", file=sys.stderr)
        return 2
    root = Path(argv[1])
    try:
        baselines = _load_baselines(_BASELINES)
    except (OSError, ValueError) as exc:
        print(f"error: could not read baselines {_BASELINES}: {exc}", file=sys.stderr)
        return 2

    failed = False
    for name, relative in _CASES.items():
        expected = baselines.get(name)
        if not isinstance(expected, dict):
            print(f"{name}: no baseline entry")
            failed = True
            continue
        report_path = root / relative
        if not report_path.is_file():
            print(f"{name}: missing report {report_path}")
            failed = True
            continue
        try:
            report = read_report(report_path)
        except (OSError, ValueError, TypeError) as exc:
            print(f"{name}: report failed strict validation: {exc}")
            failed = True
            continue
        problems = _compare_case(name, report, expected)
        if problems:
            failed = True
            for problem in problems:
                print(f"{name}: {problem}")
        else:
            print(f"{name}: ok ({report.overall_verdict.value})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
