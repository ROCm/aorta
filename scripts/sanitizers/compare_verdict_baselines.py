#!/usr/bin/env python3
"""Compare sanitizer nightly reports against committed verdict baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: compare_verdict_baselines.py <results-root>", file=sys.stderr)
        return 2
    root = Path(argv[1])
    baselines = json.loads(
        Path("recipes/sanitizers/fixtures/expected/verdict_baselines.json").read_text()
    )
    cases = {
        "waitcheck_gemm": root / "waitcheck" / "sanitizer_report.json",
        "consan_clean": root / "consan-clean" / "sanitizer_report.json",
        "consan_racy": root / "consan-racy" / "sanitizer_report.json",
    }
    failed = False
    for name, report_path in cases.items():
        if not report_path.is_file():
            print(f"missing report for {name}: {report_path}")
            failed = True
            continue
        report = _load_report(report_path)
        expected = baselines[name]["overall_verdict"]
        actual = report.get("overall_verdict")
        if actual != expected:
            print(f"{name}: expected overall_verdict={expected!r}, got {actual!r}")
            failed = True
            continue
        print(f"{name}: ok ({actual})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
