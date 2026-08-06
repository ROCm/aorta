"""Persistence helpers for versioned sanitizer reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .models import CheckResult, KernelWorklist, SanitizerReport


def build_report(
    *,
    target: str,
    worklist: KernelWorklist,
    checks: tuple[CheckResult, ...],
) -> SanitizerReport:
    if any(kernel.identity.target != target for kernel in worklist.kernels):
        raise ValueError("worklist target does not match report target")
    return SanitizerReport(target=target, worklist=worklist, checks=checks)


def write_report(report: SanitizerReport, path: Path) -> None:
    """Atomically write and validate a non-empty JSON report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            report.to_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if temporary.stat().st_size == 0:
        raise OSError(f"sanitizer report write produced an empty file: {temporary}")
    temporary.replace(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"sanitizer report is missing or empty: {path}")


def read_report(path: Path) -> SanitizerReport:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"sanitizer report is missing or empty: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise TypeError("sanitizer report root must be an object")
    data: Mapping[str, object] = raw
    return SanitizerReport.from_dict(data)
