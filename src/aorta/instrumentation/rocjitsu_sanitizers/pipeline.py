"""Public Phase-0 sanitizer pipeline.

Waitcheck is executable for exact worklist entries. ConSan is intentionally
fail-closed until RocJITsu can enforce a kernel allowlist.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .consan import scoped_consan_not_checked
from .models import CheckResult, KernelWorklist, SanitizerReport
from .report import build_report, write_report
from .waitcheck import run_waitcheck

_KNOWN_SANITIZERS = frozenset({"waitcheck", "consan"})


def run_sanitizers(
    worklist: KernelWorklist,
    *,
    target: str,
    sanitizers: Iterable[str],
    output_dir: Path,
    waitcheck_binary: Path | None = None,
    timeout_seconds: float = 900.0,
) -> SanitizerReport:
    """Run supported checks and persist one versioned report."""

    requested = tuple(dict.fromkeys(sanitizers))
    unknown = sorted(set(requested) - _KNOWN_SANITIZERS)
    if unknown:
        raise ValueError(f"unknown sanitizers: {unknown}")
    if not requested:
        raise ValueError("at least one sanitizer must be requested")
    if any(kernel.identity.target != target for kernel in worklist.kernels):
        raise ValueError("worklist target does not match requested target")

    results: list[CheckResult] = []
    for sanitizer in requested:
        if sanitizer == "waitcheck":
            results.append(
                run_waitcheck(
                    worklist,
                    output_dir=output_dir / "waitcheck",
                    binary=waitcheck_binary,
                    timeout_seconds=timeout_seconds,
                )
            )
        elif sanitizer == "consan":
            results.append(scoped_consan_not_checked(worklist))

    report = build_report(
        target=target,
        worklist=worklist,
        checks=tuple(results),
    )
    write_report(report, output_dir / "sanitizer_report.json")
    return report
