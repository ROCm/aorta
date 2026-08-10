"""Public sanitizer pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .consan import run_consan, scoped_consan_not_checked
from .models import CheckResult, KernelIdentity, KernelWorklist, SanitizerReport
from .report import build_report, write_report
from .waitcheck import run_waitcheck

_KNOWN_SANITIZERS = frozenset({"waitcheck", "consan"})

# Default wall-clock ceiling for a single sanitizer subprocess (waitcheck or
# consan). ConSan's MOI transform of large production code objects can be heavy,
# so a recipe may raise this via ``sanitizer_plan.policy.timeout_seconds``.
DEFAULT_TIMEOUT_SECONDS = 900.0


def run_sanitizers(
    worklist: KernelWorklist,
    *,
    target: str,
    sanitizers: Iterable[str],
    output_dir: Path,
    waitcheck_binary: Path | None = None,
    consan_command: Path | None = None,
    consan_hook: Path | None = None,
    consan_log: bool = True,
    consan_policy: str = "strict",
    on_missing_backend: str = "fail",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    report_name: str = "sanitizer_report.json",
    consan_target: KernelIdentity | None = None,
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
            if consan_command is not None:
                combined = run_consan(
                    worklist,
                    command=consan_command,
                    hook_lib=consan_hook,
                    output_dir=output_dir / "consan",
                    consan_log=consan_log,
                    timeout_seconds=timeout_seconds,
                    strict=consan_policy == "strict",
                    target=consan_target,
                )
                # Surface the mandatory combined-hook Waitcheck preflight as its
                # own check so an unhealthy preflight fails the run closed at
                # report scope instead of being masked by a clean ConSan verdict.
                results.append(combined.consan)
                results.append(combined.waitcheck_preflight)
            else:
                # ConSan is always fail-closed when no targeted repro command is
                # provisioned: it never falls back to whole-application
                # instrumentation, so every on_missing_backend policy resolves to
                # not_checked here. The recipe loader restricts the value to
                # "fail" so a typo can't imply a (non-existent) softer policy.
                _ = on_missing_backend
                results.append(scoped_consan_not_checked(worklist))

    report = build_report(
        target=target,
        worklist=worklist,
        checks=tuple(results),
    )
    write_report(report, output_dir / report_name)
    return report
