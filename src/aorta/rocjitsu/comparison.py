"""Deterministic comparison of expert claims and independent ConSan evidence."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ComparisonStatus(str, Enum):
    CONFIRMED_BY_CONSAN = "confirmed_by_consan"
    NOT_CONFIRMED_BY_CONSAN = "not_confirmed_by_consan"
    CONSAN_FINDING_EXPERT_NONE = "consan_finding_expert_none"
    DIFFERENT_FINDINGS = "different_findings"
    NO_CONSAN = "no_consan"
    EXPERT_NONE = "expert_none"
    EXPERT_ERROR = "expert_error"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class ShadowComparison:
    status: ComparisonStatus
    expert_class: str = ""
    expert_kernel: str = ""
    consan_kernels: tuple[str, ...] = ()
    rationale: str = ""


def _consan(report: dict[str, Any] | None) -> tuple[bool, tuple[str, ...], bool]:
    if not report:
        return False, (), False
    checks = [check for check in report.get("checks", []) if check.get("sanitizer") == "consan"]
    if not checks:
        return False, (), False
    findings = [finding for check in checks for finding in check.get("findings", [])]
    kernels = tuple(
        dict.fromkeys(
            finding["kernel_name"]
            for finding in findings
            if isinstance(finding.get("kernel_name"), str) and finding["kernel_name"]
        )
    )
    return True, kernels, bool(findings)


def compare_reports(
    expert: dict[str, Any],
    sanitizer: dict[str, Any] | None,
) -> ShadowComparison:
    """Compare two reports without changing either report's verdict."""
    if expert.get("live_job_input_analyzed") is not True:
        return ShadowComparison(
            ComparisonStatus.NOT_APPLICABLE,
            rationale="expert report did not analyze the live CIA job input",
        )
    if expert.get("status") != "complete":
        return ShadowComparison(
            ComparisonStatus.EXPERT_ERROR,
            rationale=f"expert status is {expert.get('status', 'missing')}",
        )
    answer = expert.get("answer") or {}
    expert_class = str(answer.get("class") or "")
    expert_kernel = str(answer.get("kernel") or "")
    ran_consan, consan_kernels, has_findings = _consan(sanitizer)
    if not ran_consan:
        return ShadowComparison(
            ComparisonStatus.NO_CONSAN,
            expert_class,
            expert_kernel,
            rationale="no independent ConSan result exists",
        )
    if expert_class in ("", "none"):
        return ShadowComparison(
            ComparisonStatus.CONSAN_FINDING_EXPERT_NONE
            if has_findings
            else ComparisonStatus.EXPERT_NONE,
            expert_class,
            expert_kernel,
            consan_kernels,
            "ConSan found a race the expert missed" if has_findings else "neither report found a bug",
        )
    if not has_findings:
        return ShadowComparison(
            ComparisonStatus.NOT_CONFIRMED_BY_CONSAN,
            expert_class,
            expert_kernel,
            consan_kernels,
            "ConSan ran without a matching finding; the expert claim remains unconfirmed",
        )
    if not expert_kernel or not consan_kernels:
        return ShadowComparison(
            ComparisonStatus.NOT_CONFIRMED_BY_CONSAN,
            expert_class,
            expert_kernel,
            consan_kernels,
            "ConSan finding lacks the kernel identity needed for confirmation",
        )
    if expert_kernel and consan_kernels and expert_kernel not in consan_kernels:
        return ShadowComparison(
            ComparisonStatus.DIFFERENT_FINDINGS,
            expert_class,
            expert_kernel,
            consan_kernels,
            "expert and ConSan named different kernels",
        )
    return ShadowComparison(
        ComparisonStatus.CONFIRMED_BY_CONSAN,
        expert_class,
        expert_kernel,
        consan_kernels,
        "ConSan independently observed a matching race",
    )


def summarize(comparisons: Iterable[ShadowComparison]) -> dict[str, Any]:
    statuses = Counter(comparison.status.value for comparison in comparisons)
    return {"total": sum(statuses.values()), "statuses": dict(sorted(statuses.items()))}


def compare_job(job_dir: Path) -> Path:
    """Write a comparison beside the expert report, leaving Autopsy untouched."""
    job_dir = job_dir.resolve()
    expert_path = job_dir / "rj_expert" / "report.json"
    sanitizer_path = job_dir / "bundle" / "aorta" / "sanitizer_report.json"
    expert = json.loads(expert_path.read_text(encoding="utf-8"))
    sanitizer = (
        json.loads(sanitizer_path.read_text(encoding="utf-8"))
        if sanitizer_path.is_file()
        else None
    )
    comparison = compare_reports(expert, sanitizer)
    output = expert_path.parent / "shadow_comparison.json"
    payload = json.dumps(asdict(comparison), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "ComparisonStatus",
    "ShadowComparison",
    "compare_job",
    "compare_reports",
    "summarize",
]
