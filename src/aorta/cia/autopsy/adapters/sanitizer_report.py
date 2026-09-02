from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aorta.cia.autopsy.adapters.base import AdapterArtifact, BundleContext

SCHEMA = "aorta.sanitizer_report/0.1"

# Signals this adapter contributes to the evidence stream.
SIGNAL_RACE = "SAN_CONSAN_RACE"
SIGNAL_HAZARD = "SAN_WAITCHECK_HAZARD"
SIGNAL_CLEAN = "SAN_CLEAN"
SIGNAL_NOT_CHECKED = "SAN_NOT_CHECKED"

# A backend that never ran says nothing about the workload, so these reasons are
# reported as a tooling gap rather than a clean bill of health.
MISSING_BACKEND_REASONS = frozenset(
    {
        "rj_waitcheck_not_found",
        "consan_hook_not_found",
        "consan_command_not_provisioned",
        "dry_run",
    }
)


@dataclass(frozen=True)
class SanitizerClassification:
    category: str
    confidence: float
    rationale: str
    signals: list[str]


def _findings(check: dict[str, Any]) -> list[dict[str, Any]]:
    """All findings for a check, including any attributed per-kernel results."""
    out: list[dict[str, Any]] = list(check.get("findings") or [])
    for kernel in check.get("kernel_results") or []:
        out.extend(kernel.get("findings") or [])
    return out


def _signal_for(finding: dict[str, Any]) -> str:
    severity = str(finding.get("severity") or "").lower()
    sanitizer = str(finding.get("sanitizer") or "").lower()
    if severity == "race" or sanitizer == "consan":
        return SIGNAL_RACE
    return SIGNAL_HAZARD


def classify_sanitizer(report: dict[str, Any]) -> SanitizerClassification:
    """Rule-based verdict from a sanitizer report, before any LLM involvement.

    An observed ConSan race is a definite finding, so it outranks a waitcheck
    hazard, which is a static warning that a kernel *may* be missing a wait and
    needs a dynamic run to confirm.
    """
    checks = report.get("checks") or []
    races: list[dict[str, Any]] = []
    hazards: list[dict[str, Any]] = []
    not_checked: list[str] = []

    for check in checks:
        state = str(check.get("state") or "")
        if state != "ran":
            reason = str(check.get("reason") or state or "unknown")
            not_checked.append(f"{check.get('sanitizer', '?')}: {reason}")
            continue
        for finding in _findings(check):
            if _signal_for(finding) == SIGNAL_RACE:
                races.append(finding)
            else:
                hazards.append(finding)

    target = str(report.get("target") or "unknown")
    verdict = str(report.get("overall_verdict") or "")

    if races:
        first = races[0]
        where = first.get("kernel_name") or first.get("code_object") or "an unattributed kernel"
        return SanitizerClassification(
            category="gpu_race",
            confidence=0.93,
            rationale=(
                f"ConSan reported {len(races)} race finding(s) on {target} during "
                f"record/replay; first involves {where}. A replay divergence means two "
                f"waves reached the same memory without ordering between them."
            ),
            signals=[SIGNAL_RACE],
        )

    if hazards:
        first = hazards[0]
        where = first.get("kernel_name") or first.get("code_object") or "an unattributed kernel"
        return SanitizerClassification(
            category="gpu_race",
            confidence=0.55,
            rationale=(
                f"waitcheck flagged {len(hazards)} wait hazard(s) on {target}; first in "
                f"{where}. This is a static missing-s_waitcnt warning, not an observed "
                f"race — a ConSan record/replay run would confirm whether it is reachable."
            ),
            signals=[SIGNAL_HAZARD],
        )

    if not_checked:
        return SanitizerClassification(
            category="tooling_gap",
            confidence=0.0,
            rationale=(
                "No sanitizer actually executed, so the run is inconclusive: "
                + "; ".join(not_checked[:4])
            ),
            signals=[SIGNAL_NOT_CHECKED],
        )

    return SanitizerClassification(
        category="unknown",
        confidence=0.2,
        rationale=(
            f"Sanitizers ran on {target} and reported no findings "
            f"(overall_verdict={verdict or 'unset'})."
        ),
        signals=[SIGNAL_CLEAN],
    )


class SanitizerReportAdapter:
    """Read an Aorta sanitizer_report.json and surface ConSan/waitcheck findings."""

    adapter_id = "sanitizer_report"

    def collect(self, ctx: BundleContext) -> AdapterArtifact:
        report_path = ctx.path("sanitizer_report")
        if report_path is None or not report_path.is_file():
            return AdapterArtifact(adapter=self.adapter_id)

        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:
            return AdapterArtifact(
                adapter=self.adapter_id,
                tooling_gaps=[
                    {
                        "description": f"sanitizer_report.json is unreadable: {e}",
                        "missing_signal": SCHEMA,
                        "suggested_tool": "aorta sweep run",
                    }
                ],
            )

        schema = str(report.get("schema") or "")
        if schema != SCHEMA:
            return AdapterArtifact(
                adapter=self.adapter_id,
                tooling_gaps=[
                    {
                        "description": f"unexpected sanitizer report schema {schema!r}, want {SCHEMA}",
                        "missing_signal": SCHEMA,
                        "suggested_tool": "aorta sweep run",
                    }
                ],
            )

        rel = _bundle_rel(ctx, report_path)
        classification = classify_sanitizer(report)
        evidence: list[dict[str, Any]] = []
        signals: list[str] = []
        tooling_gaps: list[dict[str, Any]] = []
        next_probes: list[dict[str, Any]] = []
        per_sanitizer: dict[str, str] = {}

        for check in report.get("checks") or []:
            name = str(check.get("sanitizer") or "?")
            per_sanitizer[name] = str(check.get("verdict") or "")
            state = str(check.get("state") or "")
            reason = str(check.get("reason") or "")

            if state != "ran":
                if reason in MISSING_BACKEND_REASONS or state == "not_checked":
                    tooling_gaps.append(
                        {
                            "description": (
                                f"{name} did not run ({reason or state}); the RocJITsu "
                                f"backend must be built and ROCJITSU_BUILD exported."
                            ),
                            "missing_signal": SIGNAL_RACE if name == "consan" else SIGNAL_HAZARD,
                            "suggested_tool": "aorta sweep run",
                        }
                    )
                    if SIGNAL_NOT_CHECKED not in signals:
                        signals.append(SIGNAL_NOT_CHECKED)
                continue

            for finding in _findings(check):
                signal = _signal_for(finding)
                if signal not in signals:
                    signals.append(signal)
                # The upstream message already names code object, target and offset,
                # so it is the most useful excerpt we can carry.
                message = str(finding.get("message") or "").strip()
                detail = " ".join(
                    part
                    for part in (
                        f"[{finding.get('severity')}/{finding.get('code')}]",
                        message,
                        f"kernel={finding['kernel_name']}" if finding.get("kernel_name") else "",
                        f"object={finding['code_object']}" if finding.get("code_object") else "",
                    )
                    if part
                )
                evidence.append(
                    {
                        "uri": rel,
                        "line_start": 1,
                        "line_end": 1,
                        "excerpt": detail[:500],
                        "adapter": self.adapter_id,
                        "signal": signal,
                    }
                )

        if not evidence and not tooling_gaps:
            signals.append(SIGNAL_CLEAN)
            evidence.append(
                {
                    "uri": rel,
                    "line_start": 1,
                    "line_end": 1,
                    "excerpt": (
                        f"sanitizers ran clean on {report.get('target', 'unknown')} "
                        f"(overall_verdict={report.get('overall_verdict', 'unset')})"
                    ),
                    "adapter": self.adapter_id,
                    "signal": SIGNAL_CLEAN,
                }
            )

        # A static hazard is worth a dynamic confirmation; an observed race is not.
        if SIGNAL_HAZARD in signals and SIGNAL_RACE not in signals:
            next_probes.append(
                {
                    "tool": "aorta sweep run",
                    "reason": (
                        "waitcheck flagged a static wait hazard — run a ConSan "
                        "record/replay recipe on the same kernel to confirm reachability."
                    ),
                    "overhead_class": "medium",
                }
            )

        return AdapterArtifact(
            adapter=self.adapter_id,
            evidence=evidence,
            signals=signals,
            summary={
                "target": report.get("target"),
                "overall_verdict": report.get("overall_verdict"),
                "execution_status": report.get("execution_status"),
                "per_sanitizer": per_sanitizer,
                "kernel_count": (report.get("worklist") or {}).get("kernel_count"),
                "category": classification.category,
                "confidence": classification.confidence,
            },
            next_probes=next_probes,
            tooling_gaps=tooling_gaps,
        )


def _bundle_rel(ctx: BundleContext, path: Path) -> str:
    try:
        return str(path.relative_to(ctx.root)).replace("\\", "/")
    except ValueError:
        return path.name
