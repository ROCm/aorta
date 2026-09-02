from __future__ import annotations

from typing import Any


def build_report(
    *,
    session_id: str,
    generated_at: str,
    bundle_job_id: str,
    bundle_root: str,
    kb_version: str | None,
    category: str,
    confidence: float,
    rationale: str,
    evidence: list[dict[str, Any]],
    next_probes: list[dict[str, Any]],
    tooling_gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "0.1",
        "session_id": session_id,
        "generated_at": generated_at,
        "phase": "autopsy",
        "bundle": {"job_id": bundle_job_id, "root": bundle_root},
        "category": category,
        "confidence": round(confidence, 3),
        "rationale": rationale[:2000],
        "evidence": evidence,
        "next_probes": next_probes,
        "tooling_gaps": tooling_gaps,
    }
    if kb_version:
        report["kb_version"] = kb_version
    return report
