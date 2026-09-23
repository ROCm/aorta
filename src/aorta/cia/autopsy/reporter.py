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
    confidence_source: dict[str, Any] | None = None,
    escalation: dict[str, Any] | None = None,
    laya: dict[str, Any] | None = None,
    rationale_caveat: str = "",
) -> dict[str, Any]:
    """Assemble the Autopsy report.

    The last four arguments say where ``confidence`` came from, what the
    escalation cutoff was applied to, which checkpoint answered when one did,
    and anything that has to be said about the number before a reader trusts
    it. They are optional so that a caller assembling a report from something
    other than ``run_autopsy`` still works, and they are written only when
    supplied so that a bundle nothing new touched produces the bytes it always
    did.

    Recording provenance inline rather than beside the run is rule 2 of
    Decision 22 in ``docs/laya-packaging.md``: a report is copied into a ticket
    and read on its own, and until now it could not answer "which thing said
    this?" for a field two different sources have always been able to write.

    *rationale_caveat* is appended **after** the 2000-character cap rather than
    before it, and that ordering is the whole reason it is a parameter instead
    of something the caller concatenates. The cap exists because a model can
    write at length; a disclosure concatenated upstream would be the tail of a
    long rationale and therefore the first thing the slice deleted, silently,
    on exactly the verbose reports most likely to be read carefully.
    """
    report: dict[str, Any] = {
        "schema_version": "0.1",
        "session_id": session_id,
        "generated_at": generated_at,
        "phase": "autopsy",
        "bundle": {"job_id": bundle_job_id, "root": bundle_root},
        "category": category,
        "confidence": round(confidence, 3),
        "rationale": rationale[:2000] + rationale_caveat,
        "evidence": evidence,
        "next_probes": next_probes,
        "tooling_gaps": tooling_gaps,
    }
    if kb_version:
        report["kb_version"] = kb_version
    if confidence_source:
        report["confidence_source"] = confidence_source
    if escalation:
        report["escalation"] = escalation
    if laya:
        report["laya"] = laya
    return report
