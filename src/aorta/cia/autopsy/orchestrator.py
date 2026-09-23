from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aorta.cia.autopsy.adapters.aorta_matrix import AortaMatrixAdapter, load_matrix, MatrixClassification, classify_matrix
from aorta.cia.autopsy.adapters.base import AdapterArtifact, BundleContext, load_manifest
from aorta.cia.autopsy.adapters.sanitizer_report import (
    SanitizerClassification,
    SanitizerReportAdapter,
    classify_sanitizer,
)
from aorta.cia.autopsy.adapters.rocgdb import (
    RocgdbAdapter,
    RocgdbClassification,
    classify_rocgdb,
    parse_rocgdb_session,
)
from aorta.cia.autopsy.adapters.stderr_watch import StderrWatchAdapter
from aorta.cia.autopsy import escalation
from aorta.cia.autopsy.reporter import build_report


log = logging.getLogger(__name__)


def _reviewed_rationale(
    classification: MatrixClassification, router_rationale: str
) -> str:
    """Keep a debugger-proven cause/fix when the LLM reviews the evidence.

    ROCgDB's adapter deterministically turns ``mean_sq=0``, ``inv_rms=inf`` and
    ``y=NaN`` into the missing-epsilon root cause and exact ``rsqrtf`` fix. The
    router used to replace that rationale wholesale; a perfectly plausible
    review could retain the values while dropping the fix, leaving the final
    chatbot to invent alternatives such as skipping padding rows.
    """
    reviewed = str(router_rationale or "").strip()
    if "DBG_NAN_TRAP" not in classification.signals:
        return reviewed or classification.rationale

    proven = classification.rationale.strip()
    if not reviewed or reviewed == proven:
        return proven
    return f"{proven} LLM review: {reviewed}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_autopsy(
    bundle_root: Path,
    *,
    kb_version: str | None = None,
    use_llm: bool = True,
    job: "Any | None" = None,
    head_node: str = "",
    stop: Stop = None,
) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    manifest = load_manifest(bundle_root)
    ctx = BundleContext(
        root=bundle_root,
        manifest=manifest,
        job_id=str(manifest.get("job_id", bundle_root.name)),
    )

    adapters = [
        AortaMatrixAdapter(),
        StderrWatchAdapter(),
        SanitizerReportAdapter(),
        RocgdbAdapter(),
    ]
    artifacts: list[AdapterArtifact] = []
    for adapter in adapters:
        artifacts.append(adapter.collect(ctx))

    matrix_adapter = artifacts[0]
    stderr_adapter = artifacts[1]
    # Through the same loader the adapter uses, so a matrix that cannot be read
    # is a tooling gap here too rather than an exception out of run_autopsy. It
    # used to be parsed twice, uncaught, and the second parse aborted the whole
    # autopsy -- discarding the sanitizer and watchdog evidence that had already
    # been collected, over an artifact that may simply have been half-written.
    matrix = load_matrix(ctx.path("aorta_matrix"))
    if matrix is not None:
        classification = classify_matrix(matrix)
    else:
        classification = MatrixClassification(
            category="tooling_gap",
            confidence=0.0,
            rationale="No readable Aorta matrix artifact in bundle.",
            signals=[],
        )

    classification = merge_watchdog_matrix(classification, stderr_adapter)

    sanitizer_path = ctx.path("sanitizer_report")
    if sanitizer_path and sanitizer_path.is_file():
        try:
            sanitizer_report = json.loads(sanitizer_path.read_text(encoding="utf-8"))
        except Exception:
            sanitizer_report = {}
        if sanitizer_report:
            classification = merge_sanitizer(classification, classify_sanitizer(sanitizer_report))

    rocgdb_path = ctx.path("rocgdb_session")
    if rocgdb_path and rocgdb_path.is_file():
        try:
            session = parse_rocgdb_session(
                rocgdb_path.read_text(encoding="utf-8", errors="replace")
            )
        except (OSError, ValueError) as exc:
            log.warning("Could not read the rocgdb session at %s: %s", rocgdb_path, exc)
        else:
            classification = merge_rocgdb(classification, classify_rocgdb(session))

    all_evidence: list[dict[str, Any]] = []
    all_next: list[dict[str, Any]] = []
    all_gaps: list[dict[str, Any]] = []
    for art in artifacts:
        all_evidence.extend(art.evidence)
        all_next.extend(art.next_probes)
        all_gaps.extend(art.tooling_gaps)

    # The adapters' own figure, which exists for every bundle whether or not any
    # model runs. It is the source the 0.85 escalation cutoff was chosen
    # against -- the prompt's `~0.62` is the same 0.62 `merge_watchdog_matrix`
    # hard-codes -- so it is what `escalation.decide` falls back to when the
    # reported confidence comes from somewhere that cutoff does not describe.
    rule_based = escalation.Confidence(classification.confidence, escalation.ADAPTER_RULES)
    reported = rule_based
    laya_threshold: float | None = None
    laya_fields: dict[str, Any] | None = None

    # LLM router — re-classifies based on all adapter evidence
    if use_llm and all_evidence:
        try:
            from aorta.cia.autopsy.router import TriageRouter
            # The root goes to the constructor, not into the call the model
            # can influence: the evidence tool is bound to it there.
            router = TriageRouter(bundle_root)
            pred = router(
                evidence=all_evidence,
                job_context=ctx.job_id,
            )
            category = getattr(pred, "category", classification.category)
            confidence = float(getattr(pred, "confidence", classification.confidence))
            rationale = _reviewed_rationale(
                classification,
                getattr(pred, "rationale", classification.rationale),
            )
            next_probe = getattr(pred, "next_probe", "none")
            next_probe_reason = getattr(pred, "next_probe_reason", "")
            # Which of the two things behind the router produced that number.
            # The tier writes itself onto the prediction when it answered, and
            # its absence is how Autopsy says the model self-reported -- the
            # distinction the escalation cutoff has always depended on and has
            # never until now been able to see.
            observation = getattr(pred, "laya", None)
            if observation is None:
                reported = escalation.Confidence(confidence, escalation.LLM_SELF_REPORT)
            else:
                reported = escalation.Confidence(
                    confidence,
                    escalation.LAYA,
                    observation.model_id,
                    caveat=observation.caveat,
                )
                laya_threshold = observation.escalation_threshold
                laya_fields = observation.as_report_fields()
        except Exception as e:
            # A verdict reached without the router is a weaker claim than one
            # reached with it, and the difference is invisible in the category
            # alone -- the adapters are confident on their own. Record it as a
            # tooling gap so a reader of the report can tell, rather than only
            # someone watching stderr at the time.
            log.warning(
                "Autopsy router unavailable (%s); classifying from adapter "
                "evidence alone. The verdict stands but is not LLM-reviewed.",
                e,
            )
            all_gaps.append(
                {
                    "description": f"Autopsy router unavailable ({e}); "
                                   "verdict derived from adapters alone.",
                    "missing_signal": "LLM_ROUTER_REVIEW",
                    "suggested_tool": "none",
                }
            )
            category = classification.category
            if category == "unknown" and matrix_adapter.tooling_gaps:
                category = "tooling_gap"
            confidence = classification.confidence
            rationale = classification.rationale
            next_probe = all_next[0]["tool"] if all_next else "none"
            next_probe_reason = ""
            reported = rule_based
    else:
        category = classification.category
        if category == "unknown" and matrix_adapter.tooling_gaps:
            category = "tooling_gap"
        confidence = classification.confidence
        rationale = classification.rationale
        next_probe = all_next[0]["tool"] if all_next else "none"
        next_probe_reason = ""

    # Escalate: run the production Aorta sweep when the router recommended one
    # and the confidence behind it is low. Which confidence that is, and against
    # which cutoff, is `escalation.decide`'s to answer -- the two were an
    # unwritten pairing until a third source of the number arrived.
    decision = escalation.decide(
        next_probe, reported, rule_based=rule_based, laya_threshold=laya_threshold
    )
    if decision.escalate and job is not None:
        print(f"[autopsy] {decision.reason} — escalating to Aorta production sweep")
        from aorta.cia.autopsy.probe import run_aorta_probe
        matrix_path = run_aorta_probe(bundle_root, job, head_node=head_node, stop=stop)
        if matrix_path:
            # Re-run with the new production matrix (use_llm stays True, no infinite loop
            # because a production matrix classifies above the cutoff)
            return run_autopsy(
                bundle_root, kb_version=kb_version, use_llm=use_llm,
                job=None, head_node=head_node,
            )

    if not all_evidence and category == "tooling_gap":
        all_evidence.append(
            {
                "uri": "manifest.yaml",
                "line_start": 1,
                "line_end": 1,
                "excerpt": "missing aorta_matrix path",
                "adapter": "aorta_matrix",
                "signal": "MISSING_ARTIFACT",
            }
        )

    report = build_report(
        session_id=str(uuid.uuid4()),
        generated_at=_utc_now(),
        bundle_job_id=ctx.job_id,
        bundle_root=str(bundle_root),
        kb_version=kb_version,
        category=category,
        confidence=confidence,
        rationale=rationale,
        evidence=all_evidence,
        next_probes=all_next,
        tooling_gaps=all_gaps,
        confidence_source=reported.as_report_fields(),
        escalation=decision.as_report_fields(),
        laya=laya_fields,
        # The rationale is the one field of this report a person actually
        # reads, so it is where a "this number is not calibrated" has to end
        # up. Passed separately rather than concatenated here because
        # ``build_report`` caps the rationale and the cap must not be what
        # deletes the disclosure.
        rationale_caveat=reported.caveat,
    )
    return report


def merge_sanitizer(
    base: MatrixClassification,
    sanitizer: SanitizerClassification,
) -> MatrixClassification:
    """Fold a sanitizer verdict into the matrix/log verdict.

    ConSan names a concrete ordering violation, which is a harder statement than the
    statistical NaN evidence a matrix or watch log can offer, so an observed race
    wins. A sanitizer that never ran must not drag down an otherwise sound verdict,
    which is why the inconclusive categories only apply when nothing else concluded.
    """
    if sanitizer.category in {"unknown", "tooling_gap"}:
        if base.category in {"unknown", "tooling_gap"} and base.confidence <= 0.0:
            return MatrixClassification(
                category=sanitizer.category,
                confidence=sanitizer.confidence,
                rationale=sanitizer.rationale,
                signals=list(dict.fromkeys([*base.signals, *sanitizer.signals])),
            )
        return base

    if sanitizer.confidence < base.confidence and base.category not in {"unknown", "tooling_gap"}:
        return base

    return MatrixClassification(
        category=sanitizer.category,
        confidence=sanitizer.confidence,
        rationale=sanitizer.rationale,
        signals=list(dict.fromkeys([*sanitizer.signals, *base.signals])),
    )


def merge_rocgdb(
    base: MatrixClassification,
    debugger: RocgdbClassification,
) -> MatrixClassification:
    """Fold a ROCgDB device-assert verdict into the existing verdict.

    A trapped assert names the kernel, the source line, and the register values
    that went non-finite, which is a harder statement than a log signature or a
    statistical matrix, so it wins when it concluded anything. An inconclusive
    session must not drag down a verdict another adapter already reached.
    """
    if debugger.category == "unknown":
        return base

    return MatrixClassification(
        category=debugger.category,
        confidence=max(debugger.confidence, base.confidence),
        rationale=debugger.rationale,
        signals=list(dict.fromkeys([*debugger.signals, *base.signals])),
    )


def merge_watchdog_matrix(
    matrix_cls: MatrixClassification,
    stderr_art: AdapterArtifact,
) -> MatrixClassification:
    watch_nan = "WATCH_NUMERIC_NAN" in stderr_art.signals

    if matrix_cls.category == "numeric_silent":
        if not watch_nan:
            return matrix_cls
        return MatrixClassification(
            category=matrix_cls.category,
            confidence=min(0.95, matrix_cls.confidence + 0.05),
            rationale=(
                f"{matrix_cls.rationale} Watchdog log corroborates NaN/non-finite loss."
            ),
            signals=list(dict.fromkeys([*matrix_cls.signals, "WATCH_NUMERIC_NAN"])),
        )

    if watch_nan:
        return MatrixClassification(
            category="numeric_silent",
            confidence=0.62,
            rationale=(
                "Watchdog detected NaN/non-finite loss in training log. "
                "Aorta matrix is smoke-only or inconclusive — run the production "
                "Residual-NaN recipe to confirm TF32 vs deterministic isolation."
            ),
            signals=list(dict.fromkeys(["WATCH_NUMERIC_NAN", *matrix_cls.signals])),
        )

    return matrix_cls
