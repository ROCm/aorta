from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dspy

from aorta.agent.llm import AUTOPSY_CATEGORIES
from aorta.cia.llm import ensure_configured


# ---------------------------------------------------------------------------
# Tools — the rule-based adapters become callable tools for the LLM
# ---------------------------------------------------------------------------

def classify_matrix(matrix_json: str) -> dict:
    """Run the rule-based AortaMatrixAdapter classifier on a matrix JSON string.
    Returns {category, confidence, signals, rationale}."""
    from aorta.cia.autopsy.adapters.aorta_matrix import classify_matrix as _classify
    try:
        matrix = json.loads(matrix_json)
        result = _classify(matrix)
        return {
            "category": result.category,
            "confidence": result.confidence,
            "signals": result.signals,
            "rationale": result.rationale,
        }
    except Exception as e:
        return {"error": str(e)}


def scan_stderr(log_text: str) -> dict:
    """Run regex patterns on log text to detect NaN, hang, OOM signals.
    Returns {signal, alert, hits}."""
    from aorta.cia.autopsy.adapters.stderr_watch import scan_stderr_text
    scan = scan_stderr_text(log_text)
    return {
        "signal": scan.signal,
        "alert": scan.alert,
        "hits": [{"line": ln, "text": ex} for ln, ex in scan.hits[:10]],
    }


def scan_sanitizer(report_json: str) -> dict:
    """Run the rule-based sanitizer classifier on a sanitizer_report.json string.
    Returns {category, confidence, signals, rationale, per_sanitizer}."""
    from aorta.cia.autopsy.adapters.sanitizer_report import classify_sanitizer
    try:
        report = json.loads(report_json)
        result = classify_sanitizer(report)
        return {
            "category": result.category,
            "confidence": result.confidence,
            "signals": result.signals,
            "rationale": result.rationale,
            "target": report.get("target"),
            "overall_verdict": report.get("overall_verdict"),
            "per_sanitizer": {
                str(c.get("sanitizer")): {"state": c.get("state"), "verdict": c.get("verdict")}
                for c in (report.get("checks") or [])
            },
        }
    except Exception as e:
        return {"error": str(e)}


def read_evidence_file(uri: str, bundle_root: str) -> str:
    """Read a specific evidence file from the bundle by its URI.
    Returns up to 200 lines of the file content."""
    try:
        p = Path(bundle_root) / uri
        if not p.is_file():
            return f"[file not found: {uri}]"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[:200])
    except Exception as e:
        return f"[error: {e}]"


def list_signals(evidence_json: str) -> list[str]:
    """Extract all unique signal slugs from the evidence list."""
    try:
        evidence = json.loads(evidence_json)
        return list(dict.fromkeys(e.get("signal", "") for e in evidence if e.get("signal")))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# DSPy signature + module
# ---------------------------------------------------------------------------

_CATEGORY_DESC = "Tier-1 category, exactly one of: " + " | ".join(sorted(AUTOPSY_CATEGORIES))


def coerce_category(category: str) -> str:
    """A category outside the shared vocabulary is not a verdict.

    The field is free text, so a model can return a plausible-looking label
    nobody downstream knows how to act on. Reporting that as 'unknown' is
    honest; passing it through would put an unhandled string in the report.
    """
    cleaned = (category or "").strip()
    return cleaned if cleaned in AUTOPSY_CATEGORIES else "unknown"


class TriageDecision(dspy.Signature):
    """
    You are the Autopsy router for GPU cluster training failures on AMD hardware.
    You have structured evidence from tool runs (adapters). Use the tools to
    inspect the evidence, then emit a triage decision.

    Rules:
    - Always call list_signals first to see what signals are present.
    - If AORTA_MATRIX_REPRO and AORTA_MITIGATION_CLEAN are both present,
      category is almost certainly numeric_silent.
    - If only WATCH_NUMERIC_NAN with no matrix, confidence should be ~0.62
      and next_probe should be 'aorta sweep run'. This applies ONLY when the
      watch signal stands alone — if any DBG_* signal is also present, use the
      debugger rule below instead, because the log signature is then the weakest
      evidence in the bundle rather than the only evidence.
    - DBG_DEVICE_ASSERT means a device-side assert trapped on the GPU and a
      debugger read the stopped wave before its registers were lost, naming the
      kernel, the source line and the workgroup. DBG_NAN_TRAP means that captured
      state included a non-finite value. Together they are the hardest evidence
      an Autopsy bundle can carry — harder than a matrix, and harder than a log
      signature, which only says a bad number reached the loss without saying
      where it came from. Category is numeric_silent with confidence >= 0.9, and
      next_probe is 'none': the failure has already been caught in the act, so
      there is nothing left to reproduce. Quote the captured register values and
      the source line from the evidence excerpts in the rationale.
    - If AORTA_MATRIX_INFRA_OK (smoke only, no repro cells), next_probe is
      'aorta sweep run' (need production matrix).
    - SAN_CONSAN_RACE means ConSan observed a real device-side ordering violation
      during record/replay: category is gpu_race with confidence >= 0.9. This is
      not illegal_mem — the access is in bounds, it is just unsynchronised. Name
      the kernel and code object from the evidence excerpt in the rationale.
    - SAN_WAITCHECK_HAZARD alone is a *static* missing-s_waitcnt warning, not an
      observed failure: category is gpu_race but confidence ~0.55, and next_probe
      is 'aorta sweep run' to confirm it dynamically with ConSan.
    - SAN_NOT_CHECKED means the sanitizer backend never ran, so the run proves
      nothing: category is tooling_gap, confidence 0.0. Never read it as clean.
    - SAN_CLEAN with no other failure signal means the sanitizers passed; prefer
      unknown over inventing a failure.
    - Use scan_sanitizer on a sanitizer_report.json URI to read its verdicts.
    - Do not guess — cite specific signal slugs and evidence URIs in rationale.
    - next_probe must be exactly 'aorta sweep run' or 'none'.
    """
    evidence_json: str = dspy.InputField(desc="JSON list of adapter evidence items with signals and URIs")
    bundle_root: str = dspy.InputField(desc="Absolute path to the bundle directory")
    job_context: str = dspy.InputField(desc="job_id, node, recipe")

    category: str = dspy.OutputField(desc=_CATEGORY_DESC)
    confidence: float = dspy.OutputField(desc="0.0-1.0")
    rationale: str = dspy.OutputField(desc="One paragraph citing specific signal slugs and evidence URIs")
    next_probe: str = dspy.OutputField(desc="'aorta sweep run' or 'none'")
    next_probe_reason: str = dspy.OutputField(desc="Why this probe is needed, or empty if none")


class TriageRouter(dspy.Module):
    def __init__(self):
        ensure_configured(model="claude-sonnet-4-6", max_tokens=2048)
        self.react = dspy.ReAct(
            TriageDecision,
            tools=[classify_matrix, scan_stderr, scan_sanitizer, read_evidence_file, list_signals],
            max_iters=6,
        )

    def forward(self, evidence: list[dict[str, Any]], bundle_root: str, job_context: str) -> dspy.Prediction:
        prediction = self.react(
            evidence_json=json.dumps(evidence),
            bundle_root=bundle_root,
            job_context=job_context,
        )
        prediction.category = coerce_category(getattr(prediction, "category", ""))
        return prediction
