from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aorta.cia.autopsy.adapters.base import AdapterArtifact, BundleContext

MITIGATION_ISOLATION = frozenset({"tf32_off", "deterministic"})
#: Complete numeric-failure tokens, not substrings. The former ``inf`` branch
#: matched ``inference`` and ``infrastructure``, so a failed inference cell
#: beside a clean mitigation became numeric corruption at 0.88 confidence even
#: when the cell said only that its infrastructure failed. ``residual`` was
#: equally broad: it names a workload family, not a value that went non-finite.
NUMERIC_HINTS = re.compile(
    r"\b(?:nan|inf(?:inity)?|non[- ]?finite)\b",
    re.I,
)

#: Failures that are plainly not numeric corruption. A cell matching one of
#: these has said what went wrong, and it was not the arithmetic.
OOM_HINTS = re.compile(r"out of memory|\boom\b|cuda error: out of memory|hip.*out of memory", re.I)
LAUNCH_HINTS = re.compile(
    r"launch fail|no such file|command not found|module.*not found|"
    r"importerror|modulenotfounderror|permission denied|exit code 127",
    re.I,
)


def _numeric_evidence(cell: dict[str, Any]) -> bool:
    """Whether this cell says its failure was numeric.

    Required before ``numeric_silent`` because "the cell failed" was the whole
    test before: any non-zero exit satisfied it, so an OOM, a missing module
    and a killed process all came back as silent numeric corruption at 0.88.
    The check that was meant to catch this was written and then not used --
    ``if is_repro and not NUMERIC_HINTS.search(...): pass`` -- so it read like a
    guard and did nothing.

    Hints or the cell's own name, plus the structured fields a matrix carries
    when the harness recorded what it saw.
    """
    hints = " ".join(cell.get("failure_hints") or [])
    if NUMERIC_HINTS.search(hints) or NUMERIC_HINTS.search(cell.get("name", "")):
        return True
    counts = cell.get("exit_status_counts") or {}
    if int(counts.get("numeric_nan") or 0) > 0:
        return True
    return bool(cell.get("nan_detected") or cell.get("non_finite_count"))


def _named_failure(cells: list[dict[str, Any]]) -> tuple[str, str] | None:
    """A category these cells name outright, and the evidence for it.

    Only for failures that said what they were. Anything else stays unknown
    rather than being given the nearest-looking label.
    """
    blob = " ".join(
        " ".join(c.get("failure_hints") or []) + " " + str(c.get("name", ""))
        for c in cells
    )
    if OOM_HINTS.search(blob):
        return "oom_fragment", "the cells report running out of memory"
    if LAUNCH_HINTS.search(blob):
        return "launch_error", "the cells report the workload failing to start"
    return None


@dataclass(frozen=True)
class MatrixClassification:
    category: str
    confidence: float
    rationale: str
    signals: list[str]


log = logging.getLogger(__name__)


def load_matrix(path: Path | None) -> dict[str, Any] | None:
    """The matrix at *path*, or None when it cannot be read as one.

    Autopsy runs on a bundle assembled from a job that has just failed, and a
    sweep interrupted mid-write leaves a matrix.json that is half a document.
    Reading it is therefore expected to fail sometimes, and one artifact that
    cannot be parsed must not take the whole verdict with it -- the sanitizer
    path has always degraded to a tooling gap, and this is the same policy in
    the same place for both readers of this file.
    """
    if path is None or not path.is_file():
        return None
    try:
        matrix = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Could not read the Aorta matrix at %s: %s", path, exc)
        return None
    return matrix if isinstance(matrix, dict) else None


class AortaMatrixAdapter:
    """Read Aorta triage matrix.json and infer numeric_silent + mechanism."""

    adapter_id = "aorta_matrix"

    def collect(self, ctx: BundleContext) -> AdapterArtifact:
        matrix_path = ctx.path("aorta_matrix")
        matrix_md_path = ctx.path("aorta_matrix_md")
        if matrix_path is None or not matrix_path.is_file():
            return AdapterArtifact(
                adapter=self.adapter_id,
                tooling_gaps=[
                    {
                        "description": "No aorta_matrix path in bundle manifest.",
                        "missing_signal": "aorta.matrix.json",
                        "suggested_tool": "aorta sweep run",
                    }
                ],
            )

        matrix = load_matrix(matrix_path)
        if matrix is None:
            # Same shape as the no-matrix branch above: an unreadable artifact
            # and an absent one are both "this did not tell us anything", and
            # neither is a reason to abandon the other adapters.
            return AdapterArtifact(
                adapter=self.adapter_id,
                tooling_gaps=[
                    {
                        "description": f"Aorta matrix at {matrix_path.name} could not be read.",
                        "missing_signal": "aorta.matrix.json",
                        "suggested_tool": "aorta sweep run",
                    }
                ],
            )
        classification = classify_matrix(matrix)
        evidence = build_evidence(matrix_path, matrix_md_path, matrix, classification)
        if "AORTA_MATRIX_INFRA_OK" in classification.signals:
            evidence.append(
                {
                    "uri": (
                        f"aorta/{matrix_path.name}"
                        if matrix_path.parent.name == "aorta"
                        else matrix_path.name
                    ),
                    "line_start": 1,
                    "line_end": 1,
                    "excerpt": f"all {len(matrix.get('cells') or [])} cells passed (smoke/infra)",
                    "adapter": "aorta_matrix",
                    "signal": "AORTA_MATRIX_INFRA_OK",
                }
            )

        artifact = AdapterArtifact(
            adapter=self.adapter_id,
            evidence=evidence,
            signals=classification.signals,
            summary=summarize_matrix(matrix),
        )

        if classification.category == "unknown":
            steps = matrix.get("steps_per_trial")
            if steps is not None and steps <= 10:
                artifact.next_probes.append(
                    {
                        "tool": "aorta sweep run",
                        "reason": (
                            "Smoke matrix passed cleanly; run production Residual-NaN "
                            "recipe (16×1000 steps) to surface NaN rates."
                        ),
                        "overhead_class": "medium",
                    }
                )
                artifact.tooling_gaps.append(
                    {
                        "description": (
                            "Matrix shows no failing repro cells — likely smoke/infra "
                            "validation only."
                        ),
                        "missing_signal": "AORTA_MATRIX_REPRO",
                        "suggested_tool": "Residual-NaN-Repro.yaml",
                    }
                )
        return artifact


def classify_matrix(matrix: dict[str, Any]) -> MatrixClassification:
    cells = matrix.get("cells") or []
    repro_failures = []
    clean_mitigations = []

    for cell in cells:
        name = cell.get("name", "")
        mitigations = set(cell.get("mitigations") or [])
        failure_rate = float(cell.get("failure_rate") or 0.0)
        failed = int(cell.get("failed_count") or 0)
        hints = " ".join(cell.get("failure_hints") or [])
        exit_counts = cell.get("exit_status_counts") or {}
        workload_failed = int(exit_counts.get("workload_failed") or 0)

        is_repro = failure_rate > 0 or failed > 0 or workload_failed > 0
        if is_repro:
            repro_failures.append(cell)
        if failure_rate == 0 and failed == 0 and mitigations & MITIGATION_ISOLATION:
            clean_mitigations.append(cell)

    # numeric_silent is a claim about arithmetic, so it needs a cell that said
    # something about arithmetic. Without one the pattern below -- repro cells
    # failing, mitigation cells clean -- is equally consistent with a repro
    # configuration that runs out of memory and a mitigation configuration that
    # does not.
    numeric = [c for c in repro_failures if _numeric_evidence(c)]

    if repro_failures and not numeric:
        named = _named_failure(repro_failures)
        repro_names = ", ".join(c["name"] for c in repro_failures[:3])
        if named:
            category, because = named
            return MatrixClassification(
                category=category,
                confidence=0.6,
                rationale=(
                    f"Aorta matrix reports failures in repro cells "
                    f"({repro_names}), and {because}. This is not a numeric "
                    "signature."
                ),
                signals=["AORTA_MATRIX_REPRO"],
            )
        return MatrixClassification(
            category="unknown",
            confidence=0.3,
            rationale=(
                f"Aorta matrix reports failures in repro cells ({repro_names}), "
                "but nothing in them says what failed: no NaN or non-finite "
                "signature, and no recognised launch or memory error. A "
                "non-zero exit on its own does not identify a cause."
            ),
            signals=["AORTA_MATRIX_REPRO"],
        )

    if numeric and clean_mitigations:
        repro_names = ", ".join(c["name"] for c in numeric[:3])
        clean_names = ", ".join(c["name"] for c in clean_mitigations[:3])
        mit = sorted(
            m for c in clean_mitigations for m in (c.get("mitigations") or []) if m in MITIGATION_ISOLATION
        )
        mechanism = mit[0] if mit else "mitigation"
        return MatrixClassification(
            category="numeric_silent",
            confidence=0.88,
            rationale=(
                f"Aorta matrix shows repro cells failing ({repro_names}) while "
                f"mitigation cells stay clean ({clean_names}) — consistent with "
                f"silent numeric corruption suppressed by {mechanism}."
            ),
            signals=["AORTA_MATRIX_REPRO", "AORTA_MITIGATION_CLEAN"],
        )

    if numeric and not clean_mitigations:
        repro_names = ", ".join(c["name"] for c in numeric[:3])
        return MatrixClassification(
            category="numeric_silent",
            confidence=0.72,
            rationale=(
                f"Aorta matrix reports failures in repro cells ({repro_names}) "
                "without a clean mitigation column — numeric_silent likely, "
                "mechanism not yet isolated."
            ),
            signals=["AORTA_MATRIX_REPRO"],
        )

    all_ok = all(
        float(c.get("failure_rate") or 0) == 0 and int(c.get("failed_count") or 0) == 0
        for c in cells
    )
    if all_ok and cells:
        return MatrixClassification(
            category="unknown",
            confidence=0.35,
            rationale=(
                "All matrix cells passed — infra/smoke validation only; "
                "no numeric failure signature at this step count."
            ),
            signals=["AORTA_MATRIX_INFRA_OK"],
        )

    return MatrixClassification(
        category="unknown",
        confidence=0.2,
        rationale="Matrix present but no repro/mitigation pattern matched.",
        signals=[],
    )


def summarize_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    cells = matrix.get("cells") or []
    return {
        "ticket": matrix.get("ticket"),
        "workload": matrix.get("workload"),
        "run_timestamp": matrix.get("run_timestamp"),
        "steps_per_trial": matrix.get("steps_per_trial"),
        "trials_per_cell": matrix.get("trials_per_cell"),
        "cell_count": len(cells),
        "cells": [
            {
                "name": c.get("name"),
                "failure_rate": c.get("failure_rate"),
                "failed_count": c.get("failed_count"),
                "mitigations": c.get("mitigations"),
            }
            for c in cells
        ],
    }


def build_evidence(
    matrix_path: Path,
    matrix_md_path: Path | None,
    matrix: dict[str, Any],
    classification: MatrixClassification,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    bundle_rel = (
        f"aorta/{matrix_path.name}"
        if matrix_path.parent.name == "aorta"
        else matrix_path.name
    )

    rendered = json.dumps(matrix, indent=2)
    lines = rendered.splitlines()

    for cell in matrix.get("cells") or []:
        name = cell.get("name")
        if not name:
            continue
        fr = float(cell.get("failure_rate") or 0)
        failed = int(cell.get("failed_count") or 0)
        mit = set(cell.get("mitigations") or [])
        is_repro = fr > 0 or failed > 0
        is_clean_mitigation = fr == 0 and failed == 0 and bool(mit & MITIGATION_ISOLATION)
        if not is_repro and not is_clean_mitigation:
            continue
        signal = "AORTA_MITIGATION_CLEAN" if is_clean_mitigation else "AORTA_MATRIX_REPRO"
        needle = f'"name": "{name}"'
        line_start = next((i + 1 for i, line in enumerate(lines) if needle in line), 1)
        line_end = min(line_start + 8, len(lines))
        evidence.append(
            {
                "uri": bundle_rel,
                "line_start": line_start,
                "line_end": line_end,
                "excerpt": (
                    f"{name}: failure_rate={fr}, failed={failed}, mitigations={sorted(mit)}"
                )[:500],
                "adapter": "aorta_matrix",
                "signal": signal,
            }
        )

    if matrix_md_path and matrix_md_path.is_file():
        for i, line in enumerate(matrix_md_path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.startswith("| bf16_"):
                evidence.append(
                    {
                        "uri": "aorta/matrix.md",
                        "line_start": i,
                        "line_end": i,
                        "excerpt": line.strip()[:500],
                        "adapter": "aorta_matrix",
                        "signal": "AORTA_MATRIX_TABLE",
                    }
                )

    if not evidence:
        evidence.append(
            {
                "uri": bundle_rel,
                "line_start": 1,
                "line_end": min(5, len(lines)),
                "excerpt": (
                    f"ticket={matrix.get('ticket')} cells={len(matrix.get('cells') or [])}"
                ),
                "adapter": "aorta_matrix",
                "signal": classification.signals[0] if classification.signals else "AORTA_MATRIX",
            }
        )
    return evidence
