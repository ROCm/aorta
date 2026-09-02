from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aorta.cia.autopsy.adapters.base import AdapterArtifact, BundleContext

MITIGATION_ISOLATION = frozenset({"tf32_off", "deterministic"})
NAN_HINTS = re.compile(r"nan|residual|non.?finite|inf", re.I)


@dataclass(frozen=True)
class MatrixClassification:
    category: str
    confidence: float
    rationale: str
    signals: list[str]


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

        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
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

        if is_repro and not NAN_HINTS.search(hints) and not NAN_HINTS.search(name):
            # Numeric silent often lacks explicit hint text; repro cell name is enough.
            pass

    if repro_failures and clean_mitigations:
        repro_names = ", ".join(c["name"] for c in repro_failures[:3])
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

    if repro_failures and not clean_mitigations:
        repro_names = ", ".join(c["name"] for c in repro_failures[:3])
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
