#!/usr/bin/env python3
"""Validate the two env-probe files required for a Buck workload handoff."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    severity: str
    message: str


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _nonempty(value: object) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def validate_pair(
    client: dict[str, object],
    workload: dict[str, object],
) -> list[Finding]:
    """Return actionable errors/warnings for a Buck client/workload pair."""
    findings: list[Finding] = []

    try:
        schema = tuple(int(part) for part in str(client["schema_version"]).split("."))
    except (KeyError, TypeError, ValueError):
        schema = ()
    if schema < (1, 13):
        findings.append(Finding("ERROR", "client schema must be 1.13 or newer"))

    if _object(client.get("build_system")).get("kind") != "buck2":
        findings.append(Finding("ERROR", "client snapshot did not detect Buck2"))

    invocation = _object(client.get("buck_invocation"))
    if invocation.get("status") != "success":
        findings.append(Finding("ERROR", "Buck dependency lookup did not succeed"))
    if invocation.get("context_source") not in {
        "default_confirmed",
        "explicit",
    }:
        findings.append(Finding("ERROR", "Buck invocation context was not confirmed"))
    if not _nonempty(invocation.get("context_fingerprint")):
        findings.append(Finding("ERROR", "Buck invocation fingerprint is missing"))
    if not _nonempty(client.get("library_introspection")):
        findings.append(Finding("ERROR", "no recognized libraries were found in the Buck graph"))

    pytorch_build = _object(workload.get("pytorch_build"))
    runtime_rocm = (
        _object(workload.get("rocm")).get("version")
        or _object(workload.get("hip")).get("version")
        or pytorch_build.get("hip_version")
    )
    required_workload_fields = {
        "python_version": workload.get("python_version"),
        "pytorch_version": workload.get("pytorch_version"),
        "pytorch_build.git_commit": pytorch_build.get("git_commit"),
        "ROCm/HIP identity": runtime_rocm,
        "gpu_arch.gfx_targets": _object(workload.get("gpu_arch")).get("gfx_targets"),
    }
    for name, value in required_workload_fields.items():
        if not _nonempty(value):
            findings.append(Finding("ERROR", f"workload snapshot is missing {name}"))

    probe_invocation = _object(workload.get("execution_context")).get("probe_invocation")
    if probe_invocation not in {"buck2_run", "buck2_action"}:
        findings.append(Finding("ERROR", "workload snapshot was not labeled as Buck-launched"))

    for label, document in (("client", client), ("workload", workload)):
        reasons = document.get("partial_reasons")
        if isinstance(reasons, list):
            for reason in reasons:
                findings.append(Finding("WARNING", f"{label}: {reason}"))

    return findings


def _load(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{path} is not a JSON object")
    return {str(key): value for key, value in document.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Buck client and workload env-probe snapshots."
    )
    parser.add_argument("client", type=Path, help="env.buck-client.json")
    parser.add_argument("workload", type=Path, help="env.workload.json")
    args = parser.parse_args()

    try:
        findings = validate_pair(_load(args.client), _load(args.workload))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    for finding in findings:
        print(f"{finding.severity}: {finding.message}")
    if any(finding.severity == "ERROR" for finding in findings):
        return 1
    print("PASS: Buck dependency and workload runtime snapshots are usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
