"""Tests for the customer-facing Buck snapshot-pair validator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "validate_buck_env_pair.py"
_SPEC = importlib.util.spec_from_file_location(
    "validate_buck_env_pair",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
validator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validator
_SPEC.loader.exec_module(validator)


def _valid_client() -> dict[str, object]:
    return {
        "schema_version": "1.13",
        "build_system": {"kind": "buck2"},
        "buck_invocation": {
            "status": "success",
            "context_source": "explicit",
            "context_fingerprint": "sha256:" + "a" * 64,
        },
        "library_introspection": [{"name": "pytorch"}],
        "partial_reasons": [],
    }


def _valid_workload() -> dict[str, object]:
    return {
        "python_version": "3.12.0",
        "pytorch_version": "2.10.0",
        "pytorch_build": {
            "git_commit": "a" * 40,
            "hip_version": "7.0",
        },
        "gpu_arch": {"gfx_targets": ["gfx950"]},
        "execution_context": {"probe_invocation": "buck2_run"},
        "partial_reasons": [],
    }


def test_valid_pair_has_no_errors():
    findings = validator.validate_pair(_valid_client(), _valid_workload())
    assert not [finding for finding in findings if finding.severity == "ERROR"]


def test_missing_client_context_and_workload_torch_are_errors():
    client = _valid_client()
    client["buck_invocation"] = {
        "status": "success",
        "context_source": "unspecified",
        "context_fingerprint": None,
    }
    workload = _valid_workload()
    workload["pytorch_version"] = None

    messages = [
        finding.message
        for finding in validator.validate_pair(client, workload)
        if finding.severity == "ERROR"
    ]
    assert any("context was not confirmed" in message for message in messages)
    assert any("fingerprint is missing" in message for message in messages)
    assert any("pytorch_version" in message for message in messages)


def test_partial_reasons_are_warnings_not_errors():
    client = _valid_client()
    client["partial_reasons"] = ["system_health: optional tool unavailable"]
    findings = validator.validate_pair(client, _valid_workload())
    assert any(finding.severity == "WARNING" for finding in findings)
    assert not [finding for finding in findings if finding.severity == "ERROR"]


def test_pytorch_hip_version_is_valid_runtime_rocm_identity():
    workload = _valid_workload()
    workload["rocm"] = {"version": None}
    workload["hip"] = {"version": None}
    findings = validator.validate_pair(_valid_client(), workload)
    assert not [finding for finding in findings if finding.severity == "ERROR"]
