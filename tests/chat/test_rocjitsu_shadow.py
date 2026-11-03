"""Kernel triage can record shadow facts without changing its user result."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("dspy", reason="rocjitsu shadow recording reaches the CIA tools")

from aorta.chat.config import settings
from aorta.chat.tools import cluster


def _result(job_dir: Path) -> dict:
    return {"ok": True, "job_id": "cia-demo", "job_dir": str(job_dir)}


def test_kernel_shadow_uses_the_compiled_job_binary(tmp_path, monkeypatch):
    job = tmp_path / "job"
    job.mkdir()
    binary = job / "kernel.bin"
    binary.write_bytes(b"binary")
    seen = {}
    monkeypatch.setattr(settings, "rocjitsu_shadow_enabled", True)
    monkeypatch.setattr(settings, "rocjitsu_shadow_max_kernels", 7)
    monkeypatch.setattr(cluster, "run_triage", lambda argv, stop: _result(job))
    monkeypatch.setattr(cluster, "_format_result", lambda *args, **kwargs: "primary result")
    monkeypatch.setattr(
        "aorta.rocjitsu.shadow.capture_shadow_recording",
        lambda **kwargs: seen.update(kwargs) or job / "rj_expert" / "live_recording.json",
    )

    rendered = cluster._run_triage([], "kernel", shadow_job_binary=True)

    assert rendered == "primary result"
    assert seen["code_object"] == binary
    assert seen["max_kernels"] == 7


def test_assembly_shadow_uses_the_staged_code_object(tmp_path, monkeypatch):
    job = tmp_path / "job"
    job.mkdir()
    binary = tmp_path / "kernel.hsaco"
    binary.write_bytes(b"binary")
    seen = {}
    monkeypatch.setattr(settings, "rocjitsu_shadow_enabled", True)
    monkeypatch.setattr(cluster, "run_triage", lambda argv, stop: _result(job))
    monkeypatch.setattr(cluster, "_format_result", lambda *args, **kwargs: "primary result")
    monkeypatch.setattr(
        "aorta.rocjitsu.shadow.capture_shadow_recording",
        lambda **kwargs: seen.update(kwargs) or job / "rj_expert" / "live_recording.json",
    )

    rendered = cluster._run_triage([], "assembly", shadow_object=binary)

    assert rendered == "primary result"
    assert seen["code_object"] == binary


def test_shadow_failure_does_not_fail_primary_triage(tmp_path, monkeypatch):
    job = tmp_path / "job"
    job.mkdir()
    binary = tmp_path / "kernel.hsaco"
    binary.write_bytes(b"binary")
    formatted = {}
    monkeypatch.setattr(settings, "rocjitsu_shadow_enabled", True)
    monkeypatch.setattr(cluster, "run_triage", lambda argv, stop: _result(job))
    monkeypatch.setattr(
        cluster,
        "_format_result",
        lambda result, *args, **kwargs: formatted.update(result) or "primary result",
    )

    def fail(**_kwargs):
        raise RuntimeError("recorder unavailable")

    monkeypatch.setattr("aorta.rocjitsu.shadow.capture_shadow_recording", fail)

    assert cluster._run_triage([], "assembly", shadow_object=binary) == "primary result"
    assert formatted["rocjitsu_shadow"]["status"] == "error"


def test_shadow_is_inert_by_default(tmp_path, monkeypatch):
    job = tmp_path / "job"
    job.mkdir()
    binary = tmp_path / "kernel.hsaco"
    binary.write_bytes(b"binary")
    monkeypatch.setattr(settings, "rocjitsu_shadow_enabled", False)
    monkeypatch.setattr(cluster, "run_triage", lambda argv, stop: _result(job))
    monkeypatch.setattr(cluster, "_format_result", lambda *args, **kwargs: "primary result")

    assert cluster._run_triage([], "assembly", shadow_object=binary) == "primary result"
    assert not (job / "rj_expert").exists()
