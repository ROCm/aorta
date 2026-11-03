"""Live recorder contracts never infer facts from configuration defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.instrumentation.rocjitsu_sanitizers import KernelCodeObject, KernelInventory
from aorta.rocjitsu import recording


def _object(path: Path, processor: str = "gfx950", kernels=("kernel_a",)) -> KernelInventory:
    return KernelInventory(
        path=str(path),
        code_objects=(
            KernelCodeObject(
                triple=f"amdgcn-amd-amdhsa--{processor}",
                processor=processor,
                features=(),
                asan_instrumented=False,
                kernels=tuple(kernels),
            ),
        ),
    )


def test_observed_architecture_outranks_the_requested_default(tmp_path, monkeypatch):
    binary = tmp_path / "kernel.hsaco"
    binary.write_bytes(b"object")
    monkeypatch.setattr(recording, "kernel_inventory", lambda path: _object(path))

    facts = recording.inspect_code_object(binary, requested_arch="gfx942")

    assert facts.applicable is True
    assert facts.requested_arch == "gfx942"
    assert facts.observed_arch == "gfx950"
    assert facts.kernels == ("kernel_a",)


def test_unsupported_architecture_is_not_applicable(tmp_path, monkeypatch):
    binary = tmp_path / "kernel.hsaco"
    binary.write_bytes(b"object")
    monkeypatch.setattr(
        recording,
        "kernel_inventory",
        lambda path: _object(path, processor="gfx942"),
    )

    facts = recording.inspect_code_object(binary, requested_arch="gfx950")

    assert facts.applicable is False
    assert facts.reason == "unsupported_architecture:gfx942"


def test_kernel_limit_is_a_fact_not_a_timeout(tmp_path, monkeypatch):
    binary = tmp_path / "kernel.hsaco"
    binary.write_bytes(b"object")
    monkeypatch.setattr(
        recording,
        "kernel_inventory",
        lambda path: _object(path, kernels=("a", "b", "c")),
    )

    facts = recording.inspect_code_object(binary, max_kernels=2)

    assert facts.applicable is False
    assert facts.reason == "kernel_limit_exceeded:3>2"


def test_an_unchecked_output_cannot_pass() -> None:
    with pytest.raises(ValueError, match="cannot pass"):
        recording.OutputCheck(recording.OutputCheckState.PASS)

    unavailable = recording.OutputCheck(recording.OutputCheckState.UNAVAILABLE)
    assert recording.render_output_check(unavailable) == "no output check"


def test_recording_is_versioned_and_written_atomically(tmp_path) -> None:
    facts = recording.CodeObjectFacts(
        path="/kernel.hsaco",
        sha256="a" * 64,
        requested_arch="gfx950",
        observed_arch="gfx950",
        kernels=("kernel_a",),
        applicable=True,
    )
    observation = recording.OperationObservation(
        operation="waitcheck",
        arguments={"kernel": "kernel_a"},
        machine_result=[],
        rendered_result="waitcheck: no diagnostics.",
        output_check=recording.OutputCheck(recording.OutputCheckState.NOT_RUN),
        elapsed_seconds=0.1,
    )
    path = tmp_path / "recording.json"

    recording.write_recording(path, recording.LiveRecording(facts, (observation,)))

    document = json.loads(path.read_text())
    assert document["schema"] == recording.RECORDING_SCHEMA
    assert document["rocjitsu_commit"] == recording.PINNED_ROCJITSU_COMMIT
    assert document["operations"][0]["output_check"]["state"] == "not_run"
    assert not list(tmp_path.glob(".recording.json.*"))
