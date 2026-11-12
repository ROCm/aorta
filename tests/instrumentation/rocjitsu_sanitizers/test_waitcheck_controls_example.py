"""Portable example validation with inert inputs and fake AORTA processes only."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from aorta.instrumentation.rocjitsu_sanitizers.execution import ProcessResult
from aorta.instrumentation.rocjitsu_sanitizers.models import KernelCheckResult, aggregate_diagnostic_summaries
from aorta.instrumentation.rocjitsu_sanitizers.recipe import load_sanitizer_recipe, _resolve_observations
from aorta.instrumentation.rocjitsu_sanitizers.report import build_report, write_report
from aorta.instrumentation.rocjitsu_sanitizers.selection import select_kernels
from aorta.instrumentation.rocjitsu_sanitizers.waitcheck import run_waitcheck

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".agents/skills/rocm-sanitizer-triage/scripts/waitcheck_controls.py"
SPEC = importlib.util.spec_from_file_location("waitcheck_controls_example", SCRIPT)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


@pytest.fixture
def prepared(tmp_path):
    missing, corrected = tmp_path / "original-missing", tmp_path / "original-corrected"
    missing.write_bytes(b"inert missing control fixture")
    corrected.write_bytes(b"inert corrected control fixture")
    output = tmp_path / "prepared"
    args = argparse.Namespace(output=output, missing_object=missing, corrected_object=corrected,
                              kernel_name="test_control", build_id="synthetic-no-compiler",
                              missing_index=2, missing_entry=16, corrected_index=3, corrected_entry=32)
    helper.prepare(args)
    return output


@pytest.fixture
def completed(tmp_path, prepared, monkeypatch):
    # Linux platform.platform() may probe `uname -p` when its cache is cold.
    # Keep metadata deterministic so the process mock remains AORTA-only.
    monkeypatch.setattr(helper.platform, "_platform_cache", None)
    monkeypatch.setattr(helper.platform, "platform", lambda: "synthetic-no-probe-platform")
    aorta = tmp_path / "aorta"
    aorta.write_text("inert fake CLI", encoding="utf-8")
    aorta.chmod(0o755)
    bundle = tmp_path / "bundle"
    (bundle / "bin").mkdir(parents=True)
    scanner = bundle / "bin" / "rj_waitcheck"
    scanner.write_text("inert fake tool", encoding="utf-8")
    scanner.chmod(0o755)
    calls, backend_calls = [], []

    def fake_popen(argv, **kwargs):
        calls.append(argv)
        assert argv[:4] == [str(aorta), "sweep", "run", "--recipe"]
        assert kwargs["shell"] is False
        assert kwargs["env"]["ROCJITSU_PREBUILT"] == str(bundle)
        recipe = load_sanitizer_recipe(Path(argv[4]))
        worklist = select_kernels(_resolve_observations(recipe), requirement=recipe.requirement, top_n=1)
        output = Path(argv[6])
        label = Path(argv[4]).stem

        def execute(native_argv, *, timeout_seconds, **unused):
            backend_calls.append(native_argv)
            assert timeout_seconds == 120
            identity = worklist.kernels[0].identity
            count = 1 if label == "missing" else 0
            instructions = 3 if count else 4
            prefix = f"{identity.code_object}:gfx1250[{identity.code_object_index}]"
            log = (f"rj_waitcheck: {prefix}:kernel=.text+0x{identity.entry_offset:x}: "
                   f"instructions={instructions} memory-events=2 diagnostics={count}\n")
            if count:
                log += (f"{prefix}:.text+0x18: missing s_wait_kmcnt <= 0 before use of s4\n"
                        "  producer .text+0x10: s_load_b32 s4, s[0:1], 0x0\n"
                        "  consumer .text+0x18: s_mov_b32 s5, s4\n")
            return ProcessResult(tuple(native_argv), 4 if count else 0, log, "")

        check = run_waitcheck(worklist, output_dir=output / "waitcheck", binary=scanner,
                              timeout_seconds=120, max_diagnostics=64, execute=execute)
        write_report(build_report(target="gfx1250", worklist=worklist, checks=(check,)),
                     output / "sanitizer_report.json")

        class Process:
            def wait(self, timeout):
                assert timeout == 180
                return 1 if label == "mismatch" else 0
        return Process()

    monkeypatch.setattr(helper.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(helper.importlib.metadata, "version", lambda name: "test-only")
    output = tmp_path / "run"
    helper.run(argparse.Namespace(prepared=prepared, aorta=aorta, prebuilt=bundle,
                                  tool_sha256=helper.digest(scanner), output=output))
    return output, calls, backend_calls


def test_prepare_preserves_inputs_and_explicit_nonzero_scope(prepared):
    manifest = helper.check_prepared(prepared)
    assert manifest["controls"]["missing"]["index"] == 2
    assert manifest["controls"]["missing"]["entry"] == 16
    assert (prepared.parent / "original-missing").read_bytes() == b"inert missing control fixture"
    original = load_sanitizer_recipe(prepared / "missing.json")
    assert original.kernel_specs[0].entry_offset == 16
    mismatch = load_sanitizer_recipe(prepared / "mismatch.json")
    assert mismatch.kernel_specs[0].code_object_sha256 == "0" * 64
    assert mismatch.kernel_specs[0].entry_offset == 32


def test_three_cases_only_aorta_called_and_mismatch_never_scanned(completed):
    output, calls, backend_calls = completed
    assert len(calls) == 3
    assert len(backend_calls) == 2
    summary = helper.verify(output)
    assert "missing: CLI=0, native=4, state=ran, verdict=warn" in summary
    assert "mismatch: CLI=1, native=None, state=error, verdict=error" in summary
    assert not (output / "mismatch/waitcheck/waitcheck-0.log").exists()


def test_cold_platform_metadata_does_not_escape_aorta_process_mock(completed):
    output, calls, backend_calls = completed
    manifest = helper.load(output / "run.json")
    assert helper.platform._platform_cache is None
    assert manifest["platform"] == "synthetic-no-probe-platform"
    assert len(calls) == 3 and all(argv[1:4] == ["sweep", "run", "--recipe"] for argv in calls)
    assert len(backend_calls) == 2


def test_downloaded_run_verifies_without_original_runtime_paths(completed, tmp_path):
    original = completed[0]
    relocated = tmp_path / "download"
    shutil.copytree(original, relocated)
    # Remove only the test-owned run tree, so no original runtime input paths exist.
    shutil.rmtree(original)
    assert "expected result verified" in helper.verify(relocated)


@pytest.mark.parametrize("mutation", ["identity", "truncated", "wrong_verdict", "backend"])
def test_inconsistent_report_rejected(completed, mutation):
    output = completed[0]
    path = output / "corrected/sanitizer_report.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "identity":
        data["checks"][0]["kernel_results"][0]["identity"]["entry_offset"] = 0
    elif mutation == "truncated":
        data["checks"][0]["diagnostics_truncated"] = True
    elif mutation == "backend":
        data["checks"][0]["backend"]["sha256"] = "f" * 64
    else:
        data["overall_verdict"] = "warn"
        data["checks"][0]["verdict"] = "warn"
        data["checks"][0]["kernel_results"][0]["verdict"] = "warn"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        helper.verify(output)


@pytest.mark.parametrize("foreign", ["object", "target", "image"])
def test_foreign_diagnostic_identity_rejected_even_when_report_and_log_agree(completed, foreign):
    output = completed[0]
    path = output / "missing/sanitizer_report.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    check = data["checks"][0]
    identity = check["kernel_results"][0]["identity"]
    original = check["findings"][0]["message"]
    prefix = f"{identity['code_object']}:{identity['target']}[{identity['code_object_index']}]:"
    replacements = {
        "object": f"{identity['code_object']}-foreign:{identity['target']}[{identity['code_object_index']}]:",
        "target": f"{identity['code_object']}:gfx999[{identity['code_object_index']}]:",
        "image": f"{identity['code_object']}:{identity['target']}[{identity['code_object_index'] + 1}]:",
    }
    changed = original.replace(prefix, replacements[foreign], 1)
    assert changed != original
    check["findings"][0]["message"] = changed
    check["kernel_results"][0]["findings"][0]["message"] = changed
    log = output / "missing/waitcheck/waitcheck-0.log"
    log.write_text(log.read_text(encoding="utf-8").replace(original, changed), encoding="utf-8")
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        helper.verify(output)


@pytest.mark.parametrize("label,updates", [
    ("corrected", {"reported_count": 9}),
    ("missing", {"reported_count": 9, "emitted_records": 9}),
    ("missing", {"requested_limit": 1}),
    ("corrected", {"count_meaning": "exact", "details": "complete", "native_analysis": "complete"}),
    ("missing", {"count_meaning": "exact"}),
    ("mismatch", {"reported_count": 9}),
])
def test_coherently_declared_metadata_must_agree_with_raw_control_evidence(completed, label, updates):
    output = completed[0]
    path = output / label / "sanitizer_report.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    check = data["checks"][0]
    entry = check["kernel_results"][0]
    entry["diagnostic_summary"].update(updates)
    # Keep check aggregation valid so the regression reaches the control verifier.
    check["diagnostic_summary"] = aggregate_diagnostic_summaries((KernelCheckResult.from_dict(entry),)).to_dict()
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic"):
        helper.verify(output)


@pytest.mark.parametrize("nullable", [False, True])
def test_legacy_controls_allow_absent_or_null_optional_diagnostic_summary(completed, nullable):
    output = completed[0]
    for label in helper.LABELS:
        path = output / label / "sanitizer_report.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        check = data["checks"][0]
        for item in (check, check["kernel_results"][0]):
            if nullable:
                item["diagnostic_summary"] = None
            else:
                item.pop("diagnostic_summary", None)
        path.write_text(json.dumps(data), encoding="utf-8")
    assert "expected result verified" in helper.verify(output)


def test_changed_recipe_refused_before_process(prepared, monkeypatch, tmp_path):
    path = prepared / "missing.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["sanitizer_plan"]["sanitizers"] = ["asan"]
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(helper.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="recipe changed"):
        helper.run(argparse.Namespace(prepared=prepared, output=tmp_path / "run"))


@pytest.mark.parametrize("mutation", ["argv", "identity_path", "extra_log", "environment"])
def test_changed_execution_scope_or_extra_output_rejected(completed, mutation):
    output = completed[0]
    path = output / "run.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "argv":
        manifest["commands"]["corrected"]["argv"][4] = manifest["commands"]["missing"]["argv"][4]
    elif mutation == "identity_path":
        manifest["identities"]["corrected"]["code_object"] = manifest["identities"]["missing"]["code_object"]
    elif mutation == "environment":
        manifest["environment_override"]["ROCJITSU_PREBUILT"] += "-other"
    else:
        log = output / "corrected/waitcheck/waitcheck-0.log"
        with log.open("a", encoding="utf-8") as stream:
            stream.write("unexpected analysis error\n")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        helper.verify(output)


def test_control_cannot_be_used_as_aorta_executable(prepared, tmp_path, monkeypatch):
    fake_cli = tmp_path / "control-copy"
    shutil.copyfile(prepared / "inputs/missing.bin", fake_cli)
    fake_cli.chmod(0o755)
    bundle = tmp_path / "bundle"
    (bundle / "bin").mkdir(parents=True)
    scanner = bundle / "bin/rj_waitcheck"
    scanner.write_bytes(b"inert scanner")
    scanner.chmod(0o755)
    monkeypatch.setattr(helper.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not execute input"))
    with pytest.raises(ValueError, match="control input"):
        helper.run(argparse.Namespace(prepared=prepared, output=tmp_path / "run", aorta=fake_cli,
                                      prebuilt=bundle, tool_sha256=helper.digest(scanner)))


def test_timeout_captures_failure_without_clean_claim(tmp_path, monkeypatch):
    class Process:
        killed = False
        def wait(self, timeout):
            if not self.killed:
                raise subprocess.TimeoutExpired("fake", timeout)
            return -9
        def kill(self):
            self.killed = True
    process = Process()
    monkeypatch.setattr(helper.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(helper.os, "name", "nt")
    row = helper.capture(["fake-aorta"], tmp_path, "timeout", {})
    assert process.killed and row["timed_out"] and row["returncode"] == -9
    assert (tmp_path / "timeout.stderr.txt").exists()


def test_existing_and_dangling_outputs_refused(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("unchanged", encoding="utf-8")
    with pytest.raises(ValueError):
        helper.fresh(existing)
    assert (existing / "keep").read_text(encoding="utf-8") == "unchanged"
    link, target = tmp_path / "link", tmp_path / "target"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(ValueError):
        helper.fresh(link)
    assert not target.exists()
