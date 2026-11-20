"""Preparation, selection and directory coverage with synthetic native inventory."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aorta.instrumentation.rocjitsu_sanitizers.models import (
    CheckResult,
    ExecutionState,
    Finding,
    FindingSeverity,
    KernelCheckResult,
    KernelIdentity,
    KernelObservation,
    KernelWorklist,
    SelectionRequirement,
    Verdict,
)
from aorta.instrumentation.rocjitsu_sanitizers.recipe import (
    execute_sanitizer_run,
    load_sanitizer_recipe,
)
from aorta.instrumentation.rocjitsu_sanitizers.report import build_report, read_report, write_report

SCRIPTS = Path(__file__).resolve().parents[3] / ".agents/skills/rocm-sanitizer-triage/scripts"


@pytest.fixture
def prep(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import waitcheck_prepare

    monkeypatch.chdir(tmp_path)
    return waitcheck_prepare


def inventory(obj, *, other="other"):
    base = {"schema": "rj-waitcheck-corpus-inventory-v1", "input": str(obj), "target": "gfx950"}
    rows = []
    for image, name, offset in [(0, other, 0), (1, "want", 0x120)]:
        rows.append(dict(base, kind="code-object", code_object_index=image, kernel_count=1))
        rows.append(
            dict(
                base,
                kind="kernel",
                code_object_index=image,
                kernel_name=name,
                kernel_entry=offset,
                kernel_entry_hex=hex(offset),
                descriptor_vaddr=4096,
            )
        )
    return rows


def fake_native(monkeypatch, *, mutate=lambda rows: None, after=None, error=None):
    calls = []

    def run(command, *, stdin, stdout, stderr, timeout):
        assert stdin == subprocess.DEVNULL and timeout == 15
        assert command[0] == str(Path(sys.executable).resolve())
        obj = Path(command[1])
        calls.append(command)
        if command[2:] == ["--list-code-objects"]:
            stdout.write(b"gfx942: 0\ngfx950: 2\n")
        else:
            assert command[2:] == ["--target", "gfx950", "--list-kernels"]
            if error == "timeout":
                raise subprocess.TimeoutExpired(command, 15)
            if error == "exit":
                stderr.write(b"listing refused")
                return subprocess.CompletedProcess(command, 2)
            rows = inventory(obj)
            mutate(rows)
            stdout.write("\n".join(json.dumps(row) for row in rows).encode())
            if after:
                after(obj)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def prepare_args(*args):
    return [
        "prepare",
        "--waitcheck",
        sys.executable,
        "--target",
        "gfx950",
        "--output",
        "plan",
        *args,
    ]


@pytest.mark.parametrize("name,index,offset", [("want", 1, 0x120), ("other", 0, 0)])
def test_single_from_other_cwd(prep, tmp_path, monkeypatch, name, index, offset):
    obj = tmp_path / "kernel with spaces.hsaco"
    obj.write_bytes(b"a compiled fixture")
    calls = fake_native(monkeypatch)
    args = prepare_args(
        "--object",
        obj.name,
        "--kernel",
        name,
        "--waitcheck-sha256",
        prep.digest(Path(sys.executable).resolve()),
    )
    assert prep.main(args) == 0
    assert len(calls) == 2
    plan = tmp_path / "plan"
    recipe = load_sanitizer_recipe(plan / "entry-0001/recipe.yaml")
    (spec,) = recipe.kernel_specs
    assert (spec.name, spec.code_object, spec.code_object_index, spec.entry_offset) == (
        name,
        str(obj),
        index,
        offset,
    )
    assert spec.code_object_sha256 == prep.digest(obj)
    assert recipe.target == "gfx950" and recipe.sanitizers == ("waitcheck",)
    assert recipe.waitcheck_max_diagnostics == 64 and recipe.timeout_seconds == 120
    original = (plan / "manifest.json").read_bytes()
    assert prep.main(args) == 1
    assert original == (plan / "manifest.json").read_bytes() and len(calls) == 2


def test_explicit_path_preserved_with_identical_copy_in_cwd(prep, tmp_path, monkeypatch):
    selected_dir = tmp_path / "selected"
    selected_dir.mkdir()
    selected = selected_dir / "kernel.bin"
    selected.write_bytes(b"synthetic compiled entry")
    previous_dir = tmp_path / "previous"
    previous_dir.mkdir()
    duplicate = previous_dir / "kernel.bin"
    duplicate.write_bytes(selected.read_bytes())
    assert prep.digest(selected) == prep.digest(duplicate)
    monkeypatch.chdir(previous_dir)
    calls = fake_native(monkeypatch)

    assert prep.main(prepare_args("--object", str(selected), "--kernel", "want")) == 0

    plan = previous_dir / "plan"
    recipe_path = plan / "entry-0001/recipe.yaml"
    recipe = load_sanitizer_recipe(recipe_path)
    assert all(command[1] == str(selected) for command in calls)
    assert recipe.kernel_specs[0].code_object == str(selected)
    manifest = json.loads((plan / "manifest.json").read_text())
    assert manifest["scope"] == str(selected)
    assert manifest["entries"][0]["identity"]["code_object"] == str(selected)

    report_path = execute_sanitizer_run(
        recipe_path, output_dir=plan / "entry-0001/dry-run", dry_run=True
    )
    report = read_report(report_path)
    assert report.worklist.kernels[0].identity.code_object == str(selected)
    assert report.checks[0].reason == "dry_run"
    assert len(calls) == 2


@pytest.mark.parametrize(
    "change",
    [
        lambda rows: rows[-1].update(kernel_name="other"),  # no match
        lambda rows: rows[1].update(kernel_name="want"),  # two matches
        lambda rows: rows[-1].pop("kernel_entry"),  # descriptor isn't an entry
        lambda rows: rows[-1].update(kernel_entry_hex="0x999"),
        lambda rows: rows[-1].update(input="/wrong-object"),
        lambda rows: rows[-1].update(target="gfx942"),
        lambda rows: rows[-1].update(code_object_index=False),
        lambda rows: rows[0].update(kernel_count=2),
        lambda rows: rows.append(rows[1].copy()),
        lambda rows: rows.pop(0),
        lambda rows: rows[-1].update(schema="unsupported"),
    ],
)
def test_uncertain_selection_leaves_receipts_without_recipes(prep, tmp_path, monkeypatch, change):
    obj = tmp_path / "k.hsaco"
    obj.write_bytes(b"object")
    fake_native(monkeypatch, mutate=change)
    assert prep.main(prepare_args("--object", str(obj), "--kernel", "want")) == 1
    assert (tmp_path / "plan/inventory-0001/kernels.stdout").is_file()
    assert not (tmp_path / "plan/manifest.json").exists()
    assert not list((tmp_path / "plan").rglob("recipe.yaml"))


def test_overlong_offset_without_hex_never_becomes_whole_object(prep, tmp_path, monkeypatch):
    def change(rows):
        rows[-1]["kernel_entry"] = "9" * 5000
        rows[-1].pop("kernel_entry_hex")

    obj = tmp_path / "k.hsaco"
    obj.write_bytes(b"fixture")
    fake_native(monkeypatch, mutate=change)
    assert prep.main(prepare_args("--object", str(obj), "--kernel", "want")) == 1
    assert not (tmp_path / "plan/manifest.json").exists()
    assert not list((tmp_path / "plan").rglob("recipe.yaml"))


def test_directory_with_zero_listed_entries_cannot_finalize_empty_plan(prep, tmp_path, monkeypatch):
    def empty_images(rows):
        rows[:] = [dict(row, kernel_count=0) for row in rows if row["kind"] == "code-object"]

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "k.hsaco").write_bytes(b"fixture")
    fake_native(monkeypatch, mutate=empty_images)
    assert prep.main(prepare_args("--directory", str(corpus))) == 1
    assert not (tmp_path / "plan/manifest.json").exists()
    assert not list((tmp_path / "plan").rglob("recipe.yaml"))


@pytest.mark.parametrize("failure", ["object_changed", "timeout", "exit"])
def test_failed_listings_stop_and_preserve_status(prep, tmp_path, monkeypatch, failure):
    obj = tmp_path / "test.hsaco"
    obj.write_bytes(b"object")
    change = (lambda p: p.write_bytes(b"changed")) if failure == "object_changed" else None
    fake_native(monkeypatch, error=failure, after=change)
    assert prep.main(prepare_args("--object", str(obj), "--kernel", "want")) == 1
    assert (tmp_path / "plan/inventory-0001/kernels.status.json").exists()
    assert not (tmp_path / "plan/manifest.json").exists()


def save_report(folder, identity, verdict):
    identity = KernelIdentity.from_dict(identity)
    worklist = KernelWorklist(
        requirement=SelectionRequirement.TOP_DISPATCH_COUNT,
        top_n=1,
        kernels=(KernelObservation(identity=identity, sources=("fixture",)),),
    )
    findings = ()
    if verdict == Verdict.WARN:
        findings = (
            Finding(
                sanitizer="waitcheck",
                severity=FindingSeverity.WARNING,
                code="wait_hazard",
                message="synthetic wait finding",
            ),
        )
    result = KernelCheckResult(
        identity=identity,
        state=ExecutionState.RAN,
        verdict=verdict,
        findings=findings,
        returncode=4 if findings else 0,
    )
    check = CheckResult(
        sanitizer="waitcheck",
        state=ExecutionState.RAN,
        verdict=verdict,
        findings=findings,
        kernel_results=(result,),
    )
    folder.mkdir()
    write_report(
        build_report(target=identity.target, worklist=worklist, checks=(check,)),
        folder / "sanitizer_report.json",
    )


def test_directory_coverage_keeps_collisions_missing_and_mismatched_reports(
    prep, tmp_path, monkeypatch
):
    root = tmp_path / "corpus"
    for folder in ["a", "b"]:
        (root / folder).mkdir(parents=True)
        (root / folder / "same.hsaco").write_bytes(folder.encode())
    (root / "not-in-scope.py").write_text("# source")
    fake_native(monkeypatch)
    assert prep.main(prepare_args("--directory", str(root))) == 0
    plan = tmp_path / "plan"
    manifest = json.loads((plan / "manifest.json").read_text())
    rows = manifest["entries"]
    assert [row["identity"]["name"] for row in rows] == ["other", "want", "other", "want"]
    assert len({row["id"] for row in rows}) == 4
    assert len({row["identity"]["code_object_sha256"] for row in rows}) == 2
    for i, verdict in [(0, Verdict.WARN), (1, Verdict.PASS), (2, Verdict.PASS)]:
        # The second object's first report is a misplaced first-object report.
        identity = rows[0 if i == 2 else i]["identity"]
        save_report(plan / rows[i]["id"] / "scan", identity, verdict)
    args = ["aggregate", "--plan", str(plan), "--output", "coverage.json"]
    assert prep.main(args) == 0
    coverage = json.loads((tmp_path / "coverage.json").read_text())
    assert coverage["requested_entries"] == 4
    assert coverage["outcome_counts"] == {
        "reported_warn": 1,
        "reported_pass": 1,
        "identity_mismatch": 1,
        "missing_report": 1,
    }
    assert coverage["entries"][0]["native_exit"] == 4
    assert coverage["entries"][0]["raw_log"] is None
    assert (
        coverage["entries"][2]["expected_identity"] != coverage["entries"][2]["reported_identity"]
    )
    assert prep.main(args) == 1


def test_wrong_tool_pin_stops_before_native(prep, tmp_path, monkeypatch):
    (tmp_path / "k.hsaco").write_bytes(b"object")
    calls = fake_native(monkeypatch)
    assert (
        prep.main(
            prepare_args("--object", "k.hsaco", "--kernel", "want", "--waitcheck-sha256", "f" * 64)
        )
        == 1
    )
    assert calls == [] and not (tmp_path / "plan").exists()
