"""Integration test for the nightly-eval harness (scripts/ci/nightly_eval.py).

Monkeypatches the GPU / subprocess boundary (run_entry, gpu_count) so the
end-to-end evaluate() logic can be exercised on the CPU gate with a synthetic
matrix.json -- no torch / aorta / GPU required.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / "ci" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


nightly_eval = _load("nightly_eval")


def _write_matrix(path: Path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    return path


def test_evaluate_record_pass_fail_and_skip(tmp_path, monkeypatch):
    matrix_doc = {
        "entries": [
            {"name": "gpu_smoke", "recipe": "r1.yaml"},
            {"name": "inference_offline", "recipe": "r2.yaml"},
            {"name": "race", "recipe": "r3.yaml", "nproc": 2, "min_gpus": 2},
        ]
    }
    baselines = {
        "baselines": {
            # inference has a blessed baseline it satisfies -> pass
            "inference_offline::baseline-local": {"passed": True, "step_time_ms": {"max": 10.0}},
        }
    }

    # Pretend the runner has exactly 1 GPU (so the 2-GPU race entry is skipped).
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 1)

    def fake_run_entry(entry, out_dir):
        name = entry["name"]
        mpath = out_dir / name / "matrix.json"
        if name == "gpu_smoke":
            _write_matrix(mpath, [{"name": "baseline-local", "error": None,
                                   "passed_count": 1, "failed_count": 0, "error_count": 0,
                                   "mean_step_time_ms": 2.0, "metrics_summary": {}}])
        elif name == "inference_offline":
            _write_matrix(mpath, [{"name": "baseline-local", "error": None,
                                   "passed_count": 1, "failed_count": 0, "error_count": 0,
                                   "mean_step_time_ms": 4.0, "metrics_summary": {}}])
        return 0, mpath, False

    monkeypatch.setattr(nightly_eval, "run_entry", fake_run_entry)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {"amd_aorta_version": "x"})

    doc = nightly_eval.evaluate(matrix_doc, baselines, tmp_path)

    by_entry = {(e["entry"], e["cell"]): e for e in doc["entries"]}
    assert by_entry[("gpu_smoke", "baseline-local")]["verdict"] == "record"
    assert by_entry[("inference_offline", "baseline-local")]["verdict"] == "pass"
    assert by_entry[("race", None)]["verdict"] == "skip"
    assert doc["summary"]["record"] == 1
    assert doc["summary"]["pass"] == 1
    assert doc["summary"]["skip"] == 1
    assert doc["summary"]["fail"] == 0


def test_evaluate_fails_when_no_matrix_json(tmp_path, monkeypatch):
    matrix_doc = {"entries": [{"name": "broken", "recipe": "r.yaml"}]}
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(nightly_eval, "run_entry", lambda entry, out_dir: (17, None, False))
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})

    doc = nightly_eval.evaluate(matrix_doc, {"baselines": {}}, tmp_path)
    assert doc["summary"]["fail"] == 1
    assert doc["entries"][0]["verdict"] == "fail"


def test_evaluate_fails_when_all_entries_skip(tmp_path, monkeypatch):
    # Zero work (e.g. no GPUs) must FAIL, not go green.
    matrix_doc = {"entries": [{"name": "race", "recipe": "r.yaml", "nproc": 2, "min_gpus": 2}]}
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 0)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})

    doc = nightly_eval.evaluate(matrix_doc, {"baselines": {}}, tmp_path)
    assert doc["summary"]["fail"] == 1
    assert any(e["error"] == "zero_work" for e in doc["entries"])


def test_evaluate_timeout_records_failure(tmp_path, monkeypatch):
    matrix_doc = {"entries": [{"name": "hang", "recipe": "r.yaml"}]}
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(nightly_eval, "run_entry", lambda entry, out_dir: (124, None, True))
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})

    doc = nightly_eval.evaluate(matrix_doc, {"baselines": {}}, tmp_path)
    assert doc["summary"]["fail"] == 1
    assert doc["entries"][0]["error"] == "timeout"


def test_evaluate_timeout_is_authoritative_even_with_matrix(tmp_path, monkeypatch):
    # rank 0 wrote a matrix.json but a worker hung -> must still fail.
    matrix_doc = {"entries": [{"name": "hang", "recipe": "r.yaml"}]}
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})

    def fake(entry, out_dir):
        mpath = _write_matrix(out_dir / entry["name"] / "matrix.json",
                              [{"name": "c", "error": None, "passed_count": 1,
                                "failed_count": 0, "error_count": 0, "metrics_summary": {}}])
        return 124, mpath, True

    monkeypatch.setattr(nightly_eval, "run_entry", fake)
    doc = nightly_eval.evaluate(matrix_doc, {"baselines": {}}, tmp_path)
    assert doc["summary"]["fail"] == 1
    assert doc["entries"][0]["error"] == "timeout"


def test_evaluate_empty_matrix_entry_fails(tmp_path, monkeypatch):
    matrix_doc = {"entries": [{"name": "nocells", "recipe": "r.yaml"}]}
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})

    def fake(entry, out_dir):
        mpath = _write_matrix(out_dir / entry["name"] / "matrix.json", [])  # zero cells
        return 0, mpath, False

    monkeypatch.setattr(nightly_eval, "run_entry", fake)
    doc = nightly_eval.evaluate(matrix_doc, {"baselines": {}}, tmp_path)
    assert doc["summary"]["fail"] == 1
    assert doc["entries"][0]["error"] == "empty_matrix"


def test_evaluate_corrupt_matrix_json_fails_with_artifact(tmp_path, monkeypatch):
    # A truncated matrix.json must produce a per-entry fail (not abort evaluate()
    # before results JSON is written).
    matrix_doc = {"entries": [{"name": "corrupt", "recipe": "r.yaml"}]}
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})

    def fake(entry, out_dir):
        mpath = out_dir / entry["name"] / "matrix.json"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text('{"cells": [', encoding="utf-8")  # truncated JSON
        return 0, mpath, False

    monkeypatch.setattr(nightly_eval, "run_entry", fake)
    doc = nightly_eval.evaluate(matrix_doc, {"baselines": {}}, tmp_path)
    assert doc["summary"]["fail"] == 1
    assert doc["entries"][0]["error"] == "corrupt_matrix"


refresh_baselines = _load("refresh_baselines")


def test_refresh_carries_over_baseline_for_gpu_skipped_entry(tmp_path, monkeypatch):
    # An 8-GPU entry that can't run on a 1-GPU box must NOT poison the refresh:
    # its existing baseline is carried over instead of dropped to record-only.
    matrix_doc = {"entries": [
        {"name": "gpu_smoke", "recipe": "r1.yaml"},
        {"name": "race_8gpu", "recipe": "r8.yaml", "nproc": 8, "min_gpus": 8},
    ]}
    existing = {"race_8gpu::baseline-local": {"passed": True}}

    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)

    def fake_run_entry(entry, out_dir):
        mpath = _write_matrix(out_dir / entry["name"] / "matrix.json",
                              [{"name": "baseline-local", "error": None, "passed_count": 1,
                                "failed_count": 0, "error_count": 0, "metrics_summary": {}}])
        return 0, mpath, False

    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", fake_run_entry)

    doc = refresh_baselines.build_baselines(
        matrix_doc, tmp_path, 0.25, 0.15, False, existing_baselines=existing)

    assert doc["baselines"]["race_8gpu::baseline-local"] == {"passed": True}
    assert "gpu_smoke::baseline-local" in doc["baselines"]


def test_refresh_gpu_skipped_without_existing_is_not_fatal(tmp_path, monkeypatch):
    matrix_doc = {"entries": [
        {"name": "gpu_smoke", "recipe": "r1.yaml"},
        {"name": "race_8gpu", "recipe": "r8.yaml", "nproc": 8, "min_gpus": 8},
    ]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)

    def fake_run_entry(entry, out_dir):
        mpath = _write_matrix(out_dir / entry["name"] / "matrix.json",
                              [{"name": "baseline-local", "error": None, "passed_count": 1,
                                "failed_count": 0, "error_count": 0, "metrics_summary": {}}])
        return 0, mpath, False

    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", fake_run_entry)

    doc = refresh_baselines.build_baselines(matrix_doc, tmp_path, 0.25, 0.15, False)
    assert "race_8gpu::baseline-local" not in doc["baselines"]
    assert "gpu_smoke::baseline-local" in doc["baselines"]


def test_refresh_refuses_on_corrupt_matrix_json(tmp_path, monkeypatch):
    # A corrupt matrix.json must refuse cleanly (atomic message), not traceback.
    import pytest
    matrix_doc = {"entries": [{"name": "gpu_smoke", "recipe": "r1.yaml"}]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 8)

    def fake_run_entry(entry, out_dir):
        mpath = out_dir / entry["name"] / "matrix.json"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text('{"cells": [', encoding="utf-8")  # truncated JSON
        return 0, mpath, False

    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", fake_run_entry)

    with pytest.raises(SystemExit, match="unreadable matrix.json"):
        refresh_baselines.build_baselines(matrix_doc, tmp_path, 0.25, 0.15, False)


def test_refresh_refuses_when_entry_ran_but_failed(tmp_path, monkeypatch):
    # A genuine did-not-pass (ran but failed) must still refuse atomically.
    import pytest
    matrix_doc = {"entries": [{"name": "gpu_smoke", "recipe": "r1.yaml"}]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 8)

    def fake_run_entry(entry, out_dir):
        mpath = _write_matrix(out_dir / entry["name"] / "matrix.json",
                              [{"name": "baseline-local", "error": None, "passed_count": 0,
                                "failed_count": 1, "error_count": 0, "metrics_summary": {}}])
        return 0, mpath, False

    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", fake_run_entry)

    with pytest.raises(SystemExit):
        refresh_baselines.build_baselines(matrix_doc, tmp_path, 0.25, 0.15, False)


# ---------------------------------------------------------------------------
# Dashboard metadata: the `rocm` column on both install layouts (issue #381)
# ---------------------------------------------------------------------------


def _roots(core: Path, libraries: Path | None = None):
    """Stand-in for rocm_paths.RocmRoots with only what build_metadata reads."""
    libraries = libraries or core
    return SimpleNamespace(
        core=core,
        libraries=libraries,
        version_file=core / ".info" / "version",
        version_dev_file=core / ".info" / "version-dev",
        lib_dir=libraries / "lib",
    )


def _write_version(root: Path, name: str, text: str) -> Path:
    info = root / ".info"
    info.mkdir(parents=True, exist_ok=True)
    path = info / name
    path.write_text(text, encoding="utf-8")
    return path


def test_metadata_reads_rocm_version_on_a_classic_layout(tmp_path, monkeypatch):
    root = tmp_path / "opt_rocm"
    _write_version(root, "version", "7.2.4\n")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    assert nightly_eval.build_metadata()["rocm"] == "7.2.4"


def test_metadata_reads_rocm_version_on_a_wheel_layout(tmp_path, monkeypatch):
    """The regression #381 fixed: a wheel install left this column null.

    `torch` and `hip` still populated from the same run, so the dashboard row
    looked complete while the ROCm version -- the thing rows are compared
    across -- was silently missing.
    """
    core = tmp_path / "site-packages" / "_rocm_sdk_core"
    _write_version(core, "version", "7.14.0\n")
    libraries = tmp_path / "site-packages" / "_rocm_sdk_libraries"
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(core, libraries))
    assert nightly_eval.build_metadata()["rocm"] == "7.14.0"


def test_metadata_falls_back_to_version_dev(tmp_path, monkeypatch):
    root = tmp_path / "opt_rocm"
    _write_version(root, "version-dev", "7.2.4.50311-abc1234\n")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    assert nightly_eval.build_metadata()["rocm"] == "7.2.4.50311-abc1234"


def test_metadata_rocm_is_null_when_no_install_is_found(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(tmp_path / "absent"))
    assert nightly_eval.build_metadata()["rocm"] is None
