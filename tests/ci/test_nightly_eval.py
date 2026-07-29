"""Integration test for the nightly-eval harness (scripts/ci/nightly_eval.py).

Monkeypatches the GPU / subprocess boundary (run_entry, gpu_count) so the
end-to-end evaluate() logic can be exercised on the CPU gate with a synthetic
matrix.json -- no torch / aorta / GPU required.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
        return 0, mpath

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
    monkeypatch.setattr(nightly_eval, "run_entry", lambda entry, out_dir: (17, None))
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})

    doc = nightly_eval.evaluate(matrix_doc, {"baselines": {}}, tmp_path)
    assert doc["summary"]["fail"] == 1
    assert doc["entries"][0]["verdict"] == "fail"
