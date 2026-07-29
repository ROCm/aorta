"""Unit tests for the nightly-eval pure logic (scripts/ci/eval_lib.py).

Dependency-light (no torch / aorta / GPU) so it runs on the CPU gate. Validates
the matrix.json harvester and the baseline comparator, including the
record-only-when-missing behavior.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_LIB = _REPO_ROOT / "scripts" / "ci" / "eval_lib.py"


def _load_eval_lib():
    spec = importlib.util.spec_from_file_location("eval_lib", _EVAL_LIB)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


eval_lib = _load_eval_lib()


def _matrix_json(tmp_path: Path, cells: list[dict]) -> Path:
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    return p


def test_cell_passed_true_for_clean_cell():
    cell = {"error": None, "passed_count": 2, "failed_count": 0, "error_count": 0}
    assert eval_lib.cell_passed(cell) is True


@pytest.mark.parametrize(
    "cell",
    [
        {"error": "boom", "passed_count": 1, "failed_count": 0, "error_count": 0},
        {"error": None, "passed_count": 1, "failed_count": 1, "error_count": 0},
        {"error": None, "passed_count": 1, "failed_count": 0, "error_count": 1},
        {"error": None, "passed_count": 0, "failed_count": 0, "error_count": 0},
    ],
)
def test_cell_passed_false_for_bad_cells(cell):
    assert eval_lib.cell_passed(cell) is False


def test_extract_metrics_pulls_step_time_and_throughput():
    cell = {
        "mean_step_time_ms": 4.5,
        "mean_wall_clock_sec": 1.2,
        "step_time_source": "per_step",
        "metrics_summary": {"gflops": {"mean": 120.0}, "gbps": {"mean": 55.0}},
    }
    m = eval_lib.extract_metrics(cell)
    assert m["mean_step_time_ms"] == 4.5
    assert m["throughput"] == {"gflops": 120.0, "gbps": 55.0}


def test_harvest_matrix_json(tmp_path):
    p = _matrix_json(
        tmp_path,
        [
            {"name": "baseline-local", "error": None, "passed_count": 1,
             "failed_count": 0, "error_count": 0, "trials": 1, "mean_step_time_ms": 3.0,
             "metrics_summary": {"gflops": {"mean": 100.0}}},
        ],
    )
    harvested = eval_lib.harvest_matrix_json(p)
    assert len(harvested) == 1
    assert harvested[0]["cell"] == "baseline-local"
    assert harvested[0]["passed"] is True
    assert harvested[0]["metrics"]["throughput"]["gflops"] == 100.0


def test_compare_record_only_when_no_baseline():
    harvested = {"cell": "c", "passed": True, "error": None, "metrics": {}}
    out = eval_lib.compare_to_baseline(harvested, None)
    assert out["verdict"] == "record"


def test_compare_pass_when_within_baseline():
    harvested = {
        "passed": True, "error": None,
        "metrics": {"mean_step_time_ms": 4.0, "throughput": {"gflops": 130.0}},
    }
    baseline = {"passed": True, "step_time_ms": {"max": 5.0}, "throughput": {"gflops": {"min": 100.0}}}
    out = eval_lib.compare_to_baseline(harvested, baseline)
    assert out["verdict"] == "pass", out


def test_compare_fail_on_slow_step_time():
    harvested = {"passed": True, "error": None, "metrics": {"mean_step_time_ms": 9.0, "throughput": {}}}
    baseline = {"passed": True, "step_time_ms": {"max": 5.0}}
    out = eval_lib.compare_to_baseline(harvested, baseline)
    assert out["verdict"] == "fail"
    assert any("mean_step_time_ms" in r for r in out["reasons"])


def test_compare_fail_on_low_throughput_and_missing_metric():
    harvested = {"passed": True, "error": None, "metrics": {"mean_step_time_ms": 1.0, "throughput": {"gflops": 50.0}}}
    baseline = {"passed": True, "throughput": {"gflops": {"min": 100.0}, "gbps": {"min": 10.0}}}
    out = eval_lib.compare_to_baseline(harvested, baseline)
    assert out["verdict"] == "fail"
    assert any("gflops" in r for r in out["reasons"])
    assert any("gbps" in r and "missing" in r for r in out["reasons"])


def test_compare_fail_when_expected_pass_but_cell_failed():
    harvested = {"passed": False, "error": "workload_failed", "metrics": {"throughput": {}}}
    baseline = {"passed": True}
    out = eval_lib.compare_to_baseline(harvested, baseline)
    assert out["verdict"] == "fail"


def test_summarize_counts_verdicts():
    entries = [{"verdict": "pass"}, {"verdict": "fail"}, {"verdict": "record"}, {"verdict": "skip"}]
    s = eval_lib.summarize(entries)
    assert s == {"total": 4, "pass": 1, "fail": 1, "record": 1, "skip": 1}
