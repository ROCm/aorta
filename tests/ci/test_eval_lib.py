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


def test_extract_metrics_pulls_step_time_and_summary():
    cell = {
        "mean_step_time_ms": 4.5,
        "mean_wall_clock_sec": 1.2,
        "step_time_source": "per_step",
        "metrics_summary": {"gflops": {"mean": 120.0}, "decode_latency_ms": {"mean": 55.0}},
    }
    m = eval_lib.extract_metrics(cell)
    assert m["mean_step_time_ms"] == 4.5
    assert m["summary"] == {"gflops": 120.0, "decode_latency_ms": 55.0}


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
    assert harvested[0]["metrics"]["summary"]["gflops"] == 100.0


def test_metric_policy_lookup():
    assert eval_lib.metric_policy("gflops") == "min"
    assert eval_lib.metric_policy("decode_latency_ms") == "max"
    assert eval_lib.metric_policy("logits_checksum") == "equal"
    assert eval_lib.metric_policy("unknown_metric") is None


def test_compare_record_only_when_no_baseline_and_passed():
    harvested = {"cell": "c", "passed": True, "error": None, "metrics": {}}
    out = eval_lib.compare_to_baseline(harvested, None)
    assert out["verdict"] == "record"


def test_compare_fail_when_no_baseline_and_cell_failed():
    # BLOCKER: a failed cell without a baseline must be fail, never record.
    harvested = {"cell": "c", "passed": False, "error": "boom", "metrics": {}}
    out = eval_lib.compare_to_baseline(harvested, None)
    assert out["verdict"] == "fail"
    assert "boom" in out["reasons"][0]


def test_compare_pass_when_within_baseline():
    harvested = {
        "passed": True, "error": None,
        "metrics": {"mean_step_time_ms": 4.0, "summary": {"gflops": 130.0}},
    }
    baseline = {"passed": True, "step_time_ms": {"max": 5.0},
                "metrics": {"gflops": {"policy": "min", "value": 100.0}}}
    out = eval_lib.compare_to_baseline(harvested, baseline)
    assert out["verdict"] == "pass", out


def test_compare_fail_on_slow_step_time():
    harvested = {"passed": True, "error": None, "metrics": {"mean_step_time_ms": 9.0, "summary": {}}}
    baseline = {"passed": True, "step_time_ms": {"max": 5.0}}
    out = eval_lib.compare_to_baseline(harvested, baseline)
    assert out["verdict"] == "fail"
    assert any("mean_step_time_ms" in r for r in out["reasons"])


def test_compare_fail_on_missing_step_time_with_configured_max():
    harvested = {"passed": True, "error": None, "metrics": {"mean_step_time_ms": None, "summary": {}}}
    baseline = {"passed": True, "step_time_ms": {"max": 5.0}}
    out = eval_lib.compare_to_baseline(harvested, baseline)
    assert out["verdict"] == "fail"
    assert any("missing" in r for r in out["reasons"])


def test_compare_metric_policies_min_max_equal_and_missing():
    harvested = {
        "passed": True, "error": None,
        "metrics": {"summary": {"gflops": 50.0, "decode_latency_ms": 20.0, "logits_checksum": 7}},
    }
    baseline = {
        "passed": True,
        "metrics": {
            "gflops": {"policy": "min", "value": 100.0},          # 50 < 100 -> fail
            "decode_latency_ms": {"policy": "max", "value": 10.0},  # 20 > 10 -> fail
            "logits_checksum": {"policy": "equal", "value": 9},     # 7 != 9 -> fail
            "gbps": {"policy": "min", "value": 5.0},                # missing -> fail
        },
    }
    out = eval_lib.compare_to_baseline(harvested, baseline)
    assert out["verdict"] == "fail"
    joined = " ".join(out["reasons"])
    assert "gflops" in joined and "decode_latency_ms" in joined
    assert "logits_checksum" in joined and "gbps" in joined


def test_compare_optional_metric_missing_is_ok():
    harvested = {"passed": True, "error": None, "metrics": {"summary": {}}}
    baseline = {"passed": True, "metrics": {"gbps": {"policy": "min", "value": 5.0, "required": False}}}
    out = eval_lib.compare_to_baseline(harvested, baseline)
    assert out["verdict"] == "pass"


def test_compare_fail_when_expected_pass_but_cell_failed():
    harvested = {"passed": False, "error": "workload_failed", "metrics": {"summary": {}}}
    baseline = {"passed": True}
    out = eval_lib.compare_to_baseline(harvested, baseline)
    assert out["verdict"] == "fail"


def test_summarize_counts_verdicts():
    entries = [{"verdict": "pass"}, {"verdict": "fail"}, {"verdict": "record"}, {"verdict": "skip"}]
    s = eval_lib.summarize(entries)
    assert s == {"total": 4, "pass": 1, "fail": 1, "record": 1, "skip": 1}
