"""Integration test for the nightly-eval harness (scripts/ci/nightly_eval.py).

Monkeypatches the GPU / subprocess boundary (run_entry, gpu_count) so the
end-to-end evaluate() logic can be exercised on the CPU gate with a synthetic
matrix.json -- no torch / aorta / GPU required.
"""

from __future__ import annotations

import importlib.util
import json
import re
import unicodedata
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
# Scoping --perf-gate to one entry, and not auto-arming a near-zero metric.
# Both exist so serving perf gating can be rolled out on its own; see
# docs/tokenspeed-gating-rollout.md.
# ---------------------------------------------------------------------------


_SERVING_SUMMARY = {
    "median_ttft_ms": {"mean": 46.3},
    "median_tpot_ms": {"mean": 1.94},
    "median_itl_ms": {"mean": 0.0},
    "p99_itl_ms": {"mean": 21.4},
    "output_throughput": {"mean": 3538.0},
}


def _serving_matrix_runner(summary=None):
    summary = _SERVING_SUMMARY if summary is None else summary

    def fake_run_entry(entry, out_dir):
        mpath = _write_matrix(out_dir / entry["name"] / "matrix.json",
                              [{"name": "baseline", "error": None, "passed_count": 1,
                                "failed_count": 0, "error_count": 0,
                                "mean_step_time_ms": 1100.0,
                                "metrics_summary": summary}])
        return 0, mpath, False

    return fake_run_entry


def test_perf_gate_can_be_scoped_to_a_single_entry(tmp_path, monkeypatch):
    """Rolling gating out per workload must not arm every other entry too.

    refresh_baselines rewrites the whole baseline file, so an unscoped
    --perf-gate would derive step-time ceilings for every unrelated entry from
    whatever single run happened to be under way -- the one-observation
    threshold this rollout exists to avoid, applied to workloads nobody looked at.
    """
    matrix_doc = {"entries": [
        {"name": "tokenspeed_serve_smoke", "recipe": "ts.yaml"},
        {"name": "gpu_smoke", "recipe": "r1.yaml"},
    ]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())

    doc = refresh_baselines.build_baselines(
        matrix_doc, tmp_path, 0.25, 0.15, True,
        perf_gate_entries={"tokenspeed_serve_smoke"})

    gated = doc["baselines"]["tokenspeed_serve_smoke::baseline"]
    ungated = doc["baselines"]["gpu_smoke::baseline"]
    assert gated["step_time_ms"] == {"max": 1375.0}
    assert gated["metrics"]["median_tpot_ms"] == {"policy": "max", "value": 2.425}
    assert gated["metrics"]["output_throughput"] == {"policy": "min", "value": 3007.3}
    # The unscoped entry stays record-only for performance: no step-time ceiling
    # and no metric bounds at all.
    assert "step_time_ms" not in ungated
    assert "metrics" not in ungated


def test_perf_gate_without_a_scope_still_gates_every_entry(tmp_path, monkeypatch):
    """The default is unchanged, so existing callers keep their behaviour."""
    matrix_doc = {"entries": [
        {"name": "tokenspeed_serve_smoke", "recipe": "ts.yaml"},
        {"name": "gpu_smoke", "recipe": "r1.yaml"},
    ]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())

    doc = refresh_baselines.build_baselines(matrix_doc, tmp_path, 0.25, 0.15, True)
    for key in ("tokenspeed_serve_smoke::baseline", "gpu_smoke::baseline"):
        assert doc["baselines"][key]["step_time_ms"] == {"max": 1375.0}


def test_perf_gate_does_not_arm_a_near_zero_median_itl(tmp_path, monkeypatch):
    """`median_itl_ms` is measured at ~0, so `value * 1.25` is a ceiling of 0.0
    that no later run with any inter-token gap can satisfy. It stays charted and
    hand-gateable; the refresher must not bless a bound for it."""
    matrix_doc = {"entries": [{"name": "tokenspeed_serve_smoke", "recipe": "ts.yaml"}]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())

    doc = refresh_baselines.build_baselines(matrix_doc, tmp_path, 0.25, 0.15, True)
    metrics = doc["baselines"]["tokenspeed_serve_smoke::baseline"]["metrics"]
    assert "median_itl_ms" not in metrics
    # Its tail counterpart is a real number and stays gateable.
    assert metrics["p99_itl_ms"] == {"policy": "max", "value": 26.75}


# A gpu_smoke cell that has already been blessed with performance bounds, in
# the shape --perf-gate writes them. This is the state the scope is supposed to
# protect: sixteen cells belonging to other people's workloads.
_BLESSED_GPU_SMOKE = {
    "gpu_smoke::baseline": {
        "passed": True,
        "step_time_ms": {"max": 2.5},
        "metrics": {
            "tokens_per_sec": {"policy": "min", "value": 900.0},
            "logits_checksum": {"policy": "equal", "value": "old-checksum"},
        },
    },
}


def test_a_scoped_refresh_leaves_other_entries_bounds_untouched(tmp_path, monkeypatch):
    """The scope exists to PROTECT the other entries, so it must not disarm them.

    The baseline file is rewritten whole, so rebuilding an out-of-scope cell as
    `{"passed": True}` plus correctness metrics deletes whatever bounds it had.
    That makes the rollout the scope was added for impossible in the other
    direction: bless serving, and the next scoped bless of anything else
    silently ungates serving again.
    """
    matrix_doc = {"entries": [
        {"name": "tokenspeed_serve_smoke", "recipe": "ts.yaml"},
        {"name": "gpu_smoke", "recipe": "r1.yaml"},
    ]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(
        refresh_baselines.nightly_eval, "run_entry",
        _serving_matrix_runner({**_SERVING_SUMMARY, "logits_checksum": {"mean": "fresh"}}))

    doc = refresh_baselines.build_baselines(
        matrix_doc, tmp_path, 0.25, 0.15, True,
        existing_baselines=_BLESSED_GPU_SMOKE,
        perf_gate_entries={"tokenspeed_serve_smoke"})

    kept = doc["baselines"]["gpu_smoke::baseline"]
    assert kept["step_time_ms"] == {"max": 2.5}, "the out-of-scope step-time ceiling was disarmed"
    assert kept["metrics"]["tokens_per_sec"] == {"policy": "min", "value": 900.0}, (
        "the out-of-scope throughput floor was disarmed"
    )
    # Correctness data is still refreshed for it -- carrying the bounds over is
    # not the same as skipping the entry.
    assert kept["metrics"]["logits_checksum"] == {"policy": "equal", "value": "fresh"}
    # And the entry actually under test does get its bounds derived.
    assert doc["baselines"]["tokenspeed_serve_smoke::baseline"]["step_time_ms"] == {"max": 1375.0}


def test_a_correctness_only_refresh_does_not_disarm_existing_gates(tmp_path, monkeypatch):
    """The same hole through the other door.

    Omitting `--perf-gate` means "do not arm new gates", not "disarm the ones
    already blessed" -- and the default dispatch of refresh-baselines.yml is
    exactly this mode, so a routine correctness refresh must not silently revert
    every gated cell to record-only.
    """
    matrix_doc = {"entries": [{"name": "gpu_smoke", "recipe": "r1.yaml"}]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())

    doc = refresh_baselines.build_baselines(
        matrix_doc, tmp_path, 0.25, 0.15, False, existing_baselines=_BLESSED_GPU_SMOKE)

    kept = doc["baselines"]["gpu_smoke::baseline"]
    assert kept["step_time_ms"] == {"max": 2.5}
    assert kept["metrics"]["tokens_per_sec"] == {"policy": "min", "value": 900.0}


def test_a_carried_bound_is_not_mutated_by_the_refresh(tmp_path, monkeypatch):
    """Carried verbatim, and by value: the written document must not alias the
    input, or a later edit to one would silently change the other."""
    matrix_doc = {"entries": [{"name": "gpu_smoke", "recipe": "r1.yaml"}]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())

    existing = {"gpu_smoke::baseline": {"passed": True, "step_time_ms": {"max": 2.5},
                                        "metrics": {"tokens_per_sec": {"policy": "min",
                                                                       "value": 900.0}}}}
    doc = refresh_baselines.build_baselines(
        matrix_doc, tmp_path, 0.25, 0.15, False, existing_baselines=existing)

    doc["baselines"]["gpu_smoke::baseline"]["step_time_ms"]["max"] = 999.0
    assert existing["gpu_smoke::baseline"]["step_time_ms"]["max"] == 2.5


def test_an_in_scope_entry_has_its_stale_bounds_replaced(tmp_path, monkeypatch):
    """The carry-over must not shadow the derivation it is protecting: a
    re-bless of the entry under test replaces its old bounds rather than
    preserving them."""
    matrix_doc = {"entries": [{"name": "tokenspeed_serve_smoke", "recipe": "ts.yaml"}]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())

    existing = {"tokenspeed_serve_smoke::baseline": {
        "passed": True, "step_time_ms": {"max": 1.0},
        "metrics": {"median_tpot_ms": {"policy": "max", "value": 0.001}},
    }}
    doc = refresh_baselines.build_baselines(
        matrix_doc, tmp_path, 0.25, 0.15, True, existing_baselines=existing,
        perf_gate_entries={"tokenspeed_serve_smoke"})

    spec = doc["baselines"]["tokenspeed_serve_smoke::baseline"]
    assert spec["step_time_ms"] == {"max": 1375.0}
    assert spec["metrics"]["median_tpot_ms"] == {"policy": "max", "value": 2.425}


def test_a_re_bless_keeps_a_hand_written_no_auto_gate_bound(tmp_path, monkeypatch):
    """Deriving is not re-deriving *everything*.

    `median_itl_ms` is on the gating allowlist but in `_NO_AUTO_GATE`: gateable
    when a baseline names it by hand, never armed by `--perf-gate`, because it
    is measured at ~0 and a relative margin around zero is not a bound. So the
    refresher has nothing to put back in its place -- and carrying bounds over
    only when the run is NOT deriving meant a scoped re-bless of this very entry
    silently deleted it. A refresh may decline to arm a gate; it must never
    disarm one.
    """
    matrix_doc = {"entries": [{"name": "tokenspeed_serve_smoke", "recipe": "ts.yaml"}]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())

    existing = {"tokenspeed_serve_smoke::baseline": {
        "passed": True,
        "metrics": {
            "median_itl_ms": {"policy": "max", "value": 5.0},
            "median_tpot_ms": {"policy": "max", "value": 0.001},
        },
    }}
    doc = refresh_baselines.build_baselines(
        matrix_doc, tmp_path, 0.25, 0.15, True, existing_baselines=existing,
        perf_gate_entries={"tokenspeed_serve_smoke"})

    metrics = doc["baselines"]["tokenspeed_serve_smoke::baseline"]["metrics"]
    assert metrics["median_itl_ms"] == {"policy": "max", "value": 5.0}, (
        "the hand-written median_itl_ms ceiling was dropped by a re-bless of its own entry"
    )
    # The observation is 0.0, so had it been auto-armed the ceiling would be 0.0
    # -- a gate that cannot pass. That is the whole reason it is _NO_AUTO_GATE.
    assert metrics["median_itl_ms"]["value"] != 0.0
    # And the auto-gateable neighbour is still re-derived rather than carried.
    assert metrics["median_tpot_ms"] == {"policy": "max", "value": 2.425}


def test_a_re_bless_keeps_a_bound_on_a_metric_off_the_allowlist(tmp_path, monkeypatch):
    """Same argument, one step further out: a spec naming a metric the allowlist
    does not know is hand-written by definition, so `--perf-gate` can never
    reproduce it and must not delete it either."""
    matrix_doc = {"entries": [{"name": "tokenspeed_serve_smoke", "recipe": "ts.yaml"}]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())

    existing = {"tokenspeed_serve_smoke::baseline": {
        "passed": True, "metrics": {"kv_cache_hit_rate": {"policy": "min", "value": 0.9}},
    }}
    doc = refresh_baselines.build_baselines(
        matrix_doc, tmp_path, 0.25, 0.15, True, existing_baselines=existing,
        perf_gate_entries={"tokenspeed_serve_smoke"})

    metrics = doc["baselines"]["tokenspeed_serve_smoke::baseline"]["metrics"]
    assert metrics["kv_cache_hit_rate"] == {"policy": "min", "value": 0.9}


def test_a_re_bless_without_a_step_time_observation_keeps_the_old_ceiling(tmp_path, monkeypatch):
    """`mean_step_time_ms` is read with `.get`, so a cell can report none.

    With no observation there is no new ceiling to write, and dropping the old
    one would disarm it on a run that never measured it. `compare_to_baseline`
    treats a missing observation against a live ceiling as a failure, which is
    the loud outcome rather than the silent one.
    """
    def run_without_step_time(entry, out_dir):
        mpath = _write_matrix(out_dir / entry["name"] / "matrix.json",
                              [{"name": "baseline", "error": None, "passed_count": 1,
                                "failed_count": 0, "error_count": 0,
                                "mean_step_time_ms": None,
                                "metrics_summary": _SERVING_SUMMARY}])
        return 0, mpath, False

    matrix_doc = {"entries": [{"name": "tokenspeed_serve_smoke", "recipe": "ts.yaml"}]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", run_without_step_time)

    existing = {"tokenspeed_serve_smoke::baseline": {
        "passed": True, "step_time_ms": {"max": 1462.0},
    }}
    doc = refresh_baselines.build_baselines(
        matrix_doc, tmp_path, 0.25, 0.15, True, existing_baselines=existing,
        perf_gate_entries={"tokenspeed_serve_smoke"})

    assert doc["baselines"]["tokenspeed_serve_smoke::baseline"]["step_time_ms"] == {"max": 1462.0}


def test_a_re_bless_does_not_resurrect_a_correctness_metric(tmp_path, monkeypatch):
    """The carry-over's one deliberate exclusion, re-pinned on the derive path.

    Correctness metrics are re-derived from this run -- that is the point of a
    refresh -- so a stale checksum must not survive alongside the fresh one."""
    matrix_doc = {"entries": [{"name": "tokenspeed_serve_smoke", "recipe": "ts.yaml"}]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(
        refresh_baselines.nightly_eval, "run_entry",
        _serving_matrix_runner({**_SERVING_SUMMARY, "logits_checksum": {"mean": "fresh"}}))

    existing = {"tokenspeed_serve_smoke::baseline": {
        "passed": True, "metrics": {"logits_checksum": {"policy": "equal", "value": "stale"}},
    }}
    doc = refresh_baselines.build_baselines(
        matrix_doc, tmp_path, 0.25, 0.15, True, existing_baselines=existing,
        perf_gate_entries={"tokenspeed_serve_smoke"})

    metrics = doc["baselines"]["tokenspeed_serve_smoke::baseline"]["metrics"]
    assert metrics["logits_checksum"] == {"policy": "equal", "value": "fresh"}


def test_a_hand_written_bound_without_a_policy_is_also_carried(tmp_path, monkeypatch):
    """`compare_to_baseline` falls back to the allowlist when a spec names no
    policy, so such a spec is a live gate and must survive a refresh too."""
    matrix_doc = {"entries": [{"name": "gpu_smoke", "recipe": "r1.yaml"}]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())

    existing = {"gpu_smoke::baseline": {
        "passed": True, "metrics": {"median_tpot_ms": {"value": 2.0}},
    }}
    doc = refresh_baselines.build_baselines(
        matrix_doc, tmp_path, 0.25, 0.15, False, existing_baselines=existing)

    assert doc["baselines"]["gpu_smoke::baseline"]["metrics"]["median_tpot_ms"] == {"value": 2.0}


# ---------------------------------------------------------------------------
# The refresh path and `needs_docker_daemon`.
#
# `refresh-baselines.yml` mounts no docker socket, so on that lane a
# daemon-dependent entry cannot run at all. The flag was only honoured by
# nightly_eval.evaluate, which left build_baselines running the entry anyway --
# producing no matrix.json, routing into `incomplete`, and aborting EVERY
# baseline refresh as partial for as long as such an entry is live.
# ---------------------------------------------------------------------------


def _daemon_matrix():
    return {"entries": [
        {"name": "tokenspeed_serve_smoke", "recipe": "ts.yaml", "needs_docker_daemon": True},
        {"name": "gpu_smoke", "recipe": "r1.yaml"},
    ]}


def test_a_refresh_skips_a_daemon_dependent_entry_rather_than_failing(tmp_path, monkeypatch):
    """An unrelated refresh must still complete. This is the whole defect: the
    refresh lane has no socket, so without this the serving row breaks blessing
    for gpu_smoke, inference_offline and every other workload in the matrix."""
    ran: list[str] = []

    def run(entry, out_dir):
        ran.append(entry["name"])
        return _serving_matrix_runner()(entry, out_dir)

    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", run)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "docker_daemon",
                        lambda: (False, "docker info failed: Cannot connect to the Docker daemon"))

    doc = refresh_baselines.build_baselines(_daemon_matrix(), tmp_path, 0.25, 0.15, False)

    assert ran == ["gpu_smoke"], "the daemon-dependent entry was run without a daemon"
    assert "gpu_smoke::baseline" in doc["baselines"]
    assert not any(k.startswith("tokenspeed_serve_smoke::") for k in doc["baselines"])


def test_a_daemon_dependent_entry_keeps_its_baseline_on_a_lane_with_no_socket(
        tmp_path, monkeypatch):
    """Same contract as an 8-GPU entry on a 1-GPU box: an entry this lane cannot
    exercise keeps whatever it was blessed with, rather than being reverted to
    record-only by a refresh that never measured it."""
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())
    monkeypatch.setattr(refresh_baselines.nightly_eval, "docker_daemon",
                        lambda: (False, "no docker client on PATH"))

    existing = {"tokenspeed_serve_smoke::baseline": {
        "passed": True, "step_time_ms": {"max": 1462.0},
        "metrics": {"median_tpot_ms": {"policy": "max", "value": 2.4}},
    }}
    doc = refresh_baselines.build_baselines(
        _daemon_matrix(), tmp_path, 0.25, 0.15, False, existing_baselines=existing)

    assert doc["baselines"]["tokenspeed_serve_smoke::baseline"] == \
        existing["tokenspeed_serve_smoke::baseline"]


def test_blessing_a_daemon_dependent_entry_that_cannot_run_is_refused(tmp_path, monkeypatch):
    """The one case where skipping is wrong. Scoping perf gating to an entry
    that never ran would emit a correctness-only refresh that reads, in the PR
    diff, exactly like a successful bless -- the same failure the unknown-name
    check refuses."""
    import pytest

    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())
    monkeypatch.setattr(refresh_baselines.nightly_eval, "docker_daemon",
                        lambda: (False, "no docker client on PATH"))

    with pytest.raises(SystemExit, match="needs a docker daemon this lane cannot reach"):
        refresh_baselines.build_baselines(
            _daemon_matrix(), tmp_path, 0.25, 0.15, True,
            perf_gate_entries={"tokenspeed_serve_smoke"})


def test_a_daemon_dependent_entry_is_refreshed_when_a_daemon_is_reachable(tmp_path, monkeypatch):
    """The other half: the skip disappears on a lane that does have the socket,
    with no further edit -- so the flag cannot silently make the entry
    unblessable forever."""
    ran: list[str] = []

    def run(entry, out_dir):
        ran.append(entry["name"])
        return _serving_matrix_runner()(entry, out_dir)

    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", run)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "docker_daemon",
                        lambda: (True, "daemon 29.7.2"))

    doc = refresh_baselines.build_baselines(
        _daemon_matrix(), tmp_path, 0.25, 0.15, True,
        perf_gate_entries={"tokenspeed_serve_smoke"})

    assert ran == ["tokenspeed_serve_smoke", "gpu_smoke"]
    assert doc["baselines"]["tokenspeed_serve_smoke::baseline"]["step_time_ms"] == {"max": 1375.0}


def test_the_probe_is_only_consulted_by_entries_that_declare_it(tmp_path, monkeypatch):
    """A broken daemon must not quarantine the rest of the refresh, exactly as
    it does not on the nightly path."""
    probed: list[bool] = []

    def probe():
        probed.append(True)
        return False, "no docker client on PATH"

    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())
    monkeypatch.setattr(refresh_baselines.nightly_eval, "docker_daemon", probe)

    matrix_doc = {"entries": [{"name": "gpu_smoke", "recipe": "r1.yaml"}]}
    doc = refresh_baselines.build_baselines(matrix_doc, tmp_path, 0.25, 0.15, False)

    assert probed == []
    assert "gpu_smoke::baseline" in doc["baselines"]


def _refresh_cli(*argv):
    import subprocess
    import sys
    return subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "ci" / "refresh_baselines.py"), *argv],
        capture_output=True, text=True, timeout=120)


def test_perf_gate_entry_rejects_a_name_the_matrix_does_not_have():
    """A typo would scope perf gating to nothing and produce a correctness-only
    refresh that reads, in the PR diff, exactly like a successful bless."""
    out = _refresh_cli("--perf-gate", "--perf-gate-entry", "tokenspeed_serve_smoek")
    assert out.returncode != 0
    assert "no such matrix entry" in out.stderr
    # The message has to be actionable from the terminal it appeared in.
    assert "gpu_smoke" in out.stderr


def test_perf_gate_entry_without_perf_gate_is_rejected():
    """Silently ignoring it would leave the operator believing gates were armed."""
    out = _refresh_cli("--perf-gate-entry", "gpu_smoke")
    assert out.returncode != 0
    assert "no effect without --perf-gate" in out.stderr


def test_an_unknown_entry_is_not_a_valid_perf_gate_scope():
    """Scoping to a name that is not in `entries` -- a staged entry, or a typo --
    must fail rather than quietly refresh everything correctness-only."""
    out = _refresh_cli("--perf-gate", "--perf-gate-entry", "not_a_real_entry")
    assert out.returncode != 0
    assert "no such matrix entry" in out.stderr


# ---------------------------------------------------------------------------
# The committed matrix file itself, through the loader the nightly uses.
# ---------------------------------------------------------------------------


def _real_matrix():
    return nightly_eval._load_yaml(nightly_eval.MATRIX)


def test_the_committed_matrix_loads_and_every_recipe_exists():
    doc = _real_matrix()
    assert doc["version"] == 1
    entries = doc["entries"]
    assert entries and len({e["name"] for e in entries}) == len(entries)
    for entry in entries:
        assert (nightly_eval.REPO_ROOT / entry["recipe"]).is_file(), entry["recipe"]


def test_pending_entries_are_inert(tmp_path, monkeypatch):
    """`pending_entries` documents staged work; nothing may execute it.

    Both consumers iterate `entries` only, and this pins that -- a future reader
    who wires the key up would otherwise start a blocked entry by accident.
    """
    doc = _real_matrix()
    # Synthesised rather than read from the file: the key is legitimately empty
    # between staged entries, and the property being pinned -- that a future
    # reader who wires the key up starts a blocked entry -- must hold then too.
    doc = dict(doc)
    doc["pending_entries"] = [
        {"name": "staged_thing", "recipe": "recipes/ci/gpu-smoke.yaml", "blocked_on": "x"},
        *(doc.get("pending_entries") or []),
    ]
    pending = {e["name"] for e in doc["pending_entries"]}

    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 8)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})
    ran: list[str] = []

    def fake(entry, out_dir):
        ran.append(entry["name"])
        return 0, _write_matrix(out_dir / entry["name"] / "matrix.json",
                                [{"name": "c", "error": None, "passed_count": 1,
                                  "failed_count": 0, "error_count": 0,
                                  "metrics_summary": {}}]), False

    monkeypatch.setattr(nightly_eval, "run_entry", fake)
    result = nightly_eval.evaluate(doc, {"baselines": {}}, tmp_path)

    assert not (pending & set(ran))
    assert not (pending & {e["entry"] for e in result["entries"]})


def _docker_matrix():
    return {"entries": [
        {"name": "needs_daemon", "recipe": "r.yaml", "needs_docker_daemon": True},
        {"name": "plain", "recipe": "r2.yaml"},
    ]}


def _fake_run(ran):
    def run(entry, out_dir):
        ran.append(entry["name"])
        return 0, _write_matrix(out_dir / entry["name"] / "matrix.json",
                                [{"name": "c", "error": None, "passed_count": 1,
                                  "failed_count": 0, "error_count": 0,
                                  "metrics_summary": {}}]), False
    return run


def test_an_entry_needing_a_daemon_skips_rather_than_fails_when_there_is_none(
        tmp_path, monkeypatch):
    """The socket is a per-lane opt-in that grants effective root, so it is
    granted to the nightly lane and to nothing else -- which makes "no daemon"
    the normal state everywhere else, not a fault. An entry that needs one must
    therefore skip: failing would redden a lane for declining a privilege it was
    right to decline."""
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 8)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})
    monkeypatch.setattr(nightly_eval, "docker_daemon",
                        lambda: (False, "no docker client on PATH"))
    ran: list[str] = []
    monkeypatch.setattr(nightly_eval, "run_entry", _fake_run(ran))

    result = nightly_eval.evaluate(_docker_matrix(), {"baselines": {}}, tmp_path)
    by_name = {e["entry"]: e for e in result["entries"]}

    assert by_name["needs_daemon"]["verdict"] == "skip"
    assert "no docker client on PATH" in by_name["needs_daemon"]["reasons"][0]
    assert "needs_daemon" not in ran
    # And the probe must not quarantine anything that did not ask for it.
    assert by_name["plain"]["verdict"] != "skip"
    assert ran == ["plain"]


def test_an_entry_needing_a_daemon_runs_when_one_is_reachable(tmp_path, monkeypatch):
    """The other half: the skip must disappear the moment a lane mounts the
    socket, with no further edit to the matrix."""
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 8)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})
    monkeypatch.setattr(nightly_eval, "docker_daemon", lambda: (True, "daemon 29.4.2"))
    ran: list[str] = []
    monkeypatch.setattr(nightly_eval, "run_entry", _fake_run(ran))

    nightly_eval.evaluate(_docker_matrix(), {"baselines": {}}, tmp_path)
    assert ran == ["needs_daemon", "plain"]


def test_the_daemon_probe_reports_a_client_that_cannot_reach_a_daemon(monkeypatch):
    """A mounted-but-dead socket is the interesting case: the client is on PATH,
    so a `which docker` check would call the capability present and the entry
    would fail on connect. The probe must actually talk to the daemon."""
    import subprocess as sp

    nightly_eval.docker_daemon.cache_clear()
    monkeypatch.setattr(nightly_eval.shutil, "which", lambda _: "/usr/local/bin/docker")
    monkeypatch.setattr(
        nightly_eval.subprocess, "run",
        lambda *a, **k: sp.CompletedProcess(
            a[0], 1, "", "Cannot connect to the Docker daemon at unix:///var/run/docker.sock."),
    )
    try:
        reachable, detail = nightly_eval.docker_daemon()
        assert reachable is False
        assert "Cannot connect to the Docker daemon" in detail
    finally:
        nightly_eval.docker_daemon.cache_clear()


def test_the_live_serving_entry_declares_the_capability_it_needs():
    """Promoted on a demonstrated launch, but only safe in `entries` because it
    declares the capability. Without the flag it is a guaranteed nightly failure
    on every runner that has not opted into the socket."""
    live = {e["name"]: e for e in _real_matrix()["entries"]}
    entry = live["tokenspeed_serve_smoke"]
    assert entry["needs_docker_daemon"] is True
    assert "blocked_on" not in entry
    assert int(entry["timeout_sec"]) >= 3600


def test_the_gating_recipe_discards_two_steps_before_measuring():
    """A step-0 Triton compile excursion reaches the metrics at `warmup_steps: 1`.

    Measured at 1 cell-run in 13: a ~10x first-token spike (465 ms against a
    43-47 ms clean range) and a 2825 ms step against 1108-1178 ms. The excursion
    lands on the first *measured* step, after the single warmup step has already
    been discarded, so it breaches three of the four originally-proposed gates on
    a run that is not a regression.

    `num_warmups` is a different knob and cannot substitute: it warms requests
    within one bench invocation, not the compile cache across invocations. The
    excursion is positional, so one more discarded step removes it by
    construction -- which is the only reason the duration-derived metrics can be
    promoted later at all.

    The matrix carries no `workload_config` override (nightly_eval.py reads only
    recipe/min_gpus/timeout_sec/needs_docker_daemon), so the recipe is the single
    place this can be set, and it is shared with the baseline refresher -- which
    is what keeps the blessed numbers and the nightly on the same measurement.
    """
    import yaml

    entry = {e["name"]: e for e in _real_matrix()["entries"]}["tokenspeed_serve_smoke"]
    recipe_path = nightly_eval.REPO_ROOT / entry["recipe"]
    config = yaml.safe_load(recipe_path.read_text("utf-8"))["workload_config"]

    assert config["warmup_steps"] == 2, (
        "the gating recipe must discard two bench steps; at 1 the measured "
        "step-0 compile excursion enters the metrics the nightly gates on"
    )

    # The workload derives its internal budget from
    # ready_timeout + (steps + warmup_steps) * bench_timeout, so an extra warmup
    # step lengthens it. The binding cap is the entry's own timeout_sec, and a
    # bench step here is ~1.1s against a 3600s budget dominated by bring-up --
    # but assert the relationship rather than trusting the arithmetic to hold.
    assert int(entry["timeout_sec"]) >= 3600


def test_the_rollout_doc_marks_its_variance_data_as_the_old_configuration():
    """The 13-cell-run variance table was measured at `warmup_steps: 1`, which the
    recipe no longer uses, so it is the rationale for the change and not the
    baseline to bless against. It is deliberately retained -- there is no
    measurement at the new setting yet, and the ten-night record-only window is
    what produces one. This guards the caveat, not the table."""
    doc = (nightly_eval.REPO_ROOT / "docs/tokenspeed-gating-rollout.md").read_text("utf-8")
    assert "warmup_steps: 1" in doc, "the provenance of the variance data is gone"
    assert "warmup_steps: 2" in doc, "the doc does not state the new setting"
    # The table itself must survive.
    assert "Clean range over 12 cell-runs" in doc
    # And the window has to be pinned to the setting the gate will run at.
    assert re.search(
        r"ten nightlies, at `warmup_steps: 2`", doc
    ), "the record-only window is not pinned to warmup_steps: 2"


def _gated_names(spec: dict) -> set[str]:
    """Every bound in one baseline entry that can red a cell, under one naming.

    `step_time_ms` is not in `metrics`. `eval_lib.compare_to_baseline` reads it
    as a *sibling* key (`baseline["step_time_ms"]["max"]`, eval_lib.py:218) and
    compares it against the harness-synthesised `mean_step_time_ms`, so a
    reader of `spec["metrics"]` alone sees an armed step-time ceiling as no
    gate at all. That is precisely the bound this bless pruned by hand, which
    makes it the one a future bless is most likely to restore by accident --
    `refresh_baselines.py:231` writes it from `mean_step_time_ms` without being
    asked. Folded in under the `step_time_ms.max` spelling the docs use, so the
    tripwires below and the prose they check are talking about one name.
    """
    names = set(spec.get("metrics") or {})
    names |= {f"step_time_ms.{bound}" for bound in (spec.get("step_time_ms") or {})}
    return names


def _gated_serving_metrics() -> dict[str, set[str]]:
    """``cell -> gated metric names`` for tokenspeed_serve_smoke, from the real file."""
    import yaml

    baselines = yaml.safe_load(
        (nightly_eval.REPO_ROOT / "config/ci/regression_baselines.yaml").read_text("utf-8")
    )["baselines"]
    return {
        key.split("::", 1)[1]: _gated_names(spec)
        for key, spec in baselines.items()
        if key.startswith("tokenspeed_serve_smoke::")
    }


#: The two serving cells, spelled out. `assert gated` and a one-element set of
#: frozensets are both satisfied by a single cell, so neither says what the docs
#: say -- deleting the `no-scratch-reclaim` block left all three tripwires green
#: while both documents went on claiming "gated on **both** cells".
_SERVING_CELLS = frozenset({"baseline", "no-scratch-reclaim"})

#: The two derivations step 6 of the rollout doc documents: `max × 1.25` on the
#: window maximum for a `max` metric, `min × 0.85` on the window minimum for a
#: `min` one. Stated here so the derivation check below has something to check
#: against; the doc text for both is asserted alongside them so they cannot
#: drift.
#:
#: Only `max` is armed today. `_FLOOR_MARGIN` is here because step 7 names
#: `output_throughput` -- a `min` metric -- as ready for promotion, and a check
#: that only knows ceilings would fail that promotion for having the wrong
#: policy rather than the wrong number.
_BLESS_MARGIN = 1.25
_FLOOR_MARGIN = 0.85

_BACKTICKED = re.compile(r"`([A-Za-z0-9_.]+)`")


def _table_rows(doc: str, header: tuple[str, ...]) -> list[list[str]]:
    """The body rows of the markdown table whose header cells are ``header``.

    Anchored on the header rather than on a line number or a nearby heading, so
    moving the table within the file is not a failure and renaming its columns
    is -- a renamed column is the edit that would silently empty the row list
    and turn every check built on it into a comparison against nothing. Callers
    assert the result is non-empty for that reason.
    """
    rows: list[list[str]] = []
    collecting = False
    for line in doc.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            collecting = False
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if tuple(c.lower() for c in cells) == tuple(h.lower() for h in header):
            collecting = True
            continue
        if not collecting:
            continue
        if all(set(cell) <= set("-: ") for cell in cells):
            continue          # the |---|---| separator
        rows.append(cells)
    return rows


def _documented_gate_sets() -> dict[str, dict[str, set[str]]]:
    """``doc -> {"gated": names, "record_only": names}``, read from the tables.

    The previous spelling of this check asked whether each gated name appeared
    *anywhere* in each file. Both documents already name every record-only
    metric, so that condition is satisfied by any metric that could ever be
    gated and the check could not fail: arming `median_ttft_ms` on both cells
    and changing nothing in the docs passed, and so did moving `median_tpot_ms`
    into the record-only list while it stayed gated. A substring test against a
    document that lists both sets is a test of the vocabulary, not of the
    claim.

    So the names come out of the structures that carry the claim. In
    `tokenspeed-serving.md` that is the three-column table under "Gated in the
    nightly"; in `tokenspeed-gating-rollout.md` it is the `First bless` column
    of the per-metric table, which states a verdict per metric and is the one
    place either document distinguishes the two sets row by row.
    """
    serving = (nightly_eval.REPO_ROOT / "docs/tokenspeed-serving.md").read_text("utf-8")
    serving_rows = _table_rows(serving, ("metric", "policy", "why this one"))
    assert serving_rows, (
        "docs/tokenspeed-serving.md has no `| metric | policy | why this one |` "
        "table, so the gated set cannot be read out of it and this check would "
        "compare against nothing."
    )

    rollout = (
        nightly_eval.REPO_ROOT / "docs/tokenspeed-gating-rollout.md"
    ).read_text("utf-8")
    rollout_rows = _table_rows(rollout, ("Metric", "Policy", "First bless", "Why"))
    assert rollout_rows, (
        "docs/tokenspeed-gating-rollout.md has no per-metric "
        "`| Metric | Policy | First bless | Why |` table."
    )

    def names(rows: list[list[str]], verdict: str | None) -> set[str]:
        return {
            name
            for row in rows
            if verdict is None or verdict in row[2]
            for name in _BACKTICKED.findall(row[0])
        }

    return {
        "docs/tokenspeed-serving.md": {
            # That table is the gated set; the record-only names are prose
            # beneath it, and the rollout doc states them row by row instead.
            "gated": names(serving_rows, None),
            "record_only": set(),
        },
        "docs/tokenspeed-gating-rollout.md": {
            "gated": names(rollout_rows, "**Gate**"),
            "record_only": names(rollout_rows, "**Record-only**"),
        },
    }


def test_the_docs_name_exactly_the_serving_metrics_that_are_gated():
    """Tie the prose to the config, because it already drifted once.

    Arming the first gate made three separate documents wrong at a stroke --
    they went on describing the serving cells as record-only and nothing as
    gated -- and nothing in the tree noticed, because a claim about what is
    gated lived only in prose. The same edit will be made again for step 7, so
    the failure mode is recurring rather than historical.

    Deliberately asserting the *set*, not that it is these two names. A reader
    needs to know which metrics can red the nightly, so a doc that says "gated"
    without naming them is its own inaccuracy; a doc that names a metric the
    file does not gate is worse. Both directions fail here, and a future
    promotion is then a docs edit the test demands rather than one it forbids.
    """
    gated = _gated_serving_metrics()
    assert set(gated) == set(_SERVING_CELLS), (
        f"expected both documented serving cells, got {sorted(gated)}. Both "
        "documents say the gate is armed on both, so one cell in the file is a "
        "claim neither of them makes."
    )
    assert len(set(map(frozenset, gated.values()))) == 1, (
        f"the two serving cells gate different metric sets: {gated}. Both cells "
        "measure the same thing under one mitigation difference, so a metric "
        "worth gating on one is worth gating on the other."
    )
    names = next(iter(gated.values()))

    for relative, documented in _documented_gate_sets().items():
        assert documented["gated"] == names, (
            f"{relative} says the nightly gates {sorted(documented['gated'])} "
            f"and config/ci/regression_baselines.yaml gates {sorted(names)}. "
            "Whichever is right, a reader of that document is being told the "
            "wrong thing about which metrics can red the run."
        )
        wrongly_listed = sorted(documented["record_only"] & names)
        assert not wrongly_listed, (
            f"{relative} lists {wrongly_listed} as record-only while "
            "config/ci/regression_baselines.yaml gates them."
        )

    # The specific claims that were false the moment the bless landed. Narrow on
    # purpose: a reworded sentence is not what this catches, an unchanged one is.
    stale = {
        "docs/tokenspeed-serving.md": "no serving baseline has been blessed",
        "docs/tokenspeed-gating-rollout.md": "nothing is gated, because no serving baseline",
        "scripts/ci/dashboard_metadata.py": "record-only until a baseline is blessed",
    }
    for relative, claim in stale.items():
        doc = (nightly_eval.REPO_ROOT / relative).read_text("utf-8")
        assert claim not in doc, (
            f"{relative} still says {claim!r} while "
            f"config/ci/regression_baselines.yaml gates {sorted(names)}."
        )


def _unquoted(doc: str) -> str:
    """``doc`` with markdown blockquote markers stripped, line by line.

    The current-state box is a blockquote, so its table's rows begin ``> |``
    and `_table_rows` -- which requires a line to start with ``|`` -- reads
    none of them. Stripping here rather than loosening `_table_rows` keeps
    that function's other callers reading exactly the tables they read today.
    """
    return "\n".join(
        line.lstrip()[1:].lstrip() if line.lstrip().startswith(">") else line
        for line in doc.splitlines()
    )


def _bullets(section: str) -> list[str]:
    """Top-level ``- `` items in ``section``, each with its indented continuation.

    A bullet ends at the first non-blank line that is not indented, so prose
    *after* the list is not absorbed into the last item. Splitting on ``\\n- ``
    instead gave the final bullet everything to the end of the section, which
    made a paragraph below the list look like part of it -- and a check that
    reads which names a bullet carries would then read names that belong to the
    commentary.
    """
    items: list[str] = []
    current: list[str] | None = None
    for line in section.splitlines():
        if line.startswith("- "):
            if current is not None:
                items.append("\n".join(current))
            current = [line]
        elif current is not None:
            if line.strip() and not line.startswith(" "):
                items.append("\n".join(current))
                current = None
            else:
                current.append(line)
    if current is not None:
        items.append("\n".join(current))
    return items


def _step_seven(doc: str) -> str:
    """Step 7's text, from its numbered heading to the next `##` section.

    Sliced rather than grepped for over the whole file because the claim being
    checked is about *where* a statement is. The current-state box and step 7
    are allowed to say different things about different metrics; what they may
    not do is give contradictory instructions about the same ones, and that is
    only visible if the two regions are read separately.
    """
    start = doc.index("**7. Promote the record-only metrics")
    end = doc.index("\n## ", start)
    return doc[start:end]


def test_step_seven_groups_the_nine_the_way_the_current_state_box_does():
    """The doc gave two answers about when step 7 may begin, and they disagreed.

    The header said the nine record-only metrics are "what the next window is
    for" while step 4, further down the same file, recorded that the completed
    window already clears the excursion blocker on `median_ttft_ms` and
    `output_throughput`. Following the header costs ten nights nobody needs; the
    two halves were added by different commits and nothing read them together.

    Fixing the header was not enough, and the first spelling of this test only
    checked the header. Step 7 went on saying "after another ten nightlies"
    and calling the `p99_*` metrics blocked on repeat data the window in fact
    recorded -- so the instruction a step-7 author actually follows still sent
    them back for nights nobody needs, and the tripwire passed because it was
    looking at one sentence somewhere else in the file.

    So this checks the partition rather than any sentence. The current-state
    box states three groups of the nine; step 7's bullets must name the same
    three, each whole and none merged. A reword passes. Splitting a group,
    merging two, or dropping a metric does not -- and neither does the
    original defect, because the old step 7 put `median_ttft_ms` and
    `output_throughput` in a bullet with the `p99_*` reasoning about
    excursions.
    """
    doc = (nightly_eval.REPO_ROOT / "docs/tokenspeed-gating-rollout.md").read_text("utf-8")

    groups = _table_rows(
        _unquoted(doc), ("record-only metric", "what step 7 is waiting for")
    )
    assert groups, (
        "the rollout doc has no `| record-only metric | what step 7 is waiting "
        "for |` box, so what step 7 must agree with cannot be read and this "
        "check would compare against nothing."
    )
    expected = [frozenset(_BACKTICKED.findall(row[0])) for row in groups]
    assert all(expected) and len(set(expected)) == len(expected), (
        f"the current-state box's groups are not distinct and non-empty: {expected}"
    )
    nine = frozenset().union(*expected)
    assert len(nine) == 9, (
        f"the box names {len(nine)} record-only metrics, not nine: {sorted(nine)}. "
        "Update this count with the rollout rather than around it."
    )

    bullets = _bullets(_step_seven(doc))
    assert bullets, "step 7 has no bullet list to read its groups out of"
    found = [g for g in (frozenset(_BACKTICKED.findall(b)) & nine for b in bullets) if g]
    assert sorted(map(sorted, found)) == sorted(map(sorted, expected)), (
        f"step 7 groups the nine as {sorted(map(sorted, found))} while the "
        f"current-state box groups them as {sorted(map(sorted, expected))}. The "
        "two have to give the same answer about the same metric: the box is "
        "what a reader sees first and step 7 is what a promoter acts on."
    )

    # The two instructions that were false while the header check above passed.
    # Kept as literals because they are the specific wrong things this document
    # said, and a grep is the honest way to say "not that again".
    for stale, why in (
        (
            "need another window before any of them can be promoted",
            "step 4 of the same document contradicts it for median_ttft_ms and "
            "output_throughput",
        ),
        (
            "After another ten nightlies",
            "the completed window already holds what two of the nine need, and "
            "step 7 must not open by scheduling ten more",
        ),
        (
            "blocked on having no repeat data",
            "the nightly harvests every allowlisted metric, so the window "
            "recorded the p99 series; what is missing is the analysis of it",
        ),
    ):
        assert stale not in doc, f"the rollout doc still says {stale!r}: {why}."

    assert "deferred to a separate PR" in doc, (
        "nothing records that median_ttft_ms and output_throughput are held "
        "back for attributability rather than for evidence, so a step-7 author "
        "cannot tell which of the nine are actually waiting on a measurement."
    )


def test_the_rollout_doc_does_not_still_say_the_window_is_outstanding():
    """Two rollout states in one document, four lines apart.

    The current-state box says the window was taken and the gate is armed; the
    paragraph immediately under it said "we do not yet have a window to derive
    thresholds from", which is the state before this whole PR. A reader who
    stops at the first paragraph after the box gets the wrong answer about
    whether there is anything to do.

    Narrow in both directions, like the step-7 check beside it: the present
    tense must be gone, and the argument it carried must still be there --
    it is the one step 7 has to satisfy for the nine, and it is the procedure
    for the next workload's first bless, so a rewrite that deletes it rather
    than re-tensing it loses something the document is for.
    """
    doc = (
        nightly_eval.REPO_ROOT / "docs/tokenspeed-gating-rollout.md"
    ).read_text("utf-8")
    stale = "we do not yet have a window to derive thresholds from"
    assert stale not in doc, (
        f"the rollout doc still says {stale!r} while its own current-state box "
        "records the completed 2026-09-08..09-17 window and two armed gates."
    )
    assert "A threshold derived from a single observation" in doc, (
        "the single-observation argument is gone; it is what step 7 still has "
        "to satisfy for the nine record-only metrics."
    )


def test_each_blessed_bound_is_its_windows_extremum_times_the_policys_margin():
    """The numbers themselves, not just the prose around them.

    The other tripwires here guard the *docs* against drifting from the
    baseline file. The four floats in it are the part that decides whether the
    nightly reds, and they were hand-written from a window table in a document
    -- so a transposed digit passed every test in the tree and would have been
    found by a nightly that stopped failing, or started.

    The window extrema are committed in `tokenspeed-gating-rollout.md`, so the
    derivation is checkable rather than merely stated: each bound must be its
    cell's and metric's window extremum times the margin its policy documents.
    That also makes the next hand-written bless show its working, which is the
    habit worth having rather than this particular set of four numbers being
    right.

    **Both policies, though only `max` is armed today.** Step 6 documents two
    derivations -- `max × 1.25` on the window maximum and `min × 0.85` on the
    window minimum -- and step 7 names `output_throughput`, a `min` metric, as
    one of the two whose evidence is already in hand. Asserting `policy ==
    "max"` for every armed key would have failed that promotion *even when its
    floor was derived exactly as the document says*, which makes this test an
    obstacle to the rollout it exists to protect. The policy is read from the
    baseline entry and picks the column and the multiplier; an unknown policy
    is a failure rather than a skip, so a third one cannot arrive unchecked.

    Both directions of coverage are asserted -- a gated key with no window row
    is a bound nothing sized, and a window row with no gated key is a metric
    the table measured and the bless silently dropped.
    """
    doc = (
        nightly_eval.REPO_ROOT / "docs/tokenspeed-gating-rollout.md"
    ).read_text("utf-8")
    for margin, policy in ((_BLESS_MARGIN, "max"), (_FLOOR_MARGIN, "min")):
        assert f"`{policy} × {margin}`" in doc, (
            f"the rollout doc no longer states the {policy} margin as "
            f"{policy} x {margin}, so the derivation checked below is not the "
            "one it documents."
        )

    rows = _table_rows(
        doc, ("cell", "metric", "min", "median", "max", "max/median", "full range")
    )
    assert rows, "the rollout doc has no per-cell window table to derive from"
    #: policy -> {(cell, metric): the extremum that policy's bound anchors on}.
    #: `min` anchors on the window minimum and `max` on the maximum: a floor is
    #: sized by the worst throughput seen, a ceiling by the worst latency.
    window = {
        "min": {(row[0].strip("`"), row[1].strip("`")): float(row[2]) for row in rows},
        "max": {(row[0].strip("`"), row[1].strip("`")): float(row[4]) for row in rows},
    }
    margins = {"max": _BLESS_MARGIN, "min": _FLOOR_MARGIN}

    gated = _gated_serving_metrics()
    measured = set(window["max"])
    armed = {(cell, name) for cell, names in gated.items() for name in names}
    assert armed == measured, (
        f"gated keys {sorted(armed)} and window rows {sorted(measured)} do not "
        "cover each other: a bound with no row was sized by nothing, and a row "
        "with no bound is a measurement the bless dropped without saying so."
    )

    import yaml

    baselines = yaml.safe_load(
        (nightly_eval.REPO_ROOT / "config/ci/regression_baselines.yaml").read_text("utf-8")
    )["baselines"]
    for cell, metric in sorted(armed):
        spec = baselines[f"tokenspeed_serve_smoke::{cell}"]["metrics"][metric]
        policy = spec["policy"]
        assert policy in margins, (
            f"{cell}/{metric} is policy {policy!r}, which step 6 of the "
            f"rollout doc gives no derivation for; it documents {sorted(margins)}. "
            "Add the rule there and the margin here rather than exempting the key."
        )
        anchor = window[policy][(cell, metric)]
        expected = round(anchor * margins[policy], 4)
        assert spec["value"] == expected, (
            f"{cell}/{metric} is policy {policy} and blessed at {spec['value']}, "
            f"but its window {policy} {anchor} x {margins[policy]} is {expected}."
        )


def test_the_docs_say_which_serving_metrics_are_not_gated():
    """"Gated" is only half the answer; the ungated set is the operational half.

    `step_time_ms.max` is the one that has to be named. It is what `--perf-gate`
    always writes, it is not a recorded metric at all, and it was pruned from
    this bless by hand -- so a doc that omits it reads as though the pruning did
    not happen, and the next hand-written bless puts it back.
    """
    gated = _gated_serving_metrics()
    names = next(iter(gated.values()))
    armed = sorted(n for n in names if n.split(".", 1)[0] == "step_time_ms")
    assert not armed, (
        f"{armed} is armed on a serving cell; the measured step-0 compile "
        "excursion (2825 ms) clears any ceiling derived from a clean night. "
        "See docs/tokenspeed-gating-rollout.md step 6."
    )
    for relative in ("docs/tokenspeed-serving.md", "docs/tokenspeed-gating-rollout.md"):
        doc = (nightly_eval.REPO_ROOT / relative).read_text("utf-8")
        assert "step_time_ms.max" in doc, (
            f"{relative} does not mention step_time_ms.max, so nothing records "
            "that it was left out on purpose."
        )
        assert "record-only" in doc, (
            f"{relative} no longer says which serving metrics stay record-only."
        )


#: What the completed 2026-09-08..09-17 window cleared for promotion: the
#: step-0 excursion was the only blocker on these two, and it did not recur in
#: twenty cell-runs. A fact about that window, so it stays true after step 7
#: promotes them.
_CLEARED_BY_THE_WINDOW = frozenset({"median_ttft_ms", "output_throughput"})

#: The step-time bound under both of its names -- the key `--perf-gate` writes
#: and the observation it is compared against. The excursion blocked it too,
#: but no window clears it: the bound is synthesised at bless time from one
#: night's mean, so there is no per-night series of it for a window to measure.
_STEP_TIME_BOUND = frozenset({"step_time_ms.max", "mean_step_time_ms"})

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_BREAK = re.compile(r"\s*(?:;|:(?!//)|—)\s*")

#: A clause saying the window cleared a metric: "clears the excursion blocker
#: on", "window cleared it", "clears X for promotion". Affirmative forms only.
#: English negates these as "does not clear", and the bare verb is kept out of
#: the match, so a sentence explaining why a metric is *not* cleared is not
#: read as claiming it is.
_CLEARED_CLAIM = re.compile(
    r"\bclear(?:s|ed)\b.*\b(?:blocker|for\s+promotion)\b|\bblocker\b.*\bclear(?:s|ed)\b"
)

#: A clause saying a metric can be promoted, by the same rule: "can be
#: promoted", "lets X be promoted", "evidence for promoting", and "pruned
#: until X" -- which says when the pruning ends. Not "cannot be promoted" and
#: not "promoting it needs".
_PROMOTABLE_CLAIM = re.compile(
    r"\b(?:can|could|should|may|will)\s+be\s+promoted\b"
    r"|\blets?\b.*\bbe\s+promoted\b"
    r"|\bfor\s+promotion\b"
    r"|\bevidence\s+for\s+promoting\b"
    r"|\bprune\w*\b.*\buntil\b"
)


def _prose_blocks(doc: str) -> list[list[list[str]]]:
    """``doc`` as blocks of sentences of clauses, so a claim can be read in place.

    A block is a paragraph, a bullet or one table row -- the unit a pronoun can
    reach back across. Blockquote markers and emphasis are dropped so the
    current-state box and `**bold**` leads read like any other prose; backticks
    are kept, because they carry the metric names.
    """
    blocks: list[str] = []
    current: list[str] = []

    def close() -> None:
        if current:
            blocks.append(" ".join(current))
            current.clear()

    for line in _unquoted(doc).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("|", "- ")):
            close()
        if stripped.startswith("|"):
            blocks.append(stripped)
        elif stripped:
            current.append(stripped)
    close()
    return [
        [_CLAUSE_BREAK.split(s) for s in _SENTENCE_BREAK.split(block.replace("*", ""))]
        for block in blocks
    ]


def _claims(doc: str, pattern: re.Pattern, universe: frozenset) -> list[tuple[str, frozenset]]:
    """``(clause, metrics it is about)`` for every clause ``pattern`` matches.

    A clause is about the ``universe`` names it spells out, or -- when it says
    "it", "them" or "their" -- about the names in the nearest earlier clause of
    the same block that spelled any out. That fallback is the whole reason
    this reads blocks rather than grepping lines: "The window cleared its
    excursion blocker too" names nothing, and was false because of the
    sentence before it.
    """
    found: list[tuple[str, frozenset]] = []
    for block in _prose_blocks(doc):
        about: frozenset = frozenset()
        for sentence in block:
            for clause in sentence:
                named = frozenset(_BACKTICKED.findall(clause)) & universe
                if named:
                    about = named
                if pattern.search(clause):
                    found.append((clause, about))
    return found


def test_the_rollout_doc_says_the_window_cleared_exactly_two_metrics():
    """The window cleared `median_ttft_ms` and `output_throughput`, and nothing else.

    The rollout doc said otherwise in nine places. Three made the step-time
    bound cleared or promotable -- "clears the *excursion* blocker on
    `median_ttft_ms`, `output_throughput` and `step_time_ms.max`", "The window
    cleared its excursion blocker too" with the name one sentence earlier, and
    "prune it by hand until `warmup_steps` is proven to cover the excursion" --
    and six more said "the three" where the set is two. Each sat near text
    explaining why `step_time_ms.max` is the exception, so a reader got both
    answers a paragraph apart: it is not a recorded metric, and a clean window
    has no series of it to clear.

    Read as claims, not as vocabulary. Every clause asserting that the window
    cleared a metric, or that a metric can be promoted, is resolved to the
    metrics it is about; the cleared set must be exactly the two, and neither
    kind of claim may be about the step-time bound. Sentences that say why it
    is *not* cleared use the negated verb and are not claims. A bare count is
    checked separately, because "the three can be promoted" names nothing to
    resolve.
    """
    doc = (nightly_eval.REPO_ROOT / "docs/tokenspeed-gating-rollout.md").read_text("utf-8")

    rows = _table_rows(doc, ("Metric", "Policy", "First bless", "Why"))
    dropped = {n for row in rows if "(was: gate)" in row[2] for n in _BACKTICKED.findall(row[0])}
    assert dropped == _CLEARED_BY_THE_WINDOW | _STEP_TIME_BOUND, (
        f"the per-metric table marks {sorted(dropped)} as dropped from the "
        "original gate set; this check assumes the excursion blocked "
        "median_ttft_ms, output_throughput and the step-time bound. Update the "
        "constants with the table rather than around it."
    )
    universe = frozenset(dropped)

    cleared = frozenset().union(*(about for _, about in _claims(doc, _CLEARED_CLAIM, universe)))
    assert cleared == _CLEARED_BY_THE_WINDOW, (
        f"the rollout doc says the window cleared {sorted(cleared)}; it cleared "
        f"{sorted(_CLEARED_BY_THE_WINDOW)}. step_time_ms.max is not a recorded "
        "metric, so there is no per-night series of it for a window to clear."
    )

    for pattern in (_CLEARED_CLAIM, _PROMOTABLE_CLAIM):
        for clause, about in _claims(doc, pattern, universe):
            assert not about & _STEP_TIME_BOUND, (
                f"the rollout doc says the step-time bound is cleared or "
                f"promotable from the window: {clause!r}. It is synthesised at "
                "bless time from one night's mean, and step 4 and step 7 both "
                "say no window measures it."
            )

    promotion = re.compile(
        f"{_CLEARED_CLAIM.pattern}|{_PROMOTABLE_CLAIM.pattern}|\\bstay\\s+record-only\\b"
    )
    for clause, _ in _claims(doc, promotion, universe):
        assert not re.search(r"\b(?:the|all)\s+three\b", clause), (
            f"the rollout doc counts the metrics the window decides as three: "
            f"{clause!r}. It decides two; name them."
        )


#: How the rollout doc said some of the nine were waiting on nights not yet
#: run: a count of groups that are waiting, or a count of metrics for which
#: "the next window" is the right answer or "another window" the wrong one.
#: Parameterised over the count so a reworded "two groups are waiting" is
#: caught as well as the original "only one group is waiting". A count that is
#: itself the object of "of the" is not the subject -- "none of the three
#: groups is waiting" says zero -- so it is excluded by look-behind.
_NUMBER = r"(?:one|two|three|four|five|six|seven|eight|nine|some)"
_WAITING_ON_NEW_NIGHTS = re.compile(
    rf"(?<!of the )\b(?:only\s+)?{_NUMBER}\s+"
    rf"(?:of\s+(?:them|the\s+nine|the\s+(?:{_NUMBER}\s+)?groups)\s+|groups?\s+)"
    r"(?:is|are)\s+waiting\b"
    rf"|\bright\s+answer\s+for\s+(?:only\s+)?{_NUMBER}\b"
    rf"|\bwrong\s+answer\s+for\s+{_NUMBER}\s+of\b",
    re.I,
)


def test_the_rollout_doc_does_not_say_any_group_waits_for_new_nights():
    """None of the three groups waits on nights that have not happened yet.

    The current-state box and step 7 agree on that group by group: two
    metrics have their evidence, three need the *analysis* of the recorded
    window and a second one only if that analysis says so, four are held on
    redundancy. Three summary sentences disagreed with them -- "'the next
    window' is the right answer for only three of them", "the wrong answer for
    six of the nine", and step 7's own opening, "only one group is waiting on
    nights that have not happened yet" -- each implying the `p99_*` group
    should wait before reading what it already has.

    The partition check above cannot see this: the groups were right and the
    sentences summarising them were wrong. So this reads every sentence for a
    count of groups or metrics described as waiting. It is a pattern over how
    this document quantifies waiting, not a parse of meaning, and says so.
    """
    doc = (nightly_eval.REPO_ROOT / "docs/tokenspeed-gating-rollout.md").read_text("utf-8")
    for block in _prose_blocks(doc):
        for clauses in block:
            sentence = " ".join(clauses)
            assert not _WAITING_ON_NEW_NIGHTS.search(sentence), (
                f"the rollout doc says part of the nine is waiting on new "
                f"nights: {sentence!r}. The current-state box and step 7 give "
                "every group a way forward from the recorded window."
            )


#: Opposing a failure to a chart: "not a chart", "rather than a chart", "not
#: charted", "instead of a chart entry".
_NOT_CHARTED = re.compile(
    r"\b(?:not|rather\s+than|instead\s+of)\s+(?:a\s+)?chart(?:s|ed|\s+entry|\s+entries)?\b",
    re.I,
)


def test_the_docs_say_a_breach_fails_the_nightly_and_stays_charted():
    """A gate breach is a verdict, and the observation behind it stays on the chart.

    Three places said a run over a serving ceiling is "a nightly failure, not a
    chart" (or "rather than a chart", or "not a chart entry"). The failure half
    is right. The other half is not -- see the next test for the path -- and it
    sends an operator looking for the breached value somewhere other than the
    metric history it is drawn in.

    Two halves, per text: no sentence about a failure opposes it to charting,
    and some sentence about a failure says the observation is still charted,
    so deleting the clause is not a way to pass. The dashboard text is read
    from `DASHBOARD_METADATA` rather than from source, because the source
    splits it across string literals and a grep would read the seams.
    """
    dashboard_metadata = _load("dashboard_metadata")
    texts = {
        relative: (nightly_eval.REPO_ROOT / relative).read_text("utf-8")
        for relative in ("docs/tokenspeed-gating-rollout.md", "docs/tokenspeed-serving.md")
    }
    texts["scripts/ci/dashboard_metadata.py (tokenspeed_serve_smoke success criteria)"] = (
        dashboard_metadata.DASHBOARD_METADATA["workloads"]["tokenspeed_serve_smoke"]["repro"][
            "success_criteria"
        ]
    )
    for where, text in texts.items():
        failures = [
            " ".join(clauses)
            for block in _prose_blocks(text)
            for clauses in block
            if re.search(r"\bfail", " ".join(clauses))
        ]
        conflated = [s for s in failures if _NOT_CHARTED.search(s)]
        assert not conflated, (
            f"{where} opposes a gate failure to charting: {conflated[0]!r}. "
            "nightly_eval.py records the failing cell with its metrics and "
            "gen_dashboard.py charts every entry, so the breach is both."
        )
        assert any(re.search(r"\bcharted\b", s) for s in failures), (
            f"{where} no longer says that a breached observation stays charted; "
            "an operator reading it cannot tell where to find the value that "
            "failed the run."
        )


def test_a_serving_gate_breach_fails_the_nightly_and_stays_charted(tmp_path, monkeypatch):
    """What the docs now say a breach does, checked on the path that does it.

    A night over the `median_tpot_ms` ceiling is driven through the real
    `nightly_eval.main()` against the committed baselines, and its results
    file through the real dashboard renderer beside a clean night's. The run
    must exit non-zero, the results file must still carry the breached value,
    and that cell's `median_tpot_ms` history must be the chart of both nights.
    A harness that drops a failing cell's metrics, or a history that skips
    failed nights, fails this -- and would make the prose wrong again in the
    other direction.

    The results file reaches the dashboard through two workflow steps that
    are YAML rather than Python -- the artifact upload and the publisher --
    and both have to run on a failed eval for any of this to hold, so they are
    asserted too.
    """
    import sys

    import yaml

    gen_dashboard = _load("gen_dashboard")
    workflows = nightly_eval.REPO_ROOT / ".github/workflows"
    publish = yaml.safe_load((workflows / "nightly-eval.yml").read_text("utf-8"))["jobs"]["publish"]
    assert "always()" in publish["if"], (
        "nightly-eval.yml's publish job no longer runs after a failed eval, so a "
        "breaching night never reaches the history the docs say it is charted in."
    )
    uploads = [
        step
        for step in yaml.safe_load((workflows / "eval-reusable.yml").read_text("utf-8"))["jobs"][
            "eval"
        ]["steps"]
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
        and "gpu-nightly-results.json" in step["with"]["path"]
    ]
    assert [step.get("if") for step in uploads] == ["always()"], (
        "eval-reusable.yml does not upload gpu-nightly-results.json after a failed "
        "eval, so the publisher has nothing to record."
    )

    baselines = yaml.safe_load(nightly_eval.BASELINES.read_text("utf-8"))["baselines"]
    ceiling = baselines["tokenspeed_serve_smoke::baseline"]["metrics"]["median_tpot_ms"]["value"]
    healthy, breach = 1.8906, round(ceiling * 1.5, 4)
    entry = {e["name"]: e for e in _real_matrix()["entries"]}["tokenspeed_serve_smoke"]

    real_evaluate = nightly_eval.evaluate
    monkeypatch.setattr(
        nightly_eval, "evaluate",
        lambda _matrix, blessed, work: real_evaluate({"entries": [entry]}, blessed, work),
    )
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(nightly_eval, "docker_daemon", lambda: (True, "stubbed"))
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {"amd_aorta_version": "x"})

    def night(label: str, tpot: float) -> tuple[int, dict]:
        def fake_run_entry(e, out_dir):
            mpath = out_dir / e["name"] / "matrix.json"
            _write_matrix(mpath, [
                {"name": cell, "error": None, "passed_count": 1, "failed_count": 0,
                 "error_count": 0, "mean_step_time_ms": 1130.0,
                 "metrics_summary": {"median_tpot_ms": {"mean": value},
                                     "p99_itl_ms": {"mean": 34.75}}}
                for cell, value in (("baseline", tpot), ("no-scratch-reclaim", healthy))
            ])
            return 0, mpath, False

        out = tmp_path / f"{label}.json"
        monkeypatch.setattr(nightly_eval, "run_entry", fake_run_entry)
        monkeypatch.setattr(
            sys, "argv",
            ["nightly_eval.py", "--out", str(out), "--work-dir", str(tmp_path / label)],
        )
        rc = nightly_eval.main()
        doc = json.loads(out.read_text("utf-8"))
        doc["generated_at"] = f"2026-09-{label}T11:30:00+00:00"
        return rc, doc

    clean_rc, clean = night("18", healthy)
    breach_rc, breached = night("19", breach)

    assert clean_rc == 0 and clean["summary"]["fail"] == 0, clean["summary"]
    assert breach_rc == 1, "a night over the median_tpot_ms ceiling did not fail the nightly"
    assert breached["summary"]["fail"] == 1, breached["summary"]
    cell = {e["cell"]: e for e in breached["entries"]}["baseline"]
    assert cell["verdict"] == "fail"
    assert any("median_tpot_ms" in reason for reason in cell["reasons"]), cell["reasons"]
    assert cell["metrics"]["summary"]["median_tpot_ms"] == breach, (
        "the failing cell reached the results file without the value that failed it"
    )

    html = gen_dashboard.build_dashboard_html([clean, breached])
    chart = gen_dashboard._svg_sparkline([healthy, breach])
    assert chart.startswith("<svg") and chart in html, (
        "the dashboard does not chart the breaching night in the cell's "
        "median_tpot_ms history; the docs say it does."
    )


def _heading_anchors(doc: str) -> set[str]:
    """The ``#fragment`` GitHub gives each heading of ``doc``.

    GitHub's rule: lowercase, drop every character that is not a letter, a
    digit, a space, ``-`` or ``_``, turn spaces into ``-``, and suffix a
    repeated slug ``-1``, ``-2``. Lines inside fenced code are skipped, because
    the rollout doc's shell blocks are full of ``# comment`` lines that are not
    headings and would otherwise mint anchors that do not exist.
    """
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    fenced = False
    for line in doc.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        match = None if fenced else re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not match:
            continue
        slug = "".join(
            c for c in match.group(1).lower()
            if c in " -_" or unicodedata.category(c)[0] in "LMN"
        ).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def test_the_gating_docs_link_only_to_anchors_they_define():
    """Every in-page ``](#...)`` link lands on a heading of the same file.

    Renaming "What blocks this today" to "What blocked this, and what remains"
    left the current-state notes linking to ``#what-blocks-this-today``, which
    no longer exists; GitHub renders that as a link that scrolls nowhere. The
    two gating docs cross-reference their own sections heavily -- the rollout
    doc has 25 such links -- and every heading here gets reworded as the
    rollout moves, so the anchors are checked rather than trusted.
    """
    for relative in ("docs/tokenspeed-gating-rollout.md", "docs/tokenspeed-serving.md"):
        doc = (nightly_eval.REPO_ROOT / relative).read_text("utf-8")
        anchors = _heading_anchors(doc)
        links = re.findall(r"\]\(#([^)\s]+)\)", doc)
        assert links, f"{relative} has no in-page links, so this checks nothing"
        dead = sorted({link for link in links if link not in anchors})
        assert not dead, (
            f"{relative} links to {dead}, which no heading in it defines. Point "
            "the link at the heading's current anchor, or keep the old one as an "
            "alias."
        )


def test_the_heading_anchors_follow_githubs_rule():
    """The slugger above agrees with GitHub on the cases these docs contain.

    Without this, a slugger that returned every link's own fragment would pass
    the dead-link test for any doc at all.
    """
    doc = "\n".join([
        "## What blocked this, and what remains",
        "#### One MI350X cell at `warmup_steps: 2` (2026-09-03) — a data point",
        "```bash",
        "# not a heading",
        "```",
        "## Repeated",
        "## Repeated",
    ])
    assert _heading_anchors(doc) == {
        "what-blocked-this-and-what-remains",
        "one-mi350x-cell-at-warmup_steps-2-2026-09-03--a-data-point",
        "repeated",
        "repeated-1",
    }


def test_the_rollout_doc_does_not_say_the_window_is_still_to_come():
    """The ten-night window at ``warmup_steps: 2`` has been taken; say so.

    Two places still spoke of it as future after the PR recorded it and blessed
    from it: the ``warmup_steps`` paragraph ("the ten-night record-only window
    has to be taken afresh ... before anything is blessed") and the warmup-1
    caveat ("there is no measurement at `warmup_steps: 2` yet"). Each sat a
    screen away from the step-4 text reporting that window's numbers. The
    patterns are the future-tense forms this doc used; the paired assertion
    that the taken window is still named keeps a deletion from passing. Read
    with whitespace collapsed rather than through `_prose_blocks`, which
    splits clauses on ``:`` and so would cut `warmup_steps: 2` in half.
    """
    doc = (nightly_eval.REPO_ROOT / "docs/tokenspeed-gating-rollout.md").read_text("utf-8")
    prose = " ".join(_unquoted(doc).split())
    stale = re.compile(
        r"\bwindow\s+(?:still\s+)?(?:has|have|needs?)\s+to\s+be\s+taken\b"
        r"|\bno\s+measurement\s+at\s+`warmup_steps:\s*2`\s+yet\b"
        r"|\bten-night\s+window\s+is\s+what\s+produces\s+it\b",
        re.I,
    )
    found = stale.search(prose)
    assert not found, (
        f"the rollout doc still speaks of the warmup_steps: 2 window as future: "
        f"{found.group(0)!r}. It is 2026-09-08..09-17, and the gates are blessed "
        "from it."
    )
    assert re.search(r"window at `warmup_steps: 2`,?\s+2026-09-08\.\.09-17", prose), (
        "the warmup-1 caveat no longer says where the warmup_steps: 2 measurement is"
    )


def test_pending_entries_are_valid_and_loadable():
    """Validated exactly like a live entry, so promoting one is a move, not a bet.

    A staged entry that names a deleted recipe or a misspelled field would
    otherwise only be discovered by the first red nightly after promotion.
    """
    from aorta.triage.recipe import load_recipe

    known_fields = {"name", "recipe", "nproc", "min_gpus", "timeout_sec",
                    "needs_docker_daemon", "blocked_on"}
    pending = _real_matrix().get("pending_entries") or []
    live = {e["name"] for e in _real_matrix()["entries"]}

    for entry in pending:
        assert set(entry) <= known_fields, f"{entry['name']}: {set(entry) - known_fields}"
        assert entry["name"] not in live, f"{entry['name']} is both staged and live"
        # A staged entry exists because it cannot run yet; say why, in the file.
        assert entry.get("blocked_on"), f"{entry['name']} has no blocked_on"
        path = nightly_eval.REPO_ROOT / entry["recipe"]
        assert path.is_file(), entry["recipe"]
        recipe = load_recipe(path)
        assert recipe.cells, f"{entry['name']}: recipe has no cells"
        # Every entry's budget must cover the recipe it names; the default is
        # 1800s and a serving bring-up alone has been measured at 379s a cell.
        assert int(entry.get("timeout_sec", 1800)) >= 1800


def test_serving_entry_metric_names_resolve_against_the_policy_table():
    """The names `tokenspeed bench serve` exports, checked against the allowlist.

    The workload passes its export through verbatim, so a metric the plan intends
    to gate must resolve to the right direction -- and the ones it intends never
    to gate must resolve to nothing, since an allowlist entry is what --perf-gate
    arms from.
    """
    eval_lib = _load("eval_lib")
    live = {e["name"]: e for e in _real_matrix()["entries"]}
    assert "tokenspeed_serve_smoke" in live

    # Gated from the first bless (docs/tokenspeed-gating-rollout.md). Both are
    # per-token metrics: they are measured between tokens, so the step-0 compile
    # excursion measured on this cell does not enter them.
    for name in ("median_tpot_ms", "p99_itl_ms"):
        assert eval_lib.metric_policy(name) == "max", name
        assert eval_lib.is_auto_gateable(name) is True, name

    # Gateable and correctly directed, but held record-only for the first bless
    # because all three are duration-derived or wait on the compile, and the
    # excursion carries each of them through its threshold.
    assert eval_lib.metric_policy("median_ttft_ms") == "max"
    assert eval_lib.metric_policy("output_throughput") == "min"
    for name in ("median_ttft_ms", "output_throughput"):
        assert eval_lib.is_auto_gateable(name) is True, name

    # Recorded but never gated: bring-up time is the noisiest thing the workload
    # reports (189-379s on one node, nothing changed), and the counters/totals
    # restate the recipe rather than measuring the stack.
    for name in ("server_startup_sec", "container_elapsed_sec", "duration",
                 "completed_total", "failed_total", "total_output_tokens",
                 "max_output_tokens_per_s", "max_concurrent_requests",
                 "mean_ttft_ms", "std_ttft_ms", "p50_ttft_ms", "p90_ttft_ms"):
        assert eval_lib.metric_policy(name) is None, name

    # Gateable, but not armed by a refresh.
    assert eval_lib.is_auto_gateable("median_itl_ms") is False


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


# ---------------------------------------------------------------------------
# Lane + base-image attribution for the latest-ROCm canary (issue #382)
# ---------------------------------------------------------------------------


def test_metadata_defaults_to_the_gate_lane(tmp_path, monkeypatch):
    """Existing callers keep describing themselves correctly, unchanged."""
    monkeypatch.delenv("AORTA_CI_LANE", raising=False)
    monkeypatch.delenv("AORTA_CI_BASE_IMAGE", raising=False)
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(tmp_path / "absent"))
    meta = nightly_eval.build_metadata()
    assert meta["lane"] == "gate"
    # The gated lane's digest is pinned in the Dockerfile and visible in review,
    # so there is nothing to record here.
    assert meta["base_image"] is None


def test_metadata_records_the_canary_lane_and_resolved_digest(tmp_path, monkeypatch):
    """"Ran :latest" is not attributable; the resolved digest is (#382)."""
    base = "rocm/pytorch:latest@sha256:" + "ab" * 32
    monkeypatch.setenv("AORTA_CI_LANE", "canary")
    monkeypatch.setenv("AORTA_CI_BASE_IMAGE", base)
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(tmp_path / "absent"))
    meta = nightly_eval.build_metadata()
    assert meta["lane"] == "canary"
    assert meta["base_image"] == base


def test_empty_lane_env_falls_back_to_gate(tmp_path, monkeypatch):
    """An exported-but-empty var must not produce a row labelled "" ."""
    monkeypatch.setenv("AORTA_CI_LANE", "")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(tmp_path / "absent"))
    assert nightly_eval.build_metadata()["lane"] == "gate"


def test_an_empty_version_does_not_shadow_a_valid_version_dev(tmp_path, monkeypatch):
    """An interrupted install leaves a zero-byte marker (#387).

    The loop used to break on mere existence, so the empty file won and the
    dashboard's ROCm column went blank with a perfectly good `version-dev`
    sitting behind it -- indistinguishable from having no ROCm at all.
    """
    root = tmp_path / "opt_rocm"
    _write_version(root, "version", "")
    _write_version(root, "version-dev", "7.2.4.50311-abc1234\n")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    assert nightly_eval.build_metadata()["rocm"] == "7.2.4.50311-abc1234"


def test_a_whitespace_only_version_does_not_shadow_a_valid_one(tmp_path, monkeypatch):
    root = tmp_path / "opt_rocm"
    _write_version(root, "version", "   \n\t\n")
    _write_version(root, "version-dev", "7.14.0\n")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    assert nightly_eval.build_metadata()["rocm"] == "7.14.0"


def test_an_unreadable_first_marker_falls_through_instead_of_raising(tmp_path, monkeypatch):
    """A directory named `.info/version` must not take down build_metadata().

    read_text() raised IsADirectoryError straight out of build_metadata, and
    nothing above it catches -- so the whole results document was lost over a
    cosmetic dashboard column. Same for a permission-denied marker.
    """
    root = tmp_path / "opt_rocm"
    (root / ".info" / "version").mkdir(parents=True)
    _write_version(root, "version-dev", "7.2.4\n")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    assert nightly_eval.build_metadata()["rocm"] == "7.2.4"


def test_a_non_utf8_marker_is_null_not_a_crash(tmp_path, monkeypatch):
    """Undecodable is unusable -- but it must not raise, either.

    Null matches what `environment.py` reports for the same file, so the
    dashboard column and `rocm.version` agree. The point of the test is that
    the read is fail-soft: `read_text()` raised `UnicodeDecodeError`, which is
    not an `OSError` and so escaped `build_metadata()` entirely.
    """
    root = tmp_path / "opt_rocm"
    info = root / ".info"
    info.mkdir(parents=True)
    (info / "version").write_bytes(b"\xff\xfe7.2.4")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    assert nightly_eval.build_metadata()["rocm"] is None


def test_a_non_utf8_marker_does_not_shadow_a_valid_one(tmp_path, monkeypatch):
    root = tmp_path / "opt_rocm"
    info = root / ".info"
    info.mkdir(parents=True)
    (info / "version").write_bytes(b"\xff\xfe")
    _write_version(root, "version-dev", "7.2.4\n")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    assert nightly_eval.build_metadata()["rocm"] == "7.2.4"


def test_a_multibyte_version_is_not_reported_as_corrupt(tmp_path, monkeypatch):
    """The bounded read must not turn a split character into "unreadable".

    Contrived content, but it pins the reason the decode is incremental: a
    plain bytes.decode() on a truncated buffer raises, which would report a
    perfectly readable marker as non-UTF-8.
    """
    root = tmp_path / "opt_rocm"
    _write_version(root, "version", "7.2.4-" + "\u00e9" * 3000)
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    value = nightly_eval.build_metadata()["rocm"]
    assert value is not None and value.startswith("7.2.4-")
