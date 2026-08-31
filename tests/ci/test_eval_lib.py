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


def test_serving_metrics_are_gateable():
    """The `tokenspeed_serve` workload emits these names verbatim from
    `tokenspeed bench serve`. Unlisted metrics are never gated, so a nightly
    would silently record serving regressions instead of failing on them."""
    for name in (
        "median_ttft_ms",
        "p99_ttft_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "median_itl_ms",
        "p99_itl_ms",
        "median_e2el_ms",
        "p99_e2el_ms",
    ):
        assert eval_lib.metric_policy(name) == "max", name
    for name in ("output_throughput", "total_token_throughput", "request_throughput"):
        assert eval_lib.metric_policy(name) == "min", name


def test_median_itl_is_gateable_by_hand_but_never_auto_gated():
    """Gateable and auto-gateable are different questions for a metric at ~0.

    `median_itl_ms` stays in the allowlist so a baseline that names it is still
    enforced; what it must not do is get a bound derived from a margin. See
    `_NO_AUTO_GATE`.
    """
    assert eval_lib.metric_policy("median_itl_ms") == "max"
    assert eval_lib.is_performance_metric("median_itl_ms") is True
    assert eval_lib.is_auto_gateable("median_itl_ms") is False


def test_the_no_auto_gate_set_is_narrow():
    """Every other allowlisted perf metric is still auto-gateable, and the set
    never claims a metric that is not gateable in the first place."""
    for name in ("gflops", "tokens_per_sec", "median_ttft_ms", "median_tpot_ms",
                 "p99_itl_ms", "output_throughput", "total_token_throughput"):
        assert eval_lib.is_auto_gateable(name) is True, name
    for name in ("logits_checksum", "server_startup_sec", "unknown_metric"):
        assert eval_lib.is_auto_gateable(name) is False, name
    assert all(eval_lib.is_performance_metric(n) for n in eval_lib._NO_AUTO_GATE)


# ---------------------------------------------------------------------------
# Offline gating simulation for the TokenSpeed serving rollout.
#
# The numbers below are the ones docs/tokenspeed-serving.md and docs/tokenspeed.md
# actually recorded on gfx950, not invented ones. They exist as tests because the
# rollout plan (docs/tokenspeed-gating-rollout.md) makes claims about which
# metrics tolerate the measured noise and which do not, and a claim about a
# comparator is checkable against the comparator.
#
# `_MAX_MARGIN` / `_MIN_MARGIN` are refresh_baselines.py's defaults, restated
# here so a change to them fails these tests rather than silently invalidating
# the plan.
# ---------------------------------------------------------------------------

_MAX_MARGIN = 0.25   # --step-time-margin: latency ceiling = value * 1.25
_MIN_MARGIN = 0.15   # --throughput-margin: throughput floor = value * 0.85


def _bound(value: float, policy: str) -> float:
    return round(value * (1 + _MAX_MARGIN), 4) if policy == "max" \
        else round(value * (1 - _MIN_MARGIN), 4)


def _gate(observed: float, name: str, threshold: float) -> str:
    """Run one metric through the real comparator and return its verdict."""
    harvested = {"passed": True, "error": None, "metrics": {"summary": {name: observed}}}
    baseline = {"passed": True,
                "metrics": {name: {"policy": eval_lib.metric_policy(name), "value": threshold}}}
    return eval_lib.compare_to_baseline(harvested, baseline)["verdict"]


# serve-models::qwen3-0.6b vs serve-load::conc-8 -- two separate sweeps that ran
# a byte-identical measurement configuration (Qwen3-0.6B, ISL 512 / OSL 128,
# concurrency 8, 32 prompts, 3 measured steps, 1 warmup step, ignore_eos, seed 0).
_QWEN_RUN_A = {"median_ttft_ms": 46.3, "median_tpot_ms": 1.94, "output_throughput": 3538.0}
_QWEN_RUN_B = {"median_ttft_ms": 45.9, "median_tpot_ms": 1.91, "output_throughput": 3631.0}

# serve-gptoss::baseline vs serve-gptoss-tp::tp1, which the doc calls the control
# for the TP axis ("reproduces the single-GPU numbers to within a percent").
_GPTOSS_RUN_A = {"median_ttft_ms": 67.1, "median_tpot_ms": 7.61, "output_throughput": 994.0}
_GPTOSS_RUN_B = {"median_ttft_ms": 67.2, "median_tpot_ms": 7.63, "output_throughput": 991.0}

# Startup to /health, same recipe, same node, nothing changed between them
# (docs/tokenspeed.md: "189, 276, 285, 291, 316 and 319 seconds across six runs
# ... a 1.7x spread"), plus the 379 s the multi-model sweep recorded for the
# same 0.6B model.
_STARTUP_SEC = [189, 276, 285, 291, 316, 319, 379]


@pytest.mark.parametrize("run_a,run_b", [(_QWEN_RUN_A, _QWEN_RUN_B),
                                         (_GPTOSS_RUN_A, _GPTOSS_RUN_B)])
def test_measured_run_to_run_noise_passes_the_proposed_gates(run_a, run_b):
    """The metrics the plan gates tolerate the noise we have actually measured.

    Both directions: whichever of the two runs is blessed, the other must pass.
    A gate that depends on which night it was armed is a coin flip, not a gate.
    """
    for blessed, observed in ((run_a, run_b), (run_b, run_a)):
        for name, value in blessed.items():
            policy = eval_lib.metric_policy(name)
            verdict = _gate(observed[name], name, _bound(value, policy))
            assert verdict == "pass", f"{name}: bless {value} -> {observed[name]} {verdict}"


def test_a_real_serving_regression_fails_the_proposed_gates():
    """A regression the size of a genuine stack change is caught.

    Scale reference from the same table: moving Qwen3-0.6B to Qwen3-4B took TPOT
    1.94 -> 3.77 ms and output throughput 3538 -> 1905 tok/s. A gate that cannot
    see that is not worth arming.
    """
    tpot_ceiling = _bound(max(_QWEN_RUN_A["median_tpot_ms"], _QWEN_RUN_B["median_tpot_ms"]), "max")
    thru_floor = _bound(min(_QWEN_RUN_A["output_throughput"],
                            _QWEN_RUN_B["output_throughput"]), "min")

    assert _gate(1.94 * 1.30, "median_tpot_ms", tpot_ceiling) == "fail"
    assert _gate(3538 * 0.75, "output_throughput", thru_floor) == "fail"
    assert _gate(3.77, "median_tpot_ms", tpot_ceiling) == "fail"
    assert _gate(1905.0, "output_throughput", thru_floor) == "fail"

    # And the band between measured noise (<=2.7%) and the gate is deliberately
    # left to the dashboard's 10% move detector, not to the gate: a 10% drift
    # passes here and is reported as "what changed" instead of failing the job.
    assert _gate(1.94 * 1.10, "median_tpot_ms", tpot_ceiling) == "pass"
    assert _gate(3538 * 0.90, "output_throughput", thru_floor) == "pass"


def test_a_startup_time_gate_would_fire_on_measured_noise():
    """Why `server_startup_sec` is absent from the allowlist and must stay absent.

    Arming it from a single observation -- which is what --perf-gate does -- makes
    the verdict depend on which night was blessed: from the fastest of the seven
    known runs, every other run breaches.
    """
    assert eval_lib.metric_policy("server_startup_sec") is None

    def breaches(threshold):
        return [v for v in _STARTUP_SEC
                if v > threshold]  # policy would be `max`

    assert breaches(_bound(min(_STARTUP_SEC), "max")) == [276, 285, 291, 316, 319, 379]
    # Four of the seven possible blessing nights produce a gate that breaches at all.
    breaking = [v for v in _STARTUP_SEC
                if [o for o in _STARTUP_SEC if o != v and o > _bound(v, "max")]]
    assert len(breaking) == 4


def test_a_near_zero_itl_bound_cannot_be_satisfied():
    """The concrete reason `median_itl_ms` is in `_NO_AUTO_GATE`.

    The docs record it as sitting near zero ("most recorded gaps are ~0"). A
    margin is multiplicative, so an observation of 0.0 blesses a ceiling of 0.0
    and every later run with any inter-token gap at all fails.
    """
    assert _bound(0.0, "max") == 0.0
    assert _gate(0.001, "median_itl_ms", _bound(0.0, "max")) == "fail"
    # Even a plainly non-zero-but-small observation gates on 0.01 ms of movement.
    assert _gate(0.03, "median_itl_ms", _bound(0.02, "max")) == "fail"


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


def test_compare_expected_failure_baseline():
    baseline = {"passed": False}
    # Expected to fail and did -> pass.
    failed = {"passed": False, "error": "boom", "metrics": {"summary": {}}}
    assert eval_lib.compare_to_baseline(failed, baseline)["verdict"] == "pass"
    # Expected to fail but passed -> fail (stale expected-failure baseline).
    passed = {"passed": True, "error": None, "metrics": {"summary": {}}}
    assert eval_lib.compare_to_baseline(passed, baseline)["verdict"] == "fail"


def test_compare_invalid_policy_fails():
    harvested = {"passed": True, "error": None, "metrics": {"summary": {"x": 1.0}}}
    baseline = {"passed": True, "metrics": {"x": {"policy": "bogus", "value": 1.0}}}
    out = eval_lib.compare_to_baseline(harvested, baseline)
    assert out["verdict"] == "fail"
    assert any("invalid" in r or "policy" in r for r in out["reasons"])


def test_metric_classification():
    assert eval_lib.is_correctness_metric("logits_checksum") is True
    assert eval_lib.is_correctness_metric("gflops") is False
    assert eval_lib.is_performance_metric("gflops") is True
    assert eval_lib.is_performance_metric("logits_checksum") is False
    assert eval_lib.is_performance_metric("unknown_metric") is False


def test_summarize_counts_verdicts():
    entries = [{"verdict": "pass"}, {"verdict": "fail"}, {"verdict": "record"}, {"verdict": "skip"}]
    s = eval_lib.summarize(entries)
    assert s == {"total": 4, "pass": 1, "fail": 1, "record": 1, "skip": 1}
