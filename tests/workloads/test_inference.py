"""Tests for the `inference` workload.

Config parsing/validation runs without torch. Model construction, the offline
and continuous-batch smokes, the WorkloadResult schema, and the NaN check
require torch and are skipped where it is unavailable. No test needs a GPU; the
smokes run on CPU.
"""

from __future__ import annotations

import pytest

from aorta.workloads.inference import (
    InferenceConfig,
    InferenceWorkload,
    ModelSpec,
    _percentile,
)

# --------------------------------------------------------------------------- #
# Config parsing / validation (torch-free)
# --------------------------------------------------------------------------- #


def test_config_defaults() -> None:
    cfg = InferenceConfig.from_dict({})
    assert cfg.mode == "offline_batch"
    assert cfg.model.kind == "decoder_transformer"
    assert cfg.request.batch_size == 4
    assert cfg.serving.kv_cache is True
    assert cfg.checks.fail_on_nan_logits is True
    assert cfg.is_autoregressive is True


def test_config_nested_sections_parsed() -> None:
    cfg = InferenceConfig.from_dict(
        {
            "mode": "continuous_batch",
            "dtype": "float16",
            "steps": 9,
            "request": {"batch_size": 2, "prompt_len": 16, "generate_tokens": 4},
            "model": {"kind": "mlp", "hidden_size": 64, "num_layers": 1, "vocab_size": 128},
            "serving": {"kv_cache": False, "continuous_batch": {"max_active_requests": 3}},
            "checks": {"compare_logits_checksum": False},
        }
    )
    assert cfg.mode == "continuous_batch"
    assert cfg.dtype == "float16"
    assert cfg.steps == 9
    assert cfg.request.prompt_len == 16
    assert cfg.model.kind == "mlp"
    assert cfg.serving.kv_cache is False
    assert cfg.serving.continuous_batch.max_active_requests == 3
    # mode=continuous_batch forces the nested enabled flag on.
    assert cfg.serving.continuous_batch.enabled is True
    assert cfg.checks.compare_logits_checksum is False


def test_nested_continuous_enabled_promotes_mode() -> None:
    cfg = InferenceConfig.from_dict({"serving": {"continuous_batch": {"enabled": True}}})
    assert cfg.mode == "continuous_batch"


def test_moe_driven_by_num_experts() -> None:
    cfg = InferenceConfig.from_dict({"model": {"kind": "decoder_transformer", "moe": {"num_experts": 4}}})
    assert cfg.model.num_experts == 4


def test_config_rejects_unknown_keys() -> None:
    # Dispatcher-reserved / platform keys are silently accepted.
    cfg = InferenceConfig.from_dict({"seed": 5, "_aorta_environment": {"x": 1}})
    assert cfg.seed == 5
    # True typos (e.g. missing underscore, misspelled section) must raise.
    with pytest.raises(ValueError, match="unknown inference key"):
        InferenceConfig.from_dict({"seed": 5, "bogus": 1})
    with pytest.raises(ValueError, match="unknown request key"):
        InferenceConfig.from_dict({"request": {"requst_batch": 4}})
    with pytest.raises(ValueError, match="unknown serving key"):
        InferenceConfig.from_dict({"serving": {"kvcache": True}})


@pytest.mark.parametrize(
    "bad",
    [
        {"mode": "speculative"},
        {"dtype": "int8"},
        {"device": "tpu"},
        {"steps": 0},
        {"warmup_steps": -1},
        {"model": {"kind": "rnn"}},
        {"model": {"num_layers": 0}},
        {"model": {"kind": "decoder_transformer", "hidden_size": 100, "num_heads": 3}},
        # num_heads=0 must fail cleanly (ValueError), not raise ZeroDivisionError.
        {"model": {"kind": "decoder_transformer", "num_heads": 0}},
        {"request": {"batch_size": 0}},
        {"request": {"prompt_len": 0}},
        {"request": {"generate_tokens": -1}},
        {"serving": {"continuous_batch": {"max_active_requests": 0}}},
        {"serving": {"continuous_batch": {"arrival_pattern": "poisson"}}},
        # continuous_batch requires at least one decode step per request.
        {"mode": "continuous_batch", "request": {"generate_tokens": 0}},
        # boolean fields must be real bools, not coercible strings.
        {"serving": {"kv_cache": "false"}},
        {"serving": {"continuous_batch": {"enabled": "false"}}},
        {"checks": {"fail_on_nan_logits": "false"}},
        # continuous_batch is a decode loop; encoder (non-autoregressive) is invalid.
        {"mode": "continuous_batch", "model": {"kind": "encoder_transformer"}},
    ],
)
def test_config_rejects_garbage(bad: dict) -> None:
    with pytest.raises(ValueError):
        InferenceConfig.from_dict(bad)


def test_offline_allows_zero_generate_tokens() -> None:
    # generate_tokens=0 is valid for offline/encoder-style inference.
    cfg = InferenceConfig.from_dict({"mode": "offline_batch", "request": {"generate_tokens": 0}})
    assert cfg.request.generate_tokens == 0


def test_encoder_is_not_autoregressive() -> None:
    cfg = InferenceConfig.from_dict({"model": {"kind": "encoder_transformer"}})
    assert cfg.is_autoregressive is False


def test_percentile_helper() -> None:
    assert _percentile([], 50) == 0.0
    assert _percentile([4.0], 99) == 4.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)


def test_workload_declares_contract() -> None:
    assert InferenceWorkload.name == "inference"
    assert InferenceWorkload.launch_mode == "single_process"
    assert InferenceWorkload.min_world_size == 1


# --------------------------------------------------------------------------- #
# torch-dependent: model construction, smokes, schema, NaN check
# --------------------------------------------------------------------------- #
try:
    import torch
except Exception:  # pragma: no cover - torch-less discovery env
    torch = None  # type: ignore[assignment]

requires_torch = pytest.mark.skipif(torch is None, reason="requires PyTorch")

_REQUIRED_METRICS = (
    "mode",
    "device",
    "dtype",
    "model_kind",
    "parameter_count",
    "batch_size",
    "prompt_len",
    "generate_tokens",
    "prefill_latency_ms",
    "decode_latency_ms",
    "tokens_per_sec",
    "step_time_p50",
    "step_time_p99",
)


@requires_torch
@pytest.mark.parametrize("kind", ["mlp", "encoder_transformer", "decoder_transformer"])
def test_build_model_returns_vocab_logits(kind: str) -> None:
    from aorta.workloads.inference import _build_model

    spec = ModelSpec.from_dict(
        {
            "kind": kind,
            "hidden_size": 32,
            "num_heads": 4,
            "ffn_size": 64,
            "num_layers": 2,
            "vocab_size": 48,
        }
    )
    model = _build_model(spec, seq_len=16).to(torch.float32)
    ids = torch.randint(0, 48, (2, 16))
    out = model(ids)
    assert out.shape == (2, 16, 48)
    assert torch.isfinite(out).all()


def _assert_result_schema(result, *, total: int) -> None:
    assert result.passed is True
    assert result.failure_count == 0
    assert result.first_failure_iteration is None
    assert result.total_iterations == total
    assert result.executed_iterations == total
    assert result.configured_iterations == total
    assert result.main_work_started is True
    assert len(result.step_times_ms) == total
    assert result.elapsed_sec >= 0.0
    for key in _REQUIRED_METRICS:
        assert key in result.metrics, f"missing metric {key}"


@requires_torch
def test_offline_batch_smoke() -> None:
    wl = InferenceWorkload(
        {
            "mode": "offline_batch",
            "device": "cpu",
            "dtype": "float32",
            "warmup_steps": 1,
            "steps": 3,
            "request": {"batch_size": 2, "prompt_len": 8, "generate_tokens": 4},
            "model": {"kind": "decoder_transformer", "hidden_size": 32, "num_heads": 4,
                      "num_layers": 2, "ffn_size": 64, "vocab_size": 48},
        }
    )
    try:
        wl.setup()
        result = wl.run()
    finally:
        wl.cleanup()

    _assert_result_schema(result, total=3)
    m = result.metrics
    assert m["mode"] == "offline_batch"
    assert m["model_kind"] == "decoder_transformer"
    assert m["parameter_count"] > 0
    assert m["decode_latency_ms"] is not None
    assert m["tokens_per_sec"] > 0.0
    assert "logits_checksum" in m


@requires_torch
def test_offline_mlp_smoke() -> None:
    wl = InferenceWorkload(
        {
            "mode": "offline_batch",
            "device": "cpu",
            "dtype": "float32",
            "steps": 2,
            "request": {"batch_size": 1, "prompt_len": 6, "generate_tokens": 2},
            "model": {"kind": "mlp", "hidden_size": 16, "num_layers": 1, "vocab_size": 32},
        }
    )
    try:
        wl.setup()
        result = wl.run()
    finally:
        wl.cleanup()
    _assert_result_schema(result, total=2)
    assert result.metrics["model_kind"] == "mlp"


@requires_torch
def test_encoder_has_no_decode_latency() -> None:
    wl = InferenceWorkload(
        {
            "mode": "offline_batch",
            "device": "cpu",
            "dtype": "float32",
            "steps": 2,
            "request": {"batch_size": 2, "prompt_len": 8, "generate_tokens": 4},
            "model": {"kind": "encoder_transformer", "hidden_size": 32, "num_heads": 4,
                      "num_layers": 1, "ffn_size": 64, "vocab_size": 48},
        }
    )
    try:
        wl.setup()
        result = wl.run()
    finally:
        wl.cleanup()
    _assert_result_schema(result, total=2)
    # No autoregressive decode → no decode-token timings, throughput from prefill.
    assert result.metrics["decode_latency_ms"] is None
    assert result.metrics["tokens_per_sec"] > 0.0


@requires_torch
def test_checksum_disabled_omits_metric_and_throughput_for_zero_decode() -> None:
    # compare_logits_checksum off -> no logits_checksum metric.
    # autoregressive + generate_tokens=0 -> prefill-based throughput (> 0).
    wl = InferenceWorkload(
        {
            "mode": "offline_batch",
            "device": "cpu",
            "dtype": "float32",
            "warmup_steps": 0,
            "steps": 2,
            "request": {"batch_size": 1, "prompt_len": 6, "generate_tokens": 0},
            "model": {"kind": "decoder_transformer", "hidden_size": 32, "num_heads": 4,
                      "num_layers": 1, "ffn_size": 64, "vocab_size": 48},
            "checks": {"compare_logits_checksum": False},
        }
    )
    try:
        wl.setup()
        result = wl.run()
    finally:
        wl.cleanup()
    assert "logits_checksum" not in result.metrics
    assert result.metrics["decode_latency_ms"] is None
    assert result.metrics["tokens_per_sec"] > 0.0


@requires_torch
def test_steps_override_from_dispatcher() -> None:
    wl = InferenceWorkload(
        {
            "mode": "offline_batch",
            "device": "cpu",
            "dtype": "float32",
            "steps": 5,
            "request": {"batch_size": 1, "prompt_len": 4, "generate_tokens": 1},
            "model": {"kind": "mlp", "hidden_size": 16, "num_layers": 1, "vocab_size": 32},
        }
    )
    try:
        wl.setup()
        assert wl._cfg.steps == 5
        result = wl.run()
    finally:
        wl.cleanup()
    assert result.configured_iterations == 5


@requires_torch
def test_continuous_batch_smoke() -> None:
    wl = InferenceWorkload(
        {
            "mode": "continuous_batch",
            "device": "cpu",
            "dtype": "float32",
            "warmup_steps": 0,
            "steps": 6,
            "request": {"batch_size": 4, "prompt_len": 8, "generate_tokens": 2},
            "serving": {"continuous_batch": {"enabled": True, "max_active_requests": 2}},
            "model": {"kind": "decoder_transformer", "hidden_size": 32, "num_heads": 4,
                      "num_layers": 1, "ffn_size": 64, "vocab_size": 48},
        }
    )
    try:
        wl.setup()
        result = wl.run()
    finally:
        wl.cleanup()
    _assert_result_schema(result, total=6)
    m = result.metrics
    assert m["mode"] == "continuous_batch"
    assert m["max_active_requests"] == 2
    # All 4 requests (2 tokens each) should retire within 6 ticks.
    assert m["requests_completed"] == 4


@requires_torch
def test_continuous_batch_kv_cache_disabled() -> None:
    # kv_cache=False feeds the full context window each tick; should still run.
    wl = InferenceWorkload(
        {
            "mode": "continuous_batch",
            "device": "cpu",
            "dtype": "float32",
            "warmup_steps": 0,
            "steps": 6,
            "request": {"batch_size": 4, "prompt_len": 8, "generate_tokens": 2},
            "serving": {"kv_cache": False, "continuous_batch": {"enabled": True, "max_active_requests": 2}},
            "model": {"kind": "decoder_transformer", "hidden_size": 32, "num_heads": 4,
                      "num_layers": 1, "ffn_size": 64, "vocab_size": 48},
        }
    )
    try:
        wl.setup()
        result = wl.run()
    finally:
        wl.cleanup()
    assert result.passed is True
    assert result.metrics["kv_cache"] is False
    assert result.metrics["requests_completed"] == 4


@requires_torch
def test_nan_logits_fail_the_trial(monkeypatch) -> None:
    """A model that emits NaN logits must fail the numeric check."""
    wl = InferenceWorkload(
        {
            "mode": "offline_batch",
            "device": "cpu",
            "dtype": "float32",
            "warmup_steps": 0,
            "steps": 1,
            "request": {"batch_size": 1, "prompt_len": 4, "generate_tokens": 0},
            "model": {"kind": "mlp", "hidden_size": 16, "num_layers": 1, "vocab_size": 32},
        }
    )
    wl.setup()
    try:
        real_model = wl._model

        def _nan_forward(input_ids):
            out = real_model(input_ids)
            return out * float("nan")

        monkeypatch.setattr(wl, "_model", _nan_forward)
        result = wl.run()
    finally:
        wl.cleanup()

    assert result.passed is False
    assert result.failure_count >= 1
    assert result.first_failure_iteration == 0
    assert any("nan_logits" in d["problems"] for d in result.failure_details)


@requires_torch
def test_inf_logits_caught_under_nan_flag(monkeypatch) -> None:
    """inf logits must be reported even when fail_on_nonfinite_output is off."""
    wl = InferenceWorkload(
        {
            "mode": "offline_batch",
            "device": "cpu",
            "dtype": "float32",
            "warmup_steps": 0,
            "steps": 1,
            "request": {"batch_size": 1, "prompt_len": 4, "generate_tokens": 0},
            "model": {"kind": "mlp", "hidden_size": 16, "num_layers": 1, "vocab_size": 32},
            "checks": {
                "fail_on_nan_logits": True,
                "fail_on_nonfinite_output": False,
                "compare_logits_checksum": False,
            },
        }
    )
    wl.setup()
    try:
        real_model = wl._model

        def _inf_forward(input_ids):
            return real_model(input_ids) + float("inf")

        monkeypatch.setattr(wl, "_model", _inf_forward)
        result = wl.run()
    finally:
        wl.cleanup()

    assert result.passed is False
    assert any("inf_logits" in d["problems"] for d in result.failure_details)
