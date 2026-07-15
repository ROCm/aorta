"""Tests for the `training` workload.

Config parsing/validation runs without torch. Model construction, DDP/FSDP
selection, the singleton smoke, and a stubbed multi-rank run require torch and
are skipped where it is unavailable. No test needs a GPU; the smoke uses CPU +
gloo and the multi-rank case stubs the collectives.
"""

from __future__ import annotations

import pytest

from aorta.workloads.training import (
    ChecksSpec,
    ModelSpec,
    OptimizerSpec,
    TrainingConfig,
    TrainingWorkload,
)
from aorta.workloads.training import _percentile

# --------------------------------------------------------------------------- #
# Config parsing / validation (torch-free)
# --------------------------------------------------------------------------- #


def test_config_defaults() -> None:
    cfg = TrainingConfig.from_dict({})
    assert cfg.parallelism == "fsdp"
    assert cfg.model.kind == "transformer"
    assert cfg.optimizer.kind == "adamw"
    assert cfg.checks.fail_on_nan_loss is True


def test_config_nested_sections_parsed() -> None:
    cfg = TrainingConfig.from_dict(
        {
            "parallelism": "ddp",
            "dtype": "float16",
            "steps": 8,
            "model": {
                "kind": "moe_transformer",
                "hidden_size": 128,
                "num_heads": 4,
                "moe": {"enabled": True, "num_experts": 6},
            },
            "optimizer": {"lr": 0.002, "weight_decay": 0.05},
            "checks": {"fail_on_nan_grad": False},
        }
    )
    assert cfg.parallelism == "ddp"
    assert cfg.dtype == "float16"
    assert cfg.steps == 8
    assert cfg.model.kind == "moe_transformer"
    assert cfg.model.num_experts == 6
    assert cfg.model.effective_experts == 6
    assert cfg.optimizer.lr == 0.002
    assert cfg.checks.fail_on_nan_grad is False


def test_effective_experts_is_one_for_dense_topologies() -> None:
    assert ModelSpec.from_dict({"kind": "transformer", "num_experts": 8}).effective_experts == 1
    assert ModelSpec.from_dict({"kind": "mlp"}).effective_experts == 1
    assert ModelSpec.from_dict({"kind": "moe_transformer", "moe": {"num_experts": 3}}).effective_experts == 3


def test_moe_transformer_rejects_single_expert() -> None:
    with pytest.raises(ValueError, match="num_experts must be >= 2"):
        ModelSpec.from_dict({"kind": "moe_transformer", "moe": {"num_experts": 1}})


def test_transformer_rejects_misaligned_hidden_heads() -> None:
    with pytest.raises(ValueError, match="divisible by num_heads"):
        ModelSpec.from_dict({"kind": "transformer", "hidden_size": 100, "num_heads": 3})


def test_config_ignores_unknown_keys() -> None:
    cfg = TrainingConfig.from_dict({"seed": 5, "_aorta_environment": {"x": 1}, "bogus": 1})
    assert cfg.seed == 5


@pytest.mark.parametrize(
    "bad",
    [
        {"parallelism": "tp"},
        {"dtype": "int8"},
        {"device": "tpu"},
        {"steps": 0},
        {"warmup_steps": -1},
        {"model": {"kind": "rnn"}},
        {"model": {"num_layers": 0}},
        {"model": {"kind": "moe_transformer", "moe": {"num_experts": 1}}},
        {"optimizer": {"kind": "sgd"}},
        {"optimizer": {"betas": [0.9]}},
    ],
)
def test_config_rejects_garbage(bad: dict) -> None:
    with pytest.raises(ValueError):
        TrainingConfig.from_dict(bad)


def test_optimizer_betas_list_coerced_to_tuple() -> None:
    spec = OptimizerSpec.from_dict({"betas": [0.9, 0.95]})
    assert spec.betas == (0.9, 0.95)


def test_percentile_helper() -> None:
    assert _percentile([], 50) == 0.0
    assert _percentile([4.0], 99) == 4.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)


def test_workload_declares_contract() -> None:
    assert TrainingWorkload.name == "training"
    assert TrainingWorkload.launch_mode == "distributed"
    assert TrainingWorkload.min_world_size == 1


# --------------------------------------------------------------------------- #
# torch-dependent: model construction, selection, smoke, multi-rank
# --------------------------------------------------------------------------- #
try:
    import torch
except Exception:  # pragma: no cover - torch-less discovery env
    torch = None  # type: ignore[assignment]

requires_torch = pytest.mark.skipif(torch is None, reason="requires PyTorch")


@requires_torch
@pytest.mark.parametrize("kind", ["mlp", "transformer", "moe_transformer"])
def test_build_model_returns_vocab_logits(kind: str) -> None:
    from aorta.workloads.training import _build_model

    spec = ModelSpec.from_dict(
        {
            "kind": kind,
            "hidden_size": 32,
            "num_heads": 4,
            "ffn_size": 64,
            "num_layers": 2,
            "vocab_size": 48,
            "moe": {"num_experts": 4},
        }
    )
    model = _build_model(spec, seq_len=16).to(torch.float32)
    ids = torch.randint(0, 48, (2, 16))
    out = model(ids)
    assert out.shape == (2, 16, 48)
    assert torch.isfinite(out).all()


def _selection_workload(parallelism: str) -> TrainingWorkload:
    wl = TrainingWorkload({})
    wl._cfg = TrainingConfig.from_dict({"parallelism": parallelism, "dtype": "float32"})
    wl._device = torch.device("cpu")
    wl._dtype = torch.float32
    wl._world_size = 1
    return wl


@requires_torch
def test_wrap_model_selects_ddp(monkeypatch) -> None:
    import torch.nn.parallel as parallel_mod

    class _FakeDDP(torch.nn.Module):
        def __init__(self, module, **kwargs):
            super().__init__()
            self.module = module
            self.kwargs = kwargs

    monkeypatch.setattr(parallel_mod, "DistributedDataParallel", _FakeDDP)
    wl = _selection_workload("ddp")
    wrapped = wl._wrap_model(torch.nn.Linear(4, 4))
    assert isinstance(wrapped, _FakeDDP)
    # Dense topology must not flip find_unused_parameters.
    assert wrapped.kwargs["find_unused_parameters"] is False


@requires_torch
def test_wrap_model_ddp_moe_enables_find_unused(monkeypatch) -> None:
    import torch.nn.parallel as parallel_mod

    class _FakeDDP(torch.nn.Module):
        def __init__(self, module, **kwargs):
            super().__init__()
            self.kwargs = kwargs

    monkeypatch.setattr(parallel_mod, "DistributedDataParallel", _FakeDDP)
    wl = TrainingWorkload({})
    wl._cfg = TrainingConfig.from_dict(
        {"parallelism": "ddp", "dtype": "float32", "model": {"kind": "moe_transformer"}}
    )
    wl._device = torch.device("cpu")
    wl._dtype = torch.float32
    wl._world_size = 1
    wrapped = wl._wrap_model(torch.nn.Linear(4, 4))
    assert wrapped.kwargs["find_unused_parameters"] is True


@requires_torch
def test_wrap_model_selects_fsdp(monkeypatch) -> None:
    import torch.distributed.fsdp as fsdp_mod

    class _FakeFSDP(torch.nn.Module):
        def __init__(self, module, **kwargs):
            super().__init__()
            self.module = module

    monkeypatch.setattr(fsdp_mod, "FullyShardedDataParallel", _FakeFSDP)
    wl = _selection_workload("fsdp")
    wrapped = wl._wrap_model(torch.nn.Linear(4, 4))
    assert isinstance(wrapped, _FakeFSDP)


@pytest.fixture()
def _cpu_singleton_env(monkeypatch):
    """Bare-process env so setup() forms a 1-rank gloo group.

    MASTER_PORT is left unset — the workload picks an ephemeral free port.
    """
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)
    yield


@requires_torch
def test_singleton_ddp_smoke(_cpu_singleton_env) -> None:
    wl = TrainingWorkload(
        {
            "parallelism": "ddp",
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 2,
            "seq_len": 16,
            "warmup_steps": 1,
            "steps": 3,
            "model": {"kind": "mlp", "hidden_size": 32, "num_layers": 2, "vocab_size": 48},
        }
    )
    try:
        wl.setup()
        result = wl.run()
    finally:
        wl.cleanup()

    assert result.passed is True
    assert result.failure_count == 0
    assert result.first_failure_iteration is None
    assert result.total_iterations == 3
    assert result.executed_iterations == 3
    assert result.configured_iterations == 3
    assert result.main_work_started is True
    assert len(result.step_times_ms) == 3
    assert result.elapsed_sec >= 0.0

    m = result.metrics
    for key in (
        "parallelism",
        "rank",
        "world_size",
        "device",
        "dtype",
        "model_kind",
        "parameter_count",
        "final_loss",
        "step_time_p50",
        "step_time_p99",
    ):
        assert key in m, f"missing metric {key}"
    assert m["parallelism"] == "ddp"
    assert m["model_kind"] == "mlp"
    assert m["world_size"] == 1
    assert m["parameter_count"] > 0


@requires_torch
def test_steps_override_from_dispatcher(_cpu_singleton_env) -> None:
    # Recipe says steps:2 inside workload_config, dispatcher injects steps=5.
    wl = TrainingWorkload(
        {
            "parallelism": "ddp",
            "device": "cpu",
            "dtype": "float32",
            "steps": 5,
            "model": {"kind": "mlp", "hidden_size": 16, "num_layers": 1, "vocab_size": 32},
            "seq_len": 8,
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
def test_stubbed_multi_rank_run(monkeypatch) -> None:
    """run() aggregates a multi-rank verdict via all_reduce without a real PG."""
    import torch.distributed as dist

    from aorta.workloads.training import _build_model

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(dist, "barrier", lambda *a, **k: None)

    cfg = TrainingConfig.from_dict(
        {
            "parallelism": "ddp",
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 2,
            "seq_len": 8,
            "steps": 2,
            "model": {"kind": "mlp", "hidden_size": 16, "num_layers": 1, "vocab_size": 32},
        }
    )
    wl = TrainingWorkload({})
    wl._cfg = cfg
    wl._device = torch.device("cpu")
    wl._dtype = torch.float32
    wl._rank = 1
    wl._world_size = 2
    model = _build_model(cfg.model, cfg.seq_len).to(torch.float32)
    wl._model = model
    wl._optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    wl._scheduler = None
    wl._parameter_count = sum(p.numel() for p in model.parameters())
    wl._input_gen = torch.Generator(device="cpu").manual_seed(99)

    result = wl.run()
    assert result.passed is True
    assert result.metrics["world_size"] == 2
    assert result.metrics["rank"] == 1
    assert result.total_iterations == 2
