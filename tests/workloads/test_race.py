"""Unit tests for the `race` workload adapter.

These are unit-level: no real torch.distributed. The config filter is tested
directly, and result mapping is tested by monkeypatching `create_reproducer`
and stubbing the distributed init in `setup()`.
"""

import logging

import pytest

from aorta.race.config import ReproducerConfig, ReproducerResult
from aorta.workloads.race import RaceWorkload


class _StubReproducer:
    def __init__(self, result: ReproducerResult) -> None:
        self._result = result

    def run(self) -> ReproducerResult:
        return self._result


def test_race_config_from_dict_filters_unknown(caplog):
    """Unknown keys are dropped with a warning; known keys map onto the config."""
    wl = RaceWorkload({})
    cfg_in = {
        "mode": "fsdp",
        "warmup_iterations": 0,
        "verify_iterations": 50,
        "h2d_prefetch": True,
        "fsdp_shard_size": 1_000_000,
        "dtype": "bfloat16",
        # unknown keys that must be dropped + warned:
        "mixed_precision": "bf16",
        "foo": 123,
    }
    with caplog.at_level(logging.WARNING, logger="aorta.workloads.race"):
        cfg = wl._race_config_from_dict(cfg_in)

    assert isinstance(cfg, ReproducerConfig)
    assert cfg.mode == "fsdp"
    assert cfg.warmup_iterations == 0
    assert cfg.verify_iterations == 50
    assert cfg.h2d_prefetch is True
    assert cfg.fsdp_shard_size == 1_000_000
    assert cfg.dtype == "bfloat16"

    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("mixed_precision" in m for m in warned)
    assert any("foo" in m for m in warned)
    # Known keys must NOT be warned about.
    assert not any("h2d_prefetch" in m for m in warned)
    assert not any("fsdp_shard_size" in m for m in warned)


def test_race_config_reserved_aorta_keys_not_warned(caplog):
    """`_aorta_*` platform keys are reserved and silently ignored."""
    wl = RaceWorkload({})
    with caplog.at_level(logging.WARNING, logger="aorta.workloads.race"):
        wl._race_config_from_dict({"mode": "default", "_aorta_trial_id": 7})
    assert not any("_aorta_trial_id" in r.getMessage() for r in caplog.records)


def test_race_config_steps_key_not_warned(caplog):
    """`steps` is injected into every workload config by the dispatcher; it is
    a platform key (not a race field), so it must be dropped WITHOUT a warning
    -- otherwise every real run logs a spurious unknown-key warning."""
    wl = RaceWorkload({})
    with caplog.at_level(logging.WARNING, logger="aorta.workloads.race"):
        cfg = wl._race_config_from_dict({"mode": "default", "steps": 100})
    assert not any("steps" in r.getMessage() for r in caplog.records)
    assert not hasattr(cfg, "steps")  # not a ReproducerConfig field


def test_startup_gpu_queue_env_disables_late_config_overwrite(monkeypatch):
    monkeypatch.setenv("GPU_MAX_HW_QUEUES", "2")
    cfg = RaceWorkload({})._race_config_from_dict({"gpu_max_hw_queues": 4})
    assert cfg.gpu_max_hw_queues is None


def test_race_isolated_startup_env_supplies_gpu_queue_default():
    assert RaceWorkload.isolated_startup_env({}) == {"GPU_MAX_HW_QUEUES": "4"}
    assert RaceWorkload.isolated_startup_env({"gpu_max_hw_queues": 2}) == {
        "GPU_MAX_HW_QUEUES": "2"
    }


def test_process_isolated_cleanup_leaves_sync_to_worker(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("aorta.workloads.race.dist.is_initialized", lambda: True)
    monkeypatch.setattr(
        "aorta.workloads.race.dist.barrier",
        lambda: calls.append("barrier"),
    )
    RaceWorkload({"_aorta_trial_isolation": "process"}).cleanup()
    assert calls == []


def test_race_config_from_dict_rejects_bad_mode():
    wl = RaceWorkload({})
    with pytest.raises(ValueError, match="mode must be one of"):
        wl._race_config_from_dict({"mode": "nope"})


def test_race_config_from_dict_rejects_bad_dtype():
    wl = RaceWorkload({})
    with pytest.raises(ValueError, match="dtype must be one of"):
        wl._race_config_from_dict({"dtype": "int8"})


def test_race_config_from_dict_rejects_bad_compute_type():
    wl = RaceWorkload({})
    # A typo like "transfomer" must error, not silently fall back to GEMM.
    with pytest.raises(ValueError, match="compute_type must be one of"):
        wl._race_config_from_dict({"compute_type": "transfomer"})


def test_reproducer_config_rejects_bad_compute_type_directly():
    """Validation lives in ReproducerConfig.__post_init__, so even direct
    construction (bypassing the RaceWorkload adapter, e.g. the aorta.race CLI)
    rejects a typo instead of silently running GEMM (false green)."""
    with pytest.raises(ValueError, match="compute_type must be one of"):
        ReproducerConfig(compute_type="transfomer")


def test_race_config_warns_shared_weights_without_transformer(caplog):
    wl = RaceWorkload({})
    with caplog.at_level("WARNING"):
        cfg = wl._race_config_from_dict(
            {"compute_type": "gemm", "shared_layer_weights": True}
        )
    assert cfg.compute_type == "gemm"
    assert any("shared_layer_weights" in r.message for r in caplog.records)


def test_race_workload_maps_result(monkeypatch):
    """run() maps every ReproducerResult field onto WorkloadResult."""
    stub_result = ReproducerResult(
        passed=False,
        total_iterations=42,
        corruption_count=3,
        first_corruption_iter=7,
        corruption_details=[{"iter": 7, "rank": 0}],
        elapsed_time_sec=1.5,
        avg_step_time_ms=35.7,
    )

    captured = {}

    def fake_create_reproducer(cfg, rank, world_size):
        captured["cfg"] = cfg
        captured["rank"] = rank
        captured["world_size"] = world_size
        return _StubReproducer(stub_result)

    monkeypatch.setattr("aorta.workloads.race.create_reproducer", fake_create_reproducer)

    wl = RaceWorkload({"mode": "default", "warmup_iterations": 2, "verify_iterations": 3})
    # Bypass real distributed init.
    wl._rank = 0
    wl._world = 2
    wl._cfg = wl._race_config_from_dict(wl.config)

    res = wl.run()

    assert res.passed is False
    assert res.failure_count == 3
    assert res.first_failure_iteration == 7
    assert res.failure_details == [{"iter": 7, "rank": 0}]
    assert res.total_iterations == 42
    assert res.elapsed_sec == 1.5
    assert res.executed_iterations == 42
    assert res.configured_iterations == 5  # warmup 2 + verify 3
    assert res.main_work_started is True
    assert res.metrics["avg_step_time_ms"] == 35.7
    assert res.metrics["mode"] == "default"
    assert res.metrics["rank"] == 0
    assert res.metrics["world_size"] == 2
    assert captured["rank"] == 0 and captured["world_size"] == 2
