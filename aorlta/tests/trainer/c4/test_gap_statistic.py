"""The gap gate's statistic is a setting (`trainer.gap_statistic`, T10): the gate compares exactly that per-sample
statistic with `max_generation_gap`, the config reaches the loss settings, and the run's gaps file and iteration
summary name it. The default stays Section 6.4's per-token maximum."""

from __future__ import annotations

import pytest
import torch

from aorlta.config import ConfigError, config_from_dict
from aorlta.trainer.loss import GAP_STATISTICS, LossSettings, TrainerInputError, gap_keeps, sample_gap


def _gap():
    # 99 tokens 0.01 apart and one 2.5 apart: max 2.5, mean 0.0349, median 0.01, p99 0.0349 (interpolated)
    return sample_gap("c", torch.tensor([-1.0] * 99 + [-3.5]), [-1.01] * 99 + [-1.0])


def test_the_default_statistic_is_the_per_token_maximum():
    assert LossSettings().gap_statistic == "max_abs"
    g = _gap()
    assert not gap_keeps(LossSettings(max_generation_gap=2.0), g)
    assert gap_keeps(LossSettings(max_generation_gap=2.6), g)


@pytest.mark.parametrize("stat", GAP_STATISTICS)
def test_the_gate_compares_the_configured_statistic(stat):
    g = _gap()
    v = g.value(stat)
    assert v > 0
    assert gap_keeps(LossSettings(max_generation_gap=v * 1.001, gap_statistic=stat), g)
    assert not gap_keeps(LossSettings(max_generation_gap=v * 0.999, gap_statistic=stat), g)


def test_a_robust_statistic_keeps_a_sample_with_one_outlying_token():
    g = _gap()
    assert gap_keeps(LossSettings(max_generation_gap=0.05, gap_statistic="mean_abs"), g)
    assert not gap_keeps(LossSettings(max_generation_gap=0.05, gap_statistic="max_abs"), g)
    assert not gap_keeps(LossSettings(max_generation_gap=0.05, gap_statistic="mean_abs"), None)


def test_a_non_finite_statistic_never_keeps():
    g = sample_gap("c", torch.tensor([-1.0, float("nan")]), [-1.0, -1.0])
    assert not gap_keeps(LossSettings(max_generation_gap=1e9, gap_statistic="mean_abs"), g)


def test_unknown_statistics_are_refused():
    with pytest.raises(TrainerInputError):
        LossSettings(gap_statistic="mean")
    with pytest.raises(TrainerInputError):
        _gap().value("frac_gt_1")


def _cfg(tmp_path, **trainer):
    return config_from_dict({
        "run": {"run_id": "gapstat", "source": "standin", "root": str(tmp_path / "runs")},
        "model": {"base_dir": str(tmp_path / "base"), "source": "standin"},
        "data": {"prompt_pools": [{"path": str(tmp_path / "ds" / "listings"), "source": "standin"}]},
        "rollout": {"engines": [0], "K": 4, "P": 4},
        "reward": {"profile": "fake", "reward_fn": "standin.v11"},
        "orchestrator": {"iterations": 1, "run_seed": 7},
        "trainer": {"ranks": 1, **trainer},
        "sync": {"dir": str(tmp_path / "shm")},
    })


def test_the_config_key_reaches_the_loss_settings(tmp_path):
    cfg = _cfg(tmp_path, gap_statistic="mean_abs", max_generation_gap=0.05)
    st = LossSettings.from_trainer(cfg.trainer)
    assert (st.gap_statistic, st.max_generation_gap) == ("mean_abs", 0.05)
    assert LossSettings.from_trainer(_cfg(tmp_path).trainer).gap_statistic == "max_abs"
    with pytest.raises(ConfigError):
        _cfg(tmp_path, gap_statistic="mean")
