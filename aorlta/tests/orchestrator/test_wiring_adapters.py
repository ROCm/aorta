"""The orchestrator's wiring of the reward service and the trainer: what a run config sets reaches the adapters it
builds (the per-check GPU limits, the gap gate's statistic)."""

from __future__ import annotations

import json

import pytest
import torch

from aorlta.config import config_from_dict
from aorlta.reward.clock import Clock
from aorlta.trainer.loss import sample_gap


def _gap():
    # 99 tokens 0.01 apart and one 2.5 apart: max 2.5, mean 0.0349, median 0.01, p99 0.0349 (interpolated)
    return sample_gap("c", torch.tensor([-1.0] * 99 + [-3.5]), [-1.01] * 99 + [-1.0])


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


def test_a_run_config_carries_gpu_check_s_into_the_reward_adapters_factories(tmp_path):
    from aorlta.orchestrator import wiring

    cfg = config_from_dict({
        "run": {"run_id": "w", "source": "standin", "root": str(tmp_path / "runs")},
        "model": {"base_dir": str(tmp_path / "base"), "source": "standin"},
        "data": {"prompt_pools": [{"path": str(tmp_path / "ds" / "listings"), "source": "standin"}]},
        "rollout": {"engines": [0, 1], "K": 4, "P": 4, "compile_adapter": "fake.compile"},
        "reward": {"profile": "standin-gpu", "reward_fn": "standin.v11", "gpu_check_s": {"san.consan": 25.0}},
        "orchestrator": {"iterations": 1, "run_seed": 7}, "trainer": {"ranks": 1},
        "sync": {"dir": str(tmp_path / "shm")}})
    r = wiring.RewardAdapter(cfg, clock=Clock(), spans=None)
    assert r.factories["standin.aorta_consan"]("consan/1").check_s == 25.0
    assert r.factories["standin.aorta_asan"]("asan/1").adapter.check_s == 45.0


@pytest.mark.parametrize("stat", ["max_abs", "mean_abs"])
def test_the_gaps_file_and_summary_name_the_configured_statistic(tmp_path, stat):
    from aorlta.orchestrator.wiring import TrainerAdapter
    from aorlta.spans import SpanRecorder
    from aorlta.trainer.checks import gap_summary

    cfg = _cfg(tmp_path, gap_statistic=stat)
    adapter = TrainerAdapter(cfg, spans=SpanRecorder())
    adapter.iteration = 3
    gaps = [_gap(), sample_gap("d", torch.tensor([-1.0, -1.2]), [-1.1, -1.0])]
    report = {"gaps": list(gaps), "gap": gap_summary(gaps)}
    adapter._write_gaps(report)
    assert report["gap"]["statistic"] == stat
    assert report["gap"]["median"] == report["gap"][stat]["median"]
    assert report["gap"]["p99"] == report["gap"][stat]["p99"]
    rows = [json.loads(x) for x in (cfg.run_dir / "trainer" / "gaps-000003.jsonl").read_text().splitlines()]
    assert [r["chain_id"] for r in rows] == ["c", "d"]
