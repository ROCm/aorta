"""A task that only a plugin knows, through the real wiring on CPU: its prompts, parse, feedback, compile check,
profile, check adapter and reward function are the plugin's (tests/test_plugins.py), the engines and the model are
test_wiring_e2e's. Nothing in the package names the task."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_plugins as TP  # noqa: E402 - the toy plugin
import test_wiring_e2e as E  # noqa: E402 - the tiny model, its twin engines and the probes

from aorlta.config import config_from_dict  # noqa: E402
from aorlta.orchestrator import wiring  # noqa: E402
from aorlta.rollout.engine_proc import InProcessEngine  # noqa: E402

pytestmark = pytest.mark.heavy

tiny_base = E.tiny_base
registry, plugin = TP.registry, TP.plugin


def build(cfg, base):
    engines = [E.TwinEngine(base, E.FAKE, i) for i in range(len(cfg.rollout.engines))]
    parts = wiring.Parts(tokenizer=E.ByteTokenizer(), handles=[InProcessEngine(i, e) for i, e in enumerate(engines)],
                         load_model=E.load_trainer_model, trainer_device="cpu", require_fast_path=False,
                         probes=E.PROBES)
    return wiring.build(cfg, parts=parts)


def test_a_plugin_task_trains_and_lands(tmp_path, tiny_base, plugin):
    data = TP.toy_config(tmp_path, str(plugin))
    data["model"]["base_dir"] = str(tiny_base)
    data["rollout"].update({"engines": [0, 1], "K": 4, "P": 4, "max_new_tokens": 512, "request_timeout_s": 300})
    data["trainer"] = {"ranks": 1, "lr": 1e-2, "micro_batch_tokens": 2048}
    cfg = config_from_dict(data)
    orch = build(cfg, tiny_base)
    assert [r.prompt_id for r in orch.pool.records()] == [f"toy:p{i}" for i in range(6)]
    result = orch.run()
    assert result["status"] == "completed", result
    s = E.summary(cfg, 1)
    assert s["trainer"]["step_taken"] and s["samples"]["trained"] > 0 and s["sync"]["landing"] == "LANDED"
    eps = E.episodes(cfg, 1)
    assert {e["prompt_id"] for e in eps} <= {f"toy:p{i}" for i in range(6)}
    rewards = {e["reward"] for e in eps if e["status"] == "scored"}
    assert rewards and rewards <= {1.0, -1.0} and len(rewards) == 2  # toy.v1's two values, from toy.check
    assert any(len(e["turns"]) > 1 for e in eps)  # a repair turn: toy.feedback's compile or apply turn
    plan = json.loads((cfg.run_dir / "run_plan.json").read_text())
    assert plan["plugins"][str(plugin)]["sha256"] == TP.sha256_file(plugin)
