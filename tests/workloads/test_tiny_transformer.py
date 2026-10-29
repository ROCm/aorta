"""CPU-only tests for the tiny_transformer workload and its launchers.

No test needs a GPU or a Monarch install. The loss-mismatch case stubs the
collective. The spur case stubs ``Popen``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.workloads.distributed_launch import (
    SPUR_WORKER_SCRIPT,
    is_ipv4,
    launch_monarch,
    launch_spur,
    verdict_from_workers,
)
from aorta.workloads.tiny_transformer import TinyTransformerConfig, TinyTransformerWorkload
from aorta.workloads.training import final_losses_disagree

try:
    import torch
except Exception:  # pragma: no cover - torch-less discovery env
    torch = None  # type: ignore[assignment]

requires_torch = pytest.mark.skipif(torch is None, reason="requires PyTorch")

_RECIPES = (
    "tiny-transformer.yaml",
    "tiny-transformer-monarch.yaml",
    "tiny-transformer-spur.yaml",
)


def _recipe(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "recipes" / "training" / name


def test_inprocess_calls_shared_rank_function(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_rank(config: dict) -> dict:
        captured["config"] = config
        return {
            "passed": True,
            "failure_count": 0,
            "metrics": {"world_size": 1, "final_loss": 0.5, "rank": 0},
        }

    monkeypatch.setattr("aorta.workloads.tiny_transformer.run_training_rank", fake_rank)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("AORTA_INNER", raising=False)
    wl = TinyTransformerWorkload(
        {"launcher": "inprocess", "device": "cpu", "dtype": "float32", "steps": 1}
    )
    wl.setup()
    result = wl.run()
    wl.cleanup()
    config = captured.get("config")
    assert isinstance(config, dict)
    model = config.get("model")
    assert isinstance(model, dict)
    assert model.get("hidden_size") == 256
    assert model.get("num_layers") == 4
    assert model.get("ffn_size") == 512
    assert config.get("parallelism") == "ddp"
    assert result.passed is True
    assert result.metrics.get("launcher") == "inprocess"
    assert result.metrics.get("final_loss") == 0.5
    assert result.metrics.get("world_size") == 1


def test_cleanup_error_is_recorded_on_the_result(monkeypatch) -> None:
    from aorta.workloads._base import WorkloadResult
    from aorta.workloads.distributed_launch import run_training_rank

    class _FakeWorkload:
        def __init__(self, config: dict) -> None:
            del config

        def setup(self) -> None:
            return None

        def run(self) -> WorkloadResult:
            return WorkloadResult(
                passed=True,
                metrics={"world_size": 1, "final_loss": 0.1, "rank": 0},
            )

        def cleanup(self) -> None:
            raise RuntimeError("barrier failed")

    monkeypatch.setattr(
        "aorta.workloads.distributed_launch.TrainingWorkload",
        _FakeWorkload,
    )
    data = run_training_rank({})
    metrics = data.get("metrics")
    assert isinstance(metrics, dict)
    assert metrics.get("cleanup_error") == "RuntimeError: barrier failed"
    assert data.get("passed") is True


def test_unknown_launcher_rejected() -> None:
    with pytest.raises(ValueError, match="launcher"):
        TinyTransformerConfig.from_dict({"launcher": "torchrun"})


def test_launcher_defaults_inprocess() -> None:
    cfg = TinyTransformerConfig.from_dict({})
    assert cfg.launcher == "inprocess"
    assert cfg.expected_world_size is None
    assert cfg.spur_bin == "spur"


def test_distributed_expected_world_size_must_match_mesh() -> None:
    with pytest.raises(ValueError, match="nodes \\* gpus_per_node"):
        TinyTransformerConfig.from_dict(
            {
                "launcher": "spur",
                "nodes": 2,
                "gpus_per_node": 8,
                "expected_world_size": 4,
            }
        )


def test_recipes_load_and_dry_run_does_not_spawn(tmp_path, monkeypatch) -> None:
    from aorta.triage.recipe import load_recipe
    from aorta.triage.runner import run_recipe

    def _boom(*_args, **_kwargs):
        raise AssertionError("dry-run spawned a process")

    monkeypatch.setattr("subprocess.Popen", _boom)
    monkeypatch.setattr("aorta.workloads.distributed_launch.subprocess.Popen", _boom)

    for name in _RECIPES:
        recipe = load_recipe(_recipe(name))
        assert recipe.workload == "tiny_transformer"
        assert recipe.steps == 20
        launcher = recipe.workload_config.get("launcher")
        assert launcher in {"inprocess", "monarch", "spur"}
        model = recipe.workload_config.get("model")
        assert isinstance(model, dict)
        assert model.get("hidden_size") == 256
        assert model.get("num_layers") == 4
        assert model.get("ffn_size") == 512
        assert model.get("vocab_size") == 256
        if name == "tiny-transformer.yaml":
            assert "expected_world_size" not in recipe.workload_config
        else:
            assert recipe.workload_config.get("expected_world_size") == 16
            assert recipe.workload_config.get("nodes") == 2
            assert recipe.workload_config.get("gpus_per_node") == 8
        run_recipe(recipe, output_dir=tmp_path / name, dry_run=True)


def test_spur_launch_sets_inner_rank_and_ipv4(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Proc:
        def __init__(self) -> None:
            self.pid = 424242
            self.returncode: int | None = None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.returncode = 0
            return 0

        def poll(self) -> int | None:
            return self.returncode

        def send_signal(self, sig: int) -> None:
            raise AssertionError(f"spur test signalled a process ({sig})")

    def fake_popen(argv, env=None, **kwargs):
        captured["argv"] = list(argv)
        captured["env"] = dict(env or {})
        captured["start_new_session"] = kwargs.get("start_new_session")
        if "--results-dir" in argv:
            out = Path(argv[argv.index("--results-dir") + 1])
        elif "--output-dir" in argv:
            out = Path(argv[argv.index("--output-dir") + 1])
        else:
            raise AssertionError(f"worker output dir missing from {argv!r}")
        trial = out / "tiny_transformer" / "trial_d0_m0_t0.json"
        trial.parent.mkdir(parents=True, exist_ok=True)
        trial.write_text(
            json.dumps(
                {
                    "result": {
                        "passed": True,
                        "failure_count": 0,
                        "metrics": {
                            "world_size": 16,
                            "final_loss": 5.5,
                            "rank": 0,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        return _Proc()

    monkeypatch.setattr("aorta.workloads.distributed_launch.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "aorta.workloads.distributed_launch.controller_ipv4",
        lambda: "10.190.161.245",
    )
    result = launch_spur(
        {"steps": 20},
        nodes=2,
        gpus_per_node=8,
        spur_bin="/opt/spur/bin/spur",
    )

    argv = captured.get("argv")
    env = captured.get("env")
    assert isinstance(argv, list)
    assert isinstance(env, dict)
    assert argv[:6] == [
        "/opt/spur/bin/spur",
        "srun",
        "-N",
        "2",
        "--ntasks-per-node",
        "8",
    ]
    assert SPUR_WORKER_SCRIPT in argv
    assert 'export RANK="$SLURM_PROCID"' in SPUR_WORKER_SCRIPT
    assert env.get("AORTA_INNER") == "1"
    master = env.get("MASTER_ADDR")
    assert master == "10.190.161.245"
    assert is_ipv4(master)
    assert env.get("WORLD_SIZE") == "16"
    assert "RANK" not in env
    inner = argv[argv.index("aorta-inner") + 1 :]
    assert "run" in inner
    assert "tiny_transformer" in inner
    assert captured.get("start_new_session") is True
    assert result.passed is True
    assert result.metrics.get("launcher") == "spur"
    assert result.metrics.get("world_size") == 16
    assert result.metrics.get("final_loss") == 5.5


def test_unreadable_rank0_result_fails_the_trial(monkeypatch, tmp_path) -> None:
    del tmp_path

    class _Proc:
        def __init__(self) -> None:
            self.pid = 424243
            self.returncode: int | None = None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.returncode = 0
            return 0

        def poll(self) -> int | None:
            return self.returncode

        def send_signal(self, sig: int) -> None:
            del sig

    def fake_popen(argv, env=None, **kwargs):
        del env, kwargs
        out = Path(argv[argv.index("--results-dir") + 1])
        trial = out / "trial_d0_m0_t0.json"
        trial.parent.mkdir(parents=True, exist_ok=True)
        trial.write_text("{", encoding="utf-8")
        return _Proc()

    monkeypatch.setattr("aorta.workloads.distributed_launch.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "aorta.workloads.distributed_launch.controller_ipv4",
        lambda: "10.0.0.8",
    )
    result = launch_spur({"steps": 1}, nodes=1, gpus_per_node=1, spur_bin="spur")
    assert result.passed is False
    errors = [
        item.get("error")
        for item in result.failure_details
        if isinstance(item.get("problems"), list) and "worker_result_unreadable" in item.get("problems", [])
    ]
    assert errors
    assert "trial_d0_m0_t0.json" in str(errors[0])


def test_spur_verdict_requires_exit_code_and_rank0() -> None:
    rank0 = {"passed": True, "metrics": {"world_size": 16, "final_loss": 1.25, "rank": 0}}
    failed_exit = verdict_from_workers(1, rank0, launcher="spur")
    assert failed_exit.passed is False
    assert failed_exit.metrics.get("world_size") == 16
    assert failed_exit.metrics.get("launcher") == "spur"
    problems: list[object] = []
    for item in failed_exit.failure_details:
        found = item.get("problems")
        if isinstance(found, list):
            problems.extend(found)
    assert "worker_exit" in problems

    missing = verdict_from_workers(0, None, launcher="spur")
    assert missing.passed is False
    assert missing.metrics.get("final_loss") is None
    assert missing.metrics.get("world_size") is None


def test_monarch_launcher_names_source_build_when_missing(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "monarch" or name.startswith("monarch."):
            raise ImportError("monarch is not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="pytorch_monarch source build"):
        launch_monarch({"steps": 1}, nodes=2, gpus_per_node=8)


def test_single_rank_loss_is_not_a_mismatch() -> None:
    assert final_losses_disagree(1.25, world_size=1, device=None) is False


@requires_torch
def test_loss_mismatch_stubbed_two_ranks(monkeypatch) -> None:
    import torch.distributed as dist

    peer = {"loss": 4.0}

    def all_reduce(tensor, op=None):
        del op
        local = float(tensor[0].item())
        negated = float(tensor[1].item())
        tensor[0] = max(local, peer["loss"])
        tensor[1] = max(negated, -peer["loss"])

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    assert final_losses_disagree(1.0, world_size=2, device=torch.device("cpu")) is True
    peer["loss"] = 1.0
    assert final_losses_disagree(1.0, world_size=2, device=torch.device("cpu")) is False


@requires_torch
def test_run_records_rank_loss_mismatch(monkeypatch) -> None:
    """The training loop fails when the stubbed ranks disagree, and passes when they match."""
    import torch.distributed as dist

    from aorta.workloads.training import TrainingConfig, TrainingWorkload, _build_model

    disagree = {"value": True}

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", lambda *_a, **_k: None)
    monkeypatch.setattr(dist, "barrier", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "aorta.workloads.training.final_losses_disagree",
        lambda *_a, **_k: disagree["value"],
    )

    cfg = TrainingConfig.from_dict(
        {
            "parallelism": "ddp",
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 1,
            "seq_len": 4,
            "steps": 1,
            "model": {"kind": "mlp", "hidden_size": 8, "num_layers": 1, "vocab_size": 16},
            "checks": {"fail_on_rank_loss_mismatch": True},
        }
    )
    wl = TrainingWorkload({})
    wl._cfg = cfg
    wl._device = torch.device("cpu")
    wl._dtype = torch.float32
    wl._rank = 0
    wl._world_size = 2
    model = _build_model(cfg.model, cfg.seq_len).to(torch.float32)
    wl._model = model
    wl._optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    wl._scheduler = None
    wl._parameter_count = 1
    wl._input_gen = torch.Generator(device="cpu").manual_seed(1)

    failed = wl.run()
    assert failed.passed is False
    problems: list[object] = []
    for item in failed.failure_details:
        found = item.get("problems")
        if isinstance(found, list):
            problems.extend(found)
    assert "rank_loss_mismatch" in problems

    disagree["value"] = False
    passed = wl.run()
    assert passed.passed is True
    assert passed.metrics.get("world_size") == 2
