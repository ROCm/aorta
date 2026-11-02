"""CPU-only tests for the tiny_transformer workload and its launchers.

No test needs a GPU or a Monarch install. The loss-mismatch case stubs the
collective. The spur case stubs ``Popen``.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from aorta.workloads.distributed_launch import (
    SPUR_WORKER_SCRIPT,
    _group_has_members,
    _read_rank0_result,
    _terminate_process_tree,
    is_ipv4,
    launch_monarch,
    launch_spur,
    run_worker_rank,
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
    "tiny-transformer-monarch-spur.yaml",
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

        def cleanup(self, *, collective: bool = True) -> None:
            del collective
            raise RuntimeError("barrier failed")

    monkeypatch.setattr(
        "aorta.workloads.training.TrainingWorkload",
        _FakeWorkload,
    )
    data = run_training_rank({})
    metrics = data.get("metrics")
    assert isinstance(metrics, dict)
    assert metrics.get("cleanup_error") == "RuntimeError: barrier failed"
    assert data.get("passed") is True


def test_failed_rank_does_not_enter_the_cleanup_barrier(monkeypatch) -> None:
    from aorta.workloads.distributed_launch import run_training_rank

    calls: list[bool] = []

    class _FakeWorkload:
        def __init__(self, config: dict) -> None:
            del config

        def setup(self) -> None:
            return None

        def run(self) -> dict:
            raise RuntimeError("oom during forward")

        def cleanup(self, *, collective: bool = True) -> None:
            calls.append(collective)

    monkeypatch.setattr(
        "aorta.workloads.training.TrainingWorkload",
        _FakeWorkload,
    )
    with pytest.raises(RuntimeError, match="oom during forward"):
        run_training_rank({"steps": 1})
    assert calls == [False]


@requires_torch
def test_cleanup_barrier_runs_only_after_success(monkeypatch) -> None:
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d

    from aorta.workloads.training import TrainingWorkload

    calls: list[str] = []

    class _Group:
        def abort(self) -> None:
            calls.append("abort")

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(c10d, "_get_default_group", lambda: _Group())
    workload = TrainingWorkload({})
    workload.cleanup(collective=False)
    assert calls == ["abort"]
    workload.cleanup()
    assert calls == ["abort", "barrier"]


def test_monarch_trainer_class_is_module_scoped() -> None:
    import ast

    root = Path(__file__).resolve().parents[2]
    trainer = root / "src" / "aorta" / "workloads" / "monarch_trainer.py"
    tree = ast.parse(trainer.read_text(encoding="utf-8"))
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    assert classes == ["Trainer"]
    launch = (root / "src" / "aorta" / "workloads" / "distributed_launch.py").read_text(
        encoding="utf-8"
    )
    assert 'globals()["Trainer"]' not in launch


def test_launcher_import_does_not_load_training() -> None:
    """Classifying a controller must not import the training module at import time."""
    import ast

    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "aorta"
        / "workloads"
        / "distributed_launch.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level = [
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    ]
    assert "aorta.workloads.training" not in top_level


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


def test_spur_launch_sets_inner_rank_and_ipv4(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["aorta", "triage", "run", "--recipe", "matrix.yaml", "--trials", "5"],
    )

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
        config_path = Path(captured["env"]["AORTA_WORKER_CONFIG"])
        captured["config"] = json.loads(config_path.read_text(encoding="utf-8"))
        out = Path(captured["env"]["AORTA_RESULTS_DIR"])
        (out / "rank0.json").write_text(
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
    monkeypatch.setenv("AORTA_MASTER_PORT", "29501")
    monkeypatch.setenv("AORTA_RANK0_IP", "10.190.161.245")
    monkeypatch.setenv("AORTA_CONTROLLER_IP", "10.0.0.9")
    monkeypatch.setenv("AORTA_SHARED_DIR", str(tmp_path))
    result = launch_spur(
        {
            "steps": 20,
            "device": "cpu",
            "model": {"kind": "transformer", "hidden_size": 256},
            "checks": {"fail_on_rank_loss_mismatch": True},
            "optimizer": {"lr": 0.001},
            "launcher": "spur",
            "nodes": 2,
            "_aorta_cell": "dropped",
        },
        nodes=2,
        gpus_per_node=8,
        spur_bin="/opt/spur/bin/spur",
    )

    argv = captured.get("argv")
    env = captured.get("env")
    forwarded = captured.get("config")
    assert isinstance(argv, list)
    assert isinstance(env, dict)
    assert isinstance(forwarded, dict)
    assert argv[:6] == [
        "/opt/spur/bin/spur",
        "srun",
        "-N",
        "2",
        "--ntasks-per-node",
        "8",
    ]
    assert argv[6:8] == ["--ntasks", "16"]
    script = next(arg for arg in argv if isinstance(arg, str) and SPUR_WORKER_SCRIPT in arg)
    assert script.startswith("# aorta-owned-step aorta-")
    assert 'export RANK="$SLURM_PROCID"' in SPUR_WORKER_SCRIPT
    assert 'export LOCAL_RANK="$SLURM_LOCALID"' in SPUR_WORKER_SCRIPT
    assert env.get("AORTA_INNER") == "1"
    master = env.get("MASTER_ADDR")
    assert master == "10.190.161.245"
    assert is_ipv4(master)
    assert env.get("WORLD_SIZE") == "16"
    assert env.get("MASTER_PORT") == "29501"
    assert "RANK" not in env
    inner = argv[argv.index("aorta-inner") + 1 :]
    assert inner[1] == "-c"
    assert "run_worker_rank" in inner[2]
    assert "matrix.yaml" not in argv
    assert "triage" not in argv
    assert forwarded["steps"] == 20
    assert forwarded["device"] == "cpu"
    assert forwarded["model"]["hidden_size"] == 256
    assert forwarded["checks"]["fail_on_rank_loss_mismatch"] is True
    assert forwarded["optimizer"]["lr"] == 0.001
    assert "launcher" not in forwarded
    assert "nodes" not in forwarded
    assert "_aorta_cell" not in forwarded
    assert captured.get("start_new_session") is True
    assert result.passed is True
    assert result.metrics.get("launcher") == "spur"
    assert result.metrics.get("world_size") == 16
    assert result.metrics.get("final_loss") == 5.5


def test_unreadable_rank0_result_fails_the_trial(monkeypatch, tmp_path) -> None:
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
        del argv, kwargs
        out = Path((env or {})["AORTA_RESULTS_DIR"])
        (out / "rank0.json").write_text("{", encoding="utf-8")
        return _Proc()

    monkeypatch.setattr("aorta.workloads.distributed_launch.subprocess.Popen", fake_popen)
    monkeypatch.setenv("AORTA_MASTER_PORT", "29501")
    monkeypatch.setenv("AORTA_RANK0_IP", "10.0.0.8")
    monkeypatch.setenv("AORTA_SHARED_DIR", str(tmp_path))
    result = launch_spur({"steps": 1}, nodes=1, gpus_per_node=1, spur_bin="spur")
    assert result.passed is False
    errors = [
        item.get("error")
        for item in result.failure_details
        if isinstance(item.get("problems"), list) and "worker_result_unreadable" in item["problems"]
    ]
    assert errors
    assert "rank0.json" in str(errors[0])


def test_spur_requires_shared_dir(monkeypatch) -> None:
    monkeypatch.setenv("AORTA_MASTER_PORT", "29501")
    monkeypatch.setenv("AORTA_RANK0_IP", "10.0.0.8")
    monkeypatch.delenv("AORTA_SHARED_DIR", raising=False)
    with pytest.raises(RuntimeError, match="AORTA_SHARED_DIR"):
        launch_spur({"steps": 1}, nodes=1, gpus_per_node=1, spur_bin="spur")


def test_rank0_ip_must_be_ipv4(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AORTA_MASTER_PORT", "29501")
    monkeypatch.setenv("AORTA_RANK0_IP", "fe80::1")
    monkeypatch.setenv("AORTA_SHARED_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="AORTA_RANK0_IP"):
        launch_spur({"steps": 1}, nodes=1, gpus_per_node=1, spur_bin="spur")


def test_run_worker_rank_publishes_only_rank0(monkeypatch, tmp_path) -> None:
    def fake_rank(config: dict) -> dict:
        assert config["steps"] == 3
        return {
            "passed": True,
            "failure_count": 0,
            "metrics": {"world_size": 2, "final_loss": 1.0, "rank": 0},
        }

    monkeypatch.setattr("aorta.workloads.distributed_launch.run_training_rank", fake_rank)
    monkeypatch.setenv("AORTA_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "1")
    run_worker_rank({"steps": 3})
    assert not (tmp_path / "rank0.json").exists()

    monkeypatch.setenv("RANK", "0")
    run_worker_rank({"steps": 3})
    parsed, error = _read_rank0_result(tmp_path)
    assert error is None
    assert isinstance(parsed, dict)
    assert parsed.get("passed") is True
    assert parsed["metrics"]["final_loss"] == 1.0


def test_terminate_kills_children_that_survive_the_leader(monkeypatch) -> None:
    monkeypatch.setattr("aorta.workloads.distributed_launch._TERMINATE_GRACE_SEC", 0.3)
    monkeypatch.setattr("aorta.workloads.distributed_launch._REAP_AFTER_KILL_SEC", 2.0)
    code = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    time.sleep(60)\n"
        "    os._exit(0)\n"
        "os._exit(0)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    pgid = proc.pid
    try:
        proc.wait(timeout=5)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not _group_has_members(pgid):
            time.sleep(0.05)
        assert _group_has_members(pgid)
        _terminate_process_tree(proc, pgid=pgid)
        assert not _group_has_members(pgid)
    finally:
        if _group_has_members(pgid):
            os.killpg(pgid, signal.SIGKILL)


def test_spur_requires_explicit_master_port(monkeypatch) -> None:
    monkeypatch.delenv("AORTA_MASTER_PORT", raising=False)
    monkeypatch.setattr(
        "aorta.workloads.distributed_launch.controller_ipv4",
        lambda: "10.0.0.8",
    )
    with pytest.raises(RuntimeError, match="AORTA_MASTER_PORT"):
        launch_spur({"steps": 1}, nodes=1, gpus_per_node=1, spur_bin="spur")


def test_inherited_rank_env_still_spawns(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.delenv("AORTA_INNER", raising=False)
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    workload = TinyTransformerWorkload(
        {
            "launcher": "spur",
            "nodes": 2,
            "gpus_per_node": 8,
            "expected_world_size": 16,
            "device": "cpu",
        }
    )
    workload.setup()
    assert workload._train_here is False


def test_inner_and_elastic_ranks_train_here(monkeypatch) -> None:
    spec = {
        "launcher": "spur",
        "nodes": 1,
        "gpus_per_node": 1,
        "expected_world_size": 1,
        "device": "cpu",
    }
    monkeypatch.setenv("AORTA_INNER", "1")
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    inner = TinyTransformerWorkload(spec)
    inner.setup()
    assert inner._train_here is True

    monkeypatch.delenv("AORTA_INNER", raising=False)
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    elastic = TinyTransformerWorkload(spec)
    elastic.setup()
    assert elastic._train_here is True


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


def _spur_worker_local_rank(env: dict[str, str]) -> str:
    """Run the worker prologue and print the LOCAL_RANK it exports."""
    cleaned = dict(os.environ)
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "AORTA_INNER",
        "AORTA_GPUS_PER_NODE",
    ):
        cleaned.pop(key, None)
    cleaned.update(env)
    proc = subprocess.run(
        ["bash", "-c", SPUR_WORKER_SCRIPT, "aorta-inner", "bash", "-c", 'printf %s "$LOCAL_RANK"'],
        env=cleaned,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_spur_local_rank_uses_slurm_localid_under_cyclic_placement() -> None:
    """Global ranks 0 and 2 share a node under cyclic 2x2; local ids stay distinct."""
    common = {
        "WORLD_SIZE": "4",
        "MASTER_ADDR": "10.0.0.8",
        "AORTA_INNER": "1",
        "AORTA_GPUS_PER_NODE": "2",
    }
    assert _spur_worker_local_rank({**common, "SLURM_PROCID": "0", "SLURM_LOCALID": "0"}) == "0"
    assert _spur_worker_local_rank({**common, "SLURM_PROCID": "2", "SLURM_LOCALID": "1"}) == "1"
    # Block fallback when the scheduler did not publish a local id.
    assert _spur_worker_local_rank({**common, "SLURM_PROCID": "2"}) == "0"


def test_spur_worker_rejects_a_non_numeric_local_id() -> None:
    proc = subprocess.run(
        ["bash", "-c", SPUR_WORKER_SCRIPT, "aorta-inner", "true"],
        env={
            **os.environ,
            "SLURM_PROCID": "0",
            "SLURM_LOCALID": "gpu0",
            "AORTA_GPUS_PER_NODE": "2",
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "10.0.0.8",
            "AORTA_INNER": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "SLURM_LOCALID" in proc.stderr


def test_monarch_launcher_names_source_build_when_missing(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "monarch" or name.startswith("monarch."):
            raise ImportError("monarch is not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="pytorch_monarch"):
        launch_monarch({"steps": 1}, nodes=2, gpus_per_node=8)


def _install_fake_monarch(monkeypatch, train_result):
    """Replace the Monarch import and the srun child. ``train_result`` is returned or raised."""
    import types

    stopped: list[object] = []
    captured: dict[str, object] = {}

    class _Actor:
        def enable_transport(self, url: str) -> None:
            captured["transport"] = url

        def attach_to_workers(self, **kwargs: object) -> object:
            captured["attach"] = kwargs

            class _Hosts:
                def shutdown(self) -> object:
                    captured["shutdowns"] = int(captured.get("shutdowns", 0)) + 1

                    async def _done() -> None:
                        return None

                    return _done()

            return _Hosts()

    def bootstrap(port: int, _auth: object) -> str:
        return f"boot-{port}"

    trainer_mod = types.ModuleType("aorta.workloads.monarch_trainer")
    trainer_mod.Trainer = type("Trainer", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aorta.workloads.monarch_trainer", trainer_mod)
    monkeypatch.setattr(
        "aorta.workloads.distributed_launch._import_monarch",
        lambda: (
            _Actor(),
            object(),
            bootstrap,
            "SERVICE_PROC_IDS",
            lambda nodes: [f"id-{index}" for index in range(nodes)],
            lambda ids: ",".join(ids),
            lambda addrs, ids: list(zip(addrs, ids, strict=True)),
        ),
    )
    monkeypatch.setattr(
        "aorta.workloads.distributed_launch._monarch_hostnames",
        lambda job_id, nodes: [f"node-{index}" for index in range(nodes)],
    )

    async def fake_train(hosts, spmd_mod, trainer_cls, gpus_per_node, config):
        del hosts, spmd_mod, trainer_cls
        captured["gpus_per_node"] = gpus_per_node
        captured["config"] = config
        if isinstance(train_result, Exception):
            raise train_result
        return train_result

    monkeypatch.setattr("aorta.workloads.distributed_launch._monarch_train", fake_train)

    class _Proc:
        pid = None
        returncode = None

        def poll(self) -> None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.returncode = 0
            return 0

        def send_signal(self, sig: int) -> None:
            raise AssertionError(f"monarch test signalled a process ({sig})")

    def fake_popen(argv, env=None, **kwargs):
        captured["argv"] = list(argv)
        captured["env"] = dict(env or {})
        captured["start_new_session"] = kwargs.get("start_new_session")
        captured["stdout"] = kwargs.get("stdout")
        captured["stderr"] = kwargs.get("stderr")
        return _Proc()

    monkeypatch.setattr("aorta.workloads.distributed_launch.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "aorta.workloads.distributed_launch._terminate_process_tree",
        lambda proc, pgid=None: stopped.append(proc),
    )
    monkeypatch.setenv("AORTA_CONTROLLER_IP", "10.1.2.3")
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.delenv("MONARCH_SLURM_JOB", raising=False)
    monkeypatch.delenv("SPUR_JOB_ID", raising=False)
    monkeypatch.setenv("MONARCH_PORT", "22222")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "16")
    return captured, stopped


def test_monarch_launch_selects_rank0_and_stops_workers(monkeypatch) -> None:
    captured, stopped = _install_fake_monarch(
        monkeypatch,
        {
            "later-rank": {
                "passed": True,
                "failure_count": 0,
                "metrics": {"rank": 1, "world_size": 16, "final_loss": 9.0},
            },
            "rank-zero": {
                "passed": True,
                "failure_count": 0,
                "metrics": {"rank": 0, "world_size": 16, "final_loss": 1.25},
            },
        },
    )
    result = launch_monarch(
        {"steps": 4, "device": "cpu", "launcher": "monarch", "nodes": 2},
        nodes=2,
        gpus_per_node=8,
    )
    assert result.passed is True
    assert result.metrics.get("final_loss") == 1.25
    assert result.metrics.get("world_size") == 16
    assert result.metrics.get("launcher") == "monarch"
    argv = captured.get("argv")
    env = captured.get("env")
    assert isinstance(argv, list)
    assert isinstance(env, dict)
    assert argv[:7] == [
        "srun",
        "--jobid",
        "4242",
        "--overlap",
        "-N2",
        "--ntasks=2",
        "--ntasks-per-node=1",
    ]
    assert "--kill-on-bad-exit=1" in argv
    names = [arg for arg in argv if str(arg).startswith("--job-name=")]
    assert len(names) == 1 and str(names[0]).startswith("--job-name=aorta-")
    assert captured.get("transport") == "tcp://10.1.2.3:0"
    assert captured.get("gpus_per_node") == 8
    forwarded = captured.get("config")
    assert isinstance(forwarded, dict)
    assert forwarded.get("steps") == 4
    assert "launcher" not in forwarded
    assert "RANK" not in env
    assert "LOCAL_RANK" not in env
    assert "WORLD_SIZE" not in env
    assert len(stopped) == 1
    assert captured.get("shutdowns") == 1


def test_monarch_launch_failure_stops_workers(monkeypatch) -> None:
    _captured, stopped = _install_fake_monarch(monkeypatch, RuntimeError("mesh failed"))
    result = launch_monarch({"steps": 1, "device": "cpu"}, nodes=2, gpus_per_node=8)
    assert result.passed is False
    problems: list[object] = []
    for item in result.failure_details:
        found = item.get("problems")
        if isinstance(found, list):
            problems.extend(found)
    assert "monarch_launch" in problems
    assert result.metrics.get("launcher") == "monarch"
    assert len(stopped) == 1


def test_monarch_srun_requests_one_task_per_node(monkeypatch) -> None:
    from aorta.workloads.distributed_launch import _monarch_hostnames, monarch_step_prefix

    seen: dict[str, list[str]] = {}

    def fake_check_output(argv, text=True):
        del text
        seen["argv"] = list(argv)
        return "0 node-a\n1 node-b\n"

    monkeypatch.setattr(
        "aorta.workloads.distributed_launch.subprocess.check_output",
        fake_check_output,
    )
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    monkeypatch.delenv("MONARCH_SLURM_JOB", raising=False)
    monkeypatch.delenv("SPUR_JOB_ID", raising=False)
    prefix = monarch_step_prefix(scheduler="slurm", nodes=2)
    assert _monarch_hostnames(prefix, 2) == ["node-a", "node-b"]
    argv = seen["argv"]
    assert argv[:4] == ["srun", "--jobid", "99", "--overlap"]
    assert "--ntasks=2" in argv
    assert "--ntasks-per-node=1" in argv


def test_monarch_spur_prefix_is_a_step_of_the_spur_allocation(monkeypatch) -> None:
    from aorta.workloads.distributed_launch import monarch_step_prefix

    monkeypatch.setenv("SPUR_JOB_ID", "17")
    prefix = monarch_step_prefix(scheduler="spur", nodes=2, spur_bin="/opt/spur/bin/spur")
    assert prefix == [
        "/opt/spur/bin/spur",
        "srun",
        "-N2",
        "--ntasks=2",
        "--ntasks-per-node=1",
    ]
    assert "--jobid" not in prefix
    assert "--overlap" not in prefix


def test_monarch_spur_needs_a_spur_allocation(monkeypatch) -> None:
    from aorta.workloads.distributed_launch import monarch_step_prefix

    monkeypatch.delenv("SPUR_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="spur salloc"):
        monarch_step_prefix(scheduler="spur", nodes=2)


def test_monarch_slurm_rejects_a_spur_job_id(monkeypatch) -> None:
    """Inside a Spur allocation SLURM_JOB_ID is Spur's id, not a Slurm job."""
    from aorta.workloads.distributed_launch import monarch_step_prefix

    monkeypatch.setenv("SPUR_JOB_ID", "17")
    monkeypatch.setenv("SLURM_JOB_ID", "17")
    monkeypatch.delenv("MONARCH_SLURM_JOB", raising=False)
    with pytest.raises(RuntimeError, match="scheduler: spur"):
        monarch_step_prefix(scheduler="slurm", nodes=2)
    monkeypatch.setenv("MONARCH_SLURM_JOB", "4242")
    assert monarch_step_prefix(scheduler="slurm", nodes=2)[:3] == ["srun", "--jobid", "4242"]


def test_monarch_launch_on_spur_uses_spur_steps(monkeypatch, tmp_path) -> None:
    captured, stopped = _install_fake_monarch(
        monkeypatch,
        {"rank0": {"passed": True, "metrics": {"rank": 0, "world_size": 16, "final_loss": 2.0}}},
    )
    seen: dict[str, list[str]] = {}

    def fake_hostnames(prefix, nodes):
        seen["prefix"] = list(prefix)
        return [f"node-{index}" for index in range(nodes)]

    monkeypatch.setattr("aorta.workloads.distributed_launch._monarch_hostnames", fake_hostnames)
    monkeypatch.setenv("SPUR_JOB_ID", "17")
    monkeypatch.setenv("AORTA_SHARED_DIR", str(tmp_path))
    result = launch_monarch(
        {"steps": 2, "device": "cpu"},
        nodes=2,
        gpus_per_node=8,
        scheduler="spur",
        spur_bin="/opt/spur/bin/spur",
    )
    assert result.passed is True
    step = ["/opt/spur/bin/spur", "srun", "-N2", "--ntasks=2", "--ntasks-per-node=1"]
    assert seen["prefix"] == step
    argv = captured.get("argv")
    assert isinstance(argv, list)
    assert argv[:5] == step
    assert "--jobid" not in argv
    assert "--kill-on-bad-exit=1" not in argv
    assert not any(str(arg).startswith("--job-name=") for arg in argv)
    script = next(arg for arg in argv if isinstance(arg, str) and "aorta-owned-step" in arg)
    assert "AORTA_STEP_CANCEL" in script
    assert 'exec "$@"' not in script
    worker_env = captured.get("env")
    assert isinstance(worker_env, dict)
    cancel = worker_env.get("AORTA_STEP_CANCEL")
    assert isinstance(cancel, str) and cancel.endswith("aorta-step-cancel")
    assert len(stopped) == 1


def test_scheduler_applies_only_to_monarch() -> None:
    assert TinyTransformerConfig.from_dict({"launcher": "monarch"}).scheduler == "slurm"
    spec = TinyTransformerConfig.from_dict({"launcher": "monarch", "scheduler": "spur"})
    assert spec.scheduler == "spur"
    with pytest.raises(ValueError, match="scheduler"):
        TinyTransformerConfig.from_dict({"launcher": "monarch", "scheduler": "pbs"})
    with pytest.raises(ValueError, match="only to launcher 'monarch'"):
        TinyTransformerConfig.from_dict({"launcher": "spur", "scheduler": "spur"})


def test_launcher_subprocess_logs_are_siblings_of_the_trial_log(monkeypatch, tmp_path) -> None:
    """Child output goes to ``<prefix>.subprocess.*.log``, left open until the child is gone."""
    prefix = tmp_path / "trial_d0_m0_t0"
    config = {
        "steps": 1,
        "device": "cpu",
        "_aorta_save_logs": True,
        "_aorta_log_prefix": str(prefix),
    }

    class _Proc:
        pid = None
        returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def poll(self) -> int:
            return 0

        def send_signal(self, sig: int) -> None:
            raise AssertionError(sig)

    def fake_popen(argv, env=None, stdout=None, stderr=None, **kwargs):
        del argv, env, kwargs
        assert stdout is not None and stderr is not None
        assert not stdout.closed
        stdout.write("spur worker\n")
        stdout.flush()
        out = Path(env_holder["results"])
        (out / "rank0.json").write_text(
            json.dumps({"result": {"passed": True, "failure_count": 0, "metrics": {}}}),
            encoding="utf-8",
        )
        return _Proc()

    env_holder: dict[str, str] = {}

    def recording_popen(argv, env=None, stdout=None, stderr=None, **kwargs):
        env_holder["results"] = env["AORTA_RESULTS_DIR"]
        return fake_popen(argv, env, stdout, stderr, **kwargs)

    monkeypatch.setattr(
        "aorta.workloads.distributed_launch.subprocess.Popen",
        recording_popen,
    )
    monkeypatch.setenv("AORTA_MASTER_PORT", "29501")
    monkeypatch.setenv("AORTA_RANK0_IP", "10.0.0.8")
    monkeypatch.setenv("AORTA_SHARED_DIR", str(tmp_path))
    result = launch_spur(config, nodes=1, gpus_per_node=1, spur_bin="spur")
    assert result.passed is True
    stdout_path = Path(f"{prefix}.subprocess.stdout.log")
    stderr_path = Path(f"{prefix}.subprocess.stderr.log")
    assert stdout_path.read_text(encoding="utf-8") == "spur worker\n"
    assert stderr_path.is_file()
    assert not Path(f"{prefix}.stdout.log").exists()

    captured, stopped = _install_fake_monarch(monkeypatch, {"rank0": {"passed": True, "metrics": {"rank": 0}}})
    monarch = launch_monarch(config, nodes=2, gpus_per_node=8)
    assert monarch.passed is True
    monarch_out = captured.get("stdout")
    monarch_err = captured.get("stderr")
    assert getattr(monarch_out, "name", None) == str(stdout_path)
    assert getattr(monarch_err, "name", None) == str(stderr_path)
    assert getattr(monarch_out, "closed", False) is True
    assert len(stopped) == 1
    assert captured.get("shutdowns") == 1


def test_inherited_rank_controller_still_writes_the_trial(tmp_path, monkeypatch) -> None:
    """RANK inherited from an old shell must not drop the controller's trial JSON."""
    from unittest.mock import MagicMock, patch

    from aorta.run.dispatcher import RunRequest, run_trials
    from aorta.workloads._base import WorkloadResult

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.delenv("AORTA_INNER", raising=False)
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    monkeypatch.setattr(
        "aorta.workloads.tiny_transformer.launch_spur",
        lambda *_args, **_kwargs: WorkloadResult(
            passed=True,
            total_iterations=1,
            elapsed_sec=0.1,
            metrics={"launcher": "spur", "world_size": 16, "final_loss": 1.0},
        ),
    )
    mock_ep = MagicMock()
    mock_ep.name = "tiny_transformer"
    mock_ep.load.return_value = TinyTransformerWorkload
    mock_eps = MagicMock()
    mock_eps.select.return_value = [mock_ep]
    with patch("importlib.metadata.entry_points", return_value=mock_eps):
        results = run_trials(
            RunRequest(
                workload="tiny_transformer",
                trials=1,
                results_dir=tmp_path,
                config_overrides={
                    "launcher": "spur",
                    "nodes": 2,
                    "gpus_per_node": 8,
                    "device": "cpu",
                },
            )
        )
    assert len(results) == 1
    assert results[0].exit_status == "ok"
    assert list(tmp_path.rglob("trial_*.json"))


def test_elastic_peer_rank_does_not_write_the_trial(tmp_path, monkeypatch) -> None:
    """A real torchrun rank still leaves the trial file to rank 0."""
    from unittest.mock import MagicMock, patch

    from aorta.run.dispatcher import RunRequest, run_trials

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "elastic-1")
    monkeypatch.delenv("AORTA_INNER", raising=False)
    monkeypatch.setattr(
        "aorta.workloads.tiny_transformer.run_training_rank",
        lambda _config: {
            "passed": True,
            "failure_count": 0,
            "metrics": {"world_size": 16, "final_loss": 1.0, "rank": 1},
        },
    )
    mock_ep = MagicMock()
    mock_ep.name = "tiny_transformer"
    mock_ep.load.return_value = TinyTransformerWorkload
    mock_eps = MagicMock()
    mock_eps.select.return_value = [mock_ep]
    with patch("importlib.metadata.entry_points", return_value=mock_eps):
        results = run_trials(
            RunRequest(
                workload="tiny_transformer",
                trials=1,
                results_dir=tmp_path,
                config_overrides={
                    "launcher": "spur",
                    "nodes": 2,
                    "gpus_per_node": 8,
                    "device": "cpu",
                },
            )
        )
    assert len(results) == 1
    assert not list(tmp_path.rglob("trial_*.json"))


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


def test_subprocess_log_refuses_a_symlink_leaf(tmp_path) -> None:
    """A planted log name must not truncate the file it points at."""
    from aorta.run._fsafe import TrustedAnchor
    from aorta.workloads.distributed_launch import _open_subprocess_logs

    root = tmp_path / "results"
    workload = root / "tiny_transformer"
    workload.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    prefix = workload / "trial_d0_m0_t0"
    (workload / "trial_d0_m0_t0.subprocess.stdout.log").symlink_to(outside)
    anchor = TrustedAnchor.freeze(root)
    opened = _open_subprocess_logs(
        {
            "_aorta_save_logs": True,
            "_aorta_log_prefix": str(prefix),
            "_aorta_results_root": str(anchor.path),
            "_aorta_results_root_id": list(anchor.identity or ()),
        }
    )
    assert opened == (None, None)
    assert outside.read_text(encoding="utf-8") == "keep"


def test_subprocess_log_refuses_a_symlink_directory(tmp_path) -> None:
    """A workload directory that is a symlink must not receive the child log."""
    from aorta.run._fsafe import TrustedAnchor
    from aorta.workloads.distributed_launch import _open_subprocess_logs

    root = tmp_path / "results"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "trial_d0_m0_t0.subprocess.stdout.log"
    secret.write_text("keep", encoding="utf-8")
    (root / "tiny_transformer").symlink_to(outside, target_is_directory=True)
    anchor = TrustedAnchor.freeze(root)
    opened = _open_subprocess_logs(
        {
            "_aorta_save_logs": True,
            "_aorta_log_prefix": str(root / "tiny_transformer" / "trial_d0_m0_t0"),
            "_aorta_results_root": str(anchor.path),
            "_aorta_results_root_id": list(anchor.identity or ()),
        }
    )
    assert opened == (None, None)
    assert secret.read_text(encoding="utf-8") == "keep"


def test_subprocess_log_writes_through_the_frozen_root(tmp_path) -> None:
    from aorta.run._fsafe import TrustedAnchor
    from aorta.workloads.distributed_launch import _open_subprocess_logs

    root = tmp_path / "results"
    workload = root / "tiny_transformer"
    workload.mkdir(parents=True)
    prefix = workload / "trial_d0_m0_t0"
    anchor = TrustedAnchor.freeze(root)
    stdout, stderr = _open_subprocess_logs(
        {
            "_aorta_save_logs": True,
            "_aorta_log_prefix": str(prefix),
            "_aorta_results_root": str(anchor.path),
            "_aorta_results_root_id": list(anchor.identity or ()),
        }
    )
    assert stdout is not None and stderr is not None
    stdout.write("child\n")
    stdout.close()
    stderr.close()
    written = workload / "trial_d0_m0_t0.subprocess.stdout.log"
    assert written.read_text(encoding="utf-8") == "child\n"


def test_cancel_names_only_the_owned_step(monkeypatch) -> None:
    """``scancel`` gets ``job.step`` for the token, never the allocation id."""
    from aorta.workloads.distributed_launch import _cancel_scheduler_step, _OwnedStep

    calls: list[list[str]] = []

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        job = argv[-1]
        if "show" in argv and argv[0] == "scontrol":
            # Slurm's multi-line block. The step name is ``Name=``.
            stdout = "\n".join(
                [
                    f"StepId={job}.3 UserId=1 StartTime=now TimeLimit=UNLIMITED",
                    "   Nodes=2 CPUs=2 Tasks=2 Name=aorta-9-abcd Network=(null)",
                    f"StepId={job}.4 UserId=1 StartTime=now TimeLimit=UNLIMITED",
                    "   Nodes=2 CPUs=2 Tasks=2 Name=bash Network=(null)",
                    "StepId=99.3 UserId=1 StartTime=now TimeLimit=UNLIMITED",
                    "   Nodes=1 CPUs=1 Tasks=1 Name=aorta-9-abcd Network=(null)",
                    f"StepId={job}.batch UserId=1 StartTime=now TimeLimit=UNLIMITED",
                    "   Nodes=1 CPUs=1 Tasks=1 Name=batch Network=(null)",
                ]
            )
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
        if "show" in argv:
            stdout = "\n".join(
                [
                    f"StepId={job}.3 StepName=bash -c # aorta-owned-step aorta-9-abcd",
                    f"StepId={job}.4 StepName=other-step",
                    "StepId=99.3 StepName=bash -c # aorta-owned-step aorta-9-abcd",
                    f"StepId={job}.batch StepName=allocation",
                ]
            )
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr("aorta.workloads.distributed_launch._run_scheduler", runner)
    _cancel_scheduler_step(
        _OwnedStep(scheduler="slurm", job_id="4242", token="aorta-9-abcd", spur_bin="spur")
    )
    _cancel_scheduler_step(
        _OwnedStep(
            scheduler="spur",
            job_id="17",
            token="aorta-9-abcd",
            spur_bin="/opt/spur/bin/spur",
        )
    )
    assert calls[0] == ["scontrol", "show", "step", "4242"]
    assert calls[1] == ["scancel", "--quiet", "4242.3"]
    assert calls[2] == ["/opt/spur/bin/spur", "scontrol", "show", "step", "17"]
    assert calls[3] == ["/opt/spur/bin/spur", "scancel", "--quiet", "17.3"]
    cancelled = [argv[-1] for argv in calls if "scancel" in argv]
    assert cancelled == ["4242.3", "17.3"]


def test_spur_launch_requests_gpus_unless_the_device_is_cpu(tmp_path) -> None:
    """A standalone GPU job reserves devices. A CPU job does not."""
    from aorta.workloads.distributed_launch import build_spur_launch

    common = {
        "spur_bin": "spur",
        "nodes": 2,
        "gpus_per_node": 8,
        "master_port": 29501,
        "config_path": tmp_path / "worker_config.json",
        "results_dir": tmp_path,
        "step_token": "aorta-1-abcd",
    }
    gpu_argv, gpu_env = build_spur_launch(**common, master_addr="", device="auto")
    assert gpu_argv[gpu_argv.index("--gpus-per-node") + 1] == "8"
    assert "MASTER_ADDR" not in gpu_env
    cpu_argv, cpu_env = build_spur_launch(**common, master_addr="10.0.0.8", device="cpu")
    assert "--gpus-per-node" not in cpu_argv
    assert cpu_env.get("MASTER_ADDR") == "10.0.0.8"


def test_spur_worker_publishes_rank0_address(tmp_path) -> None:
    """Rank 0 of the training step publishes MASTER_ADDR; another rank reads it."""
    env = {
        **os.environ,
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "AORTA_GPUS_PER_NODE": "1",
        "WORLD_SIZE": "1",
        "AORTA_INNER": "1",
        "AORTA_RESULTS_DIR": str(tmp_path),
    }
    env.pop("MASTER_ADDR", None)
    rank0 = subprocess.run(
        ["bash", "-c", SPUR_WORKER_SCRIPT, "aorta-inner", "bash", "-c", 'printf %s "$MASTER_ADDR"'],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert rank0.returncode == 0
    published = (tmp_path / "rank0.addr").read_text(encoding="utf-8").strip()
    assert is_ipv4(published)
    assert rank0.stdout.strip() == published

    env["SLURM_PROCID"] = "1"
    env["SLURM_LOCALID"] = "1"
    rank1 = subprocess.run(
        ["bash", "-c", SPUR_WORKER_SCRIPT, "aorta-inner", "bash", "-c", 'printf %s "$MASTER_ADDR"'],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert rank1.returncode == 0
    assert rank1.stdout.strip() == published


def test_spur_abandon_cancels_the_step_before_deleting_its_files(monkeypatch, tmp_path) -> None:
    """Ctrl-C publishes the cancel file, then deletes it only after the step is gone."""
    from aorta.workloads import distributed_launch

    calls: list[list[str]] = []
    shows = {"n": 0}

    class _Proc:
        def __init__(self) -> None:
            self.pid = None
            self.returncode: int | None = None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.returncode is None:
                raise KeyboardInterrupt
            return self.returncode

        def poll(self) -> int | None:
            return self.returncode

        def send_signal(self, sig: int) -> None:
            self.returncode = -int(sig)

    def fake_popen(argv, env=None, **kwargs):
        del kwargs
        calls.append(["popen", *argv])
        out = Path((env or {})["AORTA_RESULTS_DIR"])
        (out / "rank0.json").write_text("{}", encoding="utf-8")
        return _Proc()

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        command = " ".join(str(part) for part in calls[0])
        token = command.split("# aorta-owned-step ", 1)[1].split()[0]
        if "show" in argv:
            shows["n"] += 1
            found = list(Path(os.environ["AORTA_SHARED_DIR"]).glob("*/aorta-step-cancel"))
            assert found and found[0].is_file()
            if shows["n"] == 1:
                stdout = f"StepId=17.2 StepName=bash -c # aorta-owned-step {token}\n"
                return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(distributed_launch.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(distributed_launch, "_run_scheduler", runner)
    monkeypatch.setenv("AORTA_MASTER_PORT", "29501")
    monkeypatch.setenv("AORTA_RANK0_IP", "10.0.0.8")
    monkeypatch.setenv("AORTA_SHARED_DIR", str(tmp_path))
    monkeypatch.setenv("SPUR_JOB_ID", "17")
    with pytest.raises(KeyboardInterrupt):
        launch_spur({"steps": 1}, nodes=1, gpus_per_node=1, spur_bin="/opt/spur/bin/spur")
    assert calls[1] == ["/opt/spur/bin/spur", "scontrol", "show", "step", "17"]
    assert calls[2] == ["/opt/spur/bin/spur", "scancel", "--quiet", "17.2"]
    assert shows["n"] >= 2
    assert not list(tmp_path.glob("*/aorta-step-cancel"))


def test_spur_abandon_keeps_the_cancel_file_while_the_step_is_listed(monkeypatch, tmp_path) -> None:
    """A step that stays listed keeps the marker a late poll can still see."""
    from aorta.workloads import distributed_launch

    launched: list[list[str]] = []

    class _Proc:
        def __init__(self) -> None:
            self.pid = None
            self.returncode: int | None = None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.returncode is None:
                raise KeyboardInterrupt
            return self.returncode

        def poll(self) -> int | None:
            return self.returncode

        def send_signal(self, sig: int) -> None:
            self.returncode = -int(sig)

    def fake_popen(argv, env=None, **kwargs):
        del kwargs
        launched.append(list(argv))
        out = Path((env or {})["AORTA_RESULTS_DIR"])
        (out / "rank0.json").write_text("{}", encoding="utf-8")
        return _Proc()

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        token = " ".join(launched[0]).split("# aorta-owned-step ", 1)[1].split()[0]
        if "show" in argv:
            stdout = f"StepId=17.2 StepName=# aorta-owned-step {token}\n"
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(distributed_launch, "_STEP_GONE_TIMEOUT_SEC", 0)
    monkeypatch.setattr(distributed_launch.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(distributed_launch, "_run_scheduler", runner)
    monkeypatch.setenv("AORTA_MASTER_PORT", "29501")
    monkeypatch.setenv("AORTA_RANK0_IP", "10.0.0.8")
    monkeypatch.setenv("AORTA_SHARED_DIR", str(tmp_path))
    monkeypatch.setenv("SPUR_JOB_ID", "17")
    with pytest.raises(KeyboardInterrupt):
        launch_spur({"steps": 1, "device": "cpu"}, nodes=1, gpus_per_node=1, spur_bin="spur")
    assert list(tmp_path.glob("*/aorta-step-cancel"))


def _spur_client(returncode: int | None):
    """Fake ``spur srun`` client: exits with ``returncode``, or Ctrl-C when ``None``."""

    class _Proc:
        def __init__(self) -> None:
            self.pid = None
            self.returncode: int | None = None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.returncode is None:
                if returncode is None:
                    raise KeyboardInterrupt
                self.returncode = returncode
            return self.returncode

        def poll(self) -> int | None:
            return self.returncode

        def send_signal(self, sig: int) -> None:
            self.returncode = -int(sig)

    launched: list[list[str]] = []

    def fake_popen(argv, env=None, **kwargs):
        del kwargs
        launched.append(list(argv))
        out = Path((env or {})["AORTA_RESULTS_DIR"])
        (out / "rank0.json").write_text("{}", encoding="utf-8")
        return _Proc()

    return fake_popen, launched


def _spur_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AORTA_MASTER_PORT", "29501")
    monkeypatch.setenv("AORTA_RANK0_IP", "10.0.0.8")
    monkeypatch.setenv("AORTA_SHARED_DIR", str(tmp_path))
    monkeypatch.setenv("SPUR_JOB_ID", "17")


def test_spur_client_failure_cancels_a_step_that_is_still_listed(monkeypatch, tmp_path) -> None:
    """``spur srun`` exiting 1 is not proof the buffered step ended."""
    from aorta.workloads import distributed_launch

    fake_popen, launched = _spur_client(1)
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        token = " ".join(launched[0]).split("# aorta-owned-step ", 1)[1].split()[0]
        if "show" in argv:
            stdout = f"StepId=17.2 StepName=bash -c # aorta-owned-step {token}\n"
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(distributed_launch, "_STEP_GONE_TIMEOUT_SEC", 0)
    monkeypatch.setattr(distributed_launch.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(distributed_launch, "_run_scheduler", runner)
    _spur_env(monkeypatch, tmp_path)
    result = launch_spur({"steps": 1, "device": "cpu"}, nodes=1, gpus_per_node=1, spur_bin="spur")
    assert result.passed is False
    assert ["spur", "scancel", "--quiet", "17.2"] in calls
    assert list(tmp_path.glob("*/aorta-step-cancel"))


def test_spur_step_failure_that_finished_removes_its_files(monkeypatch, tmp_path) -> None:
    """A nonzero exit whose step is no longer listed is an ordinary failure."""
    from aorta.workloads import distributed_launch

    fake_popen, _launched = _spur_client(1)
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(distributed_launch.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(distributed_launch, "_run_scheduler", runner)
    _spur_env(monkeypatch, tmp_path)
    result = launch_spur({"steps": 1, "device": "cpu"}, nodes=1, gpus_per_node=1, spur_bin="spur")
    assert result.passed is False
    assert not any("scancel" in argv for argv in calls)
    assert not list(tmp_path.iterdir())


def test_second_interrupt_during_spur_cancel_keeps_the_files(monkeypatch, tmp_path) -> None:
    """Ctrl-C while waiting for the step must not delete the marker it waits on."""
    from aorta.workloads import distributed_launch

    fake_popen, _launched = _spur_client(None)

    def interrupted(_step: object) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(distributed_launch.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        distributed_launch,
        "_run_scheduler",
        lambda argv: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(distributed_launch, "_wait_owned_step_gone", interrupted)
    _spur_env(monkeypatch, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        launch_spur({"steps": 1, "device": "cpu"}, nodes=1, gpus_per_node=1, spur_bin="spur")
    assert list(tmp_path.glob("*/aorta-step-cancel"))


def test_second_interrupt_during_monarch_cancel_keeps_the_file(monkeypatch, tmp_path) -> None:
    """Monarch on Spur keeps its cancel file if the confirmation wait is interrupted."""
    from aorta.workloads import distributed_launch

    _install_fake_monarch(
        monkeypatch,
        {"rank0": {"passed": True, "metrics": {"rank": 0, "world_size": 2, "final_loss": 1.0}}},
    )

    async def _shutdown_fails(_hosts: object) -> None:
        raise RuntimeError("shutdown failed")

    def interrupted(_step: object) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(distributed_launch, "_await_mesh_shutdown", _shutdown_fails)
    monkeypatch.setattr(
        distributed_launch,
        "_run_scheduler",
        lambda argv: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(distributed_launch, "_wait_owned_step_gone", interrupted)
    monkeypatch.setattr(
        "aorta.workloads.distributed_launch._monarch_hostnames",
        lambda _prefix, nodes: [f"node-{index}" for index in range(nodes)],
    )
    monkeypatch.setenv("SPUR_JOB_ID", "17")
    monkeypatch.setenv("AORTA_SHARED_DIR", str(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        launch_monarch(
            {"steps": 1, "device": "cpu"},
            nodes=1,
            gpus_per_node=1,
            scheduler="spur",
            spur_bin="spur",
        )
    assert list(tmp_path.glob("*/aorta-step-cancel"))


def test_monarch_cancels_its_step_when_mesh_shutdown_fails(monkeypatch) -> None:
    from aorta.workloads import distributed_launch

    captured, _stopped = _install_fake_monarch(
        monkeypatch,
        {"rank0": {"passed": True, "metrics": {"rank": 0, "world_size": 2, "final_loss": 1.0}}},
    )
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        launched = captured.get("argv")
        assert isinstance(launched, list)
        names = [arg for arg in launched if str(arg).startswith("--job-name=")]
        assert len(names) == 1
        token = str(names[0]).split("=", 1)[1]
        if "show" in argv:
            # Slurm names the step after ``--job-name``; the probe step is ``bash``.
            stdout = (
                "StepId=4242.1 UserId=1 StartTime=now TimeLimit=UNLIMITED\n"
                "   Nodes=1 CPUs=1 Tasks=1 Name=bash Network=(null)\n"
                "StepId=4242.8 UserId=1 StartTime=now TimeLimit=UNLIMITED\n"
                f"   Nodes=1 CPUs=1 Tasks=1 Name={token} Network=(null)\n"
            )
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    async def _shutdown_fails(_hosts: object) -> None:
        raise RuntimeError("shutdown failed")

    monkeypatch.setattr(distributed_launch, "_await_mesh_shutdown", _shutdown_fails)
    monkeypatch.setattr(distributed_launch, "_run_scheduler", runner)
    result = launch_monarch({"steps": 1, "device": "cpu"}, nodes=1, gpus_per_node=1)
    assert result.passed is True
    assert calls[0] == ["scontrol", "show", "step", "4242"]
    assert calls[1] == ["scancel", "--quiet", "4242.8"]
    assert [argv[-1] for argv in calls if "scancel" in argv] == ["4242.8"]


def test_spur_worker_stops_when_the_cancel_file_appears(tmp_path) -> None:
    env = {
        **os.environ,
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "AORTA_GPUS_PER_NODE": "1",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "10.0.0.8",
        "AORTA_INNER": "1",
        "AORTA_RESULTS_DIR": str(tmp_path),
    }
    proc = subprocess.Popen(
        ["bash", "-c", SPUR_WORKER_SCRIPT, "aorta-inner", "sleep", "30"],
        env=env,
    )
    try:
        (tmp_path / "aorta-step-cancel").write_text("1\n", encoding="utf-8")
        assert proc.wait(timeout=5) != 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def _pid_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def _run_owned(tmp_path: Path, task: str, *, cancel: bool) -> tuple[int, int]:
    """Run ``task`` under the owned-step wrapper; return its exit and a descendant pid."""
    from aorta.workloads.distributed_launch import _marked_command

    cancel_file = tmp_path / "aorta-step-cancel"
    env = {**os.environ, "AORTA_STEP_CANCEL": str(cancel_file)}
    env.pop("AORTA_RESULTS_DIR", None)
    argv = _marked_command("aorta-1-abcd", ["bash", "-c", task])
    proc = subprocess.Popen(argv, env=env, cwd=tmp_path)
    pid_file = tmp_path / "descendant.pid"
    try:
        deadline = time.monotonic() + 5
        while not pid_file.is_file() or not pid_file.read_text(encoding="utf-8").strip():
            assert time.monotonic() < deadline, "the task never started its descendant"
            time.sleep(0.05)
        descendant = int(pid_file.read_text(encoding="utf-8"))
        if cancel:
            cancel_file.write_text("1\n", encoding="utf-8")
        code = proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    return code, descendant


def test_owned_step_cancel_reaches_a_descendant_that_ignores_sigterm(tmp_path) -> None:
    """The first process exits on SIGTERM; its child ignores it and is still killed."""
    task = "(trap '' TERM; exec sleep 30) & echo $! > descendant.pid; wait"
    code, descendant = _run_owned(tmp_path, task, cancel=True)
    try:
        assert code != 0
        assert _pid_gone(descendant, timeout=3)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(descendant, signal.SIGKILL)


def test_owned_step_reaps_a_descendant_left_by_a_clean_exit(tmp_path) -> None:
    """A task that exits 0 while its child runs on does not leave that child behind."""
    task = "sleep 30 & echo $! > descendant.pid; exit 0"
    code, descendant = _run_owned(tmp_path, task, cancel=False)
    try:
        assert code == 0
        assert _pid_gone(descendant, timeout=3)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(descendant, signal.SIGKILL)
