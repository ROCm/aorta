"""End-to-end tests for fresh-process trial isolation."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aorta.run import RunRequest, run_trials
from aorta.run._trial_worker import (
    _MAX_FAILURE_DETAILS_TOTAL,
    _resolve_backend_request,
    _synchronize_workload_result,
)
from aorta.run._worker_protocol import (
    WorkerProtocolError,
    read_envelope,
    validate_identity,
)
from aorta.run.dispatcher import _isolated_fallback_port
from aorta.workloads import WorkloadResult


@pytest.fixture
def isolated_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    module = tmp_path / "aorta_isolation_test_plugin.py"
    module.write_text(
        """\
import os
from aorta.workloads import Workload, WorkloadResult

IMPORT_VALUE = os.environ.get("ISOLATION_IMPORT_VALUE")
if IMPORT_VALUE is None:
    raise RuntimeError("startup env required before module import")
if IMPORT_VALUE == "raise-during-import":
    raise RuntimeError("child import failed")
STATE = 0

class IsolatedWorkload(Workload):
    trial_isolation_default = "process"
    trial_isolation_required = True
    trial_isolation_supported = frozenset({"process"})

    def setup(self):
        pass

    def run(self):
        global STATE
        STATE += 1
        return WorkloadResult(
            passed=True,
            total_iterations=1,
            metrics={
                "pid": os.getpid(),
                "import_value": IMPORT_VALUE,
                "state": STATE,
            },
        )
""",
        encoding="utf-8",
    )
    dist_info = tmp_path / "aorta_isolation_test-0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: aorta-isolation-test\nVersion: 0.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[aorta.workloads]\n"
        "isolated_test_workload = aorta_isolation_test_plugin:IsolatedWorkload\n"
        "[aorta.workload_policies]\n"
        "isolated_test_workload = "
        "aorta.run.validation:PROCESS_REQUIRED_POLICY\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    old_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(tmp_path)
    if old_pythonpath:
        pythonpath += os.pathsep + old_pythonpath
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    importlib.invalidate_caches()
    sys.modules.pop("aorta_isolation_test_plugin", None)
    return "isolated_test_workload"


def test_process_isolation_sets_env_before_import_and_resets_state(
    isolated_plugin: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ISOLATION_IMPORT_VALUE", raising=False)
    request = RunRequest(
        workload=isolated_plugin,
        trials=2,
        trial_isolation="auto",
        extra_env={"ISOLATION_IMPORT_VALUE": "from-cell"},
        results_dir=tmp_path,
    )

    results = run_trials(request)

    assert len(results) == 2
    metrics = [result.result["metrics"] for result in results]
    assert [entry["import_value"] for entry in metrics] == ["from-cell", "from-cell"]
    assert [entry["state"] for entry in metrics] == [1, 1]
    assert len({entry["pid"] for entry in metrics}) == 2
    assert "ISOLATION_IMPORT_VALUE" not in os.environ
    assert (tmp_path / isolated_plugin / "trial_d0_m0_t0.json").exists()
    assert (tmp_path / isolated_plugin / "trial_d0_m0_t1.json").exists()


def test_auto_backend_honors_device_preference() -> None:
    class AutoBackend:
        @classmethod
        def isolated_distributed_backend(cls, _config):
            return "auto"

    assert _resolve_backend_request(AutoBackend, {"device": "cpu"}) == "gloo"
    assert _resolve_backend_request(AutoBackend, {"device": "cuda"}) == "nccl"
    assert _resolve_backend_request(AutoBackend, {}) == "auto"


def test_process_isolation_rejects_launcher_identity_override(
    isolated_plugin: str,
    tmp_path: Path,
) -> None:
    request = RunRequest(
        workload=isolated_plugin,
        trials=1,
        trial_isolation="process",
        extra_env={"RANK": "7"},
        results_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="launcher identity"):
        run_trials(request)


@pytest.mark.skipif(os.name != "nt", reason="Windows env keys are case-insensitive")
def test_process_isolation_rejects_lowercase_windows_identity(
    isolated_plugin: str,
    tmp_path: Path,
) -> None:
    request = RunRequest(
        workload=isolated_plugin,
        trials=1,
        trial_isolation="process",
        extra_env={"rank": "7"},
        results_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="launcher identity"):
        run_trials(request)


def test_process_isolation_requires_json_compatible_config(
    isolated_plugin: str,
    tmp_path: Path,
) -> None:
    request = RunRequest(
        workload=isolated_plugin,
        trials=1,
        trial_isolation="process",
        config_overrides={"bad": object()},
        results_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="JSON-native"):
        run_trials(request)


def test_srun_fallback_requires_reserved_port_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AORTA_TRIAL_MASTER_PORT_BASE", raising=False)
    with pytest.raises(RuntimeError, match="AORTA_TRIAL_MASTER_PORT_BASE"):
        _isolated_fallback_port(0, 0)


def test_srun_fallback_uses_unique_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AORTA_TRIAL_MASTER_PORT_BASE", "30000")
    assert _isolated_fallback_port(0, 0) == "30000"
    assert _isolated_fallback_port(0, 1) == "30001"
    assert _isolated_fallback_port(1, 0) == "30097"


def test_process_isolation_rejects_lossy_tuple_config(
    isolated_plugin: str,
    tmp_path: Path,
) -> None:
    request = RunRequest(
        workload=isolated_plugin,
        trials=1,
        trial_isolation="process",
        config_overrides={"tuple": ("would", "become", "a-list")},
        results_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="JSON-native"):
        run_trials(request)


def test_worker_protocol_rejects_stale_version(tmp_path: Path) -> None:
    path = tmp_path / "response.json"
    path.write_text('{"protocol_version": 999}', encoding="utf-8")
    with pytest.raises(WorkerProtocolError, match="protocol_version"):
        read_envelope(path)


def test_worker_protocol_rejects_wrong_identity() -> None:
    with pytest.raises(WorkerProtocolError, match="nonce"):
        validate_identity(
            {"nonce": "wrong", "trial_id": "trial"},
            nonce="expected",
            trial_id="trial",
        )


def test_rank_result_aggregation_surfaces_nonzero_rank_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed as dist

    rank_zero = WorkloadResult(
        passed=True,
        failure_count=0,
        failure_details=[],
        metrics={},
    )
    rank_one = {
        "result": {
            "passed": False,
            "failure_count": 1,
            "failure_details": [{"error": "rank-one"}],
            "metrics": {},
        },
        "exit_status": "workload_failed",
    }

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def gather_object(_value, output, dst):
        assert dst == 0
        output[:] = [
            {
                "exit_status": "ok",
                "passed": True,
                "failure_count": 0,
                "first_failure_iteration": None,
                "failure_detail_count": 0,
                "failure_details": [],
            },
            {
                "exit_status": rank_one["exit_status"],
                "passed": rank_one["result"]["passed"],
                "failure_count": rank_one["result"]["failure_count"],
                "first_failure_iteration": None,
                "failure_detail_count": 1,
                "failure_details": rank_one["result"]["failure_details"],
            },
        ]

    monkeypatch.setattr(dist, "gather_object", gather_object)
    monkeypatch.setattr(dist, "broadcast_object_list", lambda _objects, src: None)

    canonical, status = _synchronize_workload_result(
        rank_zero,
        "ok",
        rank_local=True,
    )
    assert status == "workload_failed"
    assert canonical.passed is False
    assert canonical.failure_count == 1
    assert canonical.failure_details == [{"error": "rank-one", "rank": 1}]
    assert canonical.metrics["_rank_outcomes"] == {
        "passed": 1,
        "failed": 1,
        "world_size": 2,
    }


def test_global_result_scope_does_not_multiply_failure_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed as dist

    result = WorkloadResult(
        passed=False,
        failure_count=1,
        failure_details=[{"error": "global"}],
        metrics={},
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def gather_object(value, output, dst):
        assert dst == 0
        output[:] = [value, value]

    monkeypatch.setattr(dist, "gather_object", gather_object)
    monkeypatch.setattr(dist, "broadcast_object_list", lambda _objects, src: None)

    canonical, status = _synchronize_workload_result(
        result,
        "workload_failed",
        rank_local=False,
    )
    assert status == "workload_failed"
    assert canonical.failure_count == 1
    assert canonical.failure_details == [{"error": "global"}]


def test_rank_result_aggregation_bounds_failure_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed as dist

    world_size = 40
    details_per_rank = 20
    result = WorkloadResult(
        passed=False,
        failure_count=details_per_rank,
        failure_details=[
            {"error": f"rank-zero-{index}"}
            for index in range(details_per_rank)
        ],
        metrics={},
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)

    def gather_object(value, output, dst):
        assert dst == 0
        output[:] = [
            {
                **value,
                "failure_detail_count": details_per_rank,
                "failure_details": [
                    {"error": f"rank-{rank}-{index}"}
                    for index in range(8)
                ],
            }
            for rank in range(world_size)
        ]

    monkeypatch.setattr(dist, "gather_object", gather_object)
    monkeypatch.setattr(dist, "broadcast_object_list", lambda _objects, src: None)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda *_args, **_kwargs: pytest.fail("full-result all_gather must not be used"),
    )

    canonical, _status = _synchronize_workload_result(
        result,
        "workload_failed",
        rank_local=True,
    )
    assert len(canonical.failure_details) == _MAX_FAILURE_DETAILS_TOTAL
    assert canonical.failure_count == world_size * details_per_rank
    assert canonical.metrics["_failure_details_omitted"] == (
        world_size * details_per_rank - _MAX_FAILURE_DETAILS_TOTAL
    )


def test_distributed_setup_failure_escalates_without_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    result = WorkloadResult(passed=False, failure_count=1)
    with pytest.raises(RuntimeError, match="before rank synchronization"):
        _synchronize_workload_result(
            result,
            "workload_setup_failed",
            rank_local=True,
        )


def test_worker_bootstrap_failure_is_fatal(
    isolated_plugin: str,
    tmp_path: Path,
) -> None:
    from aorta.run._process import TrialWorkerError

    request = RunRequest(
        workload=isolated_plugin,
        trials=1,
        trial_isolation="process",
        extra_env={"ISOLATION_IMPORT_VALUE": "raise-during-import"},
        results_dir=tmp_path,
    )
    with pytest.raises(TrialWorkerError, match="not found|child import failed"):
        run_trials(request)


def test_trial_worker_module_is_invocable() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "aorta.run._trial_worker"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "usage: python -m aorta.run._trial_worker" in proc.stderr


def test_worker_spawn_oserror_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    from aorta.run import _process

    def fail_spawn(*_args, **_kwargs):
        raise OSError("no process slots")

    monkeypatch.setattr(_process.subprocess, "Popen", fail_spawn)
    with pytest.raises(_process.TrialWorkerError, match="could not start"):
        _process.launch_trial_worker(
            {"payload": {}},
            child_env=dict(os.environ),
            trial_id="trial",
        )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="this PyTorch Windows build lacks the TCPStore libuv backend used by torchrun",
)
def test_two_rank_workers_reinitialize_and_globalize_results(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    (plugin_root / "aorta_isolation_dist_plugin.py").write_text(
        """\
import os
import torch.distributed as dist
from aorta.workloads import Workload, WorkloadResult

class DistributedIsolationWorkload(Workload):
    launch_mode = "distributed"
    min_world_size = 2
    trial_isolation_supported = frozenset({"in_process", "process"})
    distributed_result_scope = "rank_local"

    def setup(self):
        if not dist.is_initialized():
            dist.init_process_group("gloo")

    def run(self):
        rank = dist.get_rank()
        return WorkloadResult(
            passed=rank == 0,
            failure_count=0 if rank == 0 else 1,
            failure_details=[] if rank == 0 else [{"error": "rank-one"}],
            metrics={"pid": os.getpid()},
        )

    def cleanup(self):
        if dist.is_initialized():
            dist.barrier()
""",
        encoding="utf-8",
    )
    dist_info = plugin_root / "aorta_isolation_dist-0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: aorta-isolation-dist\nVersion: 0.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[aorta.workloads]\n"
        "isolated_dist_workload = "
        "aorta_isolation_dist_plugin:DistributedIsolationWorkload\n"
        "[aorta.workload_policies]\n"
        "isolated_dist_workload = "
        "aorta.run.validation:PROCESS_OPTIONAL_POLICY\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        """\
import json
import os
import sys
from pathlib import Path
from aorta.run import RunRequest, run_trials

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
results = run_trials(RunRequest(
    workload="isolated_dist_workload",
    trials=2,
    trial_isolation="process",
    results_dir=out,
))
(out / f"rank{os.environ['RANK']}.json").write_text(
    json.dumps([result.to_dict() for result in results]),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["USE_LIBUV"] = "0"
    env["PYTHONPATH"] = (
        str(plugin_root)
        + os.pathsep
        + str(Path(__file__).resolve().parents[2] / "src")
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(launcher),
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    rank_results = [
        json.loads((output / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)
    ]
    for results in rank_results:
        assert len(results) == 2
        assert all(result["exit_status"] == "workload_failed" for result in results)
        assert all(result["result"]["passed"] is False for result in results)
        assert all(
            result["result"]["metrics"]["_rank_outcomes"]["failed"] == 1 for result in results
        )
        assert len({result["result"]["metrics"]["pid"] for result in results}) == 2
