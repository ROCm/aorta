# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the Monarch debugger demos under ``examples/monarch-debug``.

These are examples rather than package modules, so they are loaded by path.
CI has no Monarch, so everything here runs on the import stand-ins: where the
bug is planted, how the GPU demo finds and pins its allocation, the controller
address, the two launchers up to the point where they would start Python, and
the README's citations of script lines.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "monarch-debug"
_HAS_MONARCH = importlib.util.find_spec("monarch") is not None

_ALLOCATION_VARS = ("SLURM_JOB_ID", "MONARCH_SLURM_JOB", "SPUR_JOB_ID")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"monarch_debug_{name}", _EXAMPLE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cpu_mesh():
    return _load("cpu_mesh")


@pytest.fixture(scope="module")
def gpu_reduce():
    return _load("gpu_reduce")


@pytest.fixture
def no_allocation(monkeypatch):
    for key in _ALLOCATION_VARS:
        monkeypatch.delenv(key, raising=False)


# ---- Bug placement ------------------------------------------------------


def test_cpu_bug_is_on_ranks_2_and_6_of_8(cpu_mesh):
    assert [rank for rank in range(8) if cpu_mesh.is_bug_rank(rank)] == [2, 6]


@pytest.mark.parametrize(("hosts", "bad_rank"), [(1, 3), (2, 11)])
def test_gpu_bug_is_gpu_3_of_the_last_host(gpu_reduce, hosts, bad_rank):
    gpus = 8
    bad = [
        host * gpus + gpu
        for host in range(hosts)
        for gpu in range(gpus)
        if gpu_reduce.is_bug_rank(host, gpu, hosts)
    ]
    assert bad == [bad_rank]


def test_monarch_flag_matches_the_interpreter(cpu_mesh, gpu_reduce):
    assert cpu_mesh._MONARCH_READY is _HAS_MONARCH
    assert gpu_reduce._MONARCH_READY is _HAS_MONARCH


# ---- Allocation lookup --------------------------------------------------


def test_slurm_step_prefix_attaches_to_the_job(gpu_reduce, no_allocation, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert gpu_reduce.step_prefix(scheduler="slurm", nodes=2) == [
        "srun",
        "--jobid",
        "123",
        "--overlap",
        "-N2",
        "--ntasks=2",
        "--ntasks-per-node=1",
    ]


def test_spur_step_prefix_runs_spur_srun(gpu_reduce, no_allocation, monkeypatch):
    monkeypatch.setenv("SPUR_JOB_ID", "7")
    assert gpu_reduce.step_prefix(scheduler="spur", nodes=1, spur_bin="/opt/spur/bin/spur") == [
        "/opt/spur/bin/spur",
        "srun",
        "-N1",
        "--ntasks=1",
        "--ntasks-per-node=1",
    ]


def test_spur_step_prefix_needs_a_spur_allocation(gpu_reduce, no_allocation):
    with pytest.raises(RuntimeError, match="SPUR_JOB_ID"):
        gpu_reduce.step_prefix(scheduler="spur", nodes=1)


@pytest.mark.parametrize("nodes", [0, -1, True, 1.0])
def test_step_prefix_rejects_a_bad_node_count(gpu_reduce, no_allocation, monkeypatch, nodes):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(ValueError, match="nodes"):
        gpu_reduce.step_prefix(scheduler="slurm", nodes=nodes)


def test_step_prefix_rejects_an_unknown_scheduler(gpu_reduce, no_allocation):
    with pytest.raises(ValueError, match="scheduler"):
        gpu_reduce.step_prefix(scheduler="pbs", nodes=1)


def test_monarch_slurm_job_wins_over_slurm_job_id(gpu_reduce, no_allocation, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "9")
    monkeypatch.setenv("MONARCH_SLURM_JOB", "5")
    assert gpu_reduce._slurm_job_id() == "5"


def test_slurm_job_id_without_an_allocation_is_refused(gpu_reduce, no_allocation):
    with pytest.raises(RuntimeError, match="does not call salloc"):
        gpu_reduce._slurm_job_id()


def test_a_spur_id_is_not_taken_as_a_slurm_id(gpu_reduce, no_allocation, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("SPUR_JOB_ID", "42")
    with pytest.raises(RuntimeError, match="is a Spur allocation"):
        gpu_reduce._slurm_job_id()


def test_an_explicit_slurm_job_is_used_even_inside_spur(gpu_reduce, no_allocation, monkeypatch):
    monkeypatch.setenv("SPUR_JOB_ID", "42")
    monkeypatch.setenv("MONARCH_SLURM_JOB", "42")
    assert gpu_reduce._slurm_job_id() == "42"


def test_choose_nodes_puts_this_machine_first(gpu_reduce):
    assert gpu_reduce._choose_nodes(["a", "b", "c"], 2, "b") == (["b", "a"], ["c"])


def test_choose_nodes_from_outside_the_allocation(gpu_reduce):
    assert gpu_reduce._choose_nodes(["a", "b", "c"], 2, "login") == (["a", "b"], ["c"])


def test_choose_nodes_uses_the_whole_allocation(gpu_reduce):
    assert gpu_reduce._choose_nodes(["a", "b"], 2, "a") == (["a", "b"], [])


def test_choose_nodes_refuses_too_many_nodes(gpu_reduce):
    with pytest.raises(ValueError, match="needs 2 nodes, but the allocation has 1"):
        gpu_reduce._choose_nodes(["a"], 2, "a")


# ---- Probe --------------------------------------------------------------


def _short_hostname() -> str:
    return subprocess.check_output(["hostname", "-s"], text=True).strip()


def test_spur_probe_names_nodes_by_hostname_not_the_callers_nodename(gpu_reduce):
    # `env` stands in for `spur srun`: it sets SLURM_NODEID and, like Spur,
    # passes the caller's SLURMD_NODENAME through.
    prefix = ["env", "SLURM_NODEID=0", "SLURMD_NODENAME=caller-node"]
    placed = gpu_reduce._probe_nodes(prefix, 1, "spur")
    assert placed == [(_short_hostname(), socket.gethostname())]


def test_slurm_probe_uses_slurmd_nodename(gpu_reduce):
    prefix = ["env", "SLURM_NODEID=0", "SLURMD_NODENAME=node-a"]
    assert gpu_reduce._probe_nodes(prefix, 1) == [("node-a", socket.gethostname())]


def test_slurm_probe_without_a_node_name_is_refused(gpu_reduce):
    prefix = ["env", "-u", "SLURMD_NODENAME", "SLURM_NODEID=0"]
    with pytest.raises(RuntimeError, match="expected 1 nodes"):
        gpu_reduce._probe_nodes(prefix, 1)


# ---- Controller address -------------------------------------------------


@pytest.fixture
def no_controller_ip(monkeypatch):
    for key in ("AORTA_CONTROLLER_IP", "MONARCH_CONTROLLER_IP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(socket, "getfqdn", lambda *_args: "login.example")


def test_controller_ip_refuses_a_loopback_lookup(gpu_reduce, no_controller_ip, monkeypatch):
    monkeypatch.setattr(socket, "gethostbyname", lambda _name: "127.0.1.1")
    with pytest.raises(RuntimeError, match="loopback"):
        gpu_reduce._controller_ip()


def test_controller_ip_takes_the_lookup(gpu_reduce, no_controller_ip, monkeypatch):
    monkeypatch.setattr(socket, "gethostbyname", lambda _name: "10.0.0.5")
    assert gpu_reduce._controller_ip() == "10.0.0.5"


def test_controller_ip_honors_aorta_controller_ip(gpu_reduce, no_controller_ip, monkeypatch):
    def _no_lookup(_name):
        raise AssertionError("an explicit address must not be looked up")

    monkeypatch.setattr(socket, "gethostbyname", _no_lookup)
    monkeypatch.setenv("AORTA_CONTROLLER_IP", "10.1.2.3")
    monkeypatch.setenv("MONARCH_CONTROLLER_IP", "10.9.9.9")
    assert gpu_reduce._controller_ip() == "10.1.2.3"


def test_controller_ip_rejects_a_bad_explicit_address(gpu_reduce, no_controller_ip, monkeypatch):
    monkeypatch.setenv("MONARCH_CONTROLLER_IP", "node-a")
    with pytest.raises(RuntimeError, match="must be an IPv4 address"):
        gpu_reduce._controller_ip()


# ---- Worker logs and teardown -------------------------------------------


def test_open_logs_closes_stdout_when_stderr_cannot_open(gpu_reduce, tmp_path, monkeypatch):
    opened = []
    real_open = Path.open

    def _recording_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", _recording_open)
    with pytest.raises(OSError):
        gpu_reduce._open_logs(tmp_path / "gpu-workers.out", tmp_path)
    assert len(opened) == 1
    assert opened[0].closed


class _FakeStep:
    def __init__(self):
        self.signals = []

    def send_signal(self, sig):
        self.signals.append(sig)


def test_signal_group_ignores_a_group_that_already_exited(gpu_reduce, monkeypatch, capsys):
    def _gone(_pgid, _sig):
        raise ProcessLookupError

    monkeypatch.setattr(os, "getpgrp", lambda: 1, raising=False)
    monkeypatch.setattr(os, "killpg", _gone, raising=False)
    step = _FakeStep()
    gpu_reduce._signal_group(step, 4242, 15)
    assert step.signals == []
    assert capsys.readouterr().err == ""


def test_signal_group_reports_other_errors_and_signals_the_step(gpu_reduce, monkeypatch, capsys):
    def _denied(_pgid, _sig):
        raise PermissionError("not allowed")

    monkeypatch.setattr(os, "getpgrp", lambda: 1, raising=False)
    monkeypatch.setattr(os, "killpg", _denied, raising=False)
    step = _FakeStep()
    gpu_reduce._signal_group(step, 4242, 15)
    assert step.signals == [15]
    assert "could not signal process group 4242: not allowed" in capsys.readouterr().err


def test_signal_group_never_signals_its_own_group(gpu_reduce, monkeypatch):
    def _no_killpg(_pgid, _sig):
        raise AssertionError("must not signal the controller's own group")

    monkeypatch.setattr(os, "getpgrp", lambda: 4242, raising=False)
    monkeypatch.setattr(os, "killpg", _no_killpg, raising=False)
    step = _FakeStep()
    gpu_reduce._signal_group(step, 4242, 9)
    assert step.signals == [9]


class _ExitedStep:
    """A launcher that has already exited, so ``wait`` returns at once."""

    def __init__(self, returncode: int | None = 0):
        self.returncode = returncode
        self.signals: list[int] = []
        self.waits = 0

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def wait(self, timeout: float | None = None) -> int:
        self.waits += 1
        if self.returncode is None:
            raise subprocess.TimeoutExpired("step", timeout)
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode


def test_stop_workers_sigkills_a_group_that_outlives_the_launcher(gpu_reduce, monkeypatch):
    sent: list[int] = []

    def _killpg(_pgid: int, sig: int) -> None:
        sent.append(sig)
        if sig == 0 and signal.SIGKILL not in sent:
            return
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(os, "getpgrp", lambda: 1, raising=False)
    monkeypatch.setattr(os, "killpg", _killpg, raising=False)
    step = _ExitedStep()
    gpu_reduce._stop_workers(step, 4242, None)
    assert sent == [signal.SIGTERM, 0, signal.SIGKILL, 0, 0]
    assert step.signals == []


def test_stop_workers_leaves_an_empty_group_after_sigterm(gpu_reduce, monkeypatch):
    sent: list[int] = []

    def _killpg(_pgid: int, sig: int) -> None:
        sent.append(sig)
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(os, "getpgrp", lambda: 1, raising=False)
    monkeypatch.setattr(os, "killpg", _killpg, raising=False)
    gpu_reduce._stop_workers(_ExitedStep(), 4242, None)
    assert signal.SIGKILL not in sent


def test_workers_need_stop_when_the_launcher_exited_but_the_group_lives(gpu_reduce, monkeypatch):
    monkeypatch.setattr(os, "getpgrp", lambda: 1, raising=False)
    monkeypatch.setattr(os, "killpg", lambda _pgid, _sig: None, raising=False)
    assert gpu_reduce._workers_need_stop(_ExitedStep(0), 4242, None)


def test_workers_need_stop_skips_a_finished_slurm_step(gpu_reduce, monkeypatch):
    def _gone(_pgid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(os, "getpgrp", lambda: 1, raising=False)
    monkeypatch.setattr(os, "killpg", _gone, raising=False)
    assert not gpu_reduce._workers_need_stop(_ExitedStep(0), 4242, None)


def test_workers_need_stop_still_cleans_spur_after_the_client_exits(
    gpu_reduce, monkeypatch, tmp_path
):
    def _no_killpg(_pgid: int, _sig: int) -> None:
        raise AssertionError("an empty group must not be probed when cancel is set")

    monkeypatch.setattr(os, "killpg", _no_killpg, raising=False)
    assert gpu_reduce._workers_need_stop(_ExitedStep(0), 4242, tmp_path / "cancel")


def test_stop_workers_sigkills_the_group_when_wait_is_interrupted(gpu_reduce, monkeypatch):
    sent: list[int] = []

    def _killpg(_pgid: int, sig: int) -> None:
        sent.append(sig)
        if sig == 0 and signal.SIGKILL not in sent:
            return
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(os, "getpgrp", lambda: 1, raising=False)
    monkeypatch.setattr(os, "killpg", _killpg, raising=False)

    class _Interrupted(_ExitedStep):
        def wait(self, timeout: float | None = None) -> int:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        gpu_reduce._stop_workers(_Interrupted(), 4242, None)
    assert sent == [signal.SIGTERM, signal.SIGKILL, 0, 0]


def test_group_has_members_treats_permission_errors_as_alive(gpu_reduce, monkeypatch):
    def _denied(_pgid: int, _sig: int) -> None:
        raise PermissionError("not allowed")

    monkeypatch.setattr(os, "getpgrp", lambda: 1, raising=False)
    monkeypatch.setattr(os, "killpg", _denied, raising=False)
    assert gpu_reduce._group_has_members(4242)
    assert not gpu_reduce._group_has_members(None)
    assert not gpu_reduce._group_has_members(1)


def test_group_has_members_reports_other_probe_errors(gpu_reduce, monkeypatch, capsys):
    def _bad(_pgid: int, _sig: int) -> None:
        raise OSError("bad pgid")

    monkeypatch.setattr(os, "getpgrp", lambda: 1, raising=False)
    monkeypatch.setattr(os, "killpg", _bad, raising=False)
    assert not gpu_reduce._group_has_members(4242)
    assert "could not probe process group 4242: bad pgid" in capsys.readouterr().err


def _descendant_is_gone(pid: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_stop_workers_kills_a_descendant_that_ignores_sigterm(gpu_reduce, tmp_path):
    pidfile = tmp_path / "descendant.pid"
    script = tmp_path / "ignore-term.sh"
    script.write_text(
        "#!/bin/bash\n"
        'bash -c \'echo $$ > "$1"; trap "" TERM INT; '
        'while true; do read -t 0.2 || true; done\' _ "$1" &\n'
        "sleep 60\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        ["bash", str(script), str(pidfile)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pgid = os.getpgid(proc.pid)
    try:
        deadline = time.monotonic() + 5
        while not pidfile.exists():
            if proc.poll() is not None or time.monotonic() > deadline:
                raise AssertionError("descendant did not start")
            time.sleep(0.05)
        descendant = int(pidfile.read_text(encoding="utf-8").strip())
        started = time.monotonic()
        gpu_reduce._stop_workers(proc, pgid, None)
        assert time.monotonic() - started < 5
        assert _descendant_is_gone(descendant)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)


def test_spur_watch_kills_a_descendant_before_stopping_the_watcher(gpu_reduce, tmp_path):
    pidfile = tmp_path / "descendant.pid"
    cancel = tmp_path / "cancel"
    inner = (
        'bash -c \'echo $$ > "$1"; trap "" TERM INT; '
        'while true; do read -t 0.2 || true; done\' _ "$1" & sleep 60'
    )
    env = os.environ.copy()
    env["AORTA_STEP_CANCEL"] = str(cancel)
    proc = subprocess.Popen(
        [
            "bash",
            "-c",
            gpu_reduce._SPUR_WATCH,
            "debug-demo",
            "bash",
            "-c",
            inner,
            "_",
            str(pidfile),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    descendant = 0
    try:
        deadline = time.monotonic() + 5
        while not pidfile.exists():
            if proc.poll() is not None or time.monotonic() > deadline:
                raise AssertionError("descendant did not start")
            time.sleep(0.05)
        descendant = int(pidfile.read_text(encoding="utf-8").strip())
        cancel.write_text("", encoding="utf-8")
        proc.wait(timeout=10)
        assert _descendant_is_gone(descendant)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if descendant:
            try:
                os.kill(descendant, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_spur_watch_returns_when_the_child_exits_before_cancel(gpu_reduce, tmp_path):
    env = os.environ.copy()
    env["AORTA_STEP_CANCEL"] = str(tmp_path / "cancel")
    started = time.monotonic()
    proc = subprocess.run(
        ["bash", "-c", gpu_reduce._SPUR_WATCH, "debug-demo", "true"],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert time.monotonic() - started < 2


def test_debug_hints_continue_from_monarch_dbg_after_detach(cpu_mesh, gpu_reduce, capsys):
    cpu_mesh._print_debug_hint(8)
    gpu_reduce._print_debug_hint(8, 3)
    out = capsys.readouterr().out
    assert out.count("detach, then continue at monarch_dbg>") == 2
    assert "continue at (Pdb) resumes only the attached rank" in out


# ---- Worker import path -------------------------------------------------


def test_workers_can_import_gpu_reduce_by_its_module_name(gpu_reduce):
    env = os.environ.copy()
    env.update(gpu_reduce._worker_env())
    out = subprocess.run(
        [sys.executable, "-c", "import gpu_reduce; print(gpu_reduce.__file__)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        cwd="/",
    ).stdout
    assert Path(out.strip()).resolve() == _EXAMPLE / "gpu_reduce.py"


# ---- Worker attach ------------------------------------------------------


def test_worker_bootstrap_serves_under_this_nodes_service_id(gpu_reduce, tmp_path):
    fake = tmp_path / "monarch"
    fake.mkdir()
    (fake / "__init__.py").write_text("", encoding="utf-8")
    (fake / "actor.py").write_text(
        "def run_worker_loop_forever(*, address, ca):\n    print(address, ca)\n",
        encoding="utf-8",
    )
    ids = ["service<aaaaaaaaaaa>", "service<bbbbbbbbbbb>"]
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        gpu_reduce._SERVICE_IDS_ENV: ",".join(ids),
        "SLURM_NODEID": "1",
    }
    out = subprocess.run(
        [sys.executable, "-c", gpu_reduce._worker_bootstrap(22222)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.split() == [f"{ids[1]}@tcp://{socket.gethostname()}:22222", "trust_all_connections"]


def test_worker_addrs_pair_each_host_with_its_service_id(gpu_reduce):
    ids = ["service<aaaaaaaaaaa>", "service<bbbbbbbbbbb>"]
    assert gpu_reduce._worker_addrs(["h0", "h1"], ids, 22222) == [
        "service<aaaaaaaaaaa>@tcp://h0:22222",
        "service<bbbbbbbbbbb>@tcp://h1:22222",
    ]
    with pytest.raises(ValueError):
        gpu_reduce._worker_addrs(["h0"], ids, 22222)


def test_run_imports_only_service_identity_helpers_every_monarch_has(gpu_reduce):
    tree = ast.parse((_EXAMPLE / "gpu_reduce.py").read_text(encoding="utf-8"))
    imported = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("monarch._src")
        for alias in node.names
    }
    assert imported == {("monarch._src.job.service_identity", "new_service_proc_id")}


def test_cancel_path_is_unique_per_run(gpu_reduce, tmp_path, monkeypatch):
    monkeypatch.setenv("DEBUG_DEMO_SHARED_DIR", str(tmp_path))
    first, second = gpu_reduce._cancel_path(), gpu_reduce._cancel_path()
    assert first != second
    assert first.parent == second.parent == tmp_path
    assert str(os.getpid()) in first.name


# ---- README -------------------------------------------------------------

_PDB_LOCATION = re.compile(r"^> \.\.\./monarch-debug/(\w+\.py)\((\d+)\)\w+\(\)\n-> (.+)$", re.M)


def _readme_locations() -> list[tuple[str, int, str]]:
    readme = (_EXAMPLE / "README.md").read_text(encoding="utf-8")
    return [(name, int(line), code) for name, line, code in _PDB_LOCATION.findall(readme)]


def test_readme_pdb_locations_match_the_scripts():
    locations = _readme_locations()
    assert len(locations) >= 7
    for name, line, code in locations:
        source = (_EXAMPLE / name).read_text(encoding="utf-8").splitlines()
        assert source[line - 1].strip() == code, f"README cites {name}({line})"


def test_readme_line_numbers_are_ones_it_shows_in_pdb():
    readme = (_EXAMPLE / "README.md").read_text(encoding="utf-8")
    gpu_source = (_EXAMPLE / "gpu_reduce.py").read_text(encoding="utf-8").splitlines()
    init_line = next(
        number
        for number, text in enumerate(gpu_source, start=1)
        if text.strip() == "dist.init_process_group("
    )
    cited = {line for _name, line, _code in _readme_locations()} | {init_line}
    prose = {int(number) for number in re.findall(r"\b[Ll]ine\s+(\d+)\b", readme)}
    assert prose
    assert prose <= cited


# ---- Scripts ------------------------------------------------------------


@pytest.mark.parametrize("script", ["cpu_mesh.py", "gpu_reduce.py"])
def test_help_works(script):
    result = subprocess.run(
        [sys.executable, str(_EXAMPLE / script), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "Monarch debugger" in result.stdout


@pytest.mark.skipif(_HAS_MONARCH, reason="would start a real mesh")
@pytest.mark.parametrize("script", ["cpu_mesh.py", "gpu_reduce.py"])
def test_running_without_monarch_says_why(script):
    result = subprocess.run(
        [sys.executable, str(_EXAMPLE / script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert sys.executable in result.stderr
    assert "docs/monarch-install.md" in result.stderr
    assert "The import failed with ModuleNotFoundError" in result.stderr


# ---- Launchers ----------------------------------------------------------

_LAUNCHERS = ["run-slurm.sh", "run-spur.sh"]


def _launcher_env(**extra: str) -> dict[str, str]:
    # A GPU shell often exports the cluster's NCCL and Gloo settings, which the
    # fake interpreters echo back.
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _ALLOCATION_VARS
        and not key.startswith(("NCCL_", "GLOO_"))
        and key
        not in (
            "MONARCH_PYTHON",
            "DEBUG_DEMO_NET_ENV",
            "DEBUG_DEMO_SHARED_DIR",
            "AORTA_SHARED_DIR",
            "SPUR_BIN",
            "SPUR_HOME",
        )
    }
    env.update(extra)
    return env


def _launch(script: str, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_EXAMPLE / script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("script", _LAUNCHERS)
def test_launcher_parses(script):
    subprocess.run(["bash", "-n", str(_EXAMPLE / script)], check=True)


@pytest.mark.parametrize(
    ("script", "message"),
    [
        ("run-slurm.sh", "No Slurm allocation in this shell"),
        ("run-spur.sh", "No Spur allocation in this shell"),
    ],
)
def test_launcher_without_an_allocation_exits_2(script, message):
    result = _launch(script, _launcher_env(MONARCH_PYTHON="echo"))
    assert result.returncode == 2
    assert message in result.stderr
    assert result.stdout == ""


def test_slurm_launcher_passes_the_node_count():
    result = _launch(
        "run-slurm.sh",
        _launcher_env(MONARCH_PYTHON="echo", SLURM_JOB_ID="123"),
        "1",
        "--gpus-per-node",
        "4",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        str(_EXAMPLE / "gpu_reduce.py"),
        "--nodes",
        "1",
        "--gpus-per-node",
        "4",
    ]


@pytest.mark.parametrize("script", _LAUNCHERS)
def test_launcher_refuses_a_bad_node_count(script, tmp_path):
    env = _launcher_env(
        MONARCH_PYTHON="echo",
        SLURM_JOB_ID="123",
        SPUR_JOB_ID="7",
        DEBUG_DEMO_SHARED_DIR=str(tmp_path),
    )
    result = _launch(script, env, "3")
    assert result.returncode == 2
    assert "Node count must be 1 or 2" in result.stderr


def test_spur_launcher_resolves_a_spur_bin_directory(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    spur = bin_dir / "spur"
    spur.write_text("#!/bin/sh\n", encoding="utf-8")
    spur.chmod(0o755)
    shared = tmp_path / "shared"
    env = _launcher_env(
        MONARCH_PYTHON="echo",
        SPUR_JOB_ID="7",
        SPUR_BIN=str(bin_dir),
        DEBUG_DEMO_SHARED_DIR=str(shared),
    )
    result = _launch("run-spur.sh", env, "2")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        str(_EXAMPLE / "gpu_reduce.py"),
        "--scheduler",
        "spur",
        "--nodes",
        "2",
        "--spur-bin",
        str(spur),
    ]
    assert shared.is_dir()


def _fake_python(path: Path) -> Path:
    path.write_text(
        '#!/bin/sh\necho "python ifname=${NCCL_SOCKET_IFNAME:-unset} $*"\n', encoding="utf-8"
    )
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("script", _LAUNCHERS)
def test_launcher_defaults_to_python_on_path(script, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_python(bin_dir / "python")
    env = _launcher_env(
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        SLURM_JOB_ID="123",
        SPUR_JOB_ID="7",
        DEBUG_DEMO_SHARED_DIR=str(tmp_path),
    )
    result = _launch(script, env, "1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("python ifname=unset ")


@pytest.mark.parametrize("script", _LAUNCHERS)
def test_launcher_sources_the_opt_in_network_file(script, tmp_path):
    net_env = tmp_path / "net.sh"
    net_env.write_text("export NCCL_SOCKET_IFNAME=eth9\n", encoding="utf-8")
    env = _launcher_env(
        MONARCH_PYTHON=str(_fake_python(tmp_path / "fake-python")),
        DEBUG_DEMO_NET_ENV=str(net_env),
        SLURM_JOB_ID="123",
        SPUR_JOB_ID="7",
        DEBUG_DEMO_SHARED_DIR=str(tmp_path),
    )
    result = _launch(script, env, "1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("python ifname=eth9 ")


@pytest.mark.parametrize("script", _LAUNCHERS)
def test_launcher_refuses_a_missing_network_file(script, tmp_path):
    env = _launcher_env(
        MONARCH_PYTHON="echo",
        DEBUG_DEMO_NET_ENV=str(tmp_path / "missing.sh"),
        SLURM_JOB_ID="123",
        SPUR_JOB_ID="7",
    )
    result = _launch(script, env, "1")
    assert result.returncode == 2
    assert "DEBUG_DEMO_NET_ENV" in result.stderr
