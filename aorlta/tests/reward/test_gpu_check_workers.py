"""More checker workers for the stand-in GPU legs, and a per-check wall-clock limit (in one stand-in run, one kernel
that hung held the only ConSan worker for 110 s, ASan waited behind it, and checks reached the critical path).

- `reward.remote.adapters` may give an adapter several GPUs: one server on each (`servers`), the adapter's pool gets
  its profile's worker count on every server, and `RemoteFactory` spreads the workers, a restarted worker taking the
  server its predecessor left.
- `reward.gpu_check_s` limits each aorta run of one kernel (inside the adapter's own share of the pool timeout, which
  stays the outer bound). A check still running past the limit has its program killed, and a check that reached the
  limit is unsettled whatever it returned: a timeout is never clean.

CPU only: the servers are threads serving the fake adapter, and the hung "kernel" is a copy of sleep(1)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from aorlta.config import ConfigError, RewardRemote
from aorlta.reward.adapters import aorta_import, standin_aorta
from aorlta.reward.adapters.fake import FakeAdapterConfig, FakeCheckAdapter
from aorlta.reward.adapters.standin_aorta import AortaAsan, AortaConsan, kill_program
from aorlta.reward.clock import Clock
from aorlta.reward.factories import standin_factories
from aorlta.reward.profiles import load_profile, profile_from_dict
from aorlta.reward.remote import RemoteFactory, Server, servers, socket_path
from aorlta.reward.rewardfn.standin_v11 import StandinV11
from aorlta.reward.service import RewardService

from reward_helpers import reward_section


@pytest.fixture
def sock_dir():
    d = Path(tempfile.mkdtemp(prefix="gpuw-"))  # a Unix socket path must stay short
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _serve(path: Path, name: str = "fake") -> Server:
    return Server(path, name, lambda wid: FakeCheckAdapter(FakeAdapterConfig())).start()


# --- several GPUs per adapter ------------------------------------------------------------------------------------


def test_an_adapter_may_name_several_distinct_gpus_and_each_gets_its_own_server_socket():
    r = RewardRemote(socket_dir="/dev/shm/s", adapters={"standin.aorta_consan": (4, 6), "standin.aorta_asan": 5})
    assert r.gpus("standin.aorta_consan") == (4, 6) and r.gpus("standin.aorta_asan") == (5,)
    assert servers(r) == [("standin.aorta_asan", 5, Path("/dev/shm/s/standin.aorta_asan.sock")),
                          ("standin.aorta_consan", 4, Path("/dev/shm/s/standin.aorta_consan.gpu4.sock")),
                          ("standin.aorta_consan", 6, Path("/dev/shm/s/standin.aorta_consan.gpu6.sock"))]
    assert socket_path("/dev/shm/s", "standin.aorta_asan") == Path("/dev/shm/s/standin.aorta_asan.sock")
    for gpus, message in (((4, 4), "listed twice"), ((), "at least one GPU"), ((4, -1), ">= 0")):
        with pytest.raises(ConfigError, match=message):
            RewardRemote(socket_dir="/dev/shm/s", adapters={"standin.aorta_consan": gpus})


def test_the_remote_factory_spreads_workers_and_a_replacement_takes_the_server_it_left(sock_dir):
    paths = [socket_path(sock_dir, "fake", g) for g in (4, 6)]
    started = [_serve(p) for p in paths]
    try:
        f = RemoteFactory(paths, "fake", connect_timeout_s=5)
        assert f.replicas == 2
        a, b = f("consan/1"), f("consan/2")
        assert (a.path, b.path) == (paths[0], paths[1]) and f.load() == [1, 1]
        a.close()  # the watchdog's restart closes the worker's adapter before it spawns the replacement
        c = f("consan/3")
        assert c.path == paths[0] and f.load() == [1, 1]
        d = f("consan/4")
        assert d.path == paths[0] and f.load() == [2, 1]  # a tie goes to the lower index
        for x in (b, c, d):
            x.close()
        assert f.load() == [0, 0]
    finally:
        for s in started:
            s.stop()


def test_a_pool_gets_its_workers_on_every_server_of_its_adapter(tmp_path, sock_dir):
    """A ConSan-like pool (`workers: 1`) whose adapter two servers serve gets one worker on each; a local pool keeps
    its count. Every worker runs its controls on its own server."""
    profile = profile_from_dict({
        "name": "fake", "reward_fn": "standin.v11",
        "pools": {"gates": {"resource": "local_cpu", "adapter": "local", "timeout": "gate", "workers": 1},
                  "consan": {"resource": "own_gpu", "adapter": "gpu", "timeout": "sim_run", "workers": 1,
                             "controls": True}},
        "stages": [{"name": "S0", "pool": "gates", "legs": ["gate.parse"]},
                   {"name": "S2g", "pool": "consan", "legs": ["san.consan"]}],
        "requires": {"san.consan": ["gate.parse"]}})
    paths = [socket_path(sock_dir, "gpu", g) for g in (4, 6)]
    started = [_serve(p, "gpu") for p in paths]
    try:
        svc = RewardService(reward_section(), profile, StandinV11(),
                            {"local": lambda wid: FakeCheckAdapter(FakeAdapterConfig()),
                             "gpu": RemoteFactory(paths, "gpu", connect_timeout_s=5)},
                            cache_dir=tmp_path / "cache", work_dir=tmp_path / "work", clock=Clock(0.01))
        try:
            pools = svc.health()["pools"]
            assert len(pools["gates"]["workers"]) == 1 and len(pools["consan"]["workers"]) == 2
            assert sorted(str(w.adapter.path) for w in svc._pools["consan"].workers.values()) == sorted(map(str, paths))
            deadline = time.monotonic() + 30
            while min(s.jobs for s in started) < 1 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert [s.jobs >= 1 for s in started] == [True, True]  # each worker's control ran on its own server
        finally:
            svc.close()
    finally:
        for s in started:
            s.stop()


# --- the per-check wall-clock limit ------------------------------------------------------------------------------


def _models():
    return SimpleNamespace(KernelIdentity=lambda **k: k, KernelWorklist=lambda **k: k, KernelObservation=lambda **k: k,
                           SelectionRequirement=SimpleNamespace(TOP_DISPATCH_COUNT=1))


def _fake_aorta(seen: list):
    """aorta's GPU backends, answering PASS at once and recording the limit each run is given."""
    passed = SimpleNamespace(state=SimpleNamespace(value="ran"), verdict=SimpleNamespace(value="pass"), findings=(),
                             reason=None)

    def run_consan(worklist, *, command, hook_lib, output_dir, strict, timeout_seconds):
        seen.append(("san.consan", timeout_seconds))
        return SimpleNamespace(consan=passed, waitcheck_preflight=SimpleNamespace(verdict=SimpleNamespace(value="pass")))

    def run_asan(worklist, *, command, output_dir, runtime, timeout_seconds):
        seen.append(("san.asan", timeout_seconds))
        return passed

    mods = {"consan": SimpleNamespace(run_consan=run_consan),
            "asan": SimpleNamespace(run_asan=run_asan, resolve_asan_runtime=lambda: "runtime")}
    return lambda name, src: (mods.get(name), None if name in mods else "absent")


@pytest.mark.parametrize("cls,kw", [(AortaConsan, {"hook": "/nonexistent"}), (AortaAsan, {})], ids=["consan", "asan"])
def test_aorta_stops_the_program_at_the_per_check_limit_never_past_the_adapters_share(tmp_path, monkeypatch, cls, kw):
    seen: list = []
    monkeypatch.setattr(aorta_import, "load", _fake_aorta(seen))
    for limit, own, want in ((30.0, 108.0, 30.0), (None, 108.0, 108.0), (500.0, 45.0, 45.0)):
        a = cls(aorta_src="/nonexistent", timeout_s=own, check_timeout_s=limit, **kw)
        assert a.check_s == want
        assert a.bounded_check(_models(), tmp_path / "harness", "k", tmp_path / "out") == ("clean", None, [])
        assert seen[-1] == (a.leg, want)


def _sleeper(tmp_path: Path) -> Path:
    """A program whose argv[0] is its path, as the harness program aorta runs is: a copy of sleep(1). It keeps the
    name sleep, because a multi-call coreutils (uutils, as on Ubuntu 26.04) picks the tool by argv[0]'s name."""
    exe = tmp_path / "k0" / "sleep"
    exe.parent.mkdir(parents=True)
    shutil.copy(shutil.which("sleep"), exe)
    return exe


def _running(exe: Path) -> bool:
    for d in Path("/proc").iterdir():
        try:
            if d.name.isdigit() and (d / "cmdline").read_bytes().split(b"\0", 1)[0] == str(exe).encode():
                return True
        except OSError:
            pass
    return False


def test_a_hung_check_is_killed_past_the_limit_and_unsettled_and_its_worker_runs_the_next(tmp_path, monkeypatch):
    """The worst case: aorta does not stop the program (here it would wait 600 s), and when the program dies anyway it
    reports PASS. The program is killed CHECK_GRACE_S past the limit, the check is unsettled, and the same adapter
    then runs the next check at once."""
    monkeypatch.setattr(standin_aorta, "CHECK_GRACE_S", 0.5)
    exe = _sleeper(tmp_path)
    a = AortaConsan(hook="/nonexistent", aorta_src="/nonexistent", timeout_s=108.0, check_timeout_s=0.5)

    def hung(models, exe_, kernel, out):
        subprocess.run([str(exe_), "600"], timeout=600)  # aorta's own limit, as if it never fired
        return "clean", None, []

    monkeypatch.setattr(a, "check", hung)
    t0 = time.monotonic()
    state, reason, findings = a.bounded_check(_models(), exe, "k", tmp_path / "out")
    took = time.monotonic() - t0
    assert (state, findings) == ("unsettled", []) and reason.startswith("check_timeout:")
    assert 1.0 <= took < 5.0  # the limit plus one grace, not aorta's 600 s
    assert not _running(exe)  # the hung program is gone
    monkeypatch.setattr(a, "check", lambda *args: ("finding", None, [{"kind": "race"}]))
    assert a.bounded_check(_models(), exe, "k", tmp_path / "out") == ("finding", None, [{"kind": "race"}])


def test_a_check_that_reaches_the_limit_is_never_clean_and_one_inside_it_is_untouched(tmp_path, monkeypatch):
    a = AortaAsan(aorta_src="/nonexistent", timeout_s=45.0, check_timeout_s=0.3)

    def slow_clean(models, exe, kernel, out):
        time.sleep(0.5)  # aorta came back after the limit: whatever it says, the check timed out
        return "clean", None, []

    monkeypatch.setattr(a, "check", slow_clean)
    state, reason, _ = a.bounded_check(_models(), tmp_path / "harness", "k", tmp_path / "out")
    assert state == "unsettled" and reason.startswith("check_timeout:")
    monkeypatch.setattr(a, "check", lambda *args: ("clean", None, []))
    assert a.bounded_check(_models(), tmp_path / "harness", "k", tmp_path / "out") == ("clean", None, [])


def test_kill_program_signals_only_the_program_at_that_path(tmp_path):
    exe, other = _sleeper(tmp_path / "a"), _sleeper(tmp_path / "b")
    procs = [subprocess.Popen([str(p), "600"]) for p in (exe, other)]
    try:
        assert kill_program(exe) == 1
        assert procs[0].wait(timeout=10) == -9 and procs[1].poll() is None
    finally:
        for p in procs:
            p.kill()
            p.wait()


def test_the_factories_give_each_gpu_leg_its_per_check_limit_inside_its_share():
    profile = load_profile("standin-gpu")
    timeouts = {"gate": 60.0, "compile": 120.0, "sim_run": 120.0, "asan_request": 900.0}
    f = standin_factories(profile, Clock(), timeouts=timeouts, check_s={"san.consan": 30.0})
    consan, asan = f["standin.aorta_consan"]("consan/1"), f["standin.aorta_asan"]("asan/1").adapter
    assert (consan.check_s, consan.timeout_s) == (30.0, 108.0)
    assert asan.check_s == asan.timeout_s == 45.0  # not named: its own share of the ASan request timeout
    f = standin_factories(profile, Clock(), timeouts=timeouts, check_s={"san.asan": 20.0})
    assert f["standin.aorta_consan"]("consan/1").check_s == 108.0 and f["standin.aorta_asan"]("asan/1").adapter.check_s == 20.0


def test_a_checker_server_builds_its_gpu_adapter_with_the_runs_check_limit():
    """In a run the GPU legs run in the checker servers (`remote serve`), whose adapters `standin_factory` builds from
    the run config: configs/standin.yaml's ConSan gets its 30 s limit there, and ASan keeps its own share."""
    from aorlta.reward.remote import standin_factory

    cfg = str(Path(__file__).resolve().parents[2] / "configs" / "standin.yaml")
    assert standin_factory(cfg, "standin.aorta_consan")("consan/1").check_s == 30.0
    assert standin_factory(cfg, "standin.aorta_asan")("asan/1").adapter.check_s == 45.0


def test_gpu_check_s_names_only_the_gpu_legs_with_positive_limits():
    assert reward_section(gpu_check_s={"san.consan": 30.0, "san.asan": 45.0}).gpu_check_s["san.consan"] == 30.0
    assert reward_section().gpu_check_s == {}
    for value, message in (({"san.waitcheck": 30.0}, "not a stand-in GPU leg"), ({"san.consan": 0.0}, "> 0"),
                           ({"san.asan": -1.0}, "> 0")):
        with pytest.raises(ConfigError, match=message):
            reward_section(gpu_check_s=value)
