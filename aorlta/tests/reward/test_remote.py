"""Check adapters served from another process (`aorlta.reward.remote`): the stand-in GPU legs run in their own
checker process, and the reward service reaches them through `RemoteAdapter`. On CPU, with the fake adapter served
from a thread: a remote result is the in-process one, and nothing that goes wrong on the wire is ever clean."""

from __future__ import annotations

import dataclasses
import json
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest

from aorlta.reward.adapters.base import SequentialBatch
from aorlta.reward.adapters.fake import FakeAdapterConfig, FakeCheckAdapter, FakeOutcome
from aorlta.reward.clock import Clock
from aorlta.reward.profiles import load_profile
from aorlta.reward.remote import RemoteAdapter, RemoteUnavailable, Server, socket_path
from aorlta.reward.rewardfn.standin_v11 import StandinV11
from aorlta.reward.service import RewardService
from aorlta.schemas import LegJob
from aorlta.ids import sha256_text

from reward_helpers import cand, fake_rig, reward_section, wait_all


@pytest.fixture
def sock_dir():
    # A Unix socket path is limited to ~107 bytes, which a test's tmp_path can exceed; TMPDIR is honoured.
    d = Path(tempfile.mkdtemp(prefix="rmt-"))
    yield d
    shutil.rmtree(d, ignore_errors=True)  # NFS keeps a .nfs* file while a closed socket is still referenced


def job(leg="san.asan", cid="run/1/standin:k1/0", attempt=1, worker="asan/1") -> LegJob:
    return LegJob(job_id=f"{cid}#{leg}#a{attempt}", leg=leg, candidate_id=cid, control_id=None,
                  source_sha256=sha256_text("src"), task_sha256=sha256_text("task"), attempt=attempt,
                  worker_id=worker, leg_settings={"x": [1, 2]}, t_queued=1.0)


def serve(sock_dir, name="fake", factory=None, **cfg):
    made = []

    def default(wid):
        a = FakeCheckAdapter(FakeAdapterConfig(**cfg))
        made.append(a)
        return a

    server = Server(socket_path(sock_dir, name), name, factory or default).start()
    return server, made


def test_a_remote_result_is_the_in_process_one_and_controls_and_identity_come_across(sock_dir):
    server, _ = serve(sock_dir, overrides={("run/1/standin:k1/0", "san.asan"): FakeOutcome("finding")})
    try:
        remote = RemoteAdapter(socket_path(sock_dir, "fake"), "fake", "asan/1")
        local = FakeCheckAdapter(FakeAdapterConfig(overrides={("run/1/standin:k1/0", "san.asan"): FakeOutcome("finding")}))
        for leg in ("san.asan", "gate.parse", "ref.edge"):
            got, want = remote.run(job(leg)), local.run(job(leg))
            assert dataclasses.replace(got, t_start=0, t_end=0) == dataclasses.replace(want, t_start=0, t_end=0)
        assert remote.run(job()).state == "finding" and remote.run(job()).findings
        assert (remote.name, remote.version, remote.legs, remote.resource) == (local.name, local.version, local.legs,
                                                                             local.resource)
        assert remote.controls() == local.controls()
        assert remote.health()["responsive"] is True
        remote.close()
    finally:
        server.stop()


def test_a_batch_endpoint_is_served_with_its_size_and_in_band_controls(sock_dir):
    server, _ = serve(sock_dir, "fake-asan",
                      factory=lambda wid: SequentialBatch(FakeCheckAdapter(FakeAdapterConfig(legs=("san.asan",))),
                                                          max_packages=7, inband_controls=False))
    try:
        remote = RemoteAdapter(socket_path(sock_dir, "fake-asan"), "fake-asan", "asan/2")
        assert (remote.max_packages, remote.inband_controls) == (7, False)
        jobs = [job(cid=f"run/1/standin:k{i}/0") for i in range(3)]
        results = remote.run_batch(jobs)
        assert [r.state for r in results] == ["clean"] * 3
        assert [r.leg for r in results] == ["san.asan"] * 3
    finally:
        server.stop()


def test_nothing_that_goes_wrong_on_the_wire_is_clean(sock_dir):
    server, made = serve(sock_dir)
    remote = RemoteAdapter(socket_path(sock_dir, "fake"), "fake", "w/1")
    server.stop()  # the server goes away: the open connection keeps working until it is closed ...
    remote._conn.close()  # ... so close it the way a dying server would
    r = remote.run(job())
    assert (r.state, r.reason) == ("unsettled", "adapter_error")
    rs = remote.run_batch([job(), job(cid="run/1/standin:k2/0")])
    assert [(x.state, x.reason) for x in rs] == [("unsettled", "adapter_error")] * 2
    assert remote.health()["responsive"] is False


def test_a_reply_for_another_leg_or_a_malformed_reply_is_unsettled(sock_dir):
    class Liar:
        name, version, legs, resource = "fake", "1", ("san.asan", "gate.parse"), "own_gpu"

        def __init__(self, mode):
            self.mode = mode

        def run(self, j):
            r = FakeCheckAdapter(FakeAdapterConfig()).run(j)
            if self.mode == "leg":
                return dataclasses.replace(r, leg="gate.parse")
            return r

        def controls(self):
            return []

        def health(self):
            return {"responsive": True}

    for mode in ("leg",):
        server = Server(socket_path(sock_dir, "fake"), "fake", lambda wid: Liar(mode)).start()
        try:
            remote = RemoteAdapter(socket_path(sock_dir, "fake"), "fake", "w/1")
            r = remote.run(job("san.asan"))
            assert (r.state, r.reason) == ("unsettled", "adapter_error")
        finally:
            server.stop()
    # a server that answers with something that is not a LegResult
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    path = socket_path(sock_dir, "bad")
    srv.bind(str(path))
    srv.listen(1)

    def bad():
        c, _ = srv.accept()
        f = c.makefile("rwb")
        f.readline()
        f.write(json.dumps({"ok": True, "protocol": "aorlta.remote/1", "name": "fake", "version": "1",
                            "legs": ["san.asan"], "resource": "own_gpu", "controls": []}).encode() + b"\n")
        f.flush()
        f.readline()
        f.write(json.dumps({"ok": True, "result": {"leg": "san.asan", "state": "clean"}}).encode() + b"\n")
        f.flush()
        time.sleep(0.5)
        c.close()

    t = threading.Thread(target=bad)
    t.start()
    try:
        remote = RemoteAdapter(path, "fake", "w/1")
        r = remote.run(job())
        assert (r.state, r.reason) == ("unsettled", "adapter_error")
    finally:
        t.join()
        srv.close()


def test_an_unreachable_server_or_a_refused_open_is_an_error_at_construction(sock_dir):
    with pytest.raises(RemoteUnavailable):
        RemoteAdapter(socket_path(sock_dir, "nobody"), "nobody", "w/1", connect_timeout_s=0.5)
    server, _ = serve(sock_dir)
    try:
        with pytest.raises(RemoteUnavailable):  # this server serves "fake", not "fake-asan"
            RemoteAdapter(socket_path(sock_dir, "fake"), "fake-asan", "w/1", connect_timeout_s=0.5)

        def broken(wid):
            raise RuntimeError("the hook is missing")

        other = Server(socket_path(sock_dir, "other"), "other", broken).start()
        try:
            with pytest.raises(RemoteUnavailable):
                RemoteAdapter(socket_path(sock_dir, "other"), "other", "w/1", connect_timeout_s=0.5)
        finally:
            other.stop()
    finally:
        server.stop()


def test_closing_the_client_closes_the_servers_adapter_and_interrupts_its_job(sock_dir):
    server, made = serve(sock_dir, time_scale=1.0,
                         overrides={("run/1/standin:k1/0", "san.asan"): FakeOutcome("stall")}, stall_s=60.0)
    try:
        remote = RemoteAdapter(socket_path(sock_dir, "fake"), "fake", "w/1")
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("r", remote.run(job())))
        t0 = time.monotonic()
        t.start()
        time.sleep(0.5)
        remote.close()  # what the watchdog's restart does
        t.join(timeout=10)
        assert not t.is_alive() and time.monotonic() - t0 < 10  # not the 60 s stall
        assert out["r"].state == "unsettled"
        deadline = time.monotonic() + 10
        while not made[0]._stop.is_set() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert made[0]._stop.is_set()  # the server closed this worker's adapter
    finally:
        server.stop()


def test_the_reward_service_scores_through_remote_adapters_as_through_local_ones(tmp_path, sock_dir):
    """The standin-gpu arrangement on CPU: the batched (ASan) pool's adapter runs in the 'checker process'."""
    profile = load_profile("fake")
    cands = [cand(i) for i in range(6)]
    local = fake_rig(tmp_path / "local")
    try:
        want = {r.candidate_id: (r.status, r.reward) for r in wait_all([local.service.submit(c) for c in cands])}
    finally:
        local.close()

    def asan_factory(wid):
        cfg = FakeAdapterConfig()
        cfg = dataclasses.replace(cfg, legs=("san.asan",), per_leg={k: v for k, v in cfg.per_leg.items() if k == "san.asan"})
        return SequentialBatch(FakeCheckAdapter(cfg), max_packages=16, inband_controls=True)

    server = Server(socket_path(sock_dir, "fake-asan"), "fake-asan", asan_factory).start()
    try:
        svc = RewardService(reward_section(sim_workers=4), profile, StandinV11(),
                            {"fake": lambda wid: FakeCheckAdapter(FakeAdapterConfig()),
                             "fake-asan": lambda wid: RemoteAdapter(socket_path(sock_dir, "fake-asan"), "fake-asan", wid)},
                            cache_dir=tmp_path / "remote" / "cache", work_dir=tmp_path / "remote" / "work",
                            clock=Clock(0.01))  # ASan's 20 s batching wait is 0.2 s of real time
        try:
            got = {r.candidate_id: (r.status, r.reward) for r in wait_all([svc.submit(c) for c in cands])}
            assert got == want
            assert server.jobs >= 1 and server.connections >= 1
        finally:
            svc.close()
    finally:
        server.stop()
