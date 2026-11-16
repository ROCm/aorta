"""The engine process (engine_proc): the prototype's pipe protocol with its known fixes, on the fake engine
in real spawned processes (CPU only)."""

from __future__ import annotations

import socket
import threading
import time

import pytest

from aorlta.rollout import ports
from aorlta.rollout.engine_pool import engine_kwargs, engine_port
from aorlta.rollout.engine_proc import EngineProc, EngineStartError, slim
from aorlta.rollout.fake_engine import FakeEngineConfig

KW = engine_kwargs(model_dir="fake", dtype="bfloat16", P=2, K=2, n_engines=1, max_len=4096,
                   gpu_memory_utilization=0.45, run_seed=7)
SP = {"temperature": 0.7, "top_p": 0.95, "top_k": -1, "max_new_tokens": 64, "seed": 3, "stop_token_ids": [248046]}


def _proc(index=0, **fake) -> EngineProc:
    cfg = FakeEngineConfig(seed=1, min_tokens=20, max_tokens=40, **fake)
    return EngineProc(index, str(index), dict(KW, port=engine_port(index)), factory="fake", fake=cfg,
                      ready_timeout_s=120)


def test_start_reads_back_the_resolved_server_args():
    p = _proc()
    try:
        ready = p.start()
        info = ready["info"]
        assert info["sampling_backend"] == "triton" and info["enable_output_logprobs"] is True
        assert info["enable_prefix_caching"] is True and info["dtype"] == "bfloat16" and info["port"] == 20000
        assert info["max_num_seqs"] == 4 and ready["pid"] != threading.get_native_id()
        res = p.generate([1, 2, 3], SP, timeout_s=60)
        assert res["lp_ids"] == res["output_ids"] and len(res["lp_vals"]) == len(res["output_ids"])
        assert res["finish_reason"]["type"] in ("stop", "length") and res["cached_tokens"] == 0
    finally:
        p.stop()
    assert not p.alive


def test_a_request_that_does_not_return_times_out_and_its_late_reply_is_dropped():
    p = _proc(hang_rate=1.0, hang_s=3.0, time_scale=1.0)
    try:
        p.start()
        t0 = time.monotonic()
        res = p.generate([1, 2], SP, timeout_s=0.5)
        assert res["kind"] == "timeout" and time.monotonic() - t0 < 2.5
        time.sleep(3.0)  # the late reply arrives and finds no waiter
        assert p._pending == {}
    finally:
        p.stop()


def test_one_engines_death_fails_only_its_own_waiters():
    a, b = _proc(0, latency_base_s=4.0, time_scale=1.0), _proc(1, latency_base_s=1.0, time_scale=1.0)
    out: dict = {}
    try:
        a.start()
        b.start()
        threads = [threading.Thread(target=lambda: out.__setitem__("a", a.generate([1], SP, timeout_s=60))),
                   threading.Thread(target=lambda: out.__setitem__("b", b.generate([1], SP, timeout_s=60)))]
        for t in threads:
            t.start()
        time.sleep(0.3)
        a.kill("test kill")
        for t in threads:
            t.join(30)
        assert out["a"]["kind"] == "dead" and not a.alive
        assert "error" not in out["b"] and b.alive  # the other engine's request completed
        assert a.generate([1], SP, timeout_s=5)["kind"] == "dead"  # a dead engine answers at once
    finally:
        a.stop()
        b.stop()


def test_a_worker_that_cannot_start_is_an_error():
    p = EngineProc(0, "0", dict(KW, sampling_backend="flashinfer"), factory="fake", fake=FakeEngineConfig(),
                   ready_timeout_s=60)
    with pytest.raises(EngineStartError, match="failed to start"):
        p.start()


def _auto(index=0) -> EngineProc:
    """An engine with no port in its kwargs, whose fake binds TokenSpeed's ports as the engine would."""
    cfg = FakeEngineConfig(seed=1, min_tokens=20, max_tokens=40, bind_ports=True)
    return EngineProc(index, str(index), dict(KW), factory="fake", fake=cfg, ready_timeout_s=120)


def _listen(port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", port))
    s.listen(1)
    return s


def _free_bases(n: int) -> list[int]:
    out, base = [], 20000
    while len(out) < n:
        if not ports.taken(ports.cluster(base)):
            out.append(base)
        base += ports.PORT_STRIDE
    return out


def test_an_engine_without_a_port_starts_on_a_free_window_and_reports_the_ports_it_bound():
    a, b = _auto(0), _auto(1)
    try:
        ra, rb = a.start(), b.start()
        for p, ready in ((a, ra), (b, rb)):
            base = ready["port"]
            assert ready["info"]["port"] == base == p.port and ready["attempt"] == 1 and ready["earlier_attempts"] == []
            assert ready["ports"] == {"dist_init": base + 233, "tokenizer": base + 234, "rpc": base + 236,
                                      "metrics": base + 237, "scheduler_input": base + 238, "nccl_checked": base + 100}
            assert not ports.port_free(base + 233)  # bound by the engine's process while it runs
            assert ports.held_windows()[id(p)] == base
        assert abs(ra["port"] - rb["port"]) >= ports.PORT_STRIDE
    finally:
        a.stop()
        b.stop()
    assert id(a) not in ports.held_windows() and ports.port_free(ra["port"] + 233)


def test_a_start_that_loses_a_port_after_the_check_tries_another_window(monkeypatch):
    lost, _ = _free_bases(2)
    real = ports.hold_window
    calls = []

    def first_loses(holder, **kw):  # the first window's check passed, then another process took a port
        calls.append(list(kw.get("avoid")))
        return lost if len(calls) == 1 else real(holder, **kw)

    monkeypatch.setattr(ports, "hold_window", first_loses)
    p = _auto()
    with _listen(lost + 234):
        try:
            ready = p.start()
            assert ready["attempt"] == 2 and ready["port"] != lost and p.port == ready["port"]
            assert len(ready["earlier_attempts"]) == 1 and f"on port {lost}, a port was taken" in ready[
                "earlier_attempts"][0] and "Address already in use" in ready["earlier_attempts"][0]
            assert calls == [[], [lost]]  # the second choice avoids the lost window
            assert "error" not in p.generate([1, 2], SP, timeout_s=60)
        finally:
            p.stop()


def test_a_port_lost_on_every_attempt_stops_with_every_attempt_named(monkeypatch):
    bases = _free_bases(ports.START_ATTEMPTS + 1)
    held = [_listen(b + 233) for b in bases]
    monkeypatch.setattr(ports, "hold_window", lambda holder, **kw: bases[len(kw["avoid"])])
    try:
        with pytest.raises(EngineStartError, match=f"a port was taken on each of {ports.START_ATTEMPTS} starts") as e:
            _auto().start()
        assert all(f"on port {b}" in str(e.value) for b in bases[:ports.START_ATTEMPTS])
        assert f"on port {bases[ports.START_ATTEMPTS]}" not in str(e.value)
    finally:
        for s in held:
            s.close()


def test_a_start_that_fails_for_another_reason_is_not_retried(monkeypatch):
    real, calls = ports.hold_window, []
    monkeypatch.setattr(ports, "hold_window", lambda holder, **kw: calls.append(1) or real(holder, **kw))
    p = EngineProc(0, "0", dict(KW, sampling_backend="flashinfer"), factory="fake",
                   fake=FakeEngineConfig(bind_ports=True), ready_timeout_s=60)
    with pytest.raises(EngineStartError, match="failed to start") as e:
        p.start()
    assert calls == [1] and "a port was taken" not in str(e.value) and id(p) not in ports.held_windows()


def test_a_fixed_port_is_used_as_given_and_started_once(monkeypatch):
    base = _free_bases(1)[0]
    monkeypatch.setattr(ports, "hold_window", lambda *a, **k: pytest.fail("a fixed port is never chosen"))
    cfg = FakeEngineConfig(seed=1, min_tokens=20, max_tokens=40, bind_ports=True)
    p = EngineProc(0, "0", dict(KW, port=base), factory="fake", fake=cfg, ready_timeout_s=120)
    try:
        ready = p.start()
        assert ready["port"] == base and ready["ports"]["dist_init"] == base + 233 and ready["attempt"] == 1
    finally:
        p.stop()
    with _listen(base + 233), pytest.raises(EngineStartError, match=f"(?s)port {base}.*Address already in use"):
        EngineProc(0, "0", dict(KW, port=base), factory="fake", fake=cfg, ready_timeout_s=120).start()


def test_an_engine_restarted_after_a_death_starts_again_with_its_kwargs():
    p = _auto()
    try:
        first = p.start()
        p.kill("test death")
        assert not p.alive
        again = p.start()  # what the pool's restart calls, on the same handle
        assert again["attempt"] == 1 and again["ports"]["dist_init"] == again["port"] + 233
        assert p.engine_kwargs == dict(KW) and ports.held_windows()[id(p)] == again["port"]
        assert "error" not in p.generate([1, 2], SP, timeout_s=60) and first["pid"] != again["pid"]
    finally:
        p.stop()


def test_slim_reads_triples_and_reports_malformed_logprobs():
    out = {"output_ids": [5, 6], "meta_info": {"output_token_logprobs": [(-0.5, 5, None), (-1.0, 6, None)],
                                              "cached_tokens": 3, "finish_reason": {"type": "length", "length": 2}}}
    s = slim(out)
    assert s["lp_vals"] == [-0.5, -1.0] and s["lp_ids"] == [5, 6] and s["cached_tokens"] == 3
    bad = slim({"output_ids": [5], "meta_info": {"output_token_logprobs": [(-0.5,)]}})
    assert bad["lp_vals"] is None and "output_token_logprobs_malformed" in bad
    assert slim("nope")["kind"] == "engine"
