"""The engine process (engine_proc): the prototype's pipe protocol with its known fixes, on the fake engine
in real spawned processes (CPU only)."""

from __future__ import annotations

import threading
import time

import pytest

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


def test_slim_reads_triples_and_reports_malformed_logprobs():
    out = {"output_ids": [5, 6], "meta_info": {"output_token_logprobs": [(-0.5, 5, None), (-1.0, 6, None)],
                                              "cached_tokens": 3, "finish_reason": {"type": "length", "length": 2}}}
    s = slim(out)
    assert s["lp_vals"] == [-0.5, -1.0] and s["lp_ids"] == [5, 6] and s["cached_tokens"] == 3
    bad = slim({"output_ids": [5], "meta_info": {"output_token_logprobs": [(-0.5,)]}})
    assert bad["lp_vals"] is None and "output_token_logprobs_malformed" in bad
    assert slim("nope")["kind"] == "engine"
