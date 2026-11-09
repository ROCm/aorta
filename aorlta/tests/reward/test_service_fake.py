"""C2.T1-T8 with the fake checker (design Section 6.2).

Every test runs the real service with fake adapters on a scaled clock: the thresholds are the doc's
defaults (timeouts 60/120/120/900 s, stall_s 300 s, ASan max_wait_s 20 s, controls every 32 jobs or
300 s), and the clock's scale only sets how many real seconds a test takes.
"""

from __future__ import annotations

import dataclasses
import json
import random
import threading
import time

import pytest

from aorlta.reward.adapters.fake import FakeAdapterConfig, FakeLegProfile, FakeOutcome
from aorlta.reward.profiles import load_profile
from aorlta.reward.rewardfn.base import Decided
from aorlta.reward.rewardfn.standin_v11 import StandinV11
from aorlta.reward.service import RewardService, ServiceOptions
from reward_helpers import SIM_LEGS, cand, fake_rig, reward_section, wait_all

ALL_LEGS = load_profile("fake").legs
TIMEOUTS = {"gate": 60.0, "compile": 120.0, "sim_run": 120.0, "asan_request": 900.0}


def leg_timeout(leg: str) -> float:
    if leg == "gate.compile":
        return TIMEOUTS["compile"]
    if leg.startswith("gate."):
        return TIMEOUTS["gate"]
    if leg == "san.asan":
        return TIMEOUTS["asan_request"]
    return TIMEOUTS["sim_run"]


def spans_by(rig, **attrs):
    return [s for s in rig.spans.spans() if all(s.attrs.get(k) == v for k, v in attrs.items())]


def length(c):
    return -0.05 * (c.response_tokens / c.max_new_tokens)


# --- C2.T1 ------------------------------------------------------------------------------------------


def test_c2_t1_256_candidates_with_injected_latencies_all_resolve_with_spans(tmp_path):
    """Latencies shaped like a real checker's: simulator runs p10 4 s / p90 18 s, ASan ~136 s per request."""
    lat = {f"gate.{g}": FakeLegProfile(latency_s=0.2, jitter_s=0.5) for g in
           ("parse", "contract", "static", "reference", "no_artifact", "spec_invalid", "static_reject",
            "no_change_on_failing_parent")}
    lat["gate.compile"] = FakeLegProfile(latency_s=0.5, jitter_s=0.2)
    for name in SIM_LEGS:
        lat[name] = FakeLegProfile(latency_s=4.0, jitter_s=14.0)
    lat["san.asan"] = FakeLegProfile(latency_s=7.0, jitter_s=3.0)  # per package: ~136 s per 16-package request
    adapter = FakeAdapterConfig(seed=7, per_leg=lat)
    rig = fake_rig(tmp_path, scale=0.01, adapter=adapter, reward=reward_section(sim_workers=32))
    try:
        cands = [cand(i) for i in range(256)]
        futures = [rig.service.submit(c) for c in cands]
        report = rig.service.drain(rig.clock.now() + 3600)
        results = wait_all(futures, timeout=120)
        assert len(results) == 256 and all(f.done() for f in futures)
        assert report.resolved == 256 and not report.hit_deadline
        assert [r.candidate_id for r in results] == [c.candidate_id for c in cands]
        assert all(r.status == "scored" and r.reward == 1.0 + length(c) for r, c in zip(results, cands))
        for c in cands:  # one span per leg run, per candidate, plus the ASan queue
            names = {s.name for s in spans_by(rig, candidate_id=c.candidate_id)}
            assert set(ALL_LEGS) | {"asan.queue"} <= names
        assert spans_by(rig) and {s.iteration for s in rig.spans.spans()} == {1}
        runs = [s for s in rig.spans.spans() if s.name == "asan.run"]
        assert sum(s.attrs["packages"] for s in runs) == 256 + 2 * len(runs)
        assert all(s.t_end >= s.t_start for s in rig.spans.spans())
    finally:
        rig.close()


# --- C2.T2 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("slow_leg", ALL_LEGS)
def test_c2_t2_an_infrastructure_timeout_in_any_leg_is_unscored_never_clean(tmp_path, slow_leg):
    """The leg runs past its timeout on a live, heartbeating worker: the watchdog makes it
    unsettled(timeout) and the sample is unscored."""
    c = cand(0)
    adapter = FakeAdapterConfig(overrides={(c.candidate_id, slow_leg): FakeOutcome("clean", latency_s=1e5)})
    rig = fake_rig(tmp_path, scale=0.002, adapter=adapter)
    try:
        r = rig.service.submit(c).result(timeout=60)
        assert r.reward is None and r.status == "unscored", r
        assert r.legs[slow_leg].state == "unsettled" and r.legs[slow_leg].reason == "timeout"
        assert not StandinV11.claims_clean(r.reward, r.parts)
        timed = [s for s in spans_by(rig, candidate_id=c.candidate_id) if s.name == slow_leg]
        assert timed and all(s.attrs["state"] == "unsettled" for s in timed)
        assert timed[0].duration >= leg_timeout(slow_leg) - 1e-6
        assert rig.service.stats().worker_restarts.get("timeout", 0) >= 1
        assert rig.service.cache.stats()["legs"]["writes"] == sum(  # the timed-out leg was not cached
            1 for x in r.legs.values() if x.state in ("clean", "finding", "fail") and not x.cache_hit)
    finally:
        rig.close()


@pytest.mark.parametrize("leg_name", ["ref.native", "ref.edge", "san.waitcheck", "san.asan"])
def test_c2_t2_a_candidate_timeout_under_a_clean_bracket_is_scored(tmp_path, leg_name):
    c = cand(0)
    adapter = FakeAdapterConfig(overrides={(c.candidate_id, leg_name): FakeOutcome("fail", "timeout")})
    rig = fake_rig(tmp_path, adapter=adapter, reward=reward_section(sim_workers=1))
    try:
        r = rig.service.submit(c).result(timeout=60)
        assert r.legs[leg_name].state == "fail" and r.legs[leg_name].bracket["clean"] is True
        assert r.reward == -0.8 + length(c) and r.parts["base"] == -0.8
    finally:
        rig.close()


def test_c2_t2_a_candidate_timeout_without_a_clean_bracket_is_unscored(tmp_path):
    """fail(timeout) on ref.native, and the after-control is dirty on the first worker and on the worker
    the retry goes to: the timeout is never scored as -0.8."""
    c = cand(0)
    dirty = FakeOutcome("fail", "control_dirty")
    overrides = {(c.candidate_id, "ref.native"): FakeOutcome("fail", "timeout"),
                 ("control:sim/1:2:ref.native", "ref.native"): dirty,
                 ("control:sim/2:2:ref.native", "ref.native"): dirty}
    rig = fake_rig(tmp_path, adapter=FakeAdapterConfig(overrides=overrides), reward=reward_section(sim_workers=1))
    try:
        r = rig.service.submit(c).result(timeout=60)
        native = r.legs["ref.native"]
        assert r.reward is None and r.reason == "ref.native"
        assert native.state == "unsettled" and native.reason == "control_dirty" and native.bracket["clean"] is False
        assert native.attempts == 2  # retried once, on another worker
        fails = spans_by(rig, candidate_id=c.candidate_id)
        assert [s.attrs["worker"] for s in fails if s.name == "ref.native"] == ["sim/1", "sim/2"]
    finally:
        rig.close()


# --- C2.T3 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("parent_healthy", [None, True])
def test_c2_t3_a_confirmed_finding_with_an_unsettled_asan_leg_is_penalised(tmp_path, parent_healthy):
    c = cand(0, parent_healthy=parent_healthy, tokens=1500, max_new=4096)
    overrides = {(c.candidate_id, "san.consan"): FakeOutcome("finding"),
                 (c.candidate_id, "san.asan"): FakeOutcome("error", "adapter_error")}
    rig = fake_rig(tmp_path, adapter=FakeAdapterConfig(overrides=overrides))
    try:
        r = rig.service.submit(c).result(timeout=60)
        assert r.legs["san.consan"].state == "finding"
        assert r.legs["san.asan"].state == "unsettled" and r.legs["san.asan"].attempts == 2  # one infra retry
        base = -0.5 * 1.0 / 3.0  # Appendix B: in-round counts all three sanitizers, unsettled included
        broke = -0.5 if parent_healthy else 0.0
        assert r.parts == {"base": base, "broke_healthy": broke, "length": length(c)}
        assert r.reward == base + broke + length(c) and r.status == "scored"
    finally:
        rig.close()


# --- C2.T4 ------------------------------------------------------------------------------------------


def test_c2_t4_a_dirty_after_control_unsettles_exactly_its_bracket_and_each_leg_is_retried_once_elsewhere(tmp_path):
    adapter = FakeAdapterConfig(overrides={("control:sim/1:3:ref.native", "ref.native"): FakeOutcome("fail", "control_dirty")})
    section = reward_section(sim_workers=2)
    section = dataclasses.replace(section, controls=dataclasses.replace(section.controls, every_jobs=4))
    rig = fake_rig(tmp_path, adapter=adapter, reward=section)
    try:
        cands = [cand(i) for i in range(24)]
        results = {r.candidate_id: r for r in wait_all([rig.service.submit(c) for c in cands])}
        ledger = {rec.control_run_id: rec for rec in rig.service.ledger_controls.snapshot()}
        c2, c3 = ledger["control:sim/1:2"], ledger["control:sim/1:3"]
        assert c2.clean and not c3.clean
        # Independent oracle: the first attempts that ran on sim/1 between its 2nd and 3rd controls.
        expected = {(s.attrs["candidate_id"], s.name) for s in rig.spans.spans()
                    if s.attrs.get("worker") == "sim/1" and s.attrs.get("attempt") == 1 and s.name in SIM_LEGS
                    and s.t_start >= c2.t_end and s.t_end <= c3.t_start}
        # at most every_jobs: an idle worker closes its bracket early when its queue runs dry
        assert 1 <= len(expected) <= 4
        retried = [e for e in rig.service.events_log if e["event"] == "retry"]
        assert {(e["candidate_id"], e["leg"]) for e in retried} == expected
        assert all(e["reason"] == "control_dirty" and e["from_worker"] == "sim/1" for e in retried)
        assert len(retried) == len(expected)  # once each
        for cid, r in results.items():
            assert r.status == "scored"
            for name in SIM_LEGS:
                x = r.legs[name]
                if (cid, name) in expected:
                    assert x.attempts == 2 and x.worker_id != "sim/1" and x.bracket["clean"] is True
                else:
                    assert x.attempts == 1
    finally:
        rig.close()


def test_c2_t4_a_quarantined_worker_takes_no_jobs_until_a_clean_control(tmp_path):
    dirty = FakeOutcome("fail", "control_dirty")
    overrides = {(f"control:sim/1:{n}:ref.native", "ref.native"): dirty for n in (2, 3)}
    rig = fake_rig(tmp_path, adapter=FakeAdapterConfig(overrides=overrides), reward=reward_section(sim_workers=1))
    try:
        r = rig.service.submit(cand(0)).result(timeout=60)
        assert r.status == "scored"
        ledger = [rec for rec in rig.service.ledger_controls.snapshot() if rec.worker_id == "sim/1"]
        assert [rec.clean for rec in ledger[:4]] == [True, False, False, True]
        jobs_on_sim1 = [s for s in rig.spans.spans() if s.attrs.get("worker") == "sim/1" and s.name in SIM_LEGS]
        # nothing ran on sim/1 between the dirty control and the next clean one
        assert all(not (ledger[1].t_end <= s.t_start <= ledger[3].t_start) for s in jobs_on_sim1)
    finally:
        rig.close()


# --- C2.T5 ------------------------------------------------------------------------------------------


def _runs(rig) -> int:
    return sum(n for states in rig.service.stats().leg_runs.values() for n in states.values())


def test_c2_t5_the_same_source_task_and_settings_are_checked_once(tmp_path):
    rig = fake_rig(tmp_path)
    try:
        a = rig.service.submit(cand(0, text="```cpp\nsame\n```")).result(timeout=60)
        n = _runs(rig)
        assert n == len(ALL_LEGS)
        # the same content in another candidate: a reward-cache hit
        b = rig.service.submit(cand(1, text="```cpp\nsame\n```")).result(timeout=60)
        assert b.reward == a.reward and b.timings.get("reward_cache_hit") is True and _runs(rig) == n
        # different token count, same source: the reward key differs, every leg is a cache hit
        c = rig.service.submit(cand(2, text="```cpp\nsame\n```", tokens=7)).result(timeout=60)
        assert _runs(rig) == n and all(x.cache_hit for x in c.legs.values())
        assert c.reward == 1.0 + -0.05 * (7 / 4096)
    finally:
        rig.close()


def test_c2_t5_a_changed_setting_or_checker_version_misses(tmp_path):
    rig = fake_rig(tmp_path)
    try:
        rig.service.submit(cand(0, text="```cpp\nk\n```")).result(timeout=60)
    finally:
        rig.close()
    # same cache directory, a changed compile setting: only the compile leg misses
    profile = load_profile("fake")
    changed = dataclasses.replace(profile, leg_settings={"gate.compile": {"command": "hipcc --offload-arch=gfx950 -c -O2"}})
    rig2 = fake_rig(tmp_path, profile=changed)
    try:
        r = rig2.service.submit(cand(1, text="```cpp\nk\n```", tokens=3)).result(timeout=60)
        missed = {n for n, x in r.legs.items() if not x.cache_hit}
        assert missed == {"gate.compile"}
    finally:
        rig2.close()
    # a new checker version: every leg misses
    rig3 = fake_rig(tmp_path, adapter=FakeAdapterConfig(version="2"))
    try:
        r = rig3.service.submit(cand(2, text="```cpp\nk\n```", tokens=5)).result(timeout=60)
        assert not any(x.cache_hit for x in r.legs.values())
    finally:
        rig3.close()


def test_c2_t5_unsettled_results_are_never_cached(tmp_path):
    c = cand(0, text="```cpp\nu\n```")
    adapter = FakeAdapterConfig(overrides={(c.candidate_id, "san.waitcheck"): FakeOutcome("error", "adapter_error")})
    rig = fake_rig(tmp_path, adapter=adapter)
    try:
        r = rig.service.submit(c).result(timeout=60)
        assert r.reward is None and r.legs["san.waitcheck"].state == "unsettled"
        key = rig.service._cands[c.candidate_id].leg_keys["san.waitcheck"]
        rig.service.cache.flush()
        assert not (tmp_path / "cache" / "legs" / key[:2] / f"{key}.json").exists()
        before = rig.service.stats().leg_runs["san.waitcheck"]
        r2 = rig.service.submit(cand(1, text="```cpp\nu\n```")).result(timeout=60)
        assert not r2.legs["san.waitcheck"].cache_hit
        assert sum(rig.service.stats().leg_runs["san.waitcheck"].values()) > sum(before.values())
    finally:
        rig.close()


@pytest.mark.parametrize("slow_leg", ["gate.static", "ref.edge"])  # a plain leg and a bracketed one
def test_c2_t5_timeouts_are_never_cached(tmp_path, slow_leg):
    c = cand(0, text="```cpp\nt\n```")
    adapter = FakeAdapterConfig(overrides={(c.candidate_id, slow_leg): FakeOutcome("clean", latency_s=1e5)})
    rig = fake_rig(tmp_path, scale=0.002, adapter=adapter)
    try:
        r = rig.service.submit(c).result(timeout=60)
        assert r.legs[slow_leg].reason == "timeout"
        key = rig.service._cands[c.candidate_id].leg_keys[slow_leg]
        assert rig.service.cache.get_leg(key, rig.service._cands[c.candidate_id].leg_fields[slow_leg]) is None
        rig.service.cache.flush()
        assert not (tmp_path / "cache" / "legs" / key[:2] / f"{key}.json").exists()
        assert rig.service.cache.stats()["legs"]["rejected"] >= 1  # it was offered and refused
    finally:
        rig.close()


@pytest.mark.parametrize("damage", ["garbage", "wrong_key", "bad_schema", "unsettled_inside"])
def test_c2_t5_a_corrupt_cache_file_is_a_miss(tmp_path, damage):
    # a 60 s gate timeout is 0.6 s of real time at this scale: room for a scheduling stall under load
    rig = fake_rig(tmp_path, scale=0.01)
    try:
        c = cand(0, text="```cpp\nz\n```")
        rig.service.submit(c).result(timeout=60)
        key = rig.service._cands[c.candidate_id].leg_keys["san.consan"]
    finally:
        rig.close()
    path = tmp_path / "cache" / "legs" / key[:2] / f"{key}.json"
    entry = json.loads(path.read_text())
    if damage == "garbage":
        path.write_text("{not json")
    elif damage == "wrong_key":
        entry["key"] = "0" * 64
        path.write_text(json.dumps(entry))
    elif damage == "bad_schema":
        entry["result"]["state"] = "clean-ish"
        path.write_text(json.dumps(entry))
    else:
        entry["result"]["state"] = "unsettled"
        path.write_text(json.dumps(entry))
    rig2 = fake_rig(tmp_path, scale=0.01)  # a fresh process: nothing in the LRU
    try:
        r = rig2.service.submit(cand(1, text="```cpp\nz\n```", tokens=11)).result(timeout=60)
        assert not r.legs["san.consan"].cache_hit and r.legs["san.consan"].state == "clean"
        assert all(x.cache_hit for n, x in r.legs.items() if n != "san.consan")
        rig2.service.cache.flush()
        assert json.loads(path.read_text())["result"]["state"] == "clean"  # recomputed and rewritten
    finally:
        rig2.close()


# --- C2.T6 ------------------------------------------------------------------------------------------


def test_c2_t6_checks_start_while_generation_is_still_running(tmp_path):
    lat = {name: FakeLegProfile(latency_s=2.0, jitter_s=4.0) for name in ALL_LEGS}
    rig = fake_rig(tmp_path, scale=0.01, adapter=FakeAdapterConfig(per_leg=lat))
    try:
        rng = random.Random(1)
        times = sorted(rng.uniform(0, 120.0) for _ in range(32))
        t0 = rig.clock.now()
        futures = []
        for t in times:  # chains end spread over 120 s
            rig.clock.sleep(t0 + t - rig.clock.now())
            futures.append(rig.service.submit(cand(len(futures))))
        gen_end = rig.clock.to_wall(rig.clock.now())
        wait_all(futures)
        starts = {stage: min(s.t_start for s in rig.spans.spans() if s.name in legs)
                  for stage, legs in (("S0", [n for n in ALL_LEGS if n.startswith("gate.") and n != "gate.compile"]),
                                      ("S1", ["gate.compile"]), ("S2", list(SIM_LEGS)))}
        assert all(t < gen_end for t in starts.values()), (starts, gen_end)
        assert starts["S2"] < rig.clock.to_wall(t0 + times[0] + 60.0)  # the first candidate did not wait for the rest
    finally:
        rig.close()


# --- C2.T7 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("inband", [True, False])
def test_c2_t7_asan_requests_are_bounded_in_size_and_wait_and_the_last_partial_batch_is_flushed(tmp_path, inband):
    rng = random.Random(3)
    lat = {name: FakeLegProfile(latency_s=1.0, jitter_s=4.0) for name in SIM_LEGS}
    lat["san.asan"] = FakeLegProfile(latency_s=5.0, jitter_s=4.0)
    c_dec = {}
    overrides = {}
    cands = [cand(i) for i in range(70)]
    for c in cands[::7]:  # decided in S2: these must never reach ASan
        overrides[(c.candidate_id, "ref.native")] = FakeOutcome("fail", "wrong_output")
        c_dec[c.candidate_id] = True
    # 1 s of tolerance is 100 ms of real time at this scale: room for a GC pause, not for a late timer
    rig = fake_rig(tmp_path, scale=0.1, adapter=FakeAdapterConfig(per_leg=lat, overrides=overrides), inband=inband,
                   reward=reward_section(sim_workers=8))
    try:
        t0 = rig.clock.now()
        futures = []
        for c, t in zip(cands, sorted(rng.uniform(0, 150.0) for _ in cands)):
            rig.clock.sleep(t0 + t - rig.clock.now())
            futures.append(rig.service.submit(c))
        t_drain = rig.clock.now()  # the generation barrier
        report = rig.service.drain(t_drain + 3600)
        results = {r.candidate_id: r for r in wait_all(futures)}
        reqs = [e for e in rig.service.events_log if e["event"] == "asan_request"]
        assert reqs
        assert all(size <= 16 for e in reqs for size in e["request_sizes"])
        waits = [w for e in reqs for w in e["waits"]]
        assert max(waits) <= 20.0 + 1.0, max(waits)
        sent = [cid for e in reqs for cid in e["candidates"]]
        assert len(sent) == len(set(sent))  # each package once
        assert set(sent) == {cid for cid in results if cid not in c_dec}
        # once generation has ended nothing waits for a request to fill: the last partial batch goes out
        after = [e for e in reqs if e["generation_ended"]]
        assert after and reqs[-1]["generation_ended"] is True
        for e in after:
            for w in e["waits"]:
                assert e["t"] - max(e["t"] - w, t_drain) <= 1.0
        for e in reqs:  # ASan only after the candidate's S2 had settled
            sent_at = rig.clock.to_wall(e["t"])
            for cid in e["candidates"]:
                s2_end = max(s.t_end for s in spans_by(rig, candidate_id=cid) if s.name in SIM_LEGS)
                assert s2_end <= sent_at
        assert report.resolved == len(cands)
    finally:
        rig.close()


def test_c2_t7_max_inflight_bounds_outstanding_requests(tmp_path):
    lat = {"san.asan": FakeLegProfile(latency_s=8.0)}
    section = reward_section(sim_workers=8)
    section = dataclasses.replace(section, asan=dataclasses.replace(section.asan, max_inflight=1))
    rig = fake_rig(tmp_path, scale=0.002, adapter=FakeAdapterConfig(per_leg=lat), reward=section)
    try:
        wait_all([rig.service.submit(cand(i)) for i in range(60)])
        runs = sorted((s.t_start, s.t_end) for s in rig.spans.spans() if s.name == "asan.run")
        assert len(runs) >= 4
        assert all(b[0] >= a[1] for a, b in zip(runs, runs[1:]))  # never two at once
    finally:
        rig.close()


# --- C2.T8 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stalled_leg, detected_by", [("ref.native", "deadline"), ("san.asan", "stall_s")])
def test_c2_t8_a_stalled_worker_is_restarted_within_stall_s_plus_15_and_its_job_retried(tmp_path, stalled_leg, detected_by):
    """The fake stalls for 10^4 s and goes silent. A simulator leg's 120 s deadline comes first and, the
    worker being silent, counts as a stall; an ASan request's deadline is 900 s, so stall_s finds it."""
    c = cand(0)
    adapter = FakeAdapterConfig(stall_s=1e4, overrides={(c.candidate_id, stalled_leg, 1): FakeOutcome("stall")})
    rig = fake_rig(tmp_path, scale=0.01, adapter=adapter, reward=reward_section(sim_workers=2))
    try:
        r = rig.service.submit(c).result(timeout=60)
        restarts = [e for e in rig.service.events_log if e["event"] == "restart"]
        assert len(restarts) == 1 and restarts[0]["reason"] == "worker_stalled"
        took = restarts[0]["t"] - restarts[0]["item_started"]
        assert took <= 300.0 + 15.0
        assert took >= (120.0 if detected_by == "deadline" else 300.0) - 2.0
        retry = [e for e in rig.service.events_log if e["event"] == "retry"]
        assert [(e["leg"], e["reason"], e["attempt"]) for e in retry] == [(stalled_leg, "worker_stalled", 2)]
        assert r.status == "scored" and r.legs[stalled_leg].attempts == 2
        assert r.legs[stalled_leg].worker_id != restarts[0]["worker"]
    finally:
        rig.close()


def test_c2_t8_a_stall_on_every_attempt_stays_unsettled(tmp_path):
    c = cand(0)
    adapter = FakeAdapterConfig(overrides={(c.candidate_id, "san.consan"): FakeOutcome("stall")})
    rig = fake_rig(tmp_path, scale=0.002, adapter=adapter, reward=reward_section(sim_workers=2))
    try:
        r = rig.service.submit(c).result(timeout=60)
        assert r.reward is None and r.legs["san.consan"].reason == "worker_stalled"
        assert r.legs["san.consan"].attempts == 2
    finally:
        rig.close()


def test_no_progress_raises_an_alarm(tmp_path):
    c = cand(0)
    adapter = FakeAdapterConfig(overrides={(c.candidate_id, n): FakeOutcome("clean", latency_s=1e5) for n in SIM_LEGS})
    rig = fake_rig(tmp_path, scale=0.002, adapter=adapter, options=ServiceOptions(no_progress_s=100.0))
    try:
        r = rig.service.submit(c).result(timeout=60)
        assert r.reward is None
        assert any(a["kind"] == "no_progress" for a in rig.service.health()["alarms"])
        assert list((tmp_path / "work" / "alarms").glob("no_progress-*.json"))
    finally:
        rig.close()


# --- the service's own rules ------------------------------------------------------------------------


def test_drain_at_the_deadline_makes_every_pending_leg_unsettled_timeout(tmp_path):
    cs = [cand(i) for i in range(5)]
    adapter = FakeAdapterConfig(overrides={(c.candidate_id, "san.asan"): FakeOutcome("clean", latency_s=500.0) for c in cs})
    rig = fake_rig(tmp_path, scale=0.002, adapter=adapter)
    try:
        futures = [rig.service.submit(c) for c in cs]
        report = rig.service.drain(rig.clock.now() + 200.0)
        results = wait_all(futures)
        assert report.hit_deadline and report.resolved == 5 and report.scored == 0
        assert all(r.reward is None and r.legs["san.asan"].state == "unsettled"
                   and r.legs["san.asan"].reason == "timeout" for r in results)
        assert {o["leg"] for o in report.outstanding_at_deadline} == {"san.asan"}
        late = rig.service.submit(cand(9)).result(timeout=10)
        assert late.reward is None and late.reason == "not_accepting"
    finally:
        rig.close()


def test_submit_is_idempotent_and_refuses_bad_candidates(tmp_path):
    rig = fake_rig(tmp_path)
    try:
        c = cand(0)
        f1, f2 = rig.service.submit(c), rig.service.submit(c)
        assert f1 is f2
        assert rig.service.submit(cand(1, profile="standin-static")).result(timeout=10).reason == "profile_mismatch"
        bad = dataclasses.replace(cand(2), response_tokens=5000)
        assert rig.service.submit(bad).result(timeout=10).reason == "bad_candidate"
        assert f1.result(timeout=60).status == "scored"
    finally:
        rig.close()


class _Broken(StandinV11):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def compute(self, legs, c):
        reward, parts = super().compute(legs, c)
        if self.mode == "out_of_range" and reward is not None:
            return 1.5, {"base": 1.5, "broke_healthy": 0.0, "length": 0.0}
        if self.mode == "raises":
            raise RuntimeError("boom")
        return reward, parts

    def decide(self, partial, c):
        d = super().decide(partial, c)
        if self.mode == "decide_lies" and isinstance(d, Decided):
            return Decided(1.0, {"base": 1.0, "broke_healthy": 0.0, "length": 0.0})
        return d


@pytest.mark.parametrize("mode, reason", [("out_of_range", "reward_invariant_violated"),
                                          ("raises", "reward_fn_error"),
                                          ("decide_lies", "reward_invariant_violated")])
def test_a_reward_fn_that_breaks_the_rules_gives_unscored_and_an_alarm(tmp_path, mode, reason):
    rig = fake_rig(tmp_path, rewardfn=_Broken(mode))
    try:
        r = rig.service.submit(cand(0)).result(timeout=60)
        assert r.reward is None and r.reason == reason and r.status == "unscored"
        assert any(a["kind"] == reason for a in rig.service.health()["alarms"])
    finally:
        rig.close()


def test_an_unknown_gate_reason_is_unscored_and_a_compile_error_stops_the_candidate(tmp_path):
    a, b = cand(0), cand(1)
    overrides = {(a.candidate_id, "gate.static"): FakeOutcome("fail", "mystery"),
                 (b.candidate_id, "gate.compile"): FakeOutcome("fail", "compile_error")}
    rig = fake_rig(tmp_path, adapter=FakeAdapterConfig(overrides=overrides))
    try:
        ra, rb = wait_all([rig.service.submit(a), rig.service.submit(b)])
        assert ra.reward is None and ra.reason == "unknown_gate_reason"
        assert rb.reward == -0.9 + length(b)
        assert all(rb.legs[n].state == "not_run" for n in ("ref.native", "san.consan", "san.asan"))
        assert not spans_by(rig, candidate_id=b.candidate_id, state="clean") or all(
            s.name.startswith("gate.") for s in spans_by(rig, candidate_id=b.candidate_id))
    finally:
        rig.close()


def test_the_record_shows_legs_still_running_at_the_decision_as_not_run(tmp_path):
    c = cand(0)
    overrides = {(c.candidate_id, "ref.native"): FakeOutcome("fail", "wrong_output"),
                 (c.candidate_id, "san.waitcheck"): FakeOutcome("clean", latency_s=50.0)}
    rig = fake_rig(tmp_path, scale=0.002, adapter=FakeAdapterConfig(overrides=overrides))
    try:
        r = rig.service.submit(c).result(timeout=60)
        assert r.reward == -0.7 + length(c)
        assert r.legs["san.waitcheck"].state == "not_run" and r.legs["san.waitcheck"].reason == "decided_before_result"
        assert r.legs["san.asan"].state == "not_run"
    finally:
        rig.close()


def test_close_leaves_no_candidate_waiting(tmp_path):
    c = cand(0)
    rig = fake_rig(tmp_path, scale=0.002, adapter=FakeAdapterConfig(
        overrides={(c.candidate_id, "ref.native"): FakeOutcome("clean", latency_s=1e5)}))
    fut = rig.service.submit(c)
    time.sleep(0.2)
    rig.close()
    r = fut.result(timeout=10)
    assert r.reward is None and r.reason == "service_closed"


def test_reusing_the_rollout_compile_is_refused_until_it_can_be_proven():
    section = dataclasses.replace(reward_section(), reuse_rollout_compile=True)
    with pytest.raises(ValueError, match="reuse_rollout_compile"):
        fake_rig(None, reward=section)
