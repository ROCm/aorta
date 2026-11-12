"""The v1 loop (Section 6.6): C6.T2 (stalls, deadlines and the D7 policy), C6.T3 (the generation barrier),
streaming accumulation, and the failure table. All on the virtual clock: a 10^4 s stall costs no real time."""

import dataclasses

import pytest

from aorlta.episodes.writer import complete_iterations, read_episodes, read_summary
from aorlta.orchestrator.fakes import build_fake_run
from aorlta.orchestrator.state import complete_checkpoints

NO_ERRORS = {"error_rate": 0.0}  # every chain hands over a candidate, so the counts below are exact


def run(spec, **kw):
    r = build_fake_run(spec, **kw)
    return r, r.orchestrator.run()


def scored_ids(make_spec, n_iterations=1, **kw):
    """The chain ids of iteration 1 that score in a run without faults, to plant faults on."""
    r, res = run(make_spec("baseline", iterations=n_iterations, eval_every=None, engine=NO_ERRORS, **kw))
    assert res["status"] == "completed"
    return [e["chain_id"] for e in read_episodes(r.cfg.run_dir, 1) if e["status"] == "scored"]


def by_chain(run_dir, n):
    return {e["chain_id"]: e for e in read_episodes(run_dir, n)}


# --- C6.T2 ----------------------------------------------------------------------------------------------------


def test_c6_t2_a_1e4_s_stall_ends_at_the_scoring_deadline_and_the_iteration_trains(make_spec):
    stalled = scored_ids(make_spec)[5]
    r, res = run(make_spec(iterations=1, eval_every=None, engine=NO_ERRORS,
                           reward={"stall": [stalled], "faulty_attempts": [1, 2]}))
    assert res["status"] == "completed"
    s = read_summary(r.cfg.run_dir, 1)
    assert s["deadlines"]["scoring_s"] == 1800.0
    assert s["deadlines"]["scoring_hit"] is True
    assert s["deadlines"]["scoring_dur_s"] == pytest.approx(1800.0, abs=1e-6)  # not the stall's 10^4 s
    drain = r.reward.drains[0]
    assert drain["t_done"] - drain["t_drain"] == pytest.approx(1800.0, abs=1e-6)
    assert s["policy"]["decision"] == "train" and s["policy"]["retry"] is None  # 31 of 32 >= 90%
    assert s["samples"]["scored"] == 31 and s["scored_fraction"] == pytest.approx(31 / 32)
    [out] = s["deadlines"]["outstanding_at_deadline"]
    assert out["candidate_id"] == stalled and out["leg"] == "san.asan" and out["where"].startswith("fake-w")
    eps = by_chain(r.cfg.run_dir, 1)
    e = eps[stalled]
    assert (e["status"], e["unscored_reason"], e["legs"], e["trained"], e["excluded_reason"]) == (
        "unscored", "sanitizer_not_settled", {"san.asan": "unsettled"}, False, "unscored")
    group = [x for x in eps.values() if x["prompt_id"] == e["prompt_id"]]
    assert sum(x["trained"] for x in group) == 7  # the group trains on its 7 scored members
    assert complete_checkpoints(r.cfg.run_dir / "ckpt") == [1]


def test_c6_t2_below_the_threshold_the_stalled_legs_are_retried_once_with_1_5x_the_deadline(make_spec):
    stalled = scored_ids(make_spec)[:4]
    r, res = run(make_spec(iterations=1, eval_every=None, engine=NO_ERRORS,
                           reward={"stall": stalled, "faulty_attempts": [1]}))
    assert res["status"] == "completed"
    s = read_summary(r.cfg.run_dir, 1)
    p = s["policy"]
    assert p["first"]["decision"] == "retry" and p["first"]["scored"] == 28  # 28 of 32 < 90%
    assert p["retry"]["candidates"] == 4 and p["retry"]["deadline_s"] == 2700.0
    assert p["retry"]["decision"] == "train" and p["decision"] == "train" and s["samples"]["scored"] == 32
    assert sorted(r.reward.resubmits) == sorted(stalled)
    first, second = r.reward.drains
    assert first["t_done"] - first["t_drain"] == pytest.approx(1800.0, abs=1e-6)
    assert second["t_done"] - second["t_drain"] < 2700.0  # the retried legs settled before the longer deadline
    eps = by_chain(r.cfg.run_dir, 1)
    assert all(eps[c]["status"] == "scored" and eps[c]["trained"] for c in stalled)


def test_c6_t2_a_stall_that_survives_the_retry_stops_the_run_untrained(make_spec):
    stalled = scored_ids(make_spec)[:4]
    r, res = run(make_spec(iterations=1, eval_every=None, engine=NO_ERRORS,
                           reward={"stall": stalled, "faulty_attempts": [1, 2]}))
    digest_before = r.trainer.__class__(r.trainer.c).state_digest()
    assert res["status"] == "stopped"
    assert res["stop"]["reason"] == "scored_fraction_below_threshold"
    d = res["stop"]["detail"]
    assert d["policy"]["retry"]["scored"] == 28 and d["policy"]["decision"] == "stop"
    first, second = r.reward.drains
    assert second["t_done"] - second["t_drain"] == pytest.approx(2700.0, abs=1e-6)  # the retry's deadline, 1.5 x
    assert sorted({o["candidate_id"] for o in d["outstanding_at_deadline"]}) == sorted(stalled)
    assert {u["candidate_id"] for u in d["unsettled"]} == set(stalled)
    assert all(u["worker_id"].startswith("fake-w") and u["leg"] == "san.asan" for u in d["unsettled"])
    # not trained: no step, nothing published, nothing committed
    assert r.trainer.steps == 0 and r.trainer.state_digest() == digest_before
    assert [c for c in r.sync.calls if c[0] == "publish"] == []
    assert complete_iterations(r.cfg.run_dir) == [] and complete_checkpoints(r.cfg.run_dir / "ckpt") == []


def test_c6_t2_exactly_the_threshold_trains_without_a_retry(make_spec):
    # P = 5, K = 8: 36 of 40 scored is exactly 90%
    stalled = scored_ids(make_spec, P=5, pool_size=10)[:4]
    r, res = run(make_spec(iterations=1, eval_every=None, engine=NO_ERRORS, P=5, pool_size=10,
                           reward={"stall": stalled, "faulty_attempts": [1, 2]}))
    s = read_summary(r.cfg.run_dir, 1)
    assert res["status"] == "completed" and s["policy"]["decision"] == "train" and s["policy"]["retry"] is None
    assert s["samples"]["scored"] == 36 and s["samples"]["total"] == 40 and r.reward.resubmits == []


def test_c6_t2_a_retry_that_could_not_reach_the_threshold_is_not_attempted(make_spec):
    ids = scored_ids(make_spec)
    r, res = run(make_spec(iterations=1, eval_every=None, engine=NO_ERRORS,
                           reward={"unscored": ids[:4], "stall": ids[4:5], "faulty_attempts": [1, 2]}))
    assert res["status"] == "stopped" and res["stop"]["detail"]["policy"]["first"]["retryable"] == 1
    assert "even if" in res["stop"]["detail"]["policy"]["first"]["why"]
    assert r.reward.resubmits == [] and len(r.reward.drains) == 1


def test_the_scoring_deadline_follows_twice_the_rolling_median(make_spec):
    r, res = run(make_spec(iterations=3, eval_every=None, engine=NO_ERRORS, scoring_s=200.0,
                           reward={"latency_s": 150.0, "jitter_s": 0.0}))
    assert res["status"] == "completed"
    s = [read_summary(r.cfg.run_dir, n)["deadlines"] for n in (1, 2, 3)]
    assert s[0]["scoring_dur_s"] == pytest.approx(150.0)
    assert [d["scoring_s"] for d in s] == [200.0, pytest.approx(300.0), pytest.approx(300.0)]


def test_chains_still_running_at_the_generation_deadline_end_and_are_recorded(make_spec):
    r, res = run(make_spec(iterations=1, eval_every=None, engine=NO_ERRORS, generation_s=25.0,
                           min_scored_fraction=0.1))
    assert res["status"] == "completed"
    s = read_summary(r.cfg.run_dir, 1)
    d = s["deadlines"]
    assert d["generation_s"] == 25.0 and d["generation_hit"] is True and d["generation_outstanding"]
    eps = by_chain(r.cfg.run_dir, 1)
    for cid in d["generation_outstanding"]:
        assert eps[cid]["end_reason"] == "engine_error" and eps[cid]["status"] == "unscored"
    assert d["generation_dur_s"] <= 25.0 + 1e-6


# --- C6.T3 ----------------------------------------------------------------------------------------------------


def test_c6_t3_requests_in_flight_at_the_generation_barrier_stop_the_run_before_any_reload(make_spec):
    r = build_fake_run(make_spec(iterations=1, eval_every=None))
    r.rollout.leak = 1
    res = r.orchestrator.run()
    assert res["status"] == "stopped" and res["stop"]["reason"] == "barrier_violation"
    assert res["stop"]["detail"]["where"] == "generation barrier"
    assert [c for c in r.sync.calls if c[0] == "publish"] == []


def test_c6_t3_a_reload_attempted_with_requests_in_flight_raises_and_nothing_is_published(make_spec):
    r = build_fake_run(make_spec(iterations=1, eval_every=None))
    finish = r.trainer.finish

    def finish_then_leak():  # an engine restart's self-test request, say, arrives during the step
        report = finish()
        r.rollout.leak = 2
        return report

    r.trainer.finish = finish_then_leak
    res = r.orchestrator.run()
    assert res["status"] == "stopped" and res["stop"]["reason"] == "barrier_violation"
    assert res["stop"]["detail"]["where"] == "reload"
    assert [c for c in r.sync.calls if c[0] == "publish"] == []
    assert complete_checkpoints(r.cfg.run_dir / "ckpt") == []


def test_the_barrier_counts_requests_at_the_moment_of_the_reload():
    from aorlta.orchestrator.loop import GenerationBarrier
    from aorlta.orchestrator.ports import BarrierViolation

    held = [0]
    b = GenerationBarrier(lambda: held[0])
    b.assert_idle("x")
    held[0] = 1
    with pytest.raises(BarrierViolation):
        b.assert_idle("x")


# --- streaming accumulation (D5) -------------------------------------------------------------------------------


def _trace(r, monkeypatch):
    """Points, and the moment each group is handed to the trainer thread (synchronous, in the completing thread;
    the trainer thread itself runs in real time while the virtual clock jumps, so its timing says nothing)."""
    from aorlta.orchestrator.loop import TrainerWorker

    events = []
    point = r.orchestrator._point

    def traced(name, iteration):
        events.append((name, iteration))
        point(name, iteration)

    r.orchestrator._point = traced
    submit = TrainerWorker.submit

    def traced_submit(self, group):
        events.append(("queued", group.prompt_id, len(group.members),
                       sum(1 for m in group.members if m.result is not None and m.result.status == "scored")))
        return submit(self, group)

    monkeypatch.setattr(TrainerWorker, "submit", traced_submit)
    return events


def test_groups_are_handed_to_the_trainer_as_they_complete_before_the_generation_barrier(make_spec, monkeypatch):
    r = build_fake_run(make_spec(iterations=1, eval_every=None, reward={"latency_s": 1.0, "jitter_s": 1.0}))
    events = _trace(r, monkeypatch)
    assert r.orchestrator.run()["status"] == "completed"
    barrier = events.index(("gen.done", 1))
    early = [e for e in events[:barrier] if e[0] == "queued"]
    assert early, "no group was handed to the trainer while chains still ran"
    assert all(e[2] == 8 for e in early)  # a group goes only once all K members are in


def test_a_group_with_an_unsettled_member_waits_for_the_policy_and_trains_on_the_retried_result(make_spec, monkeypatch):
    unsettled = scored_ids(make_spec, reward={"latency_s": 1.0, "jitter_s": 1.0})[:4]
    r = build_fake_run(make_spec(iterations=1, eval_every=None, engine=NO_ERRORS,
                                 reward={"latency_s": 1.0, "jitter_s": 1.0, "unsettled": unsettled,
                                         "faulty_attempts": [1]}))
    events = _trace(r, monkeypatch)
    res = r.orchestrator.run()
    assert res["status"] == "completed"
    assert read_summary(r.cfg.run_dir, 1)["policy"]["first"]["decision"] == "retry"
    first_drain = events.index(("score.drained", 1))  # the retry starts after this
    held = {cid.split("/")[2] for cid in unsettled}
    queued = {e[1]: (i, e[3]) for i, e in enumerate(events) if e[0] == "queued"}
    assert set(queued) == {cid.split("/")[2] for cid in by_chain(r.cfg.run_dir, 1)}
    for pid in held:
        at, scored = queued[pid]
        assert at > first_drain, f"group {pid} went to the trainer before its retry settled"
        assert scored == 8, f"group {pid} trained on {scored} scored members, not the retried 8"
    assert any(i < first_drain for p, (i, _) in queued.items() if p not in held)  # the others did not wait
    eps = by_chain(r.cfg.run_dir, 1)
    for pid in held:
        members = [e for e in eps.values() if e["prompt_id"] == pid]
        assert len(members) == 8 and all(e["status"] == "scored" and e["advantage"] is not None for e in members)


def test_every_group_reaches_the_trainer_once(make_spec):
    r, res = run(make_spec(iterations=2, eval_every=None))  # 8 prompts, P = 4: two disjoint iterations
    assert res["status"] == "completed"
    assert len(r.trainer.added) == 8 and len(set(r.trainer.added)) == 8


# --- the failure table ----------------------------------------------------------------------------------------


def test_a_failed_trainer_check_stops_before_the_weights_are_published(make_spec):
    r, res = run(make_spec(iterations=2, eval_every=None, trainer={"fail_check_at_step": 1}))
    assert res["status"] == "stopped" and res["stop"]["reason"] == "trainer_check_failed"
    assert [c for c in r.sync.calls if c[0] == "publish"] == [] and complete_iterations(r.cfg.run_dir) == []


def test_a_failed_landing_check_stops_the_run(make_spec):
    r, res = run(make_spec(iterations=3, eval_every=None, fail_publish_at={"2": "NOT_LANDED"}))
    assert res["status"] == "stopped" and res["stop"]["reason"] == "landing_failed"
    assert complete_iterations(r.cfg.run_dir) == [1] and complete_checkpoints(r.cfg.run_dir / "ckpt") == [1]


def test_noop_iterations_warn_at_two_and_stop_at_five(make_spec):
    r, res = run(make_spec(iterations=7, eval_every=None, pool_size=16, trainer={"lr": 0.0}))
    assert res["status"] == "stopped" and res["stop"]["reason"] == "noop_streak"
    assert complete_iterations(r.cfg.run_dir) == [1, 2, 3, 4, 5]
    s = [read_summary(r.cfg.run_dir, n) for n in (1, 2, 5)]
    assert [x["noop_streak"] for x in s] == [1, 2, 5] and [x["sync"]["landing"] for x in s] == ["NOOP"] * 3
    assert "noop_steps" not in s[0]["warnings"] and "noop_steps" in s[1]["warnings"]


def test_the_eval_being_down_never_stops_training(make_spec):
    r = build_fake_run(make_spec(iterations=3, eval_every=1))

    def down(*a, **k):
        raise ConnectionError("eval endpoint down")

    r.exporter.export = down
    res = r.orchestrator.run()
    assert res["status"] == "completed" and res["iterations_completed"] == 3
    failed = [e for e in r.hook.events if e["event"] == "export_failed"]
    assert {e["iteration"] for e in failed} == {1, 2, 3}


def test_a_resume_onto_a_changed_prompt_pool_is_refused(make_spec):
    spec = make_spec(iterations=2, eval_every=None)
    assert run(spec)[1]["status"] == "completed"
    again = build_fake_run(dataclasses.replace(spec, pool_size=9))  # same config, one more prompt in the pool
    res = again.orchestrator.run(resume=True)
    assert res["status"] == "stopped" and res["stop"]["reason"] == "prompt_pool_changed"
    assert again.rollout.runs == []


def test_rollout_stop_reasons_stop_the_run(make_spec):
    r = build_fake_run(make_spec(iterations=1, eval_every=None))
    run_ = r.rollout.run

    def with_stop(*a, **k):
        out = run_(*a, **k)
        out.stop_reasons = ["2 of 4 groups have all 8 answers identical at T 0.7 (sampler suspected)"]
        return out

    r.rollout.run = with_stop
    res = r.orchestrator.run()
    assert res["status"] == "stopped" and res["stop"]["reason"] == "rollout_stop"
    assert "sampler suspected" in res["stop"]["detail"]["stop_reasons"][0]
