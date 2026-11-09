"""D7's retry (Section 6.6 step 7) through the reward service: `resubmit` re-checks a candidate's unsettled legs only,
reusing every settled leg from the cache, as the next attempt of the same check; and a stand-in profile may leave out
legs its reward reads."""

from __future__ import annotations

from aorlta.reward.adapters.fake import FakeAdapterConfig, FakeOutcome
from aorlta.reward.profiles import profile_from_dict
from aorlta.reward.rewardfn.standin_v11 import StandinV11
from aorlta.reward.service import RewardService

from reward_helpers import cand, fake_rig, reward_section


def runs(stats) -> dict[str, int]:
    return {leg: sum(by_state.values()) for leg, by_state in stats.leg_runs.items()}


def test_resubmit_reruns_only_the_unsettled_leg_as_its_next_attempt_and_reuses_every_settled_leg(tmp_path):
    c = cand(1)
    rig = fake_rig(tmp_path, adapter=FakeAdapterConfig(
        overrides={(c.candidate_id, "san.asan", 1): FakeOutcome("error", "timeout")}))  # a watchdog-style timeout
    try:
        f1 = rig.service.submit(c)
        first = f1.result(timeout=60)
        assert (first.status, first.reason) == ("unscored", "sanitizer_not_settled")
        assert (first.legs["san.asan"].state, first.legs["san.asan"].reason) == ("unsettled", "timeout")
        assert rig.service.submit(c) is f1  # submit stays idempotent: it cannot re-check anything
        before = runs(rig.service.stats())
        f2 = rig.service.resubmit(c)
        assert f2 is not f1
        second = f2.result(timeout=60)
        assert second.status == "scored" and second.reward is not None
        asan = second.legs["san.asan"]
        assert (asan.state, asan.attempts, asan.cache_hit) == ("clean", 2, False)
        settled = [leg for leg, r in first.legs.items() if r.state in ("clean", "finding", "fail")]
        assert settled and all(second.legs[leg].cache_hit for leg in settled)
        after = runs(rig.service.stats())
        assert {leg: n - before.get(leg, 0) for leg, n in after.items() if n != before.get(leg, 0)} == {"san.asan": 1}
        assert rig.service.stats().resubmitted == 1
    finally:
        rig.close()


def test_resubmit_leaves_a_scored_or_pending_candidate_alone_and_submits_an_unknown_one(tmp_path):
    rig = fake_rig(tmp_path)
    try:
        c1, c2 = cand(1), cand(2)
        f1 = rig.service.submit(c1)
        assert f1.result(timeout=60).status == "scored"
        assert rig.service.resubmit(c1) is f1  # scored: nothing to re-check
        f2 = rig.service.resubmit(c2)  # never submitted: an ordinary submit
        assert f2.result(timeout=60).status == "scored"
        assert rig.service.resubmit(c2) is f2
        assert rig.service.stats().resubmitted == 0
    finally:
        rig.close()


def test_a_twice_unsettled_leg_stays_unscored_never_clean(tmp_path):
    c = cand(3)
    rig = fake_rig(tmp_path, adapter=FakeAdapterConfig(
        overrides={(c.candidate_id, "san.asan"): FakeOutcome("error", "timeout")}))  # on every attempt
    try:
        assert rig.service.submit(c).result(timeout=60).status == "unscored"
        again = rig.service.resubmit(c).result(timeout=60)
        assert (again.status, again.reward, again.legs["san.asan"].state) == ("unscored", None, "unsettled")
        assert again.legs["san.asan"].attempts == 2
    finally:
        rig.close()


def _profile(name: str, legs: list[str]):
    return profile_from_dict({"name": name, "reward_fn": "standin.v11",
                              "pools": {"cpu": {"resource": "local_cpu", "adapter": "fake", "timeout": "gate"}},
                              "stages": [{"name": "S0", "pool": "cpu", "legs": legs}]})


def test_a_standin_profile_may_leave_out_legs_its_reward_scores():
    svc = RewardService(reward_section(), _profile("standin-static", ["gate.parse", "gate.compile"]), StandinV11(),
                        {"fake": _fake})
    svc.close()


def _fake(wid):
    from aorlta.reward.adapters.fake import FakeCheckAdapter

    return FakeCheckAdapter(FakeAdapterConfig())
