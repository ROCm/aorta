"""C9 per-256 metric (C9.T2)."""

import json

import pytest

from aorlta.measure import per256 as p
from aorlta.measure.per256 import EpisodeCounts, IterationStats, Per256Error
from aorlta.spans import Span

T0 = 1_793_600_000.0

# The synthetic run: 5 iterations of P = 25 prompts x K = 10 chains = 250 samples. 250 of the 1,250
# samples are unscored (20%), unevenly: 50, 40, 60, 50, 50. Iteration 4 hit the scoring deadline.
WALLS = {1: 900.0, 2: 1000.0, 3: 1200.0, 4: 5000.0, 5: 1100.0}
UNSCORED = {1: 50, 2: 40, 3: 60, 4: 50, 5: 50}
STALLS = {4: "scoring deadline"}

# Hand calculation, per256(n) = 256 * wall(n) / scored(n):
#   it 1: 256*900/200  = 1152                  (reported nowhere: the metric starts at iteration 2)
#   it 2: 256*1000/210 = 25600/21 = 1219.047619047619
#   it 3: 256*1200/190 = 30720/19 = 1616.842105263158
#   it 4: 256*5000/200 = 6400                  (stall)
#   it 5: 256*1100/200 = 1408
# sorted 2..5: 1219.048, 1408, 1616.842, 6400
#   median = (1408 + 1616.842)/2                          = 1512.421052631579
#   q1 at rank 0.75: 1219.048 + 0.75*(1408 - 1219.048)    = 1360.761904761905
#   q3 at rank 2.25: 1616.842 + 0.25*(6400 - 1616.842)    = 2812.631578947368
#   mean with stalls    = (1219.048 + 1616.842 + 6400 + 1408)/4 = 2660.972431077694
#   mean without stalls = (1219.048 + 1616.842 + 1408)/3        = 1414.629908103592
HAND = {"per256": {1: 1152.0, 2: 1219.047619047619, 3: 1616.842105263158, 4: 6400.0, 5: 1408.0},
        "median": 1512.421052631579, "q1": 1360.761904761905, "q3": 2812.631578947368,
        "mean_with": 2660.972431077694, "mean_without": 1414.629908103592}


def _iteration_spans(n, origin, wall):
    o = origin
    spans = [Span("select", n, o, o + 0.2, {})]
    for k in range(10):  # overlapping chains: generation alone is busy far beyond its wall-clock length
        spans.append(Span("gen.turn0", n, o + 0.2, o + 0.3 * wall + k, {"chain_id": f"run/{n}/standin:p0/{k}"}))
    spans += [Span("score.wait", n, o + 0.3 * wall + 9, o + 0.9 * wall, {}),
              Span("san.asan", n, o + 0.25 * wall, o + 0.9 * wall - 1, {"candidate_id": f"run/{n}/standin:p0/0"}),
              Span("train.final", n, o + 0.9 * wall, o + wall - 4, {}),
              Span("merge", n, o + wall - 4, o + wall - 3, {}), Span("write", n, o + wall - 3, o + wall - 2, {}),
              Span("reload", n, o + wall - 2, o + wall - 1, {}),
              Span("landing", n, o + wall - 1, o + wall, {"verdict": ["LANDED"] * 4}),
              Span("record", n, o + wall, o + wall + 2, {}), Span("ckpt", n, o + wall + 2, o + wall + 5, {})]
    return spans


def synthetic_run():
    spans, origin = [Span("engine.start", 0, T0 - 120, T0 - 1, {"engines": 4})], T0
    episodes = []
    for n, wall in WALLS.items():
        spans += _iteration_spans(n, origin, wall)
        origin += wall + 5 + 30  # record, ckpt and 30 s before the next selection: outside every wall
        for j in range(250):
            cid = f"run/{n}/standin:p{j // 10}/{j % 10}"
            unscored = j < UNSCORED[n]
            episodes.append({"schema": "aorlta.episode/1", "iteration": n, "chain_id": cid,
                             "status": "unscored" if unscored else "scored",
                             "reward": None if unscored else -0.5 + (j % 7) / 10,
                             "unscored_reason": ("asan_timeout" if j % 2 else "worker_stalled") if unscored else None,
                             "turns": [{"finish_reason": {"type": "length" if j % 25 == 3 else "stop"}}],
                             "final_turn": 0})
    return spans, episodes


def run_summary():
    spans, episodes = synthetic_run()
    stats = p.iteration_stats(p.walls_from_spans(spans), p.counts_from_episodes(episodes), STALLS)
    return stats, p.summarize(stats)


# --- C9.T2 -----------------------------------------------------------------------------------------


def test_c9_t2_per256_on_a_run_with_20_percent_unscored_equals_the_hand_calculation():
    stats, s = run_summary()
    assert sum(x.samples - x.scored for x in stats) / sum(x.samples for x in stats) == 0.2
    assert {x.iteration: x.wall_s for x in stats} == WALLS  # from the markers, not from summed spans
    assert s.per256_s == pytest.approx(HAND["per256"], rel=1e-12)
    assert s.iterations == (2, 3, 4, 5)
    assert s.median_s == pytest.approx(HAND["median"], rel=1e-12)
    assert (s.q1_s, s.q3_s) == pytest.approx((HAND["q1"], HAND["q3"]), rel=1e-12)
    assert s.iqr_s == pytest.approx(HAND["q3"] - HAND["q1"], rel=1e-12)
    assert s.mean_with_stalls_s == pytest.approx(HAND["mean_with"], rel=1e-12)
    assert s.mean_without_stalls_s == pytest.approx(HAND["mean_without"], rel=1e-12)
    assert s.stalled == (4,)
    assert (s.scored, s.samples) == (800, 1000)


def test_c9_t2_resources_per_256_scored_count_every_gpu_held():
    _, s = run_summary()
    r = p.resources(s, gpus_held=8, cpu_nodes_held=1)  # 4 trainer/engine GPUs + 4 dedicated ASan GPUs
    assert r["gpu_hours_per256_median"] == pytest.approx(8 * HAND["median"] / 3600, rel=1e-12)
    assert r["cpu_node_hours_per256_median"] == pytest.approx(HAND["median"] / 3600, rel=1e-12)
    assert r["gpu_hours_per256_mean_without_stalls"] == pytest.approx(8 * HAND["mean_without"] / 3600, rel=1e-12)


def test_c9_t2_episode_counts_carry_unscored_by_reason_truncation_and_the_scored_mean():
    _, episodes = synthetic_run()
    c = p.counts_from_episodes(episodes)[2]
    assert (c.samples, c.scored, c.unscored_by_reason) == (250, 210, {"worker_stalled": 20, "asan_timeout": 20})
    assert c.truncated == 10  # j % 25 == 3, on the final turn
    rewards = [-0.5 + (j % 7) / 10 for j in range(40, 250)]
    assert c.mean_train_reward == pytest.approx(sum(rewards) / len(rewards))


# --- rules -----------------------------------------------------------------------------------------


def test_each_iteration_is_divided_by_its_own_scored_count():
    stats = [IterationStats(1, 100.0, 256, 256, False), IterationStats(2, 100.0, 256, 128, False),
             IterationStats(3, 100.0, 256, 256, False)]
    s = p.summarize(stats)
    assert s.per256_s == {1: 100.0, 2: 200.0, 3: 100.0}
    assert s.median_s == 150.0 and s.mean_with_stalls_s == 150.0


def test_an_unscored_iteration_1_is_not_reported_and_does_not_block_the_summary():
    s = p.summarize([IterationStats(1, 10.0, 256, 0, False), IterationStats(2, 20.0, 256, 256, False)])
    assert s.per256_s == {1: None, 2: 20.0} and s.median_s == 20.0 and s.iterations == (2,)
    assert s.to_json()["per256_s"] == {"1": None, "2": 20.0}


def test_a_single_reported_iteration_has_no_iqr():
    s = p.summarize([IterationStats(1, 10.0, 256, 256, False), IterationStats(2, 20.0, 256, 256, False)])
    assert (s.median_s, s.q1_s, s.q3_s, s.iqr_s) == (20.0, None, None, None)


def test_with_every_reported_iteration_stalled_there_is_no_calm_mean():
    s = p.summarize([IterationStats(1, 10.0, 256, 256, False), IterationStats(2, 20.0, 256, 256, True, "deadline")])
    assert s.mean_without_stalls_s is None and s.mean_with_stalls_s == 20.0
    assert p.resources(s, gpus_held=4, cpu_nodes_held=1)["gpu_hours_per256_mean_without_stalls"] is None


@pytest.mark.parametrize("stats, message", [
    ([], "no iterations"),
    ([IterationStats(1, 10.0, 256, 256, False)], "no iterations from 2"),
    ([IterationStats(1, 10.0, 256, 256, False), IterationStats(3, 10.0, 256, 256, False)], r"missing from 1\.\.3"),
    ([IterationStats(2, 10.0, 256, 256, False)] * 2, "appears twice"),
    ([IterationStats(1, 10.0, 256, 256, False), IterationStats(2, 10.0, 256, 0, False)], "no scored samples"),
])
def test_summarize_fails_closed(stats, message):
    with pytest.raises(Per256Error, match=message):
        p.summarize(stats)


def test_last_must_be_covered():
    stats = [IterationStats(n, 10.0, 256, 256, False) for n in (1, 2, 3)]
    assert p.summarize(stats, last=2).iterations == (2,)
    with pytest.raises(Per256Error, match="missing"):
        p.summarize(stats, last=4)


@pytest.mark.parametrize("kwargs, message", [
    (dict(iteration=0), "iteration must be"),
    (dict(wall_s=0.0), "wall_s"),
    (dict(wall_s=float("nan")), "wall_s"),
    (dict(scored=300), "scored 300 > samples"),
    (dict(scored=True), "scored must be an int"),
    (dict(stalled=1), "stalled must be"),
    (dict(stalled=True), "needs its reason"),
])
def test_iteration_stats_are_checked(kwargs, message):
    base = dict(iteration=2, wall_s=10.0, samples=256, scored=200, stalled=False)
    with pytest.raises(Per256Error, match=message):
        IterationStats(**{**base, **kwargs})


def test_the_span_envelope_is_never_a_wall_for_the_metric():
    spans, _ = synthetic_run()
    with pytest.raises(Per256Error, match="v1 or v2"):
        p.walls_from_spans(spans, "envelope")
    no_select = [s for s in spans if not (s.name == "select" and s.iteration == 3)]
    with pytest.raises(p.MeasureError, match="iteration 3: 0 'select'"):
        p.walls_from_spans(no_select)


def test_v2_walls_include_the_time_between_iterations():
    spans, _ = synthetic_run()
    spans.append(Span("landing", 0, T0 - 2, T0 - 1, {"verdict": "LANDED"}))
    walls = p.walls_from_spans(spans, "v2")
    assert walls[2] == pytest.approx(WALLS[2] + 35)  # the previous record, ckpt and 30 s idle are inside
    assert walls[1] == pytest.approx(WALLS[1] + 1)


def _episode(**over):
    rec = {"schema": "aorlta.episode/1", "iteration": 1, "chain_id": "r/1/p/0", "status": "scored", "reward": 0.5,
           "unscored_reason": None, "turns": [{"finish_reason": {"type": "stop"}}], "final_turn": 0}
    rec.update(over)
    return rec


@pytest.mark.parametrize("records, message", [
    ([_episode(schema="aorlta.episode/2")], "schema"),
    ([_episode(status="scored", reward=None)], "scored, but its reward is None"),
    ([_episode(status="scored", reward=float("nan"))], "scored, but"),
    ([_episode(status="unscored", reward=-1.0, unscored_reason="x")], "unscored needs"),
    ([_episode(status="unscored", reward=None, unscored_reason=None)], "unscored needs"),
    ([_episode(status="quarantined", reward=-1.0, unscored_reason="quarantined")], "quarantined needs"),
    ([_episode(status="quarantined", reward=None, unscored_reason=None)], "quarantined needs"),
    ([_episode(status="pending")], "status must be"),
    ([_episode(), _episode()], "appears twice"),
    ([_episode(final_turn=1)], "final turn"),
    ([_episode(turns=[{"finish_reason": {}}])], "finish_reason"),
    ([_episode(iteration=0)], "iteration >= 1"),
])
def test_episode_counts_fail_closed(records, message):
    with pytest.raises(Per256Error, match=message):
        p.counts_from_episodes(records)


def _no_reply(**over):
    """An engine_error chain whose engine failed before its first reply."""
    return _episode(**{"chain_id": "r/1/p/1", "status": "unscored", "reward": None, "unscored_reason": "engine_error",
                       "end_reason": "engine_error", "turns": [], "final_turn": -1, **over})


def test_a_chain_the_engine_lost_before_its_first_reply_is_unscored_and_not_truncated():
    c = p.counts_from_episodes([_episode(), _no_reply()])[1]
    assert (c.samples, c.scored, c.unscored_by_reason, c.truncated) == (2, 1, {"engine_error": 1}, 0)


@pytest.mark.parametrize("over", [
    dict(final_turn=0),
    dict(final_turn=-2),
    dict(turns=[{"finish_reason": {"type": "stop"}}]),
    dict(end_reason="compiled"),
    dict(end_reason=None),
    dict(status="scored", reward=0.5, unscored_reason=None),
])
def test_no_final_turn_is_refused_unless_an_unscored_engine_error_has_no_turns(over):
    with pytest.raises(Per256Error, match="final turn"):
        p.counts_from_episodes([_no_reply(**over)])


def test_a_quarantined_chain_is_unscored_under_its_reason_and_out_of_the_scored_count():
    q = _episode(chain_id="r/1/p/1", status="quarantined", reward=None, unscored_reason="quarantined",
                 end_reason="quarantined")
    c = p.counts_from_episodes([_episode(), q])[1]
    assert (c.samples, c.scored, c.unscored_by_reason, c.mean_train_reward) == (2, 1, {"quarantined": 1}, 0.5)


def test_truncation_is_read_from_the_final_turn():
    rec = _episode(turns=[{"finish_reason": {"type": "length"}}, {"finish_reason": {"type": "stop"}}], final_turn=1)
    assert p.counts_from_episodes([rec])[1].truncated == 0
    assert p.counts_from_episodes([dict(rec, final_turn=0)])[1].truncated == 1


def test_joining_walls_counts_and_stalls_needs_the_same_iterations():
    counts = {1: EpisodeCounts(1, 10, 10), 2: EpisodeCounts(2, 10, 10)}
    with pytest.raises(Per256Error, match="differ"):
        p.iteration_stats({1: 5.0}, counts, {})
    with pytest.raises(Per256Error, match="no data"):
        p.iteration_stats({1: 5.0, 2: 5.0}, counts, {7: "deadline"})


def test_read_episodes(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(_episode()) + "\n\n" + json.dumps(_episode(chain_id="r/1/p/1")) + "\n")
    assert [r["chain_id"] for r in p.read_episodes([path])] == ["r/1/p/0", "r/1/p/1"]
    path.write_text("{not json\n")
    with pytest.raises(Per256Error, match="episodes.jsonl:1"):
        p.read_episodes([path])


def test_per256_refuses_no_scored_samples():
    with pytest.raises(Per256Error, match="undefined"):
        p.per256(100.0, 0)
    assert p.per256(100.0, 512) == 50.0
