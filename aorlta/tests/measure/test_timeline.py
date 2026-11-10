"""C9 timeline: bounds, interval arithmetic and the critical path (C9.T1)."""

import json
import random

import pytest
from measure_helpers import C9_T1_PATH, T0, sp, v1_iteration

from aorlta.ids import canonical_json
from aorlta.measure import timeline as tl
from aorlta.measure.timeline import GAP, IDLE, Bounds, TimelineError
from aorlta.spans import Span, SpanRecorder


def rel(segments):
    return [(round(s.t_start - T0, 6), round(s.t_end - T0, 6), s.name) for s in segments]


# --- C9.T1 -----------------------------------------------------------------------------------------


def test_c9_t1_overlapping_spans_give_the_known_critical_path():
    spans = v1_iteration()
    (t,) = tl.summarize_run(spans)
    assert (t.bounds.t_begin - T0, t.bounds.t_end - T0, t.wall_s) == (0, 176, 176)
    assert rel(t.path) == C9_T1_PATH
    # the chain that set the pace is followed through its own turns
    assert {s.span.attrs.get("chain_id") for s in t.path if s.name in ("gen.turn0", "gen.repair", "compile")} \
        == {"run/1/standin:p/1"}
    assert t.critical_s == pytest.approx({"select": 1, "gen.turn0": 29, "compile": 2, "gen.repair": 49,
                                          "score.wait": 69.5, "train.final": 19.5, GAP: 1, "merge": 1,
                                          "write": 1.5, "reload": 1.5, "landing": 1})
    assert sum(t.critical_s.values()) == pytest.approx(t.wall_s)
    # the long ASan run and the streaming backward passes ran under other work: off the path
    assert "asan.run" not in t.critical_s and "train.stream" not in t.critical_s


def test_c9_t1_the_path_is_not_the_sum_of_durations():
    (t,) = tl.summarize_run(v1_iteration())
    busy = sum(t.busy_s.values())
    assert busy == pytest.approx(521)  # every span inside the wall, overlap counted again and again
    assert busy > 2.9 * t.wall_s
    segs = t.path
    assert segs[0].t_start == t.bounds.t_begin and segs[-1].t_end == t.bounds.t_end
    assert all(x.t_end == y.t_start for x, y in zip(segs, segs[1:]))  # contiguous, no overlap
    assert sum(s.duration for s in segs) == pytest.approx(t.wall_s)


def test_c9_t1_coverage_and_exclusive_time_count_wall_clock_once():
    (t,) = tl.summarize_run(v1_iteration())
    assert t.category_coverage_s["generation"] == pytest.approx(79)  # [1, 80], four overlapping spans
    assert t.category_coverage_s["check"] == pytest.approx(106.5)  # [30, 31] and [45, 150.5]
    # an instant goes to the first active category: generation > training > check > sync > other
    assert t.category_exclusive_s == pytest.approx({"generation": 79, "training": 52.5, "check": 37.5,
                                                    "sync": 5, "other": 1, "unclassified": 0, IDLE: 1})
    assert sum(t.category_exclusive_s.values()) == pytest.approx(t.wall_s)
    assert t.uncovered_s == pytest.approx(1) and not t.gap_flagged
    assert t.outside == {"record": 1, "ckpt": 1}  # after the landing: outside wall(n)
    assert t.coverage_s["gen.turn0"] == pytest.approx(49) and t.busy_s["gen.turn0"] == pytest.approx(122)
    assert t.category_critical_s == pytest.approx({"other": 1, "generation": 78, "check": 71.5, "training": 19.5,
                                                   GAP: 1, "sync": 5})


def _dag_schedule(rng, n_tasks):
    """Tasks in topological order; a task starts exactly when its last dependency ends, roots start
    at 0. Durations are continuous, so ties have probability 0. Returns (spans, known critical chain)."""
    names = ["gen.turn0", "gen.repair", "compile", "gate.parse", "san.consan", "train.stream", "merge"]
    tasks = []  # (start, end, deps)
    for i in range(n_tasks):
        deps = rng.sample(range(i), k=min(i, rng.randint(0, 3))) if i else []
        start = max((tasks[d][1] for d in deps), default=0.0)
        tasks.append((start, start + rng.uniform(0.1, 40.0), deps))
    sink = max(range(n_tasks), key=lambda i: tasks[i][1])
    chain, i = [], sink
    while True:
        chain.append(i)
        if not tasks[i][2]:
            break
        i = max(tasks[i][2], key=lambda d: tasks[d][1])
    chain.reverse()
    spans = [sp(rng.choice(names), s, e, 1, task=j) for j, (s, e, _) in enumerate(tasks)]
    return spans, chain, tasks[sink][1]


@pytest.mark.parametrize("seed", range(40))
def test_c9_t1_a_dag_schedule_gives_back_its_critical_chain(seed):
    rng = random.Random(seed)
    spans, chain, end = _dag_schedule(rng, rng.randint(5, 120))
    rng.shuffle(spans)
    path = tl.critical_path(spans, T0, T0 + end)
    assert GAP not in [s.name for s in path]
    assert [s.span.attrs["task"] for s in path] == chain
    assert sum(s.duration for s in path) == pytest.approx(end)


# --- the walk's rule against a direct transcription of it ------------------------------------------


def _reference_walk(spans, lo, hi):
    pool = [s for s in spans if s.t_end > s.t_start and lo < s.t_end <= hi]
    cursor, after, out = hi, None, []
    while cursor > lo:
        cands = [s for s in pool if s.t_end <= cursor]
        if not cands:
            out.append((lo, cursor, GAP))
            break
        latest = max(s.t_end for s in cands)
        best = sorted((s for s in cands if s.t_end == latest),
                      key=lambda s: (after is None or tl.causal_key(s) != after, -s.t_start, s.name,
                                     canonical_json(s.to_json())))[0]
        if latest < cursor:
            out.append((latest, cursor, GAP))
        start = max(best.t_start, lo)
        out.append((start, latest, best.name))
        cursor, after = start, tl.causal_key(best)
    return out[::-1]


@pytest.mark.parametrize("seed", range(60))
def test_the_walk_matches_a_direct_transcription_with_ties_and_zero_length_spans(seed):
    rng = random.Random(1000 + seed)
    spans = []
    for _ in range(rng.randint(1, 80)):
        s = rng.randint(-10, 200) / 2  # half-second grid: many exact ties
        e = s + rng.choice([0, 0.5, 1, rng.randint(1, 80) / 2])
        attrs = {"chain_id": f"c{rng.randint(0, 4)}"} if rng.random() < 0.7 else {}
        spans.append(sp(rng.choice(["compile", "gen.turn0", "gate.parse", "score.wait"]), s, e, 1, **attrs))
    lo, hi = T0 + rng.randint(-5, 20), T0 + rng.randint(60, 140)
    got = [(s.t_start, s.t_end, s.name) for s in tl.critical_path(spans, lo, hi)]
    assert got == _reference_walk(spans, lo, hi)
    rng.shuffle(spans)
    assert [(s.t_start, s.t_end, s.name) for s in tl.critical_path(spans, lo, hi)] == got
    assert got[0][0] == lo and got[-1][1] == hi and all(x[1] == y[0] for x, y in zip(got, got[1:]))


def test_an_exact_tie_goes_to_the_same_chain_then_to_the_later_start():
    x, y = "run/1/standin:p/0", "run/1/standin:p/1"
    spans = [sp("compile", 10, 20, chain_id=y), sp("compile", 12, 20, chain_id=x),
             sp("gen.repair", 20, 30, chain_id=x)]
    assert rel(tl.critical_path(spans, T0, T0 + 30)) == [(0, 12, GAP), (12, 20, "compile"), (20, 30, "gen.repair")]
    spans[1] = sp("compile", 8, 20, chain_id=x)  # still the same chain, though it started earlier
    assert rel(tl.critical_path(spans, T0, T0 + 30))[1] == (8, 20, "compile")
    spans[1] = sp("compile", 8, 20, chain_id="other")  # no chain match: the later start wins
    assert rel(tl.critical_path(spans, T0, T0 + 30))[1] == (10, 20, "compile")


def test_gaps_are_explicit_and_spans_outside_the_window_are_not_candidates():
    spans = [sp("gen.turn0", -5, 4), sp("merge", 6, 8), sp("record", 8, 12), sp("compile", 3, 3)]
    assert rel(tl.critical_path(spans, T0, T0 + 10)) == [(0, 4, "gen.turn0"), (4, 6, GAP), (6, 8, "merge"),
                                                          (8, 10, GAP)]
    assert rel(tl.critical_path([], T0, T0 + 3)) == [(0, 3, GAP)]
    with pytest.raises(TimelineError):
        tl.critical_path(spans, T0 + 5, T0 + 5)


def test_a_gap_above_five_percent_of_wall_is_flagged():
    spans = [sp("select", 0, 1), sp("gen.turn0", 1, 90), sp("landing", 95.9, 100, verdict="LANDED")]
    (t,) = tl.summarize_run(spans)
    assert t.uncovered_s == pytest.approx(5.9) and t.gap_flagged
    spans[2] = sp("landing", 94, 100, verdict="LANDED")
    (t,) = tl.summarize_run(spans)
    assert t.uncovered_s == pytest.approx(4) and not t.gap_flagged


def test_unknown_span_names_are_reported_not_dropped():
    spans = [sp("select", 0, 1), sp("gen.turn0", 1, 5), sp("mystery.phase", 5, 9),
             sp("landing", 9, 10, verdict="NOOP")]
    (t,) = tl.summarize_run(spans)
    assert t.unclassified == ("mystery.phase",)
    assert t.category_exclusive_s["unclassified"] == pytest.approx(4)


# --- bounds ------------------------------------------------------------------------------------------


def test_v1_bounds_run_from_selection_to_the_last_passed_landing():
    spans = v1_iteration() + [sp("landing", 40, 41, verdict=["LANDED", "LANDED"])]  # e.g. after an engine restart
    b = tl.bounds_v1(spans, 1)
    assert (b.t_begin - T0, b.t_end - T0, b.begin_from, b.end_from) == (0, 176, "select", "landing:LANDED")
    noop = [s if s.name != "landing" else sp("landing", 175, 176, verdict=["NOOP", "NOOP"]) for s in v1_iteration()]
    assert tl.bounds_v1(noop, 1).end_from == "landing:NOOP"


@pytest.mark.parametrize("change, message", [
    (lambda s: [x for x in s if x.name != "select"], "0 'select' spans"),
    (lambda s: s + [sp("select", 2, 3)], "2 'select' spans"),
    (lambda s: [x for x in s if x.name != "landing"], "no 'landing' span"),
    (lambda s: [x if x.name != "landing" else sp("landing", 175, 176) for x in s], "no 'verdict'"),
    (lambda s: [x if x.name != "landing" else sp("landing", 175, 176, verdict="NOT_LANDED") for x in s],
     "did not pass"),
    (lambda s: [x if x.name != "landing" else sp("landing", 175, 176, verdict=["LANDED", "MISMATCH"]) for x in s],
     "did not pass"),
    (lambda s: s + [sp("landing", 200, 201, verdict="MISMATCH")], "did not pass"),
])
def test_v1_bounds_fail_closed(change, message):
    with pytest.raises(TimelineError, match=message):
        tl.bounds_v1(change(v1_iteration()), 1)


def _wp7_shaped(it=1):
    """Spans as WP7's loop and fake sync emit them: `landing` carries only `version`, and the verdict
    goes to the iteration summary; a D7 retry is a `score.retry` span; set-up spans live in iteration 0."""
    landing_free = [s for s in v1_iteration(it) if s.name != "landing"]
    return [sp("setup.engines", -50, -10, it=0), sp("boot.custom", -10, -5, it=0)] + landing_free + [
        sp("score.retry", 150.5, 150.6, it, candidates=1, deadline_s=2700.0),
        sp("landing", 175, 176, it, version=it)]


def test_a_landing_without_its_own_verdict_takes_the_summarys():
    spans = _wp7_shaped()
    with pytest.raises(TimelineError, match="no 'verdict' attr and no verdict was given"):
        tl.run_bounds(spans)
    (t,) = tl.summarize_run(spans, verdicts={1: ["LANDED"] * 4})
    assert t.wall_s == 176 and t.bounds.end_from == "landing:LANDED"
    assert t.category_coverage_s["check"] == pytest.approx(106.6)  # the retry follows score.wait: +0.1 s
    assert t.unclassified == ()
    with pytest.raises(TimelineError, match="did not pass"):
        tl.run_bounds(spans, verdicts={1: ["LANDED", "NOT_LANDED"]})


def test_a_summary_verdict_that_contradicts_the_span_is_refused():
    with pytest.raises(TimelineError, match=r"the landing span says \['LANDED'\], its summary says \['NOOP'\]"):
        tl.bounds_v1(v1_iteration(), 1, {1: ["NOOP"]})
    assert tl.bounds_v1(v1_iteration(), 1, {1: ["LANDED", "LANDED"]}).end_from == "landing:LANDED"
    assert tl.bounds_v1(v1_iteration(), 1, {2: ["NOOP"]}).end_from == "landing:LANDED"  # another iteration's


def test_v2_bounds_run_between_consecutive_publications():
    spans = [sp("landing", 5, 6, it=0, verdict="LANDED")] + v1_iteration(1) + \
        [Span(s.name, 2, s.t_start + 200, s.t_end + 200, s.attrs) for s in v1_iteration(2)]
    b1, b2 = tl.bounds_v2(spans, 1), tl.bounds_v2(spans, 2)
    assert (b1.t_begin - T0, b1.t_end - T0, b1.begin_from) == (6, 176, "landing(0)")
    assert (b2.t_begin - T0, b2.t_end - T0, b2.wall_s) == (176, 376, 200)
    with pytest.raises(TimelineError, match="iteration 0: no 'landing'"):
        tl.bounds_v2(spans[1:], 1)


def _no_step_iteration(it=1, at=152.0):
    """An iteration that took no optimizer step (every group scored alike): no step and no sync tail, and the
    loop's zero-length `no_step` marker where the landing would have ended it."""
    tail = {"train.final", "merge", "write", "reload", "landing"}
    return [s for s in v1_iteration(it) if s.name not in tail] + [sp("no_step", at, at, it, reason="zero_advantage")]


def test_an_iteration_that_took_no_step_ends_at_its_no_step_marker():
    b = tl.bounds_v1(_no_step_iteration(), 1)
    assert (b.t_begin - T0, b.t_end - T0, b.begin_from, b.end_from) == (0, 152, "select", "no_step")
    (t,) = tl.summarize_run(_no_step_iteration(), verdicts={1: []})
    assert t.wall_s == 152 and sum(t.critical_s.values()) == pytest.approx(152) and t.unclassified == ()
    spans = [sp("landing", 5, 6, it=0, verdict="LANDED")] + _no_step_iteration(1) + \
        [Span(s.name, 2, s.t_start + 200, s.t_end + 200, s.attrs) for s in v1_iteration(2)]
    b2 = tl.bounds_v2(spans, 2)
    assert (b2.t_begin - T0, b2.t_end - T0, b2.begin_from) == (152, 376, "no_step(1)")
    assert tl.bounds_v2(spans, 1).end_from == "no_step"


@pytest.mark.parametrize("spans, verdicts, message", [
    (_no_step_iteration() + [sp("landing", 175, 176, verdict="LANDED")], None, "1 'landing' spans"),
    (_no_step_iteration() + [sp("no_step", 160, 160)], None, "2 'no_step' markers"),
    (_no_step_iteration(), {1: ["LANDED"]}, "took no step, but its summary gives landing verdicts"),
])
def test_a_no_step_marker_fails_closed(spans, verdicts, message):
    with pytest.raises(TimelineError, match=message):
        tl.bounds_v1(spans, 1, verdicts)


def test_envelope_bounds_are_marked_as_such():
    spans = [sp("gen.turn0", 3, 10), sp("compile", 10, 11)]
    b = tl.bounds_envelope(spans, 1)
    assert (b.t_begin - T0, b.t_end - T0, b.from_markers) == (3, 11, False)
    assert tl.bounds_v1(v1_iteration(), 1).from_markers
    with pytest.raises(TimelineError, match="unknown mode"):
        tl.run_bounds(spans, "v3")


def test_iteration_zero_is_setup_and_not_summarized():
    spans = [sp("engine.start", -100, -2, it=0)] + v1_iteration()
    assert [t.iteration for t in tl.summarize_run(spans)] == [1]


def test_spans_from_the_recorder_round_trip_through_the_file(tmp_path):
    rec = SpanRecorder()
    for s in v1_iteration():
        rec.record(s.name, s.iteration, s.t_start, s.t_end, **s.attrs)
    path = rec.write_jsonl(tmp_path / "spans.jsonl")
    assert tl.main([str(path), "--path", "1", "--json", str(tmp_path / "t.json")]) == 0
    (out,) = json.loads((tmp_path / "t.json").read_text())
    assert [(round(s["t_start"] - T0, 6), round(s["t_end"] - T0, 6), s["name"]) for s in out["critical_path"]] \
        == C9_T1_PATH


def test_the_cli_refuses_a_run_without_markers(tmp_path, capsys):
    rec = SpanRecorder()
    rec.record("gen.turn0", 1, T0, T0 + 5)
    path = rec.write_jsonl(tmp_path / "spans.jsonl")
    assert tl.main([str(path)]) == 1
    assert "'select'" in capsys.readouterr().err
    assert tl.main([str(path), "--mode", "envelope"]) == 0


# --- intervals ---------------------------------------------------------------------------------------


def test_union_merges_overlapping_and_touching_intervals():
    assert tl.union([(5, 6), (0, 2), (1, 3), (3, 4), (8, 8)]) == [(0, 4), (5, 6)]
    assert tl.measure([(0, 10), (2, 3), (9, 12)]) == 12
    with pytest.raises(TimelineError):
        tl.union([(3, 2)])
    with pytest.raises(TimelineError):
        tl.union([(0, float("inf"))])


@pytest.mark.parametrize("a, b, expected", [
    ([(0, 10)], [(2, 3), (5, 7)], [(0, 2), (3, 5), (7, 10)]),
    ([(0, 10), (15, 25)], [(5, 20)], [(0, 5), (20, 25)]),
    ([(0, 10)], [(-5, 15)], []),
    ([(0, 10)], [], [(0, 10)]),
    ([(0, 4), (6, 9)], [(4, 6)], [(0, 4), (6, 9)]),
])
def test_subtract(a, b, expected):
    assert tl.subtract(a, b) == expected


@pytest.mark.parametrize("seed", range(30))
def test_subtract_agrees_with_point_sampling(seed):
    rng = random.Random(seed)
    a = [(x, x + rng.randint(0, 9)) for x in (rng.randint(0, 50) for _ in range(rng.randint(0, 8)))]
    b = [(x, x + rng.randint(0, 9)) for x in (rng.randint(0, 50) for _ in range(rng.randint(0, 8)))]
    got = tl.subtract(a, b)
    for k in range(0, 1200):
        p = k / 20 + 0.025
        inside = lambda ivs: any(lo <= p < hi for lo, hi in ivs)  # noqa: E731
        assert inside(got) == (inside(a) and not inside(b))


def test_bounds_wall_property():
    assert Bounds(3, 10.0, 25.5, "select", "landing:LANDED").wall_s == 15.5
