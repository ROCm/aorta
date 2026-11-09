"""verify70 (C9.T3): synthetic summaries with known answers, timestamps and durations-only."""

import datetime as dt
import json

import pytest

from aorlta.measure import verify70 as v
from aorlta.measure.verify70 import IterationPhases, Verify70Error

T0 = 1_793_600_000.0

# Phase names as another training loop might log them, mapped to categories the way a mapping file would.
CLASSES = {"generate": "generation", "train": "training", "gate": "check", "scoring_wait": "check",
           "asan_settle": "check", "compile_check": "check", "compile_repair": "mixed", "weight_sync": "other"}


def ts(n, wall, *phases):
    return IterationPhases(n, t_begin=T0, t_end=T0 + wall,
                           phases=tuple((name, T0 + a, T0 + b) for name, a, b in phases))


# Timestamps. Hand answers (check-only = union of checks, minus generation and training):
TIMESTAMPS = [
    # 1: sequential phases. Checks 100 + 200 + 1200 = 1500 of 2000 -> 0.75, confirmed.
    ts(1, 2000, ("generate", 0, 300), ("compile_check", 300, 400), ("gate", 400, 600), ("scoring_wait", 600, 1800),
       ("train", 1800, 1900), ("weight_sync", 1900, 1950)),
    # 2: check phases overlap each other. Union [400, 1600] = 1200 -> 0.60, refuted.
    #    (Their summed durations, 600 + 1000 + 700 = 2300, are more than the wall.)
    ts(2, 2000, ("generate", 0, 400), ("gate", 400, 1000), ("asan_settle", 500, 1500), ("scoring_wait", 900, 1600),
       ("train", 1600, 1900)),
    # 3: checks overlap generation and training. Union of checks [200, 1700] = 1500 (0.75), but only
    #    [1000, 1500] has no generation or training under it: 500 -> 0.25, refuted.
    ts(3, 2000, ("generate", 0, 1000), ("gate", 200, 1100), ("asan_settle", 300, 1700), ("scoring_wait", 1100, 1500),
       ("train", 1500, 1800)),
    # 4: a compile-repair envelope (mixed). As generation: checks [600, 1700] = 1100 -> 0.55. As check:
    #    [200, 1700] = 1500 -> 0.75. Straddles 0.7: undetermined.
    ts(4, 2000, ("generate", 0, 200), ("compile_repair", 200, 600), ("gate", 600, 700), ("scoring_wait", 700, 1700),
       ("train", 1700, 1900)),
    # 5: mixed, with its compile checks inside it, and a phase starting 50 s before the iteration (clipped).
    #    Lower: [200, 990] = 790 -> 0.79; upper: [50, 990] = 940 -> 0.94. Confirmed.
    ts(5, 1000, ("generate", -50, 50), ("compile_repair", 50, 200), ("compile_check", 100, 180), ("gate", 200, 990)),
]
TIMESTAMP_ANSWERS = {1: (0.75, 0.75, "confirmed", v.EXACT), 2: (0.60, 0.60, "refuted", v.EXACT),
                     3: (0.25, 0.25, "refuted", v.EXACT), 4: (0.55, 0.75, "undetermined", v.MIXED),
                     5: (0.79, 0.94, "confirmed", v.MIXED)}


def du(n, wall, **durations):
    return IterationPhases(n, wall_s=wall, durations={k: float(x) for k, x in durations.items()})


# Durations only. lower = longest check phase / wall; upper = min(1, (checks + compile-repair) / wall).
DURATIONS = [
    du(1, 2000, gate=300, scoring_wait=1500, asan_settle=200, compile_check=0, compile_repair=300, generate=400,
       train=200),  # lower 1500/2000 = 0.75; upper min(1, 2300/2000) = 1
    du(2, 2000, gate=200, scoring_wait=600, asan_settle=100, compile_check=0, compile_repair=300),  # 0.30, 0.60
    du(3, 2000, gate=500, scoring_wait=900, asan_settle=300, compile_check=0, compile_repair=200),  # 0.45, 0.95
    du(4, 1000, gate=100, scoring_wait=720, asan_settle=0, compile_check=0, compile_repair=50),  # 0.72, 0.87
]
DURATION_BOUNDS = {1: (0.75, 1.0), 2: (0.30, 0.60), 3: (0.45, 0.95), 4: (0.72, 0.87)}
DURATION_STATUS = {  # (phases not confirmed sequential, confirmed sequential)
    1: (("undetermined", v.UNSEQUENCED), ("confirmed", v.SEQUENTIAL)),
    2: (("refuted", v.UPPER_BELOW), ("refuted", v.UPPER_BELOW)),
    3: (("undetermined", v.STRADDLE), ("undetermined", v.STRADDLE)),
    4: (("undetermined", v.UNSEQUENCED), ("confirmed", v.SEQUENTIAL)),
}


# --- C9.T3 -----------------------------------------------------------------------------------------


def test_c9_t3_timestamps_give_the_known_shares():
    report = v.verify(TIMESTAMPS, CLASSES)
    for row in report.rows:
        lower, upper, status, reason = TIMESTAMP_ANSWERS[row.iteration]
        assert (row.lower, row.upper) == pytest.approx((lower, upper), abs=1e-12), row.iteration
        assert (row.mode, row.status, row.reason) == ("timestamps", status, reason), row.iteration
    assert report.counts == {"confirmed": 2, "refuted": 2, "undetermined": 1}
    assert report.fraction_confirmed == pytest.approx(0.4)


def test_c9_t3_overlapping_check_phases_count_once():
    (row,) = v.verify([TIMESTAMPS[1]], CLASSES).rows
    assert row.seconds["check"] == pytest.approx(1200)  # wall-clock coverage of the three check phases
    assert row.seconds["check_only_low"] == pytest.approx(1200)
    assert sum(b - a for name, a, b in TIMESTAMPS[1].phases if CLASSES[name] == "check") == pytest.approx(2300)


def test_c9_t3_check_time_under_generation_or_training_does_not_count():
    (row,) = v.verify([TIMESTAMPS[2]], CLASSES).rows
    assert row.seconds["check"] == pytest.approx(1500) and row.seconds["check_only_low"] == pytest.approx(500)
    assert row.seconds["generation"] + row.seconds["training"] == pytest.approx(1300)


def test_c9_t3_a_phase_outside_the_iteration_is_clipped_and_reported():
    (row,) = v.verify([TIMESTAMPS[4]], CLASSES).rows
    assert row.seconds["outside_iteration"] == pytest.approx(50)
    assert row.seconds["generation"] == pytest.approx(50)


def test_c9_t3_durations_only_give_the_known_bounds():
    for sequential in (False, True):
        report = v.verify(DURATIONS, CLASSES, sequential_confirmed=sequential)
        for row in report.rows:
            assert (row.lower, row.upper) == pytest.approx(DURATION_BOUNDS[row.iteration], abs=1e-12)
            assert (row.mode, (row.status, row.reason)) == ("durations", DURATION_STATUS[row.iteration][sequential])
        assert report.fraction_confirmed == (0.5 if sequential else 0.0)


def test_c9_t3_the_threshold_is_inclusive():
    exactly = ts(1, 1000, ("gate", 0, 700), ("train", 700, 1000))
    assert v.verify([exactly], CLASSES).rows[0].status == "confirmed"
    below = ts(1, 1000, ("gate", 0, 699), ("train", 699, 1000))
    assert v.verify([below], CLASSES).rows[0].status == "refuted"


def test_c9_t3_a_mixed_summary_of_both_kinds():
    report = v.verify(TIMESTAMPS[:2] + [du(7, 2000, gate=300, scoring_wait=1500, asan_settle=200, compile_check=0,
                                           compile_repair=300)], CLASSES)
    assert [(r.iteration, r.mode, r.status) for r in report.rows] == \
        [(1, "timestamps", "confirmed"), (2, "timestamps", "refuted"), (7, "durations", "undetermined")]


# --- fail closed -----------------------------------------------------------------------------------


def test_an_unmapped_phase_stops_the_run():
    with pytest.raises(Verify70Error, match=r"no category in the mapping: \['decode'\]"):
        v.verify([ts(1, 10, ("decode", 0, 5), ("gate", 5, 10))], CLASSES)
    with pytest.raises(Verify70Error, match="'decode' has no category"):
        v.verify([du(1, 10, gate=5, decode=1, scoring_wait=1, asan_settle=1, compile_check=0, compile_repair=0)],
                 CLASSES)


def test_a_missing_check_duration_is_unknown_not_zero():
    partial = du(1, 2000, gate=300, scoring_wait=1500, compile_check=0, compile_repair=300)  # no asan_settle
    with pytest.raises(Verify70Error, match=r"no duration for check phase\(s\) \['asan_settle'\]"):
        v.verify([partial], CLASSES)
    (row,) = v.verify([partial], CLASSES, absent_means_zero=["asan_settle"]).rows
    assert (row.lower, row.upper) == pytest.approx((0.75, 1.0))


@pytest.mark.parametrize("it, message", [
    (IterationPhases(1, t_begin=T0, phases=(("gate", T0, T0 + 1),)), "timestamps need"),
    (IterationPhases(1, t_begin=T0, t_end=T0 + 100, wall_s=150.0, phases=(("gate", T0, T0 + 1),)), "contradicts"),
    (IterationPhases(1, t_begin=T0, t_end=T0, phases=(("gate", T0, T0),)), "not positive"),
    (IterationPhases(1, t_begin=T0, t_end=T0 + 10, phases=(("gate", T0 + 5, T0 + 4),)), "ends before"),
    (IterationPhases(1, durations={"gate": 1.0}), "no wall time"),
    (IterationPhases(1, wall_s=10.0, durations={"gate": 20.0}), "longer than the wall"),
    (IterationPhases(1, wall_s=10.0, durations={"gate": -1.0}), "duration -1"),
])
def test_bad_iterations_are_refused(it, message):
    with pytest.raises(Verify70Error, match=message):
        v.verify([it], CLASSES, absent_means_zero=["scoring_wait", "asan_settle", "compile_check", "compile_repair"])


def test_the_wall_may_differ_from_its_timestamps_by_rounding_only():
    it = IterationPhases(1, t_begin=T0, t_end=T0 + 2000, wall_s=2000.9, phases=(("gate", T0, T0 + 1600),))
    assert v.verify([it], CLASSES).rows[0].wall_s == 2000


def test_duplicate_iterations_and_bad_thresholds_are_refused():
    with pytest.raises(Verify70Error, match="twice"):
        v.verify([TIMESTAMPS[0], TIMESTAMPS[0]], CLASSES)
    with pytest.raises(Verify70Error, match="threshold"):
        v.verify(TIMESTAMPS, CLASSES, threshold=0)
    with pytest.raises(Verify70Error, match="no iterations"):
        v.verify([], CLASSES)


# --- another loop's summary file: the mapping and the reader -------------------------------------------

MAPPING = """\
schema: aorlta.verify70.mapping/1
records: iterations
fields:
  iteration: iter
  t_start: times.start
  t_end: times.end
  wall_s: null
  phases: phase_log
  durations: null
phase_entry: {name: phase, t_start: begin, t_end: finish}
time_unit: ms
classes:
  check: [ext_gate_x, ext_wait_x, ext_asan_x, ext_cc_x]
  mixed: [ext_repair_x]
  generation: [ext_gen_x]
  training: [ext_train_x]
  other: [ext_sync_x]
"""
RENAME = {"generate": "ext_gen_x", "gate": "ext_gate_x", "scoring_wait": "ext_wait_x", "asan_settle": "ext_asan_x",
          "compile_check": "ext_cc_x", "compile_repair": "ext_repair_x", "train": "ext_train_x",
          "weight_sync": "ext_sync_x"}


def _external_file(tmp_path, iterations, as_iso=False):
    def t(x):
        if as_iso:
            return dt.datetime.fromtimestamp(x, dt.timezone.utc).isoformat().replace("+00:00", "Z")
        return round(x * 1000)

    recs = []
    for it in iterations:
        recs.append({"iter": it.iteration, "times": {"start": t(it.t_begin), "end": t(it.t_end)},
                     "note": "PLANTED-CANARY-TEXT-9f2c", "reward_mean": -0.31,
                     "phase_log": [{"phase": RENAME[name], "begin": t(a), "finish": t(b)} for name, a, b in it.phases]})
    path = tmp_path / "summaries.json"
    path.write_text(json.dumps({"run": "x", "iterations": recs}))
    return path


def test_an_external_summary_file_is_read_through_the_mapping(tmp_path):
    (tmp_path / "map.yaml").write_text(MAPPING)
    out = v.run(_external_file(tmp_path, TIMESTAMPS), tmp_path / "map.yaml")
    assert [(r["iteration"], r["status"]) for r in out["rows"]] == \
        [(1, "confirmed"), (2, "refuted"), (3, "refuted"), (4, "undetermined"), (5, "confirmed")]
    assert [x for r in out["rows"] for x in (r["lower"], r["upper"])] == \
        pytest.approx([x for n in range(1, 6) for x in TIMESTAMP_ANSWERS[n][:2]], abs=1e-9)
    assert out["fraction_confirmed"] == 0.4 and out["counts"]["refuted"] == 2
    text = json.dumps(out)
    for planted in ["PLANTED-CANARY-TEXT-9f2c", "reward_mean", *RENAME.values()]:
        assert planted not in text  # only numbers and fixed codes come back


def test_iso_timestamps_read_the_same_as_epoch_milliseconds(tmp_path):
    (tmp_path / "map.yaml").write_text(MAPPING)
    its = TIMESTAMPS[:4]
    ms = v.run(_external_file(tmp_path, its), tmp_path / "map.yaml")
    (tmp_path / "map.yaml").write_text(MAPPING.replace("time_unit: ms", "time_unit: s"))
    iso = v.run(_external_file(tmp_path, its, as_iso=True), tmp_path / "map.yaml")
    bounds = lambda out: [x for r in out["rows"] for x in (r["lower"], r["upper"])]  # noqa: E731
    assert bounds(iso) == pytest.approx(bounds(ms), abs=1e-9)
    assert bounds(ms) == pytest.approx([0.75, 0.75, 0.6, 0.6, 0.25, 0.25, 0.55, 0.75], abs=1e-9)


def test_object_phases_and_jsonl(tmp_path):
    (tmp_path / "map.yaml").write_text("schema: aorlta.verify70.mapping/1\n"
                                       "classes: {check: [gate, wait], generation: [gen]}\n")
    lines = [{"iteration": 1, "t_start": 0, "t_end": 100, "phases": {"gen": [0, 20], "gate": [[20, 50], [60, 90]],
                                                                       "wait": [45, 70]}},
             {"iteration": 2, "wall_s": 100, "durations": {"gate": 75, "wait": 10}}]
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    out = v.run(path, tmp_path / "map.yaml")
    assert [(r["iteration"], r["mode"], r["lower"], r["upper"]) for r in out["rows"]] == \
        [(1, "timestamps", 0.7, 0.7), (2, "durations", 0.75, 0.85)]


@pytest.mark.parametrize("mapping, message", [
    ("schema: aorlta.verify70.mapping/1\nclasses: {check: [a], generation: [a]}\n", "both check and generation"),
    ("schema: aorlta.verify70.mapping/1\nclasses: {}\n", "classes no phase"),
    ("schema: aorlta.verify70.mapping/1\nclasses: {check: [a]}\nmystery: 1\n", "unknown key"),
    ("schema: aorlta.verify70.mapping/1\nclasses: {check: [a]}\nabsent_means_zero: [b]\n", "not check or mixed"),
    ("schema: aorlta.verify70.mapping/2\nclasses: {check: [a]}\n", "expected one of"),
    ("schema: aorlta.verify70.mapping/1\nclasses: {check: [a]}\nsequential_confirmed: yes\n", "expected a bool"),
])
def test_the_mapping_is_strict(tmp_path, mapping, message):
    (tmp_path / "map.yaml").write_text(mapping)
    with pytest.raises(Verify70Error, match=message):
        v.load_mapping(tmp_path / "map.yaml")


def test_mixed_zone_aware_and_naive_times_are_refused(tmp_path):
    (tmp_path / "map.yaml").write_text("schema: aorlta.verify70.mapping/1\nclasses: {check: [gate]}\n")
    path = tmp_path / "s.json"
    path.write_text(json.dumps([{"iteration": 1, "t_start": "2026-11-02T10:00:00Z", "t_end": "2026-11-02T10:10:00",
                                 "phases": [{"name": "gate", "t_start": "2026-11-02T10:00:00Z",
                                             "t_end": "2026-11-02T10:08:00Z"}]}]))
    with pytest.raises(Verify70Error, match="mixes aware and naive"):
        v.run(path, tmp_path / "map.yaml")


def test_the_cli_prints_the_table_and_writes_numbers(tmp_path, capsys):
    (tmp_path / "map.yaml").write_text(MAPPING)
    path = _external_file(tmp_path, TIMESTAMPS[:4])
    assert v.main([str(path), "--mapping", str(tmp_path / "map.yaml"), "--json", str(tmp_path / "o.json")]) == 0
    printed = capsys.readouterr().out
    assert "Confirmed (>= 0.7) in 1 of 4 iterations (25%)" in printed and "0.550 to 0.750" in printed
    assert json.loads((tmp_path / "o.json").read_text())["iterations"] == 4
    path.write_text("[]")
    assert v.main([str(path), "--mapping", str(tmp_path / "map.yaml")]) == 1


# --- AORLTA's spans, through the same arithmetic -----------------------------------------------------


def test_aorlta_spans_give_the_timelines_check_only_time():
    from measure_helpers import v1_iteration  # the C9.T1 iteration

    its, classes = v.phases_from_spans(v1_iteration())
    (row,) = v.verify(its, classes).rows
    assert row.mode == "timestamps" and row.reason == v.EXACT
    assert row.lower == pytest.approx(37.5 / 176) and row.status == "refuted"


def test_set_up_spans_of_iteration_zero_are_not_phases():
    from measure_helpers import sp, v1_iteration

    spans = [sp("boot.custom", -10, -5, it=0)] + v1_iteration()
    its, classes = v.phases_from_spans(spans)
    assert "boot.custom" not in classes and [it.iteration for it in its] == [1]
    landing_free = [s if s.name != "landing" else sp("landing", 175, 176, version=1) for s in v1_iteration()]
    its, classes = v.phases_from_spans(landing_free, verdicts={1: ["NOOP"]})
    assert v.verify(its, classes).rows[0].lower == pytest.approx(37.5 / 176)


def test_an_unclassified_span_stops_the_span_path():
    from aorlta.spans import Span

    spans = [Span("select", 1, T0, T0 + 1, {}), Span("odd.phase", 1, T0 + 1, T0 + 2, {}),
             Span("landing", 1, T0 + 2, T0 + 3, {"verdict": "LANDED"})]
    with pytest.raises(Verify70Error, match="'odd.phase' has no category"):
        v.phases_from_spans(spans)
