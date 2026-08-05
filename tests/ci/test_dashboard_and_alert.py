"""Unit tests for the dashboard generator + alert decision (pure logic)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / "ci" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


gen_dashboard = _load("gen_dashboard")
alert_issue = _load("alert_issue")
eval_lib = _load("eval_lib")


def _results(generated, entries, build=None, **summary):
    base = {"total": 0, "pass": 0, "fail": 0, "record": 0, "skip": 0}
    base.update(summary)
    info = {"amd_aorta_version": "0.2.1rc"}
    info.update(build or {})
    return {"generated_at": generated, "build": info, "summary": base, "entries": entries}


def test_sparkline_handles_short_and_normal_series():
    assert "n/a" in gen_dashboard._svg_sparkline([1.0])
    svg = gen_dashboard._svg_sparkline([1.0, 2.0, 1.5])
    assert svg.startswith("<svg") and "polyline" in svg


def test_dashboard_renders_status_and_rows():
    r1 = _results("2026-07-28T00:00:00Z",
                  [{"entry": "inference_offline", "cell": "baseline-local", "verdict": "pass",
                    "reasons": [], "metrics": {"mean_step_time_ms": 4.0}}],
                  total=1, **{"pass": 1})
    r2 = _results("2026-07-29T00:00:00Z",
                  [{"entry": "inference_offline", "cell": "baseline-local", "verdict": "fail",
                    "reasons": ["mean_step_time_ms 9 > max 5"], "metrics": {"mean_step_time_ms": 9.0}}],
                  total=1, fail=1)
    html = gen_dashboard.build_dashboard_html([r1, r2])
    assert "aorta nightly CI" in html
    assert "failing" in html  # latest run failed
    # Cells are grouped under their workload, so the two names render separately.
    assert "inference_offline" in html
    assert "baseline-local" in html
    assert "0.2.1rc" in html


def test_dashboard_empty_history():
    html = gen_dashboard.build_dashboard_html([])
    assert "no results yet" in html


def test_dashboard_links_to_sanitizer_nightly():
    html = gen_dashboard.build_dashboard_html([])
    assert 'href="sanitizers/"' in html
    assert "sanitizer nightly" in html


def test_dashboard_renders_summary_metric_series():
    r = _results("2026-07-29T00:00:00Z",
                 [{"entry": "inference_offline", "cell": "baseline-local", "verdict": "pass",
                   "reasons": [], "metrics": {"mean_step_time_ms": 4.0,
                                              "summary": {"decode_latency_ms": 12.5}}}],
                 total=1, **{"pass": 1})
    html = gen_dashboard.build_dashboard_html([r])
    assert "Workloads" in html
    assert "decode_latency_ms" in html
    assert "ms" in html  # unit rendered


def test_dashboard_record_only_history_keeps_trend_fallback_visible():
    # A record-only build has no graded cells, so pass-rate is None for every
    # build and the sparkline falls back to "n/a". That fallback must stay
    # legible: styling the trend card with font-size:0 would blank it out.
    r = _results("2026-08-03T00:00:00Z",
                 [{"entry": "gpu_smoke", "cell": "baseline-local", "verdict": "record",
                   "reasons": ["no baseline (record-only)"],
                   "metrics": {"mean_step_time_ms": 254.0}}],
                 total=1, record=1)
    html = gen_dashboard.build_dashboard_html([r])

    trend_card = html.split('<div class="card trend">', 1)[1].split("</div></div>", 1)[0]
    assert "n/a" in trend_card
    assert "<svg" not in trend_card

    assert ".card.trend svg { display:block; }" in html
    assert "font-size:0" not in html


def test_alert_render_lists_failures():
    results = _results("2026-07-29T00:00:00Z",
                       [{"entry": "race", "cell": "smoke", "verdict": "fail",
                         "reasons": ["expected passing cell but it did not pass"], "error": None},
                        {"entry": "gpu_smoke", "cell": "baseline-local", "verdict": "pass",
                         "reasons": [], "error": None}],
                       total=2, **{"pass": 1, "fail": 1})
    assert len(alert_issue.failing_entries(results)) == 1
    title, body = alert_issue.render_issue(results, "http://run/1")
    assert "1 failing" in title
    assert "race::smoke" in body
    assert "http://run/1" in body


def test_alert_no_failures():
    results = _results("2026-07-29T00:00:00Z", [], total=0)
    assert alert_issue.failing_entries(results) == []


def test_alert_md_cell_escapes_pipes_and_newlines():
    assert alert_issue._md_cell("a | b\nc") == "a \\| b c"


def test_alert_render_sanitizes_table_cells():
    results = _results("2026-07-29T00:00:00Z",
                       [{"entry": "w", "cell": "c", "verdict": "fail",
                         "reasons": ["boom | pipe\nnewline"], "error": None}],
                       total=1, fail=1)
    _, body = alert_issue.render_issue(results, None)
    # No raw pipe/newline inside the reasons cell that would break the table.
    reason_line = [ln for ln in body.splitlines() if ln.startswith("| `w::c`")][0]
    assert "boom \\| pipe newline" in reason_line


def test_find_open_issue_ignores_pull_requests(monkeypatch):
    # The issues endpoint returns PRs too; a PR carrying the label must be ignored.
    monkeypatch.setattr(
        alert_issue, "_req",
        lambda *a, **k: [{"number": 10, "pull_request": {"url": "x"}}, {"number": 11}],
    )
    found = alert_issue._find_open_issue("ROCm/aorta", "tok")
    assert found is not None and found["number"] == 11


def test_dashboard_escapes_untrusted_reason():
    r = _results("2026-07-29T00:00:00Z",
                 [{"entry": "w", "cell": "c", "verdict": "fail",
                   "reasons": ["<script>alert(1)</script>"], "metrics": {"mean_step_time_ms": 1.0}}],
                 total=1, fail=1)
    html = gen_dashboard.build_dashboard_html([r])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_dashboard_always_explains_a_failure():
    # Uninformative columns get collapsed, but a failing cell's reason must
    # never be one of them -- a bare "fail" badge with no explanation anywhere
    # is worse than a redundant column.
    entries = [
        {"entry": "w", "cell": f"c{i}", "verdict": "pass", "reasons": [],
         "metrics": {"mean_step_time_ms": 1.0}}
        for i in range(4)
    ]
    entries.append({"entry": "w", "cell": "bad", "verdict": "fail",
                    "reasons": ["mean_step_time_ms 9 > max 5"],
                    "metrics": {"mean_step_time_ms": 9.0}})
    html = gen_dashboard.build_dashboard_html(
        [_results("2026-07-30T00:00:00Z", entries, total=5, fail=1, **{"pass": 4})])
    assert "mean_step_time_ms 9 &gt; max 5" in html


def test_dashboard_collapses_a_note_shared_by_every_cell():
    # When every cell says the same thing, say it once instead of repeating it
    # down a column -- but it still has to appear somewhere.
    entries = [
        {"entry": "w", "cell": f"c{i}", "verdict": "record",
         "reasons": ["no baseline (record-only)"],
         "metrics": {"mean_step_time_ms": 1.0}}
        for i in range(3)
    ]
    html = gen_dashboard.build_dashboard_html(
        [_results("2026-07-30T00:00:00Z", entries, total=3, record=3)])
    assert "no baseline (record-only)" in html
    assert "<th scope='col'>notes</th>" not in html


def test_dashboard_record_only_run_is_not_reported_as_passing():
    # Nothing was graded, so "passing" would overstate the result.
    r = _results("2026-07-30T00:00:00Z",
                 [{"entry": "w", "cell": "c", "verdict": "record",
                   "reasons": ["no baseline (record-only)"],
                   "metrics": {"mean_step_time_ms": 1.0}}],
                 total=1, record=1)
    html = gen_dashboard.build_dashboard_html([r])
    assert "recording" in html
    assert ">passing<" not in html


def test_dashboard_zero_work_run_reports_the_failure_the_pipeline_records():
    # This is what a real all-skip nightly looks like: nightly_eval appends a
    # synthetic _nightly_eval "fail" entry when every cell skips, so the build
    # carries fail>=1 and the honest headline is "failing", not "skipping".
    # Asserting the shape without that entry would test a document this
    # pipeline never emits.
    r = _results("2026-07-30T00:00:00Z",
                 [{"entry": "w", "cell": "c", "verdict": "skip",
                   "reasons": ["needs 8 GPUs, runner exposes 2"], "metrics": {}},
                  {"entry": "_nightly_eval", "cell": None, "verdict": "fail",
                   "reasons": ["no matrix entry ran (empty matrix or all skipped -- check GPUs)"],
                   "metrics": {}}],
                 total=2, fail=1, skip=1)
    html = gen_dashboard.build_dashboard_html([r])
    assert "failing" in html
    assert ">passing<" not in html
    assert "skipping" not in html
    # The reason a zero-work build failed must be on the page, not just a badge.
    assert "no matrix entry ran" in html
    assert "No baselines are blessed yet" not in html


def test_dashboard_all_skip_summary_is_labelled_skipping_not_passing():
    # Defensive branch: unreachable for nightly_eval's own output (the test
    # above covers that), but _latest_status must still classify an all-skip
    # summary from any other producer, and every label but "skipping" would
    # overclaim. Kept so a future relaxation of that invariant cannot silently
    # resurrect the "skip-only reads as passing" bug.
    r = _results("2026-07-30T00:00:00Z",
                 [{"entry": "w", "cell": "c", "verdict": "skip",
                   "reasons": ["needs 8 GPUs, runner exposes 2"], "metrics": {}}],
                 total=1, skip=1)
    html = gen_dashboard.build_dashboard_html([r])
    assert "skipping" in html
    assert ">passing<" not in html
    assert "No baselines are blessed yet" not in html
    assert "all 1 results were" in html


def test_dashboard_partial_record_does_not_claim_every_result_recorded():
    # record + skip must not be described as "recorded all N results".
    entries = [{"entry": "w", "cell": f"r{i}", "verdict": "record",
                "reasons": ["no baseline (record-only)"],
                "metrics": {"mean_step_time_ms": 1.0}} for i in range(3)]
    entries.append({"entry": "w", "cell": "s0", "verdict": "skip",
                    "reasons": ["needs 8 GPUs, runner exposes 2"], "metrics": {}})
    html = gen_dashboard.build_dashboard_html(
        [_results("2026-07-30T00:00:00Z", entries, total=4, record=3, skip=1)])
    assert "recording" in html
    assert "3 of 4 results (1 skipped)" in html
    assert "all 4 results" not in html


def test_dashboard_metric_trends_survive_a_missing_step_time():
    # A cell can report metrics.summary with no mean_step_time_ms (harvest
    # leaves it None). Its metric history is real, so the per-metric sparkline
    # must still draw and the "needs two runs" notice must not claim otherwise,
    # even though there is no step-time history to chart.
    def doc(when, latency):
        return _results(when,
                        [{"entry": "w", "cell": "c", "verdict": "record", "reasons": [],
                          "metrics": {"mean_step_time_ms": None,
                                      "summary": {"decode_latency_ms": latency}}}],
                        total=1, record=1)

    html = gen_dashboard.build_dashboard_html(
        [doc("2026-07-30T00:00:00Z", 12.5), doc("2026-07-31T00:00:00Z", 13.5)])
    assert "<svg" in html
    assert "Trend charts need at least two nightly runs" not in html
    # The step-time column still has nothing to chart, so it stays collapsed.
    assert "trend (step ms)" not in html


def test_dashboard_groups_cells_under_their_workload():
    entries = [
        {"entry": "llm_determinism", "cell": c, "verdict": "record",
         "reasons": [], "metrics": {"mean_step_time_ms": 10.0}}
        for c in ("bf16-12L", "tf32-24L")
    ]
    html = gen_dashboard.build_dashboard_html(
        [_results("2026-07-30T00:00:00Z", entries, total=2, record=2)])
    # One group header for the workload, and each cell listed beneath it.
    assert html.count("llm_determinism") == 1
    assert "bf16-12L" in html and "tf32-24L" in html


def test_dashboard_omits_trend_column_without_history():
    # A single build cannot draw a trend; the column is dropped rather than
    # rendered as a full column of "n/a".
    r = _results("2026-07-30T00:00:00Z",
                 [{"entry": "w", "cell": "c", "verdict": "record", "reasons": [],
                   "metrics": {"mean_step_time_ms": 1.0,
                               "summary": {"decode_latency_ms": 12.5}}}],
                 total=1, record=1)
    one = gen_dashboard.build_dashboard_html([r])
    assert "trend (step ms)" not in one

    r2 = _results("2026-07-31T00:00:00Z",
                  [{"entry": "w", "cell": "c", "verdict": "record", "reasons": [],
                    "metrics": {"mean_step_time_ms": 2.0,
                                "summary": {"decode_latency_ms": 13.5}}}],
                  total=1, record=1)
    two = gen_dashboard.build_dashboard_html([r, r2])
    assert "trend (step ms)" in two


def test_dashboard_does_not_treat_a_boolean_as_a_measurement():
    # bool is a subclass of int, so a stray `true` in the results JSON would
    # otherwise format as "1.0 ms", scale a bar, and count towards a trend.
    assert gen_dashboard._fmt_ms(True) == "—"
    assert gen_dashboard._fmt_num(True) == "—"
    assert gen_dashboard._bar(True, 10.0) == ""
    assert not gen_dashboard._has_trend([[True, False]])
    assert "n/a" in gen_dashboard._svg_sparkline([True, True, True])

    def doc(when, step):
        return _results(when,
                        [{"entry": "w", "cell": "c", "verdict": "record", "reasons": [],
                          "metrics": {"mean_step_time_ms": step,
                                      "summary": {"decode_latency_ms": step}}}],
                        total=1, record=1)

    html = gen_dashboard.build_dashboard_html(
        [doc("2026-07-30T00:00:00Z", True), doc("2026-07-31T00:00:00Z", False)])
    assert "1.0 ms" not in html
    assert "trend (step ms)" not in html


def test_dashboard_rejects_values_it_cannot_render():
    # json.loads accepts NaN/Infinity, and a metric mean can genuinely come out
    # NaN. Unfiltered, NaN plots as "nan" coordinates and -- worse -- draws a
    # full-width bar, since min(100.0, nan) returns 100.0. An int past the float
    # range is rejected too: it raises OverflowError in every formatter here,
    # including math.isfinite itself.
    huge = int("9" * 400)
    for bad in (float("nan"), float("inf"), float("-inf"), huge, True, None, "4"):
        assert not gen_dashboard._isnum(bad), bad
        assert gen_dashboard._fmt_num(bad) == "—", bad
        assert gen_dashboard._fmt_ms(bad) == "—", bad
        assert gen_dashboard._bar(bad, 10.0) == "", bad
    assert gen_dashboard._isnum(0) and gen_dashboard._isnum(-1.5)

    r = _results("2026-07-30T00:00:00Z",
                 [{"entry": "w", "cell": "c", "verdict": "record", "reasons": [],
                   "metrics": {"mean_step_time_ms": float("nan"),
                               "summary": {"decode_latency_ms": huge}}}],
                 total=1, record=1)
    html = gen_dashboard.build_dashboard_html([r])  # must not raise
    assert "nan" not in html.lower()
    assert 'class="bar"' not in html

    # Summary counts drive branching, division and the pass-rate card, so a
    # malformed one must not crash generation ("4" + 0 raises TypeError) nor
    # surface as "nan%". They render as "—" rather than a 0 we cannot vouch for.
    entry = [{"entry": "w", "cell": "c", "verdict": "pass", "reasons": [],
              "metrics": {}}]
    for count in (float("nan"), "4", huge, True, None):
        doc = _results("2026-07-30T00:00:00Z", entry, total=1, **{"pass": count})
        out = gen_dashboard.build_dashboard_html([doc])  # must not raise
        assert "nan" not in out.lower(), count
        assert "—" in out, count


def test_dashboard_does_not_count_records_as_configured_cells():
    # summary.total is len(entries), not the size of the matrix: a workload
    # skipped for want of GPUs emits ONE record with cell=None however many
    # cells its recipe defines. Counting those as "cells" overstates coverage
    # exactly when coverage is worst, so the page says "results" instead.
    entries = [
        {"entry": "llm_determinism", "cell": c, "verdict": "record",
         "reasons": ["no baseline (record-only)"],
         "metrics": {"mean_step_time_ms": 10.0}} for c in ("bf16-12L", "tf32-24L")
    ]
    entries.append({"entry": "training_ddp_8gpu", "cell": None, "verdict": "skip",
                    "reasons": ["needs 8 GPU(s), have 2"], "metrics": {}})
    # Summarised by the harness's own summarizer, so the document under test
    # cannot drift into a shape nightly_eval would never produce.
    doc = _results("2026-07-30T00:00:00Z", entries, **eval_lib.summarize(entries))
    assert doc["summary"]["total"] == 3  # three records, though the matrix is wider
    html = gen_dashboard.build_dashboard_html([doc])

    assert "2 of 3 results (1 skipped)" in html
    assert ">results<" in html and ">cells<" not in html  # count card label
    for wrong in ("2 of 3 cells", "3 cells", "2 cells ·", "1 cell ·"):
        assert wrong not in html, wrong
    assert "2 results ·" in html and "1 result ·" in html  # group tallies
    # A skipped workload reports duration_sec 0.0; it never ran, so claiming a
    # "workload run 0s" would be a measurement of something that did not happen.
    assert "workload run 0s" not in html
    # The skipped workload's single record is not presented as a cell name.
    assert ">None<" not in html
    assert "whole workload" in html


def test_dashboard_states_duration_once_per_workload_not_per_cell():
    # nightly_eval measures duration_sec once around the whole recipe and copies
    # it onto every record, so repeating it inside each cell's details reads as
    # per-cell time. It belongs to the workload and is stated there once.
    entries = [
        {"entry": "llm_determinism", "cell": c, "verdict": "record", "reasons": [],
         "duration_sec": 105.96, "trials": 2,
         "metrics": {"mean_step_time_ms": 10.0, "summary": {"latency_ms": 1.5}}}
        for c in ("bf16-12L", "tf32-24L", "moe4-bf16-12L", "baseline-bf16-24L")
    ]
    html = gen_dashboard.build_dashboard_html(
        [_results("2026-07-30T00:00:00Z", entries, **eval_lib.summarize(entries))])

    assert html.count("workload run 106s") == 1
    assert "ran in 106s" not in html  # never restated as if it were per cell
    assert html.count("2 trials") == 4  # genuinely per-record detail stays


def test_dashboard_omits_the_unit_when_the_value_is_unknown():
    # "— ms" reads as a measurement of nothing; a missing value gets no unit.
    r = _results("2026-07-30T00:00:00Z",
                 [{"entry": "w", "cell": "c", "verdict": "record", "reasons": [],
                   "metrics": {"mean_step_time_ms": 1.0,
                               "summary": {"decode_latency_ms": None,
                                           "latency_ms": 12.5}}}],
                 total=1, record=1)
    html = gen_dashboard.build_dashboard_html([r])
    assert "— ms" not in html
    assert "12.5 ms" in html  # a real value keeps its unit


def _run(day, entries, build=None, **summary):
    return _results(f"2026-08-{day:02d}T00:00:00Z", entries, build=build, **summary)


def _cell(entry, cell, verdict="record", step=None, summary=None, reasons=None):
    metrics = {}
    if step is not None:
        metrics["mean_step_time_ms"] = step
    if summary is not None:
        metrics["summary"] = summary
    return {"entry": entry, "cell": cell, "verdict": verdict,
            "reasons": reasons or [], "metrics": metrics}


# --- run history grid -------------------------------------------------------


def test_history_grid_needs_two_runs_before_it_says_anything():
    # One run is not a history: a grid of a single row invites reading a trend
    # into a single sample.
    one = [_run(3, [_cell("w", "c")], total=1, record=1)]
    assert gen_dashboard.build_history_grid(one) == ""
    assert gen_dashboard.build_history_grid([]) == ""
    assert "<h2>Run history</h2>" not in gen_dashboard.build_dashboard_html(one)


def test_history_grid_puts_runs_on_rows_and_workloads_on_columns():
    runs = [
        _run(3, [_cell("gpu_smoke", "c"), _cell("race", "smoke")], total=2, record=2),
        _run(4, [_cell("gpu_smoke", "c"), _cell("race", "smoke")], total=2, record=2),
    ]
    grid = gen_dashboard.build_history_grid(runs)
    assert "gpu_smoke" in grid and "race" in grid
    # Newest first: whichever date appears earlier in the markup is the top row.
    assert grid.index("2026-08-04") < grid.index("2026-08-03")


def test_history_grid_cell_leads_with_the_worst_verdict_in_the_group():
    # Three passing cells and one failure is a failing workload that night; the
    # cell has to say so rather than average the group into looking healthy.
    mixed = [_cell("w", f"c{i}", verdict="pass") for i in range(3)]
    mixed.append(_cell("w", "c3", verdict="fail"))
    runs = [
        _run(3, [_cell("w", f"c{i}", verdict="pass") for i in range(4)], total=4, **{"pass": 4}),
        _run(4, mixed, total=4, fail=1, **{"pass": 3}),
    ]
    grid = gen_dashboard.build_history_grid(runs)
    assert "1/4" in grid                                    # one of four is the worst
    assert gen_dashboard._VERDICT_COLOR["fail"] in grid
    assert "3 pass · 1 fail" in grid or "1 fail · 3 pass" in grid  # hover breakdown
    # A uniform group needs no ratio -- "4/4" would imply something was wrong.
    assert "4/4" not in grid


def test_history_grid_never_renders_an_unrecognised_verdict_as_healthy():
    # The results JSON is not ours to constrain, so a verdict this generator has
    # no colour for can appear. Counting it towards the group while never being
    # able to select it would render pass + error as a green tick over "1/2" --
    # an unrecognised result reported as healthy.
    runs = [
        _run(3, [_cell("w", "c0", verdict="pass"), _cell("w", "c1", verdict="pass")],
             total=2, **{"pass": 2}),
        _run(4, [_cell("w", "c0", verdict="pass"), _cell("w", "c1", verdict="error")],
             total=2, **{"pass": 1}),
    ]
    grid = gen_dashboard.build_history_grid(runs)
    cell = f"background:{gen_dashboard._UNKNOWN_COLOR}' title='w — 1 pass · 1 error'"
    assert cell in grid          # the mixed group takes the unknown colour, not green
    assert "? 1/2" in grid       # and reads as one unrecognised of two
    assert "1 pass · 1 error" in grid  # the unknown is still counted in the hover


def test_history_grid_ranks_a_known_failure_above_an_unrecognised_verdict():
    # Unknown outranks the benign verdicts but must not displace a real failure:
    # "something failed" is more actionable than "something is unfamiliar".
    runs = [
        _run(3, [_cell("w", "c0", verdict="pass")], total=1, **{"pass": 1}),
        _run(4, [_cell("w", "c0", verdict="fail"), _cell("w", "c1", verdict="error")],
             total=2, fail=1),
    ]
    grid = gen_dashboard.build_history_grid(runs)
    assert gen_dashboard._VERDICT_COLOR["fail"] in grid
    assert gen_dashboard._UNKNOWN_COLOR not in grid


def test_headline_does_not_claim_passing_when_the_summary_hides_an_unknown():
    # total=2 with a single known verdict means one entry carried something this
    # generator cannot classify. Reporting "passing" off the one it recognises
    # would be the same defect as a green grid cell, on the more prominent
    # surface.
    runs = [_run(3, [_cell("w", "c0", verdict="pass")], total=2, **{"pass": 1})]
    assert gen_dashboard._latest_status(runs) == (
        "unrecognised", gen_dashboard._UNKNOWN_COLOR)
    # A summary that adds up is untouched.
    ok = [_run(3, [_cell("w", "c0", verdict="pass")], total=1, **{"pass": 1})]
    assert gen_dashboard._latest_status(ok)[0] == "passing"


def test_worst_verdict_precedence_is_fail_then_unknown_then_the_benign_ones():
    assert gen_dashboard._worst_verdict({"fail": 1, "error": 1, "pass": 5}) == "fail"
    assert gen_dashboard._worst_verdict({"error": 1, "record": 2, "pass": 5}) == "error"
    assert gen_dashboard._worst_verdict({"record": 1, "pass": 5}) == "record"
    assert gen_dashboard._worst_verdict({"pass": 5}) == "pass"
    # A zero count is not a present verdict; it must not win by being unknown.
    assert gen_dashboard._worst_verdict({"error": 0, "pass": 5}) == "pass"
    assert gen_dashboard._worst_verdict({}) == ""


def test_history_grid_marks_a_workload_that_was_absent_that_night():
    runs = [
        _run(3, [_cell("w1", "c")], total=1, record=1),
        _run(4, [_cell("w1", "c"), _cell("w2", "c")], total=2, record=2),
    ]
    grid = gen_dashboard.build_history_grid(runs)
    assert "not in this run" in grid  # w2 has no cell on the older row


def test_history_grid_absence_marker_is_named_for_assistive_tech():
    # Unnamed, a screen reader reaches the bare glyph and announces "middle
    # dot", which does not convey that the workload was absent that night.
    runs = [
        _run(3, [_cell("w1", "c")], total=1, record=1),
        _run(4, [_cell("w1", "c"), _cell("w2", "c")], total=2, record=2),
    ]
    grid = gen_dashboard.build_history_grid(runs)
    assert "role='img' title='not in this run' aria-label='not in this run'" in grid


def test_history_grid_does_not_flag_aortas_own_nightly_version_as_a_bump():
    # amd_aorta_version is date-stamped, so it changes every single night. Were
    # it treated as a toolchain change, every row would carry the marker and the
    # one signal that should mean something would mean nothing.
    runs = [
        _run(3, [_cell("w", "c")], build={"amd_aorta_version": "0.2.2rc20260803",
                                          "rocm": "7.2.0", "torch": "2.9.1", "hip": "7.2"},
             total=1, record=1),
        _run(4, [_cell("w", "c")], build={"amd_aorta_version": "0.2.2rc20260804",
                                          "rocm": "7.2.0", "torch": "2.9.1", "hip": "7.2"},
             total=1, record=1),
    ]
    assert "class='bump'" not in gen_dashboard.build_history_grid(runs)


def test_history_grid_flags_the_run_where_the_stack_underneath_moved():
    runs = [
        _run(3, [_cell("w", "c")], build={"rocm": "7.2.0"}, total=1, record=1),
        _run(4, [_cell("w", "c")], build={"rocm": "7.3.0"}, total=1, record=1),
    ]
    grid = gen_dashboard.build_history_grid(runs)
    assert grid.count("class='bump'") == 1  # only the run that moved


def test_history_grid_window_bounds_the_rendered_rows():
    runs = [_run(d, [_cell("w", "c")], total=1, record=1) for d in range(1, 21)]
    grid = gen_dashboard.build_history_grid(runs, max_runs=5)
    assert grid.count("<tr><th scope='row'") == 5  # body rows, not the header
    assert "2026-08-20" in grid and "2026-08-15" not in grid  # newest five


# --- what changed since the previous run ------------------------------------


def test_change_summary_withheld_until_there_is_something_to_compare():
    assert gen_dashboard.build_change_summary([]) == ""
    assert gen_dashboard.build_change_summary(
        [_run(3, [_cell("w", "c")], total=1, record=1)]) == ""


def test_change_summary_reports_a_verdict_that_flipped_regressions_first():
    runs = [
        _run(3, [_cell("a", "c", verdict="pass"), _cell("b", "c", verdict="fail")],
             total=2, **{"pass": 1, "fail": 1}),
        _run(4, [_cell("a", "c", verdict="fail", reasons=["step time 9 > max 5"]),
                 _cell("b", "c", verdict="pass")],
             total=2, **{"pass": 1, "fail": 1}),
    ]
    html = gen_dashboard.build_change_summary(runs)
    assert "a::c" in html and "b::c" in html
    assert "step time 9 &gt; max 5" in html  # the reason travels with the flip
    # The new failure is what someone needs to act on, so it leads.
    assert html.index("a::c") < html.index("b::c")


def test_change_summary_names_the_toolchain_that_moved():
    runs = [
        _run(3, [_cell("w", "c")], build={"rocm": "7.2.0", "torch": "2.9.1"},
             total=1, record=1),
        _run(4, [_cell("w", "c")], build={"rocm": "7.3.0", "torch": "2.9.1"},
             total=1, record=1),
    ]
    html = gen_dashboard.build_change_summary(runs)
    assert "ROCm" in html and "7.2.0" in html and "7.3.0" in html
    assert "PyTorch" not in html  # unchanged fields stay quiet


def test_change_summary_lists_metrics_that_moved_past_the_threshold():
    runs = [
        _run(3, [_cell("w", "c", step=100.0, summary={"latency_ms": 10.0})],
             total=1, record=1),
        _run(4, [_cell("w", "c", step=150.0, summary={"latency_ms": 10.2})],
             total=1, record=1),
    ]
    html = gen_dashboard.build_change_summary(runs)
    assert "mean_step_time_ms" in html and "+50.0%" in html
    assert "latency_ms" not in html  # 2% is inside the noise floor


def test_change_summary_survives_a_previous_value_of_zero():
    # A percentage against zero is undefined; the run must still render.
    runs = [
        _run(3, [_cell("w", "c", step=0.0)], total=1, record=1),
        _run(4, [_cell("w", "c", step=5.0)], total=1, record=1),
    ]
    html = gen_dashboard.build_change_summary(runs)
    assert "mean_step_time_ms" not in html
    assert "Nothing moved" in html


def test_change_summary_ignores_values_it_cannot_compare():
    runs = [
        _run(3, [_cell("w", "c", summary={"m": float("nan"), "b": True, "s": "4"})],
             total=1, record=1),
        _run(4, [_cell("w", "c", summary={"m": 99.0, "b": False, "s": "400"})],
             total=1, record=1),
    ]
    html = gen_dashboard.build_change_summary(runs)
    assert "Nothing moved" in html


def test_change_summary_reports_cells_the_matrix_gained_and_lost():
    runs = [
        _run(3, [_cell("w", "old")], total=1, record=1),
        _run(4, [_cell("w", "new")], total=1, record=1),
    ]
    html = gen_dashboard.build_change_summary(runs)
    assert "new result" in html and "w::new" in html
    assert "no longer reported" in html and "w::old" in html


def test_change_summary_says_so_plainly_when_a_night_was_uneventful():
    same = [_cell("w", "c", step=10.0, summary={"latency_ms": 1.0})]
    runs = [_run(3, same, total=1, record=1), _run(4, same, total=1, record=1)]
    html = gen_dashboard.build_change_summary(runs)
    assert "Nothing moved" in html
    assert "<li>" not in html


def test_change_summary_names_a_cell_less_record_as_the_whole_workload():
    runs = [
        _run(3, [{"entry": "w", "cell": None, "verdict": "skip", "reasons": [],
                  "metrics": {}}], total=1, skip=1),
        _run(4, [{"entry": "w", "cell": None, "verdict": "fail", "reasons": [],
                  "metrics": {}}], total=1, fail=1),
    ]
    html = gen_dashboard.build_change_summary(runs)
    assert "w (whole workload)" in html
    assert "w::None" not in html


def test_history_grid_states_the_verdict_without_relying_on_colour():
    # A bare count renders identically for a passing and a failing group, so the
    # cell would be unreadable to anyone who cannot separate red from green, or
    # who is on a touch screen and cannot hover for the title.
    runs = [
        _run(3, [_cell("w", "c", verdict="pass")], total=1, **{"pass": 1}),
        _run(4, [_cell("w", "c", verdict="fail")], total=1, fail=1),
    ]
    grid = gen_dashboard.build_history_grid(runs)
    assert "✗ 1" in grid and "✓ 1" in grid
    # The breakdown reaches assistive tech, not only a hover tooltip. The role
    # is part of that: a generic span has no accessible name, so the label is
    # not guaranteed to be announced without it.
    assert "aria-label='w — 1 fail'" in grid
    assert grid.count("<span class='dot' role='img'") == 2


def test_history_grid_compares_its_oldest_row_against_the_run_before_it():
    # The window truncates the rendered rows, not the history. Comparing the
    # oldest displayed row against nothing would drop a toolchain change from
    # that row every time history grows past the window.
    runs = [
        _run(1, [_cell("w", "c")], build={"rocm": "7.2.0"}, total=1, record=1),
        _run(2, [_cell("w", "c")], build={"rocm": "7.3.0"}, total=1, record=1),
        _run(3, [_cell("w", "c")], build={"rocm": "7.3.0"}, total=1, record=1),
    ]
    # Day 2 is the oldest rendered row and it is where ROCm moved.
    grid = gen_dashboard.build_history_grid(runs, max_runs=2)
    assert grid.count("class='bump'") == 1
    assert "2026-08-01" not in grid  # genuinely outside the window


def test_history_grid_first_ever_run_has_nothing_to_be_compared_against():
    runs = [
        _run(3, [_cell("w", "c")], build={"rocm": "7.2.0"}, total=1, record=1),
        _run(4, [_cell("w", "c")], build={"rocm": "7.3.0"}, total=1, record=1),
    ]
    # The bump belongs to day 4 only; day 3 has no predecessor to differ from.
    assert gen_dashboard.build_history_grid(runs).count("class='bump'") == 1


def test_change_summary_compares_wall_clock_as_well_as_step_time():
    # nightly_eval records both, and they can disagree: a steady step time with
    # a climbing wall clock is what a slower setup or a stalling teardown looks
    # like, and it would be invisible if only step time were compared.
    runs = [
        _run(3, [{"entry": "w", "cell": "c", "verdict": "record", "reasons": [],
                  "metrics": {"mean_step_time_ms": 10.0, "mean_wall_clock_sec": 5.0}}],
             total=1, record=1),
        _run(4, [{"entry": "w", "cell": "c", "verdict": "record", "reasons": [],
                  "metrics": {"mean_step_time_ms": 10.0, "mean_wall_clock_sec": 9.0}}],
             total=1, record=1),
    ]
    html = gen_dashboard.build_change_summary(runs)
    assert "mean_wall_clock_sec" in html and "+80.0%" in html
    assert "mean_step_time_ms" not in html  # genuinely unchanged


def test_change_summary_threshold_is_the_more_than_it_claims_to_be():
    # The docs, the constant's comment and the steady-state line all promise
    # "more than" 10%, so exactly 10% must not be reported.
    def moved(before, after):
        runs = [_run(3, [_cell("w", "c", step=before)], total=1, record=1),
                _run(4, [_cell("w", "c", step=after)], total=1, record=1)]
        return "mean_step_time_ms" in gen_dashboard.build_change_summary(runs)

    assert not moved(100.0, 110.0)   # exactly +10%
    assert not moved(100.0, 90.0)    # exactly -10%
    assert moved(100.0, 110.1)       # just past it


def test_change_summary_does_not_round_a_move_back_onto_the_threshold():
    # Rounding to whole percent printed a reported 10.4% move as "+10%", inside
    # a section that promises "moved more than 10%" -- the row read as a
    # contradiction of the heading directly above it.
    runs = [
        _run(3, [_cell("w", "c", step=100.0)], total=1, record=1),
        _run(4, [_cell("w", "c", step=110.4)], total=1, record=1),
    ]
    html = gen_dashboard.build_change_summary(runs)
    assert "+10.4%" in html


def test_change_summary_carries_the_unit_for_step_time():
    # mean_step_time_ms is one of the compared harness metrics but had no entry
    # in _METRIC_UNITS, so its rows rendered bare numbers while every other
    # timing metric kept its unit.
    runs = [
        _run(3, [_cell("w", "c", step=100.0)], total=1, record=1),
        _run(4, [_cell("w", "c", step=150.0)], total=1, record=1),
    ]
    html = gen_dashboard.build_change_summary(runs)
    assert "100 → 150 ms" in html


def test_dashboard_withholds_the_js_controls_when_there_is_nothing_to_expand():
    # A run whose results carried no metrics renders no details rows, so both
    # buttons would provably do nothing.
    runs = [_run(3, [_cell("w", "c", step=1.0)], total=1, record=1),
            _run(4, [_cell("w", "c", step=1.0)], total=1, record=1)]
    html = gen_dashboard.build_dashboard_html(runs)
    assert "<details>" not in html
    assert 'document.querySelectorAll("tr.mrow details").length' in html


def test_dashboard_hides_the_js_only_controls_until_the_script_runs():
    # The page is complete without JavaScript, so a control that does nothing
    # without it must not be visible in the served markup.
    runs = [_run(3, [_cell("w", "c", summary={"m": 1.0})], total=1, record=1),
            _run(4, [_cell("w", "c", summary={"m": 1.0})], total=1, record=1)]
    html = gen_dashboard.build_dashboard_html(runs)
    assert '<div class="toolbar" id="toolbar" hidden>' in html
    assert "bar.hidden = false" in html
    assert "Expand all" in html
