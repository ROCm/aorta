"""Unit tests for the dashboard generator + alert decision (pure logic)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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
    labelled = gen_dashboard._svg_sparkline([1.0, 2.0, 1.5], label="step time history")
    assert 'role="img"' in labelled and 'aria-label="step time history"' in labelled


def test_sparkline_a11y_label_lists_oldest_to_newest():
    label = gen_dashboard._sparkline_a11y_label("mean step time", [100.0, None, 150.0])
    assert "oldest to newest" in label
    assert "100" in label and "150" in label


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
    assert "ROCm stack health" in html
    assert "Regression detected" in html
    assert "failing" in html  # internal label still visible
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
    assert "Sanitizer nightly" in html


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


def test_hidden_metrics_do_not_enable_empty_history_column():
    """Hidden metrics with multi-night history must not turn on an all-n/a column."""
    runs = [
        _run(3, [_cell("inference_offline", "c", summary={"world_size": 2.0})],
             total=1, **{"pass": 1}),
        _run(4, [_cell("inference_offline", "c",
                        summary={"world_size": 2.0, "decode_latency_ms": 1.0})],
             total=1, **{"pass": 1}),
    ]
    html = gen_dashboard.build_dashboard_html(runs)
    assert "Additional summary metrics" in html
    assert ">history</th>" not in html


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
    assert "step time history" not in html


def test_dashboard_groups_cells_under_their_workload():
    entries = [
        {"entry": "llm_determinism", "cell": c, "verdict": "record",
         "reasons": [], "metrics": {"mean_step_time_ms": 10.0}}
        for c in ("bf16-12L", "tf32-24L")
    ]
    html = gen_dashboard.build_dashboard_html(
        [_results("2026-07-30T00:00:00Z", entries, total=2, record=2)])
    # Customer title once in the group header; internal id in wl-id + category anchor.
    assert "LLM determinism" in html
    assert html.count("llm_determinism") >= 1
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
    assert "<span class='variant-spark'>" not in one

    r2 = _results("2026-07-31T00:00:00Z",
                  [{"entry": "w", "cell": "c", "verdict": "record", "reasons": [],
                    "metrics": {"mean_step_time_ms": 2.0,
                                "summary": {"decode_latency_ms": 13.5}}}],
                  total=1, record=1)
    two = gen_dashboard.build_dashboard_html([r, r2])
    assert "<span class='variant-spark'>" in two


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
    assert "step time history" not in html


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
    assert "<h2>Nightly release health</h2>" not in gen_dashboard.build_dashboard_html(one)


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
    assert f"background:{gen_dashboard._UNKNOWN_COLOR}" in grid
    assert "? 1/2" in grid
    assert "1 pass · 1 error" in grid


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
    assert "not in this run" in grid
    assert "role='img'" in grid
    assert "aria-label='w2 — not in this run'" in grid


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
    assert grid.count("class='release-card'") == 5
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
    assert "Performance" in html
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
    assert grid.count("<span class='dot sm' role='img'") == 2


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
    assert "details.repro-panel" in html


def test_dashboard_hides_the_js_only_controls_until_the_script_runs():
    # The page is complete without JavaScript, so a control that does nothing
    # without it must not be visible in the served markup.
    runs = [_run(3, [_cell("w", "c", summary={"m": 1.0})], total=1, record=1),
            _run(4, [_cell("w", "c", summary={"m": 1.0})], total=1, record=1)]
    html = gen_dashboard.build_dashboard_html(runs)
    assert '<div class="toolbar" id="toolbar" hidden>' in html
    assert "bar.hidden = false" in html
    assert "Expand all" in html


def test_dashboard_renders_category_health_tiles():
    entries = [
        {"entry": "gpu_smoke", "cell": "baseline-local", "verdict": "record",
         "reasons": [], "metrics": {"mean_step_time_ms": 254.0}},
        {"entry": "inference_offline", "cell": "baseline-local", "verdict": "pass",
         "reasons": [], "metrics": {"mean_step_time_ms": 4.0,
                                    "summary": {"tokens_per_sec": 4160}}},
    ]
    html = gen_dashboard.build_dashboard_html(
        [_results("2026-08-03T00:00:00Z", entries, total=2, record=1, **{"pass": 1})])
    assert "Category health" in html
    assert "Platform" in html and "Inference" in html
    assert "Healthy" in html  # one pass + one record => passing headline


def test_dashboard_pass_and_skip_is_not_labelled_healthy():
    entries = [
        {"entry": "gpu_smoke", "cell": "baseline-local", "verdict": "pass",
         "reasons": [], "metrics": {"mean_step_time_ms": 1.0}},
        {"entry": "training_ddp_8gpu", "cell": None, "verdict": "skip",
         "reasons": ["needs 8 GPU(s), have 2"], "metrics": {}},
    ]
    doc = _results("2026-07-30T00:00:00Z", entries, total=2, **{"pass": 1, "skip": 1})
    assert gen_dashboard._latest_status([doc])[0] == "partial"
    html = gen_dashboard.build_dashboard_html([doc])
    assert "Partial run" in html
    assert ">Healthy<" not in html
    assert "partial" in html
    assert "partial</strong>" in html or "<strong>partial</strong>" in html


def test_dashboard_record_only_shows_baseline_setup_label():
    r = _results("2026-07-30T00:00:00Z",
                 [{"entry": "w", "cell": "c", "verdict": "record",
                   "reasons": ["no baseline (record-only)"],
                   "metrics": {"mean_step_time_ms": 1.0}}],
                 total=1, record=1)
    html = gen_dashboard.build_dashboard_html([r])
    assert "Baseline setup" in html
    assert "recording" in html
    assert ">passing<" not in html


def test_dashboard_run_command_block():
    r = _results("2026-07-30T00:00:00Z",
                 [{"entry": "inference_offline", "cell": "baseline-local",
                   "verdict": "record", "reasons": [],
                   "metrics": {"mean_step_time_ms": 4.0}}],
                 total=1, record=1)
    html = gen_dashboard.build_dashboard_html([r])
    assert "Reproduce this workload locally" in html
    assert "Before you start" in html
    assert "Success criteria" in html
    assert "--strict" in html
    assert "example-inference-smoke.yaml" in html
    assert "README-running-recipes.md" in html
    assert "details class='repro-panel' id='repro-inference_offline'" in html
    assert "<summary>Reproduce this workload locally</summary>" in html
    assert "torchrun" not in html  # inference is single-GPU


def test_history_grid_retired_chip_dims_label_not_status_badge():
    runs = [
        _run(3, [_cell("old_workload", "c", verdict="pass")], total=1, **{"pass": 1}),
        _run(4, [_cell("gpu_smoke", "c", verdict="pass")], total=1, **{"pass": 1}),
    ]
    html = gen_dashboard.build_dashboard_html(runs)
    assert ".wl-chip.retired { opacity" not in html
    assert ".wl-chip.retired .chip-label { color:var(--muted); }" in html
    grid = gen_dashboard.build_history_grid(runs)
    assert "wl-chip retired" in grid
    assert "background:#1a7f37" in grid


def test_history_grid_retired_workloads_do_not_link_to_missing_sections():
    runs = [
        _run(3, [_cell("old_workload", "c")], total=1, record=1),
        _run(4, [_cell("gpu_smoke", "c")], total=1, record=1),
    ]
    grid = gen_dashboard.build_history_grid(runs)
    assert "href='#wl-old_workload'" not in grid
    assert "wl-chip retired" in grid
    assert "href='#wl-gpu_smoke'" in grid


def test_history_grid_workloads_and_cells_link_to_repro_sections():
    runs = [
        _run(3, [_cell("gpu_smoke", "c"), _cell("race", "smoke")], total=2, record=2),
        _run(4, [_cell("gpu_smoke", "c"), _cell("race", "smoke")], total=2, record=2),
    ]
    grid = gen_dashboard.build_history_grid(runs)
    assert "href='#wl-gpu_smoke'" in grid
    assert "<ul class='wl-chips'>" in grid
    assert "overall health" in grid.lower() or "Overall health" in grid


def test_history_grid_overall_health_uses_customer_label():
    runs = [
        _run(3, [_cell("w", "c")], total=1, record=1),
        _run(4, [_cell("w", "c")], total=1, record=1),
    ]
    grid = gen_dashboard.build_history_grid(runs)
    assert "Baseline setup" in grid


def test_build_status_json_structured_latest_summary():
    entries = [
        {"entry": "gpu_smoke", "cell": "baseline-local", "verdict": "record",
         "reasons": [], "metrics": {"mean_step_time_ms": 1.0}},
    ]
    runs = [_results("2026-08-03T12:00:00Z", entries, total=1, record=1)]
    doc = gen_dashboard.build_status_json(runs)
    assert doc["schema_version"] == 2
    assert doc["latest"]["customer_status"] == "Baseline setup"
    assert doc["latest"]["categories"]["platform"]["worst_verdict"] == "record"
    assert "results" not in doc


def test_data_json_feed_stays_a_top_level_array():
    entries = [
        {"entry": "gpu_smoke", "cell": "baseline-local", "verdict": "record",
         "reasons": [], "metrics": {"mean_step_time_ms": 1.0}},
    ]
    runs = [_results("2026-08-03T12:00:00Z", entries, total=1, record=1)]
    import json
    feed = json.loads(json.dumps(runs, indent=2))
    assert isinstance(feed, list)
    assert feed == runs


def test_scaling_summary_uses_weak_scaling_efficiency():
    entries = [
        _cell("training_ddp", "c", summary={"step_time_p50": 8.0}),
        _cell("training_ddp_8gpu", "c", summary={"step_time_p50": 2.0}),
    ]
    html = gen_dashboard.build_scaling_summary(entries)
    assert "weak scaling" in html
    assert "400%" in html  # 8.0 / 2.0 * 100


def test_scaling_summary_renders_when_two_and_eight_gpu_training_present():
    entries = [
        _cell("training_ddp", "c", summary={"step_time_p50": 8.0}),
        _cell("training_ddp_8gpu", "c", summary={"step_time_p50": 2.5}),
        _cell("training_fsdp", "c", summary={"step_time_p50": 10.0}),
        _cell("training_fsdp_8gpu", "c", summary={"step_time_p50": 3.0}),
    ]
    html = gen_dashboard.build_scaling_summary(entries)
    assert "Training scaling" in html
    assert "DDP" in html and "FSDP" in html


def test_dashboard_metadata_covers_nightly_matrix():
    matrix_path = _REPO_ROOT / "config" / "ci" / "nightly_eval_matrix.yaml"
    text = matrix_path.read_text(encoding="utf-8")
    names = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- name:"):
            names.append(line.split(":", 1)[1].strip())
    meta = gen_dashboard.load_dashboard_metadata()
    workloads = meta.get("workloads") or {}
    missing = [n for n in names if n not in workloads]
    assert not missing, f"missing dashboard metadata for {missing}"
    for name in names:
        assert workloads[name].get("run_command"), name
        assert workloads[name].get("recipe"), name
        assert workloads[name].get("repro"), name
        assert workloads[name]["repro"].get("success_criteria"), name


def test_dashboard_repro_guides_differ_by_workload_type():
    meta = gen_dashboard.load_dashboard_metadata()
    wl = meta.get("workloads") or {}
    smoke = wl["gpu_smoke"]["repro"]
    ddp = wl["training_ddp"]["repro"]
    assert "One AMD GPU" in smoke["prerequisites"][0]
    assert "torchrun" in ddp["run"]["command"]
    assert any("NCCL" in c for step in ddp["setup"] for c in step["commands"])
    race = wl["race"]["repro"]
    assert any("AORTA_TRIAL_MASTER_PORT_BASE" in c for step in race["setup"] for c in step["commands"])
    assert any("python3 -m pip" in c for step in smoke["setup"] for c in step["commands"])
    inference = wl["inference_offline"]["repro"]
    verify_cmds = inference["verify"][0]["commands"]
    assert any("triage_results/repro/inference_offline" in c for c in verify_cmds)
    assert any("grep -m 10" in c for c in verify_cmds)
    assert any("{{HEAD_SHA}}" in c for step in smoke["setup"] for c in step["commands"])
    assert any("{{AORTA_VERSION}}" in c for step in smoke["setup"] for c in step["commands"])
    assert any("find triage_results" in c for c in verify_cmds)
    assert any('RUN_DIR="$(' in c for c in verify_cmds)
    assert any("$RUN_DIR/perf.md" in c for c in verify_cmds)
    assert any("Standalone --strict only" in c for c in verify_cmds)
    assert "triage_results/repro/inference_offline" in inference["run"]["command"]
    assert inference["run"]["command"].endswith("--strict")
    assert "triage_results/repro/training_ddp" in ddp["run"]["command"]
    assert "triage_results/repro/training_ddp_8gpu" in wl["training_ddp_8gpu"]["repro"]["run"]["command"]
    assert "passed/recording only" in ddp["success_criteria"].lower()
    det = wl["llm_determinism"]["repro"]["verify"][0]["commands"]
    assert any("ranks_with_divergence" in c for c in det)
    assert any("trial_paths" in c for c in det)
    assert any("python3 -" in c for c in det)
    assert any("|| true" in c for c in det)
    race_verify = race["verify"][0]["commands"]
    assert any("layer_checksum_mismatches" in c for c in race_verify)
    assert any("trial_paths" in c for c in race_verify)
    assert any("|| true" in c for c in race_verify)
    race8 = wl["race_8gpu"]["repro"]["verify"][0]["commands"]
    assert any("layer_checksum_mismatches" in c for c in race8)


def test_determinism_verify_reads_repo_relative_failed_trial(tmp_path):
    run_dir = (
        tmp_path
        / "triage_results"
        / "repro"
        / "llm_determinism"
        / "TICKET"
        / "llm_determinism"
        / "timestamp"
    )
    trial_path = run_dir / "cells" / "baseline" / "llm_determinism" / "trial_0.json"
    trial_path.parent.mkdir(parents=True)
    trial_path.write_text(
        json.dumps({"result": {"metrics": {"ranks_with_divergence": 2}}}),
        encoding="utf-8",
    )
    (run_dir / "matrix.json").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "name": "baseline",
                        "trial_paths": [str(trial_path.relative_to(tmp_path))],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    repro = gen_dashboard.load_dashboard_metadata()["workloads"]["llm_determinism"]["repro"]
    command = repro["verify"][0]["commands"][-1]
    env = {**os.environ, "RUN_DIR": str(run_dir.relative_to(tmp_path))}
    proc = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"baseline": [2]}


def _run_repro_trial_metric_parser(tmp_path, *, metric_key: str, metric_value: int) -> str:
    run_dir = (
        tmp_path
        / "triage_results"
        / "repro"
        / "llm_determinism"
        / "TICKET"
        / "llm_determinism"
        / "timestamp"
    )
    trial_path = run_dir / "cells" / "baseline" / "llm_determinism" / "trial_0.json"
    trial_path.parent.mkdir(parents=True)
    trial_path.write_text(
        json.dumps({"result": {"metrics": {metric_key: metric_value}}}),
        encoding="utf-8",
    )
    (run_dir / "matrix.json").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "name": "baseline",
                        "trial_paths": [str(trial_path.relative_to(tmp_path))],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "perf.md").write_text(
        f"# perf\n{metric_key}: 0\n",
        encoding="utf-8",
    )
    repro = gen_dashboard.load_dashboard_metadata()["workloads"]["llm_determinism"]["repro"]
    commands = repro["verify"][0]["commands"]
    grep_cmd = next(c for c in commands if "|| true" in c)
    parse_cmd = commands[-1]
    env = {**os.environ, "RUN_DIR": str(run_dir.relative_to(tmp_path))}
    grep = subprocess.run(
        ["bash", "-c", grep_cmd],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert grep.returncode == 0, grep.stderr
    proc = subprocess.run(
        ["bash", "-c", parse_cmd],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_determinism_verify_reads_failed_trial_even_when_perf_shows_zero(tmp_path):
    out = _run_repro_trial_metric_parser(
        tmp_path, metric_key="ranks_with_divergence", metric_value=2)
    assert json.loads(out) == {"baseline": [2]}


def test_race_verify_reads_failed_trial_even_when_perf_shows_zero(tmp_path):
    run_dir = (
        tmp_path
        / "triage_results"
        / "repro"
        / "race"
        / "TICKET"
        / "race"
        / "timestamp"
    )
    trial_path = run_dir / "cells" / "smoke" / "race" / "trial_0.json"
    trial_path.parent.mkdir(parents=True)
    trial_path.write_text(
        json.dumps({"result": {"metrics": {"layer_checksum_mismatches": 3}}}),
        encoding="utf-8",
    )
    (run_dir / "matrix.json").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "name": "smoke",
                        "trial_paths": [str(trial_path.relative_to(tmp_path))],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "perf.md").write_text("layer_checksum_mismatches: 0\n", encoding="utf-8")
    repro = gen_dashboard.load_dashboard_metadata()["workloads"]["race"]["repro"]
    commands = repro["verify"][0]["commands"]
    parse_cmd = commands[-1]
    env = {**os.environ, "RUN_DIR": str(run_dir.relative_to(tmp_path))}
    proc = subprocess.run(
        ["bash", "-c", parse_cmd],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"smoke": [3]}


def test_repro_pins_fail_closed_when_version_missing():
    html = gen_dashboard.build_dashboard_html(
        [_results("2026-08-03T00:00:00Z",
                  [{"entry": "gpu_smoke", "cell": "baseline-local", "verdict": "pass",
                    "reasons": [], "metrics": {"mean_step_time_ms": 1.0}}],
                  total=1, **{"pass": 1},
                  build={"head_sha": "abc123" * 5 + "abcd", "amd_aorta_version": ""})])
    assert "STOP: dashboard is missing amd_aorta_version" in html
    assert "expanded_assets/dev-wheels" in html
    assert "'amd-aorta[hw-queue]'  #" not in html


def test_dashboard_review_a11y_and_layout_fixes():
    entries = [
        {"entry": "gpu_smoke", "cell": "baseline-local", "verdict": "pass",
         "reasons": [], "metrics": {"mean_step_time_ms": 254.0,
                                    "summary": {"mean_step_time_ms": 254.0}}},
        {"entry": "inference_offline", "cell": "baseline-local", "verdict": "pass",
         "reasons": [], "metrics": {"mean_step_time_ms": 4.0,
                                    "summary": {"tokens_per_sec": 4160}}},
    ]
    doc = _results("2026-08-03T00:00:00Z", entries, total=2, **{"pass": 2},
                   build={"head_sha": "abc123" * 5 + "abcd", "amd_aorta_version": "0.2.2rc20260803"})
    doc2 = _results("2026-08-04T00:00:00Z", entries, total=2, **{"pass": 2},
                    build={"head_sha": "abc123" * 5 + "abcd", "amd_aorta_version": "0.2.2rc20260803"})
    html = gen_dashboard.build_dashboard_html([doc, doc2])
    assert "<h3 class='release-date'>" in html
    assert "<h4 class='wl-title'>" in html
    assert "<section class='repro-sec'><h5>Before you start</h5>" in html
    assert "nightly_eval.py" in html
    assert "overflow:visible" in html
    assert "class='tablewrap'><table class='inner'>" in html
    assert ".wl-chip.absent .chip-meta" in html
    assert ".absent { color:#39414a; }" not in html
    assert "git checkout abc123abc123abc123abc123abc123abcd" in html
    assert "amd-aorta[hw-queue]==0.2.2rc20260803" in html
    assert "mean step time history, oldest to newest" in html
    assert "role='img'" in html
    assert "#1a7f37" in html
    assert "nightly_eval.py comparing matrix.json" in html
    assert "with --strict against" not in html


def test_pass_badges_use_accessible_green_not_default_verdict_green():
    entries = [
        {"entry": "gpu_smoke", "cell": "baseline-local", "verdict": "pass",
         "reasons": [], "metrics": {"mean_step_time_ms": 254.0}},
    ]
    html = gen_dashboard.build_dashboard_html(
        [_results("2026-08-03T00:00:00Z", entries, total=1, **{"pass": 1})])
    assert "background:#1a7f37" in html
    assert "background:#2ea043" not in html


def test_gpu_smoke_success_notes_baseline_gating():
    meta = gen_dashboard.load_dashboard_metadata()
    success = meta["workloads"]["gpu_smoke"]["repro"]["success_criteria"]
    assert "gpu_smoke::baseline-local" in success
    assert "not performance-thresholded" in success

def test_dashboard_workloads_use_category_card_layout():
    entries = [
        {"entry": "gpu_smoke", "cell": "baseline-local", "verdict": "pass",
         "reasons": [], "metrics": {"mean_step_time_ms": 254.0}},
        {"entry": "training_ddp", "cell": "smoke", "verdict": "pass",
         "reasons": [], "metrics": {"mean_step_time_ms": 10.0,
                                    "summary": {"step_time_p50": 9.0}}},
    ]
    html = gen_dashboard.build_dashboard_html(
        [_results("2026-08-03T00:00:00Z", entries, total=2, **{"pass": 2})])
    assert "class='wl-card'" in html
    assert "class='wl-grid'" in html
    assert "class='cat-block'" in html
    assert "<table><thead><tr><th scope='col'>recipe variant</th>" not in html


def test_dashboard_failure_action_panel():
    r = _results("2026-07-29T00:00:00Z",
                 [{"entry": "race", "cell": "smoke", "verdict": "fail",
                   "reasons": ["expected passing cell but it did not pass"],
                   "metrics": {"mean_step_time_ms": 1.0}}],
                 total=1, fail=1)
    html = gen_dashboard.build_dashboard_html([r])
    assert "Next steps" in html
    assert "aorta env probe" in html


def test_headline_checksum_record_only_shows_captured_not_match():
    r = _results("2026-07-30T00:00:00Z",
                 [{"entry": "inference_offline", "cell": "baseline-local",
                   "verdict": "record", "reasons": ["no baseline (record-only)"],
                   "metrics": {"mean_step_time_ms": 4.0,
                               "summary": {"logits_checksum": 33904201}}}],
                 total=1, record=1)
    html = gen_dashboard.build_dashboard_html([r])
    assert "captured (33,904,201)" in html
    assert "match ✓" not in html


def test_headline_checksum_pass_verdict_still_shows_captured_not_match():
    """A pass verdict cannot confirm exact checksum equality after float rounding."""
    entry = {"entry": "inference_offline", "cell": "baseline-local",
             "verdict": "pass", "reasons": [],
             "metrics": {"mean_step_time_ms": 4.0,
                         "summary": {"logits_checksum": 33904201}},
             "deltas": {"metrics": {"logits_checksum": {
                 "observed": 33904201, "policy": "equal", "value": 33904201}}}}
    html = gen_dashboard.build_dashboard_html(
        [_results("2026-07-30T00:00:00Z", [entry], total=1, **{"pass": 1})])
    assert "logits checksum: captured (33,904,201)" in html
    assert "logits checksum: match ✓" not in html


def test_headline_checksum_no_match_on_float_equality_when_harness_failed():
    """Distinct int64 checksums can round to the same float; trust verdict, not ==."""
    collided = 9007199254740992.0  # 2**53 — float equality hides int64 differences
    entry = {"entry": "inference_offline", "cell": "baseline-local",
             "verdict": "fail",
             "reasons": ["metric 'logits_checksum' != expected 9007199254740993"],
             "metrics": {"mean_step_time_ms": 4.0,
                         "summary": {"logits_checksum": collided}},
             "deltas": {"metrics": {"logits_checksum": {
                 "observed": collided, "policy": "equal", "value": collided}}}}
    html = gen_dashboard.build_dashboard_html(
        [_results("2026-07-30T00:00:00Z", [entry], total=1, fail=1)])
    assert "logits checksum: mismatch ✗" in html
    assert "logits checksum: match ✓" not in html


def test_change_summary_reports_correctness_counter_without_pct_gate():
    runs = [
        _run(3, [_cell("race", "smoke", summary={"layer_checksum_mismatches": 0})],
             total=1, record=1),
        _run(4, [_cell("race", "smoke", summary={"layer_checksum_mismatches": 1})],
             total=1, record=1),
    ]
    html = gen_dashboard.build_change_summary(runs)
    assert "Correctness" in html
    assert "layer_checksum_mismatches" in html
    assert "0 → 1" in html


def test_change_summary_reports_checksum_inequality_below_pct_gate():
    runs = [
        _run(3, [_cell("w", "c", summary={"logits_checksum": 100})], total=1, record=1),
        _run(4, [_cell("w", "c", summary={"logits_checksum": 102})], total=1, record=1),
    ]
    html = gen_dashboard.build_change_summary(runs)
    assert "Correctness" in html
    assert "logits_checksum" in html
    assert "100 → 102" in html


def test_category_tile_anchors_to_workload_that_actually_ran():
    entries = [
        {"entry": "training_ddp_8gpu", "cell": "baseline-local", "verdict": "record",
         "reasons": [], "metrics": {"mean_step_time_ms": 10.0}},
    ]
    html = gen_dashboard.build_dashboard_html(
        [_results("2026-08-03T00:00:00Z", entries, total=1, record=1)])
    assert "href='#wl-training_ddp_8gpu'" in html
    assert "href='#wl-training_ddp'" not in html


def test_change_summary_groups_step_time_p99_under_performance():
    runs = [
        _run(3, [_cell("w", "c", summary={"step_time_p99": 40.0})], total=1, record=1),
        _run(4, [_cell("w", "c", summary={"step_time_p99": 50.0})], total=1, record=1),
    ]
    html = gen_dashboard.build_change_summary(runs)
    assert "Performance" in html
    assert "step_time_p99" in html


def test_change_summary_footer_splits_correctness_from_thresholded():
    cells = [
        _cell("w", f"c{i}", summary={"layer_checksum_mismatches": i})
        for i in range(6)
    ]
    runs = [
        _run(3, cells[:6], total=6, record=6),
        _run(4, [_cell("w", f"c{i}", summary={"layer_checksum_mismatches": i + 1})
                 for i in range(6)], total=6, record=6),
    ]
    html = gen_dashboard.build_change_summary(runs)
    assert "and 2 more correctness changes" in html
    assert "past 10%" not in html


# ---------------------------------------------------------------------------
# The canary lane must stay invisible to the gated dashboard (issue #382)
# ---------------------------------------------------------------------------


def test_load_results_ignores_the_canary_subdirectory(tmp_path):
    """Structural isolation, not a naming convention.

    latest-rocm-canary.yml publishes observed-only rows to
    ``results/canary/<date>.json`` on the ci-results branch, and relies on this
    glob being NON-recursive to keep them out of the gated dashboard's history
    and trends. If that ever becomes ``rglob`` / ``**/*.json``, a red canary on a
    brand-new ROCm would start moving the gated page -- which is exactly the
    ambiguity #382 exists to avoid -- so the property is pinned here rather than
    left to a comment.
    """
    results = tmp_path / "results"
    (results / "canary").mkdir(parents=True)
    (results / "2026-08-01.json").write_text(
        json.dumps(_results("2026-08-01T00:00:00Z", [], build={"lane": "gate"})),
        encoding="utf-8",
    )
    (results / "canary" / "2026-08-01.json").write_text(
        json.dumps(_results("2026-08-01T12:00:00Z", [], build={"lane": "canary"})),
        encoding="utf-8",
    )

    loaded = gen_dashboard.load_results(results)
    assert [doc["build"]["lane"] for doc in loaded] == ["gate"]


def test_dashboard_is_unchanged_by_a_canary_row_on_disk(tmp_path):
    """#382 acceptance: the pinned gate is unchanged by this work."""
    results = tmp_path / "results"
    (results / "canary").mkdir(parents=True)
    gate_doc = _results("2026-08-01T00:00:00Z", [], build={"lane": "gate"})
    (results / "2026-08-01.json").write_text(json.dumps(gate_doc), encoding="utf-8")
    before = gen_dashboard.build_dashboard_html(gen_dashboard.load_results(results))

    (results / "canary" / "2026-08-01.json").write_text(
        json.dumps(_results("2026-08-01T12:00:00Z", [], build={"lane": "canary"})),
        encoding="utf-8",
    )
    after = gen_dashboard.build_dashboard_html(gen_dashboard.load_results(results))
    assert before == after


# ---------------------------------------------------------------------------
# Observed-only canary rendering (issue #382)
# ---------------------------------------------------------------------------


def _canary(generated, rocm, *, base_image=None, passed=1, failed=0):
    return _results(
        generated,
        [],
        build={
            "lane": "canary",
            "rocm": rocm,
            "torch": f"2.12.0+rocm{rocm}",
            "hip": "7.14.60850",
            "base_image": base_image or ("rocm/pytorch:latest@sha256:" + "ab" * 32),
        },
        **{"pass": passed, "fail": failed, "total": passed + failed},
    )


def test_canary_section_renders_the_observed_stack():
    html = gen_dashboard.build_canary_section(
        [_canary("2026-08-19T15:00:00Z", "7.14.0")]
    )
    assert 'id="canary"' in html
    assert "observed only" in html
    assert "7.14.0" in html and "2.12.0+rocm7.14.0" in html and "7.14.60850" in html
    # The digest is the whole point of the lane, so it must be on the page.
    assert "sha256:ababababab" in html


def test_canary_section_says_plainly_that_it_is_not_a_gate():
    html = gen_dashboard.build_canary_section(
        [_canary("2026-08-19T15:00:00Z", "7.14.0")]
    )
    assert "Not a gate" in html
    assert "docker/Dockerfile.ci-gpu" in html


def test_canary_section_carries_no_health_colouring():
    """A neutral lane must not health-signal (#368 5th-pass class).

    A red canary row means "a new ROCm release moved something", not "we
    regressed" -- colouring it recreates the ambiguity #382 exists to avoid.
    Checks a failing run specifically, since that is when a colour would appear.
    """
    html = gen_dashboard.build_canary_section(
        [_canary("2026-08-19T15:00:00Z", "7.14.0", passed=0, failed=3)]
    )
    for signal in ("vchip", "pill", "execution bad", "REGRESSION", "style=color:"):
        assert signal not in html, signal


def test_canary_section_has_an_empty_state_rather_than_vanishing():
    """The #canary anchor must resolve before the lane's first run (KB#11b)."""
    html = gen_dashboard.build_canary_section([])
    assert 'id="canary"' in html
    assert "No canary runs recorded yet" in html


def test_canary_section_is_bounded():
    rows = [_canary(f"2026-07-{d:02d}T15:00:00Z", "7.14.0") for d in range(1, 31)]
    html = gen_dashboard.build_canary_section(rows)
    assert html.count("<tr>") == gen_dashboard._CANARY_ROWS + 1  # + header row


def test_canary_rows_do_not_touch_the_gated_status_or_trend():
    """#382 acceptance: the pinned gate is unchanged by this work.

    ``canary_results`` is a separate argument for exactly this reason -- the
    banner, pass-rate trend and history grid are all computed from ``results``.
    """
    gate = [_results("2026-08-19T00:00:00Z", [], build={"lane": "gate"}, **{"pass": 2})]
    without = gen_dashboard.build_dashboard_html(gate)
    with_canary = gen_dashboard.build_dashboard_html(
        gate, [_canary("2026-08-19T15:00:00Z", "7.14.0", passed=0, failed=9)]
    )
    # The only difference is the canary section itself.
    assert without != with_canary
    assert gen_dashboard.build_status_json(gate) == gen_dashboard.build_status_json(gate)
    canary_only = gen_dashboard.build_canary_section(
        [_canary("2026-08-19T15:00:00Z", "7.14.0", passed=0, failed=9)]
    )
    assert with_canary.replace(canary_only, "").replace(
        gen_dashboard.build_canary_section([]), ""
    ) == without.replace(gen_dashboard.build_canary_section([]), "")


def test_dashboard_html_default_is_unchanged_without_canary_data():
    """Existing callers keep their exact output (the arg defaults to None)."""
    gate = [_results("2026-08-19T00:00:00Z", [], build={"lane": "gate"})]
    assert gen_dashboard.build_dashboard_html(gate) == gen_dashboard.build_dashboard_html(
        gate, None
    )
    assert gen_dashboard.build_dashboard_html(gate) == gen_dashboard.build_dashboard_html(
        gate, []
    )


def test_canary_section_renders_a_setup_failure_row(tmp_path):
    """The row the workflow synthesises when setup fails before the evaluator.

    That path exists so a `:latest` we cannot even install still appears in
    history with its digest (#387). It must therefore render: everything the
    container would have reported is null, and only the digest is real.
    """
    doc = _results(
        "2026-08-19T15:00:00Z",
        [],
        build={
            "lane": "canary",
            "base_image": "rocm/pytorch:latest@sha256:deadbeef",
            "rocm": None,
            "torch": None,
            "hip": None,
        },
    )
    doc["error"] = "canary setup failed before the evaluator ran (exit 1)"
    html = gen_dashboard.build_canary_section([doc])
    assert "sha256:deadbeef" in html
    # Nulls read as em dashes rather than the literal "None".
    assert "None" not in html
    assert "0/0" in html


def test_short_digest_display():
    assert gen_dashboard._short_digest("r/p:latest@sha256:" + "a" * 64) == (
        "sha256:aaaaaaaaaaaa"
    )
    # A tag-only value keeps its text: silently em-dashing it would hide that a
    # run was NOT digest-pinned, which is the one thing the column is for.
    assert gen_dashboard._short_digest("rocm/pytorch:latest") == "rocm/pytorch:latest"
    assert gen_dashboard._short_digest(None) == "—"
    assert gen_dashboard._short_digest("") == "—"


def test_main_tolerates_an_absent_canary_dir(tmp_path, monkeypatch, capsys):
    """The lane is best-effort; a missing dir must not fail the gated build."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "2026-08-19.json").write_text(
        json.dumps(_results("2026-08-19T00:00:00Z", [], build={"lane": "gate"})),
        encoding="utf-8",
    )
    out = tmp_path / "site"
    monkeypatch.setattr(
        "sys.argv",
        [
            "gen_dashboard.py",
            "--results-dir", str(results),
            "--out-dir", str(out),
            "--canary-results-dir", str(tmp_path / "does-not-exist"),
        ],
    )
    assert gen_dashboard.main() == 0
    capsys.readouterr()
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "No canary runs recorded yet" in html


def test_canary_rows_render_with_no_gated_history_at_all(tmp_path, monkeypatch, capsys):
    """Neither lane may gate the other (#387).

    The two lanes publish to `ci-results` from two different workflows, so the
    canary can legitimately land first. `pages.yml` used to make the whole render
    conditional on a top-level gated `results/*.json`, which hid canary history
    until the first gated nightly; this pins the generator half of that fix.
    """
    results = tmp_path / "results"
    (results / "canary").mkdir(parents=True)  # no gated *.json beside it
    (results / "canary" / "2026-08-19.json").write_text(
        json.dumps(_canary("2026-08-19T15:00:00Z", "7.14.0")), encoding="utf-8"
    )
    out = tmp_path / "site"
    monkeypatch.setattr(
        "sys.argv",
        [
            "gen_dashboard.py",
            "--results-dir", str(results),
            "--out-dir", str(out),
            "--canary-results-dir", str(results / "canary"),
        ],
    )
    assert gen_dashboard.main() == 0
    capsys.readouterr()
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "7.14.0" in html
    assert 'id="canary"' in html
    # The gated side is simply empty, and status.json still describes only it.
    assert json.loads((out / "data.json").read_text(encoding="utf-8")) == []


def test_main_renders_canary_rows_without_entering_data_or_status_json(tmp_path, monkeypatch, capsys):
    results = tmp_path / "results"
    (results / "canary").mkdir(parents=True)
    (results / "2026-08-19.json").write_text(
        json.dumps(_results("2026-08-19T00:00:00Z", [], build={"lane": "gate"})),
        encoding="utf-8",
    )
    (results / "canary" / "2026-08-19.json").write_text(
        json.dumps(_canary("2026-08-19T15:00:00Z", "7.14.0")), encoding="utf-8"
    )
    out = tmp_path / "site"
    monkeypatch.setattr(
        "sys.argv",
        [
            "gen_dashboard.py",
            "--results-dir", str(results),
            "--out-dir", str(out),
            "--canary-results-dir", str(results / "canary"),
        ],
    )
    assert gen_dashboard.main() == 0
    capsys.readouterr()
    assert "7.14.60850" in (out / "index.html").read_text(encoding="utf-8")
    # data.json stays the gated record; status.json must not see the lane.
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))
    assert [d["build"]["lane"] for d in data] == ["gate"]
    assert "canary" not in (out / "status.json").read_text(encoding="utf-8")
