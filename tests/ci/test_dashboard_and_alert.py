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


def _results(generated, entries, **summary):
    base = {"total": 0, "pass": 0, "fail": 0, "record": 0, "skip": 0}
    base.update(summary)
    return {"generated_at": generated, "build": {"amd_aorta_version": "0.2.1rc"}, "summary": base, "entries": entries}


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


def test_dashboard_skip_only_run_is_neither_passing_nor_recording():
    # Every cell skipped for want of GPUs establishes nothing, so neither
    # "passing" nor the record-only baseline story applies.
    r = _results("2026-07-30T00:00:00Z",
                 [{"entry": "w", "cell": "c", "verdict": "skip",
                   "reasons": ["needs 8 GPUs, runner exposes 2"], "metrics": {}}],
                 total=1, skip=1)
    html = gen_dashboard.build_dashboard_html([r])
    assert "skipping" in html
    assert ">passing<" not in html
    assert "No baselines are blessed yet" not in html
    assert "all 1 cells were" in html


def test_dashboard_partial_record_does_not_claim_every_cell_recorded():
    # record + skip must not be described as "recorded all N cells".
    entries = [{"entry": "w", "cell": f"r{i}", "verdict": "record",
                "reasons": ["no baseline (record-only)"],
                "metrics": {"mean_step_time_ms": 1.0}} for i in range(3)]
    entries.append({"entry": "w", "cell": "s0", "verdict": "skip",
                    "reasons": ["needs 8 GPUs, runner exposes 2"], "metrics": {}})
    html = gen_dashboard.build_dashboard_html(
        [_results("2026-07-30T00:00:00Z", entries, total=4, record=3, skip=1)])
    assert "recording" in html
    assert "3 of 4 cells (1 skipped)" in html
    assert "all 4 cells" not in html


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
