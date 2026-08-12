"""Unit tests for the sanitizer dashboard generator (pure logic)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load():
    path = _REPO_ROOT / "scripts" / "sanitizers" / "gen_sanitizer_dashboard.py"
    spec = importlib.util.spec_from_file_location("gen_sanitizer_dashboard", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load()


def _waitcheck_report() -> dict:
    finding = {
        "sanitizer": "waitcheck", "severity": "warning", "code": "wait_hazard",
        "message": "/a/b/sol_1.hsaco:gfx950[0]:.text+0x1: missing s_waitcnt lgkmcnt(0)",
        "kernel_name": "gemm_x", "code_object": "/a/b/sol_1.hsaco",
        "entry_offset": None, "metadata": {},
    }
    return {
        "schema": "aorta.sanitizer_report/0.1", "target": "gfx950",
        "overall_verdict": "warn", "execution_status": "complete",
        "worklist": {
            "schema": "aorta.kernel_worklist/0.1", "requirement": "top_dispatch_count",
            "top_n": 3, "kernel_count": 1,
            "kernels": [{
                "identity": {
                    "name": "gemm_x", "target": "gfx950", "code_object": "/a/b/sol_1.hsaco",
                    "code_object_sha256": "93f09ae670abcdef", "code_object_index": 0,
                    "entry_offset": None,
                },
                "total_time_ms": 0.0, "dispatch_count": 406, "sources": ["gemm_csv"],
            }],
        },
        "checks": [{
            "sanitizer": "waitcheck", "state": "ran", "verdict": "warn",
            "reason": None, "returncode": 0, "findings": [finding],
            "kernel_results": [{
                "identity": {"name": "gemm_x", "target": "gfx950"},
                "state": "ran", "verdict": "warn", "findings": [finding],
                "reason": None, "returncode": 0,
            }],
            "coverage": [],
            "backend": {"path": "/tmp/build/tools/rj_waitcheck", "sha256": "472fcf288714beef"},
        }],
    }


def _consan_racy_report() -> dict:
    race = {
        "sanitizer": "consan", "severity": "race", "code": "1",
        "message": "[rocjitsu-dbi-hooks] ConSan MOI auto replay diagnostic conflict=true",
        "kernel_name": None, "code_object": None, "entry_offset": None, "metadata": {},
    }
    return {
        "schema": "aorta.sanitizer_report/0.1", "target": "gfx950",
        "overall_verdict": "fail", "execution_status": "complete",
        "worklist": {
            "schema": "aorta.kernel_worklist/0.1", "requirement": "top_dispatch_count",
            "top_n": 1, "kernel_count": 1,
            "kernels": [{
                "identity": {
                    "name": "consan_lds_race_2wave", "target": "gfx950", "code_object": None,
                    "code_object_sha256": None, "code_object_index": None, "entry_offset": None,
                },
                "total_time_ms": 0.0, "dispatch_count": 1, "sources": ["consan_repro"],
            }],
        },
        "checks": [{
            "sanitizer": "consan", "state": "ran", "verdict": "fail",
            "reason": None, "returncode": 0, "findings": [race, race],
            "kernel_results": [],
            "coverage": [{"object_id": "reader=1,load=2", "access": "2/2"}],
            "backend": {},
        }],
    }


def _incomplete_execution_report() -> dict:
    report = _waitcheck_report()
    report["execution_status"] = "not_checked"
    return report


def test_summarize_waitcheck_reports_per_kernel_detail():
    case = gen.summarize_case(_waitcheck_report(), "warn")

    assert case["verdict"] == "warn"
    assert case["match"] is True
    assert case["backend"] == {"name": "rj_waitcheck", "sha": "472fcf288714"}
    assert len(case["kernels"]) == 1
    kernel = case["kernels"][0]
    assert kernel["name"] == "gemm_x"
    assert kernel["dispatch"] == 406
    assert kernel["verdict"] == "warn"
    assert kernel["findings"] == 1
    assert kernel["code_object"] == "sol_1.hsaco"
    assert kernel["sha"] == "93f09ae670"
    # finding groups collapse by (sanitizer, code, severity) with a cleaned example
    assert case["finding_groups"][0]["code"] == "wait_hazard"
    assert case["finding_groups"][0]["count"] == 1
    assert case["finding_groups"][0]["example"].startswith("sol_1.hsaco:")


def test_summarize_consan_credits_process_findings_to_single_kernel():
    case = gen.summarize_case(_consan_racy_report(), "fail")

    assert case["verdict"] == "fail" and case["match"] is True
    assert case["coverage"] == "2/2"
    # kernel_results is empty and race findings carry kernel_name=None; the single
    # worklist kernel must still be credited with them.
    assert case["kernels"][0]["findings"] == 2
    assert case["kernels"][0]["verdict"] == "fail"
    assert case["finding_groups"][0]["severity"] == "race"
    assert case["finding_groups"][0]["count"] == 2


def test_missing_report_is_marked_absent():
    case = gen.summarize_case(None, "pass")
    assert case["present"] is False
    assert case["verdict"] == "—"
    assert case["match"] is False


def test_summary_md_expected_warn_and_fail_are_healthy():
    rows = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-clean": gen.summarize_case(_consan_racy_report(), "fail"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    runs = [{"meta": {"run": "r1", "commit": "abc", "date": "d"}, "rows": rows, "gate": True}]
    md = gen.build_summary_md(runs)
    assert "**HEALTHY** — 3/3 sanitizer outcomes match their baselines" in md
    assert md.count("✅ **Expected outcome**") >= 3
    assert "positive-control outcomes" in md
    assert "| Baseline status | Observed | Expected | Execution |" in md
    assert "| `fail` | `fail` |" in md
    assert "**REGRESSION**" not in md
    assert "Kernel details" in md
    assert "`gemm_x`" in md and "sol_1.hsaco" in md
    assert "Observed sanitizer verdict `fail` · expected `fail`" in md
    # per-case observation summary is surfaced on the guardrail (Tab 1) MD too
    assert "Observation: waitcheck warn" in md
    assert "| Kernel | Dispatch | Observed sanitizer verdict |" in md
    assert "✅ **Match**<br>Observed: `warn`" in md
    assert "Healthy |" in md
    assert "green" not in md and "red" not in md


def test_summary_md_mismatch_is_actionable_regression():
    rows = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        # baseline says pass but report is fail -> mismatch
        "consan-clean": gen.summarize_case(_consan_racy_report(), "pass"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    gate = all(r["match"] for r in rows.values())
    runs = [{"meta": {"run": "r1", "commit": "abc", "date": "d"}, "rows": rows, "gate": gate}]
    md = gen.build_summary_md(runs)
    assert "**REGRESSION** — investigate 1/3 sanitizer outcomes" in md
    assert "❌ **Unexpected outcome**" in md
    assert "| `fail` | `pass` |" in md
    assert "❌ **Mismatch**<br>Observed: `fail`; expected `pass`" in md
    assert "Regression |" in md


def test_build_html_expected_warn_and_fail_are_healthy_and_neutral():
    rows = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-clean": gen.summarize_case(_consan_racy_report(), "fail"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    runs = [
        {
            "meta": {"run": "r1", "commit": "abc", "date": "d", "gpu": "gfx950"},
            "rows": rows,
            "gate": True,
        }
    ]
    html = gen.build_html(runs)
    assert "<!doctype html>" in html
    assert "HEALTHY — 3/3 sanitizer outcomes match their baselines" in html
    assert html.count('<span class="pill ok">Expected outcome</span>') >= 3
    assert '<span class="observed">warn</span>' in html
    assert '<span class="observed">fail</span>' in html
    assert "positive-control outcomes" in html
    assert "<th>Baseline status</th><th>Observed</th>" in html
    assert "<th>Expected</th><th>Execution</th>" in html
    assert "gemm_x" in html and "Kernel details" in html
    assert "Observed sanitizer verdict" in html
    assert '<span class="pill ok">Match</span>' in html
    assert '<span class="pill pass">Healthy</span>' in html
    assert ">green<" not in html and ">red<" not in html


def test_build_html_mismatch_is_primary_regression_signal():
    rows = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-clean": gen.summarize_case(_consan_racy_report(), "pass"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    runs = [
        {
            "meta": {"run": "r1", "commit": "abc", "date": "d", "gpu": "gfx950"},
            "rows": rows,
            "gate": False,
        }
    ]

    html = gen.build_html(runs)

    assert "REGRESSION — investigate 1/3 sanitizer outcomes" in html
    assert '<span class="pill bad">Unexpected outcome</span>' in html
    assert '<span class="pill bad">Mismatch</span>' in html
    assert 'Observed <span class="observed">fail</span>' in html
    assert 'expected <span class="observed">pass</span>' in html
    assert '<span class="pill fail">Regression</span>' in html


def test_renderers_make_missing_report_explicit():
    rows = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-clean": gen.summarize_case(None, "pass"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    runs = [
        {
            "meta": {"run": "r1", "commit": "abc", "date": "d", "gpu": "gfx950"},
            "rows": rows,
            "gate": False,
        }
    ]

    html = gen.build_html(runs)
    md = gen.build_summary_md(runs)

    assert '<span class="pill bad">Report missing</span>' in html
    assert 'Observed <span class="observed">—</span>' in html
    assert "❌ **Report missing**" in md
    assert "Observed sanitizer verdict: `—`" in md
    assert "❌ **Report missing**<br>Observed: `—`" in md
    # the per-case observation summary renders for a missing guardrail report too,
    # on both Tab 1 renderers (parity with the survey missing branch / #367).
    assert '<div class="secondary">Observation: report missing</div>' in html
    assert "Observation: report missing" in md


def test_gate_summary_distinguishes_missing_from_regression():
    # An absent report and an observed verdict mismatch both fail the gate, but
    # only a present-and-mismatched verdict is a regression; missing reports and
    # the combined case get their own aggregate labels.
    all_match = {c: gen.summarize_case(_waitcheck_report(), "warn") for c, *_ in gen.CASES}
    assert gen._gate_summary(all_match) == {
        "ok": True,
        "label": "HEALTHY",
        "detail": "3/3 sanitizer outcomes match their baselines",
        "short": "Healthy",
    }

    missing_only = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-clean": gen.summarize_case(None, "pass"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    incomplete = gen._gate_summary(missing_only)
    assert incomplete["ok"] is False and incomplete["label"] == "INCOMPLETE"
    assert incomplete["detail"] == "1/3 sanitizer report(s) are missing"

    mismatch_only = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-clean": gen.summarize_case(_consan_racy_report(), "pass"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    regression = gen._gate_summary(mismatch_only)
    assert regression["label"] == "REGRESSION" and "1/3" in regression["detail"]

    combined = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-clean": gen.summarize_case(None, "pass"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "pass"),
    }
    unhealthy = gen._gate_summary(combined)
    assert unhealthy["label"] == "UNHEALTHY"
    assert "1/3 mismatched" in unhealthy["detail"] and "1/3 missing" in unhealthy["detail"]


def test_missing_only_run_is_incomplete_not_regression():
    rows = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-clean": gen.summarize_case(None, "pass"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    runs = [
        {
            "meta": {"run": "r1", "commit": "abc", "date": "d", "gpu": "gfx950"},
            "rows": rows,
            "gate": False,
        }
    ]

    html = gen.build_html(runs)
    md = gen.build_summary_md(runs)

    assert "INCOMPLETE — 1/3 sanitizer report(s) are missing" in html
    assert '<span class="pill fail">Incomplete</span>' in html
    assert "**INCOMPLETE** — 1/3 sanitizer report(s) are missing" in md
    assert "Incomplete |" in md
    # a missing report must not be surfaced as a verdict regression anywhere
    assert "Regression" not in html and "REGRESSION" not in html
    assert "Regression" not in md and "REGRESSION" not in md
    # per-case missing status stays explicit
    assert '<span class="pill bad">Report missing</span>' in html
    assert "❌ **Report missing**" in md


def test_combined_mismatch_and_missing_is_unhealthy():
    rows = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-clean": gen.summarize_case(None, "pass"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "pass"),
    }
    runs = [
        {
            "meta": {"run": "r1", "commit": "abc", "date": "d", "gpu": "gfx950"},
            "rows": rows,
            "gate": False,
        }
    ]

    html = gen.build_html(runs)
    md = gen.build_summary_md(runs)

    detail = "investigate 1/3 mismatched outcome(s) and 1/3 missing report(s)"
    assert f"UNHEALTHY — {detail}" in html
    assert '<span class="pill fail">Unhealthy</span>' in html
    assert f"**UNHEALTHY** — {detail}" in md
    assert "Unhealthy |" in md


def test_summary_md_emphasizes_noncomplete_execution():
    rows = {
        "waitcheck": gen.summarize_case(_incomplete_execution_report(), "warn"),
        "consan-clean": gen.summarize_case(_consan_racy_report(), "fail"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    runs = [{"meta": {"run": "r1", "commit": "abc", "date": "d"}, "rows": rows, "gate": True}]

    md = gen.build_summary_md(runs)

    # non-complete execution is error-emphasized in Markdown, matching _execution_html
    assert "❌ **not_checked**" in md
    # complete executions stay neutral (no false emphasis)
    assert "❌ **complete**" not in md


def test_build_html_empty_shows_placeholder_and_back_link():
    # Rendered for the /sanitizers/ route before the first successful nightly so
    # it never 404s. Must carry the back-link and a clear "no runs yet" message.
    html = gen.build_html([])
    assert "<!doctype html>" in html
    assert 'href="../"' in html
    assert "No sanitizer runs yet" in html


def test_build_html_renders_stale_banner_when_unhealthy():
    rows = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-clean": gen.summarize_case(_consan_racy_report(), "fail"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    runs = [{"meta": {"run": "r1", "commit": "abc", "date": "d", "gpu": "gfx950"}, "rows": rows, "gate": True}]
    status = {
        "healthy": False, "conclusion": "failure", "run_id": "42",
        "run_url": "https://github.com/ROCm/aorta/actions/runs/42", "date": "2026-08-05",
    }
    html = gen.build_html(runs, status=status)
    assert "did not complete successfully" in html
    assert "failure" in html
    assert "actions/runs/42" in html


def test_build_html_no_banner_when_healthy():
    runs = [{"meta": {"run": "r1", "commit": "abc", "date": "d", "gpu": "gfx950"},
             "rows": {c: gen.summarize_case(_waitcheck_report(), "warn") for c, *_ in gen.CASES},
             "gate": True}]
    healthy = {"healthy": True, "conclusion": "success", "run_id": "42", "run_url": "", "date": "d"}
    assert "did not complete successfully" not in gen.build_html(runs, status=healthy)
    # An unhealthy banner also renders on the empty-state page.
    empty = gen.build_html([], status={"healthy": False, "conclusion": "failure",
                                       "run_id": "7", "run_url": "", "date": ""})
    assert "did not complete successfully" in empty


def test_build_summary_md_stale_banner():
    status = {"healthy": False, "conclusion": "failure", "run_id": "9",
              "run_url": "https://x/9", "date": "d"}
    md = gen.build_summary_md([], status=status)
    assert "Stale" in md and "https://x/9" in md


def test_main_empty_runs_root_publishes_placeholder(tmp_path, monkeypatch):
    # The empty-runs-root path Pages uses to guarantee /sanitizers/index.html.
    baselines = _REPO_ROOT / "recipes" / "sanitizers" / "fixtures" / "expected" / "verdict_baselines.json"
    out = tmp_path / "out"
    argv = [
        "gen_sanitizer_dashboard",
        "--runs-root", str(tmp_path / "empty"),
        "--baselines", str(baselines),
        "--out-dir", str(out),
    ]
    (tmp_path / "empty").mkdir()
    monkeypatch.setattr(sys, "argv", argv)
    assert gen.main() == 0
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "No sanitizer runs yet" in index


def test_main_writes_status_json_and_banner(tmp_path, monkeypatch):
    baselines = _REPO_ROOT / "recipes" / "sanitizers" / "fixtures" / "expected" / "verdict_baselines.json"
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({
        "healthy": False, "conclusion": "failure", "run_id": "13",
        "run_url": "https://github.com/ROCm/aorta/actions/runs/13",
        "ref": "refs/heads/main", "date": "2026-08-05",
    }))
    out = tmp_path / "out"
    (tmp_path / "empty").mkdir()
    argv = [
        "gen_sanitizer_dashboard",
        "--runs-root", str(tmp_path / "empty"),
        "--baselines", str(baselines),
        "--status", str(status_file),
        "--out-dir", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert gen.main() == 0
    assert (out / "status.json").is_file()
    assert "did not complete successfully" in (out / "index.html").read_text(encoding="utf-8")


def test_runs_from_results_dir_matches_baselines(tmp_path):
    baselines = {
        "waitcheck_gemm": {"overall_verdict": "warn"},
        "consan_clean": {"overall_verdict": "pass"},
        "consan_racy": {"overall_verdict": "fail"},
    }
    (tmp_path / "waitcheck").mkdir()
    (tmp_path / "waitcheck" / "sanitizer_report.json").write_text(json.dumps(_waitcheck_report()))
    (tmp_path / "consan-racy").mkdir()
    (tmp_path / "consan-racy" / "sanitizer_report.json").write_text(json.dumps(_consan_racy_report()))
    # consan-clean intentionally absent -> present False, gate must be unhealthy

    runs = gen.runs_from_results_dir(tmp_path, baselines, meta={"run": "r", "commit": "c"})
    assert len(runs) == 1
    rows = runs[0]["rows"]
    assert rows["waitcheck"]["match"] is True
    assert rows["consan-racy"]["match"] is True
    assert rows["consan-clean"]["present"] is False
    assert runs[0]["gate"] is False


def _consan_clean_report() -> dict:
    return {
        "schema": "aorta.sanitizer_report/0.1", "target": "gfx950",
        "overall_verdict": "pass", "execution_status": "complete",
        "worklist": {
            "schema": "aorta.kernel_worklist/0.1", "requirement": "top_dispatch_count",
            "top_n": 1, "kernel_count": 1,
            "kernels": [{
                "identity": {
                    "name": "consan_lds_clean", "target": "gfx950", "code_object": None,
                    "code_object_sha256": None, "code_object_index": None, "entry_offset": None,
                },
                "total_time_ms": 0.0, "dispatch_count": 1, "sources": ["consan_repro"],
            }],
        },
        "checks": [{
            "sanitizer": "consan", "state": "ran", "verdict": "pass",
            "reason": None, "returncode": 0, "findings": [], "kernel_results": [],
            "coverage": [{"object_id": "reader=1,load=1", "access": "1/1"}], "backend": {},
        }],
    }


def _baselines() -> dict:
    return {
        "waitcheck_gemm": {"overall_verdict": "warn"},
        "consan_clean": {"overall_verdict": "pass"},
        "consan_racy": {"overall_verdict": "fail"},
    }


def _write_history_run(root: Path, run_id: str, *, meta: dict | None = None) -> Path:
    """Lay out one published run: root/<id>/<case>/sanitizer_report.json + meta.json."""
    run_dir = root / run_id
    for case, report in (
        ("waitcheck", _waitcheck_report()),
        ("consan-clean", _consan_clean_report()),
        ("consan-racy", _consan_racy_report()),
    ):
        (run_dir / case).mkdir(parents=True)
        (run_dir / case / "sanitizer_report.json").write_text(json.dumps(report))
    manifest = {
        "run": run_id, "commit": "0123456789abcdef", "date": "2026-08-05",
        "gpu": "gfx950", "run_url": "https://github.com/ROCm/aorta/actions/runs/99",
        "gate": True,
    }
    if meta:
        manifest.update(meta)
    (run_dir / "meta.json").write_text(json.dumps(manifest))
    return run_dir


def test_history_root_enumerates_newest_first(tmp_path):
    root = tmp_path / "runs"
    for run_id in ("2026-08-03-11", "2026-08-05-33", "2026-08-04-22"):
        _write_history_run(root, run_id)

    runs = gen.runs_from_history_root(root, _baselines())

    ids = [r["meta"]["run"] for r in runs]
    assert ids == ["2026-08-05-33", "2026-08-04-22", "2026-08-03-11"]
    # rel + report_rel are threaded through, relative to the dashboard root.
    latest = runs[0]
    assert latest["rel"] == "runs/2026-08-05-33"
    assert (
        latest["rows"]["waitcheck"]["report_rel"]
        == "runs/2026-08-05-33/waitcheck/sanitizer_report.json"
    )
    # meta.json is read back (commit is shortened to 12 chars for display).
    assert latest["meta"]["commit"] == "0123456789ab"
    assert latest["meta"]["gpu"] == "gfx950"
    assert latest["meta"]["run_url"].endswith("/runs/99")


def test_history_root_keep_caps_to_newest_n(tmp_path):
    root = tmp_path / "runs"
    for i in range(5):
        _write_history_run(root, f"2026-08-0{i + 1}-{i + 1}")

    runs = gen.runs_from_history_root(root, _baselines(), keep=2)

    assert [r["meta"]["run"] for r in runs] == ["2026-08-05-5", "2026-08-04-4"]


def test_history_root_orders_variable_width_run_ids_numerically(tmp_path):
    root = tmp_path / "runs"
    # Same day, variable-width run ids: 100 > 10 > 9 numerically, but a plain
    # string sort would put "-9" ahead of "-10"/"-100" and pick the wrong latest.
    for run_id in ("2026-08-05-9", "2026-08-05-10", "2026-08-05-100"):
        _write_history_run(root, run_id)

    runs = gen.runs_from_history_root(root, _baselines())

    assert [r["meta"]["run"] for r in runs] == [
        "2026-08-05-100", "2026-08-05-10", "2026-08-05-9",
    ]


def test_history_root_preserves_recorded_comparator_gate_failure(tmp_path):
    root = tmp_path / "runs"
    # Every overall_verdict matches its baseline, but the authoritative comparator
    # gate (meta.gate, the strict compare_verdict_baselines.py result) recorded a
    # failure. The dashboard must fail closed, not render the run HEALTHY.
    _write_history_run(root, "2026-08-05-33", meta={"gate": False})

    run = gen.runs_from_history_root(root, _baselines())[0]

    # run.gate agrees with meta.gate (no contradiction in data.json).
    assert run["gate"] is False
    assert run["meta"]["gate"] is False
    summary = gen._gate_summary(run["rows"], recorded_gate=run["meta"].get("gate"))
    assert summary["ok"] is False
    assert summary["short"] == "Failed"
    # The per-run landing page and the history gate pill reflect the failure.
    page = gen.build_run_index_html(run)
    assert "FAILED" in page and "HEALTHY" not in page
    assert 'class="pill fail"' in gen._history_gate_html(run)


def test_history_root_ignores_malformed_run_dirs(tmp_path):
    root = tmp_path / "runs"
    _write_history_run(root, "2026-08-05-33")
    # A stray non-conforming child (e.g. a leftover `source/` from a nested
    # --history-root layout) must not be enumerated as a run: without filtering it
    # would sort ahead of every dated id under reverse=True and show as "latest".
    (root / "source" / "waitcheck").mkdir(parents=True)
    (root / "not-a-run").mkdir()

    runs = gen.runs_from_history_root(root, _baselines())

    assert [r["meta"]["run"] for r in runs] == ["2026-08-05-33"]


def test_history_root_missing_report_has_no_report_rel(tmp_path):
    root = tmp_path / "runs"
    run_dir = _write_history_run(root, "2026-08-05-33")
    # Remove one case's report -> present False, and no dead link is emitted.
    (run_dir / "consan-clean" / "sanitizer_report.json").unlink()

    runs = gen.runs_from_history_root(root, _baselines())
    rows = runs[0]["rows"]
    assert rows["consan-clean"]["present"] is False
    assert rows["consan-clean"]["report_rel"] is None
    assert runs[0]["gate"] is False  # a missing report fails the gate closed


def test_build_html_emits_relative_report_links(tmp_path):
    root = tmp_path / "runs"
    _write_history_run(root, "2026-08-05-33")
    _write_history_run(root, "2026-08-04-22")
    runs = gen.runs_from_history_root(root, _baselines())

    html = gen.build_html(runs)

    # Latest-run table links each case to its raw JSON, relative to /sanitizers/.
    assert 'href="runs/2026-08-05-33/waitcheck/sanitizer_report.json"' in html
    # A "view raw report" link appears in the kernel-detail section.
    assert "view raw report" in html
    # The run area is linked from the latest-run header and the history table.
    assert 'href="runs/2026-08-05-33/"' in html
    assert 'href="runs/2026-08-04-22/"' in html
    # All emitted links are relative (no site-absolute or protocol-absolute hrefs
    # to the run area / reports).
    assert 'href="/runs/' not in html
    assert "https://runs/" not in html


def test_build_html_no_report_column_links_without_history():
    # results-dir / runs-root modes carry no rel/report_rel: the Report column is
    # present but renders an em dash, and no run-area links are emitted.
    rows = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-clean": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    runs = [gen._run_record({"run": "r1", "commit": "abc", "date": "d"}, rows)]
    html = gen.build_html(runs)
    assert "<th>Report</th>" in html
    assert "sanitizer_report.json" not in html
    assert "runs/" not in html


def test_build_run_index_html_lists_reports(tmp_path):
    root = tmp_path / "runs"
    _write_history_run(root, "2026-08-05-33")
    run = gen.runs_from_history_root(root, _baselines())[0]

    page = gen.build_run_index_html(run)

    assert "<!doctype html>" in page
    assert "2026-08-05-33" in page
    # Report links are case-local (the page lives inside runs/<id>/).
    assert 'href="waitcheck/sanitizer_report.json"' in page
    assert 'href="consan-racy/sanitizer_report.json"' in page
    # Links back to the workflow run and up to the dashboard.
    assert "/runs/99" in page
    assert 'href="../../"' in page


def test_build_run_index_html_missing_report_reads_incomplete(tmp_path):
    root = tmp_path / "runs"
    run_dir = _write_history_run(root, "2026-08-05-33")
    (run_dir / "consan-clean" / "sanitizer_report.json").unlink()
    run = gen.runs_from_history_root(root, _baselines())[0]

    page = gen.build_run_index_html(run)

    # A missing report must read as INCOMPLETE, not a misleading verdict mismatch.
    assert "INCOMPLETE \u2014 1/3 sanitizer report(s) are missing" in page
    assert "verdict mismatch" not in page


def test_main_history_root_writes_per_run_landing_pages(tmp_path, monkeypatch):
    baselines = _REPO_ROOT / "recipes" / "sanitizers" / "fixtures" / "expected" / "verdict_baselines.json"
    root = tmp_path / "runs"
    _write_history_run(root, "2026-08-05-33")
    _write_history_run(root, "2026-08-04-22")
    out = tmp_path / "dashboard"
    argv = [
        "gen_sanitizer_dashboard",
        "--history-root", str(root),
        "--keep", "30",
        "--baselines", str(baselines),
        "--out-dir", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert gen.main() == 0

    assert (out / "index.html").is_file()
    # A per-run landing page is written next to each retained run's reports.
    assert (out / "runs" / "2026-08-05-33" / "index.html").is_file()
    assert (out / "runs" / "2026-08-04-22" / "index.html").is_file()
    # The raw reports are co-located under the output tree so every emitted
    # runs/<id>/<case>/sanitizer_report.json link resolves even though the
    # history-root here is a separate directory from <out-dir>/runs.
    assert (out / "runs" / "2026-08-05-33" / "waitcheck" / "sanitizer_report.json").is_file()
    assert (out / "runs" / "2026-08-04-22" / "consan-racy" / "sanitizer_report.json").is_file()
    # The run manifest is carried too, so the published runs/<id>/ layout is complete.
    assert (out / "runs" / "2026-08-05-33" / "meta.json").is_file()
    # data.json mirrors the rel / report_rel fields for machine consumers.
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))
    assert data[0]["rel"] == "runs/2026-08-05-33"
    assert (
        data[0]["rows"]["waitcheck"]["report_rel"]
        == "runs/2026-08-05-33/waitcheck/sanitizer_report.json"
    )
    # gate stays a JSON boolean (not coerced to the string "True") for consumers.
    assert data[0]["meta"]["gate"] is True


def test_main_history_root_clears_stale_output_runs(tmp_path, monkeypatch):
    # When --history-root is a separate tree from <out-dir>/runs and the output is
    # reused, a run dropped from the history must not linger as stale published
    # output. main() clears out/runs before copying the current retained set.
    baselines = _REPO_ROOT / "recipes" / "sanitizers" / "fixtures" / "expected" / "verdict_baselines.json"
    root = tmp_path / "runs"
    _write_history_run(root, "2026-08-05-33")
    out = tmp_path / "dashboard"
    stale = out / "runs" / "2026-08-01-01" / "waitcheck"
    stale.mkdir(parents=True)
    (stale / "sanitizer_report.json").write_text("{}")
    argv = [
        "gen_sanitizer_dashboard",
        "--history-root", str(root),
        "--baselines", str(baselines),
        "--out-dir", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert gen.main() == 0

    assert not (out / "runs" / "2026-08-01-01").exists()
    assert (out / "runs" / "2026-08-05-33" / "waitcheck" / "sanitizer_report.json").is_file()


def test_main_history_root_nested_under_output_is_not_wiped(tmp_path, monkeypatch):
    # If --history-root nests beneath <out-dir>/runs, the stale-clear must NOT
    # rmtree runs_out (that would delete the source reports before they're copied).
    baselines = _REPO_ROOT / "recipes" / "sanitizers" / "fixtures" / "expected" / "verdict_baselines.json"
    out = tmp_path / "dashboard"
    root = out / "runs" / "source"  # nested beneath <out-dir>/runs
    _write_history_run(root, "2026-08-05-33")
    argv = [
        "gen_sanitizer_dashboard",
        "--history-root", str(root),
        "--baselines", str(baselines),
        "--out-dir", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert gen.main() == 0

    # Source reports survive (not wiped) and are still published under the output.
    assert (root / "2026-08-05-33" / "waitcheck" / "sanitizer_report.json").is_file()
    assert (out / "runs" / "2026-08-05-33" / "waitcheck" / "sanitizer_report.json").is_file()


def test_main_empty_history_root_publishes_placeholder(tmp_path, monkeypatch):
    # Mirrors the Pages empty-state invocation (pages.yml publish_placeholder).
    baselines = _REPO_ROOT / "recipes" / "sanitizers" / "fixtures" / "expected" / "verdict_baselines.json"
    out = tmp_path / "out"
    (tmp_path / "empty").mkdir()
    argv = [
        "gen_sanitizer_dashboard",
        "--history-root", str(tmp_path / "empty"),
        "--baselines", str(baselines),
        "--out-dir", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert gen.main() == 0
    assert "No sanitizer runs yet" in (out / "index.html").read_text(encoding="utf-8")


def test_main_fails_closed_on_missing_baselines(tmp_path, monkeypatch):
    # A missing/unreadable baselines file must not paint a false-healthy gate: main
    # exits non-zero instead of falling back to an empty baseline set.
    argv = [
        "gen_sanitizer_dashboard",
        "--results-dir", str(tmp_path / "incoming"),
        "--baselines", str(tmp_path / "missing.json"),
        "--out-dir", str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert gen.main() == 2
    assert not (tmp_path / "out").exists()


def _informational_report(reason: str) -> dict:
    return {
        "schema": "aorta.sanitizer_report/0.1", "target": "gfx950",
        "overall_verdict": "error", "execution_status": "error",
        "worklist": {
            "schema": "aorta.kernel_worklist/0.1", "requirement": "top_dispatch_count",
            "top_n": 1, "kernel_count": 1,
            "kernels": [{"identity": {"name": "gemm_f32_ss", "target": "gfx950"},
                         "total_time_ms": 0.0, "dispatch_count": 1, "sources": ["kernel_list"]}],
        },
        "checks": [{
            "sanitizer": "consan", "state": "error", "verdict": "error",
            "reason": reason, "returncode": None, "findings": [],
            "kernel_results": [], "coverage": [], "backend": {},
        }, {
            "sanitizer": "waitcheck_preflight", "state": "error", "verdict": "error",
            "reason": reason, "returncode": None, "findings": [],
            "kernel_results": [], "coverage": [], "backend": {},
        }],
    }


def test_survey_cases_from_informational_dir_folds_caller_supplied_cases(tmp_path):
    # Caller-supplied ConSan cases (#347) fold into the Tab 2 workload-survey shape:
    # observed-only (expected None / match True), the observation carries the
    # fail-closed reason, the provenance workload references #347, and report_rel
    # points at the co-published survey area when a run rel is supplied.
    root = tmp_path / "informational"
    (root / "consan-gemm").mkdir(parents=True)
    (root / "consan-gemm" / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    (root / "empty-case").mkdir()  # no report -> skipped
    # A structurally invalid best-effort report (valid JSON, but not an object)
    # must be skipped like an unreadable one, never crash the whole publication.
    (root / "list-case").mkdir()
    (root / "list-case" / "sanitizer_report.json").write_text("[]")
    (root / "scalar-case").mkdir()
    (root / "scalar-case" / "sanitizer_report.json").write_text("null")

    entries = gen.survey_cases_from_informational_dir(root, rel="runs/2026-08-05-33")
    assert [e["name"] for e in entries] == ["consan-gemm"]  # non-object reports skipped
    case = entries[0]
    assert case["cls"] == "survey"
    assert case["summary"]["expected"] is None
    assert case["summary"]["match"] is True
    assert case["summary"]["verdict"] == "error"
    assert "combined_hook_timeout" in case["summary"]["observation"]
    # kernel-group + sanitizer are parsed from the case name (both-sanitizer grouping)
    assert case["group"] == "gemm"
    assert case["sanitizer"] == "consan"
    # provenance is a copy-paste reproduce command, not an "experimental /
    # caller-supplied" label (#374 B): the user-facing workload/source is dropped.
    assert case["workload"] is None
    assert case["command"] == "aorta sweep run --recipe recipes/sanitizers/daily-consan-gemm.yaml"
    assert (
        case["report_rel"]
        == "runs/2026-08-05-33/survey/consan-gemm/sanitizer_report.json"
    )

    # Without a run rel (results-dir / runs-root modes) no dead link is emitted.
    no_rel = gen.survey_cases_from_informational_dir(root, rel=None)
    assert no_rel[0]["report_rel"] is None

    # A non-existent dir degrades to an empty list.
    assert gen.survey_cases_from_informational_dir(tmp_path / "does-not-exist") == []


def test_main_history_root_informational_folds_into_survey_tab(tmp_path, monkeypatch):
    # End-to-end: a history run plus a caller-supplied ConSan dir renders the case as
    # a Tab 2 workload-survey entry, co-publishes its report under the run's survey
    # area (so the raw-report link resolves), drops the old informational heading, and
    # threads the case into data.json's runs[0]["survey"].
    baselines = (
        _REPO_ROOT / "recipes" / "sanitizers" / "fixtures" / "expected" / "verdict_baselines.json"
    )
    root = tmp_path / "runs"
    _write_history_run(root, "2026-08-05-33")
    info = tmp_path / "informational"
    (info / "consan-gemm").mkdir(parents=True)
    (info / "consan-gemm" / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    out = tmp_path / "dashboard"
    argv = [
        "gen_sanitizer_dashboard",
        "--history-root", str(root),
        "--baselines", str(baselines),
        "--informational-results-dir", str(info),
        "--out-dir", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert gen.main() == 0

    html = (out / "index.html").read_text(encoding="utf-8")
    # The caller-supplied case renders in the survey tab (its kernel name + a raw
    # report link into the co-published survey area).
    assert "gemm_f32_ss" in html
    assert 'href="runs/2026-08-05-33/survey/consan-gemm/sanitizer_report.json"' in html
    # The old informational section heading is gone.
    assert "Informational \u00b7 caller-supplied code objects (non-gating)" not in html
    assert "Informational · caller-supplied code objects" not in html
    # The report is co-published under the run's survey area so the link resolves.
    assert (
        out / "runs" / "2026-08-05-33" / "survey" / "consan-gemm" / "sanitizer_report.json"
    ).is_file()
    # data.json threads the case into the latest run's survey list.
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))
    survey = data[0]["survey"]
    assert [c["name"] for c in survey] == ["consan-gemm"]
    assert survey[0]["cls"] == "survey"
    assert survey[0]["summary"]["expected"] is None


def test_survey_recipe_for_uses_real_recipe_not_derived_guess():
    # Most survey cases map to daily-<case>.yaml, but the gemm waitcheck survey uses
    # a dedicated object recipe -- the gated Tab-1 daily-waitcheck-gemm guardrail is
    # never reused, so the displayed reproduce command must not point at it.
    assert gen._survey_recipe_for("consan-gemm") == "daily-consan-gemm"
    assert gen._survey_recipe_for("waitcheck-lds-dispatch") == "daily-waitcheck-lds-dispatch"
    assert gen._survey_recipe_for("waitcheck-tiny") == "daily-waitcheck-tiny"
    assert gen._survey_recipe_for("waitcheck-gemm") == "daily-waitcheck-gemm-object"
    assert gen._survey_recipe_for("waitcheck-gemm") != "daily-waitcheck-gemm"


def test_verdict_chip_html_colors_by_verdict():
    # #374 D: solid color-coded verdict chips keyed on the verdict string.
    assert gen._verdict_chip_html("error") == '<span class="vchip err">error</span>'
    assert gen._verdict_chip_html("fail") == '<span class="vchip err">fail</span>'
    assert gen._verdict_chip_html("warn") == '<span class="vchip warn">warn</span>'
    assert gen._verdict_chip_html("pass") == '<span class="vchip pass">pass</span>'
    assert (
        gen._verdict_chip_html("not_checked") == '<span class="vchip neutral">not_checked</span>'
    )
    assert gen._verdict_chip_html("\u2014") == '<span class="vchip neutral">\u2014</span>'


def test_survey_intro_is_readable_and_drops_experimental_framing():
    # #374 A/B: the intro is body-size readable copy (not 12px muted fine print) and
    # frames the tab as real kernels under both sanitizers, dropping the old
    # "experimental" / "caller-supplied code objects" framing from user-facing copy.
    html = gen.build_html([_healthy_guardrail_run()], survey=[])
    assert 'class="survey-intro"' in html
    assert ".survey-intro { font-size:14px" in html  # readable body size
    assert "real GPU kernels" in html
    assert "waitcheck" in html and "ConSan" in html
    assert "No expected-behavior comparison on this tab" in html
    assert "experimental" not in html.lower()
    assert "caller-supplied" not in html.lower()


def test_survey_intro_qualifies_the_both_sanitizers_claim():
    # Review (#374): the absolute "shown under both sanitizers" claim is false on the
    # supported degraded path (an absent/unreadable report skips a sanitizer). The
    # HTML intro + MD mirror must qualify it so the UI never claims both ran when only
    # one report exists.
    html = gen.build_html([_healthy_guardrail_run()], survey=[])
    # the old absolute claim is gone; the qualified degraded-path copy is present
    assert "shown under <b>both</b> sanitizers." not in html
    assert "only the sanitizer(s) that actually ran appear" in html
    md = gen.build_summary_md([_healthy_guardrail_run()], survey=[])
    assert "shown under both." not in md
    assert "only the sanitizer(s) that ran appear" in md


def _errored_report_with_partial_findings(reason: str = "combined_hook_timeout") -> dict:
    # An errored ConSan run that still emitted a partial finding before aborting
    # (e.g. coverage-incomplete): overall_verdict=error AND a finding is present.
    partial = {
        "sanitizer": "consan", "severity": "race", "code": "data_race",
        "message": "[rocjitsu-dbi-hooks] partial conflict observed before hook timeout",
        "kernel_name": None, "code_object": None, "entry_offset": None, "metadata": {},
    }
    return {
        "schema": "aorta.sanitizer_report/0.1", "target": "gfx950",
        "overall_verdict": "error", "execution_status": "error",
        "worklist": {
            "schema": "aorta.kernel_worklist/0.1", "requirement": "top_dispatch_count",
            "top_n": 1, "kernel_count": 1,
            "kernels": [{"identity": {"name": "gemm_f32_ss", "target": "gfx950"},
                         "total_time_ms": 0.0, "dispatch_count": 1, "sources": ["kernel_list"]}],
        },
        "checks": [{
            "sanitizer": "consan", "state": "error", "verdict": "error",
            "reason": reason, "returncode": None, "findings": [partial],
            "kernel_results": [], "coverage": [], "backend": {},
        }],
    }


def test_survey_message_error_reason_takes_precedence_over_partial_findings():
    # Review (#374): an errored report that also carries partial findings must show
    # its error REASON inline, not a Finding that hides it. warn/fail cases keep
    # finding-first. Enforced in the shared parts helper + both HTML and MD twins.
    errored = gen.summarize_case(_errored_report_with_partial_findings(), None)
    assert errored["verdict"] == "error" and errored["findings"] == 1  # partial finding present
    assert gen._survey_message_parts(errored) == ("Reason", "combined_hook_timeout")
    assert gen._survey_message_html(errored) == (
        '<div class="survey-msg">Reason: combined_hook_timeout</div>'
    )
    assert gen._survey_message_md(errored) == "Reason: `combined_hook_timeout`"

    # a warn case with a finding still leads with the finding (unchanged behavior)
    warn = gen.summarize_case(_waitcheck_report(), None)
    assert gen._survey_message_parts(warn)[0] == "Finding"
    assert gen._survey_message_html(warn).startswith('<div class="survey-msg">Finding:')
    assert gen._survey_message_md(warn).startswith("Finding: `")

    # the summary-table note also surfaces the errored group's reason ahead of a
    # sibling sanitizer's finding code (sweep of the same precedence class)
    entries = gen.survey_cases_from_spec(
        {"cases": [
            {"name": "waitcheck-obj", "group": "obj", "sanitizer": "waitcheck",
             "report": _waitcheck_report()},
            {"name": "consan-obj", "group": "obj", "sanitizer": "consan",
             "report": _errored_report_with_partial_findings()},
        ]}
    )
    groups = gen._group_survey_entries(entries)
    assert gen._survey_group_note(groups[0][1]) == "combined_hook_timeout"
    # rendered end-to-end: the errored reason shows on the page, gate stays HEALTHY
    html = gen.build_html([_healthy_guardrail_run()], survey=entries)
    assert "Reason: combined_hook_timeout" in html
    assert "HEALTHY" in html
    assert "REGRESSION" not in html and "Regression" not in html


def test_survey_informational_dir_isolates_malformed_nested_reports(tmp_path):
    # Review (#374): a top-level dict with a malformed NESTED shape (e.g.
    # {"checks": null}) passes the isinstance guard but makes the reduction raise
    # mid-iteration. One broken best-effort report must not abort the whole
    # non-gating publication -- the case is skipped (fails closed), good cases render.
    root = tmp_path / "informational"
    (root / "consan-bad").mkdir(parents=True)
    (root / "consan-bad" / "sanitizer_report.json").write_text(json.dumps({"checks": None}))
    (root / "consan-bad2").mkdir(parents=True)
    (root / "consan-bad2" / "sanitizer_report.json").write_text(
        json.dumps({"checks": [], "worklist": [1, 2, 3]})  # non-dict worklist
    )
    (root / "consan-good").mkdir(parents=True)
    (root / "consan-good" / "sanitizer_report.json").write_text(
        json.dumps(_consan_clean_report())
    )

    entries = gen.survey_cases_from_informational_dir(root, rel="runs/2026-08-05-33")
    # the two malformed cases are dropped; the healthy one still renders
    assert [e["name"] for e in entries] == ["consan-good"]
    # and it still renders end-to-end without raising
    html = gen.build_html([_healthy_guardrail_run()], survey=entries)
    assert "HEALTHY" in html


def test_survey_informational_groups_both_sanitizers_with_chips_and_repro(tmp_path):
    # #374 C/D/E: the same kernel scanned by both sanitizers renders under ONE
    # kernel-group heading with a sanitizer sub-block each, color-coded verdict chips,
    # a copy-paste reproduce command, and the actual message inline on the page.
    root = tmp_path / "informational"
    (root / "consan-gemm").mkdir(parents=True)
    (root / "consan-gemm" / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    (root / "waitcheck-gemm").mkdir(parents=True)
    (root / "waitcheck-gemm" / "sanitizer_report.json").write_text(
        json.dumps(_waitcheck_report())
    )

    entries = gen.survey_cases_from_informational_dir(root, rel="runs/2026-08-05-33")
    assert {e["group"] for e in entries} == {"gemm"}
    assert {e["sanitizer"] for e in entries} == {"consan", "waitcheck"}
    by_name = {e["name"]: e for e in entries}
    # the waitcheck gemm survey case's report comes from its own survey dir and its
    # reproduce command points at the dedicated object recipe (not the gated one).
    assert by_name["waitcheck-gemm"]["command"] == (
        "aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gemm-object.yaml"
    )
    assert (
        by_name["waitcheck-gemm"]["report_rel"]
        == "runs/2026-08-05-33/survey/waitcheck-gemm/sanitizer_report.json"
    )

    html = gen.build_html([_healthy_guardrail_run()], survey=entries)
    # one kernel-group heading, both sanitizer sub-blocks
    assert 'class="survey-group">gemm</h3>' in html
    assert "waitcheck (static wait-count scan)" in html
    assert "ConSan (dynamic data-race check)" in html
    # solid color-coded chips: consan error -> red, waitcheck warn -> amber
    assert '<span class="vchip err">error</span>' in html
    assert '<span class="vchip warn">warn</span>' in html
    # per-case copy-paste reproduce command reflects the ACTUAL recipe: the gemm
    # waitcheck survey uses its dedicated object recipe, NOT the gated Tab-1
    # daily-waitcheck-gemm guardrail (which is never reused for the survey).
    assert "aorta sweep run --recipe recipes/sanitizers/daily-consan-gemm.yaml" in html
    assert (
        "aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gemm-object.yaml"
        in html
    )
    assert "recipes/sanitizers/daily-waitcheck-gemm.yaml" not in html
    # the actual message is inline on the page (not only behind the raw-report link):
    # the errored ConSan case shows its reason; the waitcheck case shows a finding.
    assert "Reason: combined_hook_timeout" in html
    assert "Finding:" in html
    # gate stays healthy; no regression vocabulary leaks from the survey
    assert "HEALTHY" in html
    assert "REGRESSION" not in html and "Regression" not in html


def _waitcheck_pass_report() -> dict:
    report = _waitcheck_report()
    report["overall_verdict"] = "pass"
    report["checks"][0]["verdict"] = "pass"
    report["checks"][0]["findings"] = []
    report["checks"][0]["kernel_results"][0]["verdict"] = "pass"
    report["checks"][0]["kernel_results"][0]["findings"] = []
    return report


def _survey_mixed_entries() -> list[dict]:
    # Three kernel groups spanning the full verdict spectrum, mirroring the nightly
    # survey layout: gemm (waitcheck warn + ConSan fail), obj (ConSan error only ->
    # waitcheck absent), tiny (both pass).
    return gen.survey_cases_from_spec(
        {"cases": [
            {"name": "waitcheck-gemm", "group": "gemm", "sanitizer": "waitcheck",
             "report": _waitcheck_report()},
            {"name": "consan-gemm", "group": "gemm", "sanitizer": "consan",
             "report": _consan_racy_report()},
            {"name": "consan-obj", "group": "obj", "sanitizer": "consan",
             "report": _informational_report("combined_hook_timeout")},
            {"name": "waitcheck-tiny", "group": "tiny", "sanitizer": "waitcheck",
             "report": _waitcheck_pass_report()},
            {"name": "consan-tiny", "group": "tiny", "sanitizer": "consan",
             "report": _consan_clean_report()},
        ]}
    )


def test_survey_summary_stats_counts_kernels_runs_and_verdicts():
    # #374 summary roll-up: the aggregation counts kernel groups, sanitizer runs
    # (present reports only), and a per-verdict breakdown that sums to the run count.
    groups = gen._group_survey_entries(_survey_mixed_entries())
    stats = gen._survey_summary_stats(groups)
    assert stats["kernels"] == 3
    assert stats["runs"] == 5
    assert stats["verdicts"] == {
        "pass": 2, "warn": 1, "fail": 1, "error": 1, "not_checked": 0
    }
    # the breakdown sums exactly to the sanitizer-run count (honest arithmetic)
    assert sum(stats["verdicts"].values()) == stats["runs"]
    # headline is readable and omits the zero bucket
    headline = gen._survey_headline(stats)
    assert headline == (
        "Surveyed 3 kernels across 5 sanitizer runs "
        "\u2014 2 pass \u00b7 1 warn \u00b7 1 fail \u00b7 1 error"
    )
    assert "not_checked" not in headline


def test_survey_summary_stats_absent_sanitizer_is_not_a_run():
    # A group whose sanitizer has no report (didn't run / non-GPU host) is not
    # counted as a sanitizer run and never fabricates a verdict bucket.
    entries = gen.survey_cases_from_spec(
        {"cases": [
            {"name": "waitcheck-ghost", "group": "ghost", "sanitizer": "waitcheck"},
            {"name": "consan-real", "group": "real", "sanitizer": "consan",
             "report": _consan_clean_report()},
        ]}
    )
    stats = gen._survey_summary_stats(gen._group_survey_entries(entries))
    assert stats == {
        "kernels": 2, "runs": 1,
        "verdicts": {"pass": 1, "warn": 0, "fail": 0, "error": 0, "not_checked": 0},
    }


def test_survey_summary_table_renders_chips_emdash_and_gate_stays_healthy():
    # #374: the roll-up table renders one row per kernel with the solid-color verdict
    # chips per sanitizer, an em dash where a sanitizer did not run, the headline
    # stat, and never flips the HEALTHY gate or leaks regression vocabulary.
    entries = _survey_mixed_entries()
    groups = gen._group_survey_entries(entries)
    table = gen._survey_summary_table_html(groups)

    # readable headline stat (not tiny muted text) with the accurate breakdown
    assert 'class="survey-headline"' in table
    assert "Surveyed 3 kernels across 5 sanitizer runs" in table
    for part in ("2 pass", "1 warn", "1 fail", "1 error"):
        assert part in table
    # one row per kernel group with a stable header
    assert "<th>Kernel</th><th>waitcheck</th><th>ConSan</th>" in table
    # solid-color chips for every observed verdict (error/fail red, warn amber, pass green)
    assert '<span class="vchip warn">warn</span>' in table
    assert '<span class="vchip err">fail</span>' in table
    assert '<span class="vchip err">error</span>' in table
    assert '<span class="vchip pass">pass</span>' in table
    # the obj group's waitcheck did not run -> em dash cell, never a fake verdict
    assert "<td>&mdash;</td>" in table

    # rendered on the page: gate stays HEALTHY, no regression vocabulary
    html = gen.build_html([_healthy_guardrail_run()], survey=entries)
    assert 'class="survey-headline"' in html
    assert "HEALTHY — 3/3 sanitizer outcomes match their baselines" in html
    assert "REGRESSION" not in html and "Regression" not in html
    assert "Unexpected outcome" not in html and "Mismatch" not in html


def test_survey_summary_empty_survey_renders_gracefully():
    # Empty survey: no roll-up table (empty string), the empty-state note instead,
    # and the page still renders healthy without a 0-row table.
    assert gen._survey_summary_table_html([]) == ""
    assert gen._survey_summary_md([]) == []
    html = gen.build_html([_healthy_guardrail_run()], survey=[])
    assert "No workload-survey kernels in this run." in html
    assert 'class="survey-summary"' not in html
    assert "HEALTHY" in html


def test_survey_summary_md_mirrors_headline_and_table():
    # #374: the GitHub job-summary markdown mirrors the roll-up (headline + table).
    md = gen.build_summary_md([_healthy_guardrail_run()], survey=_survey_mixed_entries())
    assert "Surveyed 3 kernels across 5 sanitizer runs" in md
    assert "| Kernel | waitcheck | ConSan | Findings | Note |" in md
    # a kernel row with both sanitizer verdicts as markdown code spans
    assert "| gemm | `warn` | `fail` |" in md
    # the obj group's absent waitcheck renders as an em dash, not a fake verdict
    assert f"| obj | {gen._DASH} | `error` |" in md
    # observed-only: no regression vocabulary in the mirror
    assert "REGRESSION" not in md and "Mismatch" not in md


def test_build_html_has_no_informational_section():
    # The separate informational section is fully removed: a normally-rendered page
    # never carries its heading, and build_html no longer accepts an informational arg.
    html = gen.build_html([_healthy_guardrail_run()])
    assert "Informational · caller-supplied code objects" not in html
    md = gen.build_summary_md([_healthy_guardrail_run()])
    assert "Informational · caller-supplied code objects" not in md


# --- #367: two-tab (guardrail + workload survey) kernel-level dashboard ---


def test_summarize_case_survey_is_observational_with_primary_and_observation():
    # expected=None marks a survey (observed-only) case: match is True (no baseline
    # comparison) and the folded primary/observation fields are surfaced for both tabs.
    case = gen.summarize_case(_consan_racy_report(), None)
    assert case["expected"] is None
    assert case["match"] is True
    assert case["primary"] == {
        "sanitizer": "consan", "verdict": "fail", "reason": None, "preflight": None
    }
    assert case["observation"].startswith("consan fail")
    # a guardrail case (expected set) keeps its baseline match semantics
    guard = gen.summarize_case(_consan_racy_report(), "pass")
    assert guard["match"] is False and guard["expected"] == "pass"
    # a missing report degrades to an explicit, observational placeholder; with no
    # expectation it is NOT a mismatch (survey contract), while a missing guardrail
    # report (non-null expected) stays a mismatch so the gate fails closed.
    missing = gen.summarize_case(None, None)
    assert missing["present"] is False and missing["observation"] == "report missing"
    assert missing["match"] is True
    assert gen.summarize_case(None, "pass")["match"] is False
    # A structurally invalid report (``_load`` admits any JSON value, so a list or
    # scalar can reach here) is treated as absent instead of crashing on ``.get``.
    for bad in ([], "oops", 0, False):
        degraded = gen.summarize_case(bad, "pass")
        assert degraded["present"] is False and degraded["match"] is False


def test_primary_prefers_last_nonpreflight_and_captures_preflight():
    report = _informational_report("consan_strict_load_rejection")
    primary = gen._primary_checks(report["checks"])
    assert primary["sanitizer"] == "consan"
    assert primary["verdict"] == "error"
    assert primary["reason"] == "consan_strict_load_rejection"
    # the waitcheck_preflight check is captured separately, not as the primary
    assert primary["preflight"] == "error"


def test_run_record_tags_rows_as_guardrail():
    rows = {c: gen.summarize_case(_waitcheck_report(), "warn") for c, *_ in gen.CASES}
    rec = gen._run_record({"run": "r"}, rows)
    assert all(r["cls"] == "guardrail" for r in rec["rows"].values())


def test_survey_cases_from_spec_classifies_and_threads_links(tmp_path):
    (tmp_path / "wc.json").write_text(json.dumps(_waitcheck_report()))
    spec = {
        "cases": [
            {
                "name": "top5", "label": "top-5 f32 GEMM \u00b7 ConSan",
                "backend": "consan (dynamic)", "workload": "internal:gemm_top5",
                "report_rel": "runs/x/survey/top5/sanitizer_report.json",
                "report": _consan_racy_report(),
            },
            {"name": "wc", "label": "waitcheck survey", "report_path": "wc.json"},
            {"name": "missing", "label": "absent case", "report_path": "does-not-exist.json"},
        ]
    }
    entries = gen.survey_cases_from_spec(spec, base_dir=tmp_path)

    assert [e["cls"] for e in entries] == ["survey", "survey", "survey"]
    top = entries[0]
    # observed-only: expected None, match True -> never a regression
    assert top["summary"]["expected"] is None and top["summary"]["match"] is True
    assert top["summary"]["verdict"] == "fail"
    assert top["report_rel"] == "runs/x/survey/top5/sanitizer_report.json"
    assert top["workload"] == "internal:gemm_top5"
    assert top["summary"]["observation"].startswith("consan fail")
    # report_path is loaded relative to base_dir
    assert entries[1]["summary"]["verdict"] == "warn"
    # a missing report degrades gracefully: still an entry, no dead link
    assert entries[2]["summary"]["present"] is False
    assert entries[2]["report_rel"] is None
    # a bare list (no "cases" wrapper) is accepted too
    assert gen.survey_cases_from_spec([{"name": "x", "report": _waitcheck_report()}])[0][
        "cls"
    ] == "survey"


def test_survey_cases_from_spec_degrades_on_malformed_specs():
    # A malformed wrapper value must not raise (the CLI promises the survey tab
    # degrades to its empty-state note rather than aborting the whole render).
    assert gen.survey_cases_from_spec({"cases": 1}) == []
    assert gen.survey_cases_from_spec({"cases": "nope"}) == []
    assert gen.survey_cases_from_spec({}) == []
    # A non-string report_path (e.g. a numeric JSON value) is ignored instead of
    # raising in Path(); the case still renders, just as an absent report.
    entries = gen.survey_cases_from_spec(
        {"cases": [{"name": "bad", "label": "bad path", "report_path": 5}]}
    )
    assert len(entries) == 1
    assert entries[0]["summary"]["present"] is False
    assert entries[0]["report_rel"] is None


def test_survey_report_rel_rejects_unsafe_links():
    # report_rel is untrusted caller JSON: a URL scheme, an absolute path, or a
    # protocol-relative link must be dropped (HTML-escaping does not make a URL
    # safe), while a plain relative path is preserved.
    for unsafe in (
        "javascript:alert(1)",
        "/etc/passwd",
        "//evil.example/x",
        "\\\\unc\\share",
        "data:text/html,<script>1</script>",
        " runs/x/report.json",  # leading whitespace browsers would strip
        "../runs/x/report.json",  # parent-directory traversal
        "a/../../b/report.json",  # traversal buried mid-path
    ):
        spec = {"cases": [{"name": "s", "report_rel": unsafe, "report": _consan_racy_report()}]}
        entry = gen.survey_cases_from_spec(spec)[0]
        assert entry["report_rel"] is None, unsafe

    safe = "runs/2026-08-05-33/survey/top5/sanitizer_report.json"
    spec = {"cases": [{"name": "s", "report_rel": safe, "report": _consan_racy_report()}]}
    assert gen.survey_cases_from_spec(spec)[0]["report_rel"] == safe

    # An unsafe report_rel must never reach the rendered HTML as an href.
    bad = gen.survey_cases_from_spec(
        {"cases": [{"name": "s", "label": "xss", "report_rel": "javascript:alert(1)",
                    "report": _consan_racy_report()}]}
    )
    html = gen.build_html([_healthy_guardrail_run()], survey=bad)
    assert "javascript:alert(1)" not in html


def test_survey_report_path_rejects_traversal_and_absolute(tmp_path):
    # report_path is untrusted caller JSON used to read a file off disk. It must
    # be a validated relative path beneath base_dir: an absolute path or ``..``
    # traversal must NOT read the file (else a spec could pull JSON from outside
    # the spec dir and serialize it into the public dashboard). The case still
    # renders, just as an absent report.
    outside = tmp_path / "secret.json"
    outside.write_text(json.dumps(_waitcheck_report()))
    base = tmp_path / "spec_dir"
    base.mkdir()
    (base / "ok.json").write_text(json.dumps(_waitcheck_report()))

    for evil in ("../secret.json", str(outside), "a/../../secret.json"):
        spec = {"cases": [{"name": "s", "label": "evil", "report_path": evil}]}
        entry = gen.survey_cases_from_spec(spec, base_dir=base)[0]
        assert entry["summary"]["present"] is False, evil
        assert entry["report_rel"] is None, evil

    # A validated relative path beneath base_dir still loads normally.
    ok = gen.survey_cases_from_spec(
        {"cases": [{"name": "s", "label": "ok", "report_path": "ok.json"}]}, base_dir=base
    )[0]
    assert ok["summary"]["present"] is True
    assert ok["summary"]["verdict"] == "warn"


def test_build_html_tabs_have_keyboard_focus_style():
    # The visually-hidden radios (opacity:0) must project a focus indicator onto
    # their labels so keyboard users can see which tab is focused.
    html = gen.build_html([_healthy_guardrail_run()])
    assert ':focus-visible ~ .tabbar label[for="tab-guardrails"]' in html
    assert ':focus-visible ~ .tabbar label[for="tab-survey"]' in html


def _healthy_guardrail_run() -> dict:
    rows = {c: gen.summarize_case(_waitcheck_report(), "warn") for c, *_ in gen.CASES}
    return gen._run_record(
        {"run": "r1", "commit": "abc", "date": "d", "gpu": "gfx950"}, rows
    )


def test_survey_execution_status_renders_neutral_not_health_colored():
    # An observed-only survey error must not be health-colored on Tab 2. The
    # committed ConSan survey reports carry execution_status="error"; the meta
    # line must render it with the neutral `observed` style, never the red
    # `execution bad` class the guardrail tab uses for a broken run.
    err = {
        "schema": "aorta.sanitizer_report/0.1", "target": "gfx950",
        "overall_verdict": "error", "execution_status": "error",
        "worklist": {
            "schema": "aorta.kernel_worklist/0.1", "requirement": "top_dispatch_count",
            "top_n": 1, "kernel_count": 1,
            "kernels": [{"identity": {"name": "k", "target": "gfx950"},
                         "total_time_ms": 0.0, "dispatch_count": 1, "sources": ["s"]}],
        },
        "checks": [{"sanitizer": "consan", "state": "error", "verdict": "error",
                    "reason": None, "returncode": None, "findings": [],
                    "kernel_results": [], "coverage": [], "backend": {}}],
    }
    survey = gen.survey_cases_from_spec({"cases": [{"name": "s", "label": "obs", "report": err}]})
    # Only non-complete execution on the page is the survey one (guardrail runs
    # are complete), so a neutral survey render means no `execution bad` at all.
    html = gen.build_html([_healthy_guardrail_run()], survey=survey)
    assert "execution bad" not in html
    # execution status stays neutral (never health-colored) on the survey tab
    assert '<span class="observed">error</span>' in html
    # but the observed *verdict* is now a solid color-coded chip (#374 D): an error
    # verdict is a solid-red chip. This is descriptive only -- the gate stays healthy.
    assert '<span class="vchip err">error</span>' in html
    assert "HEALTHY — 3/3 sanitizer outcomes match their baselines" in html
    assert "REGRESSION" not in html and "Regression" not in html

    # Unit: the survey meta line neutralizes execution; the guardrail one colors it.
    row = survey[0]["summary"]
    assert "execution bad" not in gen._meta_line_html(row, observed=True)
    assert "execution bad" in gen._meta_line_html(row, observed=False)


def test_build_html_renders_two_self_contained_tabs():
    survey = gen.survey_cases_from_spec(
        {"cases": [{"name": "top5", "label": "top-5 GEMM survey",
                    "backend": "consan (dynamic)", "report": _consan_racy_report()}]}
    )
    html = gen.build_html([_healthy_guardrail_run()], survey=survey)

    # both tabs, radio-toggled, guardrails default-selected
    assert "id=tab-guardrails class=tabradio checked" in html
    assert "id=tab-survey class=tabradio" in html
    assert "Expected behavior (guardrails)" in html
    assert "Workload survey (observed-only)" in html
    # self-contained: no external JS/CSS, no CDN, no protocol-absolute links
    assert "<script" not in html.lower()
    assert "cdn" not in html.lower()
    assert "http://" not in html and "https://" not in html
    # explicit observed-only note on the survey tab
    assert "No expected-behavior comparison on this tab" in html


def test_build_html_survey_fail_is_color_coded_but_never_a_regression():
    # Guardrails are all healthy warn; the survey case observed a fail. The survey
    # fail must render as a solid color-coded (red) verdict chip (#374 D) and must
    # NOT turn the gate red or be labelled a regression / mismatch / unexpected.
    survey = gen.survey_cases_from_spec(
        {"cases": [{"name": "s", "label": "survey fail case",
                    "report": _consan_racy_report()}]}
    )
    html = gen.build_html([_healthy_guardrail_run()], survey=survey)

    assert "HEALTHY — 3/3 sanitizer outcomes match their baselines" in html
    # the fail is a solid-red verdict chip (color-coded), never the neutral grey chip
    assert '<span class="vchip err">fail</span>' in html
    assert '<span class="observed">fail</span>' not in html
    # none of the guardrail regression vocabulary leaks in from the survey, and the
    # gate banner stays healthy (a survey fail/error is observational, never a gate).
    assert "REGRESSION" not in html and "Regression" not in html
    assert "Unexpected outcome" not in html and "Mismatch" not in html
    # no baseline/expected column on the survey tab (no "expected <verdict>" phrasing
    # is emitted for the survey case; that phrasing is guardrail-only)
    assert "survey fail case" in html
    # observation summary + inline message are present on the tab
    assert "Observation:" in html
    assert "Finding:" in html


def test_build_html_survey_report_link_and_graceful_absence():
    survey = gen.survey_cases_from_spec(
        {"cases": [
            {"name": "linked", "label": "linked survey",
             "report_rel": "runs/x/survey/linked/sanitizer_report.json",
             "report": _consan_racy_report()},
            {"name": "nolink", "label": "nolink survey", "report": _waitcheck_report()},
        ]}
    )
    html = gen.build_html([_healthy_guardrail_run()], survey=survey)

    # the linked survey case drills down to its raw report; the unlinked one still
    # renders (degrades to no link). Guardrail rows carry no report_rel here, so the
    # only "view raw report" link comes from the linked survey case.
    assert 'href="runs/x/survey/linked/sanitizer_report.json"' in html
    assert html.count("view raw report") == 1
    assert "linked survey" in html and "nolink survey" in html


def test_kernel_tables_html_links_each_row_to_report():
    # #367 per-row acceptance: each kernel row links to the case's raw report via a
    # Report column, and degrades to an em dash (never a dead/unsafe link) when the
    # case carries no safe link.
    row = gen.summarize_case(_waitcheck_report(), "warn")
    assert len(row["kernels"]) == 1
    rel = "runs/2026-08-05-33/waitcheck/sanitizer_report.json"

    linked = gen._kernel_tables_html(row, report_rel=rel)
    assert "<th>Report</th>" in linked
    assert linked.count(f'<a href="{rel}">report</a>') == len(row["kernels"])

    # absent link -> Report column present but each row shows an em dash, no link
    absent = gen._kernel_tables_html(row, report_rel=None)
    assert "<th>Report</th>" in absent
    assert "sanitizer_report.json" not in absent
    assert ">report</a>" not in absent

    # unsafe caller value is rejected in the shared helper too (defense in depth)
    unsafe = gen._kernel_tables_html(row, report_rel="javascript:alert(1)")
    assert "javascript:alert(1)" not in unsafe
    assert ">report</a>" not in unsafe

    # empty-kernel case spans the full (now 7-column) row
    empty = gen._kernel_tables_html({**row, "kernels": []}, report_rel=rel)
    assert "colspan=7>no kernels selected" in empty


def test_build_html_kernel_rows_link_reports_on_both_tabs(tmp_path):
    root = tmp_path / "runs"
    _write_history_run(root, "2026-08-05-33")
    runs = gen.runs_from_history_root(root, _baselines())
    survey = gen.survey_cases_from_spec(
        {"cases": [{"name": "s", "label": "survey linked",
                    "report_rel": "runs/x/survey/s/sanitizer_report.json",
                    "report": _consan_racy_report()}]}
    )
    html = gen.build_html(runs, survey=survey)

    # Guardrail kernel-detail row links to its case report (in addition to the
    # latest-run table's Report cell) -> the waitcheck report link now appears
    # twice: the latest-run table cell and the per-kernel-row Report column.
    wc = 'href="runs/2026-08-05-33/waitcheck/sanitizer_report.json">report</a>'
    assert html.count(wc) == 2
    # Survey kernel row links to the survey case's report_rel (the heading uses the
    # distinct "view raw report" text, so a "report" link here is the per-row one).
    assert 'href="runs/x/survey/s/sanitizer_report.json">report</a>' in html
    # no dead/unsafe links leaked in
    assert 'href="/runs/' not in html and "javascript:" not in html


def test_build_html_empty_survey_shows_placeholder_note():
    html = gen.build_html([_healthy_guardrail_run()], survey=[])
    assert "Workload survey (observed-only)" in html
    assert "No workload-survey kernels in this run." in html


def test_build_summary_md_mirrors_two_tab_split():
    survey = gen.survey_cases_from_spec(
        {"cases": [{"name": "top5", "label": "top-5 survey",
                    "workload": "internal:gemm_top5", "report": _consan_racy_report()}]}
    )
    rows = {c: gen.summarize_case(_waitcheck_report(), "warn") for c, *_ in gen.CASES}
    run = gen._run_record({"run": "r1", "commit": "abc", "date": "d"}, rows)
    md = gen.build_summary_md([run], survey=survey)

    # guardrail section stays labelled and keeps the substring existing tests rely on
    assert "Expected behavior (guardrails) \u00b7 Kernel details" in md
    assert "Kernel details" in md
    # survey section present, observed-only, non-gating
    assert "## Workload survey (observed-only)" in md
    assert "no expected-behavior comparison on this tab" in md.lower()
    assert "observed `fail`" in md
    assert "source `internal:gemm_top5`" in md
    # the survey fail never flips the gate
    assert "**HEALTHY**" in md
    assert "**REGRESSION**" not in md


def test_main_survey_flows_into_html_and_data_json(tmp_path, monkeypatch):
    baselines = (
        _REPO_ROOT / "recipes" / "sanitizers" / "fixtures" / "expected" / "verdict_baselines.json"
    )
    incoming = tmp_path / "incoming"
    for case, report in (
        ("waitcheck", _waitcheck_report()),
        ("consan-clean", _consan_clean_report()),
        ("consan-racy", _consan_racy_report()),
    ):
        (incoming / case).mkdir(parents=True)
        (incoming / case / "sanitizer_report.json").write_text(json.dumps(report))

    survey_spec = tmp_path / "survey.json"
    survey_spec.write_text(json.dumps({"cases": [
        {"name": "top5", "label": "top-5 GEMM survey", "backend": "consan (dynamic)",
         "workload": "internal:gemm_top5",
         "report_rel": "survey/top5/sanitizer_report.json",
         "report": _consan_racy_report()},
    ]}))
    out = tmp_path / "out"
    argv = [
        "gen_sanitizer_dashboard",
        "--results-dir", str(incoming),
        "--baselines", str(baselines),
        "--survey", str(survey_spec),
        "--out-dir", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert gen.main() == 0

    html = (out / "index.html").read_text(encoding="utf-8")
    assert "Workload survey (observed-only)" in html and "top-5 GEMM survey" in html
    assert 'href="survey/top5/sanitizer_report.json"' in html

    data = json.loads((out / "data.json").read_text(encoding="utf-8"))
    # guardrail/survey case-class split is mirrored into data.json (additive fields)
    assert data[0]["rows"]["waitcheck"]["cls"] == "guardrail"
    survey = data[0]["survey"]
    assert len(survey) == 1 and survey[0]["cls"] == "survey"
    assert survey[0]["summary"]["expected"] is None
    assert survey[0]["report_rel"] == "survey/top5/sanitizer_report.json"

    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Workload survey (observed-only)" in md


def test_main_without_survey_omits_survey_rows_but_renders_tab(tmp_path, monkeypatch):
    # The survey flag is additive: existing invocations (no --survey) still work and
    # render Tab 2 with its empty-state note; data.json carries an empty survey list.
    baselines = (
        _REPO_ROOT / "recipes" / "sanitizers" / "fixtures" / "expected" / "verdict_baselines.json"
    )
    incoming = tmp_path / "incoming"
    for case, report in (
        ("waitcheck", _waitcheck_report()),
        ("consan-clean", _consan_clean_report()),
        ("consan-racy", _consan_racy_report()),
    ):
        (incoming / case).mkdir(parents=True)
        (incoming / case / "sanitizer_report.json").write_text(json.dumps(report))
    out = tmp_path / "out"
    argv = [
        "gen_sanitizer_dashboard",
        "--results-dir", str(incoming),
        "--baselines", str(baselines),
        "--out-dir", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert gen.main() == 0
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "No workload-survey kernels in this run." in html
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))
    assert data[0]["survey"] == []
