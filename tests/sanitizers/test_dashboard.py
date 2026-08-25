"""Unit tests for the sanitizer dashboard generator (pure logic)."""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import re
import shutil
import subprocess
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
    # the gate hero splits state and detail onto two lines
    assert "<strong>HEALTHY</strong>3/3 sanitizer outcomes match their baselines" in html
    assert html.count('<span class="pill ok">Expected outcome</span>') >= 3
    # verdicts are tinted badges; the solid baseline pill carries regression health
    assert '<span class="v warn">warn</span>' in html
    assert '<span class="v fail">fail</span>' in html
    assert "This tab is the regression gate." in html
    assert "<th>Baseline status</th><th>Observed</th>" in html
    assert "<th>Expected</th><th>Execution</th>" in html
    assert "gemm_x" in html and "Kernel details" in html
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

    assert "<strong>REGRESSION</strong>investigate 1/3 sanitizer outcomes" in html
    assert '<span class="pill bad">Unexpected outcome</span>' in html
    assert '<span class="pill bad">Mismatch</span>' in html
    assert '<span class="v fail">fail</span>' in html
    assert '<span class="dash">expected</span> <span class="v pass">pass</span>' in html
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
    assert '<span class="v neutral">—</span>' in html
    assert "❌ **Report missing**" in md
    assert "Observed sanitizer verdict: `—`" in md
    assert "❌ **Report missing**<br>Observed: `—`" in md
    # the per-case observation summary renders for a missing guardrail report too,
    # on both Tab 1 renderers (parity with the survey missing branch / #367).
    assert '<span class="lbl">Observation</span>report missing' in html
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

    assert "<strong>INCOMPLETE</strong>1/3 sanitizer report(s) are missing" in html
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
    assert f"<strong>UNHEALTHY</strong>{detail}" in html
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


def test_empty_page_styles_its_stale_banner():
    # The empty-runs branch still renders the stale banner, which is the most
    # safety-relevant thing on the page. It uses the shared .stale class, so the
    # branch has to embed _CSS -- otherwise a failed nightly with no data shows
    # the warning as an unstyled div and it reads as a broken page, not an alert.
    status = {
        "healthy": False, "conclusion": "failure", "run_id": "13",
        "run_url": "https://github.com/ROCm/aorta/actions/runs/13", "date": "2026-08-05",
    }
    html = gen.build_html([], status=status)
    assert 'class="stale"' in html
    assert ".stale {" in html, "empty-runs page renders .stale without embedding _CSS"
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
    # Review (#378): the banner is drawn by the .stale rules, so the empty-state
    # document has to embed the shared sheet. Without it the one page a reader
    # lands on when the nightly is broken shows its warning as an unstyled div.
    assert '<div class="stale">' in empty
    assert ".stale {" in empty


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


# --- Timestamped run ids (#392) ------------------------------------------------


def _publish_step() -> str:
    """The nightly's publish step, so tests can exercise its shell directly."""
    return (_REPO_ROOT / ".github/workflows/sanitizers-nightly.yml").read_text(
        encoding="utf-8"
    )


def _run_shell(script: str, cwd: Path, env: dict[str, str]) -> str:
    proc = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _dedent_block(workflow: str, start: str, end: str) -> str:
    """Lift a run-step fragment out of the workflow, YAML indentation removed."""
    lines = workflow.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip().startswith(start)]
    # A marker matching twice would lift some other step's block, and that shell
    # can still exit 0 -- the test would then pass without running the subject.
    assert len(starts) == 1, f"{start!r} matches {len(starts)} lines, want 1"
    first = starts[0]
    last = next(
        i for i, line in enumerate(lines[first:], first)
        if line.strip().startswith(end)
    )
    block = lines[first : last + 1]
    pad = min(len(line) - len(line.lstrip()) for line in block if line.strip())
    return "\n".join(line[pad:] for line in block)


def test_run_id_accepts_both_shapes_and_still_rejects_junk():
    # Runs published before the time was added keep their date-only names -- they
    # are never renamed, so both shapes are live in the retained window.
    for name in ("2026-08-23T094112-32638584704", "2026-08-23-32638584704",
                 "2026-08-23-9", "2026-08-23T000000-1"):
        assert gen._is_run_id(name), name
    for name in ("source", "not-a-run", "2026-08-23", "2026-08-23T0941-7",
                 "2026-08-23T094112", "2026-08-23T094112-", "2026-8-3T094112-7"):
        assert not gen._is_run_id(name), name


def test_history_order_is_correct_across_a_mixed_shape_history(tmp_path):
    # The two shapes have to sort against each other without special-casing: a
    # bare date is a prefix of any timestamped id for the same day, so it reads
    # as the earlier one, and the variable-width run id still breaks ties.
    root = tmp_path / "runs"
    ids = [
        "2026-08-23-32638584704",          # pre-change, same day
        "2026-08-23T094112-32638584705",
        "2026-08-24T031500-32700000001",
        "2026-08-23T094112-9",
        "2026-08-23T094112-10",
    ]
    for run_id in ids:
        _write_history_run(root, run_id)

    ordered = [r["meta"]["run"] for r in gen.runs_from_history_root(root, _baselines())]
    assert ordered == [
        "2026-08-24T031500-32700000001",
        "2026-08-23T094112-32638584705",
        "2026-08-23T094112-10",
        "2026-08-23T094112-9",
        "2026-08-23-32638584704",
    ]
    # --keep counts from the newest end of that same order.
    kept = gen.runs_from_history_root(root, _baselines(), keep=2)
    assert [r["meta"]["run"] for r in kept] == ordered[:2]


def test_workflow_prune_order_matches_the_generator_exactly(tmp_path):
    # The shell prunes and the generator renders from the same directory list, so
    # a disagreement deletes a directory the page still lists. Run the workflow's
    # own pipeline rather than a paraphrase of it.
    lines = _publish_step().splitlines()
    start = next(i for i, line in enumerate(lines) if 'ls -1 "$runs_dir"' in line)
    end = next(i for i, line in enumerate(lines[start:], start) if "| tr " in line)
    pipeline = " ".join(line.strip().rstrip("\\").strip() for line in lines[start : end + 1])

    ids = [
        "2026-08-23-32638584704",        # pre-change shape, same day
        "2026-08-23T094112-32638584705",
        "2026-08-24T031500-32700000001",
        "2026-08-23T094112-9",           # variable-width run ids on one instant
        "2026-08-23T094112-10",
        "2026-08-22T235959-1",
    ]
    runs = tmp_path / "runs"
    runs.mkdir()
    for run_id in ids:
        (runs / run_id).mkdir()

    shell = _run_shell(pipeline, tmp_path, {"runs_dir": str(runs)}).splitlines()
    assert shell == sorted(ids, key=gen._history_sort_key, reverse=True)


def test_workflow_reuses_the_directory_a_rerun_already_minted(tmp_path):
    # A re-run reuses GITHUB_RUN_ID, so it must land on the directory its first
    # attempt minted. A fresh timestamp would give one workflow run two
    # directories: two --keep slots, two history rows, and the earlier attempt's
    # reports left in place -- which is what the step's `rm -rf` exists to stop.
    block = _dedent_block(_publish_step(), 'run_dir_id=""', ': "${run_dir_id:=')
    script = f'{block}\nprintf "%s" "$run_dir_id"'
    runs = tmp_path / "runs"
    runs.mkdir()
    env = {"runs_dir": str(runs), "GITHUB_RUN_ID": "32638584704",
           "started_id": "2026-08-24T031500"}

    # Nothing published yet: mint a name from this attempt's clock.
    assert _run_shell(script, tmp_path, env) == "2026-08-24T031500-32638584704"

    # A re-run of a timestamped run reuses that name, not the current time.
    (runs / "2026-08-23T094112-32638584704").mkdir()
    assert _run_shell(script, tmp_path, env) == "2026-08-23T094112-32638584704"

    # ...and a re-run of a run published before the scheme keeps its old name,
    # so no directory is ever renamed underneath a published URL.
    shutil.rmtree(runs / "2026-08-23T094112-32638584704")
    (runs / "2026-08-23-32638584704").mkdir()
    assert _run_shell(script, tmp_path, env) == "2026-08-23-32638584704"

    # A different run id is not mistaken for this one, including a suffix match.
    shutil.rmtree(runs / "2026-08-23-32638584704")
    (runs / "2026-08-23T094112-4704").mkdir()
    assert _run_shell(script, tmp_path, env) == "2026-08-24T031500-32638584704"

    # The old scheme could leave two directories for one run id -- a re-run that
    # crossed midnight got a second date -- so reuse the newest of them rather
    # than re-publishing into the older one and stranding the newer.
    (runs / "2026-08-23-32638584704").mkdir()
    (runs / "2026-08-24-32638584704").mkdir()
    assert _run_shell(script, tmp_path, env) == "2026-08-24-32638584704"


def test_workflow_and_generator_agree_on_the_embedded_instant(tmp_path):
    # The workflow derives meta.json's date from the resolved directory name so a
    # re-run cannot report a clock its own directory contradicts. Run the step's
    # own metadata writer over both id shapes and render what it wrote: asserting
    # that its parsing pattern appears in the file would still pass if the block
    # stopped consulting the directory name at all.
    block = _dedent_block(
        _publish_step(), 'if [ "${GPU_RESULT}" = "success" ]; then gate=true', "PY"
    )
    # That block runs `python`; the interpreter running these tests may only be
    # on PATH under another name, so shim the name the workflow uses.
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "python").write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8"
    )
    (shim / "python").chmod(0o755)
    started_iso = "2026-08-24T03:15:00+00:00"
    env = {
        "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "ROCm/aorta",
        "GITHUB_RUN_ID": "32638584704",
        "GITHUB_SHA": "f" * 40,
        "GPU_RESULT": "success",
        "started_iso": started_iso,
    }

    for run_id, expected, rendered in (
        # A timestamped name reports the instant it carries, not this attempt's
        # clock -- that is what makes a re-run's manifest match its directory.
        (
            "2026-08-23T094112-32638584704",
            "2026-08-23T09:41:12+00:00",
            "2026-08-23 09:41:12 UTC",
        ),
        # A name from before the scheme carries no instant, so this publish
        # supplies its own rather than inventing that date's midnight.
        ("2026-08-23-32638584704", started_iso, "2026-08-24 03:15:00 UTC"),
    ):
        dest = tmp_path / run_id
        dest.mkdir()
        _run_shell(block, tmp_path, {**env, "dest": str(dest), "run_dir_id": run_id})

        meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
        assert meta.get("date") == expected, run_id
        assert meta.get("run") == run_id
        assert meta.get("gate") is True  # mirrors GPU_RESULT=success
        # What the generator will put on the page for that manifest -- for the
        # timestamped name, the 094112 its own directory carries. Agreement is
        # asserted through the renderer rather than by pinning a shared pattern.
        assert gen.format_instant(meta.get("date")) == rendered, run_id


def test_format_instant_renders_a_time_and_never_invents_one():
    assert gen.format_instant("2026-08-23T09:41:12+00:00") == "2026-08-23 09:41:12 UTC"
    # An explicit Z, which fromisoformat only accepts from 3.11.
    assert gen.format_instant("2026-08-23T09:41:12Z") == "2026-08-23 09:41:12 UTC"
    # Naive means UTC by construction: every writer uses `date -u` or utcnow.
    assert gen.format_instant("2026-08-23T09:41:12") == "2026-08-23 09:41:12 UTC"
    # A non-UTC offset is normalised rather than shown in the writer's zone.
    assert gen.format_instant("2026-08-23T11:41:12+02:00") == "2026-08-23 09:41:12 UTC"

    # A date-only value comes from a run published before the id carried a time.
    # fromisoformat() would accept it and render a confident 00:00:00, so the
    # value is returned untouched instead of inventing a midnight.
    assert gen.format_instant("2026-08-23") == "2026-08-23"
    # Malformed or absent values pass through: the run identity around them is
    # still worth rendering.
    for junk in ("", None, "d", "not a date", "2026-08-23T0941"):
        assert gen.format_instant(junk) == (junk or "")


def test_published_pages_show_the_instant_but_env_json_keeps_it_machine_readable(
    tmp_path, monkeypatch
):
    root = tmp_path / "runs"
    run_id = "2026-08-23T094112-32638584705"
    _write_history_run(root, run_id, meta={"date": "2026-08-23T09:41:12+00:00"})
    monkeypatch.setattr(sys, "argv", [
        "gen_sanitizer_dashboard", "--history-root", str(root),
        "--baselines",
        str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
        "--out-dir", str(tmp_path / "dashboard"),
    ])
    assert gen.main() == 0
    out = tmp_path / "dashboard"

    human = "2026-08-23 09:41:12 UTC"
    for page in (out / "index.html", out / "runs" / run_id / "waitcheck" / "index.html"):
        assert human in page.read_text(encoding="utf-8"), page
    assert human in (out / "summary.md").read_text(encoding="utf-8")

    # The manifest a consumer of aorta.sanitizer_run_area/0.1 reads keeps the
    # ISO instant: the human rendering is a display concern, not a stored one.
    env = json.loads(
        (out / "runs" / run_id / "waitcheck" / "env.json").read_text(encoding="utf-8")
    )
    assert env.get("date") == "2026-08-23T09:41:12+00:00"


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
    # The kernel-detail card footer links the case's run area (its directory),
    # not just the report, so the logs/recipe/inputs are one click away (#384).
    assert 'href="runs/2026-08-05-33/waitcheck/">run area</a>' in html
    assert "view raw report" not in html
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
    assert "<strong>INCOMPLETE</strong>1/3 sanitizer report(s) are missing" in page
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


def test_verdict_html_colors_by_verdict():
    # #374 D: verdict badges keyed on the verdict string. fail and error keep
    # distinct class names but share one red in CSS, so the hues can be split
    # later without touching markup.
    assert gen._verdict_html("error") == '<span class="v error">error</span>'
    assert gen._verdict_html("fail") == '<span class="v fail">fail</span>'
    assert gen._verdict_html("warn") == '<span class="v warn">warn</span>'
    assert gen._verdict_html("pass") == '<span class="v pass">pass</span>'
    assert gen._verdict_html("not_checked") == '<span class="v neutral">not_checked</span>'
    assert gen._verdict_html("\u2014") == '<span class="v neutral">\u2014</span>'


def _relative_luminance(hex_color: str) -> float:
    channels = []
    for offset in (0, 2, 4):
        value = int(hex_color[offset : offset + 2], 16) / 255
        channels.append(value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_with_white(hex_color: str) -> float:
    return 1.05 / (_relative_luminance(hex_color.lstrip("#")) + 0.05)


def test_every_white_on_solid_fill_clears_wcag_aa():
    # Sweep the whole class rather than one chip (precedent: the .vchip.warn
    # contrast fix). Every solid fill that carries white text must clear 4.5:1
    # for normal-size text. The step-number badges are 11.5px bold, which is
    # *not* WCAG "large text", so they need the full 4.5:1 too -- white on the
    # brand green at 85% alpha measured 2.99:1 before this was pinned.
    css = gen._CSS
    # Read the colour actually bound to each selector rather than asserting a
    # hardcoded pair: "#15803D appears somewhere in the sheet" would still pass
    # if .step-num.green regressed, because --solid-ok uses the same value.
    hexcolor = r"(#[0-9A-Fa-f]{6})"
    declarations = {
        "--solid-ok": rf"--solid-ok:\s*{hexcolor}",
        "--solid-bad": rf"--solid-bad:\s*{hexcolor}",
        ".step-num.blue": rf"\.step-num\.blue\s*\{{\s*background:\s*{hexcolor}",
        ".step-num.purple": rf"\.step-num\.purple\s*\{{\s*background:\s*{hexcolor}",
        ".step-num.green": rf"\.step-num\.green\s*\{{\s*background:\s*{hexcolor}",
    }
    for selector, pattern in declarations.items():
        match = re.search(pattern, css)
        assert match, f"no opaque solid fill declared for {selector}"
        fill = match.group(1)
        ratio = _contrast_with_white(fill)
        assert ratio >= 4.5, f"white on {selector} ({fill}) is {ratio:.2f}:1, below WCAG AA"
    # the audited custom properties only matter while the pills still consume
    # them, and the whole sweep only matters while this text is still white
    assert ".pill.ok, .pill.pass { background:var(--solid-ok); }" in css
    assert ".pill.bad, .pill.fail { background:var(--solid-bad); }" in css
    assert re.search(r"\.pill\s*\{[^}]*color:#fff", css), ".pill no longer sets white text"
    assert re.search(r"\.step-num\s*\{[^}]*color:#fff", css), ".step-num lost white text"
    # a semi-transparent fill under white text would silently reintroduce the bug
    assert "background:rgba(34,197,94,.85)" not in css


def test_table_wrapper_scrolls_instead_of_clipping():
    # Review (#378): the latest-run table carries nine columns and its headers do
    # not wrap, so a clipping wrapper put the rightmost columns -- including the
    # report links -- permanently out of reach on a narrow viewport.
    #
    # Swept to the survey flow band for the same reason: its five steps are
    # fixed-width, so without a scroll container they spill out of their panel.
    for selector in (r"\.table-wrap", r"\.flow"):
        rule = re.search(rf"{selector}\s*\{{([^}}]*)\}}", gen._CSS)
        assert rule, f"the {selector} rule is no longer in the stylesheet"
        assert "overflow-x:auto" in rule.group(1)
        assert "overflow:hidden" not in rule.group(1)


def test_group_run_count_matches_the_rollup_and_excludes_absent_reports():
    # The per-kernel panel's "N sanitizer runs" chip and the roll-up's run total
    # must agree. Counting spec entries would show 2 here and 1 above for a group
    # whose second report is missing.
    entries = gen.survey_cases_from_spec(
        {"cases": [
            {"name": "waitcheck-k", "group": "k", "sanitizer": "waitcheck",
             "report": _waitcheck_report()},
            {"name": "consan-k", "group": "k", "sanitizer": "consan"},  # no report
        ]}
    )
    present = [e for e in entries if (e.get("summary") or {}).get("present")]
    assert len(entries) == 2 and len(present) == 1

    grouped = gen._group_survey_entries(entries)
    assert gen._survey_summary_stats(grouped).get("runs") == 1

    html = gen._survey_detail_html(entries)
    assert '<span class="count">1 sanitizer run</span>' in html
    assert "2 sanitizer runs" not in html
    # the absent sanitizer still gets a card -- it is listed, not hidden, which
    # is what the "Important Notes" copy has to describe
    kcards = [tag for tag in re.findall(r"<details[^>]*>", html) if 'class="kcard' in tag]
    assert len(kcards) == 2


def test_absent_report_never_claims_no_findings():
    # An absent case also carries findings: 0, so a naive count chip would put
    # "no findings" next to a "Report missing" pill and claim a clean scan that
    # never ran. Sweep both tabs: the guardrail card and the survey card.
    missing = gen.summarize_case(None, "pass")
    assert missing.get("present") is False and missing.get("findings") == 0
    assert gen._findings_chip_html(missing) == ""

    rows = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-clean": missing,
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    html = gen.build_html(
        [{"meta": {"run": "r1", "commit": "abc", "date": "d", "gpu": "gfx950"},
          "rows": rows, "gate": False}]
    )
    assert '<span class="pill bad">Report missing</span>' in html
    # the surviving "no findings" chip belongs to a present case, never the absent one
    assert 'Report missing</span><span class="v neutral">' in html
    assert '<span class="v neutral">\u2014</span><span class="findings">' not in html

    # same on the survey tab, which has no baseline pill to fall back on
    survey = gen.survey_cases_from_spec({"cases": [{"name": "s", "label": "gone"}]})
    assert survey[0].get("summary", {}).get("present") is False
    card = gen._survey_case_html(survey[0], heading="gone")
    assert "no findings" not in card


def test_run_index_page_lays_out_header_and_run_card_together():
    # The run card only sits top-right when it shares the .topbar flex row with
    # the header; emitting it as a bare sibling silently drops the layout.
    run = {
        "meta": {"run": "2026-08-05-33", "commit": "abc", "date": "d", "gpu": "gfx950"},
        "rows": {
            case: gen.summarize_case(_waitcheck_report(), "warn")
            for case, _k, _l, _b in gen.CASES
        },
        "gate": True,
    }
    page = gen.build_run_index_html(run)
    topbar = page.index("<div class=topbar>")
    assert topbar < page.index('<header class="page-header">') < page.index('<aside class="runcard">')
    assert "</div>" in page[page.index('<aside class="runcard">'):]


def test_wide_tables_stay_reachable_on_narrow_viewports():
    # The latest-run table has nine columns and th is white-space:nowrap, so a
    # clipping wrapper would put the rightmost columns and the report links out
    # of reach on a narrow viewport instead of letting the user scroll to them.
    css = gen._CSS
    match = re.search(r"\.table-wrap\s*\{([^}]*)\}", css)
    assert match, ".table-wrap rule not found"
    rule = match.group(1)
    assert "overflow-x:auto" in rule.replace(" ", ""), f".table-wrap must scroll: {rule}"
    assert "overflow:hidden" not in rule.replace(" ", "")


def test_verdict_and_baseline_use_separate_visual_axes():
    # The verdict badge is always tinted; the baseline status pill is always
    # solid. That is what lets a red observed `fail` sit beside a green
    # "Expected outcome" pill without reading as a regression.
    row = gen.summarize_case(_consan_racy_report(), "fail")
    assert 'class="v fail"' in gen._verdict_html(row["verdict"])
    assert gen._baseline_status_html(row) == '<span class="pill ok">Expected outcome</span>'
    css = gen._CSS
    assert ".pill.ok, .pill.pass { background:var(--solid-ok); }" in css
    assert "background:rgba(239,68,68,.13)" in css  # verdict red stays tinted


def test_survey_intro_is_readable_and_drops_experimental_framing():
    # #374 A/B: the survey tab states plainly that it is observed-only and
    # non-gating, dropping the old "experimental" / "caller-supplied code
    # objects" framing from user-facing copy. The WaitCheck / ConSan
    # definitions are page-level cards (both tabs report both sanitizers).
    html = gen.build_html([_healthy_guardrail_run()], survey=[])
    assert "This tab is observational only." in html
    assert "do not affect nightly pass/fail status" in html
    assert "WaitCheck" in html and "ConSan" in html
    assert "Findings represent observed behavior, not regressions." in html
    assert "experimental" not in html.lower()
    assert "caller-supplied" not in html.lower()


def test_tool_cards_are_page_level_so_both_tabs_get_them():
    # Both tabs report waitcheck and consan results, so the definitions sit
    # above the tab bar rather than inside the survey panel.
    html = gen.build_html([_healthy_guardrail_run()], survey=[])
    tabbar = html.index('<div class=tabbar>')
    assert html.index('<section class="toolgrid">') < tabbar
    assert html.count("Static Analysis") == 1 and html.count("Runtime Analysis") == 1


def test_survey_intro_qualifies_the_both_sanitizers_claim():
    # Review (#374): the absolute "shown under both sanitizers" claim is false on the
    # supported degraded path (an absent/unreadable report skips a sanitizer). The
    # HTML notes + MD mirror must qualify it so the UI never claims both ran when only
    # one report exists.
    #
    # Review (#378): the qualification must also match what the degraded path
    # actually renders. An absent report still gets a card (marked "report
    # missing", no verdict), so copy claiming it "does not appear" is wrong.
    html = gen.build_html([_healthy_guardrail_run()], survey=[])
    # the old absolute claim is gone; the qualified degraded-path copy is present
    assert "shown under <b>both</b> sanitizers." not in html
    assert "simply does not appear" not in html
    assert "a skipped or missing scan still appears, marked report missing" in html
    md = gen.build_summary_md([_healthy_guardrail_run()], survey=[])
    assert "shown under both." not in md
    assert "only the sanitizer(s) that ran appear" not in md
    assert "still appears, marked report missing with no verdict" in md


def test_survey_absent_report_still_renders_a_card_and_is_not_counted_as_a_run():
    # Review (#378), the behavior the intro copy above promises: a survey entry
    # whose report is missing still renders its own card so the gap is visible,
    # but it is not a sanitizer run -- the kernel-group heading must agree with
    # the roll-up headline instead of counting spec entries.
    entries = gen.survey_cases_from_spec(
        {"cases": [
            {"name": "waitcheck-half", "group": "half", "sanitizer": "waitcheck",
             "report": _waitcheck_report()},
            {"name": "consan-half", "group": "half", "sanitizer": "consan"},
        ]}
    )
    detail = gen._survey_detail_html(entries)
    # both entries render a card, so the missing scan is visible rather than dropped
    assert detail.count('<details class="kcard') == 2
    assert "report missing" in detail
    # ... but only the one that produced a report counts as a run
    assert '<span class="count">1 sanitizer run</span>' in detail
    assert "2 sanitizer runs" not in detail
    stats = gen._survey_summary_stats(gen._group_survey_entries(entries))
    assert stats.get("runs") == 1


def test_survey_absent_card_keeps_its_workload_provenance():
    # Bugbot (#378): moving the survey meta line into the facts grid rendered the
    # grid only on the present branch, so an absent card silently lost the
    # "Source" pair that the pre-redesign HTML and the Markdown twin both keep.
    # The rest of the grid stays off that branch: backend / selection /
    # execution would describe a report that was never produced.
    entries = gen.survey_cases_from_spec(
        {"cases": [
            {"name": "consan-ghost", "sanitizer": "consan", "workload": "internal:gemm_top5"},
        ]}
    )
    card = gen._survey_case_html(entries[0], heading="ConSan")
    assert "report missing" in card
    assert '<span class="k">Source</span>' in card
    assert "internal:gemm_top5" in card
    assert "Backend" not in card and "Execution" not in card
    md = "\n".join(gen._survey_section_md(entries))
    assert "source `internal:gemm_top5`" in md


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
        '<div class="observation error"><span class="lbl">Reason</span>'
        '<span class="msg">combined_hook_timeout</span></div>'
    )
    assert gen._survey_message_md(errored) == "Reason: `combined_hook_timeout`"

    # a warn case with a finding still leads with the finding (unchanged behavior)
    warn = gen.summarize_case(_waitcheck_report(), None)
    assert gen._survey_message_parts(warn)[0] == "Finding"
    assert gen._survey_message_html(warn).startswith(
        '<div class="observation warn"><span class="lbl">Finding</span>'
    )
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
    assert '<span class="lbl">Reason</span><span class="msg">combined_hook_timeout' in html
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
    # one kernel-group panel, one collapsed card per sanitizer
    assert '<span class="kname">gemm</span>' in html
    assert '<span class="name">WaitCheck</span>' in html
    assert '<span class="kind">static wait-count scan</span>' in html
    assert '<span class="name">ConSan</span>' in html
    assert '<span class="kind">dynamic data-race check</span>' in html
    # color-coded verdict badges: consan error -> red, waitcheck warn -> amber
    assert '<span class="v error">error</span>' in html
    assert '<span class="v warn">warn</span>' in html
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
    assert '<span class="lbl">Reason</span><span class="msg">combined_hook_timeout' in html
    assert '<span class="lbl">Finding</span>' in html
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
    assert 'class="statline"' in table
    assert "Surveyed <b>3 kernels</b> across <b>5 sanitizer runs</b>" in table
    for part in ("2 pass", "1 warn", "1 fail", "1 error"):
        assert part in table
    # one row per kernel group with a stable header
    assert "<th>Kernel</th><th>WaitCheck</th><th>ConSan</th>" in table
    # color-coded verdict badges (error/fail red, warn amber, pass green)
    assert '<span class="v warn">warn</span>' in table
    assert '<span class="v fail">fail</span>' in table
    assert '<span class="v error">error</span>' in table
    assert '<span class="v pass">pass</span>' in table
    # the obj group's waitcheck did not run -> em dash cell, never a fake verdict
    assert '<td><span class="dash">&mdash;</span></td>' in table

    # rendered on the page: gate stays HEALTHY, no regression vocabulary
    html = gen.build_html([_healthy_guardrail_run()], survey=entries)
    assert 'class="statline"' in html
    assert "<strong>HEALTHY</strong>3/3 sanitizer outcomes match their baselines" in html
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
        # URL syntax the browser acts on but a directory does not: the tail of each
        # of these is a query/fragment on the segment before it, so the request
        # never reaches the published path -- and a percent-escape can be
        # normalised back into "..".
        "runs/x?old/survey/foo/sanitizer_report.json",
        "runs/x#old/survey/foo/sanitizer_report.json",
        "runs/%2e%2e/survey/foo/sanitizer_report.json",
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
    # committed ConSan survey reports carry execution_status="error"; the facts
    # grid must render it as plain text, never the red badge the guardrail tab
    # uses for a broken run.
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
    # are complete), so a neutral survey render means no red execution badge.
    html = gen.build_html([_healthy_guardrail_run()], survey=survey)
    # execution status stays plain (never health-colored) on the survey tab
    assert '<span class="k">Execution</span><span class="val">error</span>' in html
    # but the observed *verdict* is a color-coded badge (#374 D): an error verdict
    # is red. This is descriptive only -- the gate stays healthy.
    assert '<span class="v error">error</span>' in html
    assert "<strong>HEALTHY</strong>3/3 sanitizer outcomes match their baselines" in html
    assert "REGRESSION" not in html and "Regression" not in html

    # Unit: the survey facts grid neutralizes execution; the guardrail one flags it.
    row = survey[0]["summary"]
    assert '<span class="v fail">error</span>' not in gen._facts_html(row, observed=True)
    assert '<span class="v fail">error</span>' in gen._facts_html(row, observed=False)


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
    assert "This tab is observational only." in html
    # folding is pure HTML disclosure, so the no-JS promise still holds
    assert "<details" in html and "<summary>" in html


def test_build_html_survey_fail_is_color_coded_but_never_a_regression():
    # Guardrails are all healthy warn; the survey case observed a fail. The survey
    # fail must render as a solid color-coded (red) verdict chip (#374 D) and must
    # NOT turn the gate red or be labelled a regression / mismatch / unexpected.
    survey = gen.survey_cases_from_spec(
        {"cases": [{"name": "s", "label": "survey fail case",
                    "report": _consan_racy_report()}]}
    )
    html = gen.build_html([_healthy_guardrail_run()], survey=survey)

    assert "<strong>HEALTHY</strong>3/3 sanitizer outcomes match their baselines" in html
    # the fail is a red verdict badge (color-coded), never the neutral grey one
    assert '<span class="v fail">fail</span>' in html
    assert '<span class="v neutral">fail</span>' not in html
    # none of the guardrail regression vocabulary leaks in from the survey, and the
    # gate banner stays healthy (a survey fail/error is observational, never a gate).
    assert "REGRESSION" not in html and "Regression" not in html
    assert "Unexpected outcome" not in html and "Mismatch" not in html
    # no baseline/expected column on the survey tab (no "expected <verdict>" phrasing
    # is emitted for the survey case; that phrasing is guardrail-only)
    assert "survey fail case" in html
    # observation summary + inline message are present on the tab
    assert '<span class="lbl">Observation</span>' in html
    assert '<span class="lbl">Finding</span>' in html


def test_build_html_survey_report_link_and_graceful_absence():
    survey = gen.survey_cases_from_spec(
        {"cases": [
            # "staged" says the whole directory was published, so the card footer
            # may link it; report_rel alone only makes the JSON reachable.
            {"name": "linked", "label": "linked survey", "staged": True,
             "report_rel": "runs/x/survey/linked/sanitizer_report.json",
             "report": _consan_racy_report()},
            {"name": "nolink", "label": "nolink survey", "report": _waitcheck_report()},
        ]},
        rel="runs/x",
    )
    html = gen.build_html([_healthy_guardrail_run()], survey=survey)

    # the linked survey case drills down to its raw report (per-kernel row) and to
    # its run area (card footer); the unlinked one still renders, degrading to
    # neither link. Guardrail rows carry no report_rel here, so the only run-area
    # footer comes from the linked survey case.
    assert 'href="runs/x/survey/linked/sanitizer_report.json"' in html
    assert html.count('href="runs/x/survey/linked/">run area</a>') == 1
    assert html.count("run area</a>") == 1
    assert "linked survey" in html and "nolink survey" in html


def test_build_html_survey_report_without_staged_area_emits_no_directory_link():
    # A spec entry's report can be staged by the caller while its *directory* is
    # not: a directory link needs an index.html only the area's publisher writes.
    # Linking it anyway published a 404 on every --survey invocation.
    survey = gen.survey_cases_from_spec(
        {"cases": [{
            "name": "linked", "label": "linked survey",
            "report_rel": "runs/x/survey/linked/sanitizer_report.json",
            "report": _consan_racy_report(),
        }]}
    )
    assert survey[0]["staged"] is False
    html = gen.build_html([_healthy_guardrail_run()], survey=survey)

    # The per-kernel Report column still reaches the JSON the caller did stage.
    assert 'href="runs/x/survey/linked/sanitizer_report.json"' in html
    # ...but nothing claims the directory.
    assert 'href="runs/x/survey/linked/"' not in html
    assert "run area</a>" not in html


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
        {"cases": [{"name": "s", "label": "survey linked", "staged": True,
                    "report_rel": "runs/x/survey/s/sanitizer_report.json",
                    "report": _consan_racy_report()}]},
        rel="runs/x",
    )
    html = gen.build_html(runs, survey=survey)

    # Guardrail kernel-detail row links to its case report (in addition to the
    # latest-run table's Report cell) -> the waitcheck report link now appears
    # twice: the latest-run table cell and the per-kernel-row Report column.
    wc = 'href="runs/2026-08-05-33/waitcheck/sanitizer_report.json">report</a>'
    assert html.count(wc) == 2
    # Survey kernel row links to the survey case's report_rel (the card footer uses
    # the distinct "run area" text pointing at the directory, so a "report" link
    # here is the per-row one).
    assert 'href="runs/x/survey/s/sanitizer_report.json">report</a>' in html
    assert 'href="runs/x/survey/s/">run area</a>' in html
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


# --------------------------------------------------------------------------
# Run area (#384): the case directory a card footer links to
# --------------------------------------------------------------------------


def _publish_with_logs(tmp_path, monkeypatch, *, run_ids, keep_logs=None, survey=True):
    """Publish a history (each case carrying a sweep log) plus one survey case."""
    root = tmp_path / "runs"
    for run_id in run_ids:
        run_dir = _write_history_run(root, run_id)
        (run_dir / "waitcheck" / "waitcheck").mkdir(parents=True)
        (run_dir / "waitcheck" / "waitcheck" / "waitcheck-0.log").write_text("scan output\n")
        (run_dir / "consan-racy" / "consan").mkdir(parents=True)
        (run_dir / "consan-racy" / "consan" / "consan.log").write_text("race detected\n")
    argv = [
        "gen_sanitizer_dashboard",
        "--history-root", str(root),
        "--baselines",
        str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
        "--out-dir", str(tmp_path / "dashboard"),
    ]
    if survey:
        info = tmp_path / "informational"
        (info / "consan-gemm" / "consan").mkdir(parents=True)
        (info / "consan-gemm" / "sanitizer_report.json").write_text(
            json.dumps(_informational_report("combined_hook_timeout"))
        )
        (info / "consan-gemm" / "consan" / "consan.log").write_text("hook timeout\n")
        argv += ["--informational-results-dir", str(info)]
    if keep_logs is not None:
        argv += ["--keep-logs", str(keep_logs)]
    monkeypatch.setattr(sys, "argv", argv)
    assert gen.main() == 0
    return tmp_path / "dashboard"


def test_case_dir_rel_points_at_the_directory_and_rejects_unsafe_values():
    # The footer link is the case's directory, derived from report_rel so it
    # inherits the same relative-only guarantee as the report link.
    assert (
        gen._case_dir_rel("runs/2026-08-05-33/survey/consan-gemm/sanitizer_report.json")
        == "runs/2026-08-05-33/survey/consan-gemm/"
    )
    assert gen._case_dir_rel("runs/x/waitcheck/sanitizer_report.json") == "runs/x/waitcheck/"
    # No safe directory to link -> no link at all, never a dead or unsafe href.
    for unsafe in (
        None, "", "sanitizer_report.json", "javascript:alert(1)",
        "/runs/x/c/sanitizer_report.json", "../../etc/passwd",
        "//evil.example/x/sanitizer_report.json",
        # A query/fragment makes the emitted URL address "runs/x", not the case
        # directory, so there is no safe area link to derive from it.
        "runs/x?old/survey/foo/sanitizer_report.json",
        "runs/x#old/survey/foo/sanitizer_report.json",
    ):
        assert gen._case_dir_rel(unsafe) is None, unsafe


def test_raw_link_html_renders_run_area_or_nothing():
    linked = gen._raw_link_html("runs/x/survey/s/sanitizer_report.json")
    assert 'href="runs/x/survey/s/">run area</a>' in linked
    assert "view raw report" not in linked
    assert gen._raw_link_html(None) == ""
    assert gen._raw_link_html("javascript:alert(1)") == ""


def test_recipe_fixture_refs_splits_source_inputs_from_built_artifacts():
    # Source inputs are copied into the run area; CI-built artifacts (gitignored,
    # ~16MB for a GEMM object) are recorded by digest instead.
    recipe = (
        "sanitizer_plan:\n"
        "  source:\n"
        "    hip: fixtures/repro/consan_lds_race.hip\n"
        "    path: fixtures/gemm_shapes_unique.csv\n"
        "    code_object: fixtures/isa/consan_gemm_f32.hsaco\n"
        "    consan_command: fixtures/bin/consan_gemm_load\n"
        "    isa_dir: fixtures/isa\n"
    )
    source, built = gen._recipe_fixture_refs(recipe)
    assert source == ["fixtures/repro/consan_lds_race.hip", "fixtures/gemm_shapes_unique.csv"]
    assert built == [
        "fixtures/isa/consan_gemm_f32.hsaco",
        "fixtures/bin/consan_gemm_load",
        "fixtures/isa",
    ]


def test_report_digests_flattens_backend_and_code_object_provenance():
    # waitcheck records its binary path/sha256; every worklist kernel records its
    # code-object digest. Both are already in the report and rendered nowhere.
    # .get() throughout: whether the key was picked up at all is exactly what is
    # under test, so a miss should read as the assertion it is rather than a
    # KeyError that hides which digest went missing.
    digests = gen.report_digests(_waitcheck_report())
    assert digests.get("path") == "/tmp/build/tools/rj_waitcheck"
    assert digests.get("sha256") == "472fcf288714beef"
    assert digests.get("code_object:sol_1.hsaco") == "93f09ae670abcdef"
    # ConSan's repro command / hook digests are picked up from the same field.
    consan = _consan_clean_report()
    consan["checks"][0]["backend"] = {
        "command": "/w/fixtures/bin/consan_gemm_load", "command_sha256": "deadbeef",
        "hook": "/w/librocjitsu_dbi_hooks.so", "hook_sha256": "feedface",
    }
    assert gen.report_digests(consan).get("command_sha256") == "deadbeef"
    # A malformed report degrades to {} rather than raising.
    for bad in (None, [], "oops", {"checks": None}, {"worklist": {"kernels": [1]}}):
        assert gen.report_digests(bad) == {}


def test_main_with_a_survey_spec_emits_no_run_area_link_it_cannot_publish(
    tmp_path, monkeypatch
):
    # --survey entries render from an inline report; only
    # --informational-results-dir entries get their directory staged. Keying the
    # run-index link on report presence published a dead survey/<case>/ link on
    # exactly this invocation.
    out = tmp_path / "dashboard"
    out.mkdir()
    _write_history_run(out / "runs", "2026-08-05-33")
    spec = tmp_path / "survey.json"
    spec.write_text(json.dumps({"cases": [{
        "name": "consan-gemm", "label": "gemm - ConSan",
        "command": "aorta sweep run --recipe recipes/sanitizers/daily-consan-gemm.yaml",
        "report_rel": "runs/2026-08-05-33/survey/consan-gemm/sanitizer_report.json",
        "report": _consan_racy_report(),
    }]}))
    monkeypatch.setattr(sys, "argv", [
        "gen_sanitizer_dashboard",
        "--history-root", str(out / "runs"),
        "--baselines",
        str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
        "--out-dir", str(out),
        "--survey", str(spec),
    ])
    assert gen.main() == 0

    assert not (out / "runs/2026-08-05-33/survey").exists()
    # The case is still listed on both pages; it just carries no directory link.
    run_page = (out / "runs/2026-08-05-33/index.html").read_text(encoding="utf-8")
    assert "Workload survey" in run_page
    assert "survey/consan-gemm/" not in run_page
    # Same on the dashboard's Tab 2 card footer: no directory link is claimed.
    # (The per-row Report link to the JSON is the caller's to stage, unchanged.)
    dashboard = (out / "index.html").read_text(encoding="utf-8")
    assert 'href="runs/2026-08-05-33/survey/consan-gemm/"' not in dashboard
    # Every relative link on the run page resolves.
    for href in re.findall(r'href="([^"#]+)"', run_page):
        if href.startswith("http"):
            continue
        target = (out / "runs/2026-08-05-33" / href).resolve()
        ok = (target / "index.html").is_file() if target.is_dir() else target.exists()
        assert ok, href


def test_rebuild_hint_sources_all_exist_in_the_repo():
    # The hints name concrete source paths so a rebuild reproduces the recorded
    # digest. A rename must break here rather than silently ship a stale command.
    recipes = _REPO_ROOT / "recipes" / "sanitizers"
    for ref, source in gen._GENCO_ISA_SOURCES.items():
        assert (recipes / source).is_file(), f"{ref} -> {source}"
    for ref, (source, define, _extra) in gen._BIN_SOURCES.items():
        assert (recipes / source).is_file(), f"{ref} -> {source}"
        # A define means the binary is bound to a code object it must be told about.
        assert (define is None) == (ref not in gen._BIN_OBJECT), ref


def test_rebuild_hint_binary_flags_match_the_workflow():
    """Cross-check the flags against the workflow that actually builds them.

    Optimisation/debug flags change the executable, so a hint that drops them
    cannot reproduce the recorded `command_sha256`. The nightly builds the two
    guardrail repro binaries `-O1 -g` and the loader binaries without, and this
    asserts the map still agrees rather than trusting it to stay in step.
    """
    workflow = (_REPO_ROOT / ".github/workflows/sanitizers-nightly.yml").read_text(
        encoding="utf-8"
    )
    # Join shell line continuations so one hipcc invocation is one line.
    joined = workflow.replace("\\\n", " ")
    for ref, (_source, _define, extra) in gen._BIN_SOURCES.items():
        name = ref.rsplit("/", 1)[-1]
        lines = [
            line
            for line in joined.splitlines()
            if "hipcc" in line and f'/bin/{name}"' in line
        ]
        assert len(lines) == 1, f"{ref}: expected one build line, got {len(lines)}"
        built_with_o1g = "-O1 -g" in lines[0]
        assert built_with_o1g == ("-O1 -g" in extra), (
            f"{ref}: workflow uses -O1 -g = {built_with_o1g}, hint says {extra!r}"
        )
        hint = gen._rebuild_hints(gen.rebuild_plan([ref], target="gfx950"))[0]
        assert ("-O1 -g" in hint) == built_with_o1g, ref


def test_recipe_fixture_refs_rejects_parent_traversal():
    # _FIXTURE_REF_RE admits "." so it can match an extension, which also matched
    # a parent segment. The copy checked the source against the repo root but then
    # joined the same value onto the destination, writing outside the case dir.
    source, built = gen._recipe_fixture_refs(
        "source:\n  path: fixtures/../../../README.md\n"
        "  hip: fixtures/repro/consan_lds_race.hip\n"
        "  code_object: fixtures/isa/../../../etc/passwd\n"
    )
    assert source == ["fixtures/repro/consan_lds_race.hip"]
    assert built == []


def test_publish_recipe_inputs_never_writes_outside_the_case_inputs_dir(tmp_path):
    case_dir = tmp_path / "runs" / "id" / "case"
    case_dir.mkdir(parents=True)
    recipe = tmp_path / "repo" / "recipes" / "sanitizers" / "evil.yaml"
    recipe.parent.mkdir(parents=True)
    (tmp_path / "repo" / "secret.txt").write_text("do not publish me")
    # The ref must actually resolve to a real file inside repo_root, or the copy
    # is skipped for an unrelated reason and this test proves nothing. From
    # recipes/sanitizers/fixtures that is three parent segments up.
    recipe.write_text("source:\n  path: fixtures/../../../secret.txt\n")
    assert (recipe.parent / "fixtures/../../../secret.txt").resolve().is_file()

    copied, built = gen._publish_recipe_inputs(
        case_dir, recipe_src=recipe, repo_root=tmp_path / "repo"
    )

    assert copied == [] and built == []
    # recipe.yaml is the only thing written, and nothing escaped the case dir.
    written = sorted(p.name for p in case_dir.rglob("*") if p.is_file())
    assert written == ["recipe.yaml"]
    assert not (tmp_path / "runs" / "id" / "secret.txt").exists()
    assert not (tmp_path / "runs" / "secret.txt").exists()


def test_rebuild_hints_match_the_nightly_commands_per_artifact():
    def hint(ref: str) -> str:
        return gen._rebuild_hints(gen.rebuild_plan([ref], target="gfx950"))[0]

    hints = {
        ref: hint(ref)
        for ref in (
            "fixtures/isa/lds.hsaco",
            "fixtures/isa/consan_gemm_f32.hsaco",
            "fixtures/isa",
            "fixtures/bin/lds_dispatch",
            "fixtures/bin/consan_lds_race",
        )
    }
    # A --genco object may be a bundle or a raw ELF depending on the ROCm build,
    # and the recorded digest is of the unbundled object -- so the conditional
    # unbundle is part of the instruction, not an optional detail.
    lds = hints["fixtures/isa/lds.hsaco"]
    assert (
        "hipcc --genco --offload-arch=gfx950 "
        "recipes/sanitizers/fixtures/kernels/lds_reduce.hip" in lds
    )
    assert "__CLANG_OFFLOAD_BUNDLE__" in lds
    assert "clang-offload-bundler --type=o --unbundle" in lds
    # The GEMM objects are extracted from Tensile libraries, never compiled.
    assert "prepare_gemm_isa.py" in hints["fixtures/isa/consan_gemm_f32.hsaco"]
    assert "hipcc" not in hints["fixtures/isa/consan_gemm_f32.hsaco"]
    assert "--top-n 3" in hints["fixtures/isa"]
    # A loader binary needs the define naming its object, or it builds the wrong one.
    assert "-DLDS_HSACO=" in hints["fixtures/bin/lds_dispatch"]
    assert "recipes/sanitizers/fixtures/isa/lds.hsaco" in hints["fixtures/bin/lds_dispatch"]
    # A plain repro binary has no define to add.
    assert "-D" not in hints["fixtures/bin/consan_lds_race"]
    assert (
        "recipes/sanitizers/fixtures/repro/consan_lds_race.hip"
        in hints["fixtures/bin/consan_lds_race"]
    )
    # An unknown reference degrades to the bare path rather than a wrong command.
    assert gen._rebuild_hints(
        gen.rebuild_plan(["fixtures/other/x"], target="gfx950")
    ) == ["fixtures/other/x"]
    # Every hint's backticks are balanced, so the page renders them as <code>
    # spans rather than leaving a stray literal backtick in the markup.
    for ref, text in hints.items():
        assert text.count("`") % 2 == 0, ref


def test_rebuild_commands_are_runnable_from_the_repo_root():
    """Every emitted command must work from where REPRODUCE.md leaves the reader.

    The manifest records artifacts recipe-relative (``fixtures/isa/...``), but
    ``cd aorta`` puts you at the repo root where no top-level ``fixtures/`` exists
    -- so a command carrying the recorded path verbatim fails immediately. The
    output directories are gitignored, so they also have to be created.
    """
    refs = [
        "fixtures/isa/lds.hsaco",
        "fixtures/isa/consan_gemm_f32.hsaco",
        "fixtures/isa",
        "fixtures/bin/consan_gemm_load",
        "fixtures/bin/consan_lds_race",
    ]
    for entry in gen.rebuild_plan(refs, target="gfx950"):
        # The recorded path stays recipe-relative; only the commands are rewritten.
        assert entry["path"].startswith("fixtures/"), entry["path"]
        assert entry["commands"], entry["path"]
        assert entry["commands"][0] == gen._ROCM_LLVM_PATH_EXPORT
        assert entry["commands"][1].startswith("mkdir -p recipes/sanitizers/fixtures/")
        for command in entry["commands"]:
            # No bare recipe-relative path survives into an executable command:
            # every fixtures/ token must be reached through recipes/sanitizers/.
            unrooted = re.findall(r"(?<![\w/.])fixtures/[\w./-]+", command)
            assert not unrooted, f"{entry['path']}: unrooted {unrooted} in {command!r}"
    # Every *input* path a command names resolves in a clean checkout. The two
    # build-output roots are gitignored, so they are skipped by prefix rather than
    # by substring -- an earlier version matched "/bin/" and so happened to pass
    # only on a machine where a previous run had already created those dirs.
    generated = (
        "recipes/sanitizers/fixtures/isa",
        "recipes/sanitizers/fixtures/bin",
    )
    for entry in gen.rebuild_plan(refs, target="gfx950"):
        for command in entry["commands"]:
            for token in re.findall(r"recipes/sanitizers/fixtures/[\w./-]+", command):
                if any(token == root or token.startswith(f"{root}/") for root in generated):
                    continue
                assert (_REPO_ROOT / token).exists(), f"{entry['path']}: {token}"


def test_rebuild_commands_export_the_rocm_llvm_path():
    """The bundler is not on PATH in the image that produced these artifacts.

    The ROCm container exports only ``/opt/rocm/bin``, so the nightly adds the LLVM
    bindir before building any fixture -- hipcc shells out to
    ``clang-offload-bundler``, ``prepare_gemm_isa.py`` looks it up with
    ``shutil.which``, and the genco branch calls it directly. A pasteable command
    that names it bare fails the way that job's fixture build once did, so every
    entry opens with the export and it has to stay the workflow's own.
    """
    workflow = (_REPO_ROOT / ".github/workflows/sanitizers-nightly.yml").read_text(
        encoding="utf-8"
    )
    assert gen._ROCM_LLVM_PATH_EXPORT in workflow
    refs = [
        "fixtures/isa/lds.hsaco",  # hipcc --genco + an explicit unbundle
        "fixtures/isa/consan_gemm_f32.hsaco",  # prepare_gemm_isa.py
        "fixtures/isa",
        "fixtures/bin/consan_lds_race",  # hipcc host+device compile
    ]
    for entry in gen.rebuild_plan(refs, target="gfx950"):
        assert entry["commands"][0] == gen._ROCM_LLVM_PATH_EXPORT, entry["path"]
    # Rendered into the prose too, since the hints are generated from the plan.
    hint = gen._rebuild_hints(gen.rebuild_plan(["fixtures/isa"], target="gfx950"))[0]
    assert f"`{gen._ROCM_LLVM_PATH_EXPORT}`" in hint
    # An unrecognised reference still yields no commands rather than a bare export.
    assert gen.rebuild_plan(["fixtures/other/x"], target="gfx950")[0]["commands"] == []


def test_run_area_ships_every_source_its_rebuild_commands_read():
    """Decision 10 asks for the inputs a reproduction needs, not just the named ones.

    A recipe names only the built artifact it *consumes*: every ``daily-consan-*``
    lists a ``.hsaco`` and a loader binary and nothing else. Scanning the recipe
    text alone therefore left ``inputs/`` empty for exactly the cases whose
    published rebuild commands read the most.
    """
    recipes = _REPO_ROOT / "recipes" / "sanitizers"
    generated = ("fixtures/isa", "fixtures/bin")
    for recipe in sorted(recipes.glob("daily-*.yaml")):
        text = recipe.read_text(encoding="utf-8")
        named, built = gen._recipe_fixture_refs(text)
        published = set(named) | set(gen.rebuild_input_sources(built))
        # Every non-generated fixture path the commands name must be published.
        for entry in gen.rebuild_plan(built, target="gfx950"):
            for command in entry["commands"]:
                for token in re.findall(
                    r"recipes/sanitizers/fixtures/[\w./-]+", command
                ):
                    ref = f"fixtures{token[len('recipes/sanitizers/fixtures'):]}"
                    if any(
                        ref == root or ref.startswith(f"{root}/") for root in generated
                    ):
                        continue
                    assert ref in published, f"{recipe.name}: {ref} not in inputs/"
        # ...and each one is a real file, so the copy actually happens.
        for ref in published:
            assert (recipes / ref).is_file(), f"{recipe.name}: {ref}"


def test_consan_recipes_publish_their_transitive_sources(tmp_path, monkeypatch):
    # End-to-end: the ConSan survey case used to ship an empty inputs/ even though
    # its own commands read the shape CSV and the loader source.
    out = _publish_with_logs(tmp_path, monkeypatch, run_ids=["2026-08-05-33"])
    area = out / "runs" / "2026-08-05-33" / "survey" / "consan-gemm"

    inputs = sorted(
        p.relative_to(area / "inputs").as_posix()
        for p in (area / "inputs").rglob("*")
        if p.is_file()
    )
    assert inputs == [
        "fixtures/gemm_shapes_unique.csv",
        "fixtures/kernels/consan_load.hip",
    ]
    env = json.loads((area / "env.json").read_text(encoding="utf-8"))
    assert sorted(env["inputs"]) == inputs
    # They are listed as downloads on the landing page, like any published file.
    page = (area / "index.html").read_text(encoding="utf-8")
    for rel in inputs:
        assert f'href="inputs/{rel}"' in page


def test_drilldown_pages_link_one_stylesheet_instead_of_inlining_it(
    tmp_path, monkeypatch
):
    """_CSS was over 80% of every run and run-area page, duplicated per run.

    At ``--keep 30`` with five areas per run that is ~2.7MB of byte-identical CSS
    in the published tree -- more than everything else in it, and past the gzipped
    logs ``--keep-logs`` exists to bound.
    """
    out = _publish_with_logs(tmp_path, monkeypatch, run_ids=["2026-08-05-33"])
    asset = out / gen.RUN_AREA_CSS_REL
    assert asset.is_file()
    assert asset.read_text(encoding="utf-8") == gen.run_area_stylesheet()

    drilldowns = sorted(out.glob("runs/*/index.html")) + sorted(
        out.glob("runs/*/*/index.html")
    ) + sorted(out.glob("runs/*/survey/*/index.html"))
    assert drilldowns
    for page in drilldowns:
        html = page.read_text(encoding="utf-8")
        # Linked, not inlined...
        assert "<style>" not in html, page
        assert gen._CSS not in html, page
        href = re.search(r'href="([^"]*run-area\.css)"', html)
        assert href, page
        # ...and the link resolves from this page's depth.
        assert (page.parent / href.group(1)).resolve() == asset.resolve(), page
        # A drill-down page is now a small document.
        assert page.stat().st_size < len(gen._CSS), page

    # The root dashboard keeps its inline copy: one of it, and it must render
    # standalone (its empty state carries the stale banner).
    root = (out / "index.html").read_text(encoding="utf-8")
    assert "<style>" in root and ".stale {" in root


def test_run_area_stylesheet_carries_the_rules_those_pages_use():
    sheet = gen.run_area_stylesheet()
    assert gen._CSS in sheet
    # The rules the two drill-down pages add on top of the dashboard sheet.
    for selector in (
        ".wrap {",
        ".note {",
        # The rebuild section renders structurally now, so `.steps` (its old
        # bullet list) is gone and these carry the command block instead.
        ".rb {",
        ".rb-last {",
        ".rb .path {",
        ".rb pre {",
        ".rb pre code {",
        # `.cap`'s heading bar reads `var(--accent, ...)`, which only the root
        # dashboard's kernel cards define -- without this these pages render it
        # in the same grey as the `th` text beneath it.
        ".panel { --accent:",
    ):
        assert selector in sheet, selector
    assert ".steps {" not in sheet
    # Byte-stable, so re-publishing does not churn the data branch.
    assert gen.run_area_stylesheet() == sheet


# --- Run area rendering (#391) -------------------------------------------------


def _css_rule(sheet: str, selector: str) -> str:
    """One selector's declaration block, so a test can assert on it alone."""
    start = sheet.index(selector) + len(selector)
    return sheet[start : sheet.index("}", start)]


def test_fact_row_sizes_each_fact_to_its_content():
    # Equal-width tracks gave a 1-char Findings the same width as a 51-char
    # Recipe: ~555px of dead space on one row while Commit and Recipe wrapped
    # for want of ~330px.
    sheet = gen.run_area_stylesheet()
    row = _css_rule(sheet, ".kv {")
    assert "display:flex" in row and "flex-wrap:wrap" in row
    # The uniform grid is gone, not merely overridden by a later rule.
    assert "minmax(170px, 1fr)" not in sheet
    # A long value wraps inside its own item instead of widening the row.
    item = _css_rule(sheet, ".kv > span {")
    assert "min-width:0" in item and "max-width:100%" in item


def test_long_mono_facts_break_rather_than_overflow_their_neighbour():
    # A 40-char SHA painted over Date; a 136-char digest-pinned image ref ran off
    # the page. break-all rather than break-word: break-word would break these
    # too once the item is narrower than the token, but the opportunities it
    # introduces are not counted toward min-content, so each mono fact would keep
    # the whole token as its intrinsic minimum and the row would hold only for as
    # long as `.kv > span` keeps min-width:0. break-all's breaks do count.
    rule = _css_rule(gen.run_area_stylesheet(), ".kv .val.mono {")
    assert "word-break:break-all" in rule
    assert "break-word" not in rule

    image = (
        "rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0"
        "@sha256:" + "4" * 64
    )
    commit = "78d1ae686dc3a786e8cfdb1216efc4b7516c8896"
    page = gen.build_case_index_html(
        {
            "case": "c",
            "observed": {},
            "commit": commit,
            "date": "2026-08-23",
            "container_image": image,
        },
        [],
        built_refs=[],
        up="../../",
    )
    # Both survive in full: the fix is wrapping, not truncation. The image digest
    # is what makes the run reproducible at all.
    assert commit in page
    assert image in page


def test_reproduce_label_sits_outside_the_command_box():
    # The box is a copy target. With the label inside it, the box read as a code
    # block whose first token was REPRODUCE, and selecting it to copy the command
    # picked the label up too.
    command = "aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gemm-object.yaml"
    page = gen.build_case_index_html(
        {"case": "c", "observed": {}, "command": command},
        [], built_refs=[], up="../../",
    )
    assert '<p class="cap">Reproduce</p><div class="repro"><code>' in page
    assert 'class="lbl">Reproduce' not in page

    # The dashboard's Tab 2 strip is the same markup, so the two cannot diverge.
    strip = gen._survey_howto_html({"command": command})
    assert '<p class="cap">Reproduce</p><div class="repro"><code>' in strip
    assert 'class="lbl">Reproduce' not in strip
    assert gen._survey_howto_html({"command": ""}) == ""

    # Holding only the command, the box no longer needs a flex row.
    assert "display:block" in _css_rule(gen.run_area_stylesheet(), ".repro {")


def test_section_headings_own_their_separation_from_the_previous_section():
    # `.cap` shipped with margin-top:0, so the space above a heading was whatever
    # the previous element left -- 14px, against the 8px binding a heading to its
    # own content -- and it was delegated to a `.cap + .table-wrap` adjacency
    # rule that any element inserted between the two silently broke.
    sheet = gen.run_area_stylesheet()
    above, below = 24, 8
    assert f"margin:{above}px 0 {below}px" in _css_rule(sheet, ".cap {")
    # Separation has to be clearly greater than the binding or neither reads as
    # grouping the section with its heading.
    assert above >= 3 * below
    # A panel's first heading must not double the panel's own padding.
    assert ".cap:first-child { margin-top:0; }" in sheet
    # The fragile adjacency rule is removed, not worked around. Matching the
    # declaration rather than the bare selector, which the comment above it names.
    assert ".cap + .table-wrap {" not in sheet


def test_rebuild_section_renders_its_commands_as_one_runnable_block():
    # `_rebuild_hints` flattens path/what/commands/caveat into one sentence. That
    # is right for REPRODUCE.md and wrong here: the commands became inline <code>
    # runs joined by a prose ";", with the sentence period stuck to the final
    # path, so nothing on the page could be selected and run.
    plan = gen.rebuild_plan(["fixtures/isa/consan_gemm_f32.hsaco"], target="gfx950")
    html = gen._rebuild_section_html(plan)
    entry = plan[0]

    block = re.search(r"<pre><code>(.*?)</code></pre>", html, re.S)
    assert block
    body = block.group(1)
    # Every command, one per line, in a single block.
    assert body.split("\n") == [gen._esc(command) for command in entry["commands"]]
    # No prose inside the copy target...
    assert entry["what"] not in body
    assert entry["caveat"] not in body
    # ...no "; " joining the commands as prose, and no sentence period abutting
    # the final path (which made it a path that does not exist).
    assert "; " not in body
    assert not body.endswith(".")
    # One artifact is a titled block, not a one-item bullet list.
    assert "<ul" not in html and "<li>" not in html
    # what and caveat are still shown, outside the block.
    assert entry["what"] in html and entry["caveat"] in html


def test_rebuild_section_names_an_unknown_reference_without_inventing_a_command():
    # Same contract as the Markdown hints: never a plausible-looking guess.
    plan = gen.rebuild_plan(["fixtures/other/x"], target="gfx950")
    assert plan[0]["commands"] == []
    html = gen._rebuild_section_html(plan)
    assert "fixtures/other/x" in html
    assert "<pre>" not in html
    assert gen._rebuild_section_html([]) == ""


def test_final_rebuild_block_is_marked_by_class_not_last_of_type():
    # `.rb:last-of-type` keys off the tag, not the class: it matches the last
    # `div` sibling only if that div happens to be a `.rb`. On a real page it
    # never is -- a case with rebuild blocks has built refs, so `build_case_env`
    # also gives it an "Artifacts not published" `.table-wrap` after them.
    sheet = gen.run_area_stylesheet()
    assert ".rb-last {" in sheet
    assert ".rb:last-of-type" not in sheet

    plan = gen.rebuild_plan(
        ["fixtures/isa/lds.hsaco", "fixtures/bin/consan_load"], target="gfx950"
    )
    html = gen._rebuild_section_html(plan)
    assert html.count('<div class="rb">') == len(plan) - 1
    assert html.count('<div class="rb rb-last">') == 1
    # It is the last one, not merely one of them.
    assert html.rindex('<div class="rb rb-last">') > html.rindex('<div class="rb">')

    # A single-artifact case still gets the marker, and it reaches the page.
    single = gen._rebuild_section_html(plan[:1])
    assert single.count('<div class="rb rb-last">') == 1
    assert '<div class="rb">' not in single
    page = gen.build_case_index_html(
        {"case": "c", "observed": {}},
        [],
        built_refs=["fixtures/isa/lds.hsaco"],
        up="../../",
    )
    assert 'class="rb rb-last"' in page


def test_stored_rebuild_plan_is_only_trusted_when_both_renderers_can_read_it():
    # A retained area's own env.json is preferred over recomputation so history
    # keeps its instructions. But it is read off the data branch, and both
    # renderers index the four keys directly -- so a half-written or hand-edited
    # entry has to fall back, not KeyError the whole dashboard render.
    refs = ["fixtures/isa/lds.hsaco"]
    full = gen.rebuild_plan(refs, target="gfx950")
    assert gen.plan_from_env({"rebuild": full, "target": "gfx950"}, refs) is full

    for missing in sorted(gen._REBUILD_KEYS):
        partial = [{k: v for k, v in full[0].items() if k != missing}]
        recovered = gen.plan_from_env({"rebuild": partial, "target": "gfx950"}, refs)
        assert recovered == full, missing
        # Both renderers survive the fallback.
        assert gen._rebuild_hints(recovered)
        assert gen._rebuild_section_html(recovered)

    # A non-list or a list of non-dicts falls back the same way.
    for junk in ("nope", [1, 2], {"path": "x"}):
        assert gen.plan_from_env({"rebuild": junk, "target": "gfx950"}, refs) == full

    # Key presence is not enough: `commands` is the one field the renderers
    # iterate rather than stringify, so its type is part of the contract. An int
    # is a TypeError mid-render and a string degrades into one command per
    # character -- either way the whole dashboard render, not one area, breaks.
    for bad in (1, "hipcc x.hip", None, {"0": "hipcc x.hip"}):
        entry = [{**full[0], "commands": bad}]
        recovered = gen.plan_from_env({"rebuild": entry, "target": "gfx950"}, refs)
        assert recovered == full, bad
        assert gen._rebuild_hints(recovered)
        assert gen._rebuild_section_html(recovered)

    # A list of the right type but the wrong elements does not crash -- it
    # publishes the element's repr as a command line, in the one block on the
    # page that promises to be runnable. Fabricating a command is worse than
    # falling back to the module's own tables, so the elements are checked too.
    for bad_elements in ([{"a": 1}], [123], [None], ["ok", 7]):
        entry = [{**full[0], "commands": bad_elements}]
        recovered = gen.plan_from_env({"rebuild": entry, "target": "gfx950"}, refs)
        assert recovered == full, bad_elements
        rendered = gen._rebuild_section_html(recovered) + " ".join(
            gen._rebuild_hints(recovered)
        )
        assert "{'a': 1}" not in rendered and "None" not in rendered

    # An empty command list is legitimate (an unrecognised reference), so it is
    # trusted rather than recomputed.
    empty = [{**full[0], "commands": []}]
    assert gen.plan_from_env({"rebuild": empty, "target": "gfx950"}, refs) is empty


def test_reproduce_md_keeps_the_flattened_markdown_hints():
    # The page renders structurally now. REPRODUCE.md consumes the Markdown
    # verbatim, so it must be untouched by that change.
    refs = ["fixtures/isa/lds.hsaco"]
    env = gen.build_case_env(
        case="waitcheck", cls="guardrail", recipe="r.yaml", command="c",
        meta={"gpu": "gfx950"}, summary={}, report=None,
        built_refs=refs, inputs=[],
    )
    md = gen.build_reproduce_md(env, built_refs=refs)
    for hint in gen._rebuild_hints(gen.rebuild_plan(refs, target="gfx950")):
        assert hint in md


def test_files_caption_discloses_an_input_that_is_excluded_by_design():
    # "Every file published for this case" is true and, on its own, misleading:
    # for a kernel-source recipe the one input the recipe names is CI-built and
    # recorded by digest instead of copied. Nothing linked the two sections, so a
    # reader could not tell a deliberate omission from a missing file.
    env = {
        "case": "c",
        "observed": {},
        "logs_published": True,
        "artifacts_not_published": [
            {"path": "fixtures/isa/consan_gemm_f32.hsaco", "sha256": "ab"}
        ],
    }
    files = [("sanitizer_report.json", 12)]
    caption = gen._files_caption(env, files)
    assert "1 required input is" in caption
    assert f'href="#{gen._NOT_PUBLISHED_ID}"' in caption
    assert "its SHA-256" in caption

    page = gen.build_case_index_html(env, files, built_refs=[], up="../../")
    # The anchor the caption points at exists on the page.
    assert f'id="{gen._NOT_PUBLISHED_ID}"' in page
    # And the header no longer makes the absolute claim the caption walks back.
    assert "everything needed to reproduce" not in page

    # Plural agreement, and no claim at all when there is nothing to disclose.
    two = {
        **env,
        "artifacts_not_published": [
            {"path": "a", "sha256": "ab"}, {"path": "b", "sha256": "cd"}
        ],
    }
    assert "2 required inputs are" in gen._files_caption(two, files)
    assert "their SHA-256s" in gen._files_caption(two, files)
    assert gen._files_caption({**env, "artifacts_not_published": []}, files) == (
        "Every file published for this case."
    )


def test_files_caption_claims_only_what_the_listing_actually_carries():
    # `artifacts_not_published` records sha256 from the report by basename, so a
    # bare `isa_dir: fixtures/isa` reference has none and its row is an em dash.
    # The caption must not promise a digest for it. Rebuild commands are never
    # promised under this anchor either: that table is Path + SHA-256, the
    # commands are their own section above it, and an unrecognised reference has
    # no command at all.
    env = {"case": "c", "observed": {}, "logs_published": True}
    files = [("sanitizer_report.json", 12)]

    undigested = {**env, "artifacts_not_published": [{"path": "fixtures/isa"}]}
    caption = gen._files_caption(undigested, files)
    assert "1 required input is" in caption
    assert f'href="#{gen._NOT_PUBLISHED_ID}"' in caption
    assert "SHA-256" not in caption

    # One entry without a digest is enough to drop the claim for the whole list.
    mixed = {
        **env,
        "artifacts_not_published": [
            {"path": "fixtures/isa/x.hsaco", "sha256": "ab"},
            {"path": "fixtures/isa", "sha256": None},
        ],
    }
    assert "SHA-256" not in gen._files_caption(mixed, files)

    # No caption on any of these paths claims a rebuild command under the
    # anchor, because the anchored table does not carry one.
    for case in (undigested, mixed):
        assert "rebuild" not in gen._files_caption(case, files).lower()


def _np_body_rows(page: str) -> int:
    """Body rows of the "Artifacts not published" table, header excluded."""
    anchor = f'id="{gen._NOT_PUBLISHED_ID}"'
    assert anchor in page, "the page has no artifacts-not-published section"
    start = page.index(anchor)
    return page[start : page.index("</table>", start)].count("<tr><td")


def _np_md_rows(md: str) -> int:
    """Rows of REPRODUCE.md's "Artifacts not published" table."""
    assert "## Artifacts not published" in md, "REPRODUCE.md has no such section"
    section = md[md.index("## Artifacts not published") :].split("\n\n")[2]
    return len([line for line in section.splitlines() if line.startswith("| `")])


def test_every_reading_of_artifacts_not_published_agrees_and_survives_a_bad_entry():
    # The caption counts this field, agrees its noun and verb with the count and
    # links to it; the page renders it as a table; REPRODUCE.md renders it again.
    # All three read it off the data branch, and two of them used to index
    # `item["path"]` directly -- so one hand-edited env.json raised KeyError (a
    # dict with no path) or TypeError (a bare string) and failed the whole
    # dashboard render, while `refresh_published_case_area` was already filtering
    # the same field to derive built_refs. One reader now serves all three, so
    # the count in the prose cannot disagree with the rows beneath it.
    env = {"case": "c", "observed": {}, "logs_published": True}
    files = [("sanitizer_report.json", 12)]

    good = [{"path": "fixtures/isa/x.hsaco", "sha256": "ab"}]
    for stored in (
        good,
        [{"sha256": "ab"}],              # dict with no path
        ["fixtures/isa/x.hsaco"],        # bare string, as hand-edited
        [None],
        good + [{"sha256": "cd"}],       # one good, one unreadable
    ):
        case = {**env, "artifacts_not_published": stored}
        page = gen.build_case_index_html(case, files, built_refs=[], up="../../")
        md = gen.build_reproduce_md(case, built_refs=[])

        # The count in the caption is the number of rows in the table it links
        # to, and the same number of rows reaches REPRODUCE.md.
        assert _np_body_rows(page) == len(stored), stored
        assert _np_md_rows(md) == len(stored), stored

        # Nothing is dropped: an unreadable entry is still disclosed, because
        # silently omitting it restores the overstatement this section fixes.
        caption = gen._files_caption(case, files)
        assert f"{len(stored)} required input" in caption, stored
        assert "It is not the whole recipe" in caption, stored

    # The digest claim still tracks the digest column and nothing else. An entry
    # that lost its path but kept its sha256 does render a SHA-256, so the claim
    # holds; one with no digest withdraws it, whether or not it is readable.
    assert "SHA-256" in gen._files_caption(
        {**env, "artifacts_not_published": good}, files
    )
    assert "SHA-256" in gen._files_caption(
        {**env, "artifacts_not_published": good + [{"sha256": "cd"}]}, files
    )
    for undigested in ([{"path": "y"}], ["fixtures/isa/y.hsaco"], [None]):
        assert "SHA-256" not in gen._files_caption(
            {**env, "artifacts_not_published": good + undigested}, files
        ), undigested

    # A field that is not a list at all yields no section, not a row per char.
    for junk in ("fixtures/isa/x.hsaco", {"path": "x"}, 7):
        case = {**env, "artifacts_not_published": junk}
        assert gen._not_published_rows(case) == []
        assert f'id="{gen._NOT_PUBLISHED_ID}"' not in gen.build_case_index_html(
            case, files, built_refs=[], up="../../"
        )
        assert "Artifacts not published" not in gen.build_reproduce_md(
            case, built_refs=[]
        )
        assert gen._files_caption(case, files) == "Every file published for this case."

    # A wrongly-typed value inside a well-formed entry is a dash too, not its
    # repr: `{"path": ["a", "b"]}` publishing `['a', 'b']` would invent a path
    # the manifest never recorded, which is the fault this reader exists to stop.
    assert gen._not_published_rows(
        {"artifacts_not_published": [{"path": ["a", "b"], "sha256": 7}]}
    ) == [("", "")]
    page = gen.build_case_index_html(
        {**env, "artifacts_not_published": [{"path": ["a", "b"]}]},
        files,
        built_refs=[],
        up="../../",
    )
    assert "['a', 'b']" not in page and _np_body_rows(page) == 1


def test_a_malformed_artifact_list_does_not_abort_a_retained_area_refresh(tmp_path):
    # The renderers being careful is not enough on its own: the nightly re-renders
    # every retained area through refresh_published_case_area, which derived
    # built_refs from this same field. While it iterated the field itself, a
    # non-list value raised TypeError there -- before either renderer was
    # reached -- so one hand-edited env.json still aborted the whole render.
    area = tmp_path / "dashboard" / "runs" / "r" / "survey" / "c"
    area.mkdir(parents=True)
    (area / "sanitizer_report.json").write_text("{}")

    for junk in (7, "fixtures/isa/x.hsaco", {"path": "x"}, [None], [{"sha256": "ab"}]):
        (area / "env.json").write_text(
            json.dumps({"case": "c", "observed": {}, "artifacts_not_published": junk})
        )
        assert (
            gen.refresh_published_case_area(area, tmp_path / "dashboard", logs=True)
            is True
        ), junk
        assert "['a', 'b']" not in (area / "index.html").read_text()


def test_genco_rebuild_cleans_up_its_temporary_object():
    # The command runs in the reader's repo root, so without the trailing rm it
    # drops an untracked tmp.o next to pyproject.toml. The workflow's own
    # genco_object uses mktemp + rm -f.
    entry = gen.rebuild_plan(["fixtures/isa/lds.hsaco"], target="gfx950")[0]
    conditional = [c for c in entry["commands"] if c.startswith("if ")][0]
    assert conditional.endswith("&& rm -f tmp.o")
    # Every command that creates tmp.o is paired with a command that removes it.
    creates = [c for c in entry["commands"] if "-o tmp.o" in c]
    removes = [c for c in entry["commands"] if "rm -f tmp.o" in c]
    assert len(creates) == len(removes) == 1


def test_files_caption_only_claims_gzipped_logs_when_a_log_is_listed():
    env = {"case": "c", "observed": {}, "logs_published": True}
    with_log = gen.build_case_index_html(
        env, [("consan/consan.log.gz", 34)], built_refs=[], up="../../"
    )
    assert "Logs are gzipped." in with_log

    # An in-window case that produced no log at all claimed one anyway.
    without = gen.build_case_index_html(
        env, [("sanitizer_report.json", 12)], built_refs=[], up="../../"
    )
    assert "Every file published for this case." in without
    assert "gzipped" not in without

    pruned = gen.build_case_index_html(
        {**env, "logs_published": False}, [("sanitizer_report.json", 12)],
        built_refs=[], up="../../",
    )
    assert "pruned" in pruned and "gzipped" not in pruned


def test_genco_rebuild_encodes_the_bundle_check_as_one_command():
    # A consumer executing `commands` in order cannot branch on prose, so the
    # raw-ELF-vs-bundle decision has to live inside a command.
    entry = gen.rebuild_plan(["fixtures/isa/tiny.hsaco"], target="gfx950")[0]
    conditional = [c for c in entry["commands"] if c.startswith("if ")]
    assert len(conditional) == 1
    assert "__CLANG_OFFLOAD_BUNDLE__" in conditional[0]
    assert "clang-offload-bundler" in conditional[0]
    # ...and the else-branch copies rather than silently skipping.
    assert "else cp tmp.o recipes/sanitizers/fixtures/isa/tiny.hsaco; fi" in conditional[0]


def test_case_index_page_carries_the_reproduction_details(tmp_path, monkeypatch):
    # #384 requires the code-object SHA-256 on the landing page, and decision 9
    # renders REPRODUCE.md's reproduction details there. The artifacts table alone
    # cannot carry the digests: the waitcheck GEMM recipe names only a bare
    # `isa_dir: fixtures/isa`, so it has one row and no per-object digest.
    out = _publish_with_logs(tmp_path, monkeypatch, run_ids=["2026-08-05-33"])
    case = out / "runs" / "2026-08-05-33" / "waitcheck"
    page = (case / "index.html").read_text(encoding="utf-8")
    env = json.loads((case / "env.json").read_text(encoding="utf-8"))

    assert "Recorded digests" in page
    assert env["digests"]["code_object:sol_1.hsaco"] == "93f09ae670abcdef"
    for key, value in env["digests"].items():
        assert key in page and value in page
    # The env a reproduction needs is on the page too, rendered from this area's
    # own required_env, so the page and REPRODUCE.md cannot drift.
    note = gen.required_env_note(env["required_env"])
    assert "ROCJITSU_PREBUILT" in page
    assert "<code>ROCJITSU_PREBUILT</code>" in page
    assert note in (case / "REPRODUCE.md").read_text(encoding="utf-8")
    # This is a waitcheck case, so ConSan's variables are not claimed for it.
    assert [item["var"] for item in env["required_env"]] == ["ROCJITSU_PREBUILT"]
    assert "HSA_TOOLS_LIB" not in page
    # Execution status was in the MD twin but missing from the page.
    assert "Execution" in page
    # The rebuild commands render as code, not as literal Markdown backticks.
    assert "prepare_gemm_isa.py" in page
    assert "`" not in page


def test_required_env_is_per_case_and_the_note_names_exactly_it():
    # The prose has always said the four HSA/RJ variables are what ConSan
    # *additionally* requires, so listing them for a waitcheck reproduction would
    # make the machine-readable field contradict it.
    waitcheck = gen.required_env_for(case="waitcheck", report=_waitcheck_report())
    assert [item["var"] for item in waitcheck] == ["ROCJITSU_PREBUILT"]

    consan = gen.required_env_for(case="consan-racy", report=_consan_racy_report())
    assert [item["var"] for item in consan] == [item["var"] for item in gen._REQUIRED_ENV]
    # The internal flag is not leaked into the published manifest.
    assert all("consan_only" not in item for item in consan)

    # With no usable report the case name decides, following the naming convention
    # every guardrail and survey case uses.
    assert len(gen.required_env_for(case="consan-gemm", report=None)) == len(consan)
    assert len(gen.required_env_for(case="waitcheck-gemm", report=None)) == 1

    # The note names exactly the variables it was given -- no more, no fewer.
    for required in (waitcheck, consan):
        note = gen.required_env_note(required)
        named = set(re.findall(r"`([A-Z][A-Z0-9_]+)`", note))
        assert named == {item["var"] for item in required}
        for item in required:
            assert item["set_by"] in {"operator", "aorta"}
            assert item["purpose"]
    assert gen.required_env_note([]) == ""


def test_rebuild_plan_is_machine_readable_and_prose_is_generated_from_it():
    # A consumer of aorta.sanitizer_run_area/0.1 should read commands as data
    # rather than parse them out of English, and the prose must not be able to
    # disagree with them.
    refs = ["fixtures/isa/lds.hsaco", "fixtures/bin/lds_dispatch", "fixtures/other/x"]
    plan = gen.rebuild_plan(refs, target="gfx950")
    assert [entry["path"] for entry in plan] == refs

    lds = plan[0]
    assert lds["commands"] == [
        gen._ROCM_LLVM_PATH_EXPORT,
        "mkdir -p recipes/sanitizers/fixtures/isa",
        "hipcc --genco --offload-arch=gfx950 "
        "recipes/sanitizers/fixtures/kernels/lds_reduce.hip -o tmp.o",
        "if head -c 24 tmp.o | grep -qF __CLANG_OFFLOAD_BUNDLE__; then "
        "clang-offload-bundler --type=o --unbundle --input=tmp.o "
        "--targets=hipv4-amdgcn-amd-amdhsa--gfx950 "
        "--output=recipes/sanitizers/fixtures/isa/lds.hsaco; "
        "else cp tmp.o recipes/sanitizers/fixtures/isa/lds.hsaco; fi && rm -f tmp.o",
    ]
    # The bundle check lives in a command, not only in the prose caveat, so an
    # automated consumer can execute the branch.
    assert "__CLANG_OFFLOAD_BUNDLE__" in " ".join(lds["commands"])
    assert "--genco" in lds["caveat"]
    hipcc = plan[1]["commands"][2]
    assert "-O1 -g" not in hipcc  # loader binaries are built without
    assert "-DLDS_HSACO=" in hipcc
    # An unknown reference yields no command rather than a plausible wrong one.
    assert plan[2]["commands"] == []

    # Every command in the plan appears verbatim in the rendered prose.
    hints = gen._rebuild_hints(plan)
    assert len(hints) == len(plan)
    for index, entry in enumerate(plan):
        for command in entry["commands"]:
            assert f"`{command}`" in hints[index], entry["path"]
    assert hints[2] == "fixtures/other/x"


def test_env_json_records_required_env_and_rebuild_commands(tmp_path, monkeypatch):
    out = _publish_with_logs(tmp_path, monkeypatch, run_ids=["2026-08-05-33"])
    env = json.loads(
        (out / "runs/2026-08-05-33/survey/consan-gemm/env.json").read_text(
            encoding="utf-8"
        )
    )

    # A ConSan case, so it carries the full set.
    assert [item["var"] for item in env["required_env"]] == [
        item["var"] for item in gen._REQUIRED_ENV
    ]
    rebuild = {entry["path"]: entry for entry in env["rebuild"]}
    # The recipe's built refs each carry their own commands, at this run's target.
    assert "fixtures/isa/consan_gemm_f32.hsaco" in rebuild
    gemm_isa = rebuild["fixtures/isa/consan_gemm_f32.hsaco"]["commands"]
    assert any("prepare_gemm_isa.py" in command for command in gemm_isa)
    gemm_bin = " ".join(rebuild["fixtures/bin/consan_gemm_load"]["commands"])
    assert "--offload-arch=gfx950" in gemm_bin
    assert "-DOBJECT=" in gemm_bin


def test_md_code_to_html_escapes_before_converting_backticks():
    assert gen._md_code_to_html("use `X` now") == "use <code>X</code> now"
    # Escaping happens first, so text cannot inject markup of its own.
    assert gen._md_code_to_html("<b>&`x`") == "&lt;b&gt;&amp;<code>x</code>"
    assert gen._md_code_to_html("plain") == "plain"


def test_artifact_digest_index_covers_every_key_shape_a_sanitizer_writes():
    # The unpublished-artifact table looks digests up by basename, because the two
    # sanitizers name the same kind of artifact under different report keys.
    index = gen.artifact_digest_index({
        "path": "/tmp/build/tools/rj_waitcheck", "sha256": "aa",
        "command": "/w/fixtures/bin/consan_gemm_load", "command_sha256": "bb",
        "hook": "/w/librocjitsu_dbi_hooks.so", "hook_sha256": "cc",
        "code_object:consan_gemm_f32.hsaco": "dd",
    })
    assert index == {
        "rj_waitcheck": "aa", "consan_gemm_load": "bb",
        "librocjitsu_dbi_hooks.so": "cc", "consan_gemm_f32.hsaco": "dd",
    }
    # A half-recorded pair contributes nothing rather than a partial entry.
    assert gen.artifact_digest_index({"command": "/w/x", "sha256": "aa"}) == {}
    assert gen.artifact_digest_index({}) == {}


def test_run_area_names_the_consan_repro_binary_digest_not_an_em_dash():
    # ConSan records the repro binary as command/command_sha256. Keying only on
    # "code_object:" left every fixtures/bin/ artifact with a null digest, so the
    # "rebuild it and check the digest" instruction had nothing to check against.
    report = _consan_clean_report()
    report["checks"][0]["backend"] = {
        "command": "/w/fixtures/bin/consan_gemm_load", "command_sha256": "deadbeef",
    }
    env = gen.build_case_env(
        case="consan-gemm", cls="survey",
        recipe="recipes/sanitizers/daily-consan-gemm.yaml", command="aorta sweep run",
        meta={}, summary={}, report=report,
        built_refs=["fixtures/bin/consan_gemm_load", "fixtures/isa"], inputs=[],
    )
    by_path = {item["path"]: item["sha256"] for item in env["artifacts_not_published"]}
    assert by_path["fixtures/bin/consan_gemm_load"] == "deadbeef"
    # A bare directory reference still has no digest to record; that stays honest.
    assert by_path["fixtures/isa"] is None
    assert "deadbeef" in gen.build_reproduce_md(env, built_refs=list(by_path))


def test_case_dir_up_matches_the_published_depth_of_each_tab(tmp_path):
    out = tmp_path / "dashboard"
    guardrail = out / "runs" / "2026-08-05-33" / "waitcheck"
    survey = out / "runs" / "2026-08-05-33" / "survey" / "consan-gemm"
    guardrail.mkdir(parents=True)
    survey.mkdir(parents=True)
    # A guardrail case sits one level shallower than a survey case, so a single
    # hardcoded "up" would break the back-link on one of the two tabs.
    assert gen.case_dir_up(guardrail, out) == ("../../../", "../")
    assert gen.case_dir_up(survey, out) == ("../../../../", "../../")


def test_main_publishes_full_case_dirs_with_gzipped_logs(tmp_path, monkeypatch):
    out = _publish_with_logs(tmp_path, monkeypatch, run_ids=["2026-08-05-33"])
    case = out / "runs" / "2026-08-05-33" / "waitcheck"
    survey_case = out / "runs" / "2026-08-05-33" / "survey" / "consan-gemm"

    # The logs the verdict came from are published, gzipped, and still readable.
    log = case / "waitcheck" / "waitcheck-0.log.gz"
    assert log.is_file() and not (case / "waitcheck" / "waitcheck-0.log").exists()
    assert gzip.decompress(log.read_bytes()) == b"scan output\n"
    survey_log = survey_case / "consan" / "consan.log.gz"
    assert gzip.decompress(survey_log.read_bytes()) == b"hook timeout\n"

    # Every run area carries the inputs needed to reproduce it.
    for area in (case, survey_case):
        assert (area / "sanitizer_report.json").is_file()
        assert (area / "recipe.yaml").is_file()
        assert (area / "REPRODUCE.md").is_file()
        assert (area / "env.json").is_file()
        assert (area / "index.html").is_file()
    # Source-level recipe inputs are copied; the CI-built artifacts are not.
    assert (case / "inputs" / "fixtures" / "gemm_shapes_unique.csv").is_file()
    assert not list(case.rglob("*.hsaco"))
    assert not (survey_case / "inputs" / "fixtures" / "isa").exists()


def test_run_area_records_unpublished_artifacts_by_digest(tmp_path, monkeypatch):
    out = _publish_with_logs(tmp_path, monkeypatch, run_ids=["2026-08-05-33"])
    case = out / "runs" / "2026-08-05-33" / "waitcheck"

    env = json.loads((case / "env.json").read_text(encoding="utf-8"))
    assert env["schema"] == gen.RUN_AREA_ENV_SCHEMA
    assert env["class"] == "guardrail"
    assert env["recipe"] == "recipes/sanitizers/daily-waitcheck-gemm.yaml"
    assert env["command"] == (
        "aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gemm.yaml"
    )
    # The full SHA is recorded, not the display-shortened one, so the reproduce
    # instructions name a revision you can actually check out.
    assert env["commit"] == "0123456789abcdef"
    assert env["observed"]["verdict"] == "warn"
    assert env["digests"]["sha256"] == "472fcf288714beef"

    md = (case / "REPRODUCE.md").read_text(encoding="utf-8")
    assert "aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gemm.yaml" in md
    assert "git checkout 0123456789abcdef" in md
    assert "472fcf288714beef" in md

    # The survey case's recipe references built artifacts, so they are named with
    # their digest under "artifacts not published" rather than shipped.
    survey_env = json.loads(
        (out / "runs/2026-08-05-33/survey/consan-gemm/env.json").read_text(encoding="utf-8")
    )
    paths = [item["path"] for item in survey_env["artifacts_not_published"]]
    assert "fixtures/isa/consan_gemm_f32.hsaco" in paths
    survey_md = (
        out / "runs/2026-08-05-33/survey/consan-gemm/REPRODUCE.md"
    ).read_text(encoding="utf-8")
    assert "Artifacts not published" in survey_md
    # The GEMM object is *extracted* by prepare_gemm_isa.py, not compiled from a
    # .hip source, and the binary that loads it needs the -DOBJECT define. A
    # generic "hipcc it" hint would build a different file than the recorded
    # digest, which is the whole point of recording the digest.
    assert "prepare_gemm_isa.py" in survey_md
    # Root-relative in the command, because that is where the clone leaves you.
    assert (
        "--consan-object recipes/sanitizers/fixtures/isa/consan_gemm_f32.hsaco"
        in survey_md
    )
    assert "-DOBJECT=" in survey_md
    assert "recipes/sanitizers/fixtures/kernels/consan_load.hip" in survey_md


def test_case_index_lists_every_published_file(tmp_path, monkeypatch):
    out = _publish_with_logs(tmp_path, monkeypatch, run_ids=["2026-08-05-33"])
    case = out / "runs" / "2026-08-05-33" / "survey" / "consan-gemm"
    page = (case / "index.html").read_text(encoding="utf-8")

    # Pages does not auto-index a directory, so this page is what makes the
    # footer's directory link resolve; it must link everything actually present.
    for rel, _size in gen.run_area_files(case):
        assert f'href="{rel}"' in page
    assert 'href="consan/consan.log.gz"' in page
    assert 'href="sanitizer_report.json"' in page
    # It never links itself, and every link resolves on disk.
    assert 'href="index.html"' not in page
    for href in re.findall(r'href="([^"#]+)"', page):
        if href.startswith("http"):
            continue
        target = (case / href).resolve()
        assert target.exists(), href
        if target.is_dir():
            assert (target / "index.html").is_file(), href


def test_keep_logs_window_prunes_bulk_but_keeps_report_and_page(tmp_path, monkeypatch):
    # Reports are retained for --keep runs; logs and copied inputs only for the
    # newest --keep-logs, so the data branch stays bounded (#384).
    out = _publish_with_logs(
        tmp_path,
        monkeypatch,
        run_ids=["2026-08-05-33", "2026-08-04-22", "2026-08-03-11"],
        keep_logs=2,
        survey=False,
    )
    latest = out / "runs" / "2026-08-05-33" / "waitcheck"
    inside = out / "runs" / "2026-08-04-22" / "waitcheck"
    outside = out / "runs" / "2026-08-03-11" / "waitcheck"

    assert (inside / "waitcheck" / "waitcheck-0.log.gz").is_file()
    # The recipe and its inputs are pinned to the checkout that ran the case, so
    # only the run this job staged gets them -- publishing today's recipe into an
    # older area would contradict the commit its env.json names.
    assert (latest / "recipe.yaml").is_file()
    assert (latest / "inputs").is_dir()
    assert not (inside / "recipe.yaml").exists()
    assert not (inside / "inputs").exists()

    # The pruned run keeps what makes it still readable and still drillable.
    assert (outside / "sanitizer_report.json").is_file()
    assert (outside / "index.html").is_file()
    assert (outside / "env.json").is_file()
    assert not list(outside.rglob("*.log.gz"))
    assert not (outside / "inputs").exists()
    assert not (outside / "recipe.yaml").exists()
    # Both renderings say the area was pruned rather than promising its logs.
    env = json.loads((outside / "env.json").read_text(encoding="utf-8"))
    assert env["logs_published"] is False
    assert env["inputs"] == []
    assert "pruned" in (outside / "REPRODUCE.md").read_text(encoding="utf-8")
    assert json.loads((inside / "env.json").read_text(encoding="utf-8"))["logs_published"] is True
    # An empty per-sanitizer log dir is removed rather than left as a stub, and
    # the page lists only files that survived -- no dead links.
    assert not (outside / "waitcheck").exists()
    page = (outside / "index.html").read_text(encoding="utf-8")
    assert ".log.gz" not in page
    for href in re.findall(r'href="([^"#]+)"', page):
        if not href.startswith("http"):
            assert (outside / href).resolve().exists(), href


def test_run_index_lists_survey_cases_and_run_areas(tmp_path):
    root = tmp_path / "runs"
    _write_history_run(root, "2026-08-05-33")
    run = gen.runs_from_history_root(root, _baselines())[0]
    survey = gen.survey_cases_from_spec(
        {"cases": [{
            "name": "consan-gemm", "label": "gemm - ConSan",
            "command": "aorta sweep run --recipe recipes/sanitizers/daily-consan-gemm.yaml",
            "report_rel": "runs/2026-08-05-33/survey/consan-gemm/sanitizer_report.json",
            "report": _consan_racy_report(),
        }]}
    )

    page = gen.build_run_index_html(run, survey=survey)

    # Without this the run page lists only the three gated guardrail recipes, so
    # the top-nav "raw reports" link never mentions the survey cases beside them.
    assert "Workload survey" in page
    assert "daily-consan-gemm.yaml" in page
    # A spec entry carries no area_rel: its report parsed, but nothing staged the
    # directory, so the case is listed with no run-area link rather than a 404.
    assert "survey/consan-gemm/" not in page
    assert "&mdash;" in page
    # Once the caller has actually published the area it may claim it.
    staged = [{**survey[0], "staged": True}]
    linked = gen.build_run_index_html(run, survey=staged)
    assert 'href="survey/consan-gemm/">run area</a>' in linked
    # Guardrail rows link their own run area alongside the raw report.
    assert 'href="waitcheck/">run area</a>' in page
    assert 'href="waitcheck/sanitizer_report.json"' in page

    # A run with no survey cases renders exactly as before (no empty section).
    assert "Workload survey" not in gen.build_run_index_html(run)


def test_keep_logs_window_also_bounds_survey_logs(tmp_path, monkeypatch):
    """The log window must bound the survey areas, not just the guardrail cases.

    A survey area is only ever published under the run that was latest at the
    time, and then persists on the data branch under that run -- so it is only
    reachable for pruning on a *later* publish. Publishing once per run into one
    output tree is what the nightly does to the ``sanitizer-results`` branch, and
    it is the only way this shows up: a single publish leaves every survey area
    inside the window and looks correct.
    """
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    run_ids = ["2026-08-03-11", "2026-08-04-22", "2026-08-05-33"]
    for run_id in run_ids:
        run_dir = _write_history_run(dashboard / "runs", run_id)
        (run_dir / "consan-racy" / "consan").mkdir(parents=True)
        (run_dir / "consan-racy" / "consan" / "consan.log").write_text("race\n")
        info = tmp_path / f"informational-{run_id}"
        (info / "consan-gemm" / "consan").mkdir(parents=True)
        (info / "consan-gemm" / "sanitizer_report.json").write_text(
            json.dumps(_informational_report("combined_hook_timeout"))
        )
        (info / "consan-gemm" / "consan" / "consan.log").write_text("hook timeout\n")
        monkeypatch.setattr(sys, "argv", [
            "gen_sanitizer_dashboard",
            "--history-root", str(dashboard / "runs"),
            "--baselines",
            str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
            "--out-dir", str(dashboard),
            "--keep-logs", "1",
            "--informational-results-dir", str(info),
        ])
        assert gen.main() == 0

    latest = dashboard / "runs/2026-08-05-33/survey/consan-gemm"
    assert (latest / "consan" / "consan.log.gz").is_file()

    for run_id in run_ids[:-1]:
        area = dashboard / "runs" / run_id / "survey" / "consan-gemm"
        # ConSan survey logs are the bulkiest thing published; unbounded here they
        # would outlive the guardrail logs beside them by --keep / --keep-logs.
        assert not list(area.rglob("*.log.gz")), run_id
        assert not list(area.rglob("*.log")), run_id
        # Still readable and still drillable, like a pruned guardrail area.
        assert (area / "sanitizer_report.json").is_file(), run_id
        env = json.loads((area / "env.json").read_text(encoding="utf-8"))
        assert env["logs_published"] is False, run_id
        # The page is re-rendered from that manifest, so it cannot keep linking
        # the log it no longer has.
        page = (area / "index.html").read_text(encoding="utf-8")
        assert ".log.gz" not in page, run_id
        for href in re.findall(r'href="([^"#]+)"', page):
            if not href.startswith("http"):
                assert (area / href).resolve().exists(), f"{run_id}: {href}"


def _publish_into(dashboard, monkeypatch, *, info=None, keep_logs=None):
    argv = [
        "gen_sanitizer_dashboard",
        "--history-root", str(dashboard / "runs"),
        "--baselines",
        str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
        "--out-dir", str(dashboard),
    ]
    if info is not None:
        argv += ["--informational-results-dir", str(info)]
    if keep_logs is not None:
        argv += ["--keep-logs", str(keep_logs)]
    monkeypatch.setattr(sys, "argv", argv)
    return gen.main()


def test_a_rejected_survey_report_never_persists_and_cannot_abort_a_later_publish(
    tmp_path, monkeypatch
):
    """A malformed best-effort report must not take the whole dashboard down.

    ``survey_cases_from_informational_dir`` deliberately skips a report whose
    nested shape makes the reduction raise. The copy loop ran before that
    decision was consulted, so the directory was published anyway with no
    manifest -- and on the *next* nightly the retrofit path re-read that same
    report and raised, aborting publication days later for a non-gating input.
    """
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    info = tmp_path / "informational"
    (info / "consan-gemm").mkdir(parents=True)
    # A dict (so the isinstance guard passes) whose nested shape breaks reduction.
    (info / "consan-gemm" / "sanitizer_report.json").write_text(
        json.dumps({"schema": "aorta.sanitizer_report/0.1", "checks": None})
    )
    assert gen.survey_cases_from_informational_dir(info, rel="runs/2026-08-05-33") == []

    _write_history_run(dashboard / "runs", "2026-08-05-33")
    assert _publish_into(dashboard, monkeypatch, info=info) == 0
    # Nothing the survey loader refused to read is left in the published history.
    assert not (dashboard / "runs/2026-08-05-33/survey/consan-gemm").exists()

    # And an orphan already on the data branch degrades instead of aborting.
    orphan = dashboard / "runs/2026-08-05-33/survey/consan-gemm"
    orphan.mkdir(parents=True)
    (orphan / "sanitizer_report.json").write_text(
        json.dumps({"schema": "aorta.sanitizer_report/0.1", "checks": None})
    )
    _write_history_run(dashboard / "runs", "2026-08-06-44")
    assert _publish_into(dashboard, monkeypatch) == 0
    assert not (orphan / "index.html").exists()


def test_keep_logs_zero_prunes_the_current_runs_survey_area_too(tmp_path, monkeypatch):
    # --keep-logs is documented as applying to "guardrail and survey areas alike".
    # The survey co-publish always gzipped and defaulted logs=True, so with
    # --keep-logs 0 the survey area was the one unbounded thing left.
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    info = tmp_path / "informational"
    (info / "consan-gemm" / "consan").mkdir(parents=True)
    (info / "consan-gemm" / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    (info / "consan-gemm" / "consan" / "consan.log").write_text("hook timeout\n")
    run_dir = _write_history_run(dashboard / "runs", "2026-08-05-33")
    (run_dir / "waitcheck" / "waitcheck").mkdir(parents=True)
    (run_dir / "waitcheck" / "waitcheck" / "waitcheck-0.log").write_text("scan\n")

    assert _publish_into(dashboard, monkeypatch, info=info, keep_logs=0) == 0

    guardrail = dashboard / "runs/2026-08-05-33/waitcheck"
    area = dashboard / "runs/2026-08-05-33/survey/consan-gemm"
    assert not list(guardrail.rglob("*.log*"))
    assert not list(area.rglob("*.log*")), "survey bulk escaped --keep-logs 0"
    env = json.loads((area / "env.json").read_text(encoding="utf-8"))
    assert env["logs_published"] is False
    # Still browsable, like a pruned guardrail area.
    assert (area / "index.html").is_file()
    assert ".log" not in (area / "index.html").read_text(encoding="utf-8")


def test_disjoint_history_and_output_trees_keep_the_survey_subtree(
    tmp_path, monkeypatch
):
    # With --history-root separate from <out-dir>/runs the output tree is cleared
    # and re-copied. Copying only the guardrail case dirs dropped the survey areas
    # the history holds, so retained runs lost their survey downloads.
    history = tmp_path / "history"
    out = tmp_path / "dashboard"
    _write_history_run(history, "2026-08-05-33")
    area = history / "2026-08-05-33" / "survey" / "consan-gemm"
    (area / "consan").mkdir(parents=True)
    (area / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    (area / "consan" / "consan.log.gz").write_bytes(b"x")
    (area / "env.json").write_text(json.dumps({
        "case": "consan-gemm", "command": "aorta sweep run", "observed": {"verdict": "error"},
    }))
    monkeypatch.setattr(sys, "argv", [
        "gen_sanitizer_dashboard",
        "--history-root", str(history),
        "--baselines",
        str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
        "--out-dir", str(out),
    ])
    assert gen.main() == 0

    published = out / "runs/2026-08-05-33/survey/consan-gemm"
    assert (published / "sanitizer_report.json").is_file()
    assert (published / "index.html").is_file()
    # The source history is untouched by the copy.
    assert (area / "sanitizer_report.json").is_file()


def test_published_survey_area_keeps_its_recorded_command(tmp_path):
    """Reconstructing a published run must not recompute its reproduce command.

    ``displayed_survey`` rebuilds the newest run's Tab 2 list from the areas
    published under it. Recomputing the command from the current
    ``_survey_recipe_for`` map meant a recipe rename would rewrite a historical
    run's advertised command, even though that area's manifest records the one it
    actually ran -- the same rule the run-area renderers already follow.
    """
    published = tmp_path / "survey"
    (published / "consan-gemm").mkdir(parents=True)
    (published / "consan-gemm" / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    (published / "consan-gemm" / "env.json").write_text(
        json.dumps({"case": "consan-gemm", "command": "aorta sweep run --recipe old/name.yaml"})
    )
    entries = gen.survey_cases_from_informational_dir(published, rel="runs/x")
    assert entries[0]["command"] == "aorta sweep run --recipe old/name.yaml"

    # A fresh sweep's results dir has no manifest, so the recipe map still applies.
    fresh = tmp_path / "informational"
    (fresh / "consan-gemm").mkdir(parents=True)
    (fresh / "consan-gemm" / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    fresh_entries = gen.survey_cases_from_informational_dir(fresh, rel="runs/x")
    assert fresh_entries[0]["command"] == (
        "aorta sweep run --recipe recipes/sanitizers/daily-consan-gemm.yaml"
    )
    # A malformed manifest degrades to the recipe map rather than raising.
    (published / "consan-gemm" / "env.json").write_text("[]")
    assert gen.survey_cases_from_informational_dir(published, rel="runs/x")[0][
        "command"
    ] == fresh_entries[0]["command"]


def test_prose_is_rendered_from_the_persisted_manifest_not_current_constants():
    """A retained area's instructions must not be rewritten by a later code change.

    Every retained area is re-rendered nightly while its ``env.json`` is
    preserved, so recomputing the commands from the module's current tables would
    silently rewrite historical instructions and leave the prose contradicting the
    manifest beside it.
    """
    env = {
        "case": "consan-gemm",
        "target": "gfx950",
        "required_env": [
            {"var": "OLD_VAR", "set_by": "operator", "purpose": "what this run needed"}
        ],
        "rebuild": [{
            "path": "fixtures/isa/x.hsaco",
            "what": "an object built the way this run built it",
            "commands": ["hipcc --historical-flag x.hip -o x.hsaco"],
            "caveat": "",
        }],
        "observed": {},
        "artifacts_not_published": [{"path": "fixtures/isa/x.hsaco", "sha256": "ab"}],
    }
    md = gen.build_reproduce_md(env, built_refs=["fixtures/isa/x.hsaco"])
    assert "hipcc --historical-flag x.hip -o x.hsaco" in md
    assert "OLD_VAR" in md
    # Today's tables are not consulted when the manifest has its own answer.
    assert "prepare_gemm_isa.py" not in md
    assert "ROCJITSU_PREBUILT" not in md

    page = gen.build_case_index_html(
        env, [], built_refs=["fixtures/isa/x.hsaco"], up="../../"
    )
    assert "hipcc --historical-flag x.hip -o x.hsaco" in page
    assert "OLD_VAR" in page

    # An area published before the fields existed still gets instructions.
    legacy = {k: v for k, v in env.items() if k not in {"rebuild", "required_env"}}
    fallback = gen.build_reproduce_md(legacy, built_refs=["fixtures/isa/lds.hsaco"])
    assert "hipcc --genco" in fallback


def test_refresh_keeps_the_manifests_own_instructions(tmp_path, monkeypatch):
    # End-to-end version of the above: refreshing an older area must not restamp
    # its rebuild commands from the current module tables.
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    _write_history_run(dashboard / "runs", "2026-08-05-33")
    area = dashboard / "runs" / "2026-08-05-33" / "survey" / "consan-gemm"
    area.mkdir(parents=True)
    (area / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    (area / "env.json").write_text(json.dumps({
        "case": "consan-gemm", "command": "aorta sweep run", "target": "gfx950",
        "observed": {"verdict": "error"},
        "required_env": [{"var": "OLD_VAR", "set_by": "operator", "purpose": "historic"}],
        "rebuild": [{"path": "fixtures/isa/x.hsaco", "what": "historic",
                     "commands": ["hipcc --historical-flag"], "caveat": ""}],
        "artifacts_not_published": [{"path": "fixtures/isa/x.hsaco", "sha256": "ab"}],
    }))
    _write_history_run(dashboard / "runs", "2026-08-06-44")
    assert _publish_into(dashboard, monkeypatch) == 0

    md = (area / "REPRODUCE.md").read_text(encoding="utf-8")
    assert "hipcc --historical-flag" in md
    assert "OLD_VAR" in md
    env = json.loads((area / "env.json").read_text(encoding="utf-8"))
    assert env["rebuild"][0]["commands"] == ["hipcc --historical-flag"]


def test_copublish_replaces_the_destination_instead_of_merging(tmp_path, monkeypatch):
    # A file that vanished from this sweep must not survive and be listed on the
    # landing page as evidence for the new report.
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    _write_history_run(dashboard / "runs", "2026-08-05-33")
    stale = dashboard / "runs" / "2026-08-05-33" / "survey" / "consan-gemm"
    (stale / "consan").mkdir(parents=True)
    (stale / "consan" / "consan.log.gz").write_bytes(b"stale evidence")
    (stale / "sanitizer_report.json").write_text("{}")

    info = tmp_path / "informational"
    (info / "consan-gemm").mkdir(parents=True)
    (info / "consan-gemm" / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    assert _publish_into(dashboard, monkeypatch, info=info) == 0

    # This sweep produced no log, so none is published or listed.
    assert not list(stale.rglob("*.log.gz"))
    assert not (stale / "consan").exists()
    assert ".log.gz" not in (stale / "index.html").read_text(encoding="utf-8")


def test_in_window_survey_areas_are_gzipped_like_guardrail_areas(tmp_path, monkeypatch):
    # An area carried in from a disjoint history (or published before run areas
    # existed) holds raw *.log; the log window must compress it, not publish it.
    history = tmp_path / "history"
    out = tmp_path / "dashboard"
    _write_history_run(history, "2026-08-05-33")
    area = history / "2026-08-05-33" / "survey" / "consan-gemm"
    (area / "consan").mkdir(parents=True)
    (area / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    (area / "consan" / "consan.log").write_text("raw uncompressed log\n")
    monkeypatch.setattr(sys, "argv", [
        "gen_sanitizer_dashboard",
        "--history-root", str(history),
        "--baselines",
        str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
        "--out-dir", str(out),
    ])
    assert gen.main() == 0

    published = out / "runs/2026-08-05-33/survey/consan-gemm"
    assert (published / "consan" / "consan.log.gz").is_file()
    assert not (published / "consan" / "consan.log").exists()
    assert gzip.decompress(
        (published / "consan" / "consan.log.gz").read_bytes()
    ) == b"raw uncompressed log\n"


def test_survey_area_link_rejects_an_unsafe_caller_supplied_name():
    # `name` is caller JSON on a --survey entry and the run page's link is
    # survey/<name>/. HTML-escaping the attribute does not neutralize path
    # semantics, so a staged spec naming ../../other would traverse -- and would
    # disagree with the card footer, which derives its link from report_rel.
    for unsafe in (
        "../../other", "a/b", "..", "", "a b", "http://x", "a\\b",
        # A URL delimiter is as wrong as a path one: the browser reads the tail as a
        # fragment/query on "foo", so the link misses the published directory
        # entirely, and a percent-escape can be normalised back into traversal.
        "foo#bar", "foo?x=1", "%2e%2e", "foo%2Fbar",
    ):
        assert gen._safe_case_segment(unsafe) is None, unsafe
    for ok in ("consan-gemm", "waitcheck-gemm", "consan-lds-dispatch", "sol_1.hsaco"):
        assert gen._safe_case_segment(ok) == ok

    survey = gen.survey_cases_from_spec({"cases": [{
        "name": "../../other", "label": "traversal", "staged": True,
        "report_rel": "runs/x/survey/other/sanitizer_report.json",
        "report": _consan_racy_report(),
    }]})
    # An unsafe name cannot be claimed as staged at all, so no renderer links it.
    assert survey[0]["staged"] is False
    page = gen._run_index_survey_html(survey)
    assert "href=" not in page
    assert "run area" not in page
    # Even if an entry arrives already marked staged, the renderer still refuses to
    # build a link from it. The name may still appear as escaped label *text* --
    # that is display, not a path -- but never inside an href.
    forced = gen._run_index_survey_html([
        {"name": "../../other", "staged": True, "summary": {"verdict": "fail"}}
    ])
    assert "href=" not in forced
    assert "&mdash;" in forced


def test_current_run_is_the_staged_one_not_merely_the_newest(tmp_path, monkeypatch):
    """A re-run of an older workflow must not stamp the newer run's area.

    The dir id is ``<date>-<GITHUB_RUN_ID>`` and a re-run reuses its original,
    lower run id -- so it sorts *behind* a newer same-day run. Keying "the run this
    job produced" on list position then wrote this job's container image, rocjitsu
    bundle and recipe copy into the newer run's area, and published this job's
    survey under a run that did not produce it.
    """
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    older, newer = "2026-08-05-10", "2026-08-05-99"
    for run_id in (older, newer):
        _write_history_run(dashboard / "runs", run_id)
    info = tmp_path / "informational"
    (info / "consan-gemm").mkdir(parents=True)
    (info / "consan-gemm" / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    monkeypatch.setenv("AORTA_CI_IMAGE", "rocm/pytorch@sha256:rerun")
    monkeypatch.setattr(sys, "argv", [
        "gen_sanitizer_dashboard",
        "--history-root", str(dashboard / "runs"),
        "--baselines",
        str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
        "--out-dir", str(dashboard),
        "--informational-results-dir", str(info),
        # This job re-ran the OLDER workflow; the newer run is already published.
        "--current-run", older,
    ])
    assert gen.main() == 0

    def _env(run_id, case):
        path = dashboard / "runs" / run_id / case / "env.json"
        return json.loads(path.read_text(encoding="utf-8"))

    # The re-run's provenance lands on its own area...
    assert _env(older, "waitcheck").get("container_image") == "rocm/pytorch@sha256:rerun"
    assert (dashboard / "runs" / older / "waitcheck" / "recipe.yaml").is_file()
    # ...and never on the newer run that a different job produced.
    assert "container_image" not in _env(newer, "waitcheck")
    assert not (dashboard / "runs" / newer / "waitcheck" / "recipe.yaml").exists()
    # The survey is co-published under the run that swept it.
    assert (dashboard / "runs" / older / "survey" / "consan-gemm").is_dir()
    assert not (dashboard / "runs" / newer / "survey").exists()
    # data.json attaches the survey to that run too.
    data = json.loads((dashboard / "data.json").read_text(encoding="utf-8"))
    by_run = {run["meta"]["run"]: run for run in data}
    assert by_run[older].get("survey")
    assert not by_run[newer].get("survey")


def test_a_survey_spec_name_is_not_treated_as_staged_by_the_copublish_loop(
    tmp_path, monkeypatch
):
    # staged_now gated the run loop's refresh, but the co-publish loop only rewrites
    # --informational-results-dir entries. Including a spec name skipped an area
    # that nothing later rewrote, so it kept raw logs even under --keep-logs 0.
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    _write_history_run(dashboard / "runs", "2026-08-05-33")
    spec_area = dashboard / "runs" / "2026-08-05-33" / "survey" / "spec-case"
    (spec_area / "consan").mkdir(parents=True)
    (spec_area / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    (spec_area / "consan" / "consan.log").write_text("raw log\n")
    spec = tmp_path / "survey.json"
    spec.write_text(json.dumps({"cases": [{
        "name": "spec-case", "label": "spec case", "staged": True,
        "report_rel": "runs/2026-08-05-33/survey/spec-case/sanitizer_report.json",
        "report": _consan_racy_report(),
    }]}))
    info = tmp_path / "informational"
    (info / "consan-gemm").mkdir(parents=True)
    (info / "consan-gemm" / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    monkeypatch.setattr(sys, "argv", [
        "gen_sanitizer_dashboard",
        "--history-root", str(dashboard / "runs"),
        "--baselines",
        str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
        "--out-dir", str(dashboard),
        "--survey", str(spec),
        "--informational-results-dir", str(info),
        "--keep-logs", "0",
    ])
    assert gen.main() == 0

    # The spec area is not staged by this job, so the run loop prunes it.
    assert not list(spec_area.rglob("*.log")), "spec area escaped --keep-logs 0"
    assert not list(spec_area.rglob("*.log.gz"))


def test_a_spec_entry_cannot_publish_a_rejected_informational_case(tmp_path, monkeypatch):
    # The copy loop resolves each case dir against the survey entries, but keying
    # that lookup on the combined list let a --survey spec entry stand in for a
    # report the informational loader had refused to read: the directory was
    # published anyway, with a summary and command describing a different report.
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    _write_history_run(dashboard / "runs", "2026-08-05-33")
    info = tmp_path / "informational"
    (info / "consan-gemm").mkdir(parents=True)
    # A malformed nested shape: it passes the isinstance guard, then makes the
    # reduction raise, so survey_cases_from_informational_dir skips the case.
    (info / "consan-gemm" / "sanitizer_report.json").write_text(json.dumps({"checks": None}))
    spec = tmp_path / "survey.json"
    spec.write_text(json.dumps({"cases": [{
        # Same name, a healthy inline report, and staged -- so before the fix this
        # entry satisfied the lookup for the rejected directory.
        "name": "consan-gemm", "label": "spec case", "staged": True,
        "report_rel": "runs/2026-08-05-33/survey/consan-gemm/sanitizer_report.json",
        "report": _consan_racy_report(),
    }]}))
    monkeypatch.setattr(sys, "argv", [
        "gen_sanitizer_dashboard",
        "--history-root", str(dashboard / "runs"),
        "--baselines",
        str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
        "--out-dir", str(dashboard),
        "--survey", str(spec),
        "--informational-results-dir", str(info),
    ])
    assert gen.main() == 0

    assert not (dashboard / "runs/2026-08-05-33/survey/consan-gemm").exists()


def test_the_page_renders_the_survey_of_the_run_it_calls_latest(tmp_path, monkeypatch):
    """Tab 1 and Tab 2 must describe the same run.

    ``build_html``/``build_summary_md`` take their guardrail data from the newest
    run, so on a re-run of an older workflow (see ``--current-run``) passing this
    job's survey to them labelled the newer run "Latest run" while showing the
    older re-run's observations under it -- and regenerating data.json dropped the
    newer run's own published survey.
    """
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    older, newer = "2026-08-05-10", "2026-08-05-99"
    for run_id in (older, newer):
        _write_history_run(dashboard / "runs", run_id)
    # The newer run published a survey area of its own on the day it ran.
    published = dashboard / "runs" / newer / "survey" / "waitcheck-gemm"
    published.mkdir(parents=True)
    (published / "sanitizer_report.json").write_text(json.dumps(_waitcheck_report()))
    info = tmp_path / "informational"
    (info / "consan-gemm").mkdir(parents=True)
    (info / "consan-gemm" / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    monkeypatch.setattr(sys, "argv", [
        "gen_sanitizer_dashboard",
        "--history-root", str(dashboard / "runs"),
        "--baselines",
        str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
        "--out-dir", str(dashboard),
        "--informational-results-dir", str(info),
        "--current-run", older,  # this job re-ran the older workflow
    ])
    assert gen.main() == 0

    page = (dashboard / "index.html").read_text(encoding="utf-8")
    # The newest run's own survey is what the page shows, rebuilt from the area
    # published under it -- links, reproduce command and all.
    assert f"runs/{newer}/survey/waitcheck-gemm/" in page
    assert "daily-waitcheck-gemm-object.yaml" in page
    # ...and this job's survey is not rendered as if it belonged to that run.
    assert "daily-consan-gemm.yaml" not in page
    assert f"runs/{older}/survey/" not in page
    summary_md = (dashboard / "summary.md").read_text(encoding="utf-8")
    assert "daily-waitcheck-gemm-object.yaml" in summary_md
    assert "daily-consan-gemm.yaml" not in summary_md
    # Each run's record keeps its own survey, so neither is dropped or reattributed.
    by_run = {
        run["meta"]["run"]: run
        for run in json.loads((dashboard / "data.json").read_text(encoding="utf-8"))
    }
    assert [e["name"] for e in by_run[older].get("survey", [])] == ["consan-gemm"]
    assert [e["name"] for e in by_run[newer].get("survey", [])] == ["waitcheck-gemm"]


def test_staged_requires_the_whole_area_to_match_the_run_being_rendered():
    """The two renderers derive one link from different sources, so pin the area.

    The run page builds ``survey/<name>/`` relative to the run it renders; the card
    footer takes ``report_rel``'s directory. Comparing only the case segment left a
    report under a *different run* passing, so the footer pointed into that other
    run while the run page pointed at its own.
    """
    def spec(name, report_rel, rel="runs/x"):
        return gen.survey_cases_from_spec(
            {"cases": [{
                "name": name, "label": name, "staged": True,
                "report_rel": report_rel, "report": _consan_racy_report(),
            }]},
            rel=rel,
        )

    # Wrong case directory, and -- the case that survived the segment-only check --
    # the right case under the wrong run.
    for name, report_rel in (
        ("foo", "runs/x/survey/bar/sanitizer_report.json"),
        ("foo", "runs/other/survey/foo/sanitizer_report.json"),
        ("foo", "runs/x/foo/sanitizer_report.json"),
    ):
        entries = spec(name, report_rel)
        assert entries[0]["staged"] is False, report_rel
        assert "run area</a>" not in gen.build_html(
            [_healthy_guardrail_run()], survey=entries
        )
        assert "run area" not in gen._run_index_survey_html(entries)

    agreeing = spec("foo", "runs/x/survey/foo/sanitizer_report.json")
    assert agreeing[0]["staged"] is True
    assert 'href="survey/foo/">run area</a>' in gen._run_index_survey_html(agreeing)

    # With no run context nothing can be verified, so nothing may be claimed.
    assert spec("foo", "runs/x/survey/foo/sanitizer_report.json", rel=None)[0][
        "staged"
    ] is False


def test_pruning_is_irreversible_when_keep_logs_is_raised_later(tmp_path):
    # The logs were deleted from the published history and cannot be recovered from
    # it, so a later, larger --keep-logs must not claim they are available again.
    area = tmp_path / "dashboard" / "runs" / "r" / "survey" / "c"
    area.mkdir(parents=True)
    (area / "sanitizer_report.json").write_text("{}")
    (area / "env.json").write_text(json.dumps({
        "case": "c", "command": "aorta sweep run", "target": "gfx950",
        "observed": {}, "logs_published": False, "inputs": [],
        "artifacts_not_published": [],
    }))

    assert gen.refresh_published_case_area(
        area, tmp_path / "dashboard", logs=True
    ) is True

    env = json.loads((area / "env.json").read_text(encoding="utf-8"))
    assert env["logs_published"] is False
    page = (area / "index.html").read_text(encoding="utf-8")
    assert "pruned" in page
    assert "Logs are gzipped" not in page


def test_reproduce_md_runs_top_to_bottom(tmp_path, monkeypatch):
    # The fixtures the sweep needs are gitignored, so a document that invokes the
    # sweep before the rebuild steps fails on a fresh clone before the reader ever
    # reaches them.
    out = _publish_with_logs(tmp_path, monkeypatch, run_ids=["2026-08-05-33"])
    md = (
        out / "runs/2026-08-05-33/survey/consan-gemm/REPRODUCE.md"
    ).read_text(encoding="utf-8")

    checkout = md.index("git clone https://github.com/ROCm/aorta")
    rebuild = md.index("prepare_gemm_isa.py")
    env_note = md.index("ROCJITSU_PREBUILT")
    sweep = md.index("aorta sweep run --recipe")
    assert checkout < rebuild < sweep, "sweep must come after the rebuild steps"
    assert env_note < sweep, "required env must be set before the sweep"


def test_refresh_published_case_area_reports_an_unusable_manifest(tmp_path):
    # env.json is the only input for re-rendering an older area, so a missing or
    # malformed one must be reported (the caller then writes a fresh area) rather
    # than raising and aborting the whole dashboard.
    area = tmp_path / "dashboard" / "runs" / "r" / "survey" / "c"
    area.mkdir(parents=True)
    (area / "sanitizer_report.json").write_text("{}")

    assert gen.refresh_published_case_area(area, tmp_path / "dashboard", logs=True) is False
    (area / "env.json").write_text("[]")  # valid JSON, wrong shape
    assert gen.refresh_published_case_area(area, tmp_path / "dashboard", logs=True) is False


def test_older_run_areas_keep_their_own_provenance_not_the_current_job(tmp_path, monkeypatch):
    """An older area must not be relabelled with the publishing job's environment.

    ``AORTA_CI_IMAGE`` / ``AORTA_ROCJITSU_*`` describe the GPU run being published
    now. Every retained run's area is re-rendered on every nightly, so stamping
    them unconditionally would tell a reader that a three-week-old run used
    today's container image and sanitizer build.
    """
    dashboard = tmp_path / "dashboard"
    dashboard.mkdir()
    for run_id, image, bundle in (
        ("2026-08-04-22", "rocm/pytorch@sha256:old", "oldbundle"),
        ("2026-08-05-33", "rocm/pytorch@sha256:new", "newbundle"),
    ):
        _write_history_run(dashboard / "runs", run_id)
        monkeypatch.setenv("AORTA_CI_IMAGE", image)
        monkeypatch.setenv("AORTA_ROCJITSU_COMMIT", bundle)
        monkeypatch.setattr(sys, "argv", [
            "gen_sanitizer_dashboard",
            "--history-root", str(dashboard / "runs"),
            "--baselines",
            str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
            "--out-dir", str(dashboard),
        ])
        assert gen.main() == 0

    def _env(run_id):
        path = dashboard / "runs" / run_id / "waitcheck" / "env.json"
        return json.loads(path.read_text(encoding="utf-8"))

    assert _env("2026-08-04-22")["container_image"] == "rocm/pytorch@sha256:old"
    assert _env("2026-08-04-22")["rocjitsu_commit"] == "oldbundle"
    assert _env("2026-08-05-33")["container_image"] == "rocm/pytorch@sha256:new"
    # It also survives into the rendering, not just the manifest.
    assert "rocm/pytorch@sha256:old" in (
        dashboard / "runs/2026-08-04-22/waitcheck/REPRODUCE.md"
    ).read_text(encoding="utf-8")


def test_past_run_pages_still_list_the_survey_areas_published_under_them(
    tmp_path, monkeypatch
):
    # The live survey list only covers the run being published now, so a past
    # run's page has to be rebuilt from the areas on disk -- otherwise the areas
    # kept under it are reachable only by typing the URL, on a site with no
    # directory listing. Uses the nightly's layout (history nested under the
    # output tree); with disjoint trees runs/ is cleared and re-copied instead.
    out = tmp_path / "dashboard"
    out.mkdir()
    for run_id, with_survey in (("2026-08-05-33", True), ("2026-08-06-44", False)):
        _write_history_run(out / "runs", run_id)
        argv = [
            "gen_sanitizer_dashboard",
            "--history-root", str(out / "runs"),
            "--baselines",
            str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
            "--out-dir", str(out),
        ]
        if with_survey:
            info = tmp_path / "informational"
            (info / "consan-gemm").mkdir(parents=True)
            (info / "consan-gemm" / "sanitizer_report.json").write_text(
                json.dumps(_informational_report("combined_hook_timeout"))
            )
            argv += ["--informational-results-dir", str(info)]
        monkeypatch.setattr(sys, "argv", argv)
        assert gen.main() == 0

    older = (out / "runs/2026-08-05-33/index.html").read_text(encoding="utf-8")
    assert "Workload survey" in older
    assert 'href="survey/consan-gemm/">run area</a>' in older
    assert "None" not in older
    # The newest run has no survey cases of its own, so no empty section.
    assert "Workload survey" not in (
        out / "runs/2026-08-06-44/index.html"
    ).read_text(encoding="utf-8")


def test_older_survey_areas_published_before_run_areas_are_retrofitted(
    tmp_path, monkeypatch
):
    # History co-published before #384 holds survey dirs with only a report. They
    # get the same retrofit as an older guardrail case, so retained history is
    # uniformly browsable instead of holding areas with no landing page.
    out = tmp_path / "dashboard"
    out.mkdir()
    _write_history_run(out / "runs", "2026-08-05-33")
    bare = out / "runs" / "2026-08-05-33" / "survey" / "consan-gemm"
    bare.mkdir(parents=True)
    (bare / "sanitizer_report.json").write_text(
        json.dumps(_informational_report("combined_hook_timeout"))
    )
    _write_history_run(out / "runs", "2026-08-06-44")
    monkeypatch.setattr(sys, "argv", [
        "gen_sanitizer_dashboard",
        "--history-root", str(out / "runs"),
        "--baselines",
        str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
        "--out-dir", str(out),
    ])
    assert gen.main() == 0

    assert (bare / "index.html").is_file()
    env = json.loads((bare / "env.json").read_text(encoding="utf-8"))
    assert env["class"] == "survey"
    assert env["recipe"] == "recipes/sanitizers/daily-consan-gemm.yaml"
    # Retrofitted from the report alone, so it claims no live provenance and no
    # recipe copy -- neither is knowable for a run this job did not execute.
    assert "container_image" not in env
    assert not (bare / "recipe.yaml").exists()
    # And it is now reachable from the run page rather than only by direct URL.
    page = (out / "runs/2026-08-05-33/index.html").read_text(encoding="utf-8")
    assert 'href="survey/consan-gemm/">run area</a>' in page


def test_survey_entries_from_published_degrades_on_a_missing_manifest(tmp_path):
    run_dir = tmp_path / "runs" / "r"
    (run_dir / "survey" / "good").mkdir(parents=True)
    (run_dir / "survey" / "bare").mkdir(parents=True)
    (run_dir / "survey" / "good" / "sanitizer_report.json").write_text("{}")
    (run_dir / "survey" / "good" / "env.json").write_text(
        json.dumps({"case": "good", "command": "aorta sweep run", "observed": {}})
    )

    entries = gen.survey_entries_from_published(run_dir)

    # An area with no manifest is skipped rather than rendered from guesses.
    assert [e["name"] for e in entries] == ["good"]
    # A recorded verdict may be absent; the table must not print "None" for it.
    assert entries[0]["summary"]["verdict"] is None
    page = gen._run_index_survey_html(entries)
    assert "None" not in page and gen._DASH in page
    assert gen.survey_entries_from_published(tmp_path / "runs" / "absent") == []


def test_build_case_env_takes_provenance_rather_than_reading_the_environment(monkeypatch):
    # Keeping it a parameter is what makes it impossible to relabel an older run
    # by accident; the function stays pure and testable.
    monkeypatch.setenv("AORTA_CI_IMAGE", "rocm/pytorch@sha256:leak")
    kwargs = {
        "case": "waitcheck", "cls": "guardrail", "recipe": "r.yaml",
        "command": "aorta sweep run", "meta": {}, "summary": {},
        "report": None, "built_refs": [], "inputs": [],
    }
    assert "container_image" not in gen.build_case_env(**kwargs)
    assert gen.provenance_from_environ()["container_image"] == "rocm/pytorch@sha256:leak"
    stamped = gen.build_case_env(**kwargs, provenance=gen.provenance_from_environ())
    assert stamped["container_image"] == "rocm/pytorch@sha256:leak"


def test_run_area_renders_em_dash_for_a_report_missing_its_verdict():
    # summarize_case keeps a structurally-degraded report as present, so verdict /
    # execution can be None. Rendering those as the literal "None" is the bug this
    # guards; env.json still records null, which is honest for a machine consumer.
    env = gen.build_case_env(
        case="consan-gemm", cls="survey",
        recipe="recipes/sanitizers/daily-consan-gemm.yaml", command="aorta sweep run",
        meta={"run": "r", "commit_full": "abc", "date": "2026-08-05", "gpu": "gfx950"},
        summary=gen.summarize_case({"schema": "aorta.sanitizer_report/0.1"}, None),
        report=None, built_refs=[], inputs=[],
    )
    assert env["observed"]["verdict"] is None
    assert env["observed"]["execution"] is None

    md = gen.build_reproduce_md(env, built_refs=[])
    assert f"- Verdict: `{gen._DASH}`" in md
    assert f"- Execution: `{gen._DASH}`" in md
    assert "None" not in md

    page = gen.build_case_index_html(env, [], built_refs=[], up="../../")
    assert ">None<" not in page
    assert gen._DASH in page
