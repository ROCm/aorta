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
    assert '<span class="pill ok">Healthy</span>' in html
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
    assert '<span class="pill bad">Regression</span>' in html


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
    assert '<span class="pill bad">Incomplete</span>' in html
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
    assert '<span class="pill bad">Unhealthy</span>' in html
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
