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


def test_summary_md_gate_reflects_baseline_match():
    rows = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-clean": gen.summarize_case(_consan_racy_report(), "fail"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    runs = [{"meta": {"run": "r1", "commit": "abc", "date": "d"}, "rows": rows, "gate": True}]
    md = gen.build_summary_md(runs)
    assert "PASS" in md and "Kernel details" in md
    assert "`gemm_x`" in md and "sol_1.hsaco" in md


def test_summary_md_gate_fails_on_mismatch():
    rows = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        # baseline says pass but report is fail -> mismatch
        "consan-clean": gen.summarize_case(_consan_racy_report(), "pass"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    gate = all(r["match"] for r in rows.values() if r["present"])
    runs = [{"meta": {"run": "r1", "commit": "abc", "date": "d"}, "rows": rows, "gate": gate}]
    md = gen.build_summary_md(runs)
    assert "FAIL" in md and "want pass" in md


def test_build_html_has_gate_and_kernel_names():
    rows = {
        "waitcheck": gen.summarize_case(_waitcheck_report(), "warn"),
        "consan-clean": gen.summarize_case(_consan_racy_report(), "fail"),
        "consan-racy": gen.summarize_case(_consan_racy_report(), "fail"),
    }
    runs = [{"meta": {"run": "r1", "commit": "abc", "date": "d", "gpu": "gfx950"}, "rows": rows, "gate": True}]
    html = gen.build_html(runs)
    assert "<!doctype html>" in html
    assert "PASS" in html
    assert "gemm_x" in html and "Kernel details" in html


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
    # consan-clean intentionally absent -> present False, gate must go red

    runs = gen.runs_from_results_dir(tmp_path, baselines, meta={"run": "r", "commit": "c"})
    assert len(runs) == 1
    rows = runs[0]["rows"]
    assert rows["waitcheck"]["match"] is True
    assert rows["consan-racy"]["match"] is True
    assert rows["consan-clean"]["present"] is False
    assert runs[0]["gate"] is False


def test_main_fails_closed_on_missing_baselines(tmp_path, monkeypatch):
    # A missing/unreadable baselines file must not paint a false-green gate: main
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
