"""Tests for the public-safe generic-GEMM workload survey (Tab 2, #367).

Covers: the committed spec + fixtures parse and render as survey cases; both
sanitizers (waitcheck static + ConSan dynamic) are represented; a survey
``warn``/``error`` renders as a neutral observation and never a gate regression;
the committed spec is reproducible from ``gen_survey_spec.py``; and a scrub guard
asserts no customer/NDA tokens leak into the committed spec, fixtures, or the
rendered dashboard output (CLAUDE.md rule #4).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SURVEY_DIR = _REPO_ROOT / "recipes" / "sanitizers" / "survey"
_SPEC = _SURVEY_DIR / "generic_gemm_survey.json"
_REPORTS_DIR = _SURVEY_DIR / "reports"
_BASELINES = (
    _REPO_ROOT / "recipes" / "sanitizers" / "fixtures" / "expected" / "verdict_baselines.json"
)

# Data artifacts (the committed spec, the report fixtures, and the rendered
# dashboard output) must be fully scrubbed: no customer/NDA project name, org
# label, private absolute paths, internal run/user/ticket tokens, and not even
# the generic word "customer".
#
# Each entry is a regex carrying its OWN case sensitivity so a variant cannot
# slip past the guard without introducing false positives:
#   * Path/internal tokens (``recom``, ``/apps/``, ``vivekag``, ``pr347``,
#     ``feature_consan_waitcheck``) are matched case-INsensitively -- ``(?i:...)``
#     -- so ``RECOM``/``VivekAG`` are still caught; none collide with a legitimate
#     substring in the guarded artifacts.
#   * The org name "Meta" is matched case-SENSITIVELY in its real company forms
#     (``Meta``/``META``) with word boundaries. A blanket case-insensitive
#     ``meta`` would false-positive on the HTML ``<meta>`` tag and the
#     ``"metadata"`` key every ``sanitizer_report.json`` carries; matching only
#     the capitalized company forms catches an uppercase leak (Copilot's concern)
#     without those collisions.
#   * ``customer`` is a case-insensitive whole word (so ``Customer`` is caught but
#     ``customary`` is not).
_META_ORG = r"\b(?:Meta|META)\b"
_FORBIDDEN = (
    r"(?i:recom)",
    _META_ORG,
    r"(?i:/apps/)",
    r"(?i:vivekag)",
    r"(?i:pr347)",
    r"(?i:feature_consan_waitcheck)",
    r"(?i:\bcustomers?\b)",
)

# Reproduction recipes/scripts carry de-branding *prose* that legitimately uses
# the generic word "customer" (as existing public recipes in this repo already
# do, e.g. consan-code-objects.yaml). Guard those against customer NAMES,
# private paths, and internal tokens -- but not the generic word itself.
_FORBIDDEN_NAMES = (
    r"(?i:recom)",
    _META_ORG,
    r"(?i:/apps/)",
    r"(?i:vivekag)",
    r"(?i:pr347)",
    r"(?i:feature_consan_waitcheck)",
)


def _load_module(name: str):
    path = _REPO_ROOT / "scripts" / "sanitizers" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load_module("gen_sanitizer_dashboard")
gen_spec = _load_module("gen_survey_spec")


def _assert_no_forbidden(blob: str, where: str, tokens: tuple[str, ...] = _FORBIDDEN) -> None:
    # Each token carries its own case sensitivity via an inline ``(?i:...)`` flag,
    # so this deliberately does NOT pass a global re.IGNORECASE.
    for token in tokens:
        match = re.search(token, blob)
        assert match is None, (
            f"forbidden token {token!r} (matched {match.group(0)!r}) found in {where}"
        )


def _survey_entries() -> list[dict]:
    spec = json.loads(_SPEC.read_text(encoding="utf-8"))
    return gen.survey_cases_from_spec(spec, base_dir=_SPEC.parent)


def _healthy_guardrail_run() -> dict:
    report = {
        "schema": "aorta.sanitizer_report/0.1", "target": "gfx950",
        "overall_verdict": "warn", "execution_status": "complete",
        "worklist": {
            "schema": "aorta.kernel_worklist/0.1", "requirement": "top_dispatch_count",
            "top_n": 1, "kernel_count": 1,
            "kernels": [{
                "identity": {"name": "g", "target": "gfx950", "code_object": "isa/g.hsaco",
                             "code_object_sha256": "abc", "code_object_index": 0,
                             "entry_offset": None},
                "total_time_ms": 0.0, "dispatch_count": 1, "sources": ["gemm_csv"],
            }],
        },
        "checks": [{
            "sanitizer": "waitcheck", "state": "ran", "verdict": "warn",
            "reason": None, "returncode": 0, "findings": [], "kernel_results": [],
            "coverage": [], "backend": {"path": "tools/rj_waitcheck", "sha256": "472fcf288714"},
        }],
    }
    rows = {c: gen.summarize_case(report, "warn") for c, *_ in gen.CASES}
    return gen._run_record({"run": "r1", "commit": "abc", "date": "d", "gpu": "gfx950"}, rows)


# --- spec + fixtures parse and render as survey cases ---


def test_committed_spec_parses_into_six_present_survey_cases():
    entries = _survey_entries()
    assert len(entries) == 6
    assert all(e["cls"] == "survey" for e in entries)
    assert all(e["summary"]["present"] for e in entries)
    # observed-only: every survey case carries no baseline expectation
    assert all(e["summary"]["expected"] is None for e in entries)
    assert all(e["summary"]["match"] is True for e in entries)


def test_both_sanitizers_are_represented():
    backends = {e["backend"] for e in _survey_entries()}
    assert "waitcheck (static)" in backends
    assert "consan (dynamic)" in backends


def test_gemm_and_control_verdicts_match_recorded_reports():
    by_name = {e["name"]: e["summary"] for e in _survey_entries()}
    # generic hipBLASLt GEMM: waitcheck observes warn w/ many hazards; consan errors
    assert by_name["hipblaslt-gemm-f32-nt-128x128-waitcheck"]["verdict"] == "warn"
    assert by_name["hipblaslt-gemm-f32-nt-128x128-waitcheck"]["findings"] >= 1
    assert by_name["hipblaslt-gemm-f32-nt-128x128-consan"]["verdict"] == "error"
    # synthetic controls: waitcheck passes; consan fails closed (observed error)
    assert by_name["tiny-vecadd-waitcheck"]["verdict"] == "pass"
    assert by_name["tiny-vecadd-consan"]["verdict"] == "error"
    assert by_name["lds-reduce-waitcheck"]["verdict"] == "pass"
    assert by_name["lds-reduce-consan"]["verdict"] == "error"


def test_kernel_identities_are_generic():
    names = {k["name"] for e in _survey_entries() for k in e["summary"]["kernels"]}
    assert names == {"hipblaslt_gemm_f32_nt_128x128", "tiny_vecadd", "lds_reduce"}


def test_committed_spec_groups_into_three_kernels_with_both_sanitizers():
    # Review (#374): the spec carries `group`/`sanitizer` so the six cases collapse
    # into THREE kernel rows, each with a waitcheck AND a ConSan result -- not six
    # standalone rows with an em dash in every column. Guards the roll-up shape.
    entries = _survey_entries()
    groups = gen._group_survey_entries(entries)
    assert [key for key, _ in groups] == [
        "gemm-f32-nt-128x128", "tiny-vecadd", "lds-reduce"
    ]
    assert all(len(members) == 2 for _key, members in groups)
    stats = gen._survey_summary_stats(groups)
    assert stats["kernels"] == 3 and stats["runs"] == 6
    # each group exposes both sanitizer columns (no em-dash cell in the roll-up)
    for _key, members in groups:
        by_san = gen._survey_group_by_sanitizer(members)
        assert set(by_san) == {"waitcheck", "consan"}
    table = gen._survey_summary_table_html(groups)
    assert "Surveyed <b>3 kernels</b> across <b>6 sanitizer runs</b>" in table
    assert "&mdash;" not in table  # every cell has a real verdict badge


# --- survey warn/error is neutral, never a gate regression ---


def test_survey_error_and_warn_never_flip_the_gate():
    run = _healthy_guardrail_run()
    survey = _survey_entries()
    html = gen.build_html([run], survey=survey)
    md = gen.build_summary_md([run], survey=survey)
    # guardrail gate stays healthy despite survey warn/error rows
    assert "HEALTHY" in html
    assert "REGRESSION" not in html
    assert "**HEALTHY**" in md and "**REGRESSION**" not in md
    # survey rows are observed-only (explicit note on the tab)
    assert "This tab is observational only." in html
    assert "Findings represent observed behavior, not regressions." in html
    # each case drills down to its published report link
    for entry in survey:
        assert entry["report_rel"] in html


def test_build_html_renders_all_survey_kernels_with_drilldown():
    survey = _survey_entries()
    html = gen.build_html([_healthy_guardrail_run()], survey=survey)
    # Cases are grouped by kernel: each kernel is one panel holding a collapsed
    # waitcheck card and a collapsed ConSan card (not six standalone headings).
    for group_heading in (
        '<span class="kname">hipBLASLt GEMM f32 nt 128x128</span>',
        '<span class="kname">tiny_vecadd</span>',
        '<span class="kname">lds_reduce</span>',
    ):
        assert group_heading in html
    assert html.count('<span class="name">WaitCheck</span>') == 3
    assert html.count('<span class="name">ConSan</span>') == 3
    # every card ships collapsed, so its summary row must carry the verdict
    assert html.count("<details class=\"kcard") == 6 + len(gen.CASES)
    assert "<details open" not in html
    # every case still drills down to its published raw report
    for entry in survey:
        assert entry["report_rel"] in html


# --- reproducibility: the committed spec is regenerated byte-for-byte ---


def test_gen_survey_spec_reproduces_committed_spec():
    rebuilt = gen_spec.build_spec(_REPORTS_DIR, report_root=_SPEC.parent)
    rebuilt_text = json.dumps(rebuilt, indent=2) + "\n"
    assert rebuilt_text == _SPEC.read_text(encoding="utf-8")


# --- scrub guard: no customer/NDA tokens in committed or rendered artifacts ---


def test_committed_spec_and_fixtures_are_public_safe():
    _assert_no_forbidden(_SPEC.read_text(encoding="utf-8"), "generic_gemm_survey.json")
    fixtures = sorted(_REPORTS_DIR.glob("*/sanitizer_report.json"))
    assert len(fixtures) == 6
    for fixture in fixtures:
        text = fixture.read_text(encoding="utf-8")
        _assert_no_forbidden(text, str(fixture.relative_to(_REPO_ROOT)))
        # schema is preserved so the generator can parse it
        assert json.loads(text)["schema"] == "aorta.sanitizer_report/0.1"


def test_committed_recipes_are_public_safe():
    recipes = sorted(_SURVEY_DIR.glob("*.yaml"))
    assert recipes, "expected reproduction recipes under recipes/sanitizers/survey/"
    for recipe in recipes:
        _assert_no_forbidden(
            recipe.read_text(encoding="utf-8"), str(recipe.name), tokens=_FORBIDDEN_NAMES
        )


def test_rendered_dashboard_output_is_public_safe(tmp_path, monkeypatch):
    incoming = tmp_path / "incoming"
    for case, verdict, san in (
        ("waitcheck", "warn", "waitcheck"),
        ("consan-clean", "pass", "consan"),
        ("consan-racy", "fail", "consan"),
    ):
        (incoming / case).mkdir(parents=True)
        report = {
            "schema": "aorta.sanitizer_report/0.1", "target": "gfx950",
            "overall_verdict": verdict, "execution_status": "complete",
            "worklist": {
                "schema": "aorta.kernel_worklist/0.1", "requirement": "top_dispatch_count",
                "top_n": 1, "kernel_count": 1,
                "kernels": [{
                    "identity": {"name": f"g_{case}", "target": "gfx950",
                                 "code_object": "isa/g.hsaco", "code_object_sha256": "abc",
                                 "code_object_index": 0, "entry_offset": None},
                    "total_time_ms": 0.0, "dispatch_count": 1, "sources": ["gemm_csv"],
                }],
            },
            "checks": [{
                "sanitizer": san, "state": "ran", "verdict": verdict,
                "reason": None, "returncode": 0, "findings": [], "kernel_results": [],
                "coverage": [], "backend": {},
            }],
        }
        (incoming / case / "sanitizer_report.json").write_text(json.dumps(report))

    out = tmp_path / "out"
    argv = [
        "gen_sanitizer_dashboard",
        "--results-dir", str(incoming),
        "--baselines", str(_BASELINES),
        "--survey", str(_SPEC),
        "--out-dir", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert gen.main() == 0

    html = (out / "index.html").read_text(encoding="utf-8")
    md = (out / "summary.md").read_text(encoding="utf-8")
    data = (out / "data.json").read_text(encoding="utf-8")
    _assert_no_forbidden(html, "index.html")
    _assert_no_forbidden(md, "summary.md")
    _assert_no_forbidden(data, "data.json")

    # Tab 2 populated with the survey kernels and both sanitizers; gate healthy
    assert "hipblaslt_gemm_f32_nt_128x128" in html
    assert "Workload survey (observed-only)" in html
    parsed = json.loads(data)
    assert len(parsed[0]["survey"]) == 6
    assert parsed[0]["gate"] is True


def test_scrub_guard_is_case_insensitive_but_boundary_aware():
    # Casing variants of a forbidden token must be caught -- a case-sensitive
    # guard would let an uppercase project/codename/path slip through.
    for variant in (
        "RECOM_repro",
        "VivekAG",
        "/APPS/vivekag/run",
        "Meta",
        "PR347",
        "FEATURE_Consan_Waitcheck",
        "Customer",
        "CUSTOMERS",
    ):
        with pytest.raises(AssertionError):
            _assert_no_forbidden(f"leak {variant} here", "synthetic")

    # ...but a legitimate substring must NOT false-positive: every
    # ``sanitizer_report.json`` carries a ``"metadata"`` key, which must not trip
    # the boundary-anchored ``\bmeta\b`` name token.
    _assert_no_forbidden('{"metadata": {"instruction": 1}}', "synthetic")
    # The recipe denylist omits the generic word "customer" but still catches the
    # name/path/codename tokens case-insensitively.
    _assert_no_forbidden("a generic customer-facing repro", "synthetic", tokens=_FORBIDDEN_NAMES)
    with pytest.raises(AssertionError):
        _assert_no_forbidden("path /Apps/ leak", "synthetic", tokens=_FORBIDDEN_NAMES)
