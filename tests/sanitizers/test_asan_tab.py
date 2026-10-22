"""The ASan tab Pages adds to the sanitizer dashboard from committed manual runs."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts" / "sanitizers"
_COMMITTED = _REPO_ROOT / "docs" / "sanitizers" / "asan-manual-runs.json"


def _load():
    spec = importlib.util.spec_from_file_location(
        "gen_sanitizer_dashboard", _SCRIPTS / "gen_sanitizer_dashboard.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load()


def _row(**overrides) -> dict:
    row = {
        "name": "gpu_smoke",
        "description": "Minimal single-process PyTorch GPU check",
        "group": "workload",
        "command": "aorta run --workload gpu_smoke --sanitizer asan",
        "verdict": "pass",
        "kernels": {"dispatched": 3, "checked": 3},
        "findings": [],
    }
    row.update(overrides)
    return row


_OOB = {"code": "heap-buffer-overflow", "access": "READ", "size": 4, "kernel": "add100"}


def _control(**overrides) -> dict:
    fields = {
        "name": "HIP kernel, planted off-by-one",
        "group": "control",
        "expected": "fail",
        "verdict": "fail",
        "kernels": {"dispatched": 1, "checked": 1},
        "findings": [dict(_OOB)],
    }
    fields.update(overrides)
    row = _row(**fields)
    del row["command"]
    return row


def _runs(*rows: dict) -> list[dict]:
    return [
        {
            "date": "2026-10-22",
            "gpu": "MI350 (gfx950)",
            "stack": ["ROCm 7.0.2 with its ASan runtime libraries"],
            "run_mode": "aorta run --sanitizer asan",
            "rows": list(rows) or [_row()],
        }
    ]


def _write(tmp_path: Path, runs: list[dict], *, schema: str | None = None) -> Path:
    path = tmp_path / "asan.json"
    path.write_text(
        json.dumps({"schema": schema or gen.ASAN_RESULTS_SCHEMA, "runs": runs}), encoding="utf-8"
    )
    return path


def _nightly_page() -> str:
    rows = {case: gen.summarize_case(None, "pass") for case, *_ in gen.CASES}
    meta = {"run": "r1", "commit": "abc", "date": "d", "gpu": "gfx950"}
    return gen.build_html([{"meta": meta, "rows": rows, "gate": False}])


def test_committed_manual_runs_validate_and_render():
    runs = gen.load_asan_runs(_COMMITTED)
    page = gen.add_asan_tab(_nightly_page(), runs)
    assert "Manual runs, not a nightly lane." in page
    for run in runs:
        assert run["gpu"] in page
        for row in run["rows"]:
            assert gen._esc(row["name"]) in page
    # Public repo: the committed text names no internal paths.
    assert "/apps/" not in _COMMITTED.read_text(encoding="utf-8")


def test_tab_joins_the_nightly_tabs():
    page = gen.add_asan_tab(_nightly_page(), _runs(_row(), _control()))
    at = page.index
    assert at("id=tab-survey") < at("id=tab-asan") < at("<div class=tabbar>")
    tabbar = page[at("<div class=tabbar>") :]
    tabbar = tabbar[: tabbar.index("</div>")]
    assert '<label for="tab-asan">AddressSanitizer (manual run)</label>' in tabbar
    assert at("id=panel-survey") < at("<section class=tabpanel id=panel-asan>")
    # The panel has to stay a sibling of the radios, inside .tabs, for :checked to show it.
    assert page.endswith("</details>\n  </section>\n  </div>\n</div></body></html>\n")
    assert "#tab-asan:checked ~ #panel-asan { display:block; }" in page
    head = page[: at("</head>")]
    assert "#tab-asan:checked" in head
    # The nightly's own default tab is still the one selected.
    assert "id=tab-guardrails class=tabradio checked>" in page
    assert "id=tab-asan class=tabradio>" in page


def test_tab_only_adds_to_the_nightly_page():
    original = _nightly_page()
    runs = _runs(_row(), _control())
    page = gen.add_asan_tab(original, runs)
    for added in (
        f"<style>{gen._ASAN_TAB_CSS}</style>",
        "\n  <input type=radio name=santab id=tab-asan class=tabradio>",
        f'\n    <label for="tab-asan">{gen._ASAN_TAB_LABEL}</label>',
        f"<section class=tabpanel id=panel-asan>\n  {gen.asan_panel_html(runs)}\n  </section>\n  ",
    ):
        assert page.count(added) == 1
        page = page.replace(added, "", 1)
    assert page == original


def test_no_runs_page_gets_the_results_as_a_section():
    page = gen.add_asan_tab(gen.build_html([]), _runs())
    assert "<section id=panel-asan><h2 class=asan-heading>AddressSanitizer (manual run)</h2>" in page
    assert "tabpanel id=panel-asan" not in page
    assert "No sanitizer runs yet." in page
    assert page.endswith("</details></section>\n</div></body></html>\n")


@pytest.mark.parametrize(
    "mangle",
    [
        pytest.param(lambda p: gen.add_asan_tab(p, _runs()), id="already-has-the-tab"),
        pytest.param(lambda p: p.replace("</head>", ""), id="no-head"),
        pytest.param(lambda p: p.replace("</div></body></html>", "</div><footer></footer></body></html>"), id="tail-moved"),
        pytest.param(lambda p: p.replace('<label for="tab-survey">', '<label for="tab-x">'), id="label-without-radio"),
        pytest.param(lambda p: p + "<!-- -->", id="trailing-content"),
    ],
)
def test_refuses_a_page_it_cannot_place_the_tab_in(mangle):
    with pytest.raises(ValueError):
        gen.add_asan_tab(mangle(_nightly_page()), _runs())


def test_workload_and_control_rows_render_what_was_measured():
    unchecked = "17 hipBLASLt GEMM kernels (Cijk_*), not built with ASan"
    page = gen.asan_panel_html(
        _runs(
            _row(name="training", verdict="not_checked", kernels={"dispatched": 58, "checked": 41}, unchecked=unchecked),
            _control(),
            _control(name="in-bounds view", expected="pass", verdict="pass", findings=[]),
        )
    )
    assert '<span class="v neutral">not checked</span>' in page
    assert "<td class=num>41 / 58</td>" in page
    assert f"<td>{unchecked}</td>" in page
    assert page.count('<span class="pill ok">Expected outcome</span>') == 2
    assert "heap-buffer-overflow, READ of size 4, in add100" in page
    assert "no memory error reported" in page
    assert "aorta workloads <span class=count>1 workload</span>" in page
    assert "Planted-bug controls <span class=count>2 controls</span>" in page


def test_a_control_that_misses_its_bug_is_flagged():
    page = gen.asan_panel_html(_runs(_control(verdict="pass", findings=[])))
    assert '<span class="pill bad">Unexpected outcome</span>' in page
    assert "Expected outcome" not in page


def test_finding_text_names_the_access_region_kernel_and_location():
    text = gen._asan_finding_text(
        {
            "code": "heap-buffer-overflow",
            "access": "WRITE",
            "size": 4,
            "region": "0 bytes past the end of a 40-byte allocation",
            "kernel": "add100",
            "location": "hrx_probe_oob.cpp:12:24",
        }
    )
    assert text == (
        "heap-buffer-overflow, WRITE of size 4, 0 bytes past the end of a 40-byte "
        "allocation, in add100 at hrx_probe_oob.cpp:12:24"
    )
    assert gen._asan_finding_text({"code": "use-after-free"}) == "use-after-free"


def test_text_from_the_results_file_is_escaped():
    page = gen.asan_panel_html(_runs(_row(name="<script>x</script>", description="a & b")))
    assert "<script>x</script>" not in page
    assert "&lt;script&gt;x&lt;/script&gt;" in page and "a &amp; b" in page


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(_row(kernels={"dispatched": 3, "checked": 2}, unchecked="one"), id="pass-with-an-unchecked-kernel"),
        pytest.param(_row(kernels={"dispatched": 0, "checked": 0}), id="pass-with-no-kernel"),
        pytest.param(_row(kernels={"dispatched": 3, "checked": 4}), id="more-checked-than-dispatched"),
        pytest.param(_row(kernels={"dispatched": True, "checked": True}), id="bool-counts"),
        pytest.param(_row(kernels={"dispatched": 3, "checked": -1}), id="negative-count"),
        pytest.param(_row(verdict="not_checked", kernels={"dispatched": 3, "checked": 2}), id="unchecked-without-a-reason"),
        pytest.param(_row(unchecked="none"), id="reason-when-all-checked"),
        pytest.param(_row(verdict="not_checked"), id="not-checked-with-all-checked"),
        pytest.param(_row(verdict="fail"), id="fail-without-findings"),
        pytest.param(_row(findings=[dict(_OOB)]), id="findings-without-fail"),
        pytest.param(_row(verdict="fail", findings=[{"access": "READ"}]), id="finding-without-code"),
        pytest.param(_row(verdict="clean"), id="unknown-verdict"),
        pytest.param(_row(expected="pass"), id="workload-with-expectation"),
        pytest.param(_control(expected=None), id="control-without-expectation"),
        pytest.param(_row(group="smoke"), id="unknown-group"),
        pytest.param(_row(description=""), id="no-description"),
        pytest.param(_row(command=None), id="no-command"),
    ],
)
def test_rejects_rows_that_contradict_the_run_modes_verdicts(tmp_path, row):
    with pytest.raises(ValueError):
        gen.load_asan_runs(_write(tmp_path, _runs(row)))


def _run_with(**overrides) -> list[dict]:
    runs = _runs()
    runs[0].update(overrides)
    return runs


@pytest.mark.parametrize(
    "runs,schema",
    [
        pytest.param(_runs(), "aorta.asan_manual_runs/0", id="other-schema"),
        pytest.param([], None, id="no-runs"),
        pytest.param(_run_with(date="22 Oct 2026"), None, id="date-not-iso"),
        pytest.param(_run_with(stack=[]), None, id="empty-stack"),
        pytest.param(_run_with(notes=[""]), None, id="empty-note"),
        pytest.param(_run_with(run_mode_url="javascript:alert(1)"), None, id="url-not-https"),
        pytest.param(_run_with(rows=[]), None, id="no-rows"),
        pytest.param(_run_with(gpu=" "), None, id="blank-gpu"),
    ],
)
def test_rejects_malformed_runs(tmp_path, runs, schema):
    with pytest.raises(ValueError):
        gen.load_asan_runs(_write(tmp_path, runs, schema=schema))


def test_rejects_an_unreadable_file(tmp_path):
    path = tmp_path / "asan.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        gen.load_asan_runs(path)
    with pytest.raises(ValueError):
        gen.load_asan_runs(tmp_path / "absent.json")


def test_a_valid_file_round_trips(tmp_path):
    runs = _runs(_row(), _control())
    runs[0]["notes"] = ["a note"]
    runs[0]["run_mode_url"] = "https://github.com/ROCm/aorta/pull/567"
    loaded = gen.load_asan_runs(_write(tmp_path, copy.deepcopy(runs)))
    assert loaded == runs
    page = gen.asan_panel_html(loaded)
    assert '<a href="https://github.com/ROCm/aorta/pull/567">aorta run --sanitizer asan</a>' in page
    assert '<p class="asan-lead">a note</p>' in page


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPTS / "add_asan_tab.py"), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_rewrites_the_page_in_place(tmp_path):
    page = tmp_path / "index.html"
    page.write_text(_nightly_page(), encoding="utf-8")
    result = _cli("--results", str(_COMMITTED), "--page", str(page))
    assert result.returncode == 0, result.stderr
    assert "added the ASan tab" in result.stdout
    assert "<section class=tabpanel id=panel-asan>" in page.read_text(encoding="utf-8")


def test_cli_leaves_the_page_alone_when_the_results_are_bad(tmp_path):
    page = tmp_path / "index.html"
    original = _nightly_page()
    page.write_text(original, encoding="utf-8")
    result = _cli("--results", str(_write(tmp_path, _runs(_row(verdict="fail")))), "--page", str(page))
    assert result.returncode == 1
    assert "error:" in result.stderr
    assert page.read_text(encoding="utf-8") == original


def test_pages_adds_the_tab_after_publishing_the_sanitizer_page_and_only_warns_on_failure():
    workflow = yaml.safe_load((_REPO_ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["build-deploy"]["steps"]
    names = [step.get("name", "") for step in steps]
    publish = names.index("Publish the sanitizer dashboard at /sanitizers/")
    add = names.index("Add the manual ASan results to /sanitizers/")
    verify = names.index("Verify the published routes")
    assert publish < add < verify
    script = steps[add]["run"]
    assert "scripts/sanitizers/add_asan_tab.py" in script
    assert "docs/sanitizers/asan-manual-runs.json" in script
    assert "_site/sanitizers/index.html" in script
    # The nightly page outranks the ASan tab: a failure warns rather than failing the deploy.
    assert "if ! python3" in script and "::warning::" in script
    assert "set -e" not in script.replace("set -uo pipefail", "")
