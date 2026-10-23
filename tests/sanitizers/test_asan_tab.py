"""The ASan tab Pages adds to the sanitizer dashboard from committed manual runs."""

from __future__ import annotations

import copy
import html
import importlib.util
import json
import re
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
    assert "a manual run, not a nightly lane" in page
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
    assert page.endswith("</table></div></section>\n  </section>\n  </div>\n</div></body></html>\n")
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
    assert page.endswith("</table></div></section></section>\n</div></body></html>\n")


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


_GEMMS = "17 hipBLASLt GEMM kernels (Cijk_*): generated assembly that ASan can't instrument"


def _training(**overrides) -> dict:
    fields = {
        "name": "training",
        "verdict": "not_checked",
        "kernels": {"dispatched": 58, "checked": 41},
        "unchecked": _GEMMS,
        "unchecked_outside_asan": True,
    }
    fields.update(overrides)
    return _row(**fields)


def _table(page: str) -> list[tuple[str, str, str, str, str]]:
    """The tab's rows as (what ran, result, result colour, kernels checked, note)."""
    rows = re.findall(
        r'<tr><td><details class="asan-row"><summary>(.*?)</summary>.*?</details></td>'
        r'<td><span class="asan-r (\w+)">(.*?)</span></td>'
        r"<td class=num>(.*?)</td><td>(.*?)</td></tr>",
        page,
    )
    text = lambda cell: html.unescape(re.sub(r"<[^>]+>", "", cell))  # noqa: E731
    return [(text(name), text(word), cls, kernels, text(note)) for name, cls, word, kernels, note in rows]


def test_one_table_holds_workloads_and_checks_in_file_order():
    page = gen.asan_panel_html(
        _runs(
            _row(),
            _training(),
            _control(name="in-bounds view", expected="pass", verdict="pass", findings=[]),
            _control(),
        )
    )
    assert page.count("<table>") == 1
    assert "<th>What ran</th><th>Result</th><th>Kernels checked</th><th>Note</th>" in page
    assert _table(page) == [
        ("gpu_smoke", "pass", "pass", "3 of 3", ""),
        ("training", "not checked", "warn", "41 of 58", "17 hipBLASLt GEMM kernels outside ASan's reach"),
        ("in-bounds view", "clean", "pass", "1 of 1", "nothing planted, nothing reported"),
        ("HIP kernel, planted off-by-one", "caught", "pass", "1 of 1", "heap-buffer-overflow READ"),
    ]
    for removed in ("asan-card", "asan-bar", 'class="v ', "asan-more", "What ASan covers here"):
        assert removed not in page


def _reports(page: str) -> list[list[str]]:
    """Each row's full report: the lines inside its name's <details> disclosure."""
    return [
        [html.unescape(line) for line in re.findall(r"<span>(.*?)</span>", body)]
        for body in re.findall(r'<details class="asan-row"><summary>.*?</summary>(.*?)</details>', page)
    ]


def test_the_full_report_opens_from_a_keyboard_reachable_disclosure():
    page = gen.asan_panel_html(_runs(_training(), _control()))
    # A native <details>/<summary> is focusable and opens on Enter, Space or a tap; a
    # title attribute does none of those.
    assert " title=" not in page
    assert page.count('<details class="asan-row"><summary>') == page.count("<tr><td>") == 2
    assert _reports(page) == [
        [
            "Minimal single-process PyTorch GPU check",
            "aorta run --workload gpu_smoke --sanitizer asan",
            _GEMMS,
        ],
        ["Minimal single-process PyTorch GPU check", "heap-buffer-overflow, READ of size 4, in add100"],
    ]


def test_notes_are_one_plain_line_under_the_table():
    runs = _runs()
    runs[0]["notes"] = ["A pass covers what ASan instruments.", "a <b> note"]
    page = gen.asan_panel_html(runs)
    assert page.endswith(
        '</table></div><p class="asan-intro">A pass covers what ASan instruments. a &lt;b&gt; note</p></section>'
    )
    assert gen.asan_panel_html(_runs()).endswith("</table></div></section>")


def test_an_unchecked_kernel_asan_could_have_checked_is_not_called_out_of_reach():
    (row,) = _table(gen.asan_panel_html(_runs(_training(unchecked_outside_asan=False))))
    assert row[1:] == ("not checked", "warn", "41 of 58", "17 hipBLASLt GEMM kernels not checked")


def test_a_failed_workload_is_a_red_memory_error_naming_the_access():
    (row,) = _table(gen.asan_panel_html(_runs(_row(verdict="fail", findings=[dict(_OOB)]))))
    assert row[1:] == ("memory error", "fail", "3 of 3", "heap-buffer-overflow READ")


@pytest.mark.parametrize(
    "control,word",
    [
        pytest.param(_control(verdict="pass", findings=[]), "missed", id="planted-bug-missed"),
        pytest.param(_control(expected="pass"), "false alarm", id="clean-check-reported"),
        pytest.param(
            _control(expected="error", verdict="pass", findings=[]), "expected error, got pass", id="other-mismatch"
        ),
    ],
)
def test_a_detection_check_with_the_wrong_outcome_is_red(control, word):
    (row,) = _table(gen.asan_panel_html(_runs(control)))
    assert row[1:3] == (word, "fail")


_NOT_CHECKED = {"findings": [], "kernels": {"dispatched": 1, "checked": 0}, "unchecked": "x"}


@pytest.mark.parametrize(
    "row,result",
    [
        pytest.param(_control(verdict="error", findings=[]), ("error", "fail"), id="planted-error"),
        pytest.param(_control(expected="error", verdict="error", findings=[]), ("error", "fail"), id="expected-error"),
        pytest.param(_control(verdict="not_checked", **_NOT_CHECKED), ("not checked", "warn"), id="planted-not-checked"),
        pytest.param(
            _control(expected="not_checked", verdict="not_checked", **_NOT_CHECKED),
            ("not checked", "warn"),
            id="expected-not-checked",
        ),
        pytest.param(_row(verdict="error", kernels={"dispatched": 0, "checked": 0}), ("error", "fail"), id="workload-error"),
        pytest.param(_training(), ("not checked", "warn"), id="workload-not-checked"),
    ],
)
def test_a_result_without_a_usable_verdict_is_never_green(row, result):
    gen._check_asan_row(row, "row")
    (rendered,) = _table(gen.asan_panel_html(_runs(row)))
    assert rendered[1:3] == result


def test_an_errored_row_says_there_was_no_usable_report():
    for row in (_control(verdict="error", findings=[]), _row(verdict="error", findings=[])):
        (rendered,) = _table(gen.asan_panel_html(_runs(row)))
        assert rendered[4] == "no usable ASan report"


def test_intro_says_when_where_how_and_what_is_not_checked():
    runs = _runs()
    runs[0]["not_covered"] = ["hipBLASLt GEMM kernels: assembly", "Shared memory (LDS) in kernels: off", "LDS"]
    page = gen.asan_panel_html(runs)
    intro = re.search(r'<p class="asan-intro">(.*?)</p>', page).group(1)
    assert intro == (
        "Run by hand on 2026-10-22 on MI350 (gfx950) through aorta run --sanitizer asan; a manual run, "
        "not a nightly lane, so CI neither refreshes it nor gates on it. "
        "Stack: ROCm 7.0.2 with its ASan runtime libraries. "
        "Not checked: hipBLASLt GEMM kernels; shared memory (LDS) in kernels; LDS."
    )
    assert "Not checked" not in gen.asan_panel_html(_runs())


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
        pytest.param(_row(unchecked_outside_asan=True), id="outside-reach-when-all-checked"),
        pytest.param(_training(unchecked_outside_asan="yes"), id="outside-reach-not-bool"),
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
        pytest.param(_run_with(not_covered=[""]), None, id="empty-not-covered"),
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
    runs = _runs(_row(), _training(), _control())
    runs[0]["notes"] = ["a note"]
    runs[0]["not_covered"] = ["library GEMMs"]
    runs[0]["run_mode_url"] = "https://github.com/ROCm/aorta/pull/567"
    loaded = gen.load_asan_runs(_write(tmp_path, copy.deepcopy(runs)))
    assert loaded == runs
    page = gen.asan_panel_html(loaded)
    assert '<a href="https://github.com/ROCm/aorta/pull/567">aorta run --sanitizer asan</a>' in page


def _cli(*args: str, preexec_fn=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPTS / "add_asan_tab.py"), *args],
        capture_output=True,
        text=True,
        check=False,
        preexec_fn=preexec_fn,
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


def test_cli_leaves_the_page_whole_when_writing_it_fails(tmp_path):
    resource = pytest.importorskip("resource")
    page = tmp_path / "index.html"
    original = _nightly_page().encode("utf-8")
    page.write_bytes(original)
    # A file-size limit below the augmented page fails the write partway through.
    limit = len(original) + 1024
    augmented = gen.add_asan_tab(original.decode("utf-8"), gen.load_asan_runs(_COMMITTED))
    assert len(augmented.encode("utf-8")) > limit

    def cap_file_size():
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))

    result = _cli("--results", str(_COMMITTED), "--page", str(page), preexec_fn=cap_file_size)
    assert result.returncode == 1
    assert "error:" in result.stderr
    assert page.read_bytes() == original
    assert [p.name for p in tmp_path.iterdir()] == ["index.html"]


def test_cli_keeps_the_page_mode_and_leaves_no_temporary_file(tmp_path):
    page = tmp_path / "index.html"
    page.write_text(_nightly_page(), encoding="utf-8")
    page.chmod(0o644)
    result = _cli("--results", str(_COMMITTED), "--page", str(page))
    assert result.returncode == 0, result.stderr
    assert page.stat().st_mode & 0o777 == 0o644
    assert [p.name for p in tmp_path.iterdir()] == ["index.html"]


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
