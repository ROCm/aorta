"""Tests for the sanitizer dashboard preview: the screenshot helper and its workflow."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts" / "sanitizers"
_PREVIEW_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "sanitizer-dashboard-preview.yml"
_NIGHTLY_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "sanitizers-nightly.yml"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shots = _load("screenshot_dashboard")
gen = _load("gen_sanitizer_dashboard")


def _dashboard(root: Path, runs: list[str], *, pages: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text("<!doctype html><title>d</title>", encoding="utf-8")
    for name in runs:
        run = root / "runs" / name
        run.mkdir(parents=True)
        if pages:
            (run / "index.html").write_text(
                f"<!doctype html><title>{name}</title>", encoding="utf-8"
            )
    return root


# --- newest-run selection ------------------------------------------------------


def test_newest_run_compares_the_run_id_numerically(tmp_path):
    # A plain name sort puts ...-9 after ...-10.
    root = _dashboard(tmp_path / "d", ["2026-10-01T120000-9", "2026-10-01T120000-10"])
    assert shots.newest_run_page(root) == root / "runs" / "2026-10-01T120000-10" / "index.html"


def test_newest_run_orders_both_id_shapes_like_the_generator(tmp_path):
    names = [
        "2026-10-01T235959-900",
        "2026-10-02-5",  # date-only: that day's earliest
        "2026-10-02T000000-1",
    ]
    root = _dashboard(tmp_path / "d", names)
    assert shots.newest_run_page(root).parent.name == "2026-10-02T000000-1"
    assert max(names, key=gen._history_sort_key) == "2026-10-02T000000-1"


def test_newest_run_ignores_names_the_generator_does_not_enumerate(tmp_path):
    root = _dashboard(tmp_path / "d", ["2026-10-01T120000-1", "source", "2099-01-01T000000-x"])
    # A file with a well-formed name is not a run either (the generator wants is_dir()).
    (root / "runs" / "2099-12-31T235959-99").write_text("", encoding="utf-8")
    assert shots.newest_run_page(root).parent.name == "2026-10-01T120000-1"


@pytest.mark.parametrize("runs", [None, [], ["source"]], ids=["no-runs-dir", "empty", "stray"])
def test_newest_run_is_none_without_runs(tmp_path, runs):
    root = tmp_path / "d"
    root.mkdir()
    if runs is not None:
        (root / "runs").mkdir()
        for name in runs:
            (root / "runs" / name).mkdir()
    assert shots.newest_run_page(root) is None


# --- planned shots and argument handling -----------------------------------------


def test_planned_shots_names_the_dashboard_and_the_latest_run(tmp_path):
    root = _dashboard(tmp_path / "d", ["2026-10-01T120000-1", "2026-10-02T120000-2"])
    assert shots.planned_shots(root) == [
        (root / "index.html", "dashboard.png"),
        (root / "runs" / "2026-10-02T120000-2" / "index.html", "latest_run.png"),
    ]


def test_planned_shots_requires_a_rendered_index(tmp_path):
    with pytest.raises(ValueError, match="index.html does not exist"):
        shots.planned_shots(tmp_path)


def test_planned_shots_does_not_fall_back_to_an_older_run(tmp_path):
    # The generator writes a landing page for every retained run, so a missing one
    # is a render bug to surface, not a reason to photograph a different run.
    root = _dashboard(tmp_path / "d", ["2026-10-01T120000-1"])
    (root / "runs" / "2026-10-02T120000-2").mkdir()
    with pytest.raises(ValueError, match="2026-10-02T120000-2 has no index.html"):
        shots.planned_shots(root)


def test_planned_shots_on_the_generator_empty_state(tmp_path):
    # What the preview renders when sanitizer-results does not exist yet.
    history = tmp_path / "empty"
    history.mkdir()
    out = tmp_path / "dashboard"
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS / "gen_sanitizer_dashboard.py"),
            "--history-root",
            str(history),
            "--baselines",
            str(_REPO_ROOT / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"),
            "--out-dir",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    assert shots.planned_shots(out) == [(out / "index.html", "dashboard.png")]


def test_main_requires_both_directories():
    with pytest.raises(SystemExit) as exc:
        shots.main(["--dashboard-dir", "x"])
    assert exc.value.code == 2


def test_main_reports_a_missing_dashboard_before_needing_playwright(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "playwright", None)
    assert shots.main(["--dashboard-dir", str(tmp_path), "--out", str(tmp_path / "o")]) == 2
    assert "index.html does not exist" in capsys.readouterr().err


def test_main_explains_a_missing_playwright(tmp_path, capsys, monkeypatch):
    # None in sys.modules makes the import raise ImportError, installed or not.
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    root = _dashboard(tmp_path / "d", [])
    assert shots.main(["--dashboard-dir", str(root), "--out", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert "playwright is not usable" in err
    assert shots._INSTALL_HINT in err
    assert not (tmp_path / "o").exists()


def test_main_explains_a_broken_playwright_install(tmp_path, capsys, monkeypatch):
    # A half-installed package can fail its import with something other than
    # ImportError; that still gets the install hint, not a traceback.
    def broken():
        raise OSError("libnss3.so: cannot open shared object file")

    monkeypatch.setattr(shots, "_sync_playwright", broken)
    root = _dashboard(tmp_path / "d", [])
    assert shots.main(["--dashboard-dir", str(root), "--out", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert "libnss3.so" in err and shots._INSTALL_HINT in err


@pytest.mark.parametrize("workflow", [_PREVIEW_WORKFLOW, _NIGHTLY_WORKFLOW])
def test_both_workflows_install_the_playwright_the_hint_names(workflow):
    # The helper's hint, the preview workflow and the nightly render job all name
    # one Playwright pin and one browser install; a bump to one has to reach all.
    text = workflow.read_text(encoding="utf-8")
    assert 'pip install "playwright==1.64.1"' in shots._INSTALL_HINT
    assert 'pip install "playwright==1.64.1"' in text
    assert "playwright install --with-deps --only-shell chromium" in shots._INSTALL_HINT
    assert "playwright install --with-deps --only-shell chromium" in text


# --- the real capture (needs Playwright and a Chromium) ----------------------------

_TABBED_PAGE = """<!doctype html><html><head><style>
.tabradio { position:absolute; opacity:0; pointer-events:none; }
.tabpanel { display:none; }
#tab-guardrails:checked ~ #panel-guardrails { display:block; }
#tab-asan:checked ~ #panel-asan { display:block; }
</style></head><body><div class=tabs>
<input type=radio name=santab id=tab-guardrails class=tabradio checked>
<input type=radio name=santab id=tab-asan class=tabradio>
<div class=tabbar><label for="tab-guardrails">G</label><label for="tab-asan">A</label></div>
<section class=tabpanel id=panel-guardrails>guardrails</section>
<section class=tabpanel id=panel-asan>asan</section>
</div></body></html>"""


@pytest.fixture
def chromium():
    sync_api = pytest.importorskip("playwright.sync_api")
    try:
        with sync_api.sync_playwright() as pw:
            pw.chromium.launch().close()
    except Exception as error:  # any launch failure means "no browser here"
        pytest.skip(f"Playwright is installed but Chromium cannot launch: {error}")


@pytest.mark.usefixtures("chromium")
def test_capture_writes_every_page_and_tab(tmp_path):
    root = _dashboard(tmp_path / "d", ["2026-10-01T120000-1"])
    (root / "index.html").write_text(_TABBED_PAGE, encoding="utf-8")
    out = tmp_path / "shots"
    assert shots.main(["--dashboard-dir", str(root), "--out", str(out)]) == 0
    written = sorted(p.name for p in out.iterdir())
    assert written == ["dashboard-tab-asan.png", "dashboard.png", "latest_run.png"]
    for png in out.iterdir():
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # The ASan shot is of the other panel, not a second copy of the default one.
    assert (out / "dashboard.png").read_bytes() != (out / "dashboard-tab-asan.png").read_bytes()


# --- the preview workflow --------------------------------------------------------


def _workflow() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(_PREVIEW_WORKFLOW.read_text(encoding="utf-8"))


def _steps(workflow: dict) -> list[dict]:
    return [step for job in workflow["jobs"].values() for step in job.get("steps", [])]


def test_preview_workflow_is_manual_only():
    # PyYAML reads the bare key `on` as the boolean True.
    triggers = _workflow()[True]
    assert set(triggers) == {"workflow_dispatch"}, (
        "the preview must never start on its own; a PR author dispatches it on their branch"
    )


def test_preview_workflow_is_read_only():
    workflow = _workflow()
    assert workflow.get("permissions") == {"contents": "read"}
    for name, job in workflow["jobs"].items():
        assert job.get("permissions", {"contents": "read"}) == {"contents": "read"}, name


def test_preview_workflow_never_publishes():
    for step in _steps(_workflow()):
        run = step.get("run", "")
        uses = step.get("uses", "")
        assert not re.search(r"\bgit\b[^\n]*\bpush\b", run), step.get("name")
        assert "deploy-pages" not in uses and "upload-pages-artifact" not in uses, uses
        assert "deploy-pages" not in run, step.get("name")


def test_preview_workflow_calls_the_screenshot_helper_with_its_interface():
    runs = "\n".join(step.get("run", "") for step in _steps(_workflow()))
    assert re.search(r"screenshot_dashboard\.py\s*\\?\s*--dashboard-dir \S+ --out \S+", runs), (
        "the helper takes exactly --dashboard-dir DIR --out DIR"
    )


def _carry_over_provenance(runs_root: Path) -> dict[str, str]:
    """Run the render step's provenance carry-over against ``runs_root``."""
    render = next(s for s in _steps(_workflow()) if s.get("id") == "render")["run"]
    start = render.index('provenance="$(')
    end = render.index('eval "$provenance"') + len('eval "$provenance"')
    script = (
        f'set -euo pipefail\ntmp="{runs_root.parent.parent}"\n{render[start:end]}\n'
        "for v in AORTA_CI_IMAGE AORTA_ROCJITSU_COMMIT AORTA_ROCJITSU_RUN_URL; do\n"
        '  [ -n "${!v:-}" ] && printf "%s=%s\\n" "$v" "${!v}"\ndone; true\n'
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("AORTA_")}
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    proc = subprocess.run(
        ["bash", "-c", script], cwd=_REPO_ROOT, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.splitlines())


def test_preview_carries_the_newest_runs_provenance_into_the_rewrite(tmp_path):
    runs_root = tmp_path / "dashboard" / "runs"
    areas = {
        "2026-10-01T120000-10": {"container_image": "old@sha256:aa"},
        # Not a run id, so not "newest" however it sorts.
        "2026-10-03T120000-9x": {"container_image": "stray"},
        # The newest run, with a value the eval must not interpret.
        "2026-10-02T120000-9": {
            "container_image": "img'; touch pwned; '@sha256:bb",
            "rocjitsu_commit": "d03c6fe4",
        },
    }
    for run, recorded in areas.items():
        (runs_root / run / "waitcheck").mkdir(parents=True)
        (runs_root / run / "waitcheck" / "env.json").write_text(
            json.dumps(recorded), encoding="utf-8"
        )
    exported = _carry_over_provenance(runs_root)
    assert exported == {
        "AORTA_CI_IMAGE": "img'; touch pwned; '@sha256:bb",
        "AORTA_ROCJITSU_COMMIT": "d03c6fe4",
    }
    assert not (_REPO_ROOT / "pwned").exists()


def test_preview_carries_nothing_over_from_an_empty_history(tmp_path):
    runs_root = tmp_path / "dashboard" / "runs"
    runs_root.mkdir(parents=True)
    assert _carry_over_provenance(runs_root) == {}


def test_preview_workflow_passes_only_flags_the_generator_accepts():
    # Nothing else exercises this invocation in CI, so a flag the generator renames
    # or drops would otherwise surface only when someone dispatches a preview.
    render = next(s for s in _steps(_workflow()) if s.get("id") == "render")["run"]
    call = render[render.index("python scripts/sanitizers/gen_sanitizer_dashboard.py") :]
    call = call[: call.index("--out-dir")] + "--out-dir"
    flags = set(re.findall(r"(--[a-z][a-z-]*)", call))
    assert {"--history-root", "--baselines", "--status", "--out-dir"} <= flags
    source = (_SCRIPTS / "gen_sanitizer_dashboard.py").read_text(encoding="utf-8")
    for flag in flags:
        assert f'"{flag}"' in source, f"gen_sanitizer_dashboard.py does not accept {flag}"
