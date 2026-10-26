#!/usr/bin/env python3
"""Take full-page screenshots of a rendered sanitizer dashboard.

Reviewers of a dashboard change otherwise have to download the rendered tree and
open it locally. This writes, into ``--out``:

* ``dashboard.png`` -- ``index.html`` as it first loads (the default tab);
* ``dashboard-<tab id>.png`` -- each further tab of ``index.html``, when the page
  has more than one (the ASan tab, once ``add_asan_tab.py`` has been applied);
* ``latest_run.png`` -- the newest ``runs/<id>/index.html`` landing page. Skipped,
  with a note, when the tree has no runs (the generator's empty state).

"Newest" is decided exactly as the generator decides it: only well-formed run ids
(``_RUN_ID_RE``) are considered, ordered by ``_history_sort_key``. A plain name
sort would put ``...-9`` after ``...-10`` and photograph the wrong run.

Pages are opened through ``file://`` URLs in headless Chromium. The dashboard is
self-contained (no CDN, no external JS), so nothing needs to be served. Playwright
is not a project dependency; CI installs it ad hoc, pinned, with only the
headless shell this script launches:

    pip install "playwright==1.64.1"
    python -m playwright install --with-deps --only-shell chromium

Usage (sanitizer-dashboard-preview.yml, sanitizers-nightly.yml render job):
    python scripts/sanitizers/screenshot_dashboard.py \
        --dashboard-dir dashboard --out screenshots
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

DASHBOARD_PNG = "dashboard.png"
LATEST_RUN_PNG = "latest_run.png"
_INSTALL_HINT = (
    'pip install "playwright==1.64.1" && '
    "python -m playwright install --with-deps --only-shell chromium"
)
# Wide enough that the latest-run table does not scroll horizontally.
_VIEWPORT = {"width": 1400, "height": 900}


def _generator():
    path = Path(__file__).resolve().with_name("gen_sanitizer_dashboard.py")
    spec = importlib.util.spec_from_file_location("gen_sanitizer_dashboard", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def newest_run_page(dashboard_dir: Path) -> Path | None:
    """The newest run's landing page, or None when the tree has no runs.

    Returns the path even if the page itself is missing: the generator writes one
    for every retained run, so a missing page is a render bug for the caller to
    report rather than a reason to quietly photograph an older run instead.
    """
    gen = _generator()
    runs_dir = dashboard_dir / "runs"
    if not runs_dir.is_dir():
        return None
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir() and gen._is_run_id(p.name)]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: gen._history_sort_key(p.name)) / "index.html"


def planned_shots(dashboard_dir: Path) -> list[tuple[Path, str]]:
    """``(page, png name)`` pairs to capture; raises ValueError if a page is missing."""
    index = dashboard_dir / "index.html"
    if not index.is_file():
        raise ValueError(f"{index} does not exist; is {dashboard_dir} a rendered dashboard?")
    shots = [(index, DASHBOARD_PNG)]
    latest = newest_run_page(dashboard_dir)
    if latest is not None:
        if not latest.is_file():
            raise ValueError(f"newest run {latest.parent.name} has no index.html")
        shots.append((latest, LATEST_RUN_PNG))
    return shots


def _tab_png(tab_id: str) -> str:
    return "dashboard-" + re.sub(r"[^A-Za-z0-9_-]", "_", tab_id) + ".png"


def _sync_playwright():
    """Playwright's entry point, imported only when a capture is requested."""
    from playwright.sync_api import sync_playwright

    return sync_playwright


def capture(shots: list[tuple[Path, str]], out_dir: Path, sync_playwright) -> list[Path]:
    """Render each page in headless Chromium and write full-page PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport=_VIEWPORT)
            for source, name in shots:
                page.goto(source.resolve().as_uri(), wait_until="load")
                page.screenshot(path=str(out_dir / name), full_page=True)
                written.append(out_dir / name)
                if name != DASHBOARD_PNG:
                    continue
                # The tabs are CSS-only radios (hidden inputs toggled by their
                # labels), so click the label the way a reader would.
                tab_ids = page.eval_on_selector_all(
                    "input.tabradio", "els => els.filter(e => !e.checked).map(e => e.id)"
                )
                for tab_id in tab_ids:
                    page.click(f'label[for="{tab_id}"]')
                    page.screenshot(path=str(out_dir / _tab_png(tab_id)), full_page=True)
                    written.append(out_dir / _tab_png(tab_id))
        finally:
            browser.close()
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dashboard-dir",
        type=Path,
        required=True,
        help="rendered dashboard tree (index.html, runs/<id>/index.html)",
    )
    ap.add_argument("--out", type=Path, required=True, help="directory to write the PNGs to")
    args = ap.parse_args(argv)
    try:
        shots = planned_shots(args.dashboard_dir)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    # Exception, not ImportError: a half-installed playwright can fail its import
    # with something else, and the install hint is the right answer either way.
    try:
        sync_playwright = _sync_playwright()
    except Exception as error:
        print(
            f"error: playwright is not usable ({error}). Install it with\n  {_INSTALL_HINT}",
            file=sys.stderr,
        )
        return 2
    if len(shots) == 1:
        print(f"note: {args.dashboard_dir} has no runs; skipping {LATEST_RUN_PNG}")
    for path in capture(shots, args.out, sync_playwright):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
