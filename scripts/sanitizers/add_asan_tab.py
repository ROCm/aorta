#!/usr/bin/env python3
"""Add the AddressSanitizer tab to a rendered sanitizer dashboard page, in place.

ASan results come from manual runs committed to the repo, not from the sanitizer
nightly (see ``add_asan_tab`` in gen_sanitizer_dashboard.py), so Pages adds them
to whichever page it published at /sanitizers/.

Usage (pages.yml):
    python3 scripts/sanitizers/add_asan_tab.py \
        --results docs/sanitizers/asan-manual-runs.json \
        --page _site/sanitizers/index.html
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _generator():
    path = Path(__file__).resolve().with_name("gen_sanitizer_dashboard.py")
    spec = importlib.util.spec_from_file_location("gen_sanitizer_dashboard", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, required=True, help="committed manual ASan runs (JSON)")
    ap.add_argument(
        "--page", type=Path, required=True, help="rendered dashboard index.html, rewritten in place"
    )
    args = ap.parse_args(argv)
    gen = _generator()
    try:
        runs = gen.load_asan_runs(args.results)
        page = gen.add_asan_tab(args.page.read_text(encoding="utf-8"), runs)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    args.page.write_text(page, encoding="utf-8")
    rows = sum(len(run["rows"]) for run in runs)
    print(f"added the ASan tab to {args.page} ({len(runs)} run(s), {rows} row(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
