#!/usr/bin/env python3
"""File/update/close a GitHub issue based on the nightly eval results.

If the run has any FAIL verdicts, open (or update) a single tracking issue
labeled ``nightly-regression``; when the run is clean, comment on and close any
open one. Uses only the stdlib + GITHUB_TOKEN / GITHUB_REPOSITORY, so no extra
action dependency.

The decision + issue rendering are pure (``render_issue``) for unit testing;
only ``main`` touches the network.

Usage:
    python scripts/ci/alert_issue.py --results gpu-nightly-results.json [--run-url URL]
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

LABEL = "nightly-regression"
API = "https://api.github.com"


def failing_entries(results: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in results.get("entries", []) if e.get("verdict") == "fail"]


def _md_cell(text: str) -> str:
    """Sanitize a string for a Markdown table cell: escape pipes, flatten newlines."""
    return str(text).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_issue(results: dict[str, Any], run_url: str | None) -> tuple[str, str]:
    """Return (title, body) for the regression issue."""
    build = results.get("build", {}) or {}
    date = (results.get("generated_at", "") or "")[:10]
    fails = failing_entries(results)
    title = f"Nightly regression {date} ({len(fails)} failing)"
    lines = [
        f"Nightly evaluation found **{len(fails)} failing** entr"
        f"{'y' if len(fails) == 1 else 'ies'}.",
        "",
        f"- Build: aorta `{build.get('amd_aorta_version','?')}`, "
        f"torch `{build.get('torch','?')}`, ROCm `{build.get('rocm','?')}`",
        f"- Generated: {results.get('generated_at','')}",
    ]
    if run_url:
        lines.append(f"- Run: {run_url}")
    lines += ["", "| workload::cell | reasons |", "| --- | --- |"]
    for e in fails:
        reasons = "; ".join(e.get("reasons", []) or []) or (e.get("error") or "")
        key = _md_cell(f"{e.get('entry')}::{e.get('cell')}")
        lines.append(f"| `{key}` | {_md_cell(reasons)} |")
    lines += ["", "_Filed automatically by nightly-eval; will auto-close when green._"]
    return title, "\n".join(lines)


def _req(method: str, url: str, token: str, payload: dict | None = None) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode()
        return json.loads(body) if body else {}


def _find_open_issue(repo: str, token: str) -> dict | None:
    url = f"{API}/repos/{repo}/issues?state=open&labels={LABEL}&per_page=1"
    items = _req("GET", url, token)
    return items[0] if items else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--run-url", default=os.environ.get("RUN_URL"))
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not token or not repo:
        print("GITHUB_TOKEN / GITHUB_REPOSITORY not set; skipping alerting.", flush=True)
        return 0

    results = json.loads(args.results.read_text(encoding="utf-8"))
    fails = failing_entries(results)
    existing = _find_open_issue(repo, token)

    if fails:
        title, body = render_issue(results, args.run_url)
        if existing:
            _req("PATCH", f"{API}/repos/{repo}/issues/{existing['number']}", token,
                 {"title": title, "body": body})
            print(f"Updated regression issue #{existing['number']}", flush=True)
        else:
            _req("POST", f"{API}/repos/{repo}/issues", token,
                 {"title": title, "body": body, "labels": [LABEL]})
            print("Filed new regression issue", flush=True)
    elif existing:
        _req("POST", f"{API}/repos/{repo}/issues/{existing['number']}/comments", token,
             {"body": "Nightly eval is green again — closing."})
        _req("PATCH", f"{API}/repos/{repo}/issues/{existing['number']}", token, {"state": "closed"})
        print(f"Closed regression issue #{existing['number']}", flush=True)
    else:
        print("Nightly eval clean; no open regression issue.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
