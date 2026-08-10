#!/usr/bin/env python3
"""Render a sanitizer-nightly dashboard from ``aorta.sanitizer_report/0.1`` output.

Consumes the three daily recipe reports and the committed verdict baselines and
emits, into ``--out-dir``:

* ``index.html`` -- a self-contained (no CDN) status page: baseline-health banner,
  latest per-recipe table, **per-kernel detail** (selected worklist kernels with
  their code object / SHA-256 / dispatch count / observed sanitizer verdict and
  finding count), a findings-by-(code, severity) summary, and a cross-run
  history/trend table.
* ``summary.md`` -- a GitHub Actions job-summary fragment (append to
  ``$GITHUB_STEP_SUMMARY``) carrying the same gate, table, and kernel detail.
* ``data.json`` -- the aggregated structure for any richer consumer.
* ``status.json`` -- a copy of the ``--status`` run manifest, when supplied, so
  Pages can distinguish a healthy snapshot from a stale one (a failed nightly).

When ``--status`` points at an unhealthy manifest, an error-colored "stale" banner is
rendered at the top of both ``index.html`` and ``summary.md`` so a failed nightly
never leaves the previous healthy page looking current.

Two input shapes are supported:

* ``--results-dir DIR`` -- a single run laid out as
  ``DIR/{waitcheck,consan-clean,consan-racy}/sanitizer_report.json`` (the same
  layout ``compare_verdict_baselines.py`` consumes). This is the CI shape.
* ``--runs-root DIR`` -- a directory of ``run_*`` folders, each with an
  ``out/<case>/sanitizer_report.json``; used locally to render a history/trend.

Pure rendering lives in ``build_html`` / ``build_summary_md`` and pure
aggregation in ``summarize_case`` so they are unit testable without the FS.

Usage (CI):
    python scripts/sanitizers/gen_sanitizer_dashboard.py \
        --results-dir incoming \
        --baselines recipes/sanitizers/fixtures/expected/verdict_baselines.json \
        --commit "$GITHUB_SHA" --run-label "run $GITHUB_RUN_ID" \
        --out-dir dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from html import escape as _esc
from pathlib import Path
from typing import Any

# case dir, baseline key, recipe label, backend label
CASES: tuple[tuple[str, str, str, str], ...] = (
    ("waitcheck", "waitcheck_gemm", "daily-waitcheck-gemm", "waitcheck (static)"),
    ("consan-clean", "consan_clean", "daily-consan-clean", "consan (dynamic)"),
    ("consan-racy", "consan_racy", "daily-consan-racy", "consan (dynamic)"),
)

_DASH = "\u2014"  # em dash; kept as a name so it can sit inside f-string braces (py3.11)


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _short(value: str | None, width: int = 10) -> str:
    return (value or "")[:width]


def _basename(path: str | None) -> str:
    return path.rsplit("/", 1)[-1] if path else ""


def _clean_msg(message: str, limit: int = 160) -> str:
    """Collapse a leading absolute path to its basename and clamp length."""
    text = (message or "").strip()
    if text.startswith("/"):
        head, sep, rest = text.partition(":")
        if sep:
            text = head.rsplit("/", 1)[-1] + sep + rest
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def summarize_case(report: dict[str, Any] | None, expected: str | None) -> dict[str, Any]:
    """Reduce one report to the fields the dashboard renders (pure)."""
    if report is None:
        return {
            "present": False, "verdict": "—", "execution": "missing", "findings": 0,
            "coverage": "", "backend": None, "expected": expected, "match": False,
            "worklist": {"requirement": None, "top_n": None, "kernel_count": 0},
            "kernels": [], "finding_groups": [],
        }
    checks = report.get("checks", [])
    findings_total = sum(len(c.get("findings", [])) for c in checks)
    coverage = ", ".join(
        str(o.get("access")) for c in checks for o in c.get("coverage", []) if o.get("access")
    )
    backend = None
    for check in checks:
        raw = check.get("backend") or {}
        if raw.get("path"):
            backend = {"name": _basename(raw.get("path")), "sha": _short(raw.get("sha256"), 12)}
            break

    kr_by_name: dict[str, dict[str, Any]] = {}
    findings_by_name: dict[str | None, int] = {}
    for check in checks:
        for result in check.get("kernel_results", []):
            name = (result.get("identity") or {}).get("name")
            kr_by_name[name] = {
                "verdict": result.get("verdict"),
                "findings": len(result.get("findings", [])),
            }
        for finding in check.get("findings", []):
            key = finding.get("kernel_name")
            findings_by_name[key] = findings_by_name.get(key, 0) + 1

    worklist = report.get("worklist", {})
    kernel_entries = worklist.get("kernels", [])
    kernels: list[dict[str, Any]] = []
    for entry in kernel_entries:
        identity = entry.get("identity", {})
        name = identity.get("name")
        result = kr_by_name.get(name)
        if result is not None:
            verdict, findings = result["verdict"], result["findings"]
        else:
            findings = findings_by_name.get(name, 0)
            # dynamic ConSan attributes race findings at process scope (kernel_name
            # is null); with a single-kernel worklist, credit them to that kernel.
            if findings == 0 and len(kernel_entries) == 1:
                findings = findings_by_name.get(None, 0)
            verdict = report.get("overall_verdict") if len(kernel_entries) == 1 or findings else "—"
        kernels.append({
            "name": name,
            "dispatch": entry.get("dispatch_count"),
            "time_ms": entry.get("total_time_ms"),
            "code_object": _basename(identity.get("code_object")),
            "sha": _short(identity.get("code_object_sha256"), 10),
            "offset": identity.get("entry_offset"),
            "verdict": verdict,
            "findings": findings,
        })

    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for check in checks:
        for finding in check.get("findings", []):
            key = (finding.get("sanitizer"), finding.get("code"), finding.get("severity"))
            bucket = groups.setdefault(key, {"count": 0, "example": ""})
            bucket["count"] += 1
            if not bucket["example"]:
                bucket["example"] = _clean_msg(finding.get("message", ""))
    finding_groups = [
        {"sanitizer": s, "code": c, "severity": sev, **data}
        for (s, c, sev), data in sorted(groups.items(), key=lambda kv: (-kv[1]["count"], kv[0]))
    ]

    verdict = report.get("overall_verdict")
    return {
        "present": True, "verdict": verdict, "execution": report.get("execution_status"),
        "findings": findings_total, "coverage": coverage, "backend": backend,
        "expected": expected, "match": (expected is None) or (verdict == expected),
        "worklist": {
            "requirement": worklist.get("requirement"),
            "top_n": worklist.get("top_n"),
            "kernel_count": worklist.get("kernel_count"),
        },
        "kernels": kernels, "finding_groups": finding_groups,
    }


def _run_record(meta: dict[str, str], rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    # Fail closed: a missing report (present=False -> match=False) turns the gate
    # unhealthy, mirroring compare_verdict_baselines.py which errors on absent reports.
    gate = all(r["match"] for r in rows.values())
    return {"meta": meta, "rows": rows, "gate": gate}


def runs_from_results_dir(
    results_dir: Path, baselines: dict, *, meta: dict[str, str]
) -> list[dict[str, Any]]:
    rows = {
        case: summarize_case(
            _load(results_dir / case / "sanitizer_report.json"),
            baselines.get(key, {}).get("overall_verdict"),
        )
        for case, key, _label, _backend in CASES
    }
    return [_run_record(meta, rows)]


def _run_meta_from_env(run_dir: Path) -> dict[str, str]:
    meta = {"run": run_dir.name, "commit": "", "date": "", "gpu": ""}
    env = run_dir / "env.txt"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            low = line.strip().lower()
            if low.startswith("commit:"):
                meta["commit"] = _short(line.split(":", 1)[1].strip().split(" ")[0], 12)
            elif low.startswith("date:"):
                meta["date"] = line.split(":", 1)[1].strip()
            elif low.startswith("gpu:"):
                meta["gpu"] = line.split(":", 1)[1].strip()
    return meta


def runs_from_runs_root(runs_root: Path, baselines: dict) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for run_dir in sorted((p for p in runs_root.glob("run_*") if p.is_dir()), reverse=True):
        rows = {
            case: summarize_case(
                _load(run_dir / "out" / case / "sanitizer_report.json"),
                baselines.get(key, {}).get("overall_verdict"),
            )
            for case, key, _label, _backend in CASES
        }
        if any(r["present"] for r in rows.values()):
            runs.append(_run_record(_run_meta_from_env(run_dir), rows))
    return runs


def _status_banner_html(status: dict[str, Any] | None) -> str:
    """Staleness banner shown when the latest nightly did not publish healthily.

    A failed nightly skips its snapshot push, so Pages would otherwise republish
    the previous healthy page as if it were current. When the publish job records an
    unhealthy run status, surface it prominently with a link to the failed run.
    """
    if not status or status.get("healthy", True):
        return ""
    run_id = str(status.get("run_id", "") or "")
    url = str(status.get("run_url", "") or "")
    conclusion = str(status.get("conclusion", "") or "unknown")
    when = str(status.get("date", "") or "")
    run_txt = f" run {_esc(run_id)}" if run_id else ""
    link = (
        f' <a href="{_esc(url)}" style="color:#fff;text-decoration:underline">view failed run</a>'
        if url else ""
    )
    when_txt = f' <span style="font-weight:400">({_esc(when)})</span>' if when else ""
    return (
        '<div style="color:#fff;background:#cf222e;padding:12px 16px;border-radius:8px;'
        'font-weight:600;margin:12px 0 20px">'
        f"&#9888; Latest sanitizer nightly{run_txt} did not complete successfully "
        f"({_esc(conclusion)}) &mdash; the data below may be stale.{link}{when_txt}"
        "</div>"
    )


def _status_banner_md(status: dict[str, Any] | None) -> str:
    if not status or status.get("healthy", True):
        return ""
    run_id = str(status.get("run_id", "") or "")
    url = str(status.get("run_url", "") or "")
    conclusion = str(status.get("conclusion", "") or "unknown")
    run_txt = f" run `{run_id}`" if run_id else ""
    link = f" [view failed run]({url})" if url else ""
    return (
        f"> \u26a0\ufe0f **Stale** \u2014 latest sanitizer nightly{run_txt} did not "
        f"complete successfully ({conclusion}); the data below may be stale.{link}"
    )


def _baseline_status(row: dict[str, Any]) -> tuple[str, str]:
    """Return the human-readable baseline status and its emphasis class."""
    if not row["present"]:
        return "Report missing", "bad"
    if row["match"]:
        return "Expected outcome", "ok"
    return "Unexpected outcome", "bad"


def _baseline_status_html(row: dict[str, Any], *, history: bool = False) -> str:
    text, emphasis = _baseline_status(row)
    if history and row["present"]:
        text = "Match" if row["match"] else "Mismatch"
    return f'<span class="pill {emphasis}">{_esc(text)}</span>'


def _observed_html(verdict: Any) -> str:
    """Render an exact sanitizer verdict without regression-health coloring."""
    return f'<span class="observed">{_esc(str(verdict))}</span>'


def _execution_html(execution: Any) -> str:
    text = str(execution)
    if text == "complete":
        return _esc(text)
    return f'<span class="execution bad">{_esc(text)}</span>'


def _execution_md(execution: Any) -> str:
    """Markdown twin of ``_execution_html``: neutral when complete, else emphasized."""
    text = str(execution)
    if text == "complete":
        return text
    return f"\u274c **{text}**"


def _gate_summary(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Classify the aggregate run health for the top banner and history gate.

    A missing report and an observed verdict mismatch both fail the gate, but
    they are operationally different: only a *present* verdict that disagrees
    with its baseline is a regression. Absent reports are surfaced separately so
    an infrastructure failure is not mislabeled as a verdict regression, and a
    run with both is reported as a combined ``UNHEALTHY`` state.
    """
    total = len(rows)
    matched = sum(1 for r in rows.values() if r["match"])
    mismatches = sum(1 for r in rows.values() if r["present"] and not r["match"])
    missing = sum(1 for r in rows.values() if not r["present"])
    if matched == total:
        label = "HEALTHY"
        detail = f"{matched}/{total} sanitizer outcomes match their baselines"
    elif mismatches and missing:
        label = "UNHEALTHY"
        detail = (
            f"investigate {mismatches}/{total} mismatched outcome(s) and "
            f"{missing}/{total} missing report(s)"
        )
    elif mismatches:
        label = "REGRESSION"
        detail = (
            f"investigate {mismatches}/{total} sanitizer outcomes that do not "
            "match their baselines"
        )
    else:
        label = "INCOMPLETE"
        detail = f"{missing}/{total} sanitizer report(s) are missing"
    return {"ok": matched == total, "label": label, "detail": detail, "short": label.capitalize()}


def _history_case_html(row: dict[str, Any]) -> str:
    expected = ""
    if row["present"] and not row["match"]:
        expected = f" &middot; expected {_observed_html(row['expected'])}"
    return (
        f"{_baseline_status_html(row, history=True)}"
        f'<div class="secondary">Observed {_observed_html(row["verdict"])}{expected}</div>'
    )


def _history_gate_html(run: dict[str, Any]) -> str:
    summary = _gate_summary(run["rows"])
    emphasis = "ok" if run["gate"] else "bad"
    return f'<td><span class="pill {emphasis}">{_esc(summary["short"])}</span></td>'


def _baseline_status_md(row: dict[str, Any], *, history: bool = False) -> str:
    text, _emphasis = _baseline_status(row)
    if history and row["present"]:
        text = "Match" if row["match"] else "Mismatch"
    icon = "\u2705" if row["present"] and row["match"] else "\u274c"
    return f"{icon} **{text}**"


def _history_case_md(row: dict[str, Any]) -> str:
    expected = ""
    if row["present"] and not row["match"]:
        expected = f"; expected `{row['expected']}`"
    return f"{_baseline_status_md(row, history=True)}<br>" f"Observed: `{row['verdict']}`{expected}"


def _kernel_detail_html(rows: dict[str, dict[str, Any]]) -> str:
    blocks: list[str] = []
    for case, _key, label, _backend_label in CASES:
        row = rows[case]
        if not row["present"]:
            blocks.append(
                f"<h3>{_esc(label)} &middot; {_baseline_status_html(row)}</h3>"
                f"<p>Observed sanitizer verdict: {_observed_html(row['verdict'])}</p>"
            )
            continue
        wl = row["worklist"]
        backend = row["backend"]
        backend_txt = (
            f'{_esc(backend["name"])} <span class=mono>{_esc(backend["sha"])}</span>'
            if backend
            else "&mdash;"
        )
        krows = (
            "".join(
                f"<tr><td class=mono>{_esc(str(k['name']))}</td>"
                f"<td class=num>{_esc(str(k['dispatch']))}</td>"
                f"<td>{_observed_html(k['verdict'])}</td>"
                f"<td class=num>{_esc(str(k['findings']))}</td>"
                f"<td class=mono>{_esc(k['code_object']) or '&mdash;'}</td>"
                f"<td class=mono>{_esc(k['sha']) or '&mdash;'}</td></tr>"
                for k in row["kernels"]
            )
            or "<tr><td colspan=6>no kernels selected</td></tr>"
        )
        frows = (
            "".join(
                f"<tr><td>{_esc(str(g['sanitizer']))}</td><td class=mono>{_esc(str(g['code']))}</td>"
                f"<td>{_esc(str(g['severity']))}</td><td class=num>{g['count']}</td>"
                f"<td class=mono>{_esc(g['example'])}</td></tr>"
                for g in row["finding_groups"]
            )
            or "<tr><td colspan=5>no findings</td></tr>"
        )
        blocks.append(
            f"<h3>{_esc(label)} &middot; {_baseline_status_html(row)}</h3>"
            f'<div class="secondary">Observed sanitizer verdict '
            f"{_observed_html(row['verdict'])} &middot; expected "
            f"{_observed_html(row['expected'] or _DASH)}</div>"
            f"<div class=meta>backend {backend_txt} &middot; selection "
            f"{_esc(str(wl['requirement']))} top&#8209;{_esc(str(wl['top_n']))} &middot; "
            f"{_esc(str(wl['kernel_count']))} kernel(s) &middot; execution "
            f"{_execution_html(row['execution'])}</div>"
            "<table><tr><th>Kernel</th><th>Dispatch</th>"
            "<th>Observed sanitizer verdict</th><th>Findings</th>"
            f"<th>Code object</th><th>SHA-256</th></tr>{krows}</table>"
            "<table><tr><th>Sanitizer</th><th>Code</th><th>Severity</th><th>Count</th>"
            f"<th>Example</th></tr>{frows}</table>"
        )
    return "".join(blocks)


def build_html(
    runs: list[dict[str, Any]],
    *,
    title: str = "Sanitizers Nightly",
    status: dict[str, Any] | None = None,
) -> str:
    banner = _status_banner_html(status)
    if not runs:
        # Rendered before the first successful nightly (empty runs-root) so the
        # /sanitizers/ route never 404s, and whenever a run fails with no data.
        return (
            "<!doctype html>\n"
            "<html lang=en><head><meta charset=utf-8>\n"
            '<meta name=viewport content="width=device-width, initial-scale=1">\n'
            f"<title>{_esc(title)}</title></head>\n"
            '<body style="font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,'
            'sans-serif;max-width:1000px;margin:0 auto;padding:24px">\n'
            '<p><a href="../">back to CI dashboard</a></p>\n'
            f"{banner}"
            f"<h1>{_esc(title)}</h1>\n"
            "<p>No sanitizer runs yet. This page will populate after the first "
            "successful sanitizer nightly.</p>\n"
            "</body></html>\n"
        )
    latest = runs[0]
    meta = latest["meta"]
    summary = _gate_summary(latest["rows"])
    gate_color = "#1a7f37" if summary["ok"] else "#cf222e"
    gate_text = f"{summary['label']} \u2014 {summary['detail']}"

    latest_rows = "".join(
        f"<tr><td>{_esc(label)}</td><td>{_esc(backend)}</td>"
        f"<td>{_baseline_status_html(latest['rows'][case])}</td>"
        f"<td>{_observed_html(latest['rows'][case]['verdict'])}</td>"
        f"<td>{_observed_html(latest['rows'][case]['expected'] or _DASH)}</td>"
        f"<td>{_execution_html(latest['rows'][case]['execution'])}</td>"
        f"<td class=num>{latest['rows'][case]['findings']}</td>"
        f"<td>{_esc(latest['rows'][case]['coverage']) or '&mdash;'}</td></tr>"
        for case, _key, label, backend in CASES
    )

    hist_head = "".join(f"<th>{_esc(label)}</th>" for _c, _k, label, _b in CASES)
    hist_rows = "".join(
        f"<tr><td class=mono>{_esc(run['meta'].get('run', ''))}</td>"
        f"<td class=mono>{_esc(run['meta'].get('commit', ''))}</td>"
        f"<td>{_esc(run['meta'].get('date', ''))}</td>"
        + "".join(f"<td>{_history_case_html(run['rows'][c])}</td>" for c, _k, _l, _b in CASES)
        + _history_gate_html(run)
        + "</tr>"
        for run in runs
    )

    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         margin:0; padding:24px; background:#f6f8fa; color:#1f2328; }}
  .wrap {{ max-width: 1000px; margin: 0 auto; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:15px; margin:20px 0 8px; }}
  h3 {{ font-size:14px; margin:18px 0 4px; }}
  .meta {{ color:#57606a; font-size:12px; margin-bottom:12px; }}
  .secondary {{ color:#57606a; font-size:12px; margin:4px 0 8px; }}
  .mono {{ font-family: ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
  .gate {{ color:#fff; background:{gate_color}; padding:12px 16px; border-radius:8px;
          font-weight:600; margin:12px 0 20px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px;
          overflow:hidden; margin-bottom:16px; }}
  th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid #d0d7de; vertical-align:top; }}
  th {{ background:#f6f8fa; font-size:11px; text-transform:uppercase; letter-spacing:.03em; color:#57606a; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .observed {{ display:inline-block; color:#57606a; background:#afb8c133; border:1px solid #afb8c1;
               padding:1px 7px; border-radius:999px; font:600 12px ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .execution.bad {{ color:#cf222e; font-weight:600; }}
  .pill {{ padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }}
  .pill.ok {{ background:#1a7f3722; color:#1a7f37; }} .pill.bad {{ background:#cf222e22; color:#cf222e; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background:#0d1117; color:#e6edf3; }}
    table {{ background:#161b22; }} th {{ background:#161b22; color:#8b949e; }}
    th,td {{ border-color:#30363d; }}
    .meta,.secondary {{ color:#8b949e; }}
    .observed {{ color:#c9d1d9; background:#6e768133; border-color:#6e7681; }}
  }}
</style></head>
<body><div class=wrap>
  <p class=nav><a href="../">back to CI dashboard</a></p>
  {banner}
  <h1>{_esc(title)} &middot; gfx950</h1>
  <div class=meta>latest run <span class=mono>{_esc(meta.get('run', ''))}</span>
     &middot; commit <span class=mono>{_esc(meta.get('commit', ''))}</span>
     &middot; {_esc(meta.get('date', ''))} &middot; target {_esc(meta.get('gpu', 'gfx950'))}</div>
  <div class=gate>{_esc(gate_text)}</div>
  <p class=secondary>Observed <span class=mono>WARN</span> or <span class=mono>FAIL</span>
     verdicts may be expected positive-control outcomes. Baseline status is the
     regression-health signal.</p>

  <h2>Latest run</h2>
  <table>
    <tr><th>Recipe</th><th>Backend</th><th>Baseline status</th><th>Observed</th>
        <th>Expected</th><th>Execution</th><th>Findings</th><th>Coverage</th></tr>
    {latest_rows}
  </table>

  <h2>Kernel details</h2>
  {_kernel_detail_html(latest['rows'])}

  <h2>History / trend</h2>
  <table>
    <tr><th>Run</th><th>Commit</th><th>Date</th>{hist_head}<th>Gate</th></tr>
    {hist_rows}
  </table>
</div></body></html>
"""


def build_summary_md(
    runs: list[dict[str, Any]], *, status: dict[str, Any] | None = None
) -> str:
    banner = _status_banner_md(status)
    if not runs:
        head = "# Sanitizers Nightly\n\n"
        if banner:
            head += banner + "\n\n"
        return head + "No runs found.\n"
    latest = runs[0]
    meta = latest["meta"]
    summary = _gate_summary(latest["rows"])
    icon = "\u2705" if summary["ok"] else "\u274c"
    gate = f"{icon} **{summary['label']}** \u2014 {summary['detail']}"
    lines = ["# Sanitizers Nightly \u00b7 gfx950", ""]
    if banner:
        lines += [banner, ""]
    lines += [
        f"Run `{meta.get('run', '')}` \u00b7 commit `{meta.get('commit', '')}` \u00b7 "
        f"{meta.get('date', '')}",
        "",
        gate,
        "",
        "Observed `WARN` or `FAIL` verdicts may be expected positive-control outcomes. "
        "Baseline status is the regression-health signal.",
        "",
        "| Recipe | Backend | Baseline status | Observed | Expected | Execution | Findings | Coverage |",
        "|---|---|---|---|---|---|--:|---|",
    ]
    for case, _key, label, backend in CASES:
        r = latest["rows"][case]
        lines.append(
            f"| {label} | {backend} | {_baseline_status_md(r)} | `{r['verdict']}` | "
            f"`{r['expected'] or _DASH}` | "
            f"{_execution_md(r['execution'])} | {r['findings']} | {r['coverage'] or _DASH} |"
        )

    lines += ["", "## Kernel details", ""]
    for case, _key, label, _backend in CASES:
        r = latest["rows"][case]
        lines.append(
            f"<details><summary><b>{label}</b> \u2014 " f"{_baseline_status_md(r)}</summary>"
        )
        lines.append("")
        if not r["present"]:
            lines += [f"Observed sanitizer verdict: `{r['verdict']}`", "", "</details>", ""]
            continue
        b = r["backend"]
        wl = r["worklist"]
        backend_name = b["name"] if b else _DASH
        lines.append(
            f"Observed sanitizer verdict `{r['verdict']}` \u00b7 expected `{r['expected'] or _DASH}`"
        )
        lines.append(
            f"backend `{backend_name}`"
            + (f" `{b['sha']}`" if b else "")
            + f" \u00b7 selection `{wl['requirement']}` top-{wl['top_n']} "
            f"\u00b7 {wl['kernel_count']} kernel(s) \u00b7 execution {_execution_md(r['execution'])}"
        )
        lines += [
            "",
            "| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |",
            "|---|--:|---|--:|---|---|",
        ]
        for k in r["kernels"]:
            code_object = k["code_object"] or _DASH
            sha = k["sha"] or _DASH
            lines.append(
                f"| `{k['name']}` | {k['dispatch']} | `{k['verdict']}` | {k['findings']} | "
                f"`{code_object}` | `{sha}` |"
            )
        if r["finding_groups"]:
            lines += [
                "",
                "| Sanitizer | Code | Severity | Count | Example |",
                "|---|---|---|--:|---|",
            ]
            for g in r["finding_groups"]:
                example = g["example"].replace("|", "\\|")
                lines.append(
                    f"| {g['sanitizer']} | `{g['code']}` | {g['severity']} | {g['count']} | {example} |"
                )
        lines += ["", "</details>", ""]

    lines += [
        "## History / trend",
        "",
        "| Run | Commit | " + " | ".join(lbl for _c, _k, lbl, _b in CASES) + " | Gate |",
        "|---|---|" + "|".join(["---"] * len(CASES)) + "|---|",
    ]
    for run in runs:
        cells = " | ".join(_history_case_md(run["rows"][c]) for c, _k, _l, _b in CASES)
        lines.append(
            f"| {run['meta'].get('run', '')} | `{run['meta'].get('commit', '')}` | {cells} | "
            f"{_gate_summary(run['rows'])['short']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--results-dir", type=Path, help="DIR/<case>/sanitizer_report.json (one run)")
    src.add_argument("--runs-root", type=Path, help="dir of run_* folders (local history)")
    ap.add_argument("--baselines", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--commit", default=os.environ.get("GITHUB_SHA", ""))
    ap.add_argument("--run-label", default=os.environ.get("GITHUB_RUN_ID", ""))
    ap.add_argument(
        "--status",
        type=Path,
        help="JSON run-status manifest; renders a stale banner when it is unhealthy",
    )
    args = ap.parse_args()

    # Optional run-status manifest (see sanitizers-nightly.yml). A malformed or
    # absent file must not crash rendering -- treat it as "no status" (no banner).
    status = _load(args.status) if args.status is not None else None
    if args.status is not None and not isinstance(status, dict):
        status = None

    # Load baselines strictly: a missing/corrupt file would otherwise leave every
    # expected verdict as None (match=True) and paint a false-healthy gate.
    baselines = _load(args.baselines)
    if not isinstance(baselines, dict) or not baselines:
        print(
            f"error: baselines file {args.baselines} is missing, empty, or unreadable",
            file=sys.stderr,
        )
        return 2
    if args.results_dir is not None:
        meta = {
            "run": args.run_label or "latest",
            "commit": _short(args.commit, 12),
            "date": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "gpu": "gfx950",
        }
        runs = runs_from_results_dir(args.results_dir, baselines, meta=meta)
    else:
        runs = runs_from_runs_root(args.runs_root, baselines)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "index.html").write_text(build_html(runs, status=status), encoding="utf-8")
    (args.out_dir / "summary.md").write_text(
        build_summary_md(runs, status=status), encoding="utf-8"
    )
    (args.out_dir / "data.json").write_text(
        json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Persist the status manifest alongside the rendered page so Pages (and any
    # consumer) can tell a healthy snapshot from a stale one without re-parsing HTML.
    if status is not None:
        (args.out_dir / "status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    gate_label = _gate_summary(runs[0]["rows"])["short"].lower() if runs else "no-data"
    print(f"wrote {args.out_dir}/index.html, summary.md, data.json (gate={gate_label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
