#!/usr/bin/env python3
"""Render a sanitizer-nightly dashboard from ``aorta.sanitizer_report/0.1`` output.

Consumes the three daily recipe reports and the committed verdict baselines and
emits, into ``--out-dir``:

* ``index.html`` -- a self-contained (no CDN) status page: gate banner, latest
  per-recipe table with verdict badges, **per-kernel detail** (selected worklist
  kernels with their code object / SHA-256 / dispatch count / per-kernel verdict
  and finding count), a findings-by-(code, severity) summary, and a cross-run
  history/trend table.
* ``summary.md`` -- a GitHub Actions job-summary fragment (append to
  ``$GITHUB_STEP_SUMMARY``) carrying the same gate, table, and kernel detail.
* ``data.json`` -- the aggregated structure for any richer consumer.

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

_VERDICT_COLOR = {
    "pass": "#1a7f37",
    "warn": "#9a6700",
    "fail": "#cf222e",
    "error": "#6e7781",
    "not_checked": "#6e7781",
    "—": "#6e7781",
}


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
    # red, mirroring compare_verdict_baselines.py which errors on absent reports.
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


def _badge_html(verdict: str, expected: str | None) -> str:
    color = _VERDICT_COLOR.get(verdict, "#6e7781")
    mark = ""
    if expected is not None:
        mark = (
            '<span class="mark ok">&#10004;</span>' if verdict == expected
            else f'<span class="mark bad">&#10008; want {_esc(expected)}</span>'
        )
    return f'<span class="badge" style="background:{color}">{_esc(str(verdict))}</span>{mark}'


def _kernel_detail_html(rows: dict[str, dict[str, Any]]) -> str:
    blocks: list[str] = []
    for case, _key, label, _backend_label in CASES:
        row = rows[case]
        if not row["present"]:
            blocks.append(f"<h3>{_esc(label)}</h3><p>report missing</p>")
            continue
        wl = row["worklist"]
        backend = row["backend"]
        backend_txt = (
            f'{_esc(backend["name"])} <span class=mono>{_esc(backend["sha"])}</span>'
            if backend else "&mdash;"
        )
        krows = "".join(
            f"<tr><td class=mono>{_esc(str(k['name']))}</td>"
            f"<td class=num>{_esc(str(k['dispatch']))}</td>"
            f"<td>{_badge_html(k['verdict'], None)}</td>"
            f"<td class=num>{_esc(str(k['findings']))}</td>"
            f"<td class=mono>{_esc(k['code_object']) or '&mdash;'}</td>"
            f"<td class=mono>{_esc(k['sha']) or '&mdash;'}</td></tr>"
            for k in row["kernels"]
        ) or '<tr><td colspan=6>no kernels selected</td></tr>'
        frows = "".join(
            f"<tr><td>{_esc(str(g['sanitizer']))}</td><td class=mono>{_esc(str(g['code']))}</td>"
            f"<td>{_esc(str(g['severity']))}</td><td class=num>{g['count']}</td>"
            f"<td class=mono>{_esc(g['example'])}</td></tr>"
            for g in row["finding_groups"]
        ) or '<tr><td colspan=5>no findings</td></tr>'
        blocks.append(
            f"<h3>{_esc(label)} &middot; {_badge_html(row['verdict'], row['expected'])}</h3>"
            f"<div class=meta>backend {backend_txt} &middot; selection "
            f"{_esc(str(wl['requirement']))} top&#8209;{_esc(str(wl['top_n']))} &middot; "
            f"{_esc(str(wl['kernel_count']))} kernel(s) &middot; execution "
            f"{_esc(str(row['execution']))}</div>"
            "<table><tr><th>Kernel</th><th>Dispatch</th><th>Verdict</th><th>Findings</th>"
            f"<th>Code object</th><th>SHA-256</th></tr>{krows}</table>"
            "<table><tr><th>Sanitizer</th><th>Code</th><th>Severity</th><th>Count</th>"
            f"<th>Example</th></tr>{frows}</table>"
        )
    return "".join(blocks)


def build_html(runs: list[dict[str, Any]], *, title: str = "Sanitizers Nightly") -> str:
    if not runs:
        return f"<!doctype html><meta charset=utf-8><title>{_esc(title)}</title><p>No runs.</p>"
    latest = runs[0]
    meta = latest["meta"]
    gate_ok = latest["gate"]
    gate_color = "#1a7f37" if gate_ok else "#cf222e"
    gate_text = (
        "PASS \u2014 all verdicts match baselines" if gate_ok
        else "FAIL \u2014 verdict mismatch vs baselines"
    )

    latest_rows = "".join(
        f"<tr><td>{_esc(label)}</td><td>{_esc(backend)}</td>"
        f"<td>{_badge_html(latest['rows'][case]['verdict'], latest['rows'][case]['expected'])}</td>"
        f"<td>{_esc(str(latest['rows'][case]['execution']))}</td>"
        f"<td class=num>{latest['rows'][case]['findings']}</td>"
        f"<td>{_esc(latest['rows'][case]['coverage']) or '&mdash;'}</td></tr>"
        for case, _key, label, backend in CASES
    )

    hist_head = "".join(f"<th>{_esc(label)}</th>" for _c, _k, label, _b in CASES)
    hist_rows = "".join(
        f"<tr><td class=mono>{_esc(run['meta'].get('run', ''))}</td>"
        f"<td class=mono>{_esc(run['meta'].get('commit', ''))}</td>"
        f"<td>{_esc(run['meta'].get('date', ''))}</td>"
        + "".join(f"<td>{_badge_html(run['rows'][c]['verdict'], None)}</td>" for c, _k, _l, _b in CASES)
        + ('<td><span class="pill ok">green</span></td>' if run["gate"]
           else '<td><span class="pill bad">red</span></td>')
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
  .mono {{ font-family: ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
  .gate {{ color:#fff; background:{gate_color}; padding:12px 16px; border-radius:8px;
          font-weight:600; margin:12px 0 20px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px;
          overflow:hidden; margin-bottom:16px; }}
  th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid #d0d7de; vertical-align:top; }}
  th {{ background:#f6f8fa; font-size:11px; text-transform:uppercase; letter-spacing:.03em; color:#57606a; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .badge {{ color:#fff; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }}
  .mark {{ margin-left:8px; font-size:12px; }} .mark.ok {{ color:#1a7f37; }} .mark.bad {{ color:#cf222e; }}
  .pill {{ padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }}
  .pill.ok {{ background:#1a7f3722; color:#1a7f37; }} .pill.bad {{ background:#cf222e22; color:#cf222e; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background:#0d1117; color:#e6edf3; }}
    table {{ background:#161b22; }} th {{ background:#161b22; color:#8b949e; }}
    th,td {{ border-color:#30363d; }}
  }}
</style></head>
<body><div class=wrap>
  <h1>{_esc(title)} &middot; gfx950</h1>
  <div class=meta>latest run <span class=mono>{_esc(meta.get('run', ''))}</span>
     &middot; commit <span class=mono>{_esc(meta.get('commit', ''))}</span>
     &middot; {_esc(meta.get('date', ''))} &middot; target {_esc(meta.get('gpu', 'gfx950'))}</div>
  <div class=gate>{_esc(gate_text)}</div>

  <h2>Latest run</h2>
  <table>
    <tr><th>Recipe</th><th>Backend</th><th>Verdict</th><th>Execution</th><th>Findings</th><th>Coverage</th></tr>
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


def build_summary_md(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "# Sanitizers Nightly\n\nNo runs found.\n"
    latest = runs[0]
    meta = latest["meta"]
    gate = (
        "\u2705 **PASS** \u2014 all verdicts match baselines" if latest["gate"]
        else "\u274c **FAIL** \u2014 verdict mismatch vs baselines"
    )
    lines = [
        "# Sanitizers Nightly \u00b7 gfx950", "",
        f"Run `{meta.get('run', '')}` \u00b7 commit `{meta.get('commit', '')}` \u00b7 "
        f"{meta.get('date', '')}",
        "", gate, "",
        "| Recipe | Backend | Verdict | Baseline | Execution | Findings | Coverage |",
        "|---|---|---|---|---|--:|---|",
    ]
    for case, _key, label, backend in CASES:
        r = latest["rows"][case]
        mark = "" if r["expected"] is None else (" \u2705" if r["match"] else f" \u274c (want {r['expected']})")
        lines.append(
            f"| {label} | {backend} | `{r['verdict']}`{mark} | `{r['expected']}` | "
            f"{r['execution']} | {r['findings']} | {r['coverage'] or _DASH} |"
        )

    lines += ["", "## Kernel details", ""]
    for case, _key, label, _backend in CASES:
        r = latest["rows"][case]
        lines.append(f"<details><summary><b>{label}</b> \u2014 <code>{r['verdict']}</code></summary>")
        lines.append("")
        if not r["present"]:
            lines += ["report missing", "", "</details>", ""]
            continue
        b = r["backend"]
        wl = r["worklist"]
        backend_name = b["name"] if b else _DASH
        lines.append(
            f"backend `{backend_name}`"
            + (f" `{b['sha']}`" if b else "")
            + f" \u00b7 selection `{wl['requirement']}` top-{wl['top_n']} "
            f"\u00b7 {wl['kernel_count']} kernel(s) \u00b7 execution `{r['execution']}`"
        )
        lines += [
            "", "| Kernel | Dispatch | Verdict | Findings | Code object | SHA-256 |",
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
            lines += ["", "| Sanitizer | Code | Severity | Count | Example |", "|---|---|---|--:|---|"]
            for g in r["finding_groups"]:
                example = g["example"].replace("|", "\\|")
                lines.append(
                    f"| {g['sanitizer']} | `{g['code']}` | {g['severity']} | {g['count']} | {example} |"
                )
        lines += ["", "</details>", ""]

    lines += ["## History / trend", "",
              "| Run | Commit | " + " | ".join(lbl for _c, _k, lbl, _b in CASES) + " | Gate |",
              "|---|---|" + "|".join(["---"] * len(CASES)) + "|---|"]
    for run in runs:
        cells = " | ".join(f"`{run['rows'][c]['verdict']}`" for c, _k, _l, _b in CASES)
        lines.append(
            f"| {run['meta'].get('run', '')} | `{run['meta'].get('commit', '')}` | {cells} | "
            f"{'green' if run['gate'] else 'red'} |"
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
    args = ap.parse_args()

    baselines = _load(args.baselines) or {}
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
    (args.out_dir / "index.html").write_text(build_html(runs), encoding="utf-8")
    (args.out_dir / "summary.md").write_text(build_summary_md(runs), encoding="utf-8")
    (args.out_dir / "data.json").write_text(
        json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate = runs[0]["gate"] if runs else False
    print(f"wrote {args.out_dir}/index.html, summary.md, data.json (gate={'green' if gate else 'red'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
