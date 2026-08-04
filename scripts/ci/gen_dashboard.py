#!/usr/bin/env python3
"""Generate the nightly CI dashboard from the results history.

Reads ``results/*.json`` (each written by nightly_eval.py) and emits a
self-contained ``index.html`` (no external CDN): latest per-entry status table
plus inline-SVG performance trend sparklines and a pass-rate-over-time chart.
Also writes ``data.json`` (the aggregated history) for any richer consumer.

Pure rendering lives in ``build_dashboard_html`` so it can be unit tested.

Usage:
    python scripts/ci/gen_dashboard.py --results-dir site/results --out-dir site
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape as _esc
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import eval_lib  # for metric_policy display
    _metric_policy = eval_lib.metric_policy
except Exception:  # pragma: no cover - dashboard must render even if import fails
    def _metric_policy(_name: str):  # type: ignore
        return None

# Display units keyed by metric name (suffix-independent). Unknown -> no unit.
_METRIC_UNITS = {
    "gflops": "GFLOP/s",
    "gbps": "GB/s",
    "tokens_per_sec": "tok/s",
    "samples_per_sec": "smp/s",
    "throughput": "",
    "prefill_latency_ms": "ms",
    "decode_latency_ms": "ms",
    "latency_ms": "ms",
    "logits_checksum": "",
    "output_checksum": "",
    "checksum": "",
}

_VERDICT_COLOR = {
    "pass": "#2ea043",
    "fail": "#d1242f",
    "record": "#9a6700",
    "skip": "#57606a",
}


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load and sort all result docs by generated_at (oldest first)."""
    docs: list[dict[str, Any]] = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            docs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    docs.sort(key=lambda d: d.get("generated_at", ""))
    return docs


def _svg_sparkline(values: list[float], width: int = 160, height: int = 32) -> str:
    """Tiny inline SVG line chart for a metric's history (ignores None)."""
    pts = [v for v in values if isinstance(v, (int, float))]
    if len(pts) < 2:
        return '<span class="muted">n/a</span>'
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    n = len(values)
    coords = []
    for i, v in enumerate(values):
        if not isinstance(v, (int, float)):
            continue
        x = (i / (n - 1)) * (width - 4) + 2
        y = height - 2 - ((v - lo) / span) * (height - 4)
        coords.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(coords)
    last_x, last_y = coords[-1].split(",")
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" style="vertical-align:middle">'
        f'<polyline fill="none" stroke="#539bf5" stroke-width="1.5" '
        f'stroke-linejoin="round" stroke-linecap="round" points="{poly}"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="2" fill="#539bf5"/>'
        f"</svg>"
    )


def _latest_status(results: list[dict[str, Any]]) -> tuple[str, str]:
    if not results:
        return "unknown", "#57606a"
    s = results[-1].get("summary", {})
    if s.get("fail", 0):
        return "failing", _VERDICT_COLOR["fail"]
    if s.get("total", 0) == 0:
        return "empty", "#57606a"
    return "passing", _VERDICT_COLOR["pass"]


def build_dashboard_html(results: list[dict[str, Any]]) -> str:
    """Render the full dashboard HTML from the results history (pure)."""
    status, status_color = _latest_status(results)
    latest = results[-1] if results else {"build": {}, "summary": {}, "entries": []}
    build = latest.get("build", {}) or {}

    # The "Latest" table reflects ONLY the newest build's entries -- a cell that
    # disappeared from the current matrix must not linger as a stale pass/fail.
    latest_entries = latest.get("entries", []) or []
    latest_by_key: dict[str, dict[str, Any]] = {
        f"{e.get('entry')}::{e.get('cell')}": e for e in latest_entries
    }
    keys = list(latest_by_key.keys())

    # Step-time history (across builds) is only charted for currently-present keys.
    history: dict[str, list[float | None]] = {k: [] for k in keys}
    for doc in results:
        seen: dict[str, float | None] = {}
        for e in doc.get("entries", []) or []:
            k = f"{e.get('entry')}::{e.get('cell')}"
            seen[k] = (e.get("metrics") or {}).get("mean_step_time_ms")
        for k in keys:
            history[k].append(seen.get(k))

    passrate = []
    for doc in results:
        s = doc.get("summary", {})
        graded = (s.get("pass", 0) + s.get("fail", 0)) or 0
        passrate.append((s.get("pass", 0) / graded) if graded else None)

    # All values below originate from results/*.json (untrusted: error strings,
    # reasons, version strings) and are HTML-escaped before interpolation.
    rows = []
    for k in sorted(keys):
        e = latest_by_key.get(k, {})
        verdict = e.get("verdict", "skip")
        color = _VERDICT_COLOR.get(verdict, "#57606a")
        st = (e.get("metrics") or {}).get("mean_step_time_ms")
        st_txt = f"{st:.3f} ms" if isinstance(st, (int, float)) else "—"
        reasons = "; ".join(e.get("reasons", []) or [])
        rows.append(
            f"<tr><td class='mono'>{_esc(k)}</td>"
            f"<td class='center'><span class='badge' style='background:{color}'>{_esc(verdict)}</span></td>"
            f"<td class='num'>{_esc(st_txt)}</td>"
            f"<td class='spark'>{_svg_sparkline(history.get(k, []))}</td>"
            f"<td class='muted'>{_esc(reasons)}</td></tr>"
        )

    # Per-metric trends from metrics.summary (the newest build defines the set).
    metric_pairs: list[tuple[str, str]] = []
    for k, e in latest_by_key.items():
        for m in ((e.get("metrics") or {}).get("summary") or {}):
            metric_pairs.append((k, m))
    mhist: dict[tuple[str, str], list[float | None]] = {p: [] for p in metric_pairs}
    for doc in results:
        seen: dict[tuple[str, str], float | None] = {}
        for e in doc.get("entries", []) or []:
            kk = f"{e.get('entry')}::{e.get('cell')}"
            for mm, val in ((e.get("metrics") or {}).get("summary") or {}).items():
                seen[(kk, mm)] = val
        for p in metric_pairs:
            mhist[p].append(seen.get(p))
    metric_rows = []
    for k, m in sorted(metric_pairs):
        latest_val = ((latest_by_key[k].get("metrics") or {}).get("summary") or {}).get(m)
        unit = _METRIC_UNITS.get(m, "")
        val_txt = (
            f"{latest_val:.4g} {unit}".strip()
            if isinstance(latest_val, (int, float)) else "—"
        )
        policy = _metric_policy(m) or "(trend)"
        metric_rows.append(
            f"<tr><td class='mono'>{_esc(k)}</td>"
            f"<td class='mono'>{_esc(m)}</td>"
            f"<td class='center'>{_esc(policy)}</td>"
            f"<td class='num'>{_esc(val_txt)}</td>"
            f"<td class='spark'>{_svg_sparkline(mhist[(k, m)])}</td></tr>"
        )

    s = latest.get("summary", {})
    meta = _esc(
        f"aorta {build.get('amd_aorta_version') or '?'} · "
        f"torch {build.get('torch') or '?'} · ROCm {build.get('rocm') or '?'} · "
        f"HIP {build.get('hip') or '?'}"
    )
    generated = _esc(latest.get("generated_at", ""))
    metric_table = (
        "".join(metric_rows) if metric_rows
        else "<tr><td colspan=5 class=muted>no workload metrics captured yet</td></tr>"
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>aorta nightly CI</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#21262d; --fg:#c9d1d9; --muted:#8b949e; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          margin:0; background:var(--bg); color:var(--fg); line-height:1.45; }}
  .wrap {{ max-width:1060px; margin:0 auto; padding:2rem 1.25rem 3rem; }}
  header {{ border-bottom:1px solid var(--border); padding-bottom:1rem; margin-bottom:.5rem; }}
  h1 {{ font-size:1.5rem; margin:0 0 .5rem; }}
  h2 {{ font-size:1.05rem; margin:2rem 0 .25rem; color:#e6edf3; }}
  .status-pill {{ display:inline-block; color:#fff; font-weight:600; font-size:.78rem;
                  padding:3px 11px; border-radius:999px; background:{status_color}; vertical-align:middle; }}
  .meta {{ color:var(--muted); font-size:.82rem; margin:.55rem 0 0; }}
  .cards {{ display:flex; flex-wrap:wrap; gap:.6rem; align-items:stretch; margin:1.1rem 0 .25rem; }}
  .card {{ background:var(--panel); border:1px solid var(--border); border-radius:8px;
           padding:.5rem .85rem; min-width:80px; }}
  .card .k {{ font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }}
  .card .v {{ font-size:1.3rem; font-weight:600; line-height:1.3; }}
  .card.trend {{ flex:1; min-width:220px; }}
  /* Block-level SVG avoids inline baseline spacing without zeroing the font
     size, which would also hide the "n/a" fallback for short histories. */
  .card.trend svg {{ display:block; }}
  table {{ border-collapse:collapse; width:100%; margin-top:.5rem; background:var(--panel);
           border:1px solid var(--border); border-radius:8px; overflow:hidden; }}
  th, td {{ padding:.5rem .8rem; border-bottom:1px solid var(--border); font-size:.88rem; vertical-align:middle; }}
  thead th {{ text-align:left; color:var(--muted); font-weight:600; font-size:.7rem;
              text-transform:uppercase; letter-spacing:.05em; background:#12161b; }}
  tbody tr:last-child td {{ border-bottom:0; }}
  tbody tr:hover {{ background:#1b2129; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.center, th.center {{ text-align:center; }}
  td.spark {{ width:180px; }}
  td.spark svg {{ display:block; }}
  .badge {{ display:inline-block; min-width:60px; text-align:center; color:#fff;
            padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:600; }}
  .mono {{ font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.82rem; word-break:break-word; }}
  .muted {{ color:var(--muted); font-size:.8rem; }}
  .legend {{ color:var(--muted); font-size:.78rem; margin:.65rem 0 0; }}
</style></head>
<body>
  <div class="wrap">
    <header>
      <h1>aorta nightly CI dashboard</h1>
      <div>Latest: <span class="status-pill">{status}</span></div>
      <p class="meta">{meta}<br>generated {generated}</p>
    </header>

    <div class="cards">
      <div class="card"><div class="k">total</div><div class="v">{s.get('total', 0)}</div></div>
      <div class="card"><div class="k">pass</div><div class="v" style="color:#3fb950">{s.get('pass', 0)}</div></div>
      <div class="card"><div class="k">fail</div><div class="v" style="color:#f85149">{s.get('fail', 0)}</div></div>
      <div class="card"><div class="k">record</div><div class="v" style="color:#d29922">{s.get('record', 0)}</div></div>
      <div class="card"><div class="k">skip</div><div class="v" style="color:#8b949e">{s.get('skip', 0)}</div></div>
      <div class="card trend"><div class="k">pass-rate trend</div><div class="v">{_svg_sparkline(passrate, width=240)}</div></div>
    </div>

    <h2>Latest status</h2>
    <table>
      <colgroup><col style="width:38%"><col style="width:12%"><col style="width:13%"><col style="width:17%"><col style="width:20%"></colgroup>
      <thead><tr><th>workload::cell</th><th class="center">status</th><th class="num">step time</th>
        <th>trend (step ms)</th><th>notes</th></tr></thead>
      <tbody>
        {''.join(rows) if rows else '<tr><td colspan=5 class=muted>no results yet</td></tr>'}
      </tbody>
    </table>
    <p class="legend">Verdicts: pass/fail = vs blessed baseline · record = no baseline yet (metrics captured) · skip = insufficient GPUs.</p>

    <h2>Performance / metric trends</h2>
    <table>
      <colgroup><col style="width:34%"><col style="width:20%"><col style="width:10%"><col style="width:18%"><col style="width:18%"></colgroup>
      <thead><tr><th>workload::cell</th><th>metric</th><th class="center">policy</th>
        <th class="num">latest</th><th>trend</th></tr></thead>
      <tbody>
        {metric_table}
      </tbody>
    </table>
  </div>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-builds", type=int, default=180,
                    help="render at most the most recent N builds (bounds trend size)")
    args = ap.parse_args()

    results = load_results(args.results_dir)
    if args.max_builds > 0:
        results = results[-args.max_builds:]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "index.html").write_text(build_dashboard_html(results), encoding="utf-8")
    (args.out_dir / "data.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"dashboard: {len(results)} build(s) -> {args.out_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
