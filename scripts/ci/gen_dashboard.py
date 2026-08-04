#!/usr/bin/env python3
"""Generate the nightly CI dashboard from the results history.

Reads ``results/*.json`` (each written by nightly_eval.py) and emits a
self-contained ``index.html`` (no external CDN): a run header with the
toolchain identity, headline counts, and one workload-grouped status table
whose per-cell metrics are nested in collapsible rows.
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

# Links back to the source of truth. This dashboard only ever describes this
# repo's nightly run, so the slug is fixed rather than plumbed through.
_REPO = "ROCm/aorta"

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
    "step_time_p50": "ms",
    "step_time_p99": "ms",
    "mean_wall_clock_sec": "s",
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

# Worst-first, so a group's tally leads with whatever needs attention.
_VERDICT_ORDER = ("fail", "record", "skip", "pass")


def _isnum(v: Any) -> bool:
    """Numeric for display purposes.

    ``bool`` is a subclass of ``int``, so a stray ``true`` in the (untrusted)
    results JSON would otherwise format as a step time, scale a bar, or count
    towards a trend.
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool)


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
    pts = [v for v in values if _isnum(v)]
    if len(pts) < 2:
        return '<span class="muted">n/a</span>'
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    n = len(values)
    coords = []
    for i, v in enumerate(values):
        if not _isnum(v):
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


def _has_trend(series: list[list[float | None]]) -> bool:
    """True when at least one series has enough numeric points to draw."""
    return any(len([v for v in vals if _isnum(v)]) >= 2 for vals in series)


def _latest_status(results: list[dict[str, Any]]) -> tuple[str, str]:
    """Headline verdict for the newest build.

    Only a graded pass earns ``passing``; a build with no blessed baselines is
    ``recording``, since reporting either as "passing" would claim a result
    nothing established.

    ``skipping`` is the terminal branch of that classification and not a state
    today's pipeline can reach: ``nightly_eval`` appends a synthetic ``fail``
    entry when every cell skips, so a real zero-work build carries ``fail >= 1``
    and stops at ``failing`` above. It stays because this function must classify
    whatever summary it is handed, and for an all-skip one every other label
    would be a lie.
    """
    if not results:
        return "unknown", "#57606a"
    s = results[-1].get("summary", {}) or {}
    if s.get("fail", 0):
        return "failing", _VERDICT_COLOR["fail"]
    if (s.get("total", 0) or 0) == 0:
        return "empty", "#57606a"
    if s.get("pass", 0) or 0:
        return "passing", _VERDICT_COLOR["pass"]
    if s.get("record", 0) or 0:
        return "recording", _VERDICT_COLOR["record"]
    return "skipping", _VERDICT_COLOR["skip"]


def _fmt_num(v: Any) -> str:
    """Compact number for display; keeps large integer counts readable."""
    if not _isnum(v):
        return "—"
    if float(v).is_integer() and abs(v) < 1e12:
        return f"{int(v):,}"
    return f"{v:.4g}"


def _fmt_ms(v: Any) -> str:
    return f"{v:,.1f} ms" if _isnum(v) else "—"


def _fmt_timestamp(iso: str) -> str:
    """``2026-08-03T12:15:51.105587+00:00`` -> ``2026-08-03 12:15 UTC``.

    Deliberately absolute: the page is static between nightly runs, so a
    relative age ("2 hours ago") would freeze at generation time and lie.
    """
    if not iso:
        return ""
    try:
        date, _, rest = iso.partition("T")
        hh, mm = rest.split(":")[:2]
        return f"{date} {hh}:{mm} UTC"
    except Exception:
        return iso


def _chip(label: str, value: Any) -> str:
    return (
        f'<div class="chip"><div class="k">{_esc(label)}</div>'
        f'<div class="v mono">{_esc(str(value) if value else "unknown")}</div></div>'
    )


def _bar(value: Any, group_max: float) -> str:
    """Relative bar, scaled within the workload group.

    Scaling per group rather than globally is deliberate: comparing a smoke
    test's step time against an 8-GPU training loop's is meaningless, whereas
    comparing cells of the same workload is exactly the useful signal. Callers
    pass ``group_max`` of 0 for single-cell groups, where a bar would only ever
    compare a value against itself and read as a full-width bar.
    """
    if not _isnum(value) or not group_max:
        return ""
    pct = max(2.0, min(100.0, value / group_max * 100.0))
    return f'<div class="bartrack"><div class="bar" style="width:{pct:.1f}%"></div></div>'


def _metric_rows(
    cell_key: str,
    entry: dict[str, Any],
    mhist: dict[tuple[str, str], list[float | None]],
    show_trend: bool,
) -> str:
    summary = ((entry.get("metrics") or {}).get("summary") or {})
    out = []
    for m in sorted(summary):
        raw = summary.get(m)
        unit = _METRIC_UNITS.get(m, "")
        # A unit on an unknown value ("— ms") reads as a measurement of nothing.
        val = f"{_fmt_num(raw)} {unit}".strip() if _isnum(raw) else _fmt_num(raw)
        policy = _metric_policy(m) or "trend only"
        trend = (
            f"<td class='spark'>{_svg_sparkline(mhist.get((cell_key, m), []))}</td>"
            if show_trend else ""
        )
        out.append(
            f"<tr><td class='mono'>{_esc(m)}</td>"
            f"<td class='center muted'>{_esc(policy)}</td>"
            f"<td class='num'>{_esc(val)}</td>{trend}</tr>"
        )
    return "".join(out)


def build_dashboard_html(results: list[dict[str, Any]]) -> str:
    """Render the full dashboard HTML from the results history (pure)."""
    status, status_color = _latest_status(results)
    latest = results[-1] if results else {"build": {}, "summary": {}, "entries": []}
    build = latest.get("build", {}) or {}
    s = latest.get("summary", {}) or {}

    # The status table reflects ONLY the newest build's entries -- a cell that
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

    # Per-metric history from metrics.summary (the newest build defines the set).
    metric_pairs: list[tuple[str, str]] = []
    for k, e in latest_by_key.items():
        for m in ((e.get("metrics") or {}).get("summary") or {}):
            metric_pairs.append((k, m))
    mhist: dict[tuple[str, str], list[float | None]] = {p: [] for p in metric_pairs}
    for doc in results:
        seen_m: dict[tuple[str, str], float | None] = {}
        for e in doc.get("entries", []) or []:
            kk = f"{e.get('entry')}::{e.get('cell')}"
            for mm, val in ((e.get("metrics") or {}).get("summary") or {}).items():
                seen_m[(kk, mm)] = val
        for p in metric_pairs:
            mhist[p].append(seen_m.get(p))

    passrate = []
    for doc in results:
        ds = doc.get("summary", {}) or {}
        graded = (ds.get("pass", 0) + ds.get("fail", 0)) or 0
        passrate.append((ds.get("pass", 0) / graded) if graded else None)

    total = s.get("total", 0) or 0
    graded_now = (s.get("pass", 0) or 0) + (s.get("fail", 0) or 0)
    record_now = s.get("record", 0) or 0
    skip_now = s.get("skip", 0) or 0
    # "Nothing was graded" splits three ways and they must not be described
    # identically: every cell recorded a baseline, some recorded while others
    # were skipped, or nothing ran at all. The last of those is unreachable for
    # this pipeline's own output (see _latest_status) and only guards a summary
    # from some other producer.
    nothing_graded = bool(total) and not graded_now

    # Columns that would be uniformly empty are dropped rather than rendered as
    # a wall of "n/a" -- with one night of history that was three of five.
    # Step-time and per-metric history are judged separately: a workload can
    # report metrics.summary without a mean_step_time_ms, and gating its metric
    # sparklines on step time would discard trends that do exist.
    show_trend = _has_trend([history[k] for k in keys])
    show_metric_trend = _has_trend(list(mhist.values()))
    reason_texts = ["; ".join(e.get("reasons") or []) for e in latest_entries]
    distinct_reasons = sorted({r for r in reason_texts if r})
    # Collapse the notes column ONLY when every cell says the same thing, so it
    # can be stated once in a banner. Any variation keeps the column: a cell's
    # explanation -- above all a failure's -- must stay next to the cell.
    collapse_note = (
        len(distinct_reasons) == 1
        and len(reason_texts) > 1
        and all(r == distinct_reasons[0] for r in reason_texts)
    )
    show_notes = bool(distinct_reasons) and not collapse_note
    shared_reason = distinct_reasons[0] if collapse_note else ""

    ncols = 3 + int(show_trend) + int(show_notes)

    # Group cells under their workload so the sweep matrix stays visible.
    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(str(latest_by_key[k].get("entry")), []).append(k)

    body_parts = []
    for entry_name in sorted(groups):
        cell_keys = sorted(groups[entry_name])
        tally: dict[str, int] = {}
        step_times = []
        for k in cell_keys:
            v = latest_by_key[k].get("verdict", "skip")
            tally[v] = tally.get(v, 0) + 1
            st = (latest_by_key[k].get("metrics") or {}).get("mean_step_time_ms")
            if _isnum(st):
                step_times.append(st)
        group_max = max(step_times) if len(step_times) > 1 else 0.0
        tally_txt = " · ".join(
            f"{tally[v]} {v}" for v in _VERDICT_ORDER if tally.get(v)
        )
        cell_word = "cell" if len(cell_keys) == 1 else "cells"

        rows = [
            f"<tr class='grp'><th colspan='{ncols}' scope='rowgroup'>"
            f"<span class='wl'>{_esc(entry_name)}</span>"
            f"<span class='muted'> {len(cell_keys)} {cell_word} · {_esc(tally_txt)}</span>"
            f"</th></tr>"
        ]
        for k in cell_keys:
            e = latest_by_key[k]
            verdict = e.get("verdict", "skip")
            color = _VERDICT_COLOR.get(verdict, "#57606a")
            st = (e.get("metrics") or {}).get("mean_step_time_ms")
            cells = [
                f"<td class='cell mono'>{_esc(str(e.get('cell')))}</td>",
                f"<td class='center'><span class='badge' "
                f"style='background:{color}'>{_esc(verdict)}</span></td>",
                f"<td class='num'>{_esc(_fmt_ms(st))}{_bar(st, group_max)}</td>",
            ]
            if show_trend:
                cells.append(f"<td class='spark'>{_svg_sparkline(history.get(k, []))}</td>")
            if show_notes:
                cells.append(
                    f"<td class='muted'>{_esc('; '.join(e.get('reasons') or []))}</td>"
                )
            rows.append(f"<tr>{''.join(cells)}</tr>")

            mrows = _metric_rows(k, e, mhist, show_metric_trend)
            if mrows:
                n_metrics = len((e.get("metrics") or {}).get("summary") or {})
                recipe = e.get("recipe") or ""
                dur = e.get("duration_sec")
                prov = []
                if recipe:
                    prov.append(f"recipe <span class='mono'>{_esc(str(recipe))}</span>")
                if _isnum(dur):
                    prov.append(f"ran in {dur:,.0f}s")
                trials = e.get("trials")
                if _isnum(trials):
                    n_trials = int(trials)
                    prov.append(f"{n_trials} trial{'s' if n_trials != 1 else ''}")
                rows.append(
                    f"<tr class='mrow'><td colspan='{ncols}'><details>"
                    f"<summary>{n_metrics} metric{'s' if n_metrics != 1 else ''}</summary>"
                    f"<p class='prov'>{' · '.join(prov)}</p>"
                    f"<table class='inner'><thead><tr><th>metric</th>"
                    f"<th class='center'>policy</th><th class='num'>latest</th>"
                    f"{'<th>trend</th>' if show_metric_trend else ''}</tr></thead>"
                    f"<tbody>{mrows}</tbody></table>"
                    f"</details></td></tr>"
                )
        body_parts.append(f"<tbody>{''.join(rows)}</tbody>")

    table_body = "".join(body_parts) or (
        f"<tbody><tr><td colspan='{ncols}' class='muted'>no results yet</td></tr></tbody>"
    )

    head = [
        "<th scope='col'>cell</th>",
        "<th scope='col' class='center'>status</th>",
        "<th scope='col' class='num'>step time</th>",
    ]
    if show_trend:
        head.append("<th scope='col'>trend (step ms)</th>")
    if show_notes:
        head.append("<th scope='col'>notes</th>")

    toolchain = "".join([
        _chip("aorta", build.get("amd_aorta_version")),
        _chip("PyTorch", build.get("torch")),
        _chip("ROCm", build.get("rocm")),
        _chip("HIP", build.get("hip")),
    ])

    provenance = [f"Run of {_esc(_fmt_timestamp(latest.get('generated_at', '')))}"]
    sha = str(build.get("head_sha") or "")
    if sha:
        provenance.append(
            f"commit <a class='mono' href='https://github.com/{_REPO}/commit/{_esc(sha)}'>"
            f"{_esc(sha[:7])}</a>"
        )
    run_id = str(build.get("upstream_run_id") or "")
    if run_id:
        provenance.append(
            f"<a href='https://github.com/{_REPO}/actions/runs/{_esc(run_id)}'>workflow run</a>"
        )
    wheel = str(build.get("wheel_file") or "")
    if wheel:
        provenance.append(f"<span class='mono'>{_esc(wheel)}</span>")

    notices = []
    if not results:
        notices.append(
            "<div class='notice'>No nightly results have been published yet. "
            "This page fills in after the first Nightly Evaluation run.</div>"
        )
    elif nothing_graded and record_now:
        scope = (
            f"all {total} cells" if record_now == total
            else f"{record_now} of {total} cells ({skip_now} skipped)"
        )
        extra = f" Every cell reports: {_esc(shared_reason)}." if shared_reason else ""
        notices.append(
            f"<div class='notice'>No baselines are blessed yet, so nothing was graded "
            f"pass or fail — this run <strong>recorded</strong> metrics for {scope} "
            f"to become the reference.{extra}</div>"
        )
    elif nothing_graded:
        notices.append(
            f"<div class='notice'>Nothing ran: all {total} cells were "
            f"<strong>skipped</strong>, so this build establishes nothing. Check that "
            f"the runner exposes as many GPUs as the matrix asks for.</div>"
        )
    elif shared_reason:
        notices.append(
            f"<div class='notice'>Every cell reports: {_esc(shared_reason)}.</div>"
        )
    if results and not (show_trend or show_metric_trend):
        notices.append(
            "<div class='notice muted-notice'>Trend charts need at least two nightly "
            "runs; they appear automatically once more history accumulates.</div>"
        )

    cards = [
        ("cells", total, ""),
        ("pass", s.get("pass", 0), "#3fb950"),
        ("fail", s.get("fail", 0), "#f85149"),
        ("record", s.get("record", 0), "#d29922"),
        ("skip", s.get("skip", 0), "#8b949e"),
    ]
    # Only worth a card when something was actually graded; otherwise it would
    # sit next to the trend card reading "n/a" twice.
    if passrate and isinstance(passrate[-1], float):
        cards.append(("pass rate", f"{passrate[-1] * 100:.0f}%", ""))
    cards_html = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div>'
        f'<div class="v"{f" style=color:{c}" if c else ""}>{_esc(str(v))}</div></div>'
        for k, v, c in cards
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>aorta nightly CI</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#21262d; --fg:#c9d1d9;
           --muted:#8b949e; --accent:#539bf5; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          margin:0; background:var(--bg); color:var(--fg); line-height:1.5; }}
  a {{ color:var(--accent); }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:2rem 1.25rem 4rem; }}
  header {{ border-bottom:1px solid var(--border); padding-bottom:1.25rem; }}
  .titlebar {{ display:flex; align-items:center; gap:.7rem; flex-wrap:wrap; }}
  h1 {{ font-size:1.5rem; margin:0; }}
  h2 {{ font-size:1.1rem; margin:2.25rem 0 .35rem; color:#e6edf3; }}
  .status-pill {{ display:inline-block; color:#fff; font-weight:600; font-size:.78rem;
                  padding:3px 11px; border-radius:999px; background:{status_color}; }}
  .lede {{ color:var(--muted); font-size:.9rem; margin:.5rem 0 0; max-width:70ch; }}
  .nav {{ font-size:.85rem; margin:.5rem 0 0; }}
  /* Toolchain identity: a labelled grid, so each version is readable on its
     own instead of one dot-separated run of small grey text. */
  .chips {{ display:grid; gap:.5rem; margin:1.1rem 0 0;
            grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); }}
  .chip {{ background:var(--panel); border:1px solid var(--border);
           border-radius:8px; padding:.45rem .7rem; min-width:0; }}
  .chip .k {{ font-size:.65rem; color:var(--muted); text-transform:uppercase;
              letter-spacing:.06em; }}
  .chip .v {{ font-size:.82rem; overflow-wrap:anywhere; }}
  .prov-line {{ color:var(--muted); font-size:.8rem; margin:.8rem 0 0; }}
  .notice {{ background:#1c2128; border:1px solid var(--border);
             border-left:3px solid var(--record, #9a6700); border-radius:6px;
             padding:.6rem .85rem; margin:1.1rem 0 0; font-size:.86rem; }}
  .notice.muted-notice {{ border-left-color:var(--border); color:var(--muted); }}
  .cards {{ display:grid; gap:.6rem; margin:1.25rem 0 0;
            grid-template-columns:repeat(auto-fit, minmax(96px, 1fr)); }}
  .card {{ background:var(--panel); border:1px solid var(--border);
           border-radius:8px; padding:.5rem .85rem; }}
  .card .k {{ font-size:.65rem; color:var(--muted); text-transform:uppercase;
              letter-spacing:.06em; }}
  .card .v {{ font-size:1.35rem; font-weight:600; line-height:1.3; }}
  .card.trend {{ grid-column:span 2; min-width:0; }}
  /* Block-level SVG avoids inline baseline spacing without zeroing the font
     size, which would also hide the "n/a" fallback for short histories. */
  .card.trend svg {{ display:block; }}
  .tablewrap {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; margin-top:.5rem;
           background:var(--panel); border:1px solid var(--border);
           border-radius:8px; overflow:hidden; }}
  th, td {{ padding:.5rem .8rem; border-bottom:1px solid var(--border);
            font-size:.88rem; vertical-align:middle; text-align:left; }}
  thead th {{ color:var(--muted); font-weight:600; font-size:.68rem;
              text-transform:uppercase; letter-spacing:.06em; background:#12161b;
              position:sticky; top:0; z-index:1; }}
  tr.grp th {{ background:#12171e; font-size:.9rem; font-weight:600;
               border-top:1px solid var(--border); }}
  tr.grp .wl {{ font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
                color:#e6edf3; }}
  tr.grp .muted {{ font-weight:400; }}
  td.cell {{ padding-left:1.6rem; }}
  tbody tr:hover {{ background:#1b2129; }}
  tbody tr.grp:hover, tbody tr.mrow:hover {{ background:transparent; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums;
                    white-space:nowrap; }}
  td.center, th.center {{ text-align:center; }}
  td.spark {{ width:180px; }}
  td.spark svg {{ display:block; }}
  /* Step time is compared within its workload group, where the ratio means
     something; the number stays the primary read and the bar is a hint. */
  .bartrack {{ height:3px; background:#21262d; border-radius:2px; margin-top:4px; }}
  .bar {{ height:3px; background:var(--accent); border-radius:2px; opacity:.75; }}
  tr.mrow > td {{ padding:0 .8rem .4rem 1.6rem; }}
  tr.mrow summary {{ cursor:pointer; color:var(--muted); font-size:.78rem;
                     padding:.15rem 0; }}
  tr.mrow .prov {{ color:var(--muted); font-size:.75rem; margin:.3rem 0 .1rem; }}
  table.inner {{ margin:.3rem 0 .6rem; background:#12161b; }}
  table.inner th, table.inner td {{ font-size:.8rem; padding:.35rem .6rem; }}
  table.inner thead th {{ position:static; }}
  .badge {{ display:inline-block; min-width:62px; text-align:center; color:#fff;
            padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:600; }}
  .mono {{ font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
           font-size:.82rem; overflow-wrap:anywhere; }}
  .muted {{ color:var(--muted); font-size:.8rem; }}
  .legend {{ color:var(--muted); font-size:.78rem; margin:.7rem 0 0; }}
</style></head>
<body>
  <div class="wrap">
    <header>
      <div class="titlebar">
        <h1>aorta nightly CI</h1>
        <span class="status-pill">{status}</span>
      </div>
      <p class="lede">Every night AORTA installs the freshly built wheel on an
        MI350 runner and replays its workload sweep, comparing each result
        against a blessed baseline. This page is that run.</p>
      <p class="nav"><a href="docs/">AORTA documentation</a> ·
        <a href="https://github.com/{_REPO}">repository</a></p>
      <div class="chips">{toolchain}</div>
      <p class="prov-line">{' · '.join(provenance)}</p>
    </header>

    {''.join(notices)}

    <div class="cards">
      {cards_html}
      <div class="card trend"><div class="k">pass-rate trend</div><div class="v">{_svg_sparkline(passrate, width=240)}</div></div>
    </div>

    <h2>Workloads</h2>
    <div class="tablewrap">
      <table>
        <thead><tr>{''.join(head)}</tr></thead>
        {table_body}
      </table>
    </div>
    <p class="legend">Verdicts: <strong>pass</strong>/<strong>fail</strong> = compared
      against a blessed baseline · <strong>record</strong> = no baseline yet, metrics
      captured as the future reference · <strong>skip</strong> = not enough GPUs on the
      runner. Expand a cell to see its captured metrics and the recipe that produced
      them.</p>
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
