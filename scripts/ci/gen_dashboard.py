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
import math
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

# Colour alone cannot carry a verdict: it is invisible to a red/green colour
# deficiency and to anyone reading a printed or greyscale copy. Every coloured
# cell states its verdict with one of these too.
_VERDICT_GLYPH = {
    "pass": "✓",
    "fail": "✗",
    "record": "◆",
    "skip": "○",
}

# Worst-first, so a group's tally leads with whatever needs attention.
_VERDICT_ORDER = ("fail", "record", "skip", "pass")

# How many nightly runs the history grid shows before it starts scrolling. The
# full history still ships in data.json; this only bounds the rendered rows.
_GRID_RUNS = 14

# A metric has to move more than this to be worth reporting as a change. Step
# times on a shared runner wander a few percent between identical runs, so a
# lower bar would fill the section with noise every night.
_MOVE_PCT = 10.0

# Timings the harness records for every result, outside the workload's own
# metrics.summary. They are comparable across runs like any other measurement.
_HARNESS_METRICS = ("mean_step_time_ms", "mean_wall_clock_sec")

# Fields worth diffing between runs, in the order they are reported. AORTA's own
# version is deliberately absent: it is date-stamped and therefore changes every
# single night, so reporting it would mark every run as a change and teach the
# reader to ignore the one marker that should mean something. What can actually
# explain a regression is the stack underneath.
_TOOLCHAIN_FIELDS = (
    ("torch", "PyTorch"),
    ("rocm", "ROCm"),
    ("hip", "HIP"),
)


def _isnum(v: Any) -> bool:
    """Numeric, finite, and representable as a float — i.e. safe to render.

    Three things in the (untrusted) results JSON pass a bare isinstance check
    and should not: ``bool``, which is a subclass of ``int`` and would format as
    a step time or count towards a trend; ``NaN``/``Infinity``, which
    ``json.loads`` accepts and which plot as ``nan`` coordinates and a bar that
    silently reads full-width; and an integer too large to convert to a float,
    which raises ``OverflowError`` in every formatter below.
    """
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:  # int beyond the float range
        return False


def _count(v: Any) -> int:
    """A summary count as an int; anything unusable counts as 0.

    Counts drive branching and division, so a string in a malformed summary
    would crash generation outright ("4" + 0 raises) instead of rendering badly.
    Display still goes through ``_fmt_num``, which shows "—" rather than a 0 it
    cannot vouch for.
    """
    return int(v) if _isnum(v) else 0


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


def _svg_sparkline(
    values: list[float | None], width: int = 160, height: int = 32
) -> str:
    """Tiny inline SVG line chart for a metric's history (ignores non-numbers)."""
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
    if _count(s.get("fail")):
        return "failing", _VERDICT_COLOR["fail"]
    if _count(s.get("total")) == 0:
        return "empty", "#57606a"
    if _count(s.get("pass")):
        return "passing", _VERDICT_COLOR["pass"]
    if _count(s.get("record")):
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


def _key(entry: dict[str, Any]) -> str:
    return f"{entry.get('entry')}::{entry.get('cell')}"


def _cell_label(entry: dict[str, Any]) -> str:
    """``workload::cell`` for prose, with the cell-less form spelled out."""
    name = entry.get("cell")
    return f"{entry.get('entry')}::{name}" if name else f"{entry.get('entry')} (whole workload)"


def _measurements(entry: dict[str, Any]) -> dict[str, float]:
    """Every comparable number a result reports, keyed by metric name.

    The harness's own timings sit beside the workload's ``summary`` metrics so
    all of them move through the same comparison; a workload can report either
    without the other. Both timings matter and they can disagree: step time can
    hold steady while wall clock climbs, which is what a slower setup or a
    stalling teardown looks like.
    """
    metrics = entry.get("metrics") or {}
    out: dict[str, float] = {}
    for name in _HARNESS_METRICS:
        val = metrics.get(name)
        if _isnum(val):
            out[name] = float(val)
    for name, val in (metrics.get("summary") or {}).items():
        if _isnum(val):
            out[str(name)] = float(val)
    return out


def _tally(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in entries:
        v = str(e.get("verdict", "skip"))
        counts[v] = counts.get(v, 0) + 1
    return counts


def _tally_text(counts: dict[str, int]) -> str:
    known = " · ".join(f"{counts[v]} {v}" for v in _VERDICT_ORDER if counts.get(v))
    # A verdict this generator has no colour for still has to be counted; the
    # results JSON is not ours to constrain.
    extra = " · ".join(
        f"{counts[v]} {v}" for v in sorted(counts) if v not in _VERDICT_ORDER
    )
    return " · ".join(p for p in (known, extra) if p)


def build_history_grid(results: list[dict[str, Any]], max_runs: int = _GRID_RUNS) -> str:
    """One row per nightly run, one column per workload (pure).

    The page's other table answers "how is tonight?"; this one answers "how has
    this workload been?", which no single-run view can. Cells carry the count of
    results and take the colour of the worst verdict among them, so a column
    that turns red and stays red is visible without reading any number.
    """
    runs = results[-max_runs:] if max_runs > 0 else list(results)
    if len(runs) < 2:
        return ""
    # The run just outside the window, so the oldest displayed row is compared
    # against the run that actually preceded it. Comparing it against nothing
    # would silently drop a toolchain change from that row every time history
    # grows past the window.
    prior = results[-(len(runs) + 1)] if len(results) > len(runs) else None

    # Column order follows the newest run so tonight's matrix reads left to
    # right as configured; workloads that have since been retired trail it
    # rather than disappearing, which would hide that they ever ran.
    newest = [str(e.get("entry")) for e in (runs[-1].get("entries") or [])]
    ordered: list[str] = []
    for name in newest:
        if name not in ordered:
            ordered.append(name)
    for doc in reversed(runs):
        for e in doc.get("entries") or []:
            name = str(e.get("entry"))
            if name not in ordered:
                ordered.append(name)
    if not ordered:
        return ""

    head = "".join(
        f"<th scope='col' class='wlcol' title='{_esc(n)}'>{_esc(n)}</th>" for n in ordered
    )

    newest_first = list(reversed(runs))
    rows = []
    for i, doc in enumerate(newest_first):
        build = doc.get("build") or {}
        by_entry: dict[str, list[dict[str, Any]]] = {}
        for e in doc.get("entries") or []:
            by_entry.setdefault(str(e.get("entry")), []).append(e)

        status, color = _latest_status([doc])
        date = _fmt_timestamp(str(doc.get("generated_at") or "")).split(" ")[0] or "—"
        run_id = str(build.get("upstream_run_id") or "")
        when = (
            f"<a href='https://github.com/{_REPO}/actions/runs/{_esc(run_id)}'>{_esc(date)}</a>"
            if run_id else _esc(date)
        )

        # Mark the run where the toolchain moved: a column that changes colour on
        # exactly that row is the whole point of running this nightly.
        older = newest_first[i + 1] if i + 1 < len(newest_first) else prior
        bumped = older is not None and any(
            str((older.get("build") or {}).get(f) or "") != str(build.get(f) or "")
            for f, _ in _TOOLCHAIN_FIELDS
        )
        flag = " <span class='bump' title='toolchain changed in this run'>bump</span>" if bumped else ""

        cells = []
        for name in ordered:
            group = by_entry.get(name) or []
            if not group:
                cells.append("<td class='gcell'><span class='absent' title='not in this run'>·</span></td>")
                continue
            counts = _tally(group)
            worst = next((v for v in _VERDICT_ORDER if counts.get(v)), "")
            n = len(group)
            worst_n = counts.get(worst, 0)
            count = str(n) if worst_n == n else f"{worst_n}/{n}"
            bg = _VERDICT_COLOR.get(worst, "#57606a")
            # The glyph, not the colour, is what states the verdict: a bare
            # count renders identically for a passing and a failing group, which
            # leaves anyone who cannot separate red from green — or who is on a
            # touch screen with no hover — unable to read the row at all.
            breakdown = f"{name} — {_tally_text(counts)}"
            cells.append(
                f"<td class='gcell'><span class='dot' style='background:{bg}' "
                f"title='{_esc(breakdown)}' aria-label='{_esc(breakdown)}'>"
                f"{_esc(_VERDICT_GLYPH.get(worst, '?'))} {_esc(count)}</span></td>"
            )

        rows.append(
            f"<tr><th scope='row' class='runcell'>{when}{flag}"
            f"<span class='runmeta'>{_esc(str(build.get('amd_aorta_version') or ''))}</span></th>"
            f"<td class='center'><span class='badge sm' style='background:{color}'>"
            f"{_esc(status)}</span></td>{''.join(cells)}</tr>"
        )

    return (
        "<h2>Run history</h2>"
        "<p class='muted'>Each row is one nightly run, newest first; each column is a "
        "workload. A cell shows the worst verdict among that workload's results "
        "(✓ pass · ✗ fail · ◆ record · ○ skip) and how many there were — "
        "<span class='mono'>✗ 1/4</span> means one of four failed. Hover for the "
        "full breakdown.</p>"
        "<div class='tablewrap'><table class='grid'>"
        f"<thead><tr><th scope='col'>run</th><th scope='col' class='center'>status</th>"
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def build_change_summary(results: list[dict[str, Any]]) -> str:
    """What moved between the two most recent runs (pure).

    The nightly exists to catch what a new wheel broke, and that is a question
    about the difference between two runs, not about either one of them. Ordered
    by what would make someone act: the toolchain that changed, then verdicts
    that flipped, then the matrix gaining or losing cells, then metrics that
    moved far enough to mean something.
    """
    if len(results) < 2:
        return ""
    prev, latest = results[-2], results[-1]
    prev_by = {_key(e): e for e in (prev.get("entries") or [])}
    latest_by = {_key(e): e for e in (latest.get("entries") or [])}

    items: list[str] = []

    bumps = []
    for field, label in _TOOLCHAIN_FIELDS:
        before = str((prev.get("build") or {}).get(field) or "")
        after = str((latest.get("build") or {}).get(field) or "")
        if before != after and (before or after):
            bumps.append(
                f"{_esc(label)} <span class='mono'>{_esc(before or 'none')}</span> → "
                f"<span class='mono'>{_esc(after or 'none')}</span>"
            )
    if bumps:
        items.append(f"<li class='bumped'>Toolchain: {' · '.join(bumps)}</li>")

    flips = []
    for k in sorted(set(prev_by) & set(latest_by)):
        was = str(prev_by[k].get("verdict", ""))
        now = str(latest_by[k].get("verdict", ""))
        if was != now:
            flips.append((now, was, latest_by[k]))
    # Regressions first: a cell that started failing is the reason to read this.
    flips.sort(key=lambda f: (_VERDICT_ORDER.index(f[0]) if f[0] in _VERDICT_ORDER else 9))
    for now, was, entry in flips:
        color = _VERDICT_COLOR.get(now, "#57606a")
        why = "; ".join(entry.get("reasons") or [])
        tail = f" <span class='muted'>{_esc(why)}</span>" if why else ""
        items.append(
            f"<li><span class='mono'>{_esc(_cell_label(entry))}</span> "
            f"{_esc(was)} → <span class='badge sm' style='background:{color}'>"
            f"{_esc(now)}</span>{tail}</li>"
        )

    gained = sorted(set(latest_by) - set(prev_by))
    lost = sorted(set(prev_by) - set(latest_by))
    for k in gained:
        items.append(
            f"<li>new result <span class='mono'>{_esc(_cell_label(latest_by[k]))}</span></li>"
        )
    for k in lost:
        items.append(
            f"<li>no longer reported <span class='mono'>{_esc(_cell_label(prev_by[k]))}</span></li>"
        )

    movers = []
    for k in sorted(set(prev_by) & set(latest_by)):
        before_m = _measurements(prev_by[k])
        after_m = _measurements(latest_by[k])
        for metric in sorted(set(before_m) & set(after_m)):
            b, a = before_m[metric], after_m[metric]
            if not b:  # a move from zero has no meaningful percentage
                continue
            pct = (a - b) / abs(b) * 100.0
            if abs(pct) > _MOVE_PCT:
                movers.append((abs(pct), pct, k, metric, b, a, latest_by[k]))
    movers.sort(reverse=True)
    for _, pct, _k, metric, b, a, entry in movers[:6]:
        unit = _METRIC_UNITS.get(metric, "")
        arrow = "▲" if pct > 0 else "▼"
        items.append(
            f"<li><span class='mono'>{_esc(_cell_label(entry))}</span> "
            f"<span class='mono'>{_esc(metric)}</span> {arrow} {pct:+.0f}% "
            f"<span class='muted'>{_esc(_fmt_num(b))} → {_esc(_fmt_num(a))}"
            f"{(' ' + _esc(unit)) if unit else ''}</span></li>"
        )

    since = _fmt_timestamp(str(prev.get("generated_at") or "")).split(" ")[0]
    if not items:
        body = (
            "<p class='steady'>Nothing moved: same workloads, same verdicts, and no "
            f"metric changed by more than {_MOVE_PCT:.0f}%.</p>"
        )
    else:
        extra = len(movers) - 6
        more = (
            f"<p class='muted'>and {extra} more metric"
            f"{'' if extra == 1 else 's'} past {_MOVE_PCT:.0f}%</p>"
            if extra > 0 else ""
        )
        body = f"<ul class='changes'>{''.join(items)}</ul>{more}"

    return (
        f"<h2>What changed since {_esc(since or 'the previous run')}</h2>{body}"
    )


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
        passed, failed = _count(ds.get("pass")), _count(ds.get("fail"))
        graded = passed + failed
        passrate.append((passed / graded) if graded else None)

    total = _count(s.get("total"))
    graded_now = _count(s.get("pass")) + _count(s.get("fail"))
    record_now = _count(s.get("record"))
    skip_now = _count(s.get("skip"))
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
        n = len(cell_keys)
        head = f"{n} result{'' if n == 1 else 's'} · {tally_txt}"
        # duration_sec is measured once around the whole recipe invocation and
        # copied onto every record it produced, so it belongs to the workload and
        # is stated here once -- repeated in each cell it reads as per-cell time.
        durs = {
            d for d in (latest_by_key[k].get("duration_sec") for k in cell_keys)
            if _isnum(d)
        }
        # A skipped workload reports 0.0 -- it never ran, so it has no duration.
        if len(durs) == 1 and max(durs) > 0:
            head += f" · workload run {max(durs):,.0f}s"

        rows = [
            f"<tr class='grp'><th colspan='{ncols}' scope='rowgroup'>"
            f"<span class='wl'>{_esc(entry_name)}</span>"
            f"<span class='muted'> {_esc(head)}</span>"
            f"</th></tr>"
        ]
        for k in cell_keys:
            e = latest_by_key[k]
            verdict = e.get("verdict", "skip")
            color = _VERDICT_COLOR.get(verdict, "#57606a")
            st = (e.get("metrics") or {}).get("mean_step_time_ms")
            # A record for a workload that never reached its cells (skipped for
            # want of GPUs, or the synthetic zero-work entry) carries cell=None;
            # printing that as "None" invites reading it as a cell's name.
            cell_name = e.get("cell")
            label = (
                f"<span class='mono'>{_esc(str(cell_name))}</span>"
                if cell_name else "<span class='muted'>whole workload</span>"
            )
            cells = [
                f"<td class='cell'>{label}</td>",
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
                prov = []
                if recipe:
                    prov.append(f"recipe <span class='mono'>{_esc(str(recipe))}</span>")
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
            f"all {total} results" if record_now == total
            else f"{record_now} of {total} results ({skip_now} skipped)"
        )
        extra = f" Every result reports: {_esc(shared_reason)}." if shared_reason else ""
        notices.append(
            f"<div class='notice'>No baselines are blessed yet, so nothing was graded "
            f"pass or fail — this run <strong>recorded</strong> metrics for {scope} "
            f"to become the reference.{extra}</div>"
        )
    elif nothing_graded:
        notices.append(
            f"<div class='notice'>Nothing ran: all {total} results were "
            f"<strong>skipped</strong>, so this build establishes nothing. Check that "
            f"the runner exposes as many GPUs as the matrix asks for.</div>"
        )
    elif shared_reason:
        notices.append(
            f"<div class='notice'>Every result reports: {_esc(shared_reason)}.</div>"
        )
    if results and not (show_trend or show_metric_trend):
        notices.append(
            "<div class='notice muted-notice'>Trend charts need at least two nightly "
            "runs; they appear automatically once more history accumulates.</div>"
        )

    # Counts go through _fmt_num so a malformed summary shows "—" rather than
    # printing whatever str() makes of it ("nan", "True").
    cards = [
        ("results", _fmt_num(s.get("total", 0)), ""),
        ("pass", _fmt_num(s.get("pass", 0)), "#3fb950"),
        ("fail", _fmt_num(s.get("fail", 0)), "#f85149"),
        ("record", _fmt_num(s.get("record", 0)), "#d29922"),
        ("skip", _fmt_num(s.get("skip", 0)), "#8b949e"),
    ]
    # Only worth a card when something was actually graded; otherwise it would
    # sit next to the trend card reading "n/a" twice.
    if passrate and _isnum(passrate[-1]):
        cards.append(("pass rate", f"{passrate[-1] * 100:.0f}%", ""))
    cards_html = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div>'
        f'<div class="v"{f" style=color:{c}" if c else ""}>{_esc(str(v))}</div></div>'
        for k, v, c in cards
    )

    changes_html = build_change_summary(results)
    history_html = build_history_grid(results)

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
  .badge.sm {{ min-width:0; font-size:.68rem; padding:1px 7px; }}
  .mono {{ font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
           font-size:.82rem; overflow-wrap:anywhere; }}
  .muted {{ color:var(--muted); font-size:.8rem; }}
  .legend {{ color:var(--muted); font-size:.78rem; margin:.7rem 0 0; }}
  /* What changed: a list read top-down, so the eye lands on the toolchain bump
     and the regressions before the smaller metric moves. */
  ul.changes {{ list-style:none; margin:.5rem 0 0; padding:0; }}
  ul.changes li {{ background:var(--panel); border:1px solid var(--border);
                   border-left:3px solid var(--border); border-radius:6px;
                   padding:.45rem .75rem; margin:.35rem 0; font-size:.86rem; }}
  ul.changes li.bumped {{ border-left-color:var(--accent); }}
  .steady {{ color:var(--muted); font-size:.86rem; margin:.5rem 0 0; }}
  /* Run history: dense by design -- one glance should show a column that has
     been red for a week, so cells stay small and colour does the talking. */
  table.grid th.wlcol {{ font-size:.62rem; max-width:96px; overflow:hidden;
                         text-overflow:ellipsis; white-space:nowrap; }}
  table.grid td.gcell {{ text-align:center; padding:.35rem .4rem; }}
  table.grid th.runcell {{ white-space:nowrap; font-weight:600; font-size:.82rem; }}
  .runmeta {{ display:block; color:var(--muted); font-weight:400; font-size:.68rem;
              font-family:ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .bump {{ display:inline-block; margin-left:.4rem; padding:0 6px; border-radius:999px;
           background:#1f6feb33; color:var(--accent); font-size:.62rem;
           font-weight:600; text-transform:uppercase; letter-spacing:.05em; }}
  .dot {{ display:inline-block; min-width:44px; padding:2px 6px; border-radius:6px;
          color:#fff; font-size:.72rem; font-weight:600; white-space:nowrap;
          font-variant-numeric:tabular-nums; }}
  .absent {{ color:#39414a; }}
  .secthead {{ display:flex; align-items:baseline; justify-content:space-between;
               gap:1rem; flex-wrap:wrap; }}
  .toolbar button {{ background:var(--panel); color:var(--fg);
                     border:1px solid var(--border); border-radius:6px;
                     padding:.25rem .6rem; font-size:.75rem; cursor:pointer; }}
  .toolbar button:hover {{ border-color:var(--accent); color:var(--accent); }}
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

    {changes_html}

    {history_html}

    <div class="secthead">
      <h2>Workloads</h2>
      <div class="toolbar" id="toolbar" hidden>
        <button type="button" data-details="open">Expand all</button>
        <button type="button" data-details="close">Collapse all</button>
      </div>
    </div>
    <div class="tablewrap">
      <table>
        <thead><tr>{''.join(head)}</tr></thead>
        {table_body}
      </table>
    </div>
    <p class="legend">Verdicts: <strong>pass</strong>/<strong>fail</strong> = compared
      against a blessed baseline · <strong>record</strong> = no baseline yet, metrics
      captured as the future reference · <strong>skip</strong> = not enough GPUs on the
      runner. One row is one recorded result: a workload skipped before it reached
      its cells yields a single row for the whole workload, so a count of rows is
      not a count of configured cells. Expand a row for its metrics and recipe.</p>
  </div>
<script>
// Progressive enhancement only: the page is complete without this, so the
// controls stay hidden until the script that drives them has actually run.
(function () {{
  var bar = document.getElementById("toolbar");
  if (!bar) return;
  // Nothing to expand when no result carried metrics, and a pair of buttons
  // that provably do nothing is worse than no buttons.
  if (!document.querySelectorAll("tr.mrow details").length) return;
  bar.hidden = false;
  bar.addEventListener("click", function (ev) {{
    var want = ev.target && ev.target.getAttribute("data-details");
    if (!want) return;
    var open = want === "open";
    var all = document.querySelectorAll("tr.mrow details");
    for (var i = 0; i < all.length; i++) all[i].open = open;
  }});
}})();
</script>
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
