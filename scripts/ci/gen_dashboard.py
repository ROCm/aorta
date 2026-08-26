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
    _is_correctness_metric = eval_lib.is_correctness_metric
    _is_performance_metric = eval_lib.is_performance_metric
except Exception:  # pragma: no cover - dashboard must render even if import fails
    def _metric_policy(_name: str):  # type: ignore
        return None

    def _is_correctness_metric(_name: str) -> bool:  # type: ignore
        return _name.endswith("checksum")

    def _is_performance_metric(name: str) -> bool:  # type: ignore
        return name in _METRIC_UNITS and name not in (
            "logits_checksum", "output_checksum", "checksum"
        )

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
    "mean_step_time_ms": "ms",
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

# History-chip dots use white text at small size; pass green must meet 4.5:1 AA.
_CHIP_VERDICT_COLOR = dict(_VERDICT_COLOR)
_CHIP_VERDICT_COLOR["pass"] = "#1a7f37"

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

# A verdict this generator has no colour or glyph for. Kept distinct from skip's
# grey: an unrecognised verdict outranks skip, so rendering the two identically
# would hide that difference behind a hover.
_UNKNOWN_COLOR = "#8250df"
_UNKNOWN_GLYPH = "?"

# How many canary runs the observed-only section shows (issue #382). Short on
# purpose: the value of the lane is "which ROCm version changed something
# recently", and the full history is on the ci-results branch.
_CANARY_ROWS = 14

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

try:
    from dashboard_metadata import DASHBOARD_METADATA as _DEFAULT_METADATA
except ImportError:  # pragma: no cover
    _DEFAULT_METADATA: dict[str, Any] = {
        "categories": {}, "workloads": {},
    }

# Plain-language headline for each internal _latest_status() label.
_CUSTOMER_STATUS = {
    "unknown": "Unknown",
    "empty": "No data",
    "unrecognised": "Unrecognised verdict",
    "failing": "Regression detected",
    "passing": "Healthy",
    "partial": "Partial run",
    "recording": "Baseline setup",
    "skipping": "Incomplete run",
}


def load_dashboard_metadata() -> dict[str, Any]:
    """Load workload/category metadata for the dashboard (pure, cached per process)."""
    if not hasattr(load_dashboard_metadata, "_cache"):
        load_dashboard_metadata._cache = None  # type: ignore[attr-defined]
    if load_dashboard_metadata._cache is not None:  # type: ignore[attr-defined]
        return load_dashboard_metadata._cache  # type: ignore[attr-defined]
    load_dashboard_metadata._cache = dict(_DEFAULT_METADATA)  # type: ignore[attr-defined]
    return load_dashboard_metadata._cache  # type: ignore[attr-defined]


def _customer_status_label(status: str) -> str:
    return _CUSTOMER_STATUS.get(status, status.replace("_", " ").title())


def _workload_meta(entry_name: str) -> dict[str, Any]:
    return (load_dashboard_metadata().get("workloads") or {}).get(entry_name) or {}


# Correctness counters: any night-over-night change is significant.
_CORRECTNESS_COUNTERS = frozenset({
    "ranks_with_divergence",
    "layer_checksum_mismatches",
})


def _is_correctness_change_metric(name: str) -> bool:
    """True when any value change in this metric should surface in Correctness."""
    return _is_correctness_metric(name) or name in _CORRECTNESS_COUNTERS


def _is_dashboard_performance_metric(name: str) -> bool:
    """True for timing/throughput metrics the dashboard treats as performance.

    Uses the dashboard's own unit map (and harness timings), not the eval-lib
    gating allowlist — step_time_p50/p99 and mean_step_time_ms are headline
    metrics here but are not eval_lib performance gates.
    """
    if _is_correctness_change_metric(name):
        return False
    return name in _METRIC_UNITS or name in _HARNESS_METRICS


def _checksum_metric_failed(entry: dict[str, Any], name: str) -> bool:
    """True when the harness reported this checksum metric failed comparison."""
    needle = name.replace("_", " ")
    for reason in entry.get("reasons") or []:
        text = str(reason)
        lower = text.lower()
        if name in text or needle in text or "checksum" in lower:
            if "!=" in text or "expected" in lower or "mismatch" in lower:
                return True
    return False


def _checksum_headline(name: str, raw: Any, entry: dict[str, Any]) -> str:
    """Headline for checksum metrics — neutral unless the harness reports failure.

    Checksums are serialized as floats in nightly JSON; both ``observed == expected``
    and a ``pass`` verdict can be wrong for large int64 values that collide after
    rounding. Only surface a definite mismatch from failure reasons; otherwise
    show the captured value.
    """
    label = name.replace("_", " ")
    if not _isnum(raw):
        return f"{label}: —"
    if str(entry.get("verdict", "")) == "fail" and _checksum_metric_failed(entry, name):
        return f"{label}: mismatch ✗"
    return f"{label}: captured ({_fmt_num(raw)})"


def _format_headline_metric(
    name: str, raw: Any, entry: dict[str, Any] | None = None,
) -> str:
    """One headline metric for customer view."""
    entry = entry or {}
    if name in _CORRECTNESS_COUNTERS:
        if _isnum(raw) and float(raw) == 0:
            return f"{name.replace('_', ' ')}: 0 ✓"
        if _isnum(raw):
            return f"{name.replace('_', ' ')}: {_fmt_num(raw)} ✗"
        return f"{name.replace('_', ' ')}: —"
    if name in ("logits_checksum", "output_checksum", "checksum"):
        return _checksum_headline(name, raw, entry)
    unit = _METRIC_UNITS.get(name, "")
    if name == "mean_step_time_ms":
        return f"step time: {_fmt_ms(raw)}"
    label = name.replace("_", " ")
    if _isnum(raw):
        val = f"{_fmt_num(raw)} {unit}".strip()
        return f"{label}: {val}"
    return f"{label}: —"


def _headline_block(entry: dict[str, Any], entry_name: str) -> str:
    meta = _workload_meta(entry_name)
    names = meta.get("headline_metrics") or []
    if not names:
        return ""
    summary = ((entry.get("metrics") or {}).get("summary") or {})
    metrics = entry.get("metrics") or {}
    parts = []
    for name in names:
        if name in _HARNESS_METRICS:
            raw = metrics.get(name)
        else:
            raw = summary.get(name)
        parts.append(_format_headline_metric(str(name), raw, entry))
    if not parts:
        return ""
    return (
        "<ul class='headlines'>"
        + "".join(f"<li>{_esc(p)}</li>" for p in parts)
        + "</ul>"
    )


# Metrics hidden from the optional detailed table — still in headline view when relevant.
_ENGINEER_HIDDEN_METRICS = frozenset({
    "rank",
    "world_size",
    "local_world_size",
    "node_count",
    "parameter_count",
})


def _grading_rule_label(entry: dict[str, Any], name: str) -> str:
    """Plain-language rule for one metric in this cell's detailed table.

    Uses the policy recorded in ``entry["deltas"]`` from ``compare_to_baseline``
    when present; otherwise the metric was not gated for this run.
    """
    delta = ((entry.get("deltas") or {}).get("metrics") or {}).get(name) or {}
    policy = delta.get("policy")
    if policy == "equal":
        return "must match baseline"
    if policy == "min":
        return "higher is better"
    if policy == "max":
        return "lower is better"
    if policy:
        return str(policy)
    return "tracked only (not gated)"


def _substitute_repro_pins(text: str, build: dict[str, Any]) -> str:
    """Fill repro setup commands with the toolchain pins from the newest nightly."""
    sha = str(build.get("head_sha") or "").strip()
    ver = str(build.get("amd_aorta_version") or "").strip()
    if sha:
        text = text.replace("{{HEAD_SHA}}", sha)
    else:
        text = text.replace(
            "git checkout {{HEAD_SHA}}",
            "# git checkout <head_sha shown in the dashboard header>",
        )
    if ver:
        text = text.replace("{{AORTA_VERSION}}", ver)
    else:
        text = text.replace(
            "python3 -m pip install --upgrade --pre "
            "'amd-aorta[hw-queue]=={{AORTA_VERSION}}' "
            "-f https://github.com/ROCm/aorta/releases/expanded_assets/dev-wheels",
            "# STOP: dashboard is missing amd_aorta_version — copy the AORTA version "
            "from the dashboard header, then run:\n"
            "# python3 -m pip install --upgrade --pre 'amd-aorta[hw-queue]==<version>' "
            "-f https://github.com/ROCm/aorta/releases/expanded_assets/dev-wheels",
        )
    return text


def _repro_commands_html(commands: list[str], build: dict[str, Any]) -> str:
    return "".join(
        f"<pre class='mono repro-cmd'>{_esc(_substitute_repro_pins(c, build))}</pre>"
        for c in commands if c
    )


def _run_command_block(entry_name: str, build: dict[str, Any]) -> str:
    """Exhaustive, workload-specific reproduction guide outside CI."""
    meta = _workload_meta(entry_name)
    repro = meta.get("repro") or {}
    cmd = str((repro.get("run") or {}).get("command") or meta.get("run_command") or "")
    if not cmd:
        return ""
    cmd = _substitute_repro_pins(cmd, build)
    recipe = str(meta.get("recipe") or "")
    parts: list[str] = []

    prereq = repro.get("prerequisites") or []
    if prereq:
        items = "".join(f"<li>{_esc(p)}</li>" for p in prereq)
        parts.append(
            f"<section class='repro-sec'><h5>Before you start</h5>"
            f"<ul class='repro-list'>{items}</ul></section>"
        )

    step_no = 1
    for step in repro.get("setup") or []:
        title = str(step.get("title") or f"Setup step {step_no}")
        parts.append(
            f"<section class='repro-sec'><h5>{step_no}. {_esc(title)}</h5>"
            f"{_repro_commands_html(list(step.get('commands') or []), build)}</section>"
        )
        step_no += 1

    dry = repro.get("dry_run") or {}
    dry_cmd = str(dry.get("command") or "")
    if dry_cmd:
        dry_title = str(dry.get("title") or "Validate the recipe YAML (no GPU execution)")
        parts.append(
            f"<section class='repro-sec'><h5>{step_no}. {_esc(dry_title)}</h5>"
            f"<pre class='mono repro-cmd'>{_esc(dry_cmd)}</pre></section>"
        )
        step_no += 1

    run = repro.get("run") or {}
    run_title = str(run.get("title") or "Run the nightly recipe")
    parts.append(
        f"<section class='repro-sec'><h5>{step_no}. {_esc(run_title)}</h5>"
        f"<p class='repro-note muted'>Run from the repository root after setup completes. "
        f"A standalone <code>--strict</code> sweep only fails cells that error or never run. "
        f"Dashboard pass/record/fail comes separately from nightly_eval.py comparing "
        f"matrix.json against config/ci/regression_baselines.yaml; run that harness to "
        f"apply the dashboard gates.</p>"
        f"<pre class='mono repro-cmd'>{_esc(cmd)}</pre></section>"
    )
    step_no += 1

    for step in repro.get("verify") or []:
        title = str(step.get("title") or "Verify results")
        parts.append(
            f"<section class='repro-sec'><h5>{step_no}. {_esc(title)}</h5>"
            f"{_repro_commands_html(list(step.get('commands') or []), build)}</section>"
        )
        step_no += 1

    success = str(repro.get("success_criteria") or "")
    if success:
        parts.append(
            f"<section class='repro-sec'><h5>Success criteria</h5>"
            f"<p>{_esc(success)}</p></section>"
        )

    compare = str(repro.get("compare_notes") or "")
    if compare:
        parts.append(
            f"<section class='repro-sec'><h5>Compare with nightly numbers</h5>"
            f"<p class='muted'>{_esc(compare)}</p></section>"
        )

    if recipe:
        parts.append(
            f"<p class='repro-note muted'>Recipe YAML: "
            f"<a class='mono' href='https://github.com/{_REPO}/blob/main/{_esc(recipe)}'>"
            f"{_esc(recipe)}</a> · "
            f"<a href='https://github.com/{_REPO}/blob/main/recipes/README-running-recipes.md'>"
            f"Multi-node launcher guide</a></p>"
        )

    return (
        f"<details class='repro-panel' id='repro-{_esc(entry_name)}'>"
        f"<summary>Reproduce this workload locally</summary>"
        f"<div class='repro-body'>{''.join(parts)}</div></details>"
    )


def _category_statuses(
    latest_entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Structured category health for ``status.json`` consumers."""
    meta = load_dashboard_metadata()
    categories = meta.get("categories") or {}
    by_entry: dict[str, list[dict[str, Any]]] = {}
    for e in latest_entries:
        by_entry.setdefault(str(e.get("entry")), []).append(e)
    out: dict[str, dict[str, Any]] = {}
    for cat_id, cat in categories.items():
        workloads = cat.get("workloads") or []
        group: list[dict[str, Any]] = []
        for wl in workloads:
            group.extend(by_entry.get(str(wl), []))
        if not group:
            continue
        counts = _tally(group)
        worst = _worst_verdict(counts)
        out[str(cat_id)] = {
            "label": str(cat.get("label") or cat_id),
            "worst_verdict": worst,
            "counts": dict(counts),
            "workloads": [str(w) for w in workloads if by_entry.get(str(w))],
        }
    return out


def _headline_metrics_for_entries(
    latest_entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Latest headline metric values keyed by ``entry::cell``."""
    out: dict[str, dict[str, Any]] = {}
    for e in latest_entries:
        entry_name = str(e.get("entry") or "")
        cell = e.get("cell")
        key = f"{entry_name}::{cell}" if cell else entry_name
        metrics: dict[str, Any] = {}
        meta = _workload_meta(entry_name)
        for name in meta.get("headline_metrics") or []:
            raw = ((e.get("metrics") or {}).get("summary") or {}).get(name)
            if name in _HARNESS_METRICS:
                raw = (e.get("metrics") or {}).get(name, raw)
            if _isnum(raw):
                metrics[str(name)] = raw
        if metrics:
            out[key] = metrics
    return out


def build_status_json(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Structured latest-night summary for embeds (separate from the history feed)."""
    latest = results[-1] if results else None
    status, _ = _latest_status(results)
    latest_entries = list((latest or {}).get("entries") or [])
    latest_block: dict[str, Any] = {
        "status": status,
        "customer_status": _customer_status_label(status),
        "generated_at": (latest or {}).get("generated_at"),
        "build": (latest or {}).get("build") or {},
        "summary": (latest or {}).get("summary") or {},
        "categories": _category_statuses(latest_entries),
        "headline_metrics": _headline_metrics_for_entries(latest_entries),
    }
    return {
        "schema_version": 2,
        "latest": latest_block,
    }


def build_scaling_summary(latest_entries: list[dict[str, Any]]) -> str:
    """2-GPU vs 8-GPU step-time scaling for training workloads."""
    by_entry: dict[str, list[dict[str, Any]]] = {}
    for e in latest_entries:
        by_entry.setdefault(str(e.get("entry")), []).append(e)

    def _step_p50(entry_name: str) -> float | None:
        for e in by_entry.get(entry_name) or []:
            raw = ((e.get("metrics") or {}).get("summary") or {}).get("step_time_p50")
            if _isnum(raw):
                return float(raw)
        return None

    pairs = (
        ("training_ddp", "training_ddp_8gpu", "DDP"),
        ("training_fsdp", "training_fsdp_8gpu", "FSDP"),
    )
    rows = []
    for two_gpu, eight_gpu, label in pairs:
        p50_2 = _step_p50(two_gpu)
        p50_8 = _step_p50(eight_gpu)
        if p50_2 is None or p50_8 is None:
            continue
        ratio = p50_8 / p50_2 if p50_2 else None
        # Weak scaling: recipes keep the same per-rank batch, so global batch
        # grows 4× from 2 to 8 GPUs and ideal step time stays flat.
        efficiency = (p50_2 / p50_8 * 100.0) if p50_8 else None
        eff_txt = f"{efficiency:.0f}%" if efficiency is not None else "—"
        ratio_txt = f"{ratio:.2f}×" if ratio is not None else "—"
        rows.append(
            f"<tr><td>{_esc(label)}</td>"
            f"<td class='num'>{_esc(_fmt_ms(p50_2))}</td>"
            f"<td class='num'>{_esc(_fmt_ms(p50_8))}</td>"
            f"<td class='num'>{_esc(ratio_txt)}</td>"
            f"<td class='num'>{_esc(eff_txt)}</td></tr>"
        )
    if not rows:
        return ""
    return (
        "<h2>Training scaling (2 → 8 GPU)</h2>"
        "<p class='muted'>Reference step_time_p50 from the latest nightly. "
        "Recipes use the same per-rank batch at 2 and 8 GPUs (weak scaling), so "
        "ideal step time is flat. Efficiency is "
        "<span class='mono'>p50_2 / p50_8</span> as a percentage.</p>"
        "<div class='tablewrap'><table class='scaling'>"
        "<thead><tr><th>strategy</th><th class='num'>2 GPU p50</th>"
        "<th class='num'>8 GPU p50</th><th class='num'>slowdown</th>"
        "<th class='num'>efficiency</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def build_category_summary(
    latest_entries: list[dict[str, Any]],
) -> str:
    """Four category tiles summarising worst verdict per use-case area."""
    meta = load_dashboard_metadata()
    categories = meta.get("categories") or {}
    if not categories:
        return ""

    by_entry: dict[str, list[dict[str, Any]]] = {}
    for e in latest_entries:
        by_entry.setdefault(str(e.get("entry")), []).append(e)

    tiles = []
    for cat_id, cat in categories.items():
        label = str(cat.get("label") or cat_id)
        workloads = cat.get("workloads") or []
        group: list[dict[str, Any]] = []
        for wl in workloads:
            group.extend(by_entry.get(str(wl), []))
        if not group:
            continue
        counts = _tally(group)
        worst = _worst_verdict(counts)
        color = _VERDICT_COLOR.get(worst, _UNKNOWN_COLOR)
        glyph = _VERDICT_GLYPH.get(worst, _UNKNOWN_GLYPH)
        tally_txt = _tally_text(counts)
        present = [str(wl) for wl in workloads if by_entry.get(str(wl))]
        anchor = f"wl-{_esc(present[0])}" if present else ""
        tiles.append(
            f"<a class='cat-tile' href='#{anchor}' style='border-left-color:{color}'>"
            f"<div class='cat-k'>{_esc(label)}</div>"
            f"<div class='cat-v'>{glyph} {_esc(worst or '—')}</div>"
            f"<div class='cat-sub muted'>{_esc(tally_txt)}</div>"
            f"</a>"
        )
    if not tiles:
        return ""
    return (
        "<h2>Category health</h2>"
        "<p class='muted'>Worst verdict in each area from tonight's run. "
        "Click a tile to jump to workloads.</p>"
        f"<div class='cat-grid'>{''.join(tiles)}</div>"
    )


def _short_digest(base_image: Any) -> str:
    """``repo:tag@sha256:abcdef123456...`` -> ``sha256:abcdef123456`` for display.

    Truncated to 12 hex characters, which is enough to identify a digest in a
    table without making the column unreadable.

    Renders the whole value when it carries no digest, so a malformed or
    tag-only entry stays visible rather than silently becoming an em dash --
    the digest is the entire point of the canary row.
    """
    text = str(base_image or "").strip()
    if not text:
        return "—"
    _, _, digest = text.rpartition("@")
    if not digest:
        return text
    algo, _, hexpart = digest.partition(":")
    return f"{algo}:{hexpart[:12]}" if hexpart else digest


def build_canary_section(canary_results: list[dict[str, Any]]) -> str:
    """The observed-only latest-ROCm canary lane (issue #382).

    Rendered **neutral on purpose**: no verdict chips, no status colours, no
    contribution to the page banner or ``status.json``. This lane follows a
    moving tag, so a red row means "this ROCm release did something", not "we
    regressed" -- health-colouring it would recreate exactly the ambiguity #382
    exists to avoid, and is the trap #368's survey tab fell into (KB: a neutral
    view must sweep *every* health-signalling render path, not just the obvious
    one).

    Always renders a section, including an explicit empty state, so the
    ``#canary`` anchor resolves before the lane's first successful run instead
    of 404ing a promised route (KB#11b).
    """
    rows = []
    for doc in canary_results[-_CANARY_ROWS:]:
        build = doc.get("build") or {}
        summary = doc.get("summary") or {}
        graded = f"{_count(summary.get('pass'))}/{_count(summary.get('pass')) + _count(summary.get('fail'))}"
        # The workflow synthesises a row when setup died before the evaluator
        # ran, and puts the only explanation in top-level `error`. Without
        # rendering it such a row is an unexplained line of em dashes and 0/0 --
        # it would say a :latest was broken without saying how, which is most of
        # the value. Plain text, no health class: this lane stays neutral (a red
        # canary is a question about a new ROCm release, not a regression here).
        note = str(doc.get("error") or "").strip()
        rows.append(
            "<tr>"
            f"<td>{_esc(str(doc.get('generated_at') or '—')[:10])}</td>"
            f"<td>{_esc(str(build.get('rocm') or '—'))}</td>"
            f"<td>{_esc(str(build.get('torch') or '—'))}</td>"
            f"<td>{_esc(str(build.get('hip') or '—'))}</td>"
            f"<td><code>{_esc(_short_digest(build.get('base_image')))}</code></td>"
            f"<td>{_esc(graded)}</td>"
            f"<td>{_esc(note) if note else '—'}</td>"
            "</tr>"
        )

    if rows:
        body = (
            "<table><thead><tr><th>date</th><th>ROCm</th><th>torch</th>"
            "<th>HIP</th><th>base image</th><th>passed/graded</th>"
            "<th>note</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    else:
        body = (
            '<p class="muted">No canary runs recorded yet. The lane is scheduled '
            "daily and best-effort, so it may also be skipped when the shared "
            "GPU runner is busy.</p>"
        )

    return (
        '<section class="dash-section" id="canary">'
        '<h2>Latest ROCm canary <span class="muted">· observed only</span></h2>'
        '<p class="muted">Tracks <code>rocm/pytorch:latest</code>, resolved to a '
        "concrete digest per run. <strong>Not a gate:</strong> these rows never "
        "affect the status above, the pass-rate trend, or any required check -- "
        "the merge gate stays on the digest pinned in "
        "<code>docker/Dockerfile.ci-gpu</code>. A change here says a new ROCm "
        "release moved something, which is a question to investigate, not a "
        "regression on this branch.</p>"
        f"{body}"
        "</section>"
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


def _sparkline_a11y_label(metric: str, values: list[float | None]) -> str:
    """Plain-language history for screen readers (oldest → newest)."""
    pts = [v for v in values if _isnum(v)]
    if len(pts) < 2:
        return f"{metric} history: not enough data"
    rendered = [_fmt_ms(v) if "step time" in metric.lower() else _fmt_num(v) for v in pts]
    return f"{metric} history, oldest to newest: {', '.join(rendered)}"


def _svg_sparkline(
    values: list[float | None],
    width: int = 160,
    height: int = 32,
    *,
    label: str | None = None,
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
    svg = (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" style="vertical-align:middle">'
        f'<polyline fill="none" stroke="#539bf5" stroke-width="1.5" '
        f'stroke-linejoin="round" stroke-linecap="round" points="{poly}"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="2" fill="#539bf5"/>'
        f"</svg>"
    )
    if label:
        return (
            f'<span class="spark-a11y" role="img" aria-label="{_esc(label)}">{svg}</span>'
        )
    return svg


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

    A summary whose counts do not account for its own total is holding at least
    one verdict this generator does not know, and cannot be called ``passing``
    on the strength of the verdicts it does know.
    """
    if not results:
        return "unknown", "#57606a"
    s = results[-1].get("summary", {}) or {}
    if _count(s.get("fail")):
        return "failing", _VERDICT_COLOR["fail"]
    total = _count(s.get("total"))
    if total == 0:
        return "empty", "#57606a"
    if total > sum(_count(s.get(v)) for v in _VERDICT_ORDER):
        return "unrecognised", _UNKNOWN_COLOR
    if _count(s.get("pass")):
        # A mix of pass and skip is not a fully healthy run — some workloads
        # never executed (e.g. not enough GPUs for the 8-GPU matrix entries).
        if _count(s.get("skip")):
            return "partial", _VERDICT_COLOR["record"]
        return "passing", _CHIP_VERDICT_COLOR["pass"]
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
        if m in _ENGINEER_HIDDEN_METRICS:
            continue
        raw = summary.get(m)
        unit = _METRIC_UNITS.get(m, "")
        val = f"{_fmt_num(raw)} {unit}".strip() if _isnum(raw) else _fmt_num(raw)
        trend = (
            f"<td class='spark'>{_svg_sparkline(mhist.get((cell_key, m), []))}</td>"
            if show_trend else ""
        )
        out.append(
            f"<tr><td class='mono'>{_esc(m)}</td>"
            f"<td class='center muted'>{_esc(_grading_rule_label(entry, m))}</td>"
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


def _worst_verdict(counts: dict[str, int]) -> str:
    """The verdict a group should be judged by.

    Known failures rank first. Anything this generator does not recognise ranks
    next -- ahead of record, skip and pass -- because the results JSON is not
    ours to constrain, and a group holding an unrecognised verdict must not be
    able to render as healthy. Scanning ``_VERDICT_ORDER`` alone would count an
    unknown towards the group size while never selecting it, so ``pass`` plus
    ``error`` would show a green tick over a 1/2 ratio.
    """
    if counts.get("fail"):
        return "fail"
    unknown = sorted(v for v, n in counts.items() if n and v not in _VERDICT_ORDER)
    if unknown:
        return unknown[0]
    return next((v for v in _VERDICT_ORDER if counts.get(v)), "")


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

    active = set(newest)

    def _wl_chip(name: str, group: list[dict[str, Any]], *, retired: bool) -> str:
        if not group:
            return (
                f"<span class='wl-chip absent' role='img' "
                f"title='{_esc(name)} — not in this run' "
                f"aria-label='{_esc(name)} — not in this run'>"
                f"<span class='chip-label'>{_esc(str(_workload_meta(name).get('title') or name))}</span>"
                f"<span class='chip-meta'>·</span></span>"
            )
        counts = _tally(group)
        worst = _worst_verdict(counts)
        n = len(group)
        worst_n = counts.get(worst, 0)
        count = str(n) if worst_n == n else f"{worst_n}/{n}"
        bg = _CHIP_VERDICT_COLOR.get(worst, _UNKNOWN_COLOR)
        breakdown = f"{name} — {_tally_text(counts)}"
        dot = (
            f"<span class='dot sm' role='img' style='background:{bg}' "
            f"aria-label='{_esc(breakdown)}'>"
            f"{_esc(_VERDICT_GLYPH.get(worst, _UNKNOWN_GLYPH))} {_esc(count)}</span>"
        )
        title = str(_workload_meta(name).get("title") or name)
        inner = (
            f"{dot}<span class='chip-label'>{_esc(title)}</span>"
        )
        if retired:
            return (
                f"<span class='wl-chip retired' title='{_esc(breakdown)} (retired workload)'>"
                f"{inner}</span>"
            )
        return (
            f"<a class='wl-chip' href='#wl-{_esc(name)}' "
            f"title='{_esc(breakdown)} — jump to workload'>{inner}</a>"
        )

    newest_first = list(reversed(runs))
    cards = []
    for i, doc in enumerate(newest_first):
        build = doc.get("build") or {}
        by_entry: dict[str, list[dict[str, Any]]] = {}
        for e in doc.get("entries") or []:
            by_entry.setdefault(str(e.get("entry")), []).append(e)

        status, color = _latest_status([doc])
        customer_status = _customer_status_label(status)
        date = _fmt_timestamp(str(doc.get("generated_at") or "")).split(" ")[0] or "—"
        run_id = str(build.get("upstream_run_id") or "")
        when = (
            f"<a href='https://github.com/{_REPO}/actions/runs/{_esc(run_id)}'>{_esc(date)}</a>"
            if run_id else _esc(date)
        )

        older = newest_first[i + 1] if i + 1 < len(newest_first) else prior
        bumped = older is not None and any(
            str((older.get("build") or {}).get(f) or "") != str(build.get(f) or "")
            for f, _ in _TOOLCHAIN_FIELDS
        )
        flag = " <span class='bump' title='toolchain changed in this run'>bump</span>" if bumped else ""

        chip_items = [
            f"<li>{_wl_chip(name, by_entry.get(name) or [], retired=name not in active)}</li>"
            for name in ordered
        ]
        cards.append(
            f"<article class='release-card'>"
            f"<header class='release-hdr'>"
            f"<h3 class='release-date'>{when}{flag}</h3>"
            f"<span class='badge sm' style='background:{color}' "
            f"title='{_esc(status)}'>{_esc(customer_status)}</span>"
            f"<span class='runmeta mono'>{_esc(str(build.get('amd_aorta_version') or ''))}</span>"
            f"</header>"
            f"<ul class='wl-chips'>{''.join(chip_items)}</ul>"
            f"</article>"
        )

    return (
        "<section class='dash-section' id='release-history'>"
        "<h2>Nightly release health</h2>"
        "<p class='muted'>Each card is one nightly release (newest first). "
        "<strong>Overall health</strong> summarises that night's graded verdicts; "
        "workload chips wrap instead of scrolling sideways. Active workloads link "
        "to reproduction steps below. Retired workloads (no longer in the latest "
        "matrix) are plain text. Glyphs: ✓ pass · ✗ fail · ◆ record · ○ skip.</p>"
        f"<div class='release-list'>{''.join(cards)}</div>"
        "</section>"
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

    movers_corr: list[tuple] = []
    movers_perf: list[tuple] = []
    movers_other: list[tuple] = []
    for k in sorted(set(prev_by) & set(latest_by)):
        before_m = _measurements(prev_by[k])
        after_m = _measurements(latest_by[k])
        for metric in sorted(set(before_m) & set(after_m)):
            b, a = before_m[metric], after_m[metric]
            if b == a:
                continue
            if _is_correctness_change_metric(metric):
                movers_corr.append((0, 0, k, metric, b, a, latest_by[k]))
                continue
            if not b:  # a move from zero has no meaningful percentage
                continue
            pct = (a - b) / abs(b) * 100.0
            if abs(pct) > _MOVE_PCT:
                row = (abs(pct), pct, k, metric, b, a, latest_by[k])
                if _is_dashboard_performance_metric(metric):
                    movers_perf.append(row)
                else:
                    movers_other.append(row)
    movers_corr.sort(key=lambda r: (r[5], r[4]), reverse=True)
    movers_perf.sort(reverse=True)
    movers_other.sort(reverse=True)

    def _corr_mover_li(row: tuple) -> str:
        _, _, _k, metric, b, a, entry = row
        unit = _METRIC_UNITS.get(metric, "")
        return (
            f"<li class='corr'><span class='mono'>{_esc(_cell_label(entry))}</span> "
            f"<span class='mono'>{_esc(metric)}</span> "
            f"<span class='muted'>{_esc(_fmt_num(b))} → {_esc(_fmt_num(a))}"
            f"{(' ' + _esc(unit)) if unit else ''}</span></li>"
        )

    def _mover_li(row: tuple, css: str) -> str:
        _, pct, _k, metric, b, a, entry = row
        unit = _METRIC_UNITS.get(metric, "")
        arrow = "▲" if pct > 0 else "▼"
        return (
            f"<li class='{css}'><span class='mono'>{_esc(_cell_label(entry))}</span> "
            f"<span class='mono'>{_esc(metric)}</span> {arrow} {pct:+.1f}% "
            f"<span class='muted'>{_esc(_fmt_num(b))} → {_esc(_fmt_num(a))}"
            f"{(' ' + _esc(unit)) if unit else ''}</span></li>"
        )

    if movers_corr:
        items.append("<li class='change-hdr corr'>Correctness</li>")
        for row in movers_corr[:4]:
            items.append(_corr_mover_li(row))
    if movers_perf:
        items.append("<li class='change-hdr perf'>Performance</li>")
        for row in movers_perf[:4]:
            items.append(_mover_li(row, "perf"))
    for row in movers_other[:2]:
        items.append(_mover_li(row, ""))

    shown_corr = min(len(movers_corr), 4)
    shown_perf = min(len(movers_perf), 4)
    shown_other = min(len(movers_other), 2)
    extra_corr = len(movers_corr) - shown_corr
    extra_thresholded = (len(movers_perf) - shown_perf) + (len(movers_other) - shown_other)

    since = _fmt_timestamp(str(prev.get("generated_at") or "")).split(" ")[0]
    if not items:
        body = (
            "<p class='steady'>Nothing moved: same workloads, same verdicts, and no "
            f"metric changed by more than {_MOVE_PCT:.0f}%.</p>"
        )
    else:
        more_parts: list[str] = []
        if extra_corr:
            more_parts.append(
                f"and {extra_corr} more correctness change"
                f"{'' if extra_corr == 1 else 's'}"
            )
        if extra_thresholded:
            more_parts.append(
                f"and {extra_thresholded} more metric"
                f"{'' if extra_thresholded == 1 else 's'} past {_MOVE_PCT:.0f}%"
            )
        more = (
            "".join(f"<p class='muted'>{p}</p>" for p in more_parts)
            if more_parts else ""
        )
        body = f"<ul class='changes'>{''.join(items)}</ul>{more}"

    return (
        f"<h2>What changed since {_esc(since or 'the previous run')}</h2>{body}"
    )


def build_workload_cards(
    *,
    groups: dict[str, list[str]],
    latest_by_key: dict[str, dict[str, Any]],
    history: dict[str, list[float | None]],
    mhist: dict[tuple[str, str], list[float | None]],
    show_trend: bool,
    show_metric_trend: bool,
    show_notes: bool,
    build: dict[str, Any],
) -> str:
    """Category-grouped workload cards (replaces the wide results table)."""
    meta = load_dashboard_metadata()
    categories = meta.get("categories") or {}
    ordered_entries: list[str] = []
    seen: set[str] = set()
    for cat in categories.values():
        for wl in cat.get("workloads") or []:
            name = str(wl)
            if name in groups and name not in seen:
                ordered_entries.append(name)
                seen.add(name)
    for name in sorted(groups):
        if name not in seen:
            ordered_entries.append(name)

    def _workload_card(entry_name: str) -> str:
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
        meta_line = f"{n} result{'' if n == 1 else 's'} · {tally_txt}"
        durs = {
            d for d in (latest_by_key[k].get("duration_sec") for k in cell_keys)
            if _isnum(d)
        }
        if len(durs) == 1 and max(durs) > 0:
            meta_line += f" · workload run {max(durs):,.0f}s"

        wl_meta = _workload_meta(entry_name)
        wl_title = str(wl_meta.get("title") or entry_name)
        wl_summary = str(wl_meta.get("summary") or "")
        summary_html = (
            f"<p class='wl-sum muted'>{_esc(wl_summary)}</p>" if wl_summary else ""
        )

        variants: list[str] = []
        for k in cell_keys:
            e = latest_by_key[k]
            verdict = e.get("verdict", "skip")
            color = _CHIP_VERDICT_COLOR.get(verdict, _VERDICT_COLOR.get(verdict, "#57606a"))
            st = (e.get("metrics") or {}).get("mean_step_time_ms")
            cell_name = e.get("cell")
            label = (
                f"<span class='mono'>{_esc(str(cell_name))}</span>"
                if cell_name else "<span class='muted'>whole workload</span>"
            )
            spark_vals = history.get(k, [])
            trend_html = (
                f"<span class='variant-spark'>{_svg_sparkline(spark_vals, label=_sparkline_a11y_label('mean step time', spark_vals))}</span>"
                if show_trend else ""
            )
            notes_html = (
                f"<p class='variant-note muted'>{_esc('; '.join(e.get('reasons') or []))}</p>"
                if show_notes else ""
            )
            headlines = _headline_block(e, entry_name)
            headline_html = headlines or ""
            mrows = _metric_rows(k, e, mhist, show_metric_trend)
            metrics_html = ""
            if mrows:
                visible = [
                    m for m in ((e.get("metrics") or {}).get("summary") or {})
                    if m not in _ENGINEER_HIDDEN_METRICS
                ]
                n_metrics = len(visible)
                recipe = e.get("recipe") or ""
                prov = []
                if recipe:
                    prov.append(f"recipe <span class='mono'>{_esc(str(recipe))}</span>")
                trials = e.get("trials")
                if _isnum(trials):
                    n_trials = int(trials)
                    prov.append(f"{n_trials} trial{'s' if n_trials != 1 else ''}")
                metrics_html = (
                    f"<details class='variant-metrics'>"
                    f"<summary>Detailed metrics (optional) — {n_metrics} value"
                    f"{'s' if n_metrics != 1 else ''}</summary>"
                    f"<p class='prov muted'>Additional summary metrics for this recipe variant. "
                    f"<strong>Grading rule</strong> is how nightly CI compares the value; "
                    f"<strong>this run</strong> is tonight; "
                    f"<strong>history</strong> is recent nights (when available).</p>"
                    f"<p class='prov'>{' · '.join(prov)}</p>"
                    f"<div class='tablewrap'><table class='inner'><thead><tr><th>metric</th>"
                    f"<th class='center'>grading rule</th><th class='num'>this run</th>"
                    f"{'<th>history</th>' if show_metric_trend else ''}</tr></thead>"
                    f"<tbody>{mrows}</tbody></table></div>"
                    f"</details>"
                )
            variants.append(
                f"<div class='variant'>"
                f"<div class='variant-hdr'>"
                f"<span class='variant-name'>{label}</span>"
                f"<span class='badge' style='background:{color}'>{_esc(verdict)}</span>"
                f"<span class='num variant-step'>{_esc(_fmt_ms(st))}{_bar(st, group_max)}</span>"
                f"{trend_html}"
                f"</div>"
                f"{headline_html}{notes_html}{metrics_html}"
                f"</div>"
            )

        return (
            f"<article class='wl-card' id='wl-{_esc(entry_name)}'>"
            f"<header class='wl-card-hdr'>"
            f"<div class='wl-head'>"
            f"<h4 class='wl-title'>"
            f"<a href='#repro-{_esc(entry_name)}'>{_esc(wl_title)}</a>"
            f"<span class='mono muted wl-id'>{_esc(entry_name)}</span>"
            f"</h4></div>"
            f"{summary_html}"
            f"<p class='muted wl-meta'>{_esc(meta_line)}</p>"
            f"{_run_command_block(entry_name, build)}"
            f"</header>"
            f"<div class='variant-list'>{''.join(variants)}</div>"
            f"</article>"
        )

    blocks: list[str] = []
    for cat_id, cat in categories.items():
        cat_workloads = [str(w) for w in (cat.get("workloads") or []) if str(w) in groups]
        if not cat_workloads:
            continue
        cards = "".join(_workload_card(w) for w in cat_workloads)
        label = str(cat.get("label") or cat_id)
        summary = str(cat.get("summary") or "")
        sum_html = f"<p class='cat-block-sum muted'>{_esc(summary)}</p>" if summary else ""
        blocks.append(
            f"<div class='cat-block' id='cat-{_esc(cat_id)}'>"
            f"<h3 class='cat-block-head'>{_esc(label)}</h3>"
            f"{sum_html}"
            f"<div class='wl-grid'>{cards}</div>"
            f"</div>"
        )

    orphan = [n for n in ordered_entries if n not in {
        str(w) for c in categories.values() for w in (c.get("workloads") or [])
    }]
    if orphan:
        cards = "".join(_workload_card(w) for w in orphan)
        blocks.append(
            f"<div class='cat-block' id='cat-other'>"
            f"<h3 class='cat-block-head'>Other</h3>"
            f"<div class='wl-grid'>{cards}</div></div>"
        )

    if not blocks:
        return "<p class='muted'>no results yet</p>"
    return "".join(blocks)


def build_dashboard_html(
    results: list[dict[str, Any]],
    canary_results: list[dict[str, Any]] | None = None,
) -> str:
    """Render the full dashboard HTML from the results history (pure).

    ``canary_results`` is the observed-only latest-ROCm lane (issue #382). It is
    a separate argument rather than merged into ``results`` precisely so it
    cannot reach the status banner, the pass-rate trend, the history grid or
    ``build_status_json`` -- everything gated is computed from ``results`` alone.

    It defaults to ``None``, and ``None`` / ``[]`` / omitted all render
    identically. That is NOT byte-identical to the pre-#382 page: the canary
    section is always emitted, showing its "no runs recorded yet" state when
    there is nothing to show, so the ``#canary`` anchor resolves before the
    lane's first run instead of 404ing a promised route. What is unchanged is
    everything gated -- same banner, same trend, same history grid, same
    ``status.json`` and ``data.json``.
    """
    status, status_color = _latest_status(results)
    customer_status = _customer_status_label(status)
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
            if m in _ENGINEER_HIDDEN_METRICS:
                continue
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

    # Group cells under their workload so the sweep matrix stays visible.
    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(str(latest_by_key[k].get("entry")), []).append(k)

    workloads_html = build_workload_cards(
        groups=groups,
        latest_by_key=latest_by_key,
        history=history,
        mhist=mhist,
        show_trend=show_trend,
        show_metric_trend=show_metric_trend,
        show_notes=show_notes,
        build=build,
    )

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
            f"to become the reference. Pass/fail grading starts after the first "
            f"blessed baseline PR.{extra}</div>"
        )
    elif nothing_graded:
        notices.append(
            f"<div class='notice'>Nothing ran: all {total} results were "
            f"<strong>skipped</strong>, so this build establishes nothing. Check that "
            f"the runner exposes as many GPUs as the matrix asks for.</div>"
        )
    elif status == "partial":
        pass_now = _count(s.get("pass"))
        notices.append(
            f"<div class='notice'>This run is <strong>partial</strong>: "
            f"{pass_now} of {total} results passed but {skip_now} workload(s) were "
            f"<strong>skipped</strong> — not every configured matrix entry ran. "
            f"Check that the runner exposes as many GPUs as the matrix asks for.</div>"
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

    fail_now = _count(s.get("fail"))
    action_panel = ""
    if fail_now:
        action_panel = (
            "<div class='notice fail-actions'>"
            "<strong>Next steps</strong> "
            "Capture your environment with "
            "<code class='mono'>aorta env probe -o env.json</code>, "
            "review the failing workload below, and "
            f"<a href='https://github.com/{_REPO}/issues/new/choose'>"
            "open a ROCm issue</a> with the stack versions above."
            "</div>"
        )

    scope_notice = (
        "<div class='notice scope-notice muted-notice'>"
        "<strong>Reference hardware:</strong> single-node MI350 runner with pinned "
        "ROCm + PyTorch wheels. Numbers are nightly CI references — not a guarantee "
        "on other AMD GPUs or cluster configurations."
        "</div>"
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
    category_html = build_category_summary(latest_entries)
    scaling_html = build_scaling_summary(latest_entries)
    canary_html = build_canary_section(canary_results or [])

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ROCm stack health · AORTA nightly</title>
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
  .dash-section {{ margin-top:2rem; }}
  .dash-section:first-of-type {{ margin-top:1.5rem; }}
  .release-list {{ display:flex; flex-direction:column; gap:.65rem; margin-top:.5rem; }}
  .release-card {{ background:var(--panel); border:1px solid var(--border);
                   border-radius:8px; padding:.65rem .85rem; }}
  .release-hdr {{ display:flex; flex-wrap:wrap; align-items:center; gap:.45rem .75rem;
                  margin-bottom:.55rem; }}
  .release-date {{ font-weight:600; font-size:.88rem; margin:0; }}
  .wl-chips {{ display:flex; flex-wrap:wrap; gap:.4rem; list-style:none;
                margin:0; padding:0; }}
  .wl-chips > li {{ margin:0; padding:0; }}
  .wl-chip {{ display:inline-flex; align-items:center; gap:.35rem; padding:.25rem .55rem;
              border:1px solid var(--border); border-radius:999px; font-size:.75rem;
              text-decoration:none; color:inherit; background:#12161b; max-width:100%; }}
  a.wl-chip:hover {{ border-color:var(--accent); }}
  .wl-chip.retired {{ cursor:default; border-color:#30363d; }}
  .wl-chip.retired .chip-label {{ color:var(--muted); }}
  .wl-chip.absent {{ cursor:default; }}
  .wl-chip.absent .chip-meta {{ color:var(--muted); }}
  .wl-chip .chip-label {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
                          max-width:18ch; }}
  .wl-chip .chip-meta {{ color:var(--muted); }}
  .dot.sm {{ min-width:38px; font-size:.68rem; padding:1px 5px; }}
  .cat-block {{ margin-top:1.5rem; }}
  .cat-block-head {{ font-size:.95rem; margin:0 0 .25rem; color:#e6edf3; }}
  .cat-block-sum {{ margin:0 0 .55rem; max-width:72ch; }}
  .wl-grid {{ display:grid; gap:.75rem;
              grid-template-columns:repeat(auto-fit, minmax(min(100%, 340px), 1fr)); }}
  .wl-card {{ background:var(--panel); border:1px solid var(--border); border-radius:8px;
              overflow:visible; }}
  .wl-card-hdr {{ padding:.75rem .85rem .55rem; border-bottom:1px solid var(--border); }}
  .wl-meta {{ margin:.25rem 0 0; font-size:.78rem; }}
  .variant-list {{ padding:.55rem .85rem .75rem; display:flex; flex-direction:column;
                    gap:.65rem; }}
  .variant {{ border:1px solid var(--border); border-radius:6px; padding:.5rem .65rem;
              background:#12161b; }}
  .variant-hdr {{ display:flex; flex-wrap:wrap; align-items:center; gap:.45rem .65rem; }}
  .variant-name {{ flex:1 1 120px; min-width:0; }}
  .variant-step {{ margin-left:auto; }}
  .variant-spark svg {{ display:block; }}
  .variant-note {{ margin:.35rem 0 0; font-size:.78rem; }}
  .variant-metrics {{ margin-top:.45rem; }}
  .variant-metrics summary {{ cursor:pointer; color:var(--muted); font-size:.78rem; }}
  ul.headlines {{ margin-top:.35rem; }}
  .repro-sec {{ margin:.65rem 0 0; }}
  .repro-sec h5 {{ margin:0 0 .35rem; font-size:.78rem; color:#e6edf3;
                   text-transform:uppercase; letter-spacing:.04em; }}
  ul.repro-list {{ margin:.2rem 0 0; padding-left:1.1rem; font-size:.82rem; }}
  ul.repro-list li {{ margin:.25rem 0; }}
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
  .secthead {{ display:flex; align-items:baseline; justify-content:space-between;
               gap:1rem; flex-wrap:wrap; }}
  .toolbar button {{ background:var(--panel); color:var(--fg);
                     border:1px solid var(--border); border-radius:6px;
                     padding:.25rem .6rem; font-size:.75rem; cursor:pointer; }}
  .toolbar button:hover {{ border-color:var(--accent); color:var(--accent); }}
  .cat-grid {{ display:grid; gap:.65rem; margin:.5rem 0 0;
               grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); }}
  .cat-tile {{ display:block; background:var(--panel); border:1px solid var(--border);
               border-left:3px solid var(--border); border-radius:8px;
               padding:.65rem .85rem; text-decoration:none; color:inherit; }}
  .cat-tile:hover {{ border-color:var(--accent); }}
  .cat-k {{ font-size:.72rem; color:var(--muted); text-transform:uppercase;
            letter-spacing:.06em; }}
  .cat-v {{ font-size:1rem; font-weight:600; margin:.2rem 0; }}
  .cat-sub {{ font-size:.75rem; }}
  .scope-notice {{ margin-top:.65rem; }}
  .fail-actions {{ border-left-color:#f85149; }}
  .wl-head {{ display:flex; align-items:baseline; gap:.6rem; flex-wrap:wrap; }}
  .wl-head strong {{ color:#e6edf3; font-size:.95rem; }}
  .wl-id {{ font-size:.72rem; }}
  .wl-sum {{ margin:.25rem 0 .35rem; font-size:.82rem; }}
  ul.headlines {{ list-style:none; margin:.35rem 0 0; padding:0;
                  display:flex; flex-wrap:wrap; gap:.35rem .75rem; }}
  ul.headlines li {{ background:#12161b; border:1px solid var(--border);
                     border-radius:6px; padding:.2rem .55rem; font-size:.78rem; }}
  tr.headline-row > td {{ padding-top:0; padding-bottom:.5rem; }}
  .repro-panel {{ margin:.55rem 0 0; border:1px solid var(--border); border-radius:8px;
                  background:#12161b; }}
  .repro-panel summary {{ cursor:pointer; padding:.55rem .75rem; font-size:.82rem;
                           font-weight:600; color:var(--accent); list-style-position:outside; }}
  .repro-panel[open] summary {{ border-bottom:1px solid var(--border); }}
  .repro-body {{ padding:.65rem .75rem .75rem; }}
  ol.repro-steps {{ margin:.35rem 0 0; padding-left:1.25rem; font-size:.82rem; }}
  ol.repro-steps li {{ margin:.35rem 0; }}
  .repro-cmd {{ background:#0d1117; border:1px solid var(--border); border-radius:6px;
                padding:.5rem .65rem; margin:.35rem 0 0; overflow-x:auto;
                white-space:pre-wrap; word-break:break-word; }}
  .repro-note {{ margin:.45rem 0 0; font-size:.75rem; }}
  .wl-intro {{ margin:.35rem 0 .5rem; font-size:.86rem; max-width:72ch; }}
  h4.wl-title {{ margin:0; font-size:.92rem; font-weight:600; display:flex;
                 flex-wrap:wrap; align-items:baseline; gap:.35rem .5rem; }}
  h4.wl-title a {{ color:#e6edf3; text-decoration:none; }}
  h4.wl-title a:hover {{ color:var(--accent); text-decoration:underline; }}
  a.gcell-link {{ color:inherit; text-decoration:none; display:inline-block; }}
  a.gcell-link:hover .dot {{ outline:2px solid var(--accent); outline-offset:1px; }}
  table.scaling {{ margin-top:.5rem; }}
  table.grid th.wlcol a {{ color:var(--muted); text-decoration:none; font-size:.62rem; }}
  table.grid th.wlcol a:hover {{ color:var(--accent); text-decoration:underline; }}
  ul.changes li.change-hdr {{ background:transparent; border:none; padding:.15rem 0;
                              font-size:.72rem; text-transform:uppercase;
                              letter-spacing:.06em; color:var(--muted); }}
  ul.changes li.corr {{ border-left-color:#f85149; }}
  ul.changes li.perf {{ border-left-color:#d29922; }}
  .status-sub {{ font-size:.68rem; color:var(--muted); margin-left:.25rem; }}
</style></head>
<body>
  <div class="wrap">
    <header>
      <div class="titlebar">
        <h1>ROCm stack health</h1>
        <span class="status-pill" title="{_esc(status)}">{_esc(customer_status)}</span>
        <span class="status-sub mono">({_esc(status)})</span>
      </div>
      <p class="lede">Every night AORTA installs the freshly built wheel on a
        reference MI350 runner and replays representative PyTorch workloads —
        training, inference, and distributed correctness checks — comparing each
        result against a blessed baseline when one exists.</p>
      <p class="nav"><strong>Stack health</strong> ·
        <a href="docs/">AORTA documentation</a> ·
        <a href="sanitizers/">Sanitizer nightly</a> ·
        <a href="https://github.com/{_REPO}">Repository</a> ·
        <a href="data.json">data.json</a> ·
        <a href="status.json">status.json</a></p>
      <div class="chips">{toolchain}</div>
      <p class="prov-line">{' · '.join(provenance)}</p>
    </header>

    {scope_notice}
    {''.join(notices)}
    {action_panel}

    <section class="dash-section" id="overview">
    <div class="cards">
      {cards_html}
      <div class="card trend"><div class="k">pass-rate trend</div><div class="v">{_svg_sparkline(passrate, width=240)}</div></div>
    </div>
    </section>

    {category_html}

    {changes_html}

    {history_html}

    {scaling_html}

    <section class="dash-section" id="workloads">
    <div class="secthead">
      <h2>Workloads</h2>
      <div class="toolbar" id="toolbar" hidden>
        <button type="button" data-details="open">Expand all</button>
        <button type="button" data-details="close">Collapse all</button>
      </div>
    </div>
    <p class="muted wl-intro">Workloads are grouped by use case below. Each card shows tonight&apos;s recipe variants; expand <strong>Reproduce this workload locally</strong> for a full step-by-step guide, or <strong>Detailed metrics</strong> for additional summary values.</p>
    {workloads_html}
    <p class="legend">Results: <strong>pass</strong>/<strong>fail</strong> = compared against a blessed baseline · <strong>record</strong> = baseline not set yet · <strong>skip</strong> = not enough GPUs. Expand cards for reproduction steps or optional detailed metrics.</p>
    </section>

    {canary_html}
  </div>
<script>
(function () {{
  function openFromHash() {{
    var hash = location.hash;
    if (!hash) return;
    var target = document.querySelector(hash);
    if (!target) return;
    if (target.tagName === "DETAILS") {{
      target.open = true;
      return;
    }}
    if (target.id && target.id.indexOf("wl-") === 0) {{
      var repro = document.getElementById("repro-" + target.id.slice(3));
      if (repro && repro.tagName === "DETAILS") repro.open = true;
    }}
  }}
  openFromHash();
  window.addEventListener("hashchange", openFromHash);
}})();
(function () {{
  var bar = document.getElementById("toolbar");
  if (!bar) return;
  var panels = document.querySelectorAll(".variant-metrics, details.repro-panel");
  if (!panels.length) return;
  bar.hidden = false;
  bar.addEventListener("click", function (ev) {{
    var want = ev.target && ev.target.getAttribute("data-details");
    if (!want) return;
    var open = want === "open";
    for (var i = 0; i < panels.length; i++) panels[i].open = open;
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
    ap.add_argument(
        "--canary-results-dir",
        type=Path,
        default=None,
        help=(
            "observed-only latest-ROCm canary results (issue #382). Optional and "
            "may be absent: the section renders an empty state rather than "
            "failing, since the lane is non-gating and best-effort."
        ),
    )
    args = ap.parse_args()

    results = load_results(args.results_dir)
    if args.max_builds > 0:
        results = results[-args.max_builds:]
    # Absent dir is normal (no canary run yet, or the data branch predates the
    # lane), so this must not fail the gated dashboard build.
    canary_results: list[dict[str, Any]] = []
    if args.canary_results_dir is not None and args.canary_results_dir.is_dir():
        canary_results = load_results(args.canary_results_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "index.html").write_text(
        build_dashboard_html(results, canary_results), encoding="utf-8")
    # data.json stays the GATED history only -- machine consumers treat it as the
    # gate's record, and folding an observed-only lane into it would change that
    # contract silently. The canary's own history lives on the ci-results branch.
    (args.out_dir / "data.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.out_dir / "status.json").write_text(
        json.dumps(build_status_json(results), indent=2), encoding="utf-8")
    print(
        f"dashboard: {len(results)} build(s), {len(canary_results)} canary run(s) "
        f"-> {args.out_dir / 'index.html'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
