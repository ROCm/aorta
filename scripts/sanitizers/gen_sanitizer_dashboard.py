#!/usr/bin/env python3
"""Render a sanitizer-nightly dashboard from ``aorta.sanitizer_report/0.1`` output.

Consumes the three daily recipe reports and the committed verdict baselines and
emits, into ``--out-dir``:

* ``index.html`` -- a self-contained (no CDN / no external JS) status page with two
  tabs (a pure CSS ``:checked`` radio toggle -- switching needs no network):
  **Expected behavior (guardrails)** -- the baseline-checked regression gate:
  baseline-health banner, latest per-recipe table, **per-kernel detail** (code
  object / SHA-256 / dispatch count / observed sanitizer verdict / finding count),
  and a cross-run history/trend table; and **Workload survey (observed-only)** --
  kernels drawn from multiple workloads (including aorta-internal-sourced kernels
  supplied via ``--survey`` and the caller-supplied ConSan cases from
  ``--informational-results-dir``, #347) shown with the same kernel-detail shape
  but **no expected/match column** (a fail / not_checked here is an observation,
  never a regression). On both tabs a per-kernel-row ``Report`` column links the
  case's ``sanitizer_report.json`` (satisfying #367's per-row drill-down) and the
  card footer links its **run area** -- the published case *directory*, which also
  carries the sanitizer logs, the recipe as it ran and a REPRODUCE.md, so the
  reproduce command on the card is actionable (#384). Each case also carries a
  one-line observation summary.
* ``summary.md`` -- a GitHub Actions job-summary fragment (append to
  ``$GITHUB_STEP_SUMMARY``) carrying the same gate, table, and kernel detail.
* ``data.json`` -- the aggregated structure for any richer consumer.
* ``status.json`` -- a copy of the ``--status`` run manifest, when supplied, so
  Pages can distinguish a healthy snapshot from a stale one (a failed nightly).

When ``--status`` points at an unhealthy manifest, an error-colored "stale" banner is
rendered at the top of both ``index.html`` and ``summary.md`` so a failed nightly
never leaves the previous healthy page looking current.

Three input shapes are supported:

* ``--results-dir DIR`` -- a single run laid out as
  ``DIR/{waitcheck,consan-clean,consan-racy}/sanitizer_report.json`` (the same
  layout ``compare_verdict_baselines.py`` consumes). This is the CI shape.
* ``--runs-root DIR`` -- a directory of ``run_*`` folders, each with an
  ``out/<case>/sanitizer_report.json``; used locally to render a history/trend.
* ``--history-root DIR`` -- the PUBLISHED data-branch layout
  ``DIR/<id>/<case>/sanitizer_report.json`` plus a per-run ``DIR/<id>/meta.json``
  (keys: commit, date, gpu, run_url, gate). ``<id>`` is ``<YYYY-MM-DD>-<run_id>``
  (date-sortable, unique), enumerated newest-first and capped by ``--keep N``
  (default 30). This is the shape the nightly publishes and Pages serves under
  ``/sanitizers/``: each run's case dirs are co-located under ``runs/<id>/`` and
  linked from the rendered page, and a ``runs/<id>/index.html`` landing page is
  written for each retained run. Every case dir is completed into a *run area*
  (report + gzipped logs + recipe + inputs + REPRODUCE.md + env.json + its own
  ``index.html``); ``--keep-logs N`` (default 7) bounds how many runs keep the
  bulky parts -- guardrail and survey areas alike -- while reports stay for the
  full ``--keep`` window.

Pure rendering lives in ``build_html`` / ``build_summary_md`` /
``build_run_index_html`` / ``build_case_index_html`` / ``build_reproduce_md`` and
pure aggregation in ``summarize_case`` / ``report_digests`` so they are unit
testable without the FS.

Usage (CI, published-history mode):
    python scripts/sanitizers/gen_sanitizer_dashboard.py \
        --history-root dashboard/runs --keep 30 \
        --baselines recipes/sanitizers/fixtures/expected/verdict_baselines.json \
        --commit "$GITHUB_SHA" --run-label "run $GITHUB_RUN_ID" \
        --status status.json \
        --out-dir dashboard
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
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


# Dashboard stylesheet. Kept as a module-level plain string (not part of the
# f-string body) so CSS braces need no doubling and the design diffs cleanly.
#
# Two colour axes that must not be conflated:
#   identity  -- which sanitizer/stage this is; lives on square icon tiles only
#                (blue = infrastructure, purple = waitcheck, green = consan)
#   verdict   -- what was observed; lives on pills only (.v)
# Green therefore means "consan" on a tile and "pass" on a pill; they never
# share a surface. Baseline status (.pill) is the only *solid* fill, so the
# regression-health signal outranks the observed verdict visually.
_CSS = """
  :root {
    color-scheme: dark;
    --bg:#0B1220; --panel:#111827; --inset:#0E1626; --raised:#131C2E; --sunken:#0A101C;
    --text-1:#F9FAFB; --text-2:#D1D5DB; --text-3:#94A3B8;
    --blue:#3B82F6; --purple:#8B5CF6; --green:#22C55E;
    --v-warn:#F59E0B; --v-fail:#EF4444;
    --solid-ok:#15803D; --solid-bad:#B91C1C;
    --border:rgba(148,163,184,.13); --border-hover:rgba(148,163,184,.26);
    --fs-td:15px; --fs-th:12.5px; --fs-kvk:12.5px; --fs-kvv:14px;
    --fs-obs:14.5px; --fs-cap:14px; --fs-name:16px; --fs-kind:13.5px; --fs-badge:12.5px;
    --radius:12px; --radius-sm:10px;
    --shadow:0 1px 2px rgba(0,0,0,.30), 0 6px 20px rgba(0,0,0,.28);
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  * { box-sizing:border-box; }
  html,body { margin:0; background:var(--bg); color:var(--text-1);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1180px; margin:0 auto; padding:24px; }
  .mono { font-family:var(--mono); }
  a { color:#7CB0F8; text-decoration:none; } a:hover { text-decoration:underline; }

  .navrow { display:flex; align-items:center; justify-content:space-between; gap:12px;
    font-size:13px; margin-bottom:16px; }
  .navrow a { color:var(--text-3); } .navrow a:hover { color:var(--text-2); }

  .topbar { display:flex; align-items:flex-start; justify-content:space-between; gap:20px; margin-bottom:14px; }
  .page-header { display:flex; align-items:center; gap:14px; }
  .brand-tile { width:46px; height:46px; flex:0 0 auto; border-radius:var(--radius);
    display:grid; place-items:center; color:var(--blue);
    background:rgba(59,130,246,.12); border:1px solid rgba(59,130,246,.28); }
  .page-header h1 { font-size:30px; font-weight:600; margin:0; letter-spacing:-.02em; line-height:1.1; }
  .page-header .subtitle { font-size:14px; color:var(--text-2); margin:4px 0 0; }

  .runcard { flex:0 0 auto; width:288px; background:var(--panel);
    border:1px solid var(--border); border-radius:var(--radius-sm); padding:2px 12px; box-shadow:var(--shadow); }
  .runrow { display:flex; align-items:baseline; justify-content:space-between; gap:12px;
    padding:5px 0; border-bottom:1px solid var(--border); }
  .runrow:last-child { border-bottom:none; }
  .runrow .k { font-size:10px; text-transform:uppercase; letter-spacing:.06em;
    color:var(--text-3); font-weight:600; white-space:nowrap; }
  /* .rv, not .v -- .v is the verdict badge and would draw a pill outline here. */
  .runrow .rv { font-family:var(--mono); font-size:11.5px; color:var(--text-2);
    text-align:right; word-break:break-all; }

  .toolgrid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .feature-card { background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
    padding:16px; box-shadow:var(--shadow); display:flex; gap:14px; align-items:flex-start;
    transition:border-color .16s ease, transform .16s ease; }
  .feature-card:hover { border-color:var(--border-hover); transform:translateY(-1px); }
  .feature-card .tile { width:44px; height:44px; border-radius:var(--radius-sm); }
  .feature-card .title-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:6px; }
  .feature-card h3 { font-size:var(--fs-name); font-weight:700; margin:0; letter-spacing:-.01em; }
  .feature-card p { font-size:14px; color:var(--text-2); margin:0; line-height:1.5; }
  .feature-card code { font-family:var(--mono); font-size:13.5px; color:var(--text-1); }
  .badge { font-size:12px; font-weight:600; padding:2px 9px; border-radius:999px; border:1px solid transparent; }
  .badge.purple { color:#C4B5FD; background:rgba(139,92,246,.14); border-color:rgba(139,92,246,.30); }
  .badge.green  { color:#86EFAC; background:rgba(34,197,94,.14);  border-color:rgba(34,197,94,.30); }

  .tile { width:34px; height:34px; flex:0 0 auto; border-radius:9px; display:grid; place-items:center;
    background:rgba(148,163,184,.07); border:1px solid var(--border); }
  .tile.blue   { color:var(--blue);   border-color:rgba(59,130,246,.32); background:rgba(59,130,246,.09); }
  .tile.purple { color:var(--purple); border-color:rgba(139,92,246,.32); background:rgba(139,92,246,.09); }
  .tile.green  { color:var(--green);  border-color:rgba(34,197,94,.32);  background:rgba(34,197,94,.09); }

  /* Self-contained tabs (no external JS/CSS): radios toggle sibling panels via :checked. */
  .tabradio { position:absolute; opacity:0; pointer-events:none; }
  .tabbar { display:flex; gap:4px; border-bottom:1px solid var(--border); margin:18px 0 16px; }
  .tabbar label { cursor:pointer; padding:10px 18px; font-size:14px; font-weight:600; color:var(--text-3);
    border:1px solid transparent; border-bottom:none; border-radius:10px 10px 0 0; margin-bottom:-1px;
    transition:color .15s ease, background .15s ease; }
  .tabbar label:hover { color:var(--text-2); }
  #tab-guardrails:checked ~ .tabbar label[for="tab-guardrails"],
  #tab-survey:checked ~ .tabbar label[for="tab-survey"] {
    color:var(--text-1); background:var(--panel);
    border-color:var(--border); border-bottom:1px solid var(--panel); }
  /* The radios are visually hidden (opacity:0) so they show no native focus ring;
     mirror :focus-visible onto the label so keyboard users can see the focused tab. */
  #tab-guardrails:focus-visible ~ .tabbar label[for="tab-guardrails"],
  #tab-survey:focus-visible ~ .tabbar label[for="tab-survey"] {
    outline:2px solid var(--blue); outline-offset:-2px; }
  .tabpanel { display:none; }
  #tab-guardrails:checked ~ #panel-guardrails { display:block; }
  #tab-survey:checked ~ #panel-survey { display:block; }

  .gate { display:flex; align-items:center; gap:13px; border-radius:var(--radius);
    padding:14px 16px; margin-bottom:12px; border:1px solid; border-left-width:3px; }
  .gate .gate-icon { width:30px; height:30px; border-radius:50%; display:grid; place-items:center; flex:0 0 auto; }
  .gate .gate-text { font-size:14.5px; color:var(--text-2); }
  .gate .gate-text strong { display:block; font-size:15.5px; font-weight:700; letter-spacing:.02em; margin-bottom:1px; }
  .gate.ok  { background:rgba(34,197,94,.08); border-color:rgba(34,197,94,.26); border-left-color:var(--green); }
  .gate.ok .gate-icon { background:rgba(34,197,94,.16); color:var(--green); }
  .gate.ok .gate-text strong { color:#86EFAC; }
  .gate.bad { background:rgba(239,68,68,.08); border-color:rgba(239,68,68,.26); border-left-color:var(--v-fail); }
  .gate.bad .gate-icon { background:rgba(239,68,68,.16); color:var(--v-fail); }
  .gate.bad .gate-text strong { color:#FCA5A5; }

  .info-banner { display:flex; gap:12px; align-items:flex-start; background:rgba(59,130,246,.07);
    border:1px solid rgba(59,130,246,.26); border-left:3px solid var(--blue);
    border-radius:var(--radius); padding:13px 16px; margin-bottom:12px; }
  .info-banner > svg { flex:0 0 auto; color:var(--blue); margin-top:2px; }
  .info-banner p { margin:0; font-size:14px; color:var(--text-2); line-height:1.5; }
  .info-banner strong { color:var(--text-1); font-weight:600; }

  .stale { display:flex; gap:12px; align-items:flex-start; background:rgba(239,68,68,.10);
    border:1px solid rgba(239,68,68,.30); border-left:3px solid var(--v-fail);
    border-radius:var(--radius); padding:13px 16px; margin-bottom:12px;
    font-size:14px; color:var(--text-2); }
  .stale strong { color:#FCA5A5; }
  .stale a { color:#FCA5A5; text-decoration:underline; }

  .panel { background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
    padding:16px; box-shadow:var(--shadow); margin-bottom:12px; }
  .panel > h2 { font-size:17px; font-weight:600; margin:0 0 12px; letter-spacing:-.01em;
    display:flex; align-items:center; gap:9px; }
  .count { font-size:12px; font-weight:600; color:var(--text-3); background:var(--inset);
    border:1px solid var(--border); border-radius:999px; padding:1px 10px; }

  /* overflow-x:auto, not hidden: the latest-run table has nine columns with
     non-wrapping headers, so clipping would put the rightmost columns and the
     report links out of reach on a narrow viewport. A scroll container still
     clips to the border-radius. */
  .table-wrap { border:1px solid var(--border); border-radius:var(--radius-sm); overflow-x:auto; }
  table { width:100%; border-collapse:separate; border-spacing:0; }
  th { text-align:left; padding:10px 12px; font-size:var(--fs-th); font-weight:600;
    text-transform:uppercase; letter-spacing:.06em; color:var(--text-3);
    background:var(--inset); border-bottom:1px solid var(--border); white-space:nowrap; }
  td { padding:11px 12px; font-size:var(--fs-td); border-bottom:1px solid var(--border);
    color:var(--text-2); vertical-align:top; }
  tbody tr:last-child td { border-bottom:none; }
  tbody tr:hover td { background:rgba(148,163,184,.035); }
  /* Numeric headers right-align with their cells so a value sits under its label. */
  th.num, td.num { text-align:right; font-variant-numeric:tabular-nums; }
  td.mono { font-family:var(--mono); font-size:14px; color:var(--text-1); }
  td.wrap-any { word-break:break-word; }
  td.empty { color:var(--text-3); font-style:italic; }
  .dash { color:var(--text-3); }

  /* Verdict badge: tinted, never solid. fail and error share one red. */
  .v { display:inline-block; padding:2px 10px; border-radius:999px;
    font:600 var(--fs-badge) var(--mono); border:1px solid transparent; white-space:nowrap; }
  .v.pass  { color:#86EFAC; background:rgba(34,197,94,.13);  border-color:rgba(34,197,94,.32); }
  .v.warn  { color:#FCD34D; background:rgba(245,158,11,.13); border-color:rgba(245,158,11,.34); }
  .v.fail, .v.error { color:#FCA5A5; background:rgba(239,68,68,.13); border-color:rgba(239,68,68,.34); }
  .v.neutral { color:var(--text-3); background:rgba(148,163,184,.10); border-color:rgba(148,163,184,.24); }

  /* Baseline status: the only solid fill on the page. */
  .pill { display:inline-block; padding:3px 11px; border-radius:999px;
    font-size:var(--fs-badge); font-weight:600; color:#fff; white-space:nowrap; }
  .pill.ok, .pill.pass { background:var(--solid-ok); }
  .pill.bad, .pill.fail { background:var(--solid-bad); }

  .kcard { background:var(--inset); border:1px solid var(--border); border-radius:var(--radius-sm);
    margin-bottom:10px; overflow:hidden; }
  .kcard:last-child { margin-bottom:0; }
  .kcard.wc { --accent:var(--purple); }
  .kcard.cs { --accent:var(--green); }
  .kcard.nu { --accent:var(--blue); }
  .kcard > summary { list-style:none; cursor:pointer; user-select:none; display:flex;
    align-items:center; gap:11px; flex-wrap:wrap; padding:12px 14px; transition:background .14s ease; }
  .kcard > summary::-webkit-details-marker { display:none; }
  .kcard > summary:hover { background:rgba(148,163,184,.04); }
  .kcard > summary:focus-visible { outline:2px solid var(--blue); outline-offset:-2px; }
  .kcard[open] > summary { border-bottom:1px solid var(--border); }
  .kcard summary .name { font-family:var(--mono); font-weight:700; font-size:var(--fs-name); color:var(--text-1); }
  .kcard summary .kind { font-size:var(--fs-kind); color:var(--text-3); }
  .kcard summary .findings { font-size:13px; color:var(--text-3); background:rgba(148,163,184,.07);
    border:1px solid var(--border); border-radius:999px; padding:2px 10px; }
  .kcard summary .findings.has { color:#FCD34D; border-color:rgba(245,158,11,.30); background:rgba(245,158,11,.08); }
  .kcard summary .spacer { flex:1 1 auto; }
  .chevron { color:var(--text-3); transition:transform .2s ease; flex:0 0 auto; }
  details[open] > summary .chevron { transform:rotate(180deg); }
  .kcard-body { padding:14px; }

  /* Borderless label/value facts. Bordered boxes in a row read as tabs.
     Sized to content rather than uniformly: equal-width tracks gave a 1-char
     Findings the same width as a 51-char Recipe, so short facts sat on ~555px
     of dead space while Commit and Recipe wrapped for want of ~330px. Flex
     trades the column alignment between rows for giving each fact the width it
     needs. Derived from content on purpose -- a per-field width table would be
     wrong as soon as Run and Date grow. */
  .kv { display:flex; flex-wrap:wrap; gap:12px 28px;
    padding-bottom:14px; margin-bottom:14px; border-bottom:1px solid var(--border); }
  /* min-width:0 lets a long value wrap inside its own item instead of forcing
     the row wider; max-width keeps the widest of them within the panel. */
  .kv > span { flex:0 1 auto; min-width:0; max-width:100%; }
  .kv .k { display:block; font-size:var(--fs-kvk); color:var(--text-3); text-transform:uppercase;
    letter-spacing:.05em; font-weight:600; margin-bottom:3px; }
  .kv .val { display:block; font-size:var(--fs-kvv); color:var(--text-1); }
  /* break-all, not break-word: break-word only breaks a word that cannot fit a
     line by itself, which still leaves a digest-pinned image ref overflowing. */
  .kv .val.mono { font-family:var(--mono); word-break:break-all; }

  .observation { font-size:var(--fs-obs); color:var(--text-2); line-height:1.5;
    background:rgba(148,163,184,.05); border-left:2px solid rgba(148,163,184,.26);
    border-radius:0 8px 8px 0; padding:10px 13px; margin-bottom:14px; }
  .observation.warn { border-left-color:var(--v-warn); background:rgba(245,158,11,.06); }
  .observation.fail, .observation.error { border-left-color:var(--v-fail); background:rgba(239,68,68,.06); }
  .observation .lbl { display:block; font-size:12px; color:var(--text-3); text-transform:uppercase;
    letter-spacing:.06em; font-weight:600; margin-bottom:3px; }
  .observation .msg { font-family:var(--mono); font-size:14px; word-break:break-word; color:var(--text-1); }

  /* Holds the command and nothing else. With the label inside, this box read as
     a code block whose first token was "REPRODUCE", and selecting it to copy
     picked the label up along with the command. The label sits above it now. */
  .repro { display:block; background:var(--sunken);
    border:1px solid var(--border); border-radius:8px; padding:10px 13px; margin-bottom:14px; }
  .repro code { font-family:var(--mono); font-size:14px; color:#A5D6FF; word-break:break-all; }

  /* A section heading. Must not share colour, case and weight with the th
     beneath it -- and has to own its own separation. With margin-top:0 the space
     above a heading was whatever the previous element happened to leave (14px,
     against the 8px binding the heading to its own content), so sections ran
     together; and it was delegated to a `.cap + .table-wrap` adjacency rule, so
     inserting anything between a heading and its table silently deleted the gap.
     24px here makes separation ~3x the binding and independent of what precedes. */
  .cap { display:flex; align-items:center; gap:9px; font-size:var(--fs-cap); font-weight:700;
    text-transform:uppercase; letter-spacing:.06em; color:var(--text-1); margin:24px 0 8px; }
  .cap:first-child { margin-top:0; }
  .cap::before { content:""; width:3px; height:15px; border-radius:2px;
    background:var(--accent, var(--text-3)); flex:0 0 auto; }
  .card-foot { display:flex; justify-content:flex-end; font-size:13px; }

  details.panel { padding:0; }
  details.panel > summary { list-style:none; cursor:pointer; user-select:none; display:flex;
    align-items:center; gap:10px; padding:15px 16px; font-size:17px; font-weight:600; }
  details.panel > summary::-webkit-details-marker { display:none; }
  details.panel > summary:hover { background:rgba(148,163,184,.03); }
  details.panel > summary:focus-visible { outline:2px solid var(--blue); outline-offset:-2px; }
  details.panel[open] > summary { border-bottom:1px solid var(--border); }
  details.panel > summary .spacer { flex:1 1 auto; }
  .panel-body { padding:16px; }

  .chips { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; }
  .chip { background:var(--inset); border:1px solid var(--border); border-radius:var(--radius-sm);
    padding:13px 15px; display:flex; gap:12px; align-items:center;
    transition:border-color .16s ease, background .16s ease; }
  .chip:hover { border-color:var(--border-hover); background:var(--raised); }
  .chip .circle { width:34px; height:34px; flex:0 0 auto; border-radius:50%; display:grid; place-items:center; }
  .chip .circle.blue   { background:rgba(59,130,246,.14); color:var(--blue);   box-shadow:0 0 0 1px rgba(59,130,246,.24) inset; }
  .chip .circle.purple { background:rgba(139,92,246,.14); color:var(--purple); box-shadow:0 0 0 1px rgba(139,92,246,.24) inset; }
  .chip .circle.green  { background:rgba(34,197,94,.14);  color:var(--green);  box-shadow:0 0 0 1px rgba(34,197,94,.24) inset; }
  .chip .h { font-size:14px; font-weight:600; color:var(--text-1); display:block; }
  .chip .d { font-size:13px; color:var(--text-3); display:block; margin-top:2px; line-height:1.4; }

  .two-col { display:grid; grid-template-columns:1.85fr 1fr; gap:12px; align-items:start; margin-bottom:12px; }
  .two-col > .panel { margin-bottom:0; }

  /* Same reachability rule as .table-wrap: the five steps are fixed-width, so a
     narrow viewport scrolls the band instead of spilling it out of its panel. */
  .flow { display:flex; align-items:flex-start; justify-content:space-between; gap:4px;
    padding:2px 0; overflow-x:auto; }
  .step { display:flex; flex-direction:column; align-items:center; text-align:center; width:104px; }
  /* Solid, deepened fills: white on the identity accents at 85% alpha measured
     2.99:1 for green (WCAG AA needs 4.5:1 for this 11.5px bold text). These
     opaque shades clear it -- blue 5.2:1, purple 5.7:1, green 5.0:1. */
  .step-num { width:21px; height:21px; border-radius:50%; display:grid; place-items:center;
    font-size:11.5px; font-weight:700; margin-bottom:7px; color:#fff; }
  .step-num.blue { background:#2563EB; }
  .step-num.purple { background:#7C3AED; }
  .step-num.green { background:#15803D; }
  .step .tile { width:46px; height:46px; border-radius:var(--radius-sm); }
  .step .label { font-size:13px; font-weight:600; color:var(--text-1); margin-top:9px; line-height:1.25; }
  .step .desc { font-size:12px; color:var(--text-3); margin-top:4px; line-height:1.35; }
  .flow-arrow { color:rgba(148,163,184,.42); font-size:15px; margin-top:46px; flex:0 0 auto; }

  .panel-note { margin-top:14px; display:flex; align-items:center; gap:10px; background:var(--inset);
    border:1px solid var(--border); border-radius:var(--radius-sm); padding:11px 14px;
    font-size:13px; color:var(--text-2); }
  .panel-note svg { flex:0 0 auto; color:var(--text-3); }

  details.notes > summary { justify-content:space-between; }
  details.notes ul { list-style:none; margin:0; padding:0 16px 15px; display:flex; flex-direction:column; gap:11px; }
  details.notes li { display:flex; gap:9px; align-items:flex-start; font-size:13px; color:var(--text-2); line-height:1.45; }
  .check { flex:0 0 auto; width:17px; height:17px; border-radius:50%; display:grid; place-items:center;
    margin-top:1px; background:rgba(34,197,94,.14); color:var(--green);
    box-shadow:0 0 0 1px rgba(34,197,94,.26) inset; }

  .statline { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:12px; }
  .statline .lead { font-size:14px; color:var(--text-2); margin-right:4px; }
  .kgroup-title .kname { font-family:var(--mono); font-size:16px; }
  .survey-empty { font-size:14px; color:var(--text-2); margin:8px 0; }

  @media (max-width:1000px) {
    .topbar { flex-direction:column; } .runcard { width:100%; }
    .two-col { grid-template-columns:1fr; }
    .flow { flex-wrap:wrap; justify-content:center; gap:10px; }
  }
  @media (max-width:720px) { .toolgrid, .chips { grid-template-columns:1fr; } }
"""


# Inline SVG icons (Feather/Lucide outline, 1.8-2.0 stroke, never filled). Inline
# so the page stays self-contained -- no CDN, no sprite sheet, no web font.
_ICON_ACTIVITY = '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>'
_ICON_SHIELD = (
    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
    '<polyline points="9 12 11 14 15 10"/>'
)
_ICON_INFO = (
    '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/>'
    '<line x1="12" y1="8" x2="12.01" y2="8"/>'
)
_ICON_CHECK = '<polyline points="20 6 9 17 4 12"/>'
_ICON_CHEVRON = '<polyline points="6 9 12 15 18 9"/>'
_ICON_FILE = (
    '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
    '<polyline points="14 2 14 8 20 8"/>'
)
_ICON_FOLDER = '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>'
_ICON_CODE = '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>'
_ICON_LAYOUT = (
    '<rect x="3" y="3" width="18" height="18" rx="2"/>'
    '<line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/>'
)
_ICON_SEARCH = '<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'
_ICON_BUG = (
    '<path d="m8 2 1.9 1.9M14.1 3.9 16 2"/><path d="M9 7.1v-1a3 3 0 1 1 6 0v1"/>'
    '<path d="M12 20c-3.3 0-6-2.7-6-6v-3a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v3c0 3.3-2.7 6-6 6"/>'
    '<path d="M12 20v-9M6 13H2M22 13h-4"/>'
    '<path d="M6.5 9C4.6 8.8 3 7.1 3 5M21 5c0 2.1-1.6 3.8-3.5 4"/>'
    '<path d="M3 21c0-2.1 1.7-3.9 3.8-4M17.2 17c2.1.1 3.8 1.9 3.8 4"/>'
)
_ICON_REPEAT = (
    '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>'
    '<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>'
)


def _svg(body: str, *, size: int = 18, width: float = 2, cls: str = "") -> str:
    """Wrap an inline icon path set in a stroked, unfilled ``<svg>``."""
    class_attr = f' class="{cls}"' if cls else ""
    return (
        f'<svg{class_attr} width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="{width}" stroke-linecap="round" '
        f'stroke-linejoin="round">{body}</svg>'
    )


def _chevron_html() -> str:
    return _svg(_ICON_CHEVRON, size=18, width=2, cls="chevron")


def _tool_cards_html() -> str:
    """WaitCheck / ConSan explainer cards.

    Page level, above the tab bar: both tabs surface results from both
    sanitizers, so the definitions are common context. Tab-specific framing
    (the gate banner, the observed-only banner) lives inside each tab.
    """
    return (
        '<section class="toolgrid">'
        '<article class="feature-card">'
        f'<div class="tile purple">{_svg(_ICON_SHIELD, size=22, width=1.8)}</div>'
        "<div>"
        '<div class="title-row"><h3>WaitCheck</h3>'
        '<span class="badge purple">Static Analysis</span></div>'
        "<p>Detects potential <code>s_waitcnt</code> synchronization hazards "
        "using static code analysis.</p>"
        "</div></article>"
        '<article class="feature-card">'
        f'<div class="tile green">{_svg(_ICON_ACTIVITY, size=22, width=1.8)}</div>'
        "<div>"
        '<div class="title-row"><h3>ConSan</h3>'
        '<span class="badge green">Runtime Analysis</span></div>'
        "<p>Detects runtime data races and concurrency issues during execution.</p>"
        "</div></article>"
        "</section>"
    )


def _run_card_html(meta: dict[str, Any]) -> str:
    """Run provenance as a compact top-right card.

    Deliberately *not* a row of bordered chips: bordered boxes in a horizontal
    row directly above the tab bar read as a second set of tabs.
    """
    rows = (
        ("Run", meta.get("run", "")),
        ("Commit", meta.get("commit", "")),
        ("Date", meta.get("date", "")),
        ("Target", meta.get("gpu", "gfx950")),
    )
    body = "".join(
        f'<div class="runrow"><span class="k">{_esc(label)}</span>'
        f'<span class="rv">{_esc(str(value))}</span></div>'
        for label, value in rows
    )
    return f'<aside class="runcard">{body}</aside>'


def _gate_hero_html(summary: dict[str, Any]) -> str:
    """The Tab 1 gate banner: state on one line, detail on the next.

    Lives inside Tab 1, not the page header -- baselines do not exist on the
    survey tab, so a gate verdict is meaningless there.
    """
    cls = "ok" if summary["ok"] else "bad"
    icon = _ICON_CHECK if summary["ok"] else _ICON_INFO
    return (
        f'<div class="gate {cls}">'
        f'<span class="gate-icon">{_svg(icon, size=17, width=3)}</span>'
        f'<span class="gate-text"><strong>{_esc(summary["label"])}</strong>'
        f'{_esc(summary["detail"])}</span></div>'
    )




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


def _primary_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce a report's checks to a primary (sanitizer, verdict, reason) plus the
    ConSan ``waitcheck_preflight`` verdict.

    Folds the ``gen_new_kernels_dashboard.py`` prototype's ``_primary`` helper into
    the generator (rather than a post-hoc HTML splice): the last non-preflight
    check wins as the "primary" sanitizer/verdict/reason, and the preflight verdict
    is surfaced separately. Values are kept as-is (``None`` preserved) so callers
    decide how to render an absent field.
    """
    primary: dict[str, Any] = {
        "sanitizer": None, "verdict": None, "reason": None, "preflight": None
    }
    for check in checks:
        if check.get("sanitizer") == "waitcheck_preflight":
            primary["preflight"] = check.get("verdict")
        else:
            primary["sanitizer"] = check.get("sanitizer")
            primary["verdict"] = check.get("verdict")
            primary["reason"] = check.get("reason")
    return primary


def _observation_text(
    primary: dict[str, Any],
    findings: int,
    finding_groups: list[dict[str, Any]],
    *,
    present: bool = True,
) -> str:
    """A one-line human summary of what a case observed.

    Combines the primary sanitizer + verdict + fail-closed reason + a finding
    highlight into a compact string surfaced on both tabs (guardrail and survey).
    Observational only -- it never encodes a pass/fail health signal.
    """
    if not present:
        return "report missing"
    san, verdict = primary.get("sanitizer"), primary.get("verdict")
    head = f"{san or _DASH} {verdict or _DASH}" if (san or verdict) else "no sanitizer check ran"
    parts = [head]
    if primary.get("reason"):
        parts.append(f"reason {primary['reason']}")
    if findings:
        top = finding_groups[0]["code"] if finding_groups else None
        parts.append(f"{findings} finding(s)" + (f" ({top})" if top else ""))
    if primary.get("preflight"):
        parts.append(f"preflight {primary['preflight']}")
    return "; ".join(parts)


def summarize_case(report: dict[str, Any] | None, expected: str | None) -> dict[str, Any]:
    """Reduce one report to the fields the dashboard renders (pure).

    ``expected=None`` marks an observed-only *survey* case (``match`` is ``True``):
    the reduced row carries no baseline signal and the renderer must treat it as
    observational, never as a passing/failing gate row.
    """
    if not isinstance(report, dict):
        # Treat a missing report AND a structurally invalid one (``_load`` returns
        # any JSON value, so a report file holding ``[]``/a scalar reaches here as a
        # non-dict) as absent rather than calling ``.get`` on it and aborting the
        # whole dashboard render. Match follows the same rule as the present path: a
        # survey case (expected is None) is never a mismatch, while a guardrail case
        # keeps a non-null expectation and so an absent report stays a mismatch (the
        # gate fails closed). Hardcoding False here contradicted the survey contract.
        return {
            "present": False, "verdict": "—", "execution": "missing", "findings": 0,
            "coverage": "", "backend": None, "expected": expected, "match": expected is None,
            "worklist": {"requirement": None, "top_n": None, "kernel_count": 0},
            "kernels": [], "finding_groups": [],
            "primary": {"sanitizer": None, "verdict": None, "reason": None, "preflight": None},
            "observation": "report missing",
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
    primary = _primary_checks(checks)
    observation = _observation_text(primary, findings_total, finding_groups)
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
        "primary": primary, "observation": observation,
    }


def _run_record(
    meta: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    *,
    rel: str | None = None,
) -> dict[str, Any]:
    # Guardrail rows are the baseline-checked class (Tab 1). Tag them so data.json
    # carries the guardrail/survey case-class split; survey cases live in a separate
    # ``survey`` list attached by ``main`` to the run each list describes.
    for r in rows.values():
        r.setdefault("cls", "guardrail")
    # Fail closed: a missing report (present=False -> match=False) turns the gate
    # unhealthy, mirroring compare_verdict_baselines.py which errors on absent reports.
    gate = all(r["match"] for r in rows.values())
    # The authoritative comparator gate recorded in the run manifest (meta.gate)
    # is stricter than a row-level overall_verdict match (it also checks schema,
    # execution status, per-check verdicts, and finding shape). Honor a recorded
    # failure so a comparator-failed run whose verdicts happen to match baselines
    # is never rendered healthy, and run.gate agrees with meta.gate in data.json.
    if meta.get("gate") is False:
        gate = False
    # ``rel`` is the run's relative area under the published dashboard (e.g.
    # ``runs/<id>``); None for the results-dir / runs-root modes, which do not
    # publish co-located raw reports and therefore render no per-run links.
    return {"meta": meta, "rows": rows, "gate": gate, "rel": rel}


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


def _run_meta_from_history(run_dir: Path) -> dict[str, Any]:
    """Read a published run's ``meta.json`` (commit, date, gpu, run_url, gate).

    The run id (``<YYYY-MM-DD>-<run_id>``) is authoritative from the directory
    name; ``meta.json`` supplies the rest. A missing/corrupt manifest degrades to
    an id-only record rather than crashing the whole dashboard render.
    """
    meta: dict[str, Any] = {"run": run_dir.name, "commit": "", "date": "", "gpu": "gfx950"}
    data = _load(run_dir / "meta.json")
    if isinstance(data, dict):
        if data.get("commit"):
            meta["commit"] = _short(str(data["commit"]), 12)
        for key in ("date", "gpu", "run_url"):
            value = data.get(key)
            if value not in (None, ""):
                meta[key] = str(value)
        # The unabbreviated SHA as well: ``commit`` is shortened for display, but
        # a run area's reproduce instructions need a checkout-able revision.
        if data.get("commit"):
            meta["commit_full"] = str(data["commit"])
        # Preserve the manifest's recorded gate with its JSON type. Coercing it to
        # str would emit "True"/"False" into data.json, where "False" is truthy
        # for machine consumers and the documented boolean type is lost.
        gate = data.get("gate")
        if gate is not None:
            meta["gate"] = gate
    return meta


# A published run directory is ``<YYYY-MM-DD>-<run_id>`` (run_id an integer).
_RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d+$")


def _is_run_id(name: str) -> bool:
    """Whether a directory name is a well-formed ``<YYYY-MM-DD>-<run_id>`` id."""
    return _RUN_ID_RE.match(name) is not None


def _history_sort_key(name: str) -> tuple[str, int]:
    """Order key for a published run id (``<YYYY-MM-DD>-<run_id>``).

    Enumeration is already filtered to well-formed ids (``_is_run_id``); this key
    exists because the trailing run id is a *variable-width* integer, so a plain
    name sort mis-orders ``...-9`` after ``...-10`` and would pick the wrong
    latest run (and prune the wrong dir). Split the numeric suffix off and compare
    it as an int. The malformed fallback ``(name, -1)`` is defensive only.
    """
    head, sep, tail = name.rpartition("-")
    if sep and tail.isdigit():
        return (head, int(tail))
    return (name, -1)


def runs_from_history_root(
    history_root: Path, baselines: dict, *, keep: int = 30
) -> list[dict[str, Any]]:
    """Enumerate the PUBLISHED ``DIR/<id>/<case>/sanitizer_report.json`` layout.

    Runs are ordered newest-first by ``<id>`` (``<YYYY-MM-DD>-<run_id>``) using a
    date-then-numeric key (see ``_history_sort_key``; a plain string sort would
    put ``...-9`` after ``...-10``) and capped to the newest ``keep``. Each
    summarized row is tagged with a ``report_rel`` pointing at its raw JSON
    relative to the dashboard root, and each record with the run's ``rel`` area,
    so ``build_html`` can emit relative links that work under ``/sanitizers/``.
    """
    # Only enumerate well-formed <YYYY-MM-DD>-<run_id> dirs: a stray child (e.g. a
    # leftover `source/` from a nested --history-root layout) must not become a
    # phantom "latest" pseudo-run or consume a --keep slot.
    run_dirs = sorted(
        (p for p in history_root.iterdir() if p.is_dir() and _is_run_id(p.name)),
        key=lambda p: _history_sort_key(p.name),
        reverse=True,
    ) if history_root.is_dir() else []
    runs: list[dict[str, Any]] = []
    for run_dir in run_dirs[: max(keep, 0)]:
        rel = f"runs/{run_dir.name}"
        rows: dict[str, dict[str, Any]] = {}
        for case, key, _label, _backend in CASES:
            row = summarize_case(
                _load(run_dir / case / "sanitizer_report.json"),
                baselines.get(key, {}).get("overall_verdict"),
            )
            # Only link reports that are actually present, so the page never
            # points at a 404. Relative to the dashboard root (runs/<id>/...).
            row["report_rel"] = f"{rel}/{case}/sanitizer_report.json" if row["present"] else None
            rows[case] = row
        runs.append(_run_record(_run_meta_from_history(run_dir), rows, rel=rel))
    return runs


def _safe_report_rel(value: Any) -> str | None:
    """Accept only a same-origin *relative* path for a survey report link.

    ``report_rel`` on a survey case is caller-supplied JSON, so a value like
    ``javascript:alert(1)`` or ``//evil.example/x`` would otherwise render as an
    executable or off-origin ``<a href>`` -- HTML-escaping the attribute does not
    neutralize a URL scheme. Reject anything that is not a plain relative path:
    a URL scheme (``scheme:``), absolute or protocol-relative paths (leading
    ``/``), backslashes, ``..`` parent-directory traversal, embedded
    control/whitespace characters, and the URL delimiters ``?``, ``#`` and ``%``
    -- which address something other than the path they appear in. The
    internally-computed guardrail/history
    links already have this shape, so this only tightens the untrusted survey
    field. Rejecting ``..`` also makes the value safe to join onto a base
    directory when it is (re)used as a filesystem ``report_path`` (see
    ``_load_survey_report``): the joined path can never escape the base.
    """
    if not isinstance(value, str) or not value:
        return None
    if value != value.strip() or any(ord(ch) < 0x20 for ch in value):
        return None
    if value.startswith("/") or "\\" in value:
        return None
    # URL syntax has to go as well, not just filesystem-unsafe characters: the
    # value is used verbatim in an ``href``, where ``?`` and ``#`` turn the rest of
    # the path into a query/fragment on the segment before them, and a
    # percent-escape can be normalised back into ``..`` by URL resolution. So
    # ``runs/x?old/survey/foo/sanitizer_report.json`` passes every check below
    # while the browser requests ``runs/x``. None of the paths this generator
    # builds contain any of them.
    if any(ch in value for ch in "?#%"):
        return None
    # A relative path never carries a URL scheme; a colon in the first segment
    # (before the first "/", e.g. ``javascript:``) means it is a scheme, not a path.
    if ":" in value.split("/", 1)[0]:
        return None
    # No parent-directory traversal, so the path stays within its origin.
    if ".." in value.split("/"):
        return None
    return value


def _load_survey_report(report_path: str, base_dir: Path | None) -> dict[str, Any] | None:
    """Load a survey case's ``report_path`` (untrusted caller JSON) safely.

    Only a validated *relative* path that stays beneath ``base_dir`` is read, so a
    supplied ``--survey`` spec cannot pull JSON from outside the spec directory --
    via an absolute path, ``..`` traversal, or a URL scheme -- and serialize its
    contents into the public dashboard. This mirrors the relative-only boundary
    ``_safe_report_rel`` enforces on the ``report_rel`` link; an unsafe value
    renders the case as absent (no report) rather than reading the file.
    """
    safe_rel = _safe_report_rel(report_path)
    if safe_rel is None:
        return None
    base = (base_dir if base_dir is not None else Path(".")).resolve()
    candidate = (base / safe_rel).resolve()
    # Defense in depth against symlink escapes: the resolved target must stay at
    # or beneath base_dir even though ``..``/absolute inputs are already rejected.
    if candidate != base and base not in candidate.parents:
        return None
    return _load(candidate)


def survey_cases_from_spec(
    spec: dict[str, Any] | list[Any],
    *,
    base_dir: Path | None = None,
    rel: str | None = None,
) -> list[dict[str, Any]]:
    """Build ordered observed-only *survey* case entries from a spec (pure-ish).

    The survey tab shows kernels drawn from multiple workloads -- including kernels
    extracted from aorta-internal workloads -- with no expected/baseline signal. To
    keep this public repo free of customer/NDA identifiers (CLAUDE.md rule #4), the
    survey *data* is supplied at run time via ``--survey`` (JSON), never hardcoded
    here: an internal-hosted build feeds internal-sourced rows through the same
    renderer, while the public build feeds only scrubbed/public rows.

    Spec shape (a bare list of cases, or ``{"cases": [...]}``); each case::

        {
          "name": "top5-gemm-consan",          # stable id
          "label": "top-5 f32 GEMM \u00b7 ConSan",   # display label (defaults to name)
          "backend": "consan (dynamic)",       # backend/tool label
          "workload": "internal:gemm_top5",    # originating workload (optional)
          "report_rel": "runs/<id>/survey/top5-gemm-consan/sanitizer_report.json",  # optional link
          "report": { ...inline sanitizer_report... }   # inline report, OR
          "report_path": "relative/report.json"   # a file to load (relative to base_dir)
        }

    Each entry carries ``cls="survey"`` and a ``summarize_case(report, None)``
    summary (``expected=None`` -> observational, ``match=True``), so failures /
    ``not_checked`` are shown as data and never rendered as a regression.

    Both untrusted-JSON path fields are relative-only and validated: ``report_rel``
    (the link) via ``_safe_report_rel`` and ``report_path`` (the file to load) via
    ``_load_survey_report``, which reads only a validated relative path beneath
    ``base_dir`` (no absolute paths, ``..`` traversal, or URL schemes). ``report_rel``
    is retained only for a *present* report (no dead link, matching the guardrail
    path in ``runs_from_history_root``). A malformed spec (a non-list ``cases``
    wrapper, a non-string / unsafe ``report_path``) degrades to an empty / absent
    survey rather than raising.
    """
    cases = spec.get("cases", []) if isinstance(spec, dict) else spec
    # A malformed wrapper (e.g. ``{"cases": 1}``) must degrade to an empty survey,
    # not raise a TypeError when iterated -- matching the CLI's promise that a bad
    # --survey spec renders the empty-state note rather than aborting the render.
    if not isinstance(cases, list):
        cases = []
    entries: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        report = case.get("report")
        # Only load from a non-empty string; a non-string report_path (e.g. a
        # numeric JSON value) would otherwise raise in Path() and abort the whole
        # dashboard instead of rendering this case as absent. The path is untrusted
        # caller JSON, so _load_survey_report resolves only a validated relative
        # path beneath base_dir (no absolute paths / ``..`` traversal / schemes).
        report_path = case.get("report_path")
        if report is None and isinstance(report_path, str) and report_path:
            report = _load_survey_report(report_path, base_dir)
        summary = summarize_case(report if isinstance(report, dict) else None, None)
        name = str(case.get("name", case.get("label", "survey")))
        backend = case.get("backend")
        if not backend:
            b = summary.get("backend")
            backend = b["name"] if b else _DASH
        sanitizer = case.get("sanitizer")
        if not sanitizer:
            sanitizer = (summary.get("primary") or {}).get("sanitizer")
        entries.append(
            {
                "name": name,
                "label": str(case.get("label", name)),
                # Optional kernel-group key so a spec can render both sanitizers of
                # one kernel under a single heading (defaults to the case name -> a
                # standalone case). ``command`` is an optional copy-paste reproduce
                # line surfaced on Tab 2 (#374 B/C).
                "group": str(case.get("group", name)),
                "group_label": case.get("group_label"),
                "sanitizer": sanitizer,
                "command": case.get("command"),
                "backend": str(backend),
                "workload": case.get("workload"),
                # Keep a report link only for a present report (no dead link,
                # matching runs_from_history_root) and only when it is a safe
                # relative path (report_rel is untrusted caller JSON).
                "report_rel": (
                    _safe_report_rel(case.get("report_rel")) if summary["present"] else None
                ),
                # Whether this case's whole run-area *directory* is published, not
                # just its report. A spec entry renders from an inline report or an
                # out-of-tree report_path, so by default nothing stages the
                # directory and no run-area link may be emitted for it (a present
                # report is not evidence the directory exists). A caller that does
                # publish complete areas can say so with `"staged": true` -- but
                # only a name that is one plain path segment can be claimed, since
                # the area link is built from it and `..` would traverse.
                #
                # The two renderers derive this link from different sources -- the
                # run page builds ``survey/<name>/`` relative to the run it is
                # rendering, the card footer takes the directory of ``report_rel``
                # -- so they can only be trusted if the *whole* area is pinned to
                # one run. Comparing only the case segment was not enough: a report
                # under ``runs/other/survey/foo/`` matched ``name="foo"`` while the
                # footer pointed into a different run. Require the complete path
                # for the run being rendered, which is exactly what the
                # informational builder constructs, so the two agree by
                # construction. Without a run context (``rel=None``) nothing can be
                # verified and no spec entry may claim an area.
                "staged": (
                    case.get("staged") is True
                    and summary["present"]
                    and _safe_case_segment(name) is not None
                    and bool(rel)
                    and _safe_report_rel(case.get("report_rel"))
                    == f"{rel}/survey/{name}/sanitizer_report.json"
                ),
                "cls": "survey",
                "summary": summary,
            }
        )
    return entries


# The recipe that actually produces each best-effort survey case's report, so the
# "reproduce this yourself" command on Tab 2 is the REAL recipe rather than a
# name-derived guess. Most case dirs (``<sanitizer>-<group>``) map to
# ``daily-<case>.yaml``, but the gemm waitcheck survey uses a dedicated object recipe
# (``daily-waitcheck-gemm.yaml`` is the *gated* Tab-1 guardrail and is never reused
# for the survey). Recipe names are generic/public -- no customer/NDA identifiers.
_SURVEY_RECIPE: dict[str, str] = {
    "waitcheck-gemm": "daily-waitcheck-gemm-object",
}


def _survey_recipe_for(name: str) -> str:
    """The recipe filename stem that produces survey case ``name``'s report."""
    return _SURVEY_RECIPE.get(name, f"daily-{name}")


def _split_survey_case(name: str) -> tuple[str, str | None]:
    """Split a survey case dir name into a (kernel-group, sanitizer) pair.

    The nightly names each best-effort survey case ``<sanitizer>-<group>`` (e.g.
    ``consan-gemm``, ``waitcheck-gemm``, ``consan-lds-dispatch``), so the same kernel
    scanned by both sanitizers shares one group key and renders under one heading
    with a sanitizer sub-block each. A name without a recognized sanitizer prefix is
    its own group with an unknown sanitizer (the report's primary sanitizer is used
    for display). This parse is purely structural -- no customer/NDA identifiers.
    """
    for prefix, san in (("consan-", "consan"), ("waitcheck-", "waitcheck")):
        if name.startswith(prefix):
            return name[len(prefix):], san
    return name, None


def _published_command(case_dir: Path) -> str:
    """The reproduce command a published run area recorded, or "" (no manifest).

    A fresh sweep's results dir has no ``env.json``, so this is empty there and the
    caller falls back to the recipe map.
    """
    env = _load(case_dir / "env.json")
    command = env.get("command") if isinstance(env, dict) else None
    return command if isinstance(command, str) and command else ""


def survey_cases_from_informational_dir(
    root: Path, *, rel: str | None = None
) -> list[dict[str, Any]]:
    """Build observed-only *survey* entries from a caller-supplied results dir.

    Enumerates ``root/<case>/sanitizer_report.json`` (the layout the nightly writes
    for its best-effort survey recipes) and folds each case into the same Tab 2
    workload-survey shape ``survey_cases_from_spec`` produces -- kernel name,
    dispatch, observed verdict, findings, code object / SHA, selection meta, and a
    one-line observation carrying any fail-closed reason. Each entry is observed-only
    (``summarize_case(report, None)`` -> ``expected=None``, ``match`` True) so a
    ``fail``/``error`` here is an observation, never a gate.

    Cases are keyed into kernel groups by ``_split_survey_case`` so the same kernel
    run under both waitcheck and ConSan renders under one heading with a sanitizer
    sub-block each (#374 C). The provenance recipe is surfaced as a copy-pasteable
    reproduce ``command`` -- the REAL recipe that produced the report, resolved via
    ``_survey_recipe_for`` (a small case-name -> recipe map, defaulting to
    ``daily-<case>.yaml``) rather than an "experimental / caller-supplied" label
    (#374 B). No customer/NDA
    identifiers are hardcoded: the kernel data flows in at run time from the reports
    (CLAUDE.md rule #4); the generic CI GEMM/LDS/vecadd kernel names are public-safe.

    ``rel`` is the latest run's published area (e.g. ``runs/<id>``); when it is
    supplied and the report is present, ``report_rel`` is set to
    ``<rel>/survey/<name>/sanitizer_report.json`` (validated via ``_safe_report_rel``)
    so per-case + per-kernel raw-report links resolve to the co-published report (see
    ``main``). With ``rel=None`` (results-dir / runs-root modes, which publish no
    co-located reports) ``report_rel`` stays ``None`` so no dead link is emitted. An
    absent ``root`` returns ``[]`` and a case with no loadable report is skipped.
    """
    entries: list[dict[str, Any]] = []
    if not root.is_dir():
        return entries
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        report = _load(case_dir / "sanitizer_report.json")
        # Skip an unreadable report AND a structurally invalid one (``_load`` admits
        # any JSON value, so a best-effort survey report holding ``[]``/``null``/a
        # scalar arrives as a non-dict); rendering it would crash ``summarize_case``
        # on ``.get`` and abort the whole publication for a non-gating input.
        if not isinstance(report, dict):
            continue
        name = case_dir.name
        # A top-level dict can still hold a malformed *nested* shape (e.g.
        # ``{"checks": null}`` or a non-list ``worklist``) that makes the reduction
        # raise mid-iteration. Isolate it: one broken best-effort report must not
        # abort the whole non-gating publication -- skip the case (fails closed, the
        # same degradation as a non-object/unreadable report) rather than propagate.
        try:
            summary = summarize_case(report, None)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        backend = summary.get("backend")
        backend_name = backend["name"] if backend else _DASH
        group, san_from_name = _split_survey_case(name)
        sanitizer = san_from_name or (summary.get("primary") or {}).get("sanitizer")
        report_rel = None
        if rel is not None and summary["present"]:
            report_rel = _safe_report_rel(f"{rel}/survey/{name}/sanitizer_report.json")
        entries.append(
            {
                "name": name,
                "label": name,
                "group": group,
                "sanitizer": sanitizer,
                "backend": str(backend_name),
                "workload": None,
                # An already-published area records the command it actually ran, so
                # prefer it over recomputing from the current recipe map. This
                # function also reconstructs a published run's survey (see
                # ``displayed_survey`` in ``main``), and recomputing there let a
                # recipe rename rewrite a historical run's reproduce command --
                # the same "prose must not outlive its manifest" rule the run-area
                # renderers follow.
                "command": _published_command(case_dir)
                or (
                    "aorta sweep run --recipe "
                    f"recipes/sanitizers/{_survey_recipe_for(name)}.yaml"
                ),
                "report_rel": report_rel,
                # These are the cases ``main`` co-publishes as complete run areas
                # (report + logs + manifests + index.html) under
                # ``<rel>/survey/<name>/``, so unlike a ``--survey`` spec entry the
                # directory link is safe to emit. Gated on ``rel`` for the same
                # reason ``report_rel`` is: the results-dir / runs-root modes
                # publish nothing co-located.
                "staged": rel is not None and summary["present"],
                "cls": "survey",
                "summary": summary,
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Run area (#384)
#
# A published case directory is the "run area" a card footer links to. It holds
# everything needed to reproduce that one case locally: the report, the
# sanitizer logs the verdict was derived from, the recipe exactly as it ran,
# that recipe's source-level inputs, and REPRODUCE.md / env.json recording the
# CI-built artifacts that are deliberately NOT published, by digest.
#
# Everything here is stdlib-only on purpose: the publish job runs this script
# from a bare checkout with no ``pip install``, so PyYAML is not importable.
# ---------------------------------------------------------------------------

# Fixture paths under these prefixes are CI-built (see
# recipes/sanitizers/fixtures/.gitignore): a ~16MB Tensile code object or a
# compiled repro binary. They are never copied into a run area -- publishing
# them for every retained run would bloat the data branch and Pages -- so their
# SHA-256 is recorded instead and a local rebuild can be verified against it.
_BUILT_FIXTURE_DIRS = ("fixtures/bin/", "fixtures/isa/")

# Any fixture path a sanitizer recipe references. Recipes name them under
# several different keys (``path``, ``isa_dir``, ``code_object``,
# ``consan_command``, ``hip``, ``command``), so match the value shape rather
# than enumerating keys -- a recipe that adds a new key needs no change here.
_FIXTURE_REF_RE = re.compile(r"fixtures/[A-Za-z0-9._/-]+")


def _is_built_fixture(ref: str) -> bool:
    """Whether a recipe's fixture reference names CI-built output.

    Compares with a trailing slash appended so a bare directory reference
    (``isa_dir: fixtures/isa``) is classified alongside the files inside it; a
    plain ``startswith`` on the prefixes would let that one through as a source
    input and leave it off the "not published" list.
    """
    return f"{ref}/".startswith(_BUILT_FIXTURE_DIRS)


RUN_AREA_ENV_SCHEMA = "aorta.sanitizer_run_area/0.1"

# Provenance the workflow knows but the report does not. Absent => omitted from
# env.json rather than emitted empty, so the file only ever states what the run
# actually captured.
_RUN_AREA_ENV_VARS: tuple[tuple[str, str], ...] = (
    ("container_image", "AORTA_CI_IMAGE"),
    ("rocjitsu_commit", "AORTA_ROCJITSU_COMMIT"),
    ("rocjitsu_run_url", "AORTA_ROCJITSU_RUN_URL"),
)


def provenance_from_environ() -> dict[str, str]:
    """The publish job's environment-derived provenance for the run it just staged.

    These describe *this* job's GPU run -- the container image it ran in and the
    rocjitsu bundle its sanitizer backends came from -- so they are only valid
    for the run being published now, never for the older runs re-rendered
    alongside it (see ``write_case_run_area``).
    """
    return {key: os.environ[var] for key, var in _RUN_AREA_ENV_VARS if os.environ.get(var)}


# The characters a survey case directory is actually named with (``consan-gemm``,
# ``waitcheck-gemm``, ``consan-lds-dispatch``), leading with an alphanumeric so
# ``.``/``..`` cannot match. An allowlist rather than a list of things to reject,
# because the value is simultaneously a path segment and a URL segment: excluding
# only the filesystem-dangerous characters still admitted ``foo#bar`` and ``foo?x``
# -- which a browser reads as a fragment/query on ``foo`` -- and ``%2e%2e``, which
# URL resolution can normalise back into traversal.
_CASE_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _safe_case_segment(name: Any) -> str | None:
    """A survey case name accepted only as one plain, URL-safe path segment.

    The run page's area link is ``survey/<name>/``, and ``name`` on a ``--survey``
    entry is caller-supplied JSON. HTML-escaping that attribute does not neutralise
    path semantics, so a name like ``../../other`` would render a traversal link --
    and would also disagree with the card footer, which derives its link from the
    separately-validated ``report_rel``. Requiring a single segment makes the two
    agree by construction, since ``report_rel`` is built as
    ``<rel>/survey/<name>/sanitizer_report.json`` from this same value.
    """
    if not isinstance(name, str) or not _CASE_SEGMENT_RE.fullmatch(name):
        return None
    return name


def _case_dir_rel(report_rel: str | None) -> str | None:
    """The run-area directory holding a case's report, as a relative link.

    Derived from ``report_rel`` and re-validated through ``_safe_report_rel``,
    so the footer link inherits exactly the same relative-only guarantee as the
    report link it replaces: no URL scheme, no absolute path, no ``..``
    traversal. Returns ``None`` when there is no safe directory to link -- an
    absent or unsafe ``report_rel``, or a bare filename with no directory part
    -- so an unlinkable case still renders, just without a footer.
    """
    safe = _safe_report_rel(report_rel)
    if not safe or "/" not in safe:
        return None
    return f"{safe.rsplit('/', 1)[0]}/"


def _recipe_fixture_refs(recipe_text: str) -> tuple[list[str], list[str]]:
    """Split a recipe's fixture references into ``(source, built)`` paths (pure).

    Paths are as written in the recipe, i.e. relative to ``recipes/sanitizers/``.
    Source inputs (the ``.hip`` repro sources, the GEMM shape CSV) are small --
    hundreds of bytes to a few KB -- and are copied into the run area. Built
    artifacts are recorded by digest only. Order is first-seen and de-duplicated.

    References containing a ``..`` segment are dropped here, at the one boundary
    both consumers share. ``_FIXTURE_REF_RE`` admits ``.`` so it can match a file
    extension, which also let ``fixtures/../../../README.md`` through -- and while
    the copy validated that the *source* stayed inside the repo, it then joined the
    same unchecked value onto the destination and wrote outside the case
    directory. Nothing under ``fixtures/`` legitimately needs a parent segment,
    and rejecting it here keeps the copied inputs and the recorded
    not-published list consistent about which refs exist.
    """
    source: list[str] = []
    built: list[str] = []
    for ref in dict.fromkeys(_FIXTURE_REF_RE.findall(recipe_text)):
        if ".." in ref.split("/"):
            continue
        (built if _is_built_fixture(ref) else source).append(ref)
    return source, built


def report_digests(report: dict[str, Any] | None) -> dict[str, str]:
    """Artifact digests a run area records for the files it does not publish.

    Both sanitizers already write this provenance into every report and nothing
    renders it today: waitcheck stores its binary's ``path``/``sha256``, ConSan
    stores the repro ``command`` and hook plus their digests, and each worklist
    kernel carries its ``code_object_sha256``. Flattened into one mapping so
    REPRODUCE.md and env.json can name the exact artifacts a local rebuild has
    to match. A malformed report degrades to ``{}`` rather than raising.
    """
    digests: dict[str, str] = {}
    if not isinstance(report, dict):
        return digests
    checks = report.get("checks")
    for check in checks if isinstance(checks, list) else []:
        backend = check.get("backend") if isinstance(check, dict) else None
        if not isinstance(backend, dict):
            continue
        for key, value in backend.items():
            if isinstance(value, str) and value:
                digests.setdefault(str(key), value)
    worklist = report.get("worklist")
    kernels = worklist.get("kernels") if isinstance(worklist, dict) else None
    for entry in kernels if isinstance(kernels, list) else []:
        identity = entry.get("identity") if isinstance(entry, dict) else None
        if not isinstance(identity, dict):
            continue
        obj, sha = identity.get("code_object"), identity.get("code_object_sha256")
        if isinstance(obj, str) and obj and isinstance(sha, str) and sha:
            digests.setdefault(f"code_object:{_basename(obj)}", sha)
    return digests


def artifact_digest_index(digests: dict[str, str]) -> dict[str, str]:
    """Index recorded digests by artifact *basename* (pure).

    The two sanitizers name the same kind of thing differently: waitcheck records
    its binary as ``path``/``sha256``, ConSan records the repro binary as
    ``command``/``command_sha256`` and its hook as ``hook``/``hook_sha256``, and
    worklist kernels arrive from ``report_digests`` already keyed as
    ``code_object:<basename>``. A run area names the artifacts it did not publish
    by their recipe-relative path, so index by basename and look them up with one
    rule -- keying only on ``code_object:`` left every ``fixtures/bin/`` repro
    binary showing an em dash while its digest sat in the report unused.
    """
    index: dict[str, str] = {}
    for name_key, sha_key in (
        ("path", "sha256"),
        ("command", "command_sha256"),
        ("hook", "hook_sha256"),
    ):
        name, sha = digests.get(name_key), digests.get(sha_key)
        if name and sha:
            index.setdefault(_basename(name), sha)
    for key, value in digests.items():
        prefix, sep, base = key.partition("code_object:")
        if sep and not prefix and base:
            index.setdefault(base, value)
    return index


def _gzip_logs(case_dir: Path) -> None:
    """Compress every ``*.log`` in a published case dir to ``*.log.gz`` in place.

    ConSan runs with hook logging at debug level, so an uncompressed log is the
    bulkiest thing in a run area; gzip keeps a 7-run log window cheap on the
    data branch. ``mtime=0`` keeps the output byte-stable for identical input,
    so re-publishing an unchanged run does not churn the branch. Idempotent: a
    case dir that was already compressed has no ``*.log`` left to convert.
    """
    for src in sorted(case_dir.rglob("*.log")):
        if not src.is_file():
            continue
        dest = src.with_name(f"{src.name}.gz")
        with src.open("rb") as raw, dest.open("wb") as fh:
            with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as out:
                shutil.copyfileobj(raw, out)
        src.unlink()


def _prune_run_area_bulk(case_dir: Path) -> None:
    """Drop the heavy, regenerable parts of a run area outside the log window.

    Keeps ``sanitizer_report.json`` (and the landing page / manifests written
    next to it) so an older run still renders and still drills down; removes the
    logs, the copied inputs and the recipe copy. ``recipe.yaml`` is pruned even
    though it is tiny, so that a run area's contents depend only on its age and
    not on whether it was ever published inside the window -- ``env.json`` still
    records the recipe path, and the commit it ran at, to recover it from.

    The caller must re-render ``index.html`` afterwards: it enumerates what is
    on disk, so a page written before pruning would link files that are gone.
    """
    if not case_dir.is_dir():
        return
    for log in sorted(case_dir.rglob("*.log.gz")) + sorted(case_dir.rglob("*.log")):
        log.unlink(missing_ok=True)
    shutil.rmtree(case_dir / "inputs", ignore_errors=True)
    (case_dir / "recipe.yaml").unlink(missing_ok=True)
    # Remove the now-empty per-sanitizer log dirs so the file list is not a row
    # of empty folders.
    for sub in sorted(p for p in case_dir.iterdir() if p.is_dir()):
        if not any(sub.iterdir()):
            sub.rmdir()


def _read_text_or_empty(path: Path) -> str:
    """Read a repo file, or "" when it is missing/unreadable.

    A run area whose recipe cannot be read is still worth publishing; aborting
    the whole dashboard over it is not.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _publish_recipe_inputs(
    case_dir: Path, *, recipe_src: Path, repo_root: Path
) -> tuple[list[str], list[str]]:
    """Copy a case's recipe and its source-level inputs into the run area.

    Returns ``(copied, built)``: the recipe-relative paths copied under
    ``inputs/``, and the referenced CI-built artifacts that were deliberately
    skipped (recorded by digest by the caller). A recipe that cannot be read
    yields ``([], [])`` -- a run area missing its inputs is worth publishing,
    an aborted dashboard is not.
    """
    recipe_text = _read_text_or_empty(recipe_src)
    if not recipe_text:
        return [], []
    (case_dir / "recipe.yaml").write_text(recipe_text, encoding="utf-8")
    source_refs, built_refs = _recipe_fixture_refs(recipe_text)
    # Plus the sources the rebuild commands read. A recipe names only the built
    # artifact it consumes -- every daily-consan-* recipe lists a .hsaco and a
    # loader binary and nothing else -- so scanning the recipe text alone left
    # inputs/ empty for exactly the cases whose published commands read the most
    # (decision 10 asks for every input a reproduction needs).
    source_refs = list(dict.fromkeys(source_refs + rebuild_input_sources(built_refs)))
    recipe_dir = recipe_src.parent
    copied: list[str] = []
    inputs_root = (case_dir / "inputs").resolve()
    for ref in source_refs:
        src = recipe_dir / ref
        # Recipe text is repo-authored, but resolve defensively anyway: an input
        # must stay inside the recipe tree so a stray path cannot pull an
        # unrelated file into a published, publicly-served directory.
        try:
            resolved = src.resolve()
            resolved.relative_to(repo_root.resolve())
        except (OSError, ValueError):
            continue
        if not resolved.is_file():
            continue
        dest = case_dir / "inputs" / ref
        # ...and the same check on the *write*, not just the read.
        # ``_recipe_fixture_refs`` already drops parent segments; this is the
        # belt-and-braces half, because a validator that covers only one use of a
        # value is how the traversal got through in the first place.
        try:
            dest.resolve().relative_to(inputs_root)
        except (OSError, ValueError):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, dest)
        copied.append(ref)
    return copied, built_refs


def _human_size(num_bytes: int) -> str:
    """A compact file size for the run-area file table."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    value = float(num_bytes)
    for unit in ("KB", "MB", "GB"):
        value /= 1024.0
        if value < 1024.0:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} TB"


def run_area_files(case_dir: Path) -> list[tuple[str, int]]:
    """Every published file in a run area as ``(relative path, size)`` pairs.

    Enumerates what is actually on disk rather than what was expected, so the
    landing page never links a file that pruning removed or a run never
    produced. ``index.html`` itself is excluded (it is the page doing the
    listing). Sorted for a stable page across re-publishes.
    """
    if not case_dir.is_dir():
        return []
    files: list[tuple[str, int]] = []
    for path in sorted(case_dir.rglob("*")):
        if not path.is_file() or path.name == "index.html":
            continue
        files.append((path.relative_to(case_dir).as_posix(), path.stat().st_size))
    return files


# Both renderers of the rebuild section share this trailer: the workflow is what
# actually built the artifacts, so it -- not a paraphrase here -- is the source
# of truth for the exact invocation.
_REBUILD_NOTE = (
    "The exact invocations the nightly used are in "
    ".github/workflows/sanitizers-nightly.yml."
)

# Anchor on the landing page's "Artifacts not published" heading, so the Files
# caption can point a reader at the digest and rebuild command for an input it
# deliberately does not list.
_NOT_PUBLISHED_ID = "artifacts-not-published"

# The env a reproduction needs, as data, so a consumer of
# ``aorta.sanitizer_run_area/0.1`` can read the variables instead of parsing them
# out of a sentence. ``set_by`` says whether the operator must export it or
# ``aorta`` derives it from the recipe's policy block.
# ``consan_only`` keeps the manifest honest per case: the prose has always said
# these four are what "ConSan additionally requires", so listing them for a
# waitcheck reproduction would have the machine-readable field contradict it.
_REQUIRED_ENV: tuple[dict[str, Any], ...] = (
    {
        "var": "ROCJITSU_PREBUILT",
        "set_by": "operator",
        "purpose": "unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook",
        "consan_only": False,
    },
    {
        "var": "HSA_TOOLS_LIB",
        "set_by": "aorta",
        "purpose": "ConSan hook to load",
        "consan_only": True,
    },
    {
        "var": "HSA_TOOLS_DISABLE_REGISTER",
        "set_by": "aorta",
        "purpose": "must be 1 so the hook is not double-registered",
        "consan_only": True,
    },
    {
        "var": "RJ_CONSAN_MODE",
        "set_by": "aorta",
        "purpose": "ConSan detection mode",
        "consan_only": True,
    },
    {
        "var": "RJ_CONSAN_POLICY",
        "set_by": "aorta",
        "purpose": "ConSan reporting policy",
        "consan_only": True,
    },
)


def required_env_for(*, case: str, report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The env a *this* case's reproduction needs (pure).

    ConSan's four variables are dropped for a waitcheck-only case, so the
    machine-readable list cannot claim more than the prose does. Whether ConSan
    ran is taken from the report's own check/finding sanitizers where possible,
    falling back to the case name -- the naming convention every guardrail and
    survey case follows (``consan-*`` / ``*-consan``).
    """
    names: set[str] = set()
    checks = (report or {}).get("checks")
    for check in checks if isinstance(checks, list) else []:
        if not isinstance(check, dict):
            continue
        if isinstance(check.get("sanitizer"), str):
            names.add(check["sanitizer"].lower())
        for finding in check.get("findings") or []:
            if isinstance(finding, dict) and isinstance(finding.get("sanitizer"), str):
                names.add(finding["sanitizer"].lower())
    consan = (
        any("consan" in name for name in names)
        if names
        else "consan" in case.lower()
    )
    return [
        {key: value for key, value in item.items() if key != "consan_only"}
        for item in _REQUIRED_ENV
        if consan or not item["consan_only"]
    ]

def required_env_note(required_env: list[dict[str, Any]]) -> str:
    """Markdown prose for exactly the variables a manifest lists (pure).

    Generated from ``env["required_env"]`` rather than written beside it, so the
    sentence is per-case (a waitcheck area does not describe ConSan's variables)
    and a retained area's prose keeps describing what *it* recorded even after
    the module's own table changes.
    """
    operator = [item for item in required_env if item.get("set_by") == "operator"]
    derived = [item for item in required_env if item.get("set_by") == "aorta"]
    parts: list[str] = []
    for item in operator:
        parts.append(
            f"Set `{item['var']}` ({item.get('purpose', '')}) before running."
            if item.get("purpose")
            else f"Set `{item['var']}` before running."
        )
    if derived:
        names = ", ".join(f"`{item['var']}`" for item in derived)
        parts.append(
            f"`aorta` sets {names} for you from the recipe's policy block."
        )
    return " ".join(parts)


# The drill-down pages -- a run's landing page and each case's run area -- link
# one stylesheet instead of inlining ``_CSS``. ``_CSS`` is ~18KB and over 80% of
# such a page, and at ``--keep 30`` with five areas per run that is ~2.7MB of
# byte-identical CSS in the published tree: more than everything else in it
# combined, and well past the gzipped logs ``--keep-logs`` exists to bound. The
# root dashboard page keeps its inline copy -- there is only ever one of it, and
# it has to render standalone (including its empty-state stale banner).
RUN_AREA_CSS_REL = "assets/run-area.css"

# What those pages add on top of ``_CSS``. The union of both pages' needs, so one
# file serves both; an unused selector costs nothing.
_DRILLDOWN_CSS = """
  .wrap { max-width:900px; }
  /* `.cap`'s heading bar is `var(--accent, var(--text-3))`, and --accent is only
     defined on the root dashboard's kernel cards. These pages have no such
     ancestor, so every bar fell back to exactly the grey of the th text beneath
     it -- leaving nothing to tell a section heading from a table header. Scoped
     here rather than on `.cap` so the root dashboard keeps its per-kind hues. */
  .panel { --accent:var(--blue); }
  .note { margin:0 0 14px; color:var(--text-3); font-size:13px; }
  .note a { color:#7CB0F8; }
  /* One rebuild recipe per artifact: path, what it is, the commands as a real
     block, then the caveat. These pages had no multi-line code affordance at
     all, which is how the commands ended up as inline <code> runs inside a
     prose sentence. pre-wrap so a long command is never hidden off-screen, and
     so a soft wrap does not become a newline when the block is copied. */
  .rb { margin:0 0 18px; }
  .rb:last-of-type { margin-bottom:8px; }
  .rb .path { margin:0 0 4px; font-family:var(--mono); font-size:13.5px;
    color:var(--text-1); word-break:break-all; }
  .rb .what { margin:0 0 8px; color:var(--text-3); font-size:13px; }
  .rb .caveat { margin:0; color:var(--text-3); font-size:13px; }
  .rb pre { margin:0 0 8px; background:var(--sunken); border:1px solid var(--border);
    border-radius:8px; padding:11px 13px; }
  .rb pre code { display:block; font-family:var(--mono); font-size:13px;
    color:#A5D6FF; white-space:pre-wrap; overflow-wrap:anywhere; }
"""


def run_area_stylesheet() -> str:
    """The stylesheet the drill-down pages link, as text (pure)."""
    return f"{_CSS}{_DRILLDOWN_CSS}"


def write_shared_assets(out_dir: Path) -> Path:
    """Write the drill-down stylesheet under ``out_dir`` and return its path.

    Byte-stable for unchanged input, like the gzipped logs, so re-publishing the
    retained history does not churn the data branch.
    """
    path = out_dir / RUN_AREA_CSS_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run_area_stylesheet(), encoding="utf-8")
    return path


def _md_code_to_html(text: str) -> str:
    """Render a shared Markdown prose constant's ``backticks`` as ``<code>``.

    Lets the page and REPRODUCE.md render the same constant natively rather than
    keeping two hand-synced copies of the same sentence. Escaping happens first,
    so the split can never introduce markup from the text itself.
    """
    parts = _esc(text).split("`")
    return "".join(
        part if index % 2 == 0 else f"<code>{part}</code>"
        for index, part in enumerate(parts)
    )

# Which source each unpublished fixture is built from, and how. Paths are
# recipe-relative, matching how a recipe names them.
#
# These are per-artifact rather than one hint per directory because the commands
# are NOT interchangeable, and a generic "hipcc it" hint produces a file whose
# digest does not match the one recorded beside it:
#
# * an ``--genco`` object is a raw code object on some ROCm builds and a
#   clang-offload bundle on others, so it must be unbundled conditionally -- the
#   recorded digest is of the unbundled object the loader opens;
# * the GEMM objects are *extracted* from the shipped Tensile libraries by
#   prepare_gemm_isa.py, never compiled from a .hip source;
# * every consan_load / lds_dispatch binary is one source compiled with a define
#   naming the code object it loads, so omitting the define builds the wrong
#   binary rather than failing loudly.
_GENCO_ISA_SOURCES = {
    "fixtures/isa/lds.hsaco": "fixtures/kernels/lds_reduce.hip",
    "fixtures/isa/tiny.hsaco": "fixtures/kernels/tiny_vecadd.hip",
}
# ref -> (source, define name or None, extra flags). A define binds the binary to
# its object. ``extra`` is verbatim from the workflow: the two guardrail repro
# binaries are built ``-O1 -g`` and the loader binaries are not, and optimisation
# /debug flags change the executable -- so dropping them would not reproduce the
# recorded ``command_sha256`` any more than dropping the define would.
_BIN_SOURCES: dict[str, tuple[str, str | None, str]] = {
    "fixtures/bin/consan_lds_race": (
        "fixtures/repro/consan_lds_race.hip",
        None,
        "-O1 -g",
    ),
    "fixtures/bin/consan_lds_race_2wave": (
        "fixtures/repro/consan_lds_race_2wave.hip",
        None,
        "-O1 -g",
    ),
    "fixtures/bin/consan_tiny_load": ("fixtures/kernels/consan_load.hip", "OBJECT", ""),
    "fixtures/bin/consan_gemm_load": ("fixtures/kernels/consan_load.hip", "OBJECT", ""),
    "fixtures/bin/lds_dispatch": ("fixtures/kernels/lds_dispatch.hip", "LDS_HSACO", ""),
}
# The code object each defined binary loads, so the define can name it.
_BIN_OBJECT = {
    "fixtures/bin/consan_tiny_load": "fixtures/isa/tiny.hsaco",
    "fixtures/bin/consan_gemm_load": "fixtures/isa/consan_gemm_f32.hsaco",
    "fixtures/bin/lds_dispatch": "fixtures/isa/lds.hsaco",
}
# Recipe-relative paths (how a recipe names a fixture, and how this run area
# records one) are NOT runnable paths. REPRODUCE.md puts the reader at the repo
# root after ``cd aorta``, where there is no top-level ``fixtures/`` -- the tree
# lives here, which is also what the nightly's own ``FIXTURES`` variable is set
# to. Every emitted command is prefixed with this so it can be pasted and run;
# only the recorded ``path`` stays recipe-relative.
_FIXTURES_FROM_ROOT = "recipes/sanitizers/fixtures"

# The keys every ``rebuild_plan`` entry carries, and that both renderings index
# directly. ``plan_from_env`` checks a *stored* plan against these before
# trusting it, since that one comes off the data branch rather than from here.
_REBUILD_KEYS = frozenset({"path", "what", "commands", "caveat"})


def _runnable(ref: str) -> str:
    """A recipe-relative fixture path rewritten to run from the repo root."""
    return f"{_FIXTURES_FROM_ROOT}{ref[len('fixtures'):]}" if ref.startswith("fixtures") else ref


_GEMM_ISA_CSV = "fixtures/gemm_shapes_unique.csv"
_GEMM_ISA_SCRIPT = (
    "python scripts/sanitizers/prepare_gemm_isa.py "
    f"--csv {_runnable(_GEMM_ISA_CSV)} --out {_FIXTURES_FROM_ROOT}/isa"
)


def rebuild_input_sources(built_refs: list[str]) -> list[str]:
    """Recipe-relative source files the rebuild commands for these artifacts read.

    Pure, and derived from the same tables ``rebuild_plan`` renders from, so the
    inputs a run area copies cannot disagree with the commands it publishes.

    Needed because a recipe names only the built artifact it *consumes*: every
    ``daily-consan-*`` recipe lists a ``.hsaco`` and a loader binary and nothing
    else, so scanning the recipe text alone left ``inputs/`` empty for exactly the
    cases whose published commands read the most. A binary's loaded code object is
    folded in too, since rebuilding the binary means rebuilding that object first
    -- ``consan_gemm_load`` needs ``consan_load.hip`` *and* the GEMM shape CSV its
    object is extracted with.
    """
    refs = list(built_refs) + [
        _BIN_OBJECT[ref] for ref in built_refs if ref in _BIN_OBJECT
    ]
    sources: list[str] = []
    for ref in dict.fromkeys(refs):
        genco = _GENCO_ISA_SOURCES.get(ref)
        if genco:
            sources.append(genco)
        elif f"{ref}/".startswith("fixtures/isa/") or ref == "fixtures/isa":
            # Every isa/ artifact not built by --genco is extracted by
            # prepare_gemm_isa.py, which reads the shape CSV.
            sources.append(_GEMM_ISA_CSV)
        if ref in _BIN_SOURCES:
            sources.append(_BIN_SOURCES[ref][0])
    return list(dict.fromkeys(sources))

# Every rebuild starts here, because the tool none of them can run without is not
# on PATH in the image that built the artifact: the ROCm container exports only
# ``/opt/rocm/bin``, while ``clang-offload-bundler`` lives in the LLVM bindir. hipcc
# shells out to it, ``prepare_gemm_isa.py`` requires it (``shutil.which``), and the
# genco branch invokes it directly -- so a command that names it by bare name fails
# exactly as the nightly's own fixture build once did ("clang-offload-bundler not
# found on PATH"). Byte-identical to the export in sanitizers-nightly.yml, which
# ``test_rebuild_commands_export_the_rocm_llvm_path`` cross-checks.
_ROCM_LLVM_PATH_EXPORT = (
    'export PATH="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}/lib/llvm/bin:${PATH}"'
)


def rebuild_plan(built_refs: list[str], *, target: str) -> list[dict[str, Any]]:
    """How to rebuild each unpublished artifact, as data (pure).

    ``env.json`` carries this so a consumer of ``aorta.sanitizer_run_area/0.1``
    can read the exact commands rather than parse them out of a sentence, and
    both prose renderings are generated from it by ``_rebuild_hints`` -- so the
    landing page, REPRODUCE.md and the manifest cannot disagree about how an
    artifact was built.

    Each entry is ``{"path", "what", "commands", "caveat"}``. ``path`` stays
    recipe-relative, matching how a recipe names the artifact and how
    ``artifacts_not_published`` records it; ``commands`` are rewritten to run from
    the repo root, which is where REPRODUCE.md leaves the reader, and open with the
    two prerequisites a fresh shell lacks -- the ROCm LLVM bindir on ``PATH`` and
    the gitignored output directory. They are the nightly's own invocations
    (see the tables above): per-artifact and not interchangeable, because a
    rebuild that differs from them will not reproduce the SHA-256 recorded beside
    the artifact. An unrecognised reference yields commands ``[]`` rather than a
    guess.

    Anything a consumer must branch on is encoded *in* a command rather than left
    to ``caveat`` prose, so the list can be executed as-is.
    """

    def prelude(sub: str) -> list[str]:
        """What every rebuild needs before its first tool call, shell order."""
        return [_ROCM_LLVM_PATH_EXPORT, f"mkdir -p {_FIXTURES_FROM_ROOT}/{sub}"]

    plan: list[dict[str, Any]] = []
    for ref in built_refs:
        out = _runnable(ref)
        source = _GENCO_ISA_SOURCES.get(ref)
        if source:
            plan.append({
                "path": ref,
                "what": f"a gfx code object built from {source}",
                "commands": [
                    *prelude("isa"),
                    f"hipcc --genco --offload-arch={target} {_runnable(source)} -o tmp.o",
                    # One command, because which branch is correct depends on the
                    # ROCm build: --genco emits a raw object on some and a bundle
                    # on others, and the recorded digest is of the unbundled one.
                    # ...and clean up after itself. The workflow's genco_object
                    # builds into a mktemp file and rm -f's it; this runs in the
                    # reader's repo root, so without the trailing rm it drops an
                    # untracked tmp.o next to pyproject.toml. `&&` keeps it for
                    # inspection if the conversion failed.
                    "if head -c 24 tmp.o | grep -qF __CLANG_OFFLOAD_BUNDLE__; then "
                    "clang-offload-bundler --type=o --unbundle --input=tmp.o "
                    f"--targets=hipv4-amdgcn-amd-amdhsa--{target} --output={out}; "
                    f"else cp tmp.o {out}; fi && rm -f tmp.o",
                ],
                "caveat": (
                    "The conditional matters: --genco emits a raw code object on "
                    "some ROCm builds and a clang-offload bundle on others, and the "
                    "recorded digest is of the unbundled object."
                ),
            })
        elif ref == "fixtures/isa/consan_gemm_f32.hsaco":
            plan.append({
                "path": ref,
                "what": "one representative heavy f32 SS GEMM code object",
                "commands": [
                    *prelude("isa"),
                    f"{_GEMM_ISA_SCRIPT} --top-n 0 --consan-object {out}",
                ],
                "caveat": (
                    "Extracted from the shipped Tensile libraries, never compiled "
                    "from a .hip source."
                ),
            })
        elif f"{ref}/".startswith("fixtures/isa/") or ref == "fixtures/isa":
            # sol_<n>.hsaco, or a bare isa_dir reference covering them.
            plan.append({
                "path": ref,
                "what": "the per-transpose f32 GEMM code objects (sol_<n>.hsaco)",
                "commands": [*prelude("isa"), f"{_GEMM_ISA_SCRIPT} --top-n 3"],
                "caveat": (
                    "Extracted from the shipped Tensile libraries, never compiled "
                    "from a .hip source."
                ),
            })
        elif ref in _BIN_SOURCES:
            source, define, extra = _BIN_SOURCES[ref]
            flags = f"--offload-arch={target}"
            if extra:
                flags += f" {extra}"
            if define:
                # $(pwd) is the repo root here, so the define must carry the
                # root-relative path to the object, not the recipe-relative one.
                flags += f' -D{define}="\\"$(pwd)/{_runnable(_BIN_OBJECT[ref])}\\""'
            plan.append({
                "path": ref,
                "what": f"a host repro binary built from {source}",
                "commands": [
                    *prelude("bin"),
                    f"hipcc {flags} {_runnable(source)} -o {out}",
                ],
                "caveat": (
                    f"The -D{define} define is required: it is how the binary learns "
                    "which code object to load."
                    if define
                    else ""
                ),
            })
        else:
            plan.append({"path": ref, "what": "", "commands": [], "caveat": ""})
    return plan


def plan_from_env(env: dict[str, Any], built_refs: list[str]) -> list[dict[str, Any]]:
    """A run area's rebuild plan, preferring the one it persisted (pure).

    Every retained area is re-rendered nightly while its ``env.json`` is
    preserved, so recomputing the commands from the module's current tables would
    rewrite historical instructions and leave the prose contradicting the
    manifest sitting beside it. Fall back to recomputation only for an area
    published before the field existed -- or one whose stored entries do not
    carry the keys both renderers read. Both renderers index those keys directly,
    and this runs over a manifest read off the data branch, so accepting a
    partial entry here would turn one hand-edited or half-written ``env.json``
    into a ``KeyError`` that fails the whole dashboard render rather than one
    area's rebuild section.
    """
    stored = env.get("rebuild")
    if isinstance(stored, list) and all(
        isinstance(item, dict) and _REBUILD_KEYS <= item.keys() for item in stored
    ):
        return stored
    return rebuild_plan(built_refs, target=str(env.get("target") or "gfx950"))


def _rebuild_hints(plan: list[dict[str, Any]]) -> list[str]:
    """One prose rebuild hint per artifact, rendered from a ``rebuild_plan`` (pure).

    Markdown for REPRODUCE.md, which uses it verbatim. A reference with no known
    recipe degrades to its bare path rather than a plausible-looking wrong
    command.

    The landing page does NOT go through here: flattening the four fields into
    one sentence is right for Markdown and wrong for HTML, where it produced a
    prose paragraph with the commands as inline ``<code>`` runs joined by
    prose ``;`` -- nothing on the page a reader could select and run. That page
    renders the same plan structurally instead (``_rebuild_section_html``).
    """
    hints: list[str] = []
    for entry in plan:
        if not entry["commands"]:
            hints.append(str(entry["path"]))
            continue
        commands = "; ".join(f"`{command}`" for command in entry["commands"])
        hint = f"{entry['path']} -- {entry['what']}: {commands}."
        if entry["caveat"]:
            hint += f" {entry['caveat']}"
        hints.append(hint)
    return hints


def _rebuild_section_html(plan: list[dict[str, Any]]) -> str:
    """The landing page's rebuild section, rendered from ``rebuild_plan`` (pure).

    ``rebuild_plan`` already separates ``path`` / ``what`` / ``commands`` /
    ``caveat`` -- its docstring's whole point is that a consumer can read the
    commands "rather than parse them out of a sentence", and that anything to
    branch on is encoded *in* a command "so the list can be executed as-is".
    Rendering that structure keeps the same promise for a human: the commands
    land in one block, one per line, with no prose inside it and no sentence
    period stuck to the final path.

    Each artifact is a titled block rather than a list item, so a case with one
    unpublished artifact no longer renders a one-item bullet list.
    """
    if not plan:
        return ""
    blocks: list[str] = []
    for entry in plan:
        commands = [str(command) for command in entry.get("commands") or []]
        parts = [f'<p class="path">{_esc(str(entry["path"]))}</p>']
        what = entry.get("what")
        if what:
            parts.append(f'<p class="what">{_esc(str(what))}</p>')
        if commands:
            body = "\n".join(_esc(command) for command in commands)
            parts.append(f"<pre><code>{body}</code></pre>")
        else:
            # Same contract as ``_rebuild_hints``: an unrecognised reference is
            # named and left at that, never given a plausible-looking guess.
            parts.append(
                '<p class="what">No rebuild command is recorded for this reference.</p>'
            )
        caveat = entry.get("caveat")
        if caveat:
            parts.append(f'<p class="caveat">{_esc(str(caveat))}</p>')
        blocks.append(f'<div class="rb">{"".join(parts)}</div>')
    return (
        '<p class="cap">Rebuild the fixtures</p>'
        + "".join(blocks)
        + f'<p class="note">{_esc(_REBUILD_NOTE)}</p>'
    )


def _files_caption(env: dict[str, Any], files: list[tuple[str, int]]) -> str:
    """The Files table's caption, as HTML (pure).

    "Every file published for this case" is true and, on its own, misleading:
    for a ``source.kind: kernel`` recipe the one input the recipe names is
    CI-built, so it is recorded under *Artifacts not published* by digest
    instead of being copied here. Nothing linked the two sections, so a reader
    could not tell a deliberate omission from a missing file. Say it, and point
    at where the digest and rebuild command actually are.
    """
    if not env.get("logs_published", True):
        base = (
            "Every file published for this case. Its logs, recipe copy and inputs "
            "were pruned for falling outside the nightly's log-retention window."
        )
    elif any(rel.endswith(".log.gz") for rel, _size in files):
        base = "Every file published for this case. Logs are gzipped."
    else:
        base = "Every file published for this case."
    missing = env.get("artifacts_not_published") or []
    if not missing:
        return _esc(base)
    count = len(missing)
    noun = "input" if count == 1 else "inputs"
    verb = "is" if count == 1 else "are"
    return (
        f"{_esc(base)} It is not the whole recipe: {count} required {noun} {verb} "
        f'CI-built and excluded by design, listed under <a href="#{_NOT_PUBLISHED_ID}">'
        f"artifacts not published</a> with a SHA-256 and a command to rebuild it."
    )


def build_case_env(
    *,
    case: str,
    cls: str,
    recipe: str,
    command: str,
    meta: dict[str, Any],
    summary: dict[str, Any],
    report: dict[str, Any] | None,
    built_refs: list[str],
    inputs: list[str],
    logs: bool = True,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The machine-readable provenance manifest for one run area (pure).

    Deliberately narrow: it states only what the publishing job genuinely knows
    -- the run identity, the recipe and command, what was observed, and the
    digests already present in the report. It is NOT an env-probe snapshot;
    the nightly never runs one, so anything shaped like that schema here would
    be fabricated fields rather than captured ones.

    ``provenance`` is passed in rather than read from the environment here, so a
    caller re-rendering an *older* run's area cannot accidentally relabel it with
    the current job's image and bundle; an empty mapping omits those fields.

    ``logs`` records whether this area still carries the bulk it was published
    with, or has been pruned back to the report for falling outside the
    ``--keep-logs`` window. Both renderings read it, so neither promises files
    that a pruned area no longer has.
    """
    digests = report_digests(report)
    by_basename = artifact_digest_index(digests)
    env: dict[str, Any] = {
        "schema": RUN_AREA_ENV_SCHEMA,
        "case": case,
        "class": cls,
        "recipe": recipe,
        "command": command,
        "run": meta.get("run", ""),
        "commit": meta.get("commit_full") or meta.get("commit", ""),
        "date": meta.get("date", ""),
        "target": meta.get("gpu", ""),
        "observed": {
            "verdict": summary.get("verdict"),
            "execution": summary.get("execution"),
            "reason": (summary.get("primary") or {}).get("reason"),
            "findings": summary.get("findings", 0),
        },
        "inputs": inputs,
        "logs_published": logs,
        # Machine-readable counterparts of the two things REPRODUCE.md explains in
        # prose, so a consumer of this schema does not have to parse English to
        # learn which variables to export or how to rebuild an artifact. Both
        # renderings are generated from these, not written alongside them.
        "required_env": required_env_for(case=case, report=report),
        "rebuild": rebuild_plan(built_refs, target=str(meta.get("gpu") or "")),
        "artifacts_not_published": [
            {"path": ref, "sha256": by_basename.get(_basename(ref))} for ref in built_refs
        ],
        "digests": digests,
    }
    if meta.get("run_url"):
        env["run_url"] = meta["run_url"]
    for key, value in (provenance or {}).items():
        if value:
            env[key] = value
    return env


def build_reproduce_md(env: dict[str, Any], *, built_refs: list[str]) -> str:
    """REPRODUCE.md for one run area (pure).

    The downloadable companion to the landing page rather than a strict twin of
    it: the two share the run identity, the reproduce command and the rebuild
    section (via ``_rebuild_hints``), and this file additionally carries the
    required env and the recorded digests that the page links it for.
    """
    observed = env.get("observed") or {}
    # Point at the file table rather than restating an inventory here: what a run
    # area holds depends on its age (see ``--keep-logs``) and on whether it was
    # published by the job that ran it, and a fixed sentence would go stale.
    pruned = (
        ""
        if env.get("logs_published", True)
        else " Its logs, recipe copy and inputs were pruned for falling outside the "
        "nightly's log-retention window."
    )
    lines = [
        f"# Reproduce `{env['case']}`",
        "",
        f"Sanitizer case `{env['case']}` from run `{env.get('run', '')}` of the AORTA "
        "sanitizer nightly. This directory is the run area for that one case: its "
        "report, the sanitizer output the verdict came from, and the provenance "
        f"below.{pruned} See `index.html` for every file actually published here.",
        "",
        "## Run",
        "",
        f"- Commit: `{env.get('commit', '')}`",
        f"- Date: {env.get('date', '')}",
        f"- Target: `{env.get('target', '')}`",
        f"- Class: {env.get('class', '')} "
        + ("(gated guardrail)" if env.get("class") == "guardrail" else "(observed-only, non-gating)"),
    ]
    if env.get("run_url"):
        lines.append(f"- Workflow run: {env['run_url']}")
    if env.get("container_image"):
        lines.append(f"- Container image: `{env['container_image']}`")
    if env.get("rocjitsu_commit"):
        bundle = f"- rocjitsu bundle: `{env['rocjitsu_commit']}`"
        if env.get("rocjitsu_run_url"):
            bundle += f" ({env['rocjitsu_run_url']})"
        lines.append(bundle)
    lines += [
        "",
        "## Observed",
        "",
        f"- Verdict: `{observed.get('verdict') or _DASH}`",
        f"- Execution: `{observed.get('execution') or _DASH}`",
        f"- Findings: {observed.get('findings', 0)}",
    ]
    if observed.get("reason"):
        lines.append(f"- Reason: `{observed['reason']}`")
    # Ordered so the document runs top to bottom. The sweep command comes *last*:
    # the fixtures it needs are gitignored, so on a fresh clone it fails before the
    # reader would ever reach a rebuild section printed after it.
    lines += [
        "",
        "## Run it yourself",
        "",
        "First, the checkout this run used:",
        "",
        "```",
        "git clone https://github.com/ROCm/aorta && cd aorta",
        f"git checkout {env.get('commit', '')}",
        "pip install -e .",
        "```",
        "",
    ]
    hints = _rebuild_hints(plan_from_env(env, built_refs))
    if hints:
        lines += [
            "Then rebuild the CI-built fixtures, which are not published here (see "
            "'Artifacts not published' below). Run these from the repo root, the "
            "directory the clone above leaves you in:",
            "",
            *(f"- {hint}" for hint in hints),
            "",
            _REBUILD_NOTE,
            "",
        ]
    env_note = required_env_note(env.get("required_env") or [])
    if env_note:
        lines += [env_note, ""]
    lines += [
        "Finally, the sweep itself:",
        "",
        "```",
        f"{env.get('command', '')}",
        "```",
        "",
    ]
    if env.get("artifacts_not_published"):
        lines += [
            "## Artifacts not published",
            "",
            "These are CI-built and too large to publish for every retained run. "
            "Rebuild them as above and check the digest matches.",
            "",
            "| Path | SHA-256 |",
            "|---|---|",
        ]
        for item in env["artifacts_not_published"]:
            lines.append(f"| `{item['path']}` | `{item.get('sha256') or _DASH}` |")
        lines.append("")
    if env.get("digests"):
        lines += ["## Recorded digests", "", "| Key | Value |", "|---|---|"]
        for key, value in sorted(env["digests"].items()):
            lines.append(f"| `{key}` | `{value}` |")
        lines.append("")
    lines += [
        "## Files here",
        "",
        "See `index.html` for the browsable list. "
        + ("Logs are gzipped; " if env.get("logs_published", True) else "")
        + "`sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document "
        "the dashboard renders from.",
        "",
    ]
    return "\n".join(lines)


def case_dir_up(case_dir: Path, out_dir: Path) -> tuple[str, str]:
    """Relative paths from a case dir back to ``(dashboard root, its run dir)``.

    Derived from the published depth rather than hardcoded per caller, because
    the two tabs sit at different depths -- ``runs/<id>/<case>/`` for a guardrail
    case and ``runs/<id>/survey/<case>/`` for a survey one -- and an off-by-one
    here is a broken back-link on a page nothing else links to.
    """
    try:
        depth = len(case_dir.resolve().relative_to(out_dir.resolve()).parts)
    except ValueError:
        return "../../", "../"
    return "../" * depth, "../" * max(depth - 2, 1)


def build_case_index_html(
    env: dict[str, Any],
    files: list[tuple[str, int]],
    *,
    built_refs: list[str],
    up: str,
    run_up: str = "../",
    title: str = "Sanitizers Nightly",
) -> str:
    """The run area's landing page: what this case was, and every file to download.

    GitHub Pages does not auto-index a directory, so this page is what makes the
    card footer's directory link resolve at all. ``up`` / ``run_up`` are the
    relative paths back to the dashboard root and to this case's run area (case
    dirs sit at different depths on the two tabs; see ``case_dir_up``).
    """
    observed = env.get("observed") or {}
    rows = "".join(
        f'<tr><td class=mono><a href="{_esc(rel)}">{_esc(rel)}</a></td>'
        f"<td class=num>{_esc(_human_size(size))}</td></tr>"
        for rel, size in files
    ) or '<tr><td class=empty colspan=2>no files published for this case</td></tr>'
    facts = [
        ("Run", _esc(str(env.get("run", "")))),
        ("Commit", _esc(str(env.get("commit", "")))),
        ("Date", _esc(str(env.get("date", "")))),
        ("Target", _esc(str(env.get("target", "")))),
        ("Recipe", _esc(str(env.get("recipe", "")))),
        ("Execution", _esc(str(observed.get("execution") or _DASH))),
        ("Findings", _esc(str(observed.get("findings", 0)))),
    ]
    if env.get("container_image"):
        facts.append(("Container", _esc(str(env["container_image"]))))
    if env.get("rocjitsu_commit"):
        facts.append(("rocjitsu bundle", _esc(str(env["rocjitsu_commit"]))))
    kv = "".join(_fact_html(label, value, mono=True) for label, value in facts)
    reason = observed.get("reason")
    reason_html = (
        f'<div class="observation {_tone_for(observed.get("verdict"))}">'
        f'<span class="lbl">Reason</span><span class="msg">{_esc(str(reason))}</span></div>'
        if reason
        else ""
    )
    steps_html = _rebuild_section_html(plan_from_env(env, built_refs))
    not_published = env.get("artifacts_not_published") or []
    np_html = ""
    if not_published:
        np_rows = "".join(
            f'<tr><td class=mono>{_esc(str(item["path"]))}</td>'
            f'<td class="mono wrap-any">{_esc(str(item.get("sha256") or "")) or "&mdash;"}</td></tr>'
            for item in not_published
        )
        np_html = (
            f'<p class="cap" id="{_NOT_PUBLISHED_ID}">Artifacts not published</p>'
            '<div class="table-wrap"><table>'
            "<thead><tr><th>Path</th><th>SHA-256</th></tr></thead>"
            f"<tbody>{np_rows}</tbody></table></div>"
        )
    # #384 requires the code-object SHA-256 on this page, and the artifacts table
    # above cannot carry it on its own: a recipe that names only a bare
    # ``isa_dir: fixtures/isa`` has one row and no per-object digest, so without
    # this section those digests would exist only in env.json / REPRODUCE.md.
    digests_html = ""
    digests = env.get("digests") or {}
    if isinstance(digests, dict) and digests:
        digest_rows = "".join(
            f"<tr><td class=mono>{_esc(str(key))}</td>"
            f'<td class="mono wrap-any">{_esc(str(value))}</td></tr>'
            for key, value in sorted(digests.items())
        )
        digests_html = (
            '<p class="cap">Recorded digests</p><div class="table-wrap"><table>'
            "<thead><tr><th>Key</th><th>Value</th></tr></thead>"
            f"<tbody>{digest_rows}</tbody></table></div>"
        )
    run_link = (
        f'<a href="{_esc(str(env["run_url"]))}">workflow run</a>'
        if env.get("run_url")
        else "<span></span>"
    )
    return (
        "<!doctype html>\n"
        "<html lang=en><head><meta charset=utf-8>\n"
        '<meta name=viewport content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)} &middot; {_esc(str(env.get('case', '')))}</title>\n"
        f'<link rel=stylesheet href="{_esc(up)}{RUN_AREA_CSS_REL}"></head>\n'
        "<body><div class=wrap>\n"
        f'<div class=navrow><span><a href="{_esc(up)}">&larr; back to sanitizer dashboard</a>'
        f' &middot; <a href="{_esc(run_up)}">run {_esc(str(env.get("run", "")))}</a></span>\n'
        f"{run_link}</div>\n"
        '<div class=topbar>\n'
        '<header class="page-header"><div class="brand-tile">'
        f"{_svg(_ICON_FILE, size=24, width=2)}</div><div>\n"
        f"<h1>{_esc(str(env.get('case', '')))}</h1>\n"
        # Not "everything needed to reproduce this case locally": for a
        # kernel-source recipe the one input the recipe names is CI-built and
        # recorded by digest rather than published (see ``_files_caption``).
        '<p class="subtitle">Run area &mdash; the reproduce command, what this '
        "case observed, and every file published for it</p></div></header>\n"
        "</div>\n"
        '<section class="panel">\n'
        f"<h2>Case {_verdict_html(observed.get('verdict') or _DASH)}</h2>\n"
        f'<div class="kv">{kv}</div>\n'
        f"{reason_html}\n"
        # The label sits outside the box, so selecting the box yields a command
        # that runs rather than one prefixed with the word "Reproduce".
        '<p class="cap">Reproduce</p>'
        f'<div class="repro"><code>{_esc(str(env.get("command", "")))}</code></div>\n'
        f'<p class="note">{_md_code_to_html(required_env_note(env.get("required_env") or []))}</p>\n'
        f"{steps_html}{np_html}{digests_html}"
        "</section>\n"
        '<section class="panel"><h2>Files</h2>\n'
        f'<p class="note">{_files_caption(env, files)}</p>\n'
        '<div class="table-wrap"><table>\n'
        "<thead><tr><th>File</th><th class=num>Size</th></tr></thead>\n"
        f"<tbody>{rows}</tbody>\n"
        "</table></div></section>\n"
        "</div></body></html>\n"
    )


def write_case_run_area(
    case_dir: Path,
    *,
    case: str,
    cls: str,
    recipe_stem: str,
    command: str,
    meta: dict[str, Any],
    summary: dict[str, Any],
    report: dict[str, Any] | None,
    repo_root: Path,
    up: str,
    run_up: str = "../",
    logs: bool = True,
    provenance: dict[str, str] | None = None,
    current: bool = True,
) -> None:
    """Materialise one run area: recipe + inputs + REPRODUCE.md + env.json + index.

    Called after the case's report and logs are already in ``case_dir``, so the
    file listing on the landing page reflects exactly what was published.

    ``current`` means this run is the one the publishing job just staged, and is
    what gates the two things only that job can source correctly: the ``recipe``
    and its inputs, which come from *its* checkout, and ``provenance``, which
    describes *its* container and sanitizer build. Re-rendering an older run's
    area must leave both alone -- copying today's recipe into a three-week-old
    area would replace the recipe as it ran with whatever is on ``main`` now,
    beside an ``env.json`` still telling the reader to check out that run's
    commit. Older areas keep what they captured while they were current (see
    ``refresh_published_case_area``); this path is the fallback for a run
    published before run areas existed, which can only state what its own
    ``meta.json`` and report still say.

    ``logs=False`` (a run outside the ``--keep-logs`` window) still writes the
    manifests and the page -- which stay useful and cost bytes, not megabytes --
    and both renderings then describe the area as pruned rather than promising
    files it does not have.
    """
    recipe_rel = f"recipes/sanitizers/{recipe_stem}.yaml"
    inputs: list[str] = []
    built_refs: list[str] = []
    if current and logs:
        inputs, built_refs = _publish_recipe_inputs(
            case_dir, recipe_src=repo_root / recipe_rel, repo_root=repo_root
        )
    else:
        # The referenced built artifacts are still worth naming by digest even
        # when nothing is copied; the recipe is only read, never published.
        _, built_refs = _recipe_fixture_refs(
            _read_text_or_empty(repo_root / recipe_rel)
        )
    env = build_case_env(
        case=case,
        cls=cls,
        recipe=recipe_rel,
        command=command,
        meta=meta,
        summary=summary,
        report=report,
        built_refs=built_refs,
        inputs=inputs,
        logs=logs,
        provenance=provenance if current else None,
    )
    (case_dir / "env.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (case_dir / "REPRODUCE.md").write_text(
        build_reproduce_md(env, built_refs=built_refs), encoding="utf-8"
    )
    (case_dir / "index.html").write_text(
        build_case_index_html(
            env,
            run_area_files(case_dir),
            built_refs=built_refs,
            up=up,
            run_up=run_up,
        ),
        encoding="utf-8",
    )


def refresh_published_case_area(case_dir: Path, out_dir: Path, *, logs: bool) -> bool:
    """Re-render an already-published run area from its own persisted manifest.

    Every retained run is re-rendered on every nightly, but only the run being
    published now can be described from the live job. So an older area keeps the
    ``env.json`` it captured while it was current -- its commit, its container
    image, its rocjitsu bundle -- and only its rendering is rebuilt. That rebuild
    is what keeps the page honest once the area has been pruned: the file table
    enumerates what is on disk, so a page written before pruning would go on
    linking logs that are gone.

    Call after gzipping or pruning, with ``logs`` saying which happened. Returns
    ``False`` when there is no manifest to rebuild from (a run published before
    run areas existed), leaving the caller to write a fresh one.
    """
    env = _load(case_dir / "env.json")
    if not isinstance(env, dict):
        return False
    if not logs:
        # The manifest described an area that still had its bulk. Restate it for
        # what is left, so it does not name inputs that are no longer there.
        env["inputs"] = []
    # Pruning is one-way: the files were deleted from the published history and
    # cannot be recovered from it, so raising --keep-logs later must not flip this
    # back to true and have the page and REPRODUCE.md offer logs that are gone.
    # Only re-staging from the source sweep can restore them, which goes through
    # write_case_run_area rather than here.
    env["logs_published"] = bool(env.get("logs_published", True)) and logs
    built_refs = [
        str(item["path"])
        for item in env.get("artifacts_not_published") or []
        if isinstance(item, dict) and item.get("path")
    ]
    up, run_up = case_dir_up(case_dir, out_dir)
    (case_dir / "env.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (case_dir / "REPRODUCE.md").write_text(
        build_reproduce_md(env, built_refs=built_refs), encoding="utf-8"
    )
    (case_dir / "index.html").write_text(
        build_case_index_html(
            env,
            run_area_files(case_dir),
            built_refs=built_refs,
            up=up,
            run_up=run_up,
        ),
        encoding="utf-8",
    )
    return True


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
    link = f' <a href="{_esc(url)}">view failed run</a>' if url else ""
    when_txt = f" ({_esc(when)})" if when else ""
    return (
        '<div class="stale">'
        "<span>&#9888;</span><span><strong>Stale.</strong> "
        f"Latest sanitizer nightly{run_txt} did not complete successfully "
        f"({_esc(conclusion)}) &mdash; the data below may be stale.{link}{when_txt}"
        "</span></div>"
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


# Verdict badge classes, keyed on the observed verdict string. ``fail`` and
# ``error`` get distinct class names but share one red in CSS, matching the
# existing summary.md wording -- splitting the hues later is a CSS-only change.
# not_checked / unknown / anything unmapped stays neutral grey.
_VERDICT_CLASS: dict[str, str] = {
    "error": "error", "fail": "fail", "warn": "warn", "pass": "pass",
}


def _verdict_html(verdict: Any) -> str:
    """Render an observed sanitizer verdict as a tinted badge.

    Used by *both* tabs. Verdict colour is descriptive, never a health signal:
    on the guardrail tab a red ``fail`` sitting beside a solid green "Expected
    outcome" pill is a positive control behaving as designed. The solid baseline
    pill is what carries regression health, and it outranks this badge visually.
    """
    text = str(verdict)
    cls = _VERDICT_CLASS.get(text.strip().lower(), "neutral")
    return f'<span class="v {cls}">{_esc(text)}</span>'


def _execution_html(execution: Any, *, observed: bool = False) -> str:
    """Execution status: plain when complete, otherwise flagged.

    ``observed=True`` (the survey tab) renders a non-complete status as plain
    text. Tab 2 promises observed-only rendering, and the committed ConSan
    survey reports carry ``execution_status: "error"``; that is an observation,
    not a regression, so it must not get the guardrail tab's red badge.
    """
    text = str(execution)
    if text == "complete" or observed:
        return _esc(text)
    return f'<span class="v fail">{_esc(text)}</span>'


def _execution_md(execution: Any) -> str:
    """Markdown twin of ``_execution_html``: neutral when complete, else emphasized."""
    text = str(execution)
    if text == "complete":
        return text
    return f"\u274c **{text}**"


def _gate_summary(
    rows: dict[str, dict[str, Any]], *, recorded_gate: bool | None = None
) -> dict[str, Any]:
    """Classify the aggregate run health for the top banner and history gate.

    A missing report and an observed verdict mismatch both fail the gate, but
    they are operationally different: only a *present* verdict that disagrees
    with its baseline is a regression. Absent reports are surfaced separately so
    an infrastructure failure is not mislabeled as a verdict regression, and a
    run with both is reported as a combined ``UNHEALTHY`` state.

    ``recorded_gate`` is the authoritative comparator result persisted in the run
    manifest (``meta.gate``). It is stricter than the row-level ``overall_verdict``
    match, so when it is explicitly ``False`` the run fails closed even if every
    reduced row matches its baseline -- otherwise a comparator-failed run would
    render ``HEALTHY``.
    """
    total = len(rows)
    matched = sum(1 for r in rows.values() if r["match"])
    mismatches = sum(1 for r in rows.values() if r["present"] and not r["match"])
    missing = sum(1 for r in rows.values() if not r["present"])
    rows_ok = matched == total
    if rows_ok and recorded_gate is False:
        # Rows match their baselines but the stricter comparator gate failed.
        label = "FAILED"
        detail = "run gate failed its comparator check"
    elif rows_ok:
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
    ok = rows_ok and recorded_gate is not False
    return {"ok": ok, "label": label, "detail": detail, "short": label.capitalize()}


def _history_case_html(row: dict[str, Any]) -> str:
    expected = ""
    if row["present"] and not row["match"]:
        expected = f' <span class="dash">expected</span> {_verdict_html(row["expected"])}'
    return (
        f"{_baseline_status_html(row, history=True)} "
        f'{_verdict_html(row["verdict"])}{expected}'
    )


def _history_gate_html(run: dict[str, Any]) -> str:
    summary = _gate_summary(run["rows"], recorded_gate=run["meta"].get("gate"))
    emphasis = "pass" if run["gate"] else "fail"
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


def _report_link_html(report_rel: str | None) -> str:
    """A relative link to a case's raw ``sanitizer_report.json``, or an em dash."""
    if not report_rel:
        return "&mdash;"
    return f'<a href="{_esc(report_rel)}">report</a>'


def _run_cell_html(run: dict[str, Any]) -> str:
    """History "Run" cell: link the run id to its published ``runs/<id>/`` area."""
    label = run["meta"].get("run", "")
    rel = run.get("rel")
    if rel:
        return f'<a href="{_esc(rel)}/">{_esc(label)}</a>'
    return _esc(label)


def _backend_txt_html(backend: dict[str, Any] | None) -> str:
    if not backend:
        return '<span class="dash">&mdash;</span>'
    return f'{_esc(backend["name"])} {_esc(backend["sha"])}'


def _fact_html(label: str, value: str, *, mono: bool = False) -> str:
    cls = "val mono" if mono else "val"
    return f'<span><span class="k">{_esc(label)}</span><span class="{cls}">{value}</span></span>'


def _facts_html(
    row: dict[str, Any],
    *,
    observed: bool = False,
    extra: tuple[tuple[str, str], ...] = (),
) -> str:
    """Backend / selection / kernels / execution as a borderless label/value grid.

    Replaces the old middot-separated meta line. A row of bordered chips reads
    as a tab strip, so these are plain label-over-value pairs on a grid.
    ``extra`` prepends caller-supplied pairs (the guardrail tab leads with
    observed/expected). ``observed=True`` keeps execution un-health-coloured.
    """
    wl = row["worklist"]
    items = [_fact_html(label, value) for label, value in extra]
    items.append(_fact_html("Backend", _backend_txt_html(row["backend"]), mono=True))
    items.append(
        _fact_html(
            "Selection",
            f"{_esc(str(wl['requirement']))} top&#8209;{_esc(str(wl['top_n']))}",
            mono=True,
        )
    )
    items.append(_fact_html("Kernels", _esc(str(wl["kernel_count"]))))
    items.append(_fact_html("Execution", _execution_html(row["execution"], observed=observed)))
    return f'<div class="kv">{"".join(items)}</div>'


def _kernel_tables_html(
    row: dict[str, Any],
    *,
    report_rel: str | None = None,
) -> str:
    """The shared kernel-detail + findings tables (identical shape on both tabs).

    ``report_rel`` is the case's raw ``sanitizer_report.json`` link; when it is a
    safe relative path it is rendered in a per-row **Report** column so every
    kernel row drills down to the report (#367's per-row acceptance criterion),
    for both the guardrail and survey callers. It degrades to an em dash when the
    case carries no link -- an absent report, an unsafe caller value, or an input
    mode (results-dir / runs-root) that publishes no co-located reports -- so no
    dead or unsafe link is ever emitted. All kernel rows of one case share the
    single case-level report, so they all point at the same file.

    Both tabs render the same tinted ``_verdict_html`` badge per kernel; verdict
    colour is descriptive on either tab (see ``_verdict_html``).

    Numeric columns right-align the ``th`` as well as the ``td`` so each value
    sits directly under its own header rather than drifting to the far edge.
    """
    link = _report_link_html(_safe_report_rel(report_rel))
    krows = (
        "".join(
            f"<tr><td class=mono>{_esc(str(k['name']))}</td>"
            f"<td class=num>{_esc(str(k['dispatch']))}</td>"
            f"<td>{_verdict_html(k['verdict'])}</td>"
            f"<td class=num>{_esc(str(k['findings']))}</td>"
            f"<td class=mono>{_esc(k['code_object']) or '&mdash;'}</td>"
            f"<td class=mono>{_esc(k['sha']) or '&mdash;'}</td>"
            f"<td>{link}</td></tr>"
            for k in row["kernels"]
        )
        or '<tr><td class=empty colspan=7>no kernels selected</td></tr>'
    )
    frows = (
        "".join(
            f"<tr><td>{_esc(str(g['sanitizer']))}</td><td class=mono>{_esc(str(g['code']))}</td>"
            f"<td>{_esc(str(g['severity']))}</td><td class=num>{g['count']}</td>"
            f"<td class='mono wrap-any'>{_esc(g['example'])}</td></tr>"
            for g in row["finding_groups"]
        )
        or '<tr><td class=empty colspan=5>no findings</td></tr>'
    )
    return (
        '<p class="cap">Kernels</p><div class="table-wrap"><table>'
        "<thead><tr><th>Kernel</th><th class=num>Dispatch</th>"
        "<th>Observed</th><th class=num>Findings</th>"
        f"<th>Code object</th><th>SHA-256</th><th>Report</th></tr></thead>"
        f"<tbody>{krows}</tbody></table></div>"
        '<p class="cap">Findings</p><div class="table-wrap"><table>'
        "<thead><tr><th>Sanitizer</th><th>Code</th><th>Severity</th><th class=num>Count</th>"
        f"<th>Example</th></tr></thead><tbody>{frows}</tbody></table></div>"
    )


def _raw_link_html(report_rel: str | None, *, staged: bool = True) -> str:
    """The run-area link, as a card footer.

    Points at the case's published *directory*, not just its
    ``sanitizer_report.json``: the run area also carries the sanitizer logs the
    verdict came from, the recipe as it ran, its source-level inputs and a
    REPRODUCE.md, which is what makes the card's reproduce command actionable
    (#384). The per-kernel-row ``Report`` column still links the JSON directly
    for a schema drill-down.

    ``staged=False`` emits nothing. A directory link needs an ``index.html`` that
    only the publisher of the area can write, so a case whose *report* was staged
    by someone else (a ``--survey`` spec supplying an inline report) has a
    reachable JSON but no reachable directory. Guardrail rows default to ``True``:
    their ``report_rel`` is generator-computed for present cases, and the same
    loop publishes the area it names.

    Deliberately in the card body rather than the ``<summary>`` row: a link
    inside a summary competes with the disclosure toggle.
    """
    rel = _case_dir_rel(report_rel) if staged else None
    if not rel:
        return ""
    return f'<div class="card-foot"><a href="{_esc(rel)}">run area</a></div>'


def _findings_chip_html(row: dict[str, Any]) -> str:
    """Findings count for a collapsed card's summary row.

    An absent report also carries ``findings: 0``, but "no findings" would
    claim a clean scan where none ran. Emit nothing in that case and let the
    verdict badge (and, on the guardrail tab, the "Report missing" pill) say so.
    """
    if not row.get("present"):
        return ""
    count = int(row.get("findings") or 0)
    if not count:
        return '<span class="findings">no findings</span>'
    return f'<span class="findings has">{count} finding{"" if count == 1 else "s"}</span>'


def _observation_html(row: dict[str, Any], *, tone: str = "") -> str:
    """The observation callout: uppercase label over the observation text."""
    text = row.get("observation", "")
    if not text:
        return ""
    cls = f"observation {tone}".strip()
    return (
        f'<div class="{cls}"><span class="lbl">Observation</span>{_esc(text)}</div>'
    )


def _tone_for(verdict: Any) -> str:
    """Map a verdict onto the callout tint class.

    Returns "" for pass and for anything unmapped, so a healthy or unknown
    observation keeps the neutral grey callout rather than being tinted.
    """
    cls = _VERDICT_CLASS.get(str(verdict).strip().lower(), "")
    return "" if cls == "pass" else cls


def _sanitizer_accent(backend_label: str) -> str:
    """Identity accent class for a case: purple = waitcheck, green = consan."""
    label = backend_label.lower()
    if "waitcheck" in label:
        return "wc"
    if "consan" in label:
        return "cs"
    return "nu"


def _card_summary_html(
    *,
    accent: str,
    icon: str,
    name: str,
    kind: str,
    trailing: str,
) -> str:
    """The collapsed card's triage row.

    Every card ships closed, so this row has to carry enough status that a
    reader can decide what to open without opening anything: identity icon,
    name, kind, then the caller's status badges.
    """
    tile = {"wc": "purple", "cs": "green"}.get(accent, "blue")
    return (
        f'<summary><span class="tile {tile}">{_svg(icon, size=17, width=1.9)}</span>'
        f'<span class="name">{_esc(name)}</span>'
        f'<span class="kind">{_esc(kind)}</span>'
        f'{trailing}<span class="spacer"></span>{_chevron_html()}</summary>'
    )


def _kernel_detail_html(rows: dict[str, dict[str, Any]]) -> str:
    """Tab 1 (guardrails): expected-vs-observed kernel detail with baseline status.

    One collapsed ``<details>`` card per recipe. Pure HTML disclosure -- no JS,
    matching the CSS-radio tabs.
    """
    blocks: list[str] = []
    for case, _key, label, backend_label in CASES:
        row = rows[case]
        accent = _sanitizer_accent(backend_label)
        icon = _ICON_SHIELD if accent == "wc" else _ICON_ACTIVITY
        trailing = (
            f"{_baseline_status_html(row)}{_verdict_html(row['verdict'])}"
            f"{_findings_chip_html(row)}"
        )
        summary = _card_summary_html(
            accent=accent, icon=icon, name=label, kind=backend_label, trailing=trailing
        )
        tone = _tone_for(row["verdict"])
        if not row["present"]:
            body = _observation_html(row, tone="fail")
        else:
            body = (
                _facts_html(
                    row,
                    extra=(
                        ("Observed", _esc(str(row["verdict"]))),
                        ("Expected", _esc(str(row["expected"] or _DASH))),
                    ),
                )
                + _observation_html(row, tone=tone)
                + _kernel_tables_html(row, report_rel=row.get("report_rel"))
                + _raw_link_html(row.get("report_rel"))
            )
        blocks.append(
            f'<details class="kcard {accent}">{summary}'
            f'<div class="kcard-body">{body}</div></details>'
        )
    return "".join(blocks)


_SURVEY_CHIPS: tuple[tuple[str, str, str, str], ...] = (
    ("blue", _ICON_SEARCH, "Understand kernel behavior",
     "See how real kernels behave under sanitizers."),
    ("purple", _ICON_BUG, "Investigate sanitizer findings",
     "Review issues detected by WaitCheck or ConSan."),
    ("green", _ICON_REPEAT, "Reproduce issues locally",
     "Use provided commands to reproduce runs."),
)

# Workload -> kernel -> waitcheck -> consan -> dashboard, as a horizontal band.
# Numbered, with a one-line description per node so the diagram needs no legend.
_SURVEY_FLOW: tuple[tuple[str, str, str, str], ...] = (
    ("blue", _ICON_FOLDER, "Workload", "Run a workload or test recipe"),
    ("blue", _ICON_CODE, "Kernel Generated", "GPU kernel is compiled / extracted"),
    ("purple", _ICON_SHIELD, "WaitCheck Analysis", "Static scan for s_waitcnt hazards"),
    ("green", _ICON_ACTIVITY, "ConSan Analysis", "Runtime data-race / concurrency check"),
    ("blue", _ICON_LAYOUT, "Dashboard Results", "Reports shown under each sanitizer"),
)

_SURVEY_NOTES: tuple[str, ...] = (
    "This tab is informational only and does not affect nightly pass/fail status.",
    "Findings represent observed behavior, not regressions.",
    "Where both sanitizers produced a report the kernel appears under each; "
    "a skipped or missing scan still appears, marked report missing with no verdict.",
    "Every case lists the recipe that produced the kernel and a reproducible command.",
)


def _survey_flow_html() -> str:
    parts: list[str] = []
    for index, (accent, icon, label, desc) in enumerate(_SURVEY_FLOW, start=1):
        if parts:
            parts.append('<span class="flow-arrow">&#8594;</span>')
        parts.append(
            f'<div class="step"><span class="step-num {accent}">{index}</span>'
            f'<div class="tile {accent}">{_svg(icon, size=22, width=1.85)}</div>'
            f'<div class="label">{_esc(label)}</div>'
            f'<div class="desc">{_esc(desc)}</div></div>'
        )
    return f'<div class="flow">{"".join(parts)}</div>'


def _survey_note_html() -> str:
    """The workload-survey tab's orientation panel.

    States plainly that Tab 2 is observed-only / non-gating, lists what the tab
    is for, and shows how a result is produced. The WaitCheck / ConSan
    definitions are *not* here -- they are page-level cards shown above the tab
    bar, because Tab 1 reports the same two sanitizers and needs them too.
    """
    chips = "".join(
        f'<div class="chip"><span class="circle {accent}">{_svg(icon, size=16, width=1.95)}</span>'
        f'<span><span class="h">{_esc(head)}</span>'
        f'<span class="d">{_esc(desc)}</span></span></div>'
        for accent, icon, head, desc in _SURVEY_CHIPS
    )
    notes = "".join(
        f'<li><span class="check">{_svg(_ICON_CHECK, size=10, width=3.5)}</span>{_esc(note)}</li>'
        for note in _SURVEY_NOTES
    )
    return (
        '<div class="info-banner">'
        f"{_svg(_ICON_INFO, size=18, width=2)}"
        "<p><strong>This tab is observational only.</strong> Results do not affect "
        "nightly pass/fail status and should be interpreted independently.</p></div>"
        f'<section class="panel"><h2>What You Can Do Here</h2>'
        f'<div class="chips">{chips}</div></section>'
        '<div class="two-col">'
        '<div class="panel"><h2>How Results Are Produced</h2>'
        f"{_survey_flow_html()}"
        f'<div class="panel-note">{_svg(_ICON_FILE, size=16, width=1.8)}'
        "Each result includes the workload/recipe and a copy-paste command to "
        "reproduce the run.</div></div>"
        '<details class="panel notes" open><summary>Important Notes'
        f"{_chevron_html()}</summary><ul>{notes}</ul></details>"
        "</div>"
    )


def _sanitizer_name(entry: dict[str, Any]) -> str:
    """Card title: the sanitizer's product name."""
    san = str(entry.get("sanitizer") or "").strip().lower()
    if san == "waitcheck":
        return "WaitCheck"
    if san == "consan":
        return "ConSan"
    return str(entry.get("label") or san or "sanitizer")


def _sanitizer_kind(entry: dict[str, Any]) -> str:
    """Card subtitle: what the sanitizer does, in four words."""
    san = str(entry.get("sanitizer") or "").strip().lower()
    if san == "waitcheck":
        return "static wait-count scan"
    if san == "consan":
        return "dynamic data-race check"
    return ""


def _survey_group_label(group: str, entries: list[dict[str, Any]]) -> str:
    """A readable kernel-group heading. Prefer an entry-supplied group label, else
    prettify the group key (``lds-dispatch`` -> ``lds dispatch``)."""
    for entry in entries:
        gl = entry.get("group_label")
        if gl:
            return str(gl)
    return group.replace("-", " ").replace("_", " ")


def _survey_panel_title(group: str, entries: list[dict[str, Any]]) -> str:
    """Heading for a kernel-group panel.

    A lone ungrouped case leads with its own ``label``: the group key falls back
    to its terser ``name``, so the label is the only human name it has. Grouped
    kernels use the shared group label. Roll-up table rows keep using
    ``_survey_group_label`` so the HTML and markdown mirrors stay in parity.
    """
    if len(entries) == 1 and entries[0].get("label"):
        return str(entries[0]["label"])
    return _survey_group_label(group, entries)


def _survey_howto_html(entry: dict[str, Any]) -> str:
    """A copy-pasteable "reproduce this run yourself" strip (empty when unknown)."""
    command = entry.get("command")
    if not command:
        return ""
    # Label outside the box, matching the run area: with it inside, the box read
    # as a code block whose first token was "REPRODUCE", and selecting the box to
    # copy the command picked the label up too.
    return (
        '<p class="cap">Reproduce</p>'
        f'<div class="repro"><code>{_esc(str(command))}</code></div>'
    )


def _survey_message_parts(row: dict[str, Any]) -> tuple[str, str]:
    """Pick the inline survey message as an ``(label, text)`` pair (pure).

    Precedence is verdict-aware: an ``error`` case leads with its fail-closed
    ``primary.reason`` -- an errored report can still carry *partial* findings (e.g.
    a coverage-incomplete ConSan run), and surfacing a ``Finding:`` there would hide
    the reason the requirement says errored cases must show inline. warn/fail cases
    lead with the top finding example, falling back to a reason if present. Returns
    ``("", "")`` when there is nothing to say. Single source of truth so the HTML and
    MD twins stay in parity.
    """
    verdict = str(row.get("verdict") or "").strip().lower()
    reason = (row.get("primary") or {}).get("reason")
    reason_text = _clean_msg(str(reason), 240) if reason else ""
    groups = row.get("finding_groups") or []
    example = _clean_msg(str(groups[0].get("example", "")), 240) if groups else ""
    if verdict == "error" and reason_text:
        return "Reason", reason_text
    if example:
        return "Finding", example
    if reason_text:
        return "Reason", reason_text
    return "", ""


def _survey_message_html(row: dict[str, Any]) -> str:
    """The human-readable outcome message shown INLINE on the page (#374 E).

    Surfaces the actual finding / error text on Tab 2 itself, not only behind the
    raw-report link, with error-reason-first precedence (see ``_survey_message_parts``).
    Empty when there is nothing to say.
    """
    label, text = _survey_message_parts(row)
    if not text:
        return ""
    tone = _tone_for(row.get("verdict"))
    cls = f"observation {tone}".strip()
    return (
        f'<div class="{cls}"><span class="lbl">{_esc(label)}</span>'
        f'<span class="msg">{_esc(text)}</span></div>'
    )


_SANITIZER_RANK = {"waitcheck": 0, "consan": 1}


def _group_survey_entries(
    survey: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group survey entries by kernel-group key, preserving first-seen order (pure).

    The single source of truth for the Tab 2 kernel grouping: both the roll-up
    summary table and the per-kernel detail blocks consume this same structure, so
    a kernel scanned by both waitcheck and ConSan is one group with a sub-entry per
    sanitizer. Returns an ordered ``[(group_key, entries), ...]`` list.
    """
    order: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in survey:
        key = str(entry.get("group") or entry.get("name") or entry.get("label") or "survey")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(entry)
    return [(key, groups[key]) for key in order]


# Verdict buckets for the observed-only survey roll-up, in display order. A
# *present* survey run that observed something other than pass/warn/fail/error
# (e.g. a literal ``not_checked``/unknown verdict) folds into ``not_checked``. An
# absent sanitizer cell (no report -> the kernel was not scanned by that sanitizer)
# is NOT a run and is excluded from these counts entirely (rendered as an em dash).
_SURVEY_VERDICT_BUCKETS: tuple[str, ...] = ("pass", "warn", "fail", "error", "not_checked")


def _survey_verdict_bucket(row: dict[str, Any]) -> str:
    """Bucket a present survey row's observed verdict for the roll-up (pure)."""
    verdict = str(row.get("verdict") or "").strip().lower()
    return verdict if verdict in ("pass", "warn", "fail", "error") else "not_checked"


def _survey_present_runs(entries: list[dict[str, Any]]) -> int:
    """Sanitizer runs in a kernel group that actually produced a report (pure).

    Single source of truth for "how many runs", so the kernel-group heading and the
    roll-up headline agree: an absent report still renders a card (with no verdict)
    but it is not a run.
    """
    return sum(1 for entry in entries if (entry.get("summary") or {}).get("present"))


def _survey_summary_stats(
    groups: list[tuple[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Aggregate grouped survey entries into observed-only roll-up counts (pure).

    Returns ``{"kernels": K, "runs": M, "verdicts": {bucket: n}}`` where ``kernels``
    is the number of kernel groups, ``runs`` is the number of sanitizer runs that
    actually produced a report (absent sanitizer cells are not runs), and
    ``verdicts`` sums exactly to ``runs``. These are observations of what the
    sanitizers saw -- they never gate and carry no regression semantics.
    """
    verdicts = dict.fromkeys(_SURVEY_VERDICT_BUCKETS, 0)
    runs = 0
    for _key, entries in groups:
        runs += _survey_present_runs(entries)
        for entry in entries:
            row = entry.get("summary") or {}
            if not row.get("present"):
                continue
            verdicts[_survey_verdict_bucket(row)] += 1
    return {"kernels": len(groups), "runs": runs, "verdicts": verdicts}


def _survey_headline(stats: dict[str, Any]) -> str:
    """One-line at-a-glance stat for the survey roll-up (pure).

    Zero buckets are omitted so the line stays readable; the shown buckets always
    sum to the sanitizer-run count for honest arithmetic.
    """
    verdicts = stats.get("verdicts", {})
    parts = [f"{n} {bucket}" for bucket in _SURVEY_VERDICT_BUCKETS if (n := verdicts.get(bucket, 0))]
    breakdown = " \u00b7 ".join(parts) if parts else "no sanitizer runs"
    kernels = int(stats.get("kernels", 0))
    runs = int(stats.get("runs", 0))
    return (
        f"Surveyed {kernels} kernel{'' if kernels == 1 else 's'} across "
        f"{runs} sanitizer run{'' if runs == 1 else 's'} \u2014 {breakdown}"
    )


def _survey_headline_html(stats: dict[str, Any]) -> str:
    """The roll-up stat strip: a lead sentence plus one badge per verdict bucket.

    Same arithmetic as ``_survey_headline`` (which stays the MD twin); the
    badges reuse the shared verdict colours so the strip matches the table.
    """
    verdicts = stats.get("verdicts", {})
    kernels = int(stats.get("kernels", 0))
    runs = int(stats.get("runs", 0))
    lead = (
        f'<span class="lead">Surveyed <b>{kernels} kernel'
        f"{'' if kernels == 1 else 's'}</b> across <b>{runs} sanitizer run"
        f"{'' if runs == 1 else 's'}</b></span>"
    )
    badges = "".join(
        f'<span class="v {_VERDICT_CLASS.get(bucket, "neutral")}">{n} {_esc(bucket)}</span>'
        for bucket in _SURVEY_VERDICT_BUCKETS
        if (n := verdicts.get(bucket, 0))
    )
    return lead + (badges or '<span class="v neutral">no sanitizer runs</span>')


def _survey_group_by_sanitizer(
    entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map a kernel group's entries to ``{sanitizer: entry}`` (first wins) (pure)."""
    by_san: dict[str, dict[str, Any]] = {}
    for entry in entries:
        san = str(entry.get("sanitizer") or "").strip().lower()
        if san and san not in by_san:
            by_san[san] = entry
    return by_san


def _survey_group_findings(entries: list[dict[str, Any]]) -> int:
    """Total findings observed across a kernel group's present entries (pure)."""
    return sum(
        int((entry.get("summary") or {}).get("findings") or 0)
        for entry in entries
        if (entry.get("summary") or {}).get("present")
    )


def _survey_group_note(entries: list[dict[str, Any]]) -> str:
    """A short observation note for a kernel group's roll-up row (pure).

    Error-reason-first, mirroring ``_survey_message_parts``: if any sanitizer in the
    group errored, its fail-closed reason is the salient note even when a sibling
    produced findings; otherwise the first finding code, else the first reason, else
    empty. Descriptive only -- never regression vocabulary.
    """
    for entry in entries:
        row = entry.get("summary") or {}
        if str(row.get("verdict") or "").strip().lower() == "error":
            reason = (row.get("primary") or {}).get("reason")
            if reason:
                return str(reason)
    for entry in entries:
        row = entry.get("summary") or {}
        finding_groups = row.get("finding_groups") or []
        if finding_groups and finding_groups[0].get("code"):
            return str(finding_groups[0]["code"])
        reason = (row.get("primary") or {}).get("reason")
        if reason:
            return str(reason)
    return ""


def _survey_summary_cell_html(entry: dict[str, Any] | None) -> str:
    """A verdict-badge roll-up cell, or an em dash when the sanitizer did not run."""
    row = (entry or {}).get("summary") or {}
    if entry is None or not row.get("present"):
        return '<span class="dash">&mdash;</span>'
    return _verdict_html(row.get("verdict"))


def _survey_summary_table_html(
    groups: list[tuple[str, list[dict[str, Any]]]],
) -> str:
    """The Tab 2 roll-up: a headline stat + one row per kernel (pure).

    Columns: Kernel | waitcheck | ConSan | Findings | Note. Verdict cells reuse the
    tinted ``_verdict_html`` badges; a sanitizer not run for a kernel shows
    an em dash rather than a fabricated verdict. Observed-only: nothing here gates.
    Empty groups render nothing (the caller shows the empty-state note).
    """
    if not groups:
        return ""
    stats = _survey_summary_stats(groups)
    rows: list[str] = []
    for key, entries in groups:
        by_san = _survey_group_by_sanitizer(entries)
        note = _survey_group_note(entries)
        note_html = f"<td class=mono>{_esc(note)}</td>" if note else '<td class="dash">&mdash;</td>'
        rows.append(
            f"<tr><td class=mono>{_esc(_survey_group_label(key, entries))}</td>"
            f"<td>{_survey_summary_cell_html(by_san.get('waitcheck'))}</td>"
            f"<td>{_survey_summary_cell_html(by_san.get('consan'))}</td>"
            f"<td class=num>{_survey_group_findings(entries)}</td>"
            f"{note_html}</tr>"
        )
    return (
        '<section class="panel survey-summary"><h2>Survey summary</h2>'
        f'<div class="statline">{_survey_headline_html(stats)}</div>'
        '<div class="table-wrap"><table><thead>'
        "<tr><th>Kernel</th><th>WaitCheck</th><th>ConSan</th>"
        "<th class=num>Findings</th><th>Note</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table></div></section>'
    )


def _survey_case_html(entry: dict[str, Any], *, heading: str, kind: str = "") -> str:
    """One observed-only survey case as a collapsed ``<details>`` card.

    The summary row carries the verdict badge and findings count so a reader can
    triage every case without expanding any of them. The body holds the facts
    grid, the inline finding/reason, the reproduce command and the tables. An
    absent report keeps its originating workload but drops the facts that would
    describe a report that was never produced.
    """
    row = entry["summary"]
    san = str(entry.get("sanitizer") or "").strip().lower()
    accent = {"waitcheck": "wc", "consan": "cs"}.get(san, "nu")
    icon = _ICON_SHIELD if accent == "wc" else _ICON_ACTIVITY
    trailing = f"{_verdict_html(row['verdict'])}{_findings_chip_html(row)}"
    summary = _card_summary_html(
        accent=accent, icon=icon, name=heading, kind=kind, trailing=trailing
    )
    workload = entry.get("workload")
    extra: tuple[tuple[str, str], ...] = (
        (("Source", _esc(str(workload))),) if workload else ()
    )
    if not row["present"]:
        # Provenance only: the rest of the facts grid (backend, selection,
        # execution) would describe a report that does not exist. The source
        # still belongs here -- the MD twin names it on this branch too.
        facts = "".join(_fact_html(label, value) for label, value in extra)
        body = (
            (f'<div class="kv">{facts}</div>' if facts else "")
            + _observation_html(row, tone=_tone_for(row["verdict"]))
            + _survey_message_html(row)
            + _survey_howto_html(entry)
        )
    else:
        body = (
            _facts_html(row, observed=True, extra=extra)
            + _observation_html(row, tone=_tone_for(row["verdict"]))
            + _survey_message_html(row)
            + _survey_howto_html(entry)
            + _kernel_tables_html(row, report_rel=entry.get("report_rel"))
            + _raw_link_html(entry.get("report_rel"), staged=bool(entry.get("staged")))
        )
    return (
        f'<details class="kcard {accent}">{summary}'
        f'<div class="kcard-body">{body}</div></details>'
    )


def _survey_detail_html(survey: list[dict[str, Any]]) -> str:
    """Tab 2 (workload survey): observed-only kernel detail, no expected/match column.

    Cases sharing a ``group`` key (e.g. the same kernel run under both waitcheck and
    ConSan) render under one kernel-group heading with a sanitizer sub-block each
    (waitcheck first, then ConSan). A lone case renders on its own. Verdicts are
    solid color-coded chips and the actual message is shown inline (#374 C/D/E).
    """
    if not survey:
        return (
            f"{_survey_note_html()}"
            '<p class="survey-empty">No workload-survey kernels in this run.</p>'
        )
    grouped = _group_survey_entries(survey)
    # Roll-up summary (headline + one row per kernel) sits above the detail blocks
    # so the reader sees coverage + outcomes at a glance without scrolling.
    blocks: list[str] = [_survey_note_html(), _survey_summary_table_html(grouped)]
    for key, entries in grouped:
        ordered = sorted(
            entries, key=lambda e: _SANITIZER_RANK.get(str(e.get("sanitizer")), 99)
        )
        # Counting spec entries instead would show "2 sanitizer runs" on a group
        # whose second report is missing while the roll-up above it counts one.
        runs = _survey_present_runs(ordered)
        cards = "".join(
            _survey_case_html(
                entry,
                heading=_sanitizer_name(entry),
                kind=_sanitizer_kind(entry),
            )
            for entry in ordered
        )
        blocks.append(
            '<section class="panel"><h2 class="kgroup-title">'
            f'<span class="kname">{_esc(_survey_panel_title(key, entries))}</span>'
            f'<span class="count">{runs} sanitizer run{"" if runs == 1 else "s"}</span>'
            f"</h2>{cards}</section>"
        )
    return "".join(blocks)


def build_html(
    runs: list[dict[str, Any]],
    *,
    title: str = "Sanitizers Nightly",
    status: dict[str, Any] | None = None,
    survey: list[dict[str, Any]] | None = None,
) -> str:
    banner = _status_banner_html(status)
    if not runs:
        # Rendered before the first successful nightly (empty runs-root) so the
        # /sanitizers/ route never 404s, and whenever a run fails with no data.
        # Shares _CSS: this branch still renders the stale banner, which is the
        # most safety-relevant thing on the page, and an unstyled warning on an
        # otherwise-dark site reads as a broken page rather than an alert.
        return (
            "<!doctype html>\n"
            "<html lang=en><head><meta charset=utf-8>\n"
            '<meta name=viewport content="width=device-width, initial-scale=1">\n'
            f"<title>{_esc(title)}</title>\n"
            f"<style>{_CSS}</style></head>\n"
            "<body><div class=wrap>\n"
            '<div class=navrow><a href="../">&larr; back to CI dashboard</a></div>\n'
            f"{banner}"
            '<header class="page-header"><div class="brand-tile">'
            f"{_svg(_ICON_ACTIVITY, size=24, width=2)}</div><div>\n"
            f"<h1>{_esc(title)}</h1>\n"
            '<p class="subtitle">No runs yet</p></div></header>\n'
            '<section class="panel"><p>No sanitizer runs yet. This page will '
            "populate after the first successful sanitizer nightly.</p></section>\n"
            "</div></body></html>\n"
        )
    latest = runs[0]
    meta = latest["meta"]
    summary = _gate_summary(latest["rows"], recorded_gate=latest["meta"].get("gate"))

    latest_rows = "".join(
        f"<tr><td class=mono>{_esc(label)}</td><td>{_esc(backend)}</td>"
        f"<td>{_baseline_status_html(latest['rows'][case])}</td>"
        f"<td>{_verdict_html(latest['rows'][case]['verdict'])}</td>"
        f"<td>{_verdict_html(latest['rows'][case]['expected'] or _DASH)}</td>"
        f"<td>{_execution_html(latest['rows'][case]['execution'])}</td>"
        f"<td class=num>{latest['rows'][case]['findings']}</td>"
        f"<td class=mono>{_esc(latest['rows'][case]['coverage']) or '<span class=dash>&mdash;</span>'}</td>"
        f"<td>{_report_link_html(latest['rows'][case].get('report_rel'))}</td></tr>"
        for case, _key, label, backend in CASES
    )
    # Link to the whole run area (co-located raw reports) when published there.
    latest_rel = latest.get("rel")
    run_area_link = (
        f'<a href="{_esc(latest_rel)}/">{_svg(_ICON_FILE, size=14, width=2)} raw reports</a>'
        if latest_rel
        else "<span></span>"
    )

    hist_head = "".join(f"<th>{_esc(label)}</th>" for _c, _k, label, _b in CASES)
    hist_rows = "".join(
        f"<tr><td class=mono>{_run_cell_html(run)}</td>"
        f"<td class=mono>{_esc(run['meta'].get('commit', ''))}</td>"
        f"<td>{_esc(run['meta'].get('date', ''))}</td>"
        + "".join(f"<td>{_history_case_html(run['rows'][c])}</td>" for c, _k, _l, _b in CASES)
        + _history_gate_html(run)
        + "</tr>"
        for run in runs
    )
    run_count = len(runs)
    hist_gate = (
        f'<span class="pill {"ok" if latest["gate"] else "bad"}">'
        f'latest: {_esc(summary["short"])}</span>'
    )

    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style></head>
<body><div class=wrap>
  <div class=navrow>
    <a href="../">&larr; back to CI dashboard</a>
    {run_area_link}
  </div>
  {banner}
  <div class=topbar>
    <header class="page-header">
      <div class="brand-tile">{_svg(_ICON_ACTIVITY, size=24, width=2)}</div>
      <div>
        <h1>{_esc(title)}</h1>
        <p class="subtitle">GPU sanitizer guardrails and workload survey</p>
      </div>
    </header>
    {_run_card_html(meta)}
  </div>

  {_tool_cards_html()}

  <div class=tabs>
  <input type=radio name=santab id=tab-guardrails class=tabradio checked>
  <input type=radio name=santab id=tab-survey class=tabradio>
  <div class=tabbar>
    <label for="tab-guardrails">Expected behavior (guardrails)</label>
    <label for="tab-survey">Workload survey (observed-only)</label>
  </div>

  <section class=tabpanel id=panel-guardrails>
  {_gate_hero_html(summary)}
  <div class="info-banner">{_svg(_ICON_INFO, size=18, width=2)}
    <p><strong>This tab is the regression gate.</strong> Baseline status column
       will determine if regression failed or passed.</p></div>

  <section class="panel">
    <h2>Latest run <span class=count>{len(CASES)} recipes</span></h2>
    <div class="table-wrap"><table><thead>
      <tr><th>Recipe</th><th>Backend</th><th>Baseline status</th><th>Observed</th>
          <th>Expected</th><th>Execution</th><th class=num>Findings</th>
          <th>Coverage</th><th>Report</th></tr></thead>
      <tbody>{latest_rows}</tbody>
    </table></div>
  </section>

  <section class="panel">
    <h2>Kernel details <span class=count>{len(CASES)} recipes</span></h2>
    {_kernel_detail_html(latest['rows'])}
  </section>

  <details class="panel">
    <summary>History / trend
      <span class=count>{run_count} run{'' if run_count == 1 else 's'}</span>
      {hist_gate}<span class="spacer"></span>{_chevron_html()}</summary>
    <div class="panel-body"><div class="table-wrap"><table><thead>
      <tr><th>Run</th><th>Commit</th><th>Date</th>{hist_head}<th>Gate</th></tr></thead>
      <tbody>{hist_rows}</tbody>
    </table></div></div>
  </details>
  </section>

  <section class=tabpanel id=panel-survey>
  {_survey_detail_html(survey or [])}
  </section>
  </div>
</div></body></html>
"""


def _run_index_survey_html(survey: list[dict[str, Any]] | None) -> str:
    """The run landing page's workload-survey table (pure).

    Without this the page lists only the three gated guardrail recipes, so the
    top-nav "raw reports" link never mentions the survey cases published
    alongside them under ``survey/<case>/`` (#384). For a past run the entries
    come from ``survey_entries_from_published``, whose recorded verdict can be
    absent -- hence the em-dash fallback rather than a "None".

    A run-area link is emitted only for an entry marked ``staged`` -- i.e. one
    whose whole directory was published -- never from the case name plus report
    presence. A report can be *present* (it parsed) without its directory
    existing at all: a ``--survey`` spec's entries render from an inline report
    or an out-of-tree ``report_path``, while only ``--informational-results-dir``
    entries are staged under ``survey/<case>/``. Keying on presence therefore
    emitted a 404 on exactly that invocation.

    The link path itself goes through ``_safe_case_segment``, because ``name`` is
    caller-supplied on a spec entry: a staged spec naming ``../../other`` would
    otherwise render a traversal link here while the card footer -- which derives
    its link from the validated ``report_rel`` -- pointed somewhere else entirely.
    """
    if not survey:
        return ""

    def _area_cell(entry: dict[str, Any]) -> str:
        segment = _safe_case_segment(entry.get("name")) if entry.get("staged") else None
        if not segment:
            return '<td><span class="dash">&mdash;</span></td>'
        return f'<td><a href="survey/{_esc(segment)}/">run area</a></td>'

    rows = "".join(
        f"<tr><td class=mono>{_esc(str(entry.get('label') or entry.get('name', '')))}</td>"
        f"<td>{_verdict_html((entry.get('summary') or {}).get('verdict') or _DASH)}</td>"
        f"<td class=mono>{_esc(str(entry.get('command') or ''))}</td>"
        + _area_cell(entry)
        + "</tr>"
        for entry in survey
    )
    return (
        '<section class="panel"><h2>Workload survey '
        f'<span class=count>{len(survey)} observed-only</span></h2>'
        '<div class="table-wrap"><table>\n'
        "<thead><tr><th>Case</th><th>Observed</th><th>Reproduce</th><th>Files</th></tr></thead>\n"
        f"<tbody>{rows}</tbody>\n"
        "</table></div></section>\n"
    )


def survey_entries_from_published(run_dir: Path) -> list[dict[str, Any]]:
    """Reconstruct a past run's survey list from the areas published under it.

    The live survey list only ever covers the run being published now, so an
    older run's landing page has nothing to render its Workload survey section
    from -- leaving the survey areas still on the data branch under it reachable
    only by typing the URL, on a site that cannot auto-index a directory. Each
    area's own ``env.json`` already records everything that table shows.

    Every entry is ``staged``: these are read off disk, so the directory is known
    to exist by construction.
    """
    entries: list[dict[str, Any]] = []
    for case_dir in sorted(p for p in (run_dir / "survey").glob("*") if p.is_dir()):
        env = _load(case_dir / "env.json")
        if not isinstance(env, dict):
            continue
        observed = env.get("observed") or {}
        entries.append({
            "name": case_dir.name,
            "label": str(env.get("case") or case_dir.name),
            "command": str(env.get("command") or ""),
            "staged": True,
            "summary": {
                "verdict": observed.get("verdict"),
                "present": (case_dir / "sanitizer_report.json").is_file(),
            },
        })
    return entries


def build_run_index_html(
    run: dict[str, Any],
    *,
    title: str = "Sanitizers Nightly",
    survey: list[dict[str, Any]] | None = None,
    up: str = "../../",
) -> str:
    """A tiny, self-contained landing page for one published run (pure).

    Lives at ``runs/<id>/index.html`` alongside the run's raw reports, so the
    report links are case-local (``<case>/sanitizer_report.json``) rather than
    dashboard-root-relative. ``survey`` adds the observed-only cases published
    under ``survey/<case>/`` for this run, so the page covers both tabs.
    """
    meta = run["meta"]
    rows = run["rows"]
    # Reuse the shared aggregate classifier so a missing report reads as
    # INCOMPLETE rather than a misleading "verdict mismatch" (mirrors build_html).
    summary = _gate_summary(rows, recorded_gate=run["meta"].get("gate"))
    run_url = meta.get("run_url", "")
    run_link = (
        f'<a href="{_esc(run_url)}">workflow run</a>' if run_url else "<span></span>"
    )

    def _cell(case: str, present: bool) -> str:
        if not present:
            return '<span class="dash">report missing</span>'
        href = _esc(f"{case}/sanitizer_report.json")
        return f'<a href="{href}">sanitizer_report.json</a>'

    def _area_cell(case: str, present: bool) -> str:
        if not present:
            return '<span class="dash">&mdash;</span>'
        return f'<a href="{_esc(case)}/">run area</a>'

    report_rows = "".join(
        f"<tr><td class=mono>{_esc(label)}</td>"
        f"<td>{_baseline_status_html(rows[case])} {_verdict_html(rows[case]['verdict'])}</td>"
        f"<td>{_cell(case, rows[case]['present'])}</td>"
        f"<td>{_area_cell(case, rows[case]['present'])}</td></tr>"
        for case, _key, label, _backend in CASES
    )
    # Shares the dashboard stylesheet so a drill-down does not look like a
    # different product; only the narrower wrap differs. Linked rather than
    # inlined -- there is one of these per retained run (see RUN_AREA_CSS_REL).
    return (
        "<!doctype html>\n"
        "<html lang=en><head><meta charset=utf-8>\n"
        '<meta name=viewport content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)} &middot; {_esc(meta.get('run', ''))}</title>\n"
        f'<link rel=stylesheet href="{_esc(up)}{RUN_AREA_CSS_REL}"></head>\n'
        "<body><div class=wrap>\n"
        '<div class=navrow><a href="../../">&larr; back to sanitizer dashboard</a>\n'
        f"{run_link}</div>\n"
        '<div class=topbar>\n'
        '<header class="page-header"><div class="brand-tile">'
        f"{_svg(_ICON_ACTIVITY, size=24, width=2)}</div><div>\n"
        f"<h1>{_esc(title)}</h1>\n"
        f'<p class="subtitle">run {_esc(meta.get("run", ""))}</p></div></header>\n'
        f"{_run_card_html(meta)}\n"
        "</div>\n"
        f"{_gate_hero_html(summary)}\n"
        '<section class="panel"><h2>Raw reports</h2><div class="table-wrap"><table>\n'
        "<thead><tr><th>Recipe</th><th>Baseline status</th><th>Raw report</th>"
        "<th>Files</th></tr></thead>\n"
        f"<tbody>{report_rows}</tbody>\n"
        "</table></div></section>\n"
        f"{_run_index_survey_html(survey)}"
        "</div></body></html>\n"
    )


def _survey_message_md(row: dict[str, Any]) -> str:
    """The inline human-readable outcome message for the MD mirror (#374 E).

    Same error-reason-first precedence as the HTML twin (``_survey_message_parts``).
    """
    label, text = _survey_message_parts(row)
    return f"{label}: `{text}`" if text else ""


def _survey_summary_md(groups: list[tuple[str, list[dict[str, Any]]]]) -> list[str]:
    """Markdown mirror of the Tab 2 roll-up: headline stat + one row per kernel."""
    if not groups:
        return []
    stats = _survey_summary_stats(groups)

    def cell(entry: dict[str, Any] | None) -> str:
        row = (entry or {}).get("summary") or {}
        if entry is None or not row.get("present"):
            return _DASH
        return f"`{row.get('verdict')}`"

    lines = [
        _survey_headline(stats),
        "",
        "| Kernel | waitcheck | ConSan | Findings | Note |",
        "|---|---|---|--:|---|",
    ]
    for key, entries in groups:
        by_san = _survey_group_by_sanitizer(entries)
        note = _survey_group_note(entries)
        lines.append(
            f"| {_survey_group_label(key, entries)} | {cell(by_san.get('waitcheck'))} "
            f"| {cell(by_san.get('consan'))} | {_survey_group_findings(entries)} "
            f"| {note or _DASH} |"
        )
    lines.append("")
    return lines


def _survey_section_md(survey: list[dict[str, Any]]) -> list[str]:
    """Tab 2 mirror for the GitHub job summary: observed-only, non-gating."""
    lines = [
        "## Workload survey (observed-only)",
        "",
        "How real GPU kernels behave under AMD's sanitizers \u2014 **waitcheck** (static "
        "`s_waitcnt` wait-count scan) and **ConSan** (dynamic data-race check); where "
        "both produced a report the kernel is shown under each, and a scan that was "
        "skipped or whose report is missing still appears, marked report missing with "
        "no verdict. **No "
        "expected-behavior comparison on this tab**; an `error` / `fail` / `warn` here "
        "is an observation of how the kernel behaved, not a regression. Each case lists "
        "a copy-paste command to reproduce the run.",
        "",
    ]
    if not survey:
        lines += ["No workload-survey kernels in this run.", ""]
        return lines
    lines += _survey_summary_md(_group_survey_entries(survey))
    for entry in survey:
        r = entry["summary"]
        workload = f" \u00b7 source `{entry['workload']}`" if entry.get("workload") else ""
        lines.append(
            f"<details><summary><b>{entry['label']}</b> \u2014 observed "
            f"`{r['verdict']}`</summary>"
        )
        lines += ["", f"Observation: {r.get('observation', '')}{workload}"]
        message = _survey_message_md(r)
        if message:
            lines += ["", message]
        if entry.get("command"):
            lines += ["", f"Reproduce: `{entry['command']}`"]
        if not r["present"]:
            lines += ["", "</details>", ""]
            continue
        lines += [
            "",
            "| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |",
            "|---|--:|---|--:|---|---|",
        ]
        for k in r["kernels"]:
            lines.append(
                f"| `{k['name']}` | {k['dispatch']} | `{k['verdict']}` | {k['findings']} | "
                f"`{k['code_object'] or _DASH}` | `{k['sha'] or _DASH}` |"
            )
        lines += ["", "</details>", ""]
    return lines


def build_summary_md(
    runs: list[dict[str, Any]],
    *,
    status: dict[str, Any] | None = None,
    survey: list[dict[str, Any]] | None = None,
) -> str:
    banner = _status_banner_md(status)
    if not runs:
        head = "# Sanitizers Nightly\n\n"
        if banner:
            head += banner + "\n\n"
        return head + "No runs found.\n"
    latest = runs[0]
    meta = latest["meta"]
    summary = _gate_summary(latest["rows"], recorded_gate=latest["meta"].get("gate"))
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

    lines += [
        "",
        "Two views below: **Expected behavior (guardrails)** (baseline-checked, the gate) "
        "and **Workload survey (observed-only)** (non-gating).",
        "",
        "## Expected behavior (guardrails) \u00b7 Kernel details",
        "",
    ]
    for case, _key, label, _backend in CASES:
        r = latest["rows"][case]
        lines.append(
            f"<details><summary><b>{label}</b> \u2014 " f"{_baseline_status_md(r)}</summary>"
        )
        lines.append("")
        if not r["present"]:
            lines += [
                f"Observed sanitizer verdict: `{r['verdict']}`",
                "",
                f"Observation: {r.get('observation', '')}",
                "",
                "</details>",
                "",
            ]
            continue
        b = r["backend"]
        wl = r["worklist"]
        backend_name = b["name"] if b else _DASH
        lines.append(
            f"Observed sanitizer verdict `{r['verdict']}` \u00b7 expected `{r['expected'] or _DASH}`"
        )
        lines.append(f"Observation: {r.get('observation', '')}")
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

    lines += _survey_section_md(survey or [])

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
            f"{_gate_summary(run['rows'], recorded_gate=run['meta'].get('gate'))['short']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--results-dir", type=Path, help="DIR/<case>/sanitizer_report.json (one run)")
    src.add_argument("--runs-root", type=Path, help="dir of run_* folders (local history)")
    src.add_argument(
        "--history-root",
        type=Path,
        help="published DIR/<id>/<case>/sanitizer_report.json layout (data branch)",
    )
    ap.add_argument(
        "--keep",
        type=int,
        default=30,
        help="in --history-root mode, render only the newest N runs (default 30)",
    )
    ap.add_argument(
        "--keep-logs",
        type=int,
        default=7,
        help="keep sanitizer logs + the recipe copy and its inputs for only the "
        "newest N runs (default 7), guardrail and survey areas alike. Older "
        "retained runs keep their report, manifests and landing page; their bulk "
        "is pruned so the published tree stays bounded (#384). Note this bounds the "
        "checkout, not the branch history: a pruned blob stays reachable from the "
        "commit that added it, so anything ever published is permanent.",
    )
    ap.add_argument(
        "--current-run",
        default="",
        help="the <YYYY-MM-DD>-<run_id> this job just staged. Only that run's area "
        "is written from the live checkout and environment; without it the newest "
        "retained run is assumed, which is wrong when an older workflow is re-run "
        "after a newer same-day one (its lower run id sorts behind).",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repo checkout the recipes and their source-level fixture inputs are "
        "copied from into each run area (defaults to this script's own checkout)",
    )
    ap.add_argument("--baselines", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--commit", default=os.environ.get("GITHUB_SHA", ""))
    ap.add_argument("--run-label", default=os.environ.get("GITHUB_RUN_ID", ""))
    ap.add_argument(
        "--status",
        type=Path,
        help="JSON run-status manifest; renders a stale banner when it is unhealthy",
    )
    ap.add_argument(
        "--informational-results-dir",
        type=Path,
        help="dir of <case>/sanitizer_report.json (caller-supplied ConSan objects, #347) "
        "folded into the observed-only workload-survey tab (Tab 2) as non-gating cases; "
        "no longer a separate informational section. Absent dir => no cases.",
    )
    ap.add_argument(
        "--survey",
        type=Path,
        help="JSON spec of observed-only workload-survey cases for the survey tab "
        "(see survey_cases_from_spec). Data is supplied here at run time so this public "
        "repo hardcodes no customer/NDA identifiers; a fail/not_checked never gates.",
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
    elif args.runs_root is not None:
        runs = runs_from_runs_root(args.runs_root, baselines)
    else:
        runs = runs_from_history_root(args.history_root, baselines, keep=args.keep)

    # Which retained run this job actually produced. NOT simply the newest one:
    # re-running an older workflow reuses its lower GITHUB_RUN_ID, so its
    # <date>-<run_id> sorts *behind* a newer same-day run. Taking position 0 then
    # stamps the newer area with this job's container/bundle/recipe and publishes
    # this job's survey under the wrong run. Fall back to the newest run only when
    # the caller did not say (the results-dir / runs-root modes, and older callers).
    current_index: int | None = None
    if runs:
        if args.current_run:
            current_index = next(
                (
                    index
                    for index, run in enumerate(runs)
                    if str(run["meta"].get("run", "")) == args.current_run
                ),
                None,  # staged run already pruned out of the window: nothing is current
            )
        else:
            current_index = 0
    current_rel = runs[current_index].get("rel") if current_index is not None else None

    # Observed-only survey cases for Tab 2. A malformed/absent spec degrades to an
    # empty survey (Tab 2 renders its empty-state note) rather than crashing render.
    survey: list[dict[str, Any]] = []
    if args.survey is not None:
        spec = _load(args.survey)
        if isinstance(spec, (dict, list)):
            survey = survey_cases_from_spec(
                spec, base_dir=args.survey.parent, rel=current_rel
            )
    # Caller-supplied ConSan cases (#347) now render as observed-only workload-survey
    # (Tab 2) entries rather than a separate informational section. They are appended
    # after any explicit --survey spec cases so an explicit spec (if wired) leads, and
    # their report_rel is threaded to the latest run's published area so per-case /
    # per-kernel raw-report links resolve once the reports are co-published below.
    informational_survey: list[dict[str, Any]] = []
    if args.informational_results_dir is not None:
        informational_survey = survey_cases_from_informational_dir(
            args.informational_results_dir,
            rel=current_rel,
        )
        survey += informational_survey
    # Mirror the two-class split into data.json: attach the survey list to the run
    # this job produced (additive; existing per-run keys are untouched).
    if current_index is not None:
        runs[current_index]["survey"] = survey
    # What the dashboard page and the job summary show on Tab 2. Both take their
    # guardrail data from runs[0], so the survey rendered beside it has to describe
    # that same run -- and the run this job staged is not always the newest one (see
    # --current-run). Passing this job's survey there would label the newer run
    # "Latest run" while showing an older re-run's observations under it, and would
    # drop the newer run's own published survey from data.json, which nothing else
    # reconstructs. A published ``survey/`` has the same <case>/sanitizer_report.json
    # layout an informational results dir does, so the newest run's list is rebuilt
    # from the areas already published under it, at full card fidelity.
    displayed_survey = survey
    if runs and current_index != 0 and args.history_root is not None:
        displayed_survey = survey_cases_from_informational_dir(
            args.history_root / str(runs[0]["meta"].get("run", "")) / "survey",
            rel=runs[0].get("rel"),
        )
        runs[0]["survey"] = displayed_survey

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "index.html").write_text(
        build_html(runs, status=status, survey=displayed_survey),
        encoding="utf-8",
    )
    # In published-history mode, write each retained run's tiny landing page next
    # to its co-located raw reports (out_dir/runs/<id>/index.html). Bounded by
    # --keep, so pruned runs stop getting a page.
    if args.history_root is not None:
        # The one stylesheet every run and run-area page links, written once for
        # the whole tree rather than inlined into each of them.
        write_shared_assets(args.out_dir)
        # When the raw reports live in a separate tree from the output (the copy
        # branch below), a reused --out-dir would keep run dirs dropped by --keep
        # (or reports since removed from a retained run) as stale published output.
        # Clear the output runs/ tree first so the published set matches `runs`
        # exactly. Skip when history_root IS runs_out (CI) or nests beneath it
        # (e.g. --history-root out/runs/source --out-dir out) -- removing runs_out
        # would delete the very reports we are about to copy.
        # Describes this job's GPU run only, so it is stamped onto the run being
        # published now and never onto the older runs re-rendered beside it.
        provenance = provenance_from_environ()
        info_dir = args.informational_results_dir
        # Survey areas the co-publish loop further below rewrites from this job's
        # own results. Every *other* published area -- an older run's, or one this
        # run inherited from the history being re-rendered -- is refreshed in the
        # run loop instead, so no published area is left without a landing page.
        #
        # Only the informational entries, never a --survey spec's: the co-publish
        # loop stages only those. Including a spec name here skipped an area in the
        # run loop that nothing later rewrote, so it kept its raw logs even under
        # --keep-logs 0.
        staged_now = {str(entry.get("name")) for entry in informational_survey}
        runs_out = args.out_dir / "runs"
        runs_out_res = runs_out.resolve()
        history_res = args.history_root.resolve()
        history_within_runs_out = (
            history_res == runs_out_res or runs_out_res in history_res.parents
        )
        if runs_out.exists() and not history_within_runs_out:
            shutil.rmtree(runs_out)
        for position, run in enumerate(runs):
            rel = run.get("rel")
            if not rel:
                continue
            run_out = args.out_dir / rel
            run_out.mkdir(parents=True, exist_ok=True)
            # Logs and copied inputs are kept only for the newest --keep-logs runs;
            # every retained run still keeps its report, manifests and landing page.
            keep_logs = position < max(args.keep_logs, 0)
            # Co-locate each retained run's raw case dirs under the output tree so the
            # emitted runs/<id>/<case>/... links resolve even when --history-root is
            # not already <out-dir>/runs. In CI both resolve to the same path, so the
            # copy is skipped and only the in-place steps below apply.
            src_run = args.history_root / run["meta"].get("run", "")
            if src_run.is_dir() and src_run.resolve() != run_out.resolve():
                for case, _key, _label, _backend in CASES:
                    # Copy the whole case dir, not just the report: the sanitizer
                    # logs the verdict came from are what make the run area
                    # reproducible (#384).
                    src_case = src_run / case
                    if src_case.is_dir():
                        shutil.copytree(src_case, run_out / case, dirs_exist_ok=True)
                # The survey subtree belongs to the run too. The stale-clear above
                # emptied the output tree, so without this a retained run silently
                # loses the survey areas its history holds -- their downloads 404
                # and the regenerated run index drops the survey table.
                src_survey = src_run / "survey"
                if src_survey.is_dir():
                    shutil.copytree(src_survey, run_out / "survey", dirs_exist_ok=True)
                # Carry the run manifest too so the published layout
                # (runs/<id>/meta.json) is complete, not just the reports.
                src_meta = src_run / "meta.json"
                if src_meta.is_file():
                    shutil.copy2(src_meta, run_out / "meta.json")
            # Only the newest run was produced by this job, so only its area may be
            # written from the live checkout and environment (see
            # write_case_run_area); older ones are re-rendered from what they
            # captured themselves.
            current = position == current_index
            for case, _key, label, _backend in CASES:
                row = run["rows"][case]
                case_out = run_out / case
                if not row.get("present") or not case_out.is_dir():
                    continue
                if keep_logs:
                    _gzip_logs(case_out)
                else:
                    _prune_run_area_bulk(case_out)
                if not current and refresh_published_case_area(
                    case_out, args.out_dir, logs=keep_logs
                ):
                    continue
                up, run_up = case_dir_up(case_out, args.out_dir)
                write_case_run_area(
                    case_out,
                    case=case,
                    cls="guardrail",
                    recipe_stem=label,
                    command=f"aorta sweep run --recipe recipes/sanitizers/{label}.yaml",
                    meta=run["meta"],
                    summary=row,
                    report=_load(case_out / "sanitizer_report.json"),
                    repo_root=args.repo_root,
                    up=up,
                    run_up=run_up,
                    logs=keep_logs,
                    provenance=provenance,
                    current=current,
                )
            # A survey area is published under whichever run was latest at the
            # time, so it ages out of the log window with the rest of that run --
            # without this its ConSan log, the bulkiest thing published, would
            # outlive the guardrail logs beside it by --keep minus --keep-logs.
            for survey_case in sorted(
                p for p in (run_out / "survey").glob("*") if p.is_dir()
            ):
                if current and survey_case.name in staged_now:
                    continue  # rewritten from this job's own results further below
                # Same treatment as a guardrail area: an in-window area still has
                # to be gzipped, or a disjoint history (or an area published before
                # run areas existed) publishes its raw *.log uncompressed.
                if keep_logs:
                    _gzip_logs(survey_case)
                else:
                    _prune_run_area_bulk(survey_case)
                if refresh_published_case_area(
                    survey_case, args.out_dir, logs=keep_logs
                ):
                    continue
                # Co-published before run areas existed: retrofit one, the same way
                # an older guardrail case is handled above. Without this an area
                # with a report but no manifest stays invisible -- it gets no
                # landing page, and survey_entries_from_published skips it.
                survey_report = _load(survey_case / "sanitizer_report.json")
                if not isinstance(survey_report, dict):
                    continue
                # A top-level dict can still hold a malformed *nested* shape
                # (``{"checks": null}``) that makes the reduction raise, which is
                # why survey_cases_from_informational_dir isolates the same call.
                # Without this an orphan area left on the data branch by an earlier
                # publish aborts the entire dashboard, days later and for a
                # non-gating input. Skip it as unreadable instead.
                try:
                    survey_summary = summarize_case(survey_report, None)
                except (AttributeError, KeyError, TypeError, ValueError):
                    continue
                stem = _survey_recipe_for(survey_case.name)
                up, run_up = case_dir_up(survey_case, args.out_dir)
                write_case_run_area(
                    survey_case,
                    case=survey_case.name,
                    cls="survey",
                    recipe_stem=stem,
                    command=f"aorta sweep run --recipe recipes/sanitizers/{stem}.yaml",
                    meta=run["meta"],
                    summary=survey_summary,
                    report=survey_report,
                    repo_root=args.repo_root,
                    up=up,
                    run_up=run_up,
                    logs=keep_logs,
                    current=False,
                )
            (run_out / "index.html").write_text(
                build_run_index_html(
                    run,
                    survey=survey if current else survey_entries_from_published(run_out),
                ),
                encoding="utf-8",
            )
        # Co-publish each caller-supplied ConSan report (#347) under the survey area
        # of the run this job produced, so the report_rel links threaded above
        # (survey/<name>/...) resolve. Done after the stale-clear (shutil.rmtree)
        # and the per-run loop that (re)creates out_dir/<rel>, using the same rel +
        # case name as survey_cases_from_informational_dir so the copied path
        # matches the link. current_rel, not runs[0]: on a re-run of an older
        # workflow these differ, and staging under the newer run would attribute
        # this job's sweep to a run that did not produce it.
        if (
            current_index is not None
            and current_rel
            and info_dir is not None
            and info_dir.is_dir()
        ):
            # That run's own retention decision, so --keep-logs applies to
            # guardrail and survey areas alike as documented (with --keep-logs 0
            # the survey area is pruned too, not left as the one unbounded thing).
            # The staged run's position, never a fallback to the newest one: that
            # substitution is the mistake --current-run exists to prevent.
            staged_keeps_logs = current_index < max(args.keep_logs, 0)
            # Informational entries only, never a --survey spec's: this loop copies
            # only what info_dir holds, and a spec entry that happens to share a
            # rejected case's name would satisfy the lookup below and publish the
            # very directory the survey loader refused to read -- with a summary and
            # command describing a different report.
            survey_by_name = {
                str(entry.get("name")): entry for entry in informational_survey
            }
            for case_dir in sorted(p for p in info_dir.iterdir() if p.is_dir()):
                src_report = case_dir / "sanitizer_report.json"
                if not src_report.is_file():
                    continue
                # Resolve the entry BEFORE copying. A case whose report was
                # rejected as malformed has no entry, and copying it anyway left an
                # orphan directory with no manifest: the next nightly then treats
                # it as an older survey area and retrofits one from that same
                # unusable report. Nothing should persist into history that the
                # survey loader already decided it could not read.
                entry = survey_by_name.get(case_dir.name)
                if entry is None:
                    continue
                dest = args.out_dir / current_rel / "survey" / case_dir.name
                # Replace the destination rather than merging into it: a file that
                # disappeared from this sweep (an old log especially) would
                # otherwise survive and be listed on the landing page as evidence
                # for the *new* report. The nightly clears its run dir first, but
                # this co-publish path is documented for reusable output trees too.
                # Skip the clear if the source somehow sits underneath it, which
                # would delete the very tree about to be read.
                if dest.exists() and not (
                    dest.resolve() == case_dir.resolve()
                    or dest.resolve() in case_dir.resolve().parents
                ):
                    shutil.rmtree(dest)
                # The whole case dir, so the ConSan/waitcheck logs behind the
                # observed verdict ship with the report (#384).
                shutil.copytree(case_dir, dest, dirs_exist_ok=True)
                if staged_keeps_logs:
                    _gzip_logs(dest)
                else:
                    _prune_run_area_bulk(dest)
                up, run_up = case_dir_up(dest, args.out_dir)
                write_case_run_area(
                    dest,
                    case=case_dir.name,
                    cls="survey",
                    recipe_stem=_survey_recipe_for(case_dir.name),
                    command=str(entry.get("command") or ""),
                    meta=runs[current_index]["meta"],
                    summary=entry.get("summary") or {},
                    report=_load(dest / "sanitizer_report.json"),
                    repo_root=args.repo_root,
                    up=up,
                    run_up=run_up,
                    logs=staged_keeps_logs,
                    provenance=provenance,
                )
    (args.out_dir / "summary.md").write_text(
        build_summary_md(runs, status=status, survey=displayed_survey),
        encoding="utf-8",
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
    gate_label = (
        _gate_summary(runs[0]["rows"], recorded_gate=runs[0]["meta"].get("gate"))["short"].lower()
        if runs else "no-data"
    )
    print(f"wrote {args.out_dir}/index.html, summary.md, data.json (gate={gate_label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
