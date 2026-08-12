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
  supplied via ``--survey``) shown with the same kernel-detail shape but **no
  expected/match column** (a fail / not_checked here is an observation, never a
  regression). Both tabs link each case -- in its heading and in a per-kernel-row
  ``Report`` column -- to its ``sanitizer_report.json`` (satisfying #367's per-row
  drill-down), and carry a one-line observation summary.
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
  ``/sanitizers/``: each run's raw reports are co-located under ``runs/<id>/`` and
  linked from the rendered page, and a tiny ``runs/<id>/index.html`` landing page
  is written for each retained run.

Pure rendering lives in ``build_html`` / ``build_summary_md`` /
``build_run_index_html`` and pure aggregation in ``summarize_case`` so they are
unit testable without the FS.

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
    if report is None:
        # Match follows the same rule as the present path: a survey case
        # (expected is None) is never a mismatch, while a guardrail case keeps a
        # non-null expectation and so a missing report stays a mismatch (the gate
        # fails closed). Hardcoding False here contradicted the survey contract.
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
    # carries the guardrail/survey case-class split; survey cases live in the
    # separate ``survey`` list attached to the latest run.
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
    ``/``), backslashes, and embedded control/whitespace characters. The
    internally-computed guardrail/history links already have this shape, so this
    only tightens the untrusted survey field.
    """
    if not isinstance(value, str) or not value:
        return None
    if value != value.strip() or any(ord(ch) < 0x20 for ch in value):
        return None
    if value.startswith("/") or "\\" in value:
        return None
    # A relative path never carries a URL scheme; a colon in the first segment
    # (before the first "/", e.g. ``javascript:``) means it is a scheme, not a path.
    if ":" in value.split("/", 1)[0]:
        return None
    return value


def survey_cases_from_spec(
    spec: dict[str, Any] | list[Any], *, base_dir: Path | None = None
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
          "report_path": "relative/or/abs/report.json"   # a file to load
        }

    Each entry carries ``cls="survey"`` and a ``summarize_case(report, None)``
    summary (``expected=None`` -> observational, ``match=True``), so failures /
    ``not_checked`` are shown as data and never rendered as a regression.

    ``report_rel`` is retained only for a *present* report (no dead link, matching
    the guardrail path in ``runs_from_history_root``) and only when it is a safe
    relative path (it is untrusted caller JSON; see ``_safe_report_rel``). A
    malformed spec (a non-list ``cases`` wrapper, a non-string ``report_path``)
    degrades to an empty / absent survey rather than raising.
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
        # Only construct a path from a non-empty string; a non-string report_path
        # (e.g. a numeric JSON value) would otherwise raise in Path() and abort the
        # whole dashboard instead of rendering this case as absent.
        report_path = case.get("report_path")
        if report is None and isinstance(report_path, str) and report_path:
            path = Path(report_path)
            if base_dir is not None and not path.is_absolute():
                path = base_dir / path
            report = _load(path)
        summary = summarize_case(report if isinstance(report, dict) else None, None)
        name = str(case.get("name", case.get("label", "survey")))
        backend = case.get("backend")
        if not backend:
            b = summary.get("backend")
            backend = b["name"] if b else _DASH
        entries.append(
            {
                "name": name,
                "label": str(case.get("label", name)),
                "backend": str(backend),
                "workload": case.get("workload"),
                # Keep a report link only for a present report (no dead link,
                # matching runs_from_history_root) and only when it is a safe
                # relative path (report_rel is untrusted caller JSON).
                "report_rel": (
                    _safe_report_rel(case.get("report_rel")) if summary["present"] else None
                ),
                "cls": "survey",
                "summary": summary,
            }
        )
    return entries


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
        expected = f" &middot; expected {_observed_html(row['expected'])}"
    return (
        f"{_baseline_status_html(row, history=True)}"
        f'<div class="secondary">Observed {_observed_html(row["verdict"])}{expected}</div>'
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
        return "&mdash;"
    return f'{_esc(backend["name"])} <span class=mono>{_esc(backend["sha"])}</span>'


def _meta_line_html(row: dict[str, Any]) -> str:
    wl = row["worklist"]
    return (
        f"<div class=meta>backend {_backend_txt_html(row['backend'])} &middot; selection "
        f"{_esc(str(wl['requirement']))} top&#8209;{_esc(str(wl['top_n']))} &middot; "
        f"{_esc(str(wl['kernel_count']))} kernel(s) &middot; execution "
        f"{_execution_html(row['execution'])}</div>"
    )


def _kernel_tables_html(row: dict[str, Any], *, report_rel: str | None = None) -> str:
    """The shared kernel-detail + findings tables (identical shape on both tabs).

    ``report_rel`` is the case's raw ``sanitizer_report.json`` link; when it is a
    safe relative path it is rendered in a per-row **Report** column so every
    kernel row drills down to the report (#367's per-row acceptance criterion),
    for both the guardrail and survey callers. It degrades to an em dash when the
    case carries no link -- an absent report, an unsafe caller value, or an input
    mode (results-dir / runs-root) that publishes no co-located reports -- so no
    dead or unsafe link is ever emitted. All kernel rows of one case share the
    single case-level report, so they all point at the same file.
    """
    link = _report_link_html(_safe_report_rel(report_rel))
    krows = (
        "".join(
            f"<tr><td class=mono>{_esc(str(k['name']))}</td>"
            f"<td class=num>{_esc(str(k['dispatch']))}</td>"
            f"<td>{_observed_html(k['verdict'])}</td>"
            f"<td class=num>{_esc(str(k['findings']))}</td>"
            f"<td class=mono>{_esc(k['code_object']) or '&mdash;'}</td>"
            f"<td class=mono>{_esc(k['sha']) or '&mdash;'}</td>"
            f"<td>{link}</td></tr>"
            for k in row["kernels"]
        )
        or "<tr><td colspan=7>no kernels selected</td></tr>"
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
    return (
        "<table><tr><th>Kernel</th><th>Dispatch</th>"
        "<th>Observed sanitizer verdict</th><th>Findings</th>"
        f"<th>Code object</th><th>SHA-256</th><th>Report</th></tr>{krows}</table>"
        "<table><tr><th>Sanitizer</th><th>Code</th><th>Severity</th><th>Count</th>"
        f"<th>Example</th></tr>{frows}</table>"
    )


def _raw_link_html(report_rel: str | None) -> str:
    return f' <a class=raw href="{_esc(report_rel)}">view raw report</a>' if report_rel else ""


def _kernel_detail_html(rows: dict[str, dict[str, Any]]) -> str:
    """Tab 1 (guardrails): expected-vs-observed kernel detail with baseline status."""
    blocks: list[str] = []
    for case, _key, label, _backend_label in CASES:
        row = rows[case]
        raw_link = _raw_link_html(row.get("report_rel"))
        if not row["present"]:
            blocks.append(
                f"<h3>{_esc(label)} &middot; {_baseline_status_html(row)}</h3>"
                f"<p>Observed sanitizer verdict: {_observed_html(row['verdict'])}</p>"
                f'<div class="secondary">Observation: {_esc(row.get("observation", ""))}</div>'
            )
            continue
        blocks.append(
            f"<h3>{_esc(label)} &middot; {_baseline_status_html(row)}{raw_link}</h3>"
            f'<div class="secondary">Observed sanitizer verdict '
            f"{_observed_html(row['verdict'])} &middot; expected "
            f"{_observed_html(row['expected'] or _DASH)}</div>"
            f'<div class="secondary">Observation: {_esc(row.get("observation", ""))}</div>'
            f"{_meta_line_html(row)}"
            f"{_kernel_tables_html(row, report_rel=row.get('report_rel'))}"
        )
    return "".join(blocks)


def _survey_note_html() -> str:
    return (
        '<p class="secondary survey-note">Workload survey &mdash; observed sanitizer '
        "behavior only. <b>No expected-behavior comparison on this tab</b>: a "
        "<span class=mono>fail</span> or <span class=mono>not_checked</span> here is an "
        "observation, not a regression. Kernels may be drawn from multiple workloads, "
        "including aorta-internal-sourced kernels supplied via the survey input.</p>"
    )


def _survey_detail_html(survey: list[dict[str, Any]]) -> str:
    """Tab 2 (workload survey): observed-only kernel detail, no expected/match column."""
    if not survey:
        return (
            f"{_survey_note_html()}"
            "<p class=secondary>No workload-survey kernels in this run.</p>"
        )
    blocks: list[str] = [_survey_note_html()]
    for entry in survey:
        row = entry["summary"]
        raw_link = _raw_link_html(entry.get("report_rel"))
        workload = entry.get("workload")
        workload_txt = f" &middot; source {_esc(str(workload))}" if workload else ""
        head = (
            f"<h3>{_esc(str(entry['label']))} &middot; {_observed_html(row['verdict'])}"
            f"{raw_link}</h3>"
        )
        if not row["present"]:
            blocks.append(
                f"{head}"
                f'<div class="secondary">Observation: {_esc(row.get("observation", ""))}'
                f"{workload_txt}</div>"
            )
            continue
        blocks.append(
            f"{head}"
            f'<div class="secondary">backend {_esc(str(entry["backend"]))}{workload_txt}</div>'
            f'<div class="secondary">Observation: {_esc(row.get("observation", ""))}</div>'
            f"{_meta_line_html(row)}"
            f"{_kernel_tables_html(row, report_rel=entry.get('report_rel'))}"
        )
    return "".join(blocks)


def informational_from_dir(root: Path) -> list[dict[str, Any]]:
    """Enumerate non-gating "informational" cases under ``root/<case>/sanitizer_report.json``.

    Used for experimental ConSan runs over caller-supplied code objects / commands
    (``source.consan_command``, #347). These are rendered in a clearly-labelled section
    that does **not** feed the baseline gate, so the daily controls stay authoritative.
    Each case surfaces its observed verdict and *reason* (which the gated tables omit)
    plus the ConSan preflight verdict. Absent fields (e.g. a passing check serializes a
    ``null`` reason) render as the em-dash placeholder rather than a literal ``None``.
    """

    def _cell(value: object) -> str:
        return _DASH if value is None else str(value)

    cases: list[dict[str, Any]] = []
    if not root.is_dir():
        return cases
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        report = _load(case_dir / "sanitizer_report.json")
        if report is None:
            continue
        summary = summarize_case(report, report.get("overall_verdict"))
        primary = (_DASH, _DASH, _DASH)
        preflight = _DASH
        for check in report.get("checks", []):
            san = check.get("sanitizer")
            if san == "waitcheck_preflight":
                preflight = _cell(check.get("verdict"))
            else:
                primary = (_cell(san), _cell(check.get("verdict")), _cell(check.get("reason")))
        cases.append(
            {
                "name": case_dir.name,
                "summary": summary,
                "sanitizer": primary[0],
                "verdict": primary[1],
                "reason": primary[2],
                "preflight": preflight,
            }
        )
    return cases


def build_informational_html(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return ""
    rows = "".join(
        f"<tr><td class=mono>{_esc(c['name'])}</td>"
        f"<td class=mono>{_esc(c['sanitizer'])}</td>"
        f"<td>{_observed_html(c['verdict'])}</td>"
        f"<td class=mono>{_esc(_clean_msg(c['reason'], 200))}</td>"
        f"<td>{_observed_html(c['preflight'])}</td></tr>"
        for c in cases
    )
    return (
        "<h2>Informational &middot; caller-supplied code objects (non-gating)</h2>"
        "<p class=secondary>Experimental ConSan runs over caller-supplied kernels/objects "
        "(<span class=mono>source.consan_command</span>, #347). These do <b>not</b> affect the "
        "gate above; the table reports each case's observed verdict and reason for this run.</p>"
        "<table><tr><th>Recipe</th><th>Sanitizer</th><th>Verdict</th><th>Reason</th>"
        f"<th>ConSan preflight</th></tr>{rows}</table>"
    )


def build_informational_md(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return ""
    lines = [
        "## Informational \u00b7 caller-supplied code objects (non-gating)",
        "",
        "Experimental ConSan runs over caller-supplied kernels/objects "
        "(`source.consan_command`, #347). These do **not** affect the gate; the table reports "
        "each case's observed verdict and reason for this run.",
        "",
        "| Recipe | Sanitizer | Verdict | Reason | ConSan preflight |",
        "|---|---|---|---|---|",
    ]
    for c in cases:
        reason = _clean_msg(c["reason"], 200).replace("|", "\\|")
        lines.append(
            f"| `{c['name']}` | `{c['sanitizer']}` | `{c['verdict']}` | {reason} | `{c['preflight']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def build_html(
    runs: list[dict[str, Any]],
    *,
    title: str = "Sanitizers Nightly",
    status: dict[str, Any] | None = None,
    informational: list[dict[str, Any]] | None = None,
    survey: list[dict[str, Any]] | None = None,
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
    summary = _gate_summary(latest["rows"], recorded_gate=latest["meta"].get("gate"))
    gate_color = "#2c7d3b" if summary["ok"] else "#c92c35"
    gate_text = f"{summary['label']} \u2014 {summary['detail']}"

    latest_rows = "".join(
        f"<tr><td>{_esc(label)}</td><td>{_esc(backend)}</td>"
        f"<td>{_baseline_status_html(latest['rows'][case])}</td>"
        f"<td>{_observed_html(latest['rows'][case]['verdict'])}</td>"
        f"<td>{_observed_html(latest['rows'][case]['expected'] or _DASH)}</td>"
        f"<td>{_execution_html(latest['rows'][case]['execution'])}</td>"
        f"<td class=num>{latest['rows'][case]['findings']}</td>"
        f"<td>{_esc(latest['rows'][case]['coverage']) or '&mdash;'}</td>"
        f"<td>{_report_link_html(latest['rows'][case].get('report_rel'))}</td></tr>"
        for case, _key, label, backend in CASES
    )
    # Link to the whole run area (co-located raw reports) when published there.
    latest_rel = latest.get("rel")
    run_area_link = (
        f' &middot; <a href="{_esc(latest_rel)}/">raw reports</a>' if latest_rel else ""
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
  a.raw {{ font-size:12px; font-weight:400; margin-left:8px; }}
  /* Self-contained tabs (no external JS/CSS): radios toggle sibling panels via :checked. */
  .tabradio {{ position:absolute; opacity:0; pointer-events:none; }}
  .tabbar {{ display:flex; gap:4px; border-bottom:1px solid #d0d7de; margin:16px 0 0; }}
  .tabbar label {{ cursor:pointer; padding:8px 14px; font-weight:600; color:#57606a;
          border:1px solid transparent; border-bottom:none; border-radius:8px 8px 0 0; }}
  #tab-guardrails:checked ~ .tabbar label[for="tab-guardrails"],
  #tab-survey:checked ~ .tabbar label[for="tab-survey"] {{
          color:#1f2328; background:#fff; border-color:#d0d7de; }}
  /* The radios are visually hidden (opacity:0) so they show no native focus
     ring; mirror :focus-visible onto the associated label so keyboard users can
     see which tab control is focused. */
  #tab-guardrails:focus-visible ~ .tabbar label[for="tab-guardrails"],
  #tab-survey:focus-visible ~ .tabbar label[for="tab-survey"] {{
          outline:2px solid #0969da; outline-offset:-2px; }}
  .tabpanel {{ display:none; padding-top:8px; }}
  #tab-guardrails:checked ~ #panel-guardrails {{ display:block; }}
  #tab-survey:checked ~ #panel-survey {{ display:block; }}
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
  .pill.pass {{ background:#2c7d3b; color:#fff; }} .pill.fail {{ background:#c92c35; color:#fff; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background:#0d1117; color:#e6edf3; }}
    table {{ background:#161b22; }} th {{ background:#161b22; color:#8b949e; }}
    th,td {{ border-color:#30363d; }}
    .meta,.secondary {{ color:#8b949e; }}
    .observed {{ color:#c9d1d9; background:#6e768133; border-color:#6e7681; }}
    #tab-guardrails:checked ~ .tabbar label[for="tab-guardrails"],
    #tab-survey:checked ~ .tabbar label[for="tab-survey"] {{
            color:#e6edf3; background:#161b22; border-color:#30363d; }}
  }}
</style></head>
<body><div class=wrap>
  <p class=nav><a href="../">back to CI dashboard</a></p>
  {banner}
  <h1>{_esc(title)} &middot; gfx950</h1>
  <div class=meta>latest run <span class=mono>{_esc(meta.get('run', ''))}</span>
     &middot; commit <span class=mono>{_esc(meta.get('commit', ''))}</span>
     &middot; {_esc(meta.get('date', ''))} &middot; target {_esc(meta.get('gpu', 'gfx950'))}{run_area_link}</div>
  <div class=gate>{_esc(gate_text)}</div>
  <p class=secondary>Observed <span class=mono>WARN</span> or <span class=mono>FAIL</span>
     verdicts may be expected positive-control outcomes. Baseline status is the
     regression-health signal.</p>

  <div class=tabs>
  <input type=radio name=santab id=tab-guardrails class=tabradio checked>
  <input type=radio name=santab id=tab-survey class=tabradio>
  <div class=tabbar>
    <label for="tab-guardrails">Expected behavior (guardrails)</label>
    <label for="tab-survey">Workload survey (observed-only)</label>
  </div>

  <section class=tabpanel id=panel-guardrails>
  <p class=secondary>Baseline-checked kernels: expected vs observed sanitizer behavior.
     This tab is the regression gate.</p>
  <h2>Latest run</h2>
  <table>
    <tr><th>Recipe</th><th>Backend</th><th>Baseline status</th><th>Observed</th>
        <th>Expected</th><th>Execution</th><th>Findings</th><th>Coverage</th><th>Report</th></tr>
    {latest_rows}
  </table>

  <h2>Kernel details</h2>
  {_kernel_detail_html(latest['rows'])}

  <h2>History / trend</h2>
  <table>
    <tr><th>Run</th><th>Commit</th><th>Date</th>{hist_head}<th>Gate</th></tr>
    {hist_rows}
  </table>
  </section>

  <section class=tabpanel id=panel-survey>
  <h2>Workload survey</h2>
  {_survey_detail_html(survey or [])}
  {build_informational_html(informational or [])}
  </section>
  </div>
</div></body></html>
"""


def build_run_index_html(run: dict[str, Any], *, title: str = "Sanitizers Nightly") -> str:
    """A tiny, self-contained landing page for one published run (pure).

    Lives at ``runs/<id>/index.html`` alongside the run's raw reports, so the
    report links are case-local (``<case>/sanitizer_report.json``) rather than
    dashboard-root-relative.
    """
    meta = run["meta"]
    rows = run["rows"]
    # Reuse the shared aggregate classifier so a missing report reads as
    # INCOMPLETE rather than a misleading "verdict mismatch" (mirrors build_html).
    summary = _gate_summary(rows, recorded_gate=run["meta"].get("gate"))
    gate_color = "#1a7f37" if summary["ok"] else "#cf222e"
    gate_text = f"{summary['label']} \u2014 {summary['detail']}"
    run_url = meta.get("run_url", "")
    run_link = (
        f' &middot; <a href="{_esc(run_url)}">workflow run</a>' if run_url else ""
    )

    def _cell(case: str, present: bool) -> str:
        if not present:
            return "report missing"
        href = _esc(f"{case}/sanitizer_report.json")
        return f'<a href="{href}">sanitizer_report.json</a>'

    report_rows = "".join(
        f"<tr><td>{_esc(label)}</td>"
        f"<td>{_baseline_status_html(rows[case])}"
        f'<div class="secondary">Observed {_observed_html(rows[case]["verdict"])}</div></td>'
        f"<td>{_cell(case, rows[case]['present'])}</td></tr>"
        for case, _key, label, _backend in CASES
    )
    return (
        "<!doctype html>\n"
        "<html lang=en><head><meta charset=utf-8>\n"
        '<meta name=viewport content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)} &middot; {_esc(meta.get('run', ''))}</title>\n"
        "<style>\n"
        "  body { font: 14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;\n"
        "         max-width:900px; margin:0 auto; padding:24px; color:#1f2328; }\n"
        "  h1 { font-size:18px; margin:0 0 4px; }\n"
        "  .meta { color:#57606a; font-size:12px; margin-bottom:12px; }\n"
        "  .mono { font-family: ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }\n"
        f"  .gate {{ color:#fff; background:{gate_color}; padding:10px 14px; border-radius:8px;\n"
        "          font-weight:600; margin:12px 0 18px; }\n"
        "  table { width:100%; border-collapse:collapse; }\n"
        "  th,td { text-align:left; padding:7px 10px; border-bottom:1px solid #d0d7de; }\n"
        "  .secondary { color:#57606a; font-size:12px; margin:4px 0 0; }\n"
        "  .pill { padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }\n"
        "  .pill.ok { background:#1a7f3722; color:#1a7f37; } .pill.bad { background:#cf222e22; color:#cf222e; }\n"
        "  .observed { display:inline-block; color:#57606a; background:#afb8c133; border:1px solid #afb8c1;\n"
        "              padding:1px 7px; border-radius:999px; font:600 12px ui-monospace,SFMono-Regular,Menlo,monospace; }\n"
        "</style></head>\n"
        "<body>\n"
        '<p><a href="../../">back to sanitizer dashboard</a></p>\n'
        f"<h1>{_esc(title)} &middot; run <span class=mono>{_esc(meta.get('run', ''))}</span></h1>\n"
        f"<div class=meta>commit <span class=mono>{_esc(meta.get('commit', ''))}</span>"
        f" &middot; {_esc(meta.get('date', ''))} &middot; target {_esc(meta.get('gpu', 'gfx950'))}"
        f"{run_link}</div>\n"
        f"<div class=gate>{_esc(gate_text)}</div>\n"
        "<table>\n"
        "<tr><th>Recipe</th><th>Baseline status</th><th>Raw report</th></tr>\n"
        f"{report_rows}\n"
        "</table>\n"
        "</body></html>\n"
    )


def _survey_section_md(survey: list[dict[str, Any]]) -> list[str]:
    """Tab 2 mirror for the GitHub job summary: observed-only, non-gating."""
    lines = [
        "## Workload survey (observed-only)",
        "",
        "Observed sanitizer behavior only \u2014 **no expected-behavior comparison on this "
        "tab**; a `fail` / `not_checked` here is an observation, not a regression. Kernels "
        "may be drawn from multiple workloads, including aorta-internal-sourced kernels "
        "supplied via the survey input.",
        "",
    ]
    if not survey:
        lines += ["No workload-survey kernels in this run.", ""]
        return lines
    for entry in survey:
        r = entry["summary"]
        workload = f" \u00b7 source `{entry['workload']}`" if entry.get("workload") else ""
        lines.append(
            f"<details><summary><b>{entry['label']}</b> \u2014 observed "
            f"`{r['verdict']}`</summary>"
        )
        lines += ["", f"Observation: {r.get('observation', '')}{workload}"]
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
    informational: list[dict[str, Any]] | None = None,
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

    informational_md = build_informational_md(informational or [])
    if informational_md:
        lines += [informational_md]

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
        help="dir of <case>/sanitizer_report.json rendered as a non-gating informational "
        "section (experimental caller-supplied ConSan objects; does not affect the gate)",
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

    informational = (
        informational_from_dir(args.informational_results_dir)
        if args.informational_results_dir is not None
        else []
    )

    # Observed-only survey cases for Tab 2. A malformed/absent spec degrades to an
    # empty survey (Tab 2 renders its empty-state note) rather than crashing render.
    survey: list[dict[str, Any]] = []
    if args.survey is not None:
        spec = _load(args.survey)
        if isinstance(spec, (dict, list)):
            survey = survey_cases_from_spec(spec, base_dir=args.survey.parent)
    # Mirror the two-class split into data.json: attach the survey list to the
    # latest run record (additive; existing per-run keys are untouched).
    if runs:
        runs[0]["survey"] = survey

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "index.html").write_text(
        build_html(runs, status=status, informational=informational, survey=survey),
        encoding="utf-8",
    )
    # In published-history mode, write each retained run's tiny landing page next
    # to its co-located raw reports (out_dir/runs/<id>/index.html). Bounded by
    # --keep, so pruned runs stop getting a page.
    if args.history_root is not None:
        # When the raw reports live in a separate tree from the output (the copy
        # branch below), a reused --out-dir would keep run dirs dropped by --keep
        # (or reports since removed from a retained run) as stale published output.
        # Clear the output runs/ tree first so the published set matches `runs`
        # exactly. Skip when history_root IS runs_out (CI) or nests beneath it
        # (e.g. --history-root out/runs/source --out-dir out) -- removing runs_out
        # would delete the very reports we are about to copy.
        runs_out = args.out_dir / "runs"
        runs_out_res = runs_out.resolve()
        history_res = args.history_root.resolve()
        history_within_runs_out = (
            history_res == runs_out_res or runs_out_res in history_res.parents
        )
        if runs_out.exists() and not history_within_runs_out:
            shutil.rmtree(runs_out)
        for run in runs:
            rel = run.get("rel")
            if not rel:
                continue
            run_out = args.out_dir / rel
            run_out.mkdir(parents=True, exist_ok=True)
            # Co-locate each retained run's raw reports under the output tree so the
            # emitted runs/<id>/<case>/sanitizer_report.json links resolve even when
            # --history-root is not already <out-dir>/runs. In CI both resolve to the
            # same path, so the copy is skipped and this is a no-op.
            src_run = args.history_root / run["meta"].get("run", "")
            if src_run.is_dir() and src_run.resolve() != run_out.resolve():
                for case, _key, _label, _backend in CASES:
                    src_report = src_run / case / "sanitizer_report.json"
                    if src_report.is_file():
                        (run_out / case).mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_report, run_out / case / "sanitizer_report.json")
                # Carry the run manifest too so the published layout
                # (runs/<id>/meta.json) is complete, not just the reports.
                src_meta = src_run / "meta.json"
                if src_meta.is_file():
                    shutil.copy2(src_meta, run_out / "meta.json")
            (run_out / "index.html").write_text(
                build_run_index_html(run), encoding="utf-8"
            )
    (args.out_dir / "summary.md").write_text(
        build_summary_md(runs, status=status, informational=informational, survey=survey),
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
