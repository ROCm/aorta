"""Generate kernel-trace correlation reports.

Reads ``rank_*_metrics.jsonl`` files in a run directory and emits:

  - A JSON summary keyed by rank (kernel-event totals, NaN counts).
  - A CSV table of per-finding rows (one per NaN iteration with
    preceding kernel-event counts).
  - An HTML report rendering the same data alongside a small Markdown-
    style narrative. The HTML is intentionally dependency-free so the
    generator works with the base ``aorta`` install.

The generator depends only on the standard library; ``aorta[report]``'s
heavy plotting stack (matplotlib, pandas) is *not* required for kernel
reports.
"""

from __future__ import annotations

import csv
import html
import io
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..analysis.kernel_correlator import (
    CorrelationFinding,
    IterationRecord,
    KernelEventCorrelator,
    iter_findings_table,
)

log = logging.getLogger(__name__)


def generate_kernel_report(
    metrics_dir: Path,
    output_dir: Path,
    *,
    lookback_iterations: int = 5,
    pattern: str = "rank_*_metrics.jsonl",
) -> dict[str, Path]:
    """Build a kernel-trace report bundle for a single run directory.

    Args:
        metrics_dir: Directory containing the per-rank JSONL files.
        output_dir: Where to write the report artifacts.
        lookback_iterations: How many iterations of context to attach to
            each NaN finding.
        pattern: Glob pattern for the metrics files.

    Returns:
        Mapping of artifact name to its filesystem path.
    """
    metrics_dir = Path(metrics_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correlator = KernelEventCorrelator(lookback_iterations=lookback_iterations)
    records = correlator.load_metrics_glob(metrics_dir, pattern=pattern)
    findings = correlator.find_failures(records)
    summary = correlator.summarise(records)

    summary_path = output_dir / "kernel_trace_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "metrics_dir": str(metrics_dir),
                "summary": summary,
                "findings": [_finding_to_dict(f) for f in findings],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    csv_path = output_dir / "kernel_trace_findings.csv"
    csv_path.write_text(_render_findings_csv(findings), encoding="utf-8")

    html_path = output_dir / "kernel_trace_report.html"
    html_path.write_text(
        _render_html(metrics_dir=metrics_dir, summary=summary, findings=findings),
        encoding="utf-8",
    )

    log.info(
        "Kernel report written: summary=%s findings=%s html=%s",
        summary_path,
        csv_path,
        html_path,
    )

    return {
        "summary_json": summary_path,
        "findings_csv": csv_path,
        "html_report": html_path,
    }


# Static columns that ``iter_findings_table`` always emits, in the
# order downstream consumers (humans + spreadsheet pivots + the HTML
# table further below) expect to see them. Keeping this in one place
# ensures the with-findings header agrees with the no-findings
# placeholder. Issue raised by Copilot on PR #162: previously the
# populated path used ``sorted(...)`` of all keys, which produced a
# different column order than the placeholder header
# ``"rank,global_step,loss\n"`` and surprised CSV consumers.
_FINDINGS_STATIC_COLUMNS: tuple[str, ...] = (
    "rank",
    "global_step",
    "loss",
    "lookback_iterations",
)


def _findings_csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    """Static columns first, then sorted dynamic ``kernel_*`` columns.

    All dynamic columns are guaranteed to be ``kernel_<event>`` strings
    by ``iter_findings_table``; sort them so the column ordering is
    deterministic across runs even if the underlying event-type set
    changes between versions.
    """
    dynamic_cols = sorted(
        {key for row in rows for key in row if key not in _FINDINGS_STATIC_COLUMNS}
    )
    return list(_FINDINGS_STATIC_COLUMNS) + dynamic_cols


def _render_findings_csv(findings: Iterable[CorrelationFinding]) -> str:
    """Serialise findings as CSV with a stable column order.

    Always writes a header row whose static-column prefix matches
    ``_FINDINGS_STATIC_COLUMNS`` -- including the no-findings case --
    so a downstream pipeline can rely on ``rank,global_step,loss,
    lookback_iterations`` being the leading columns regardless of
    whether any NaN iterations were detected.
    """
    rows = list(iter_findings_table(findings))
    fieldnames = _findings_csv_fieldnames(rows)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _finding_to_dict(finding: CorrelationFinding) -> dict[str, Any]:
    return {
        "target": _record_summary(finding.target),
        "preceding_window": [_record_summary(r) for r in finding.preceding_window],
        "kernel_event_total": finding.kernel_event_total,
    }


def _record_summary(record: IterationRecord) -> dict[str, Any]:
    return {
        "rank": record.rank,
        "global_step": record.global_step,
        "loss": record.loss,
        "kernel_summary": record.kernel_summary,
        "kernel_event_count": record.kernel_event_count,
        "overlap_ms": record.overlap_ms,
    }


_HTML_HEAD = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>Kernel Trace Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2em; color: #222; }
  h1, h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }
  table { border-collapse: collapse; margin: 1em 0; }
  th, td { border: 1px solid #ccc; padding: 0.4em 0.7em; text-align: right; }
  th { background: #f4f4f4; text-align: left; }
  td.label { text-align: left; }
  .nan { color: #b00020; font-weight: 600; }
  .empty { color: #888; font-style: italic; }
  pre { background: #f4f4f4; padding: 0.7em; border-radius: 4px; overflow-x: auto; }
</style>
</head>
<body>
"""

_HTML_FOOT = "</body></html>\n"


def _render_html(
    *,
    metrics_dir: Path,
    summary: dict[str, Any],
    findings: list[CorrelationFinding],
) -> str:
    parts: list[str] = [_HTML_HEAD]
    parts.append("<h1>Kernel Trace Report</h1>")
    parts.append(f"<p>Source: <code>{html.escape(str(metrics_dir))}</code></p>")

    parts.append("<h2>Summary</h2>")
    parts.append(_render_summary_table(summary))

    parts.append("<h2>NaN Findings</h2>")
    if not findings:
        parts.append("<p class='empty'>No NaN iterations detected.</p>")
    else:
        parts.append(_render_findings_table(findings))
        parts.append("<h2>Per-finding context</h2>")
        for idx, finding in enumerate(findings):
            parts.append(_render_finding_detail(idx, finding))

    parts.append(_HTML_FOOT)
    return "".join(parts)


def _render_summary_table(summary: dict[str, Any]) -> str:
    rows = [
        ("Total iterations", summary.get("total_iterations", 0)),
        ("NaN iterations", summary.get("nan_iterations", 0)),
        ("Ranks", ", ".join(str(r) for r in summary.get("ranks", []))),
    ]
    out = ["<table>"]
    for label, value in rows:
        out.append(
            f"<tr><td class='label'>{html.escape(label)}</td>"
            f"<td>{html.escape(str(value))}</td></tr>"
        )

    totals = summary.get("kernel_event_totals", {})
    if totals:
        out.append("<tr><th class='label'>Kernel event</th><th>Count</th></tr>")
        for key in sorted(totals):
            out.append(
                f"<tr><td class='label'>{html.escape(key)}</td>"
                f"<td>{html.escape(str(totals[key]))}</td></tr>"
            )
    out.append("</table>")
    return "".join(out)


def _render_findings_table(findings: Iterable[CorrelationFinding]) -> str:
    findings = list(findings)
    if not findings:
        return ""
    keys = sorted({key for f in findings for key in f.kernel_event_total})
    headers = ["rank", "global_step", "loss", "lookback"] + keys
    out = ["<table><thead><tr>"]
    for h in headers:
        out.append(f"<th>{html.escape(h)}</th>")
    out.append("</tr></thead><tbody>")
    for f in findings:
        cells = [
            str(f.target.rank),
            str(f.target.global_step),
            f"<span class='nan'>{f.target.loss}</span>",
            str(len(f.preceding_window)),
        ]
        for key in keys:
            cells.append(str(f.kernel_event_total.get(key, 0)))
        out.append("<tr>")
        for cell in cells:
            out.append(f"<td>{cell}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _render_finding_detail(idx: int, finding: CorrelationFinding) -> str:
    target = finding.target
    out = [
        f"<h3>Finding {idx + 1}: rank {target.rank}, step {target.global_step}</h3>",
        "<p>Iterations preceding the NaN (most recent last):</p>",
        "<table><thead><tr><th>step</th><th>loss</th><th>events</th></tr></thead><tbody>",
    ]
    for record in finding.preceding_window:
        out.append(
            "<tr>"
            f"<td>{record.global_step}</td>"
            f"<td>{record.loss}</td>"
            f"<td class='label'>{html.escape(json.dumps(record.kernel_summary))}</td>"
            "</tr>"
        )
    out.append(
        "<tr>"
        f"<td><b>{target.global_step}</b></td>"
        f"<td><span class='nan'>{target.loss}</span></td>"
        f"<td class='label'>{html.escape(json.dumps(target.kernel_summary))}</td>"
        "</tr>"
    )
    out.append("</tbody></table>")
    return "".join(out)


__all__ = ["generate_kernel_report"]
