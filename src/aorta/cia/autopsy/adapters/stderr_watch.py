from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aorta.cia.autopsy.adapters.base import AdapterArtifact, BundleContext

NAN_PATTERNS = [
    re.compile(r"\bloss=nan\b", re.I),
    re.compile(r"\bloss.*\bnan\b", re.I),
    re.compile(r"non-?finite", re.I),
    re.compile(r"NaN detected", re.I),
    re.compile(r"residual.*nan", re.I),
    re.compile(r"numeric_silent", re.I),
]


@dataclass(frozen=True)
class StderrScan:
    alert: bool
    hits: list[tuple[int, str]]
    signal: str


def scan_stderr_text(text: str) -> StderrScan:
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if any(p.search(line) for p in NAN_PATTERNS):
            hits.append((i, line.strip()))
    return StderrScan(
        alert=bool(hits),
        hits=hits,
        signal="WATCH_NUMERIC_NAN" if hits else "WATCH_CLEAN",
    )


class StderrWatchAdapter:
    """Parse training stderr/watch log for silent numeric (NaN) signatures."""

    adapter_id = "stderr_watch"

    def collect(self, ctx: BundleContext) -> AdapterArtifact:
        stderr_path = ctx.path("stderr")
        if stderr_path is None or not stderr_path.is_file():
            return AdapterArtifact(adapter=self.adapter_id)

        text = stderr_path.read_text(encoding="utf-8", errors="replace")
        scan = scan_stderr_text(text)
        rel = _bundle_rel(ctx, stderr_path)
        evidence: list[dict[str, Any]] = []

        if scan.alert:
            line_start, excerpt = scan.hits[0]
            line_end = scan.hits[-1][0] if len(scan.hits) > 1 else line_start
            evidence.append(
                {
                    "uri": rel,
                    "line_start": line_start,
                    "line_end": line_end,
                    "excerpt": excerpt[:500],
                    "adapter": self.adapter_id,
                    "signal": scan.signal,
                }
            )
            next_probes = [
                {
                    "tool": "aorta sweep run",
                    "reason": (
                        "Watchdog saw NaN/non-finite in training log — run "
                        "Residual-NaN-Repro matrix to isolate TF32 vs deterministic."
                    ),
                    "overhead_class": "medium",
                }
            ]
        else:
            next_probes = []
            evidence.append(
                {
                    "uri": rel,
                    "line_start": 1,
                    "line_end": min(3, len(text.splitlines()) or 1),
                    "excerpt": "no NaN signature in watch log",
                    "adapter": self.adapter_id,
                    "signal": scan.signal,
                }
            )

        return AdapterArtifact(
            adapter=self.adapter_id,
            evidence=evidence,
            signals=[scan.signal] if scan.alert else [],
            summary={"alert": scan.alert, "hit_count": len(scan.hits)},
            next_probes=next_probes,
        )


def _bundle_rel(ctx: BundleContext, path: Path) -> str:
    try:
        return str(path.relative_to(ctx.root)).replace("\\", "/")
    except ValueError:
        return path.name
