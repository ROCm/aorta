from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aorta.cia.autopsy.adapters.base import AdapterArtifact, BundleContext

#: A quantity assigned or compared to a value that is not a number, written the
#: way a training loop writes it: the name, the operator, then the value.
#:
#: Anchoring on the value is the point. The pattern here was ``loss.*nan``,
#: which reads "checking loss for nan" as a run that has gone non-finite.
ASSIGNED_NONFINITE = re.compile(
    r"\b\w*(?:loss|grad|gradient|norm|residual|logit|score|activation)\w*\s*"
    r"(?:[=:]|\bis\b|\bbecame\b|\bwent\b|\bdiverged\s+to\b)\s*"
    r"[-+]?(?:nan|inf|infinity)\b",
    re.I,
)

#: Statements that a run *has* gone non-finite, as against statements about
#: whether it might. "NaN detection enabled" is the second kind.
DECLARED_NONFINITE = re.compile(
    r"\bnan\s+detected\b"
    r"|\bdetected\s+nan\b"
    r"|\bnon-?finite\s+(?:loss|gradient|grad|value|activation)s?\b"
    r"|\bloss\s+(?:is|became|went)\s+non-?finite\b",
    re.I,
)

#: What turns either of the above into a report that nothing is wrong. "no
#: non-finite values found" is a clean run saying so.
NEGATED = re.compile(
    r"\b(?:no|not|without|never|zero)\b[^.\n]{0,24}\b(?:nan|non-?finite)\b",
    re.I,
)

#: Kept for callers that import it; the scan uses the three above.
NAN_PATTERNS = [ASSIGNED_NONFINITE, DECLARED_NONFINITE]


@dataclass(frozen=True)
class StderrScan:
    alert: bool
    hits: list[tuple[int, str]]
    signal: str


def scan_stderr_text(text: str) -> StderrScan:
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if NEGATED.search(line):
            continue
        if ASSIGNED_NONFINITE.search(line) or DECLARED_NONFINITE.search(line):
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
                        "Watchdog saw a non-finite value in the training log — "
                        "re-run this job's own recipe to see whether it "
                        "reproduces and under which settings."
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
