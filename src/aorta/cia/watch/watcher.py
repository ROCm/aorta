from __future__ import annotations

from pathlib import Path

import dspy

from aorta.cia.llm import ensure_configured

# ---------------------------------------------------------------------------
# Tools available to the ReAct loop
# ---------------------------------------------------------------------------

def read_file_tail(path: str, lines: int = 80) -> str:
    """Read the last N lines of a file for extra context. Does not advance cursor."""
    try:
        p = Path(path)
        if not p.is_file():
            return f"[file not found: {path}]"
        text = p.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])
    except Exception as e:
        return f"[error reading {path}: {e}]"


def list_job_files(job_dir: str) -> str:
    """List files in job dir with sizes and mtimes — spot new log files."""
    import subprocess
    try:
        r = subprocess.run(
            ["find", job_dir, "-maxdepth", "4", "-type", "f",
             "-printf", "%T@ %s %p\n"],
            capture_output=True, text=True, timeout=10,
        )
        lines = sorted(r.stdout.strip().splitlines(), reverse=True)[:30]
        out = []
        for line in lines:
            parts = line.split(" ", 2)
            if len(parts) == 3:
                out.append(f"{parts[2]}  ({int(float(parts[1])):,} bytes)")
        return "\n".join(out) or "[empty directory]"
    except Exception as e:
        return f"[error: {e}]"


def count_repeated_lines(text: str) -> dict:
    """Count line frequencies — high repeat count signals a hang (same step printed over and over)."""
    from collections import Counter
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    counts = Counter(lines)
    top = counts.most_common(5)
    max_repeats = top[0][1] if top else 0
    return {
        "max_repeats": max_repeats,
        "likely_hang": max_repeats >= 5,
        "top_repeated": [{"line": l, "count": c} for l, c in top[:3]],
    }


# ---------------------------------------------------------------------------
# DSPy signature + module
# ---------------------------------------------------------------------------

class WatchAssessment(dspy.Signature):
    """
    You are monitoring a live GPU training job on an AMD GPU cluster.
    You receive the NEW log content since the last check — not the full history.
    The operator has stated what healthy training should look like.

    Your job:
    1. Read the new content carefully.
    2. Use tools to get more context if the new content is ambiguous or very short.
    3. Judge whether each expectation is being met.
    4. Alert only when an expectation is CLEARLY violated — not on startup noise,
       expected warnings, or one-off messages.

    Signal slugs to use:
    - WATCH_CLEAN          : everything looks healthy
    - WATCH_NUMERIC_NAN    : NaN, non-finite, or loss divergence detected
    - WATCH_HANG           : training appears stalled (no step progress)
    - WATCH_LOSS_STALL     : loss has plateaued and is not decreasing as expected
    - WATCH_THROUGHPUT_LOW : throughput/speed dropped significantly below expectation
    - WATCH_OOM            : out-of-memory error
    - WATCH_UNKNOWN_ERROR  : clear error but doesn't fit above categories
    """
    new_content: str = dspy.InputField(
        desc="New log lines since last poll, labelled by filename. May be empty if no new output.")
    job_context: str = dspy.InputField(
        desc="job_id, node, recipe, elapsed_time_sec, total_bytes_seen so far")
    expectations: str = dspy.InputField(
        desc="Operator-stated expectations for healthy training (plain English)")

    healthy: bool = dspy.OutputField(desc="True if no expectation is clearly violated")
    signal: str = dspy.OutputField(desc="One signal slug from the list above")
    confidence: float = dspy.OutputField(desc="0.0-1.0 — how certain you are of this assessment")
    evidence: str = dspy.OutputField(
        desc="Specific log lines (with filename) that show the violation, or 'none' if healthy")
    assessment: str = dspy.OutputField(
        desc="One paragraph: what training looks like right now based on the new content")


class LogWatcher(dspy.Module):
    def __init__(self):
        ensure_configured()
        self.react = dspy.ReAct(
            WatchAssessment,
            tools=[read_file_tail, list_job_files, count_repeated_lines],
            max_iters=4,
        )

    def forward(self, new_content: str, job_context: str, expectations: str) -> dspy.Prediction:
        return self.react.forward(
            new_content=new_content,
            job_context=job_context,
            expectations=expectations,
        )
