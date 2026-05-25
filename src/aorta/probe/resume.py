"""Resume helper for ``aorta probe`` (issue #188 Phase 1).

A trial is considered "complete" iff its ``trial_<n>/result.json`` exists,
parses as JSON, and carries a non-empty ``verdict`` field. Anything else --
missing file, empty file, syntactically invalid JSON, JSON without
``verdict``, or ``verdict == ""`` -- is treated as incomplete so the runner
re-executes the trial.

This module is deliberately tiny and pure (no FS writes, no env mutation)
so the runner can call it per trial without performance concern. The
contract is fixed by FR 1.4 in ``docs/plans/aorta-probe-188-rubric.md``.
"""

from __future__ import annotations

import json
from pathlib import Path


def is_trial_complete(trial_dir: Path) -> bool:
    """Return True iff the trial directory carries a valid completion marker.

    A trial completes when ``trial_dir/result.json`` exists, parses as
    JSON, and contains a non-empty ``verdict`` field. Partial or
    truncated writes (zero-byte file, half a ``{``, missing
    ``verdict``, blank ``verdict``) are treated as incomplete so the
    runner re-executes the trial -- the rubric's "completed -> skip,
    truncated -> re-run" contract.

    Any unexpected error (PermissionError, UnicodeDecodeError, etc.)
    is treated as "incomplete" rather than re-raised: the failure
    surfaces when the re-execution itself fails, but the resume helper
    never crashes the whole probe run.
    """
    result_path = trial_dir / "result.json"
    if not result_path.is_file():
        return False
    try:
        text = result_path.read_text(encoding="utf-8")
    except OSError:
        return False
    if not text.strip():
        return False
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(parsed, dict):
        return False
    verdict = parsed.get("verdict")
    if not isinstance(verdict, str) or not verdict:
        return False
    return True


__all__ = ["is_trial_complete"]
