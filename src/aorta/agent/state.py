"""Append-only agent event log and resume (``wake``).

State lives under ``<output>/<ticket>/agent_log.jsonl``. Each line is a
JSON object. The agent process is stateless across restarts: ``wake()``
replays the log and scans existing probe cell verdicts to rebuild
``AgentState``.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_LOG_NAME = "agent_log.jsonl"

# Cells are named ``{mitigation}-{diagnostic}``; both axes use "none" for the
# no-op baseline, so the baseline cell is ``none-none``.
_BASELINE_NAME = "none"


def winning_mitigation(cell_name: str, verdict: str | None) -> str | None:
    """Return the mitigation iff ``cell_name`` is a genuine convergence win.

    A win is a *non-baseline mitigation* passing with the *baseline
    diagnostic* -- i.e. a ``{mitigation}-none`` cell where ``mitigation !=
    "none"``. A pass on a diagnostic-only cell (``none-xnack``) or on a
    mitigation+diagnostic cell (``tf32_off-xnack``) is NOT attributable to the
    mitigation alone, and ``none`` is never a "winning mitigation". The
    diagnostic is taken as the last ``-`` segment so mitigation names that
    themselves contain ``-`` still parse correctly.
    """
    if verdict != "pass" or "-" not in cell_name:
        return None
    mitigation, diagnostic = cell_name.rsplit("-", 1)
    if mitigation == _BASELINE_NAME or diagnostic != _BASELINE_NAME:
        return None
    return mitigation


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class AgentState:
    """Reconstructed agent memory for a ticket run."""

    ticket: str
    tried_mitigations: list[str] = field(default_factory=list)
    last_category: str = "unknown"
    last_hypothesis: str = ""
    iterations_completed: int = 0
    winning_mitigation: str | None = None
    converged: bool = False


def agent_log_path(run_dir: Path) -> Path:
    return run_dir / _LOG_NAME


def append_log_event(run_dir: Path, event_type: str, payload: dict[str, Any]) -> None:
    """Append one JSON line to ``agent_log.jsonl`` (owner-only, 0600).

    The log records argv/symptom/hypothesis, which can carry sensitive
    data, so the file is created owner-only like ``probe.env`` (FR 1.10)
    rather than at the umask default. ``0o600`` in the ``os.open`` mode is
    only ever narrowed by umask (never widened), and a one-time ``chmod``
    on creation covers platforms that ignore the create mode bits.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    # Reserved metadata wins: spreading payload first means a payload key
    # named "ts"/"type" can't clobber the event envelope.
    record = {**payload, "ts": _utc_now_iso(), "type": event_type}
    path = agent_log_path(run_dir)
    fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    # Enforce owner-only on EVERY write (best-effort), not just on create: a
    # log left by an older version or created manually with broader perms must
    # be narrowed, since it carries sensitive argv/symptom/hypothesis. chmod
    # can fail on exotic filesystems or cross-owner files -- never let that
    # abort logging.
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        log.debug("could not enforce 0600 on %s", path)


def _read_log_events(run_dir: Path) -> list[dict[str, Any]]:
    path = agent_log_path(run_dir)
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # The log is the resume source of truth; a truncated/corrupt
            # tail (e.g. interrupted write) must not abort replay.
            log.warning("skipping malformed line %d in %s", lineno, path)
            continue
    return events


def read_trial_results(cell_dir: Path) -> list[dict[str, Any]]:
    """Parse every ``trial_*/result.json`` under a cell dir, ordered by index.

    Multi-trial probe cells write one ``result.json`` per trial; a verdict
    decision must consider all of them, not just ``trial_0`` (a later trial
    can fail even when ``trial_0`` passed).
    """
    indexed: list[tuple[int, dict[str, Any]]] = []
    for trial_dir in cell_dir.glob("trial_*"):
        if not trial_dir.is_dir():
            continue
        result_path = trial_dir / "result.json"
        if not result_path.is_file():
            continue
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        try:
            index = int(trial_dir.name.removeprefix("trial_"))
        except ValueError:
            index = 0
        indexed.append((index, data))
    indexed.sort(key=lambda item: item[0])
    return [data for _, data in indexed]


def aggregate_cell_verdict(trial_results: list[dict[str, Any]]) -> str | None:
    """Reduce per-trial verdicts to a single cell verdict.

    A cell is ``pass`` only if *every* trial passed; otherwise the first
    non-``pass`` verdict is returned as the representative failure. Returns
    ``None`` when no trial recorded a verdict.
    """
    verdicts = [
        data["verdict"]
        for data in trial_results
        if isinstance(data.get("verdict"), str) and data["verdict"]
    ]
    if not verdicts:
        return None
    for verdict in verdicts:
        if verdict != "pass":
            return verdict
    return "pass"


def _scan_cell_verdicts(run_dir: Path) -> dict[str, str]:
    """Map cell name -> aggregated verdict across all trials when present."""
    verdicts: dict[str, str] = {}
    if not run_dir.is_dir():
        return verdicts
    # Sorted so the winning_mitigation picked during wake() is deterministic
    # across filesystems (matches loop._read_cell_summaries).
    for cell_dir in sorted(run_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        verdict = aggregate_cell_verdict(read_trial_results(cell_dir))
        if verdict:
            verdicts[cell_dir.name] = verdict
    return verdicts


def wake(run_dir: Path, *, ticket: str) -> AgentState:
    """Rebuild :class:`AgentState` from the event log and on-disk probe cells."""
    state = AgentState(ticket=ticket)
    for event in _read_log_events(run_dir):
        etype = event.get("type")
        if etype == "mitigation_tried":
            name = event.get("mitigation")
            if isinstance(name, str) and name not in state.tried_mitigations:
                state.tried_mitigations.append(name)
        elif etype == "llm_step":
            cat = event.get("category")
            if isinstance(cat, str):
                state.last_category = cat
            hyp = event.get("hypothesis")
            if isinstance(hyp, str):
                state.last_hypothesis = hyp
        elif etype == "iteration_complete":
            state.iterations_completed += 1
        elif etype == "converged":
            state.converged = True
            win = event.get("winning_mitigation")
            if isinstance(win, str):
                state.winning_mitigation = win

    verdicts = _scan_cell_verdicts(run_dir)
    for cell_name, verdict in verdicts.items():
        if "-" not in cell_name:
            continue
        mitigation = cell_name.rsplit("-", 1)[0]
        if mitigation != _BASELINE_NAME and mitigation not in state.tried_mitigations:
            state.tried_mitigations.append(mitigation)
        win = winning_mitigation(cell_name, verdict)
        if win is not None and state.winning_mitigation is None:
            state.winning_mitigation = win
            state.converged = True

    return state


__all__ = [
    "AgentState",
    "agent_log_path",
    "aggregate_cell_verdict",
    "append_log_event",
    "read_trial_results",
    "wake",
    "winning_mitigation",
]
