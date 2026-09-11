#!/usr/bin/env python3
"""Harvest ``agent_log.jsonl`` into trajectory rows.

``aorta agent mitigate`` already writes an append-only decision log: one
``llm_step`` per iteration carrying category, hypothesis, ``next_mitigations``,
confidence and stop, a ``mitigation_tried`` per accepted name, and a terminal
event naming the outcome. That is a trajectory in everything but shape, so this
is a reshape over a run that already happened, not a second run.

Two things this file is deliberate about.

**A trajectory with no decision in it is not a trajectory.** ``run_agent_loop``
short-circuits on a passing baseline (``loop.py:399``) *before* the proposer is
ever consulted, so a clean control produces exactly two events --
``session_start`` and ``baseline_pass`` -- and zero ``llm_step``. Emitting that
as a zero-step row and counting it alongside real searches would inflate the
corpus with rows that carry no supervision. Every row therefore carries
``trainable`` and a reason, and the summary counts the two kinds separately.

**Whose decision was it?** aorta#449: a loop that records a name-resolution
failure as a genuine ``agent_stop`` misattributes its own stopping, and a
reward computed over that trajectory is computed over a corrupted signal.
``LiteLLMProposer`` filters unregistered names out of ``next_mitigations``
*before* ``validate_step`` sees them, so a hallucinated name leaves an empty
list; ``loop.py:451`` treats an empty list as a stop regardless of
``step.stop``, and ``_resolve_stop_outcome`` falls through to
``agent_requested``.

:func:`check_issue_449` separates the two cases the log can actually tell
apart. When the recorded step says ``stop: false`` and the loop stopped
anyway, the log contradicts itself and #449 is *proven*. When the step says
``stop: true`` with no reason, a dropped name and a genuine conclusion agree
on every recorded field -- that is the issue's own point -- so the row is
marked **indeterminate** rather than cleared. ``FakeLLMProposer`` selects from
the candidate set and cannot emit an unregistered name, so a fake-backend run
is structurally immune and clears nothing about a real one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TRAJECTORY_SCHEMA = "aorta.rl_trajectory/0.1"

#: Hypothesis texts ``_safe_stop`` writes verbatim (``agent/llm.py:266,273``).
#: Substring match, because the unparseable-response one interpolates the
#: exception.
_SAFE_STOP_MARKERS = ("Empty LLM response", "LLM returned unparseable response")

_TERMINAL_TYPES = frozenset(
    {
        "baseline_pass",
        "converged",
        "search_stopped",
        "approval_required",
        "policy_stop",
        "registry_error",
        "error",
    }
)


def read_log(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"warning: {path}:{lineno} is not JSON; skipped", file=sys.stderr)
    return events


def read_cells(run_dir: Path) -> list[dict[str, Any]]:
    """Final per-cell verdicts, read from the probe tree beside the log.

    The log records decisions, not outcomes; the cells are the outcomes. A
    trajectory needs both or the reward has nothing to score the decision
    against.
    """
    cells: list[dict[str, Any]] = []
    for cell_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        trials = sorted(cell_dir.glob("trial_*/result.json"))
        if not trials:
            continue
        docs = [json.loads(p.read_text()) for p in trials]
        failures: list[str] = []
        for doc in docs:
            for det in doc.get("failure_detectors_fired") or []:
                if det not in failures:
                    failures.append(det)
        verdicts = [d.get("verdict") for d in docs]
        # fail > error > pass across trials, matching aggregate_cell_verdict:
        # one reproducing trial makes the cell a reproduction.
        verdict = "fail" if "fail" in verdicts else "error" if "error" in verdicts else "pass"
        cells.append(
            {
                "cell_name": docs[0].get("cell_name", cell_dir.name),
                "verdict": verdict,
                "trial_verdicts": verdicts,
                "n_trials": len(docs),
                "failure_detectors_fired": failures,
                "exit_codes": [d.get("exit_code") for d in docs],
                "env": docs[0].get("env") or {},
            }
        )
    return cells


def check_issue_449(events: list[dict[str, Any]], backend: str | None) -> dict[str, Any]:
    """Did this loop misattribute its own stopping? See aorta#449."""
    proven: list[dict[str, Any]] = []
    indeterminate: list[dict[str, Any]] = []

    steps = [e for e in events if e.get("type") == "llm_step"]
    stopped = next((e for e in events if e.get("type") == "search_stopped"), None)
    agent_stop = bool(stopped and stopped.get("outcome") == "agent_stop")

    for index, step in enumerate(steps):
        empty = not (step.get("next_mitigations") or [])
        hypothesis = str(step.get("hypothesis") or "")

        if agent_stop and empty and not step.get("stop"):
            proven.append(
                {
                    "signature": "loop_stopped_while_the_step_said_stop_false",
                    "llm_step_index": index,
                    "hypothesis": hypothesis,
                    "detail": (
                        "the recorded step sets stop=false with an empty "
                        "next_mitigations, and the loop still recorded agent_stop. "
                        "That is aorta#449 exactly: names the proposer asked for "
                        "were filtered out before validation, and the empty list "
                        "alone satisfied the stop check"
                    ),
                }
            )
        elif any(marker in hypothesis for marker in _SAFE_STOP_MARKERS):
            proven.append(
                {
                    "signature": "backend_failure_recorded_as_agent_decision",
                    "llm_step_index": index,
                    "hypothesis": hypothesis,
                    "detail": (
                        "_safe_stop turned an empty or unparseable model reply into "
                        "stop_reason='agent_requested', so the log attributes a "
                        "backend failure to the agent (adjacent to #449, same class)"
                    ),
                }
            )
        elif agent_stop and step.get("stop") and step.get("stop_reason") is None:
            indeterminate.append(
                {
                    "signature": "agent_stop_indistinguishable_from_dropped_name",
                    "llm_step_index": index,
                    "confidence": step.get("confidence"),
                    "detail": (
                        "the step sets stop=true with no stop_reason, so the reason "
                        "was inferred by _resolve_stop_outcome. #449's own "
                        "reproduction shows a dropped name and a genuine conclusion "
                        "agreeing on every recorded field here, so the log cannot "
                        "separate them"
                    ),
                }
            )

    if agent_stop and not steps:
        indeterminate.append(
            {
                "signature": "agent_stop_without_any_recorded_step",
                "detail": "search_stopped reports agent_stop but no llm_step was logged",
            }
        )

    # Not #449, but the same question -- who stopped this loop.
    loop_failures = [
        {"type": e.get("type"), "reason": e.get("reason")}
        for e in events
        if e.get("type") in {"policy_stop", "registry_error", "error"}
    ]

    return {
        "bitten": bool(proven),
        "indeterminate": bool(indeterminate),
        "evidence": proven,
        "indeterminate_evidence": indeterminate,
        # FakeLLMProposer picks from `candidates`, so the filter that causes
        # #449 has nothing to drop. A clean fake-backend run is not evidence
        # that a litellm/openai/vllm-backed run would be clean.
        "backend_can_reach_449": backend not in (None, "fake"),
        "backend": backend,
        "loop_failure_events": loop_failures,
    }


def build_trajectory(run_dir: Path) -> dict[str, Any] | None:
    log_path = run_dir / "agent_log.jsonl"
    if not log_path.is_file():
        return None
    events = read_log(log_path)
    if not events:
        return None

    session = next((e for e in events if e.get("type") == "session_start"), {})
    terminal = next((e for e in reversed(events) if e.get("type") in _TERMINAL_TYPES), {})
    steps = [e for e in events if e.get("type") == "llm_step"]
    tried = [e.get("mitigation") for e in events if e.get("type") == "mitigation_tried"]
    cells = read_cells(run_dir)

    outcome = terminal.get("outcome") or terminal.get("type") or "unknown"
    trainable = bool(steps)
    if trainable:
        reason = f"{len(steps)} decision step(s) recorded"
    elif outcome == "baseline_pass":
        reason = (
            "baseline_pass: run_agent_loop short-circuits a passing none-none cell "
            "before consulting the proposer, so no decision was ever made"
        )
    else:
        reason = f"no llm_step events recorded (outcome={outcome})"

    return {
        "schema": TRAJECTORY_SCHEMA,
        "trajectory_id": run_dir.name,
        "ticket": session.get("ticket"),
        "source": str(log_path),
        "symptom": session.get("symptom"),
        "argv": session.get("argv") or [],
        "llm_backend": session.get("llm_backend"),
        "started_at": session.get("ts"),
        "ended_at": events[-1].get("ts"),
        "outcome": outcome,
        "stop_reason": terminal.get("stop_reason"),
        "winning_mitigation": terminal.get("winning_mitigation"),
        "converged": outcome == "converged",
        "n_events": len(events),
        "n_llm_steps": len(steps),
        "mitigations_tried": tried,
        "steps": [
            {
                "index": i,
                "ts": s.get("ts"),
                "category": s.get("category"),
                "hypothesis": s.get("hypothesis"),
                "next_mitigations": s.get("next_mitigations") or [],
                "confidence": s.get("confidence"),
                "stop": s.get("stop"),
                "stop_reason": s.get("stop_reason"),
            }
            for i, s in enumerate(steps)
        ],
        "cells": cells,
        "trainable": trainable,
        "trainable_reason": reason,
        "issue_449": check_issue_449(events, session.get("llm_backend")),
    }


def find_run_dirs(root: Path) -> list[Path]:
    if (root / "agent_log.jsonl").is_file():
        return [root]
    return sorted(p.parent for p in root.rglob("agent_log.jsonl"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, action="append",
                        help="Agent output tree (repeatable). Searched for agent_log.jsonl.")
    parser.add_argument("--out", type=Path, required=True, help="Trajectory JSONL to write.")
    parser.add_argument("--exclude-untrainable", action="store_true",
                        help="Drop rows with no decision steps instead of emitting them.")
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for root in args.results:
        if not root.exists():
            print(f"warning: {root} does not exist; skipped", file=sys.stderr)
            continue
        for run_dir in find_run_dirs(root):
            row = build_trajectory(run_dir)
            if row is not None:
                rows.append(row)

    kept = [r for r in rows if r["trainable"]] if args.exclude_untrainable else rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    trainable = sum(1 for r in rows if r["trainable"])
    bitten = sum(1 for r in rows if r["issue_449"]["bitten"])
    unclear = sum(1 for r in rows if r["issue_449"]["indeterminate"])
    exposed = sum(1 for r in rows if r["issue_449"]["backend_can_reach_449"])
    print(f"wrote {len(kept)} trajectory row(s) to {args.out}")
    print(f"  found        : {len(rows)}")
    print(f"  trainable    : {trainable}   (>=1 recorded decision step)")
    print(f"  untrainable  : {len(rows) - trainable}")
    print(f"  aorta#449    : {bitten} proven, {unclear} indeterminate")
    print(f"  #449-exposed : {exposed} row(s) ran a backend that can hallucinate a name")
    for row in rows:
        flag = "449" if row["issue_449"]["bitten"] else "  ?" if row["issue_449"]["indeterminate"] else "   "
        print(
            f"  [{flag}] {row['trajectory_id']:<32} outcome={row['outcome']:<22} "
            f"steps={row['n_llm_steps']} cells={len(row['cells'])} "
            f"trainable={row['trainable']} backend={row['llm_backend']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
