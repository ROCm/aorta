#!/usr/bin/env python3
"""Emit the frozen demo payload: two objectives over the same real scenarios.

The dashboard is built against the schema in this module's ``--help`` and in
`docs/rl-cost-aware-objective.md`; keys and types are fixed and nothing is
omitted. Where a value genuinely does not exist it is ``null`` rather than a
plausible zero, because a zero here is indistinguishable from a measurement.

Everything it reads is already on disk:

* the archived probe matrices under ``/apps/vikhande/probe-gpu`` -- the
  uninitialised-workspace NaN (8 cells x 4 trials, one resolver) and the
  fp16-overflow companion that nothing resolves. The second is not decoration:
  it is the only scenario that exercises the empty-resolver-set rule, without
  which the objective teaches "always guess";
* the recorded Qwen3-8B rollouts from Slurm job 35729, re-scored under both
  objectives from their raw wire text;
* one live joint-axis episode's ``agent_log.jsonl``, if a real one was
  produced; otherwise the ``fake``-backend episode, and ``episode_backend``
  says which. A fake episode is never labelled real.

Usage
-----

    python examples/rl/emit_demo_payload.py --out /apps/vikhande/demo/demo_payload.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aorta.agent.llm import AUTOPSY_CATEGORIES  # noqa: E402
from aorta.agent.state import read_trial_results  # noqa: E402
from cost_aware_reward import (  # noqa: E402
    BASELINE,
    DEFAULT_BUDGET_CELLS,
    W_COST,
    W_FIX,
    W_TRIAGE,
    cells_for_proposal,
    score_cost_aware,
)
from fix_reward import FIX_WEIGHT, resolution_from_matrix  # noqa: E402
from rescore_e2e import analyse, form_pass, reference_policies  # noqa: E402
from triage_reward import Label, find_probe_cells, label_trials  # noqa: E402

PROBE_ROOT = Path("/apps/vikhande/probe-gpu")
NAN_MATRIX = PROBE_ROOT / "nan_matrix" / "PROBE-NAN-WS-MATRIX"
FP16_MATRIX = PROBE_ROOT / "nan_fp16" / "PROBE-NAN-FP16"
ROLLOUTS = PROBE_ROOT / "rl-nan" / "results"
EPISODE_ROOT = Path("/apps/vikhande/demo/episode")

#: The two registry entries whose own section header in
#: ``registry/mitigations.py`` calls them diagnostics: they buy visibility
#: rather than testing a cause.
DIAGNOSTICS = ["amd_log_level_4", "hip_launch_blocking"]

#: The demo's episode budget, and the ``--max-cells`` the loop was given. One
#: number, so the reward is not normalising against a budget nothing enforces.
BUDGET_CELLS = DEFAULT_BUDGET_CELLS

SYMPTOM = (
    "loss goes NaN in a microbatch accumulation workspace; an earlier "
    "unscaled fp16 phase overflowed, and the NaN appears in a tensor no "
    "numeric op touched"
)
SYMPTOM_FP16 = (
    "loss goes NaN partway through an unscaled fp16 forward; activation "
    "peaks grow ~2.5x per layer and pass 65504 at layer 11"
)

#: The five policy names the payload contract fixes.
ORACLE, MODEL = "oracle", "qwen3-8b"
PICK_FIRST, SHOTGUN, PROSE = (
    "abstain_and_pick_first",
    "abstain_and_shotgun",
    "always_prose",
)
#: rescore_e2e's own names for the same constants, so the mapping is explicit
#: rather than incidental.
_REFERENCE_NAME = {
    "oracle_contract_perfect": ORACLE,
    "abstain_and_pick_first": PICK_FIRST,
    "abstain_and_shotgun": SHOTGUN,
    "always_prose": PROSE,
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _cell_label(matrix: Path, cell: str) -> Label | None:
    path = matrix / cell
    docs = read_trial_results(path)
    return label_trials(docs, source=str(path)) if docs else None


def _trials_per_cell(matrix: Path) -> int | None:
    counts = {len(read_trial_results(c)) for c in find_probe_cells(matrix)}
    counts.discard(0)
    return max(counts) if len(counts) == 1 else (max(counts) if counts else None)


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


def build_scenarios() -> list[dict[str, Any]]:
    """The two real archived GPU scenarios, with their honest caveats attached.

    ``true_category`` is ``null`` on both, and that is a finding rather than a
    gap in this script. The symptom is a NaN and the cause is memory nobody
    wrote, so ``numeric_instability`` would name the symptom and assert a cause
    the archived matrix contradicts; ``illegal_mem`` would be wrong too,
    because an uninitialised read is legal and silent. Neither the shipped
    8-name autopsy set nor PR #484's 11-name set has a slot for
    uninitialised-memory / stale-buffer reuse.
    """
    out: list[dict[str, Any]] = []
    for matrix, scenario_id, symptom, notes in (
        (
            NAN_MATRIX,
            "uninit_workspace_nan",
            SYMPTOM,
            "Both mechanisms are real (unscaled fp16 overflow; uninitialised "
            "torch.empty workspace) but their COMPOSITION is authored, and "
            "_plant_residue pins which freed block is recycled -- without that "
            "pinning the reproducer passed 24/24. Deterministic: no trial "
            "disagreed with its cell across ~60 trials.",
        ),
        (
            FP16_MATRIX,
            "fp16_overflow_nan",
            SYMPTOM_FP16,
            "The honest negative companion: a real NaN that NOTHING in the "
            "registry resolves (21 mitigations measured, all fail). It is the "
            "only scenario here that exercises the empty-resolver-set rule.",
        ),
    ):
        if not matrix.is_dir():
            continue
        resolution = resolution_from_matrix(matrix, scenario_id)
        out.append(
            {
                "id": scenario_id,
                "symptom": symptom,
                # See the docstring: refusing to label is the correct answer.
                "true_category": None,
                "true_resolvers": sorted(resolution.resolvers),
                "cells_total": resolution.cells,
                "trials_per_cell": _trials_per_cell(matrix),
                "authored": True,
                "notes": notes,
            }
        )
    return out


# ---------------------------------------------------------------------------
# episode
# ---------------------------------------------------------------------------


def _observed_for_axes(
    matrix: Path, mitigation_axis: list[str], diagnostic_axis: list[str]
) -> list[dict[str, Any]]:
    """What the cells of the current grid actually reported."""
    rows: list[dict[str, Any]] = []
    for m in mitigation_axis:
        for d in diagnostic_axis:
            cell = f"{m}-{d}"
            label = _cell_label(matrix, cell)
            if label is None:
                continue
            rows.append(
                {
                    "cell": cell,
                    "verdict": label.verdict,
                    "detectors": sorted(label.cited_detectors),
                }
            )
    return rows


def terminal_observed(run_dir: Path) -> list[dict[str, Any]]:
    """Every cell the episode ran, with the verdict it ended with.

    ``steps[].observed`` is the evidence the policy had *when it chose*, so a
    cell's result lands on the following step and the last step's results land
    nowhere. On a converging episode that hides the punchline: the run reports
    a winner while no ``observed`` entry anywhere shows a cell passing.

    Enumerated from the **run directory's own cell directories**, which is the
    direct statement of "cells this episode genuinely ran and paid for" -- a
    cell directory holding ``trial_*/result.json`` exists if and only if the
    workload was executed there. Deliberately *not* the archived matrix and
    deliberately not the axes: padding this out with cells the policy never
    bought would make the search look more thorough than it was, and the
    grid's meaning is "squares this policy chose to spend on".

    Same element shape as ``steps[].observed``, and the verdict is recomputed
    through ``label_trials`` -- aorta's own resolver -- rather than read off
    the artifact, which is the same seam the rewards use.
    """
    rows: list[dict[str, Any]] = []
    for cell in find_probe_cells(run_dir):
        docs = read_trial_results(cell)
        if not docs:
            continue
        label = label_trials(docs, source=str(cell))
        rows.append(
            {
                "cell": cell.name,
                "verdict": label.verdict,
                "detectors": sorted(label.cited_detectors),
            }
        )
    return rows


def build_episode(run_dir: Path, scenario_id: str) -> dict[str, Any] | None:
    """Reconstruct one episode from an ``agent_log.jsonl`` the loop wrote.

    Read from the log rather than recomputed, so the cell counts in the
    payload are the ones the loop charged at the time -- the ``axis_growth``
    event exists for exactly this. Cells observed at each step are read from
    the run's own cell directories.
    """
    log = run_dir / "agent_log.jsonl"
    if not log.is_file():
        return None
    events = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    mitigation_axis, diagnostic_axis = [BASELINE], [BASELINE]
    steps: list[dict[str, Any]] = []
    cumulative = len(mitigation_axis) * len(diagnostic_axis)
    pending: dict[str, Any] | None = None
    outcome = "gave_up"
    winner: str | None = None

    for event in events:
        etype = event.get("type")
        if etype == "llm_step":
            pending = {
                "n": len(steps) + 1,
                "action": "conclude" if event.get("stop") else "probe",
                "mitigations": list(event.get("next_mitigations") or []),
                "diagnostics": list(event.get("next_diagnostics") or []),
                "category": event.get("category"),
                "hypothesis": event.get("hypothesis") or "",
                "confidence": float(event.get("confidence") or 0.0),
                "cells_this_step": 0,
                "cells_cumulative": cumulative,
                "observed": _observed_for_axes(
                    run_dir, mitigation_axis, diagnostic_axis
                ),
            }
        elif etype == "axis_growth" and pending is not None:
            pending["cells_this_step"] = int(event.get("cells_added") or 0)
            cumulative = int(event.get("cells_total") or cumulative)
            pending["cells_cumulative"] = cumulative
            mitigation_axis = list(event.get("mitigation_axis") or mitigation_axis)
            diagnostic_axis = list(event.get("diagnostic_axis") or diagnostic_axis)
            steps.append(pending)
            pending = None
        elif etype == "converged":
            outcome = "resolved"
            winner = event.get("winning_mitigation")
        elif etype == "baseline_pass":
            outcome = "baseline_pass"
        elif etype == "policy_stop":
            outcome = "budget_exhausted"
        elif etype == "search_stopped":
            outcome = "gave_up"

    if pending is not None:
        # A concluding step never grows an axis, so it has no axis_growth to
        # close it. It still spent nothing and still belongs in the episode.
        steps.append(pending)

    return {
        "scenario_id": scenario_id,
        "budget_cells": BUDGET_CELLS,
        "steps": steps,
        "outcome": outcome,
        "winning_mitigation": winner,
        # Built here rather than by the caller so `episode` and
        # `control_episode` cannot drift in shape.
        "terminal_observed": terminal_observed(run_dir),
    }


def refused_names(run_dir: Path) -> list[str]:
    """Names the cell budget refused, read straight off the ``axis_growth`` events.

    Separate from :func:`build_episode` because the payload's step shape is
    frozen and has no field for a refusal, but a reader has to be told: a
    refused name still appears in ``step.diagnostics`` (the policy did propose
    it) and contributes nothing to ``cells_this_step``.
    """
    log = run_dir / "agent_log.jsonl"
    if not log.is_file():
        return []
    refused: set[str] = set()
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") != "axis_growth":
            continue
        for key in ("rejected_mitigations", "rejected_diagnostics"):
            refused.update(str(n) for n in (event.get(key) or []))
    return sorted(refused)


def unresolved_names(run_dir: Path) -> list[str]:
    """Diagnostics the proposer's filter dropped, off the ``llm_step`` events.

    The counterpart to :func:`refused_names`, and the distinction is the whole
    reason both exist. A *refused* name was good and the cell budget had no
    room, so it still appears in ``step.diagnostics``. An *unresolved* name
    was never on the offered axis, so it is absent from ``step.diagnostics``
    and the episode would otherwise read as though the policy never proposed
    it. Reading one key and calling it the other would merge a cost decision
    with a name-resolution failure.
    """
    log = run_dir / "agent_log.jsonl"
    if not log.is_file():
        return []
    unresolved: set[str] = set()
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") not in ("llm_step", "search_stopped"):
            continue
        unresolved.update(
            str(n) for n in (event.get("unresolved_diagnostics") or [])
        )
    return sorted(unresolved)


def find_episode() -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    """Prefer a real-model episode; fall back to fake and say so.

    Returns ``(episode, backend, provenance_extras)``. ``backend`` is what
    actually produced the steps, never what was hoped for.
    """
    provenance_path = EPISODE_ROOT / "provenance.json"
    extras: dict[str, Any] = {}
    if provenance_path.is_file():
        extras = json.loads(provenance_path.read_text(encoding="utf-8"))

    for backend, run_dir in (
        ("litellm", EPISODE_ROOT / "live" / "DEMO-JOINT-LIVE"),
        ("fake", EPISODE_ROOT / "fake" / "DEMO-JOINT-FAKE"),
    ):
        episode = build_episode(run_dir, "uninit_workspace_nan")
        # A log with no decision steps is not an episode. The commonest cause
        # is a passing baseline, which short-circuits before the proposer is
        # ever called (loop.py, _baseline_passed) -- so it would show a
        # decision sequence that never happened.
        if episode and episode["steps"]:
            extras["episode_run_dir"] = str(run_dir)
            return episode, backend, extras
    return None, "fake", extras


#: What the control run is, in the payload's own voice. The renderer shows it
#: beside the real episode, so it has to say for itself that it is not one.
CONTROL_LABEL = (
    "CONTROL, not a model: the `fake` backend (FakeLLMProposer), which walks "
    "both axes in registry order rather than reading the evidence. Same "
    "reproducer, same 10-cell budget, same two diagnostics offered. It bought "
    "both diagnostics early, reached a 3x3 grid, and stopped on the cell "
    "budget without ever proposing the resolver -- so its terminal grid has "
    "no passing cell. The contrast with the real episode is the argument for "
    "the cost term: the same budget either finds the cause or is spent on "
    "evidence, depending on the policy."
)


def find_control_episode(
    scenario_id: str, episode_run_dir: str | None
) -> dict[str, Any] | None:
    """The ``fake``-backend run, as a second episode of identical shape.

    Returns None when the control run does not exist, or when it *is* the
    episode. That second case is the one worth guarding: if the live run never
    produced steps, ``find_episode`` falls back to the fake one, and emitting
    the same run twice would stage a comparison between a policy and itself.
    """
    run_dir = EPISODE_ROOT / "fake" / "DEMO-JOINT-FAKE"
    if episode_run_dir and Path(episode_run_dir) == run_dir:
        return None
    episode = build_episode(run_dir, scenario_id)
    if not episode or not episode["steps"]:
        return None
    return {**episode, "label": CONTROL_LABEL}


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------


def policy_reads_evidence(name: str) -> bool:
    """Only the oracle and the model read the input at all.

    The three constants emit the same object for every scenario, which is what
    makes them the right comparison: where a constant ties or beats the model,
    the reward has not measured the model.
    """
    return name in (ORACLE, MODEL)


def _oracle_triage_raw(label: Label) -> str:
    return json.dumps(
        {
            "category": "unknown",
            "hypothesis": "Uninitialised workspace read; the NaN is not numeric.",
            "next_mitigations": [],
            "confidence": 0.8,
            "stop": False,
            "verdict": label.verdict,
            "detectors": sorted(label.cited_detectors),
        }
    )


def build_policies(doc: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-score the recorded rollouts under both objectives.

    "Today" is the objective as it stands on ``feat/rl-fix-half-f1``:
    ``0.5 * form-ladder + 0.5 * F1-fix-credit``, computed by
    ``rescore_e2e.analyse`` rather than reimplemented here. Note that today's
    total has **no triage term at all** -- the form ladder subsumes it -- so
    ``today.triage`` reports the existing ``triage_reward`` score for the same
    policy for comparison, not a component of ``today.total``.

    "Proposed" is :func:`cost_aware_reward.score_cost_aware`.
    """
    meta = doc["meta"]
    offered = [c for c in meta["candidates"] if c not in meta["tried"] and c != "none"]
    resolution = resolution_from_matrix(NAN_MATRIX, "uninit_workspace_nan")

    # The loop state the rollouts were generated against: `tried` is already on
    # the mitigation axis, so its cells are already paid for.
    mitigation_axis = [BASELINE] + list(meta["tried"])
    diagnostic_axis = [BASELINE]
    cells_already = len(mitigation_axis) * len(diagnostic_axis)

    form = form_pass(doc)
    today = analyse(doc, resolution=resolution, form=form, formulation="f1")

    # Per-scenario triage ground truth, and the model's own recorded triage
    # answers -- real rollouts, one per scenario, from the same job.
    triage_rows = {row["scenario_id"]: row for row in doc.get("triage", [])}
    labels = {
        cell.name: label_trials(read_trial_results(cell), source=str(cell))
        for cell in find_probe_cells(NAN_MATRIX)
    }

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    policies: list[dict[str, Any]] = []
    # Filled as we go and folded into the caveats: the model's margin over the
    # best constant, decomposed, because part of it comes from a term the
    # constants do not compete on. See `_margin_caveats`.
    audit: dict[str, float] = {}

    # --- the model ---------------------------------------------------------
    model_rows: list[dict[str, Any]] = []
    model_rows_no_triage: list[Any] = []
    for row in form.rows:
        label = labels.get(row["scenario_id"])
        triage_row = triage_rows.get(row["scenario_id"])
        # Splice the model's own recorded triage answer onto its proposal, so
        # the joint reward reads one policy rather than two. Both halves come
        # from the same model at the same temperature in the same job.
        raw_obj: dict[str, Any] | None
        try:
            raw_obj = json.loads(row["raw"])
        except json.JSONDecodeError:
            raw_obj = None
        if isinstance(raw_obj, dict) and triage_row:
            answer = triage_row.get("answer") or {}
            raw_obj.setdefault("verdict", answer.get("verdict", ""))
            raw_obj.setdefault("detectors", answer.get("detectors", []))
            raw = json.dumps(raw_obj)
        else:
            raw = row["raw"]
        score = score_cost_aware(
            raw,
            offered=offered,
            label=label,
            resolution=resolution,
            mitigation_axis=mitigation_axis,
            diagnostic_axis=diagnostic_axis,
            budget_cells=BUDGET_CELLS,
            cells_already_spent=cells_already,
        )
        model_rows.append({"row": row, "score": score})
        # The same completion with the triage WEIGHT set to zero. The three
        # constants emit no verdict, so the triage term is one the model is
        # alone in scoring on; zeroing the weight -- for the model and for
        # every constant alike, below -- is how much of its margin that is
        # worth. Withholding the label instead would not be like-for-like:
        # the constants are scored against a label and collect the passing
        # cell's empty-empty attribution credit, so they have to lose the
        # same thing the model does.
        model_rows_no_triage.append(
            score_cost_aware(
                raw,
                offered=offered,
                label=labels.get(row["scenario_id"]),
                resolution=resolution,
                mitigation_axis=mitigation_axis,
                diagnostic_axis=diagnostic_axis,
                budget_cells=BUDGET_CELLS,
                cells_already_spent=cells_already,
                w_triage=0.0,
            )
        )

    audit["model_proposed"] = _mean([r["score"].total for r in model_rows])
    audit["model_proposed_no_triage"] = _mean(
        [s.total for s in model_rows_no_triage]
    )
    audit["model_today"] = today["model_mean_after"]

    policies.append(
        {
            "name": MODEL,
            "reads_evidence": True,
            "cells_spent": round(
                _mean([float(r["score"].cells_spent) for r in model_rows])
            ),
            "today": {
                "total": round(today["model_mean_after"], 4),
                "form": round(today["fix_half"]["model_form_mean"], 4),
                "triage": round(
                    _mean([float(r["reward"]) for r in doc.get("triage", [])]), 4
                ),
                "fix": today["fix_half"]["model_fix_rate"],
            },
            "proposed": {
                "total": round(_mean([r["score"].total for r in model_rows]), 4),
                "gate_passed": all(r["score"].gate_passed for r in model_rows),
                "triage": round(_mean([r["score"].triage for r in model_rows]), 4),
                "fix": round(_mean([r["score"].fix for r in model_rows]), 4),
                "cost_penalty": round(
                    _mean([r["score"].cost_penalty for r in model_rows]), 4
                ),
            },
        }
    )

    # --- the constants -----------------------------------------------------
    reference_raw = reference_policies(offered, resolution)
    for ref_name, raw in reference_raw.items():
        name = _REFERENCE_NAME.get(ref_name)
        if name is None:
            # honest_abstainer has no slot in the frozen payload contract.
            continue
        per_scenario: list[Any] = []
        per_scenario_no_triage: list[Any] = []
        for scenario_id, label in sorted(labels.items()):
            # The oracle is the only constant that reads the evidence, so it
            # is the only one given a verdict. The others emit none, and an
            # absent verdict scores as wrong rather than defaulting to right.
            scored_raw = _oracle_triage_raw(label) if name == ORACLE else raw
            if name == ORACLE:
                body = json.loads(scored_raw)
                body["next_mitigations"] = json.loads(raw)["next_mitigations"]
                scored_raw = json.dumps(body)
            common = dict(
                offered=offered,
                label=label,
                resolution=resolution,
                mitigation_axis=mitigation_axis,
                diagnostic_axis=diagnostic_axis,
                budget_cells=BUDGET_CELLS,
                cells_already_spent=cells_already,
            )
            per_scenario.append(score_cost_aware(scored_raw, **common))
            per_scenario_no_triage.append(
                score_cost_aware(scored_raw, w_triage=0.0, **common)
            )
        if not policy_reads_evidence(name):
            audit.setdefault("constants_no_triage", 0.0)
            audit["constants_no_triage"] = max(
                audit["constants_no_triage"],
                _mean([s.total for s in per_scenario_no_triage]),
            )
        ref = today["references"][ref_name]
        policies.append(
            {
                "name": name,
                "reads_evidence": policy_reads_evidence(name),
                "cells_spent": round(_mean([float(s.cells_spent) for s in per_scenario])),
                "today": {
                    "total": ref["reward"],
                    "form": ref["form_reward"],
                    "triage": round(
                        _mean([s.triage for s in per_scenario]), 4
                    ),
                    "fix": ref["fix_credit"],
                },
                "proposed": {
                    "total": round(_mean([s.total for s in per_scenario]), 4),
                    "gate_passed": all(s.gate_passed for s in per_scenario),
                    "triage": round(_mean([s.triage for s in per_scenario]), 4),
                    "fix": round(_mean([s.fix for s in per_scenario]), 4),
                    "cost_penalty": round(
                        _mean([s.cost_penalty for s in per_scenario]), 4
                    ),
                },
            }
        )

    constants = [p for p in policies if not p["reads_evidence"]]
    audit["best_constant_today"] = max(p["today"]["total"] for p in constants)
    audit["best_constant_proposed"] = max(p["proposed"]["total"] for p in constants)
    return policies, today, audit


def _margin_caveats(audit: dict[str, float]) -> list[str]:
    """State the part of the result that flatters the proposal, with numbers.

    The model's margin over the best constant widens by a lot under the new
    objective, and some of that widening is not the objective working -- it is
    the model being the only policy with a triage answer at all. Reported here
    rather than left for a reader to notice, and the triage-free figure is the
    like-for-like one.
    """
    today_margin = audit["model_today"] - audit["best_constant_today"]
    proposed_margin = audit["model_proposed"] - audit["best_constant_proposed"]
    fair_margin = (
        audit["model_proposed_no_triage"] - audit["constants_no_triage"]
    )
    return [
        f"MARGIN, today: model {audit['model_today']:.4f} vs best constant "
        f"{audit['best_constant_today']:.4f} -- a gap of {today_margin:.4f}. "
        "Thinner than the 0.0111 margin this project already learned to "
        "distrust.",
        f"MARGIN, proposed: model {audit['model_proposed']:.4f} vs best "
        f"constant {audit['best_constant_proposed']:.4f} -- a gap of "
        f"{proposed_margin:.4f}.",
        "FLATTERING: part of that widening is NOT the objective working. The "
        "model is the only policy with a verdict, so it alone earns on the 0.3 "
        "triage term. Setting the triage WEIGHT to zero for every policy "
        f"alike gives model {audit['model_proposed_no_triage']:.4f} vs best "
        f"constant {audit['constants_no_triage']:.4f}, a gap of "
        f"{fair_margin:.4f} -- still clear of every constant, but read this "
        "figure as the like-for-like one.",
        "Also inherited from triage_reward and left unchanged per rule 2: on "
        "the one cell that PASSES, citing no detectors is perfect attribution "
        "by convention, so a policy that reads nothing collects 0.4 of the "
        "triage term there. Correct for triage in isolation; worth knowing "
        "when the term is one of three.",
    ]


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------


def build_payload() -> dict[str, Any]:
    doc = json.loads(ROLLOUTS.read_text(encoding="utf-8"))
    meta = doc["meta"]
    matrix_meta_path = PROBE_ROOT / "rl-nan" / "run-meta.json"
    matrix_meta: dict[str, Any] = (
        json.loads(matrix_meta_path.read_text(encoding="utf-8"))
        if matrix_meta_path.is_file()
        else {}
    )
    scenarios = build_scenarios()
    episode, backend, extras = find_episode()
    control = find_control_episode(
        "uninit_workspace_nan", extras.get("episode_run_dir")
    )
    policies, today, audit = build_policies(doc)

    episode_caveats: list[str] = []
    if episode is not None:
        # Reconciliation is asserted, not assumed. terminal_observed is
        # enumerated from the cell directories and cells_cumulative is
        # replayed from the log's axis_growth events -- two independent
        # sources for one quantity, so they can disagree (a cell charged whose
        # trials never landed, or a resumed run inheriting cells). When they
        # do, the discrepancy is explained on screen rather than noticed.
        for label, ep in (("episode", episode), ("control_episode", control)):
            if ep is None:
                continue
            charged = ep["steps"][-1]["cells_cumulative"] if ep["steps"] else 0
            ran = len(ep["terminal_observed"])
            if ran != charged:
                episode_caveats.append(
                    f"COUNT MISMATCH in {label}: terminal_observed holds {ran} "
                    f"cells but the log charged {charged}. terminal_observed "
                    "lists only cells whose trials are on disk, so the "
                    "difference is cells charged whose workload did not "
                    "complete. cells_cumulative remains the cost of record."
                )
        episode_caveats.append(
            f"EPISODE PROVENANCE, separate from provenance.slurm_job: the "
            f"episode ran as Slurm job {extras.get('slurm_job')} on "
            f"{extras.get('node')}, budget {extras.get('max_probe_cells')} "
            f"cells, diagnostics offered "
            f"{extras.get('diagnostics_offered')}. provenance.slurm_job/node "
            f"describe the MATRIX the ground truth came from, which is a "
            f"different run on a different node."
        )
        refused = refused_names(Path(extras.get("episode_run_dir", "")))
        if refused:
            episode_caveats.append(
                "episode.steps[].mitigations/diagnostics are what the policy "
                "PROPOSED. A name the cell budget refused still appears there "
                "and contributes nothing to cells_this_step -- in this episode "
                f"the diagnostic(s) {refused} were proposed and refused. "
                "cells_this_step is the authoritative cost."
            )
        unresolved = unresolved_names(Path(extras.get("episode_run_dir", "")))
        if unresolved:
            episode_caveats.append(
                "episode.steps[].diagnostics is what SURVIVED the proposer's "
                f"filter. The policy also proposed {unresolved}, which was not "
                "on the offered diagnostic axis and was dropped before the "
                "axis was planned -- so it appears in NEITHER "
                "steps[].diagnostics NOR the refused list above. Terminal "
                "credit computed from steps[] alone would score a proposal "
                "the policy did not make."
            )

    caveats = _margin_caveats(audit) + episode_caveats + [
        "The uninitialised-workspace scenario is an AUTHORED COMPOSITION of two "
        "real mechanisms; _plant_residue pins which freed block is recycled. "
        "Nothing is a fault injected to make a number look good, but nothing "
        "was found in the wild either.",
        "EFFECTIVE SAMPLE SIZE IS ONE SCENARIO, not eight. The eight "
        "scenario_ids are eight cells of a single reproducer sharing a single "
        "ground truth, so the policy scores are 64 completions on one problem. "
        "A second unrelated resolving scenario is worth more than more samples "
        "on this one.",
        "The two score columns are NOT on the same scale: today's objective "
        "tops out at 1.0, the proposed one at triage+fix = "
        f"{W_TRIAGE + W_FIX:.1f} (reached only by spending zero cells, which is "
        "impossible since the baseline cell always runs). Compare the RANKING, "
        "not the values.",
        "today.total has no triage term -- today's objective is "
        f"{FIX_WEIGHT} * form-ladder + {FIX_WEIGHT} * F1-fix-credit. today.triage "
        "reports the existing triage_reward score for the same policy so the "
        "two columns can be read side by side; it is not a component of "
        "today.total.",
        "The three constant policies emit no verdict field, so their triage "
        "term is 0.0 by construction. An absent verdict is scored as a wrong "
        "verdict rather than defaulted to the label's -- defaulting would hand "
        "every policy that reads nothing a free 0.6.",
        "The proposed objective's floor at 0.0 creates a flat spot: several "
        "distinct policies that earn nothing all tie at 0.0, so GRPO sees no "
        "advantage between them. Recorded as a defect of the formulation, not "
        "fixed by moving a weight.",
        "Weights were pre-registered at 0.3 / 0.5 / 0.2 before anything was "
        "measured and have not been adjusted since.",
        "true_category is null on both scenarios because that is the correct "
        "answer: neither the shipped 8-name autopsy set nor PR #484's 11-name "
        "set has a slot for uninitialised-memory / stale-buffer reuse.",
    ]
    if episode is None:
        caveats.insert(
            0,
            "NO EPISODE WAS PRODUCED. Neither a real-model nor a fake-backend "
            "run left an agent_log.jsonl with decision steps.",
        )
    elif backend == "fake":
        caveats.insert(
            0,
            "THE EPISODE IS FAKE, not a real model. FakeLLMProposer walks both "
            "axes in registry order, so it demonstrates that the loop can grow "
            "the diagnostic axis and says nothing about whether a policy would "
            "choose to.",
        )

    return {
        "generated_at": _now(),
        "provenance": {
            # slurm_job and node describe ONE run: the matrix the scenarios'
            # ground truth came from. The episode ran as a separate job on a
            # separate node, and pairing that node with this job id would
            # describe a machine-run that never happened -- so the episode's
            # own provenance goes in the caveats instead.
            "slurm_job": matrix_meta.get("slurm_job_id"),
            "node": matrix_meta.get("node"),
            "matrix_path": str(NAN_MATRIX),
            "model": extras.get("model") or meta.get("model"),
            "temperature": meta.get("temperature_injected", 0.7),
            "episode_backend": backend,
        },
        "scenarios": scenarios,
        "action_space": {
            "mitigations": sorted(
                c for c in meta["candidates"] if c not in DIAGNOSTICS and c != "none"
            ),
            "diagnostics": list(DIAGNOSTICS),
        },
        "episode": episode,
        # Additive, and null when there is no control run to show. Same shape
        # as `episode` plus `label`.
        "control_episode": control,
        "policies": policies,
        "weights": {"triage": W_TRIAGE, "fix": W_FIX, "cost": W_COST},
        "caveats": caveats,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("/apps/vikhande/demo/demo_payload.json"))
    args = parser.parse_args(argv)

    payload = build_payload()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"  scenarios      {[s['id'] for s in payload['scenarios']]}")
    print(f"  action space   {len(payload['action_space']['mitigations'])} mitigations, "
          f"{len(payload['action_space']['diagnostics'])} diagnostics")
    for key in ("episode", "control_episode"):
        ep = payload[key]
        if ep is None:
            print(f"  {key:<14} (none)")
            continue
        observed = ep["terminal_observed"]
        passing = [r["cell"] for r in observed if r["verdict"] == "pass"]
        charged = ep["steps"][-1]["cells_cumulative"] if ep["steps"] else 0
        print(
            f"  {key:<14} steps={len(ep['steps'])} outcome={ep['outcome']} "
            f"terminal_observed={len(observed)} charged={charged} "
            f"{'RECONCILES' if len(observed) == charged else 'MISMATCH'}"
        )
        print(f"                 passing cells: {passing or '(none)'}")
    print(f"  autopsy set    {len(AUTOPSY_CATEGORIES)} categories")
    print()
    print(f"  {'policy':<24} {'reads':>5} {'cells':>5} "
          f"{'today':>7} {'proposed':>9}")
    for pol in payload["policies"]:
        print(
            f"  {pol['name']:<24} {str(pol['reads_evidence']):>5} "
            f"{pol['cells_spent']:>5} {pol['today']['total']:>7.4f} "
            f"{pol['proposed']['total']:>9.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
