"""The probe agent's corpus: one example per ``propose()`` call, labelled by what ran.

This is the cheapest of the three to build, for the reason the plan gives: the
sweep already records the answer. ``run_agent_loop`` appends every decision to
``agent_log.jsonl`` -- the step it took, the mitigations it added, and, when a
cell finally passed, a ``converged`` event naming the mitigation that cleared it.
That name is a measured fact about a probe cell, not another model's opinion,
which is what makes it a label.

The state is reconstructed rather than replayed, and the difference matters. The
log records what the proposer *said*, not what it was shown, so the cell
summaries here are read back off the run directory with the same reader the loop
uses (``read_trial_results`` and ``aggregate_cell_verdict``) and then filtered to
the mitigations that had been tried by the step being labelled. That reproduces
what the proposer saw at that point without the loop having had to log it.

**One of the three labels the plan asks for is not derivable, and it is left
out rather than approximated.** The plan wants "the final category the sweep
reached" alongside the winning mitigation. Nothing in a probe run records a
category that did not come from a model: ``agent_log.jsonl``'s ``llm_step`` holds
the proposer's own category, and ``agent_report.md`` prints that same field. Using
it would train the classifier on the output of the thing it replaces. The only
independent category is an Autopsy verdict, so ``proposer_category`` examples are
emitted for a run that has a ``report.json`` beside it and skipped, counted, for
every run that does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aorta.agent.llm import (
    CLASSIFIER_CATEGORY_QUESTION,
    CLASSIFIER_MITIGATION_QUESTION,
    CLASSIFIER_STOP_QUESTION,
    CLASSIFIER_STOP_WHEN_FALSE,
    CLASSIFIER_STOP_WHEN_TRUE,
    PROBE_CATEGORIES,
)
from aorta.local_classifier.corpus.schema import BuildResult, LabelledExample, iter_json_lines
from aorta.local_classifier.predictor import Choice, Noul

#: The cell the loop treats as the no-op baseline, and the mitigation name that
#: is never a candidate. Both spelled as ``aorta.agent`` spells them.
_BASELINE_CELL = "none-none"
_BASELINE_MITIGATION = "none"

#: The questions come from ``aorta.agent.llm``, not from here.
#:
#: This module wrote its own phrasing of all three first, and that was a latent
#: defect rather than duplication for its own sake: a corpus labelled against one
#: wording and a proposer asking another does not fail, it answers slightly worse
#: for a reason nobody would go looking for. The questions belong beside the
#: decision they are about -- which is the probe agent's, not the encoder's -- and
#: this module already imported :data:`PROBE_CATEGORIES` from there, so the import
#: direction was settled before the question came up. The seam in
#: :mod:`aorta.local_classifier.predictor` deliberately holds none of them: it serves three
#: tracks, and a string registry is what it would become if each of them put its
#: own phrasing there.
_STOP = Noul(
    question=CLASSIFIER_STOP_QUESTION,
    when_true=CLASSIFIER_STOP_WHEN_TRUE,
    when_false=CLASSIFIER_STOP_WHEN_FALSE,
)

#: The fourth question, and the one exception to the paragraph above: it is
#: defined here because it exists nowhere else yet.
#:
#: ``aorta.agent.llm`` holds the three questions Track B implemented. This one
#: belongs beside them and cannot be put there yet, because the corpus is the
#: prerequisite for the shape that asks it -- writing the question into the
#: proposer before anything was labelled against it is the sequencing that would
#: have Phase 1 scoring a question nothing was trained on. So the builder is the
#: first writer, and the follow-up is for this constant to move to
#: ``aorta/agent/llm.py`` beside ``CLASSIFIER_MITIGATION_QUESTION`` when Track B
#: implements per-candidate nouls, with its drift test extended to cover it.
#:
#: A template rather than a fixed string, and that is what makes the shape one
#: forward pass rather than N. ``laya.common.build_sequence`` lays out
#: ``[CLS] <instructions> [SEP] <options> [SEP] <state> [SEP]`` per question and
#: collates every question into one batch, so N questions over *one* state is a
#: single pass and N *states* is N passes. The candidate therefore has to vary in
#: the question text; putting it in the state instead would give each candidate its
#: own state and turn twenty-one questions into twenty-one forward passes, which is
#: precisely the cost the shape exists to avoid.
CLASSIFIER_CANDIDATE_QUESTION_TEMPLATE = (
    "Would applying the mitigation {mitigation!r} make this failure stop reproducing?"
)
CLASSIFIER_CANDIDATE_WHEN_TRUE = "this mitigation addresses the cause and the repro would pass"
CLASSIFIER_CANDIDATE_WHEN_FALSE = "this mitigation is unrelated to the cause, or not enough on its own"


def candidate_question(mitigation: str) -> Noul:
    """The per-candidate noul for one mitigation.

    A function rather than a formatted literal at each call site, so the corpus
    and whatever ends up asking it cannot phrase the same question two ways -- the
    defect that put the other four constants in one place.
    """
    return Noul(
        question=CLASSIFIER_CANDIDATE_QUESTION_TEMPLATE.format(mitigation=mitigation),
        when_true=CLASSIFIER_CANDIDATE_WHEN_TRUE,
        when_false=CLASSIFIER_CANDIDATE_WHEN_FALSE,
    )


def build_proposer_corpus(runs_root: str | Path) -> BuildResult:
    """Build the proposer corpus from the agent run directories under *runs_root*.

    *runs_root* is the ``--output`` an ``aorta agent mitigate`` run was given;
    each immediate subdirectory is one ticket and holds ``agent_log.jsonl``
    beside the probe cells.
    """
    root = Path(runs_root).expanduser()
    result = BuildResult()
    if not root.is_dir():
        result.warn(f"no agent run root at {root}, so there are no proposer calls to label")
        return result

    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (run_dir / "agent_log.jsonl").is_file():
            continue
        result.scanned += 1
        _add_from_run(run_dir, result)

    _warn_about_the_shape_of_it(result)
    return result


def _add_from_run(run_dir: Path, result: BuildResult) -> None:
    events = list(iter_json_lines(run_dir / "agent_log.jsonl"))
    winner = _winning_mitigation(events)
    if winner is None:
        # A run that never converged has no observed answer to "which mitigation
        # clears this". It still has a stop label when it exhausted the
        # candidates, which is why the function carries on rather than returning.
        result.skip("run never converged, so no mitigation is observed to have cleared a cell")

    symptom = _symptom(events)
    candidates = _candidates(events, run_dir)
    if not candidates:
        result.skip("no candidate mitigations recoverable from the run")
        return
    summaries = _cell_summaries(run_dir)
    category = _independent_category(run_dir)

    steps = [event for event in events if event.get("type") == "llm_step"]
    if not steps:
        result.skip("no llm_step events, so the run consulted no proposer")
        return

    exhausted_at = _exhausted_step_index(events, steps, candidates)
    for index, step in enumerate(steps):
        tried = _tried_before(events, step)
        remaining = [m for m in candidates if m not in tried and m != _BASELINE_MITIGATION]
        if not remaining:
            # The loop short-circuits an exhausted candidate list before the
            # proposer is consulted, so an example here would be a question the
            # model is never asked.
            result.skip("step had no remaining candidates, which the loop handles before asking")
            continue
        state = _state(symptom, _visible_summaries(summaries, tried), remaining, tried)
        join_key = f"{run_dir.name}:step{index}"
        # Every row from one step shares the group, so the split cannot divide
        # them and the two question shapes below are scored over the same steps.
        group = join_key
        baselines = (
            ("dspy_mitigation", _first_proposed(step)),
            ("dspy_category", str(step.get("category") or "")),
            ("dspy_stop", "true" if step.get("stop") is True else "false"),
        )

        if winner is not None and winner in remaining:
            result.examples.append(
                LabelledExample(
                    decision="proposer_mitigation",
                    state=state,
                    question=Choice(
                        question=CLASSIFIER_MITIGATION_QUESTION,
                        options=tuple(remaining),
                    ),
                    label=winner,
                    join_key=join_key,
                    group=group,
                    source=str(run_dir / "agent_log.jsonl"),
                    baselines=baselines,
                )
            )
            # The same step, asked the other way. See _add_candidate_nouls.
            _add_candidate_nouls(
                result,
                run_dir=run_dir,
                state=state,
                remaining=remaining,
                winner=winner,
                join_key=join_key,
                group=group,
                baselines=baselines,
            )
        elif winner is not None:
            result.skip("winning mitigation had already been tried by this step")

        # Stopping is correct only once nothing untried is left. Before that
        # point a stop would have skipped the mitigation that went on to clear
        # the cell, which is observed rather than assumed.
        should_stop = exhausted_at is not None and index >= exhausted_at
        result.examples.append(
            LabelledExample(
                decision="proposer_stop",
                state=state,
                question=_STOP,
                label="true" if should_stop else "false",
                join_key=join_key,
                group=group,
                source=str(run_dir / "agent_log.jsonl"),
                baselines=baselines,
            )
        )

        if category is not None:
            result.examples.append(
                LabelledExample(
                    decision="proposer_category",
                    state=state,
                    question=Choice(
                        question=CLASSIFIER_CATEGORY_QUESTION,
                        options=tuple(sorted(PROBE_CATEGORIES)),
                    ),
                    label=category,
                    join_key=join_key,
                    group=group,
                    source=str(run_dir / "report.json"),
                    baselines=baselines,
                )
            )
        else:
            result.skip(
                "no Autopsy report beside the run, so the only category available is the "
                "proposer's own and labelling with it would be circular"
            )


def _add_candidate_nouls(
    result: BuildResult,
    *,
    run_dir: Path,
    state: str,
    remaining: list[str],
    winner: str,
    join_key: str,
    group: str,
    baselines: tuple[tuple[str, str], ...],
) -> None:
    """The same step as one yes/no question per candidate, alongside the choice.

    **Both shapes, so Phase 1 measures which is better instead of arguing it.**
    Track B's case for per-candidate nouls is a good one and it is entirely a
    priori. Its three parts hold up on inspection: N nouls over one state is one
    forward pass, which the seam's batching asymmetry guarantees and
    ``tests/cia/test_local_classifier_predictor.py`` asserts; each noul lands in the ``noul:2``
    bucket rather than ``choice:11+``, so the bucket whose shipped temperature the
    library clamps is *avoided* rather than merely disclosed; and "would this one
    help" suits an agent that proposes, observes and retries better than a
    forced-choice distribution does, because the loop never wanted a distribution.

    What none of that establishes is that the encoder answers it better. That is
    an empirical claim about a model nobody has run on ROCm logs, and Phase 1
    exists precisely so such claims are not settled by argument -- the plan's kill
    criterion is a measurement, not a review. Emitting both costs one extra row per
    candidate over a state that is already built, and ``EvalResult.by_decision``
    already scores them separately, so one eval run over one corpus answers it. If
    the nouls win, Track B implements them knowing by how much; if they lose, the
    21-way choice stays and its clamped bucket is disclosed rather than avoided,
    which is the arrangement that already ships.

    Every row shares *state* with the choice-shaped row and with the other
    candidates, which is both what keeps it one forward pass and what makes the
    comparison fair -- the two shapes differ in the question and in nothing else.
    They share *group* too, so the split keeps a step whole and both shapes are
    scored over the same steps rather than over two halves of the corpus.

    The label is the mitigation that actually cleared a cell -- a probe cell
    passing, not another model's opinion -- so every other candidate is labelled
    false. That is the honest labelling and it is also the shape's main cost: one
    true in twenty-one. ``PRIMARY_METRIC`` reads this decision on ``group_top1``
    for that reason, because accuracy over the rows would score a model that
    proposes nothing at all above 0.95.
    """
    for candidate in remaining:
        result.examples.append(
            LabelledExample(
                decision="proposer_candidate",
                state=state,
                question=candidate_question(candidate),
                label="true" if candidate == winner else "false",
                join_key=f"{join_key}:{candidate}",
                group=group,
                source=str(run_dir / "agent_log.jsonl"),
                baselines=baselines,
            )
        )


def _winning_mitigation(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if event.get("type") == "converged":
            name = event.get("winning_mitigation")
            if isinstance(name, str) and name and name != _BASELINE_MITIGATION:
                return name
    return None


def _symptom(events: list[dict[str, Any]]) -> str:
    for event in events:
        if event.get("type") == "session_start":
            symptom = event.get("symptom")
            return symptom if isinstance(symptom, str) else ""
    return ""


def _candidates(events: list[dict[str, Any]], run_dir: Path) -> list[str]:
    """The candidate set, as far as the run recorded it.

    ``_list_candidate_mitigations`` resolves the allowlist from the CLI, the
    recipe axis or the whole registry, and none of those three reaches the log.
    So the candidate set is rebuilt from what the run is observed to have
    offered: every mitigation it tried, plus every mitigation a probe cell
    exists for. That is a lower bound -- a run that converged on its first try
    looks like it had one candidate -- and a lower bound is the honest shape,
    because a candidate list padded with names the run never considered would
    make the choice question easier here than in production.
    """
    found: list[str] = []
    for event in events:
        if event.get("type") == "mitigation_tried":
            name = event.get("mitigation")
            if isinstance(name, str) and name and name not in found:
                found.append(name)
    for cell_dir in _subdirectories(run_dir):
        if "-" not in cell_dir.name:
            continue
        mitigation = cell_dir.name.rsplit("-", 1)[0]
        if mitigation != _BASELINE_MITIGATION and mitigation not in found:
            found.append(mitigation)
    return found


def _subdirectories(run_dir: Path) -> list[Path]:
    """Sorted subdirectories, tolerating a run directory that has gone away.

    Sorted so a corpus built twice over the same artifacts is byte-identical;
    the eval splits 80/20 on it, and a split that moves between builds makes two
    runs incomparable for a reason nobody would look for.
    """
    try:
        return sorted(p for p in run_dir.iterdir() if p.is_dir())
    except OSError:
        return []


def _tried_before(events: list[dict[str, Any]], step: dict[str, Any]) -> list[str]:
    """Mitigations already tried when *step* was logged, by log order.

    Ordered by position in the file rather than by the ``ts`` field: the
    timestamps are second-resolution, so several events inside one iteration
    share one, and sorting by them reorders an iteration's own history.
    """
    tried: list[str] = []
    for event in events:
        if event is step:
            break
        if event.get("type") == "mitigation_tried":
            name = event.get("mitigation")
            if isinstance(name, str) and name and name not in tried:
                tried.append(name)
    return tried


def _exhausted_step_index(
    events: list[dict[str, Any]], steps: list[dict[str, Any]], candidates: list[str]
) -> int | None:
    """The first step at which stopping was the right answer, or None.

    None means no step in this run should have stopped: either it converged
    while candidates remained, or it ended for a reason that says nothing about
    whether the search was finished -- a wall-time budget, a policy refusal, an
    approval gate, a backend error. Those runs contribute negative stop examples
    only, which is correct: at every step they reached, there was more to try.
    """
    for event in events:
        if event.get("type") == "baseline_pass":
            # The baseline passing means nothing was broken, so the very first
            # step should have stopped. The loop short-circuits this before the
            # proposer is consulted, so in practice there is no step to label.
            return 0
    outcome = _stop_outcome(events)
    if outcome != "exhausted_candidates":
        return None
    for index, step in enumerate(steps):
        tried = _tried_before(events, step)
        if not [m for m in candidates if m not in tried and m != _BASELINE_MITIGATION]:
            return index
    return len(steps) - 1


def _stop_outcome(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "search_stopped":
            return str(event.get("stop_reason") or event.get("outcome") or "")
    return ""


def _cell_summaries(run_dir: Path) -> list[dict[str, Any]]:
    """Cell verdicts and detectors, read with the loop's own reader.

    ``aorta.agent.state`` is core and imports no model, so reusing it costs
    nothing and keeps one definition of what a cell's verdict is. A second
    aggregation here is how the corpus would come to disagree with the loop
    about which cells failed.
    """
    from aorta.agent.state import aggregate_cell_verdict, read_trial_results

    summaries: list[dict[str, Any]] = []
    for cell_dir in _subdirectories(run_dir):
        trials = read_trial_results(cell_dir)
        if not trials:
            continue
        detectors: list[str] = []
        for trial in trials:
            for detector in trial.get("failure_detectors_fired") or []:
                if detector not in detectors:
                    detectors.append(detector)
        summaries.append(
            {
                "cell_name": trials[0].get("cell_name", cell_dir.name),
                "verdict": aggregate_cell_verdict(trials),
                "failure_detectors_fired": detectors,
                "exit_code": trials[0].get("exit_code"),
            }
        )
    return summaries


def _visible_summaries(
    summaries: list[dict[str, Any]], tried: list[str]
) -> list[dict[str, Any]]:
    """The cells that existed when a step was taken.

    The run directory holds every cell the whole run produced, including the
    ones a later iteration added. Showing all of them to a question asked at
    step 0 would hand the model the result of the mitigation it is being asked
    to pick, which scores beautifully and measures nothing.
    """
    allowed = {_BASELINE_MITIGATION, *tried}
    return [
        row
        for row in summaries
        if str(row.get("cell_name") or "").rsplit("-", 1)[0] in allowed
    ]


def _state(
    symptom: str,
    summaries: list[dict[str, Any]],
    remaining: list[str],
    tried: list[str],
) -> str:
    """The proposer's inputs, serialised the way its prompt serialises them.

    Deliberately the same JSON ``_build_prompt`` sends, so what the encoder reads
    and what the LLM reads differ in model and not in input. If the two diverge,
    a Phase 1 comparison between them stops being a comparison.
    """
    return json.dumps(
        {
            "symptom": symptom or None,
            "cell_summaries": summaries,
            "candidates": remaining,
            "already_tried": tried,
        },
        indent=2,
    )


def _first_proposed(step: dict[str, Any]) -> str:
    proposed = step.get("next_mitigations")
    if isinstance(proposed, list) and proposed:
        return str(proposed[0])
    return ""


def _independent_category(run_dir: Path) -> str | None:
    """A category from something other than the proposer, or None.

    Only an Autopsy report counts. See the module docstring: every other
    category on disk in a probe run is the proposer's own.
    """
    path = run_dir / "report.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None
    if not isinstance(report, dict):
        return None
    category = str(report.get("category") or "")
    return category if category in PROBE_CATEGORIES else None


def _warn_about_the_shape_of_it(result: BuildResult) -> None:
    stops = [e.label for e in result.examples if e.decision == "proposer_stop"]
    if stops and "true" not in stops:
        result.warn(
            "no proposer_stop example is labelled true: no scanned run ended by exhausting "
            "its candidates, so the stop noul has only negative examples and a classifier "
            "that never stops scores perfectly on it."
        )
    if not any(e.decision == "proposer_category" for e in result.examples):
        result.warn(
            "no proposer_category examples: labelling a category needs an Autopsy report "
            "beside the run, and the proposer's own category cannot label itself. Run "
            "Autopsy over the sweep bundles, or label the categories by hand."
        )
    mitigations = [e for e in result.examples if e.decision == "proposer_mitigation"]
    if mitigations:
        widths = {len(e.question.options) for e in mitigations}
        if widths == {1}:
            result.warn(
                "every proposer_mitigation example offers exactly one candidate, so the "
                "choice is forced and accuracy on it is 1.0 by construction. These runs "
                "converged on the first mitigation they tried."
            )
    candidates = [e for e in result.examples if e.decision == "proposer_candidate"]
    if candidates and mitigations:
        # The two shapes reach a usable sample size at wildly different rates, and
        # "the corpus has 400 examples" hides it completely. The plan's target is
        # 300-500 per decision, and on a 21-candidate run one step buys 21 rows of
        # one decision and a single row of the other.
        result.warn(
            f"the same {len(mitigations)} step(s) produced {len(mitigations)} "
            f"proposer_mitigation row(s) and {len(candidates)} proposer_candidate row(s). "
            "Both shapes describe the same decisions, so read the per-decision counts "
            "rather than the total: the choice shape needs one step per row and is the "
            "one that will be short of the 300-example target."
        )
        positives = sum(1 for e in candidates if e.label == "true")
        result.warn(
            f"{positives} of {len(candidates)} proposer_candidate rows are labelled true, "
            "which is one per step by construction. Score that decision on group_top1 -- "
            "whether the right candidate ranks first -- never on accuracy over rows."
        )


__all__ = ["build_proposer_corpus"]
