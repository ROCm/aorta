#!/usr/bin/env python3
"""Multi-step probe episodes against an archived matrix, for GRPO rollouts.

Why this exists
===============
A single-step task -- one scenario, one archived matrix, one completion scored
against it -- is an input/output pair, and an answer that is one name from a
21-name menu is memorisable by construction. A sequence of decisions, each
conditioned on what the previous one revealed, is not: there is no fixed string
that answers it. Three properties of the single-step regime reduce to the same
missing thing:

1. Several step events can never fire in one completion. ``name_already_tried``
   needs a history, and ``name_not_offered`` / ``malformed_reply`` are
   invisible to any scorer that reads the loop's log rather than the reply.
2. ``terminal_unresolvable_correct`` is structurally unreachable: making the
   claim requires stopping while proposing nothing, earning it requires having
   already tried everything, and one reply cannot do both.
3. A constant policy is a complete policy when episodes are one step long. It
   cannot follow a decision tree.

What this module is
===================
An **offline environment**. The policy proposes, an archived probe matrix
answers, the policy sees the result and proposes again, until it converges,
stops, or runs out of budget. Nothing here runs a workload or touches a GPU:
every cell verdict already exists on disk.

The substitution of an archive lookup for a probe run is exact rather than
approximate, and only on archives where it is. It needs every cell to have a
single verdict across its trials, so that the verdict a re-run would produce is
known; it would **not** be legitimate on a rate-based reproducer. And it needs
the archive to have measured every name the menu offers, which
:func:`load_scenario` enforces rather than assumes.

Design decision 1: an in-harness driver that imports the loop's decisions
========================================================================
``run_agent_loop`` is not reused; its *decision functions* are. The executor is
the only thing replaced, for two reasons that are not preferences.

Throughput: ``run_agent_loop`` is a sequential blocking loop over one episode,
and GRPO advances a whole group of episodes per scenario per iteration.
Generation has to be batched **across episodes at the same step index**, which a
lockstep driver can do and a call into ``run_agent_loop`` cannot, because the
model call is buried inside it.

Cost: the whole argument for offline episodes is that **GPU is paid once per
matrix and never per rollout step**, which is true precisely because
``_execute_probe_matrix`` is not called.

Fidelity is bought back by *import*:

===============================  ======================================
decision                         shipped function
===============================  ======================================
what the policy is shown         ``loop._read_cell_summaries``
did the baseline pass            ``loop._baseline_passed``
did a mitigation win             ``loop._find_winning_mitigation``
parse and filter the reply       ``llm._step_from_content``
validate the step                ``AgentPolicy.validate_step``
the iteration budget             ``AgentPolicy.check_iteration_budget``
the approval gate                ``AgentPolicy.pending_approvals``
classify a stop                  ``loop._resolve_stop_outcome``
the audit log                    ``state.append_log_event``
===============================  ======================================

``_read_cell_summaries`` is called **once per archive, on the archive root**,
and each episode filters that list to the cells on its own axis -- the loop's
own output, subsetted, in its own order.

The claim that this is faithful is **measured, not asserted**:
``tests/examples/test_episode_env.py`` drives the real ``run_agent_loop`` with
``_execute_probe_matrix`` patched to materialise archived cells, feeds the same
scripted replies to both, and requires the two ``agent_log.jsonl`` files to
match event for event with only timestamps removed.

⚠ **One deliberate divergence from the shipped real-LLM proposers**, which
short-circuit an empty remaining menu into an ``exhausted_candidates`` stop
without calling the model. The environment calls the policy anyway. That
shortcut is the *proposer's* stop, not the policy's, and on an unresolvable
scenario crediting ``terminal_unresolvable_correct`` to it would pay for a
conclusion nobody drew. The fidelity test's proposer has no shortcut either, so
the comparison is like for like.

Design decision 2: one advantage per episode, group = episodes on a scenario
============================================================================
The episode return is the **sum of its per-step events** (``event_reward``),
and GRPO normalises within a group of episodes on the **same scenario**. Steps
are not exchangeable samples of one task -- step 3 of a long search faces a
state step 1 never saw -- so a group of steps would have a meaningless baseline.
Episodes on one scenario are exchangeable by construction: same prompt at step
1, same answer key, same menu.

Per-step advantages were rejected: they need a per-step baseline, which needs a
value function or a group of episodes sharing a state, and neither exists here.

⚠ **The cost, stated because it does not go away:** one advantage per episode
pushes every token of a good step inside a bad episode down with the rest. That
is ordinary GRPO without a critic. What the event reward still buys is
*diagnosis* -- the histogram says which decision earned what even though the
gradient does not.

Design decision 3: the budget is the shipped one
================================================
``max_iterations = 8``, the ``AgentPolicy`` default, so the measured behaviour
is what ``aorta agent`` does today. Eight cycles is also what makes
``terminal_unresolvable_correct`` reachable and no more than that: exhausting a
21-name menu at ~3.4 names a reply takes about seven cycles, so the earned path
fits inside the budget without fitting comfortably.

Termination is guaranteed four independent ways: convergence, an explicit
stop, the iteration budget, and the menu emptying (every further name is
dropped by the filter, which ends the search as ``proposal_unresolved``).

Design decision 4: the observation, and where it bites
======================================================
The policy sees what ``_read_cell_summaries`` produces -- cell name, aggregated
verdict, detector IDs, the capture dict, exit code. Categorical fields and no
numbers, so **a mitigation that changed the run without changing the verdict is
invisible** and consecutive steps can look identical even when the underlying
cells differ. On a corpus whose every non-resolving cell is a plain ``fail``
that costs nothing; on a corpus with graded failures it would.

The failure and warn detector lists are unioned into one sorted list (see
:func:`render_cell_summary`), which a test shows is lossless on the corpus
archives because none fires a warn detector.

The menu is the full registry on every scenario
===============================================
Every scenario offers the 21 registered non-baseline mitigations, so step 1 is
the same question on every scenario and the task does not change per scenario.
The price is that a scenario may only enter the corpus if its archive measured
every offered name: an environment has to be able to answer every action it
offers.

Usage
-----

    python examples/rl/episode_env.py --corpus-root <dir> --describe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aorta.agent.llm import PROBE_CATEGORIES, AgentStep, _step_from_content
from aorta.agent.loop import (
    _baseline_passed,
    _find_winning_mitigation,
    _read_cell_summaries,
    _resolve_stop_outcome,
)
from aorta.agent.policy import AgentPolicy, PolicyViolation
from aorta.agent.state import append_log_event

sys.path.insert(0, str(Path(__file__).resolve().parent))

from event_reward import (  # noqa: E402
    BASELINE,
    BASELINE_CELL,
    POINTS,
    EventScore,
    LoggedEpisode,
    LoggedStep,
    MatrixGrid,
    StepContext,
    classify_stop,
    grid_from_matrix,
    score_completion,
    score_episode,
)
from triage_reward import Label, find_probe_cells, label_run, label_trials  # noqa: E402

__all__ = [
    "CORPUS",
    "CORPUS_ROOT_ENV",
    "SYSTEM",
    "Episode",
    "EpisodeStep",
    "Sample",
    "Scenario",
    "load_corpus",
    "load_scenario",
    "registered_mitigations",
    "rollout_scenario",
    "runnable_names",
    "user_message",
]

#: Where the archived matrices live, if ``--corpus-root`` is not given.
CORPUS_ROOT_ENV = "AORTA_RL_CORPUS_ROOT"

#: The corpus, as ``(scenario_id, archive directory name, mechanism family)``.
#: Each archive is one probe run's output directory (the parent of its
#: ``{mitigation}-{diagnostic}`` cell directories), found under the corpus root
#: by name. The archives are not in the repository; every one measured all 21
#: registered mitigations at 4 trials per cell with no cell whose trials
#: disagree, which is what :func:`load_scenario` checks.
CORPUS: tuple[tuple[str, str, str], ...] = (
    ("nan_uninit_workspace", "PROBE-NAN-WS-MATRIX22", "memory contents"),
    ("cancellation_nan", "PROBE-CANCEL-NAN", "numeric cancellation"),
    ("stream_stale_read", "PROBE-STREAM-RACE-MATRIX", "ordering"),
    ("queue_stale_read", "PROBE-QUEUE-RACE-MATRIX", "ordering"),
    ("xnack_page_fault", "PROBE-XNACK-SVM-MATRIX", "address translation"),
    ("scratch_exhaustion", "PROBE-SCRATCH-OOM-MATRIX", "resource exhaustion"),
    ("reference_mismatch", "PROBE-REFMISMATCH-MATRIX", "numeric reference"),
)

#: Seven keys: the five ``proposal_reward.REQUIRED_KEYS`` plus the triage
#: answer (``verdict``, ``detectors``), asked in the same reply so that every
#: completion carries its own read of the evidence rather than sharing one.
#: The category list is ``PROBE_CATEGORIES`` -- the ones ``validate_step``
#: accepts from a probe step -- so the prompt never offers a category the loop
#: would refuse.
TRIAGE_VERDICTS = ["pass", "warn", "fail", "error", "not_checked"]
SYSTEM = (
    "You are an AORTA probe agent. You are shown the evidence from one probe "
    "cell and a list of registered mitigation names.\n"
    "Do two things in one reply.\n"
    "(a) Triage the evidence: say what happened and cite the finding IDs that "
    f"justify it. `verdict` must be one of {TRIAGE_VERDICTS}. `detectors` must "
    "be a list of the finding IDs you were shown that fired, written exactly as "
    "they appear, and nothing else; a clean run cites an empty list.\n"
    "(b) Propose what to try next: `next_mitigations` must contain ONLY names "
    "from the candidate list, and `category` must be one of "
    f"{sorted(PROBE_CATEGORIES)}.\n"
    "Return strict JSON with exactly these keys: verdict (string), detectors "
    "(list of strings), category (string), hypothesis (string), "
    "next_mitigations (list of strings), confidence (number 0-1), stop "
    "(boolean)."
)


def registered_mitigations() -> list[str]:
    """The menu: every built-in mitigation except the baseline, sorted."""
    from aorta.registry.mitigations import BUILTIN_MITIGATIONS

    return [n for n in sorted(BUILTIN_MITIGATIONS) if n != BASELINE]


def runnable_names(raw: str, offered: Sequence[str]) -> list[str]:
    """The names in a reply that would actually become probe cells.

    Replays what the shipped path does with a reply: an unparseable reply
    yields nothing, a name outside ``offered`` is dropped before the loop sees
    it, and a reply that sets ``stop: true`` runs nothing at all, because the
    loop stops before it grows the axis. So neither a hallucinated name nor a
    name in a stopping reply counts as naming a resolver.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict):
        return []
    if obj.get("stop") is True:
        return []
    proposed = obj.get("next_mitigations")
    if not isinstance(proposed, list):
        return []
    return [str(m) for m in proposed if str(m) in offered]


def resolve_corpus_root(given: str | Path | None) -> Path:
    """``--corpus-root``, else ``$AORTA_RL_CORPUS_ROOT``, else a refusal.

    There is no default location. A default would be a path on somebody's
    machine, and a missing corpus has to be an error naming the fix rather
    than an empty corpus that trains on nothing.
    """
    value = given if given else os.environ.get(CORPUS_ROOT_ENV)
    if not value:
        raise SystemExit(
            f"no corpus: pass --corpus-root or set {CORPUS_ROOT_ENV} to the "
            f"directory holding the archived matrices "
            f"({', '.join(archive for _, archive, _ in CORPUS)})"
        )
    root = Path(value)
    if not root.is_dir():
        raise SystemExit(f"corpus root {root} is not a directory")
    return root


def render_cell_summary(row: dict[str, Any]) -> dict[str, Any]:
    """One ``_read_cell_summaries`` row in the shape the prompt uses.

    ``_read_cell_summaries`` keeps failure and warn detector IDs as two lists
    in first-seen order; the prompt shows **one sorted union** under
    ``failure_detectors_fired`` and no warn key. The prompt shape is the one the
    recorded checkpoints were trained on, so it is kept rather than switched to
    the loop's own: changing it would change the question every recorded number
    was measured against. What the policy gives up is the failure/warn split,
    which costs nothing on a corpus with no warn detector (a test asserts that
    for the real archives) and is a real difference from what a production
    proposer is shown.
    """
    detectors = set(row.get("failure_detectors_fired") or [])
    detectors.update(row.get("warn_detectors_fired") or [])
    return {
        "cell_name": row.get("cell_name"),
        "verdict": row.get("verdict"),
        "exit_code": row.get("exit_code"),
        "failure_detectors_fired": sorted(str(d) for d in detectors),
        "capture": row.get("capture") or {},
    }


def user_message(cell_summaries: list[dict[str, Any]], candidates: list[str]) -> str:
    """The user turn: the cells on the axis so far, and the remaining menu.

    Deliberately not ``llm._build_prompt``, which also carries ``symptom`` and
    ``already_tried``. ``already_tried`` is recoverable from the cell names on
    screen; ``symptom`` is an operator-written string an archive does not carry.
    This is the shape the recorded checkpoints were trained on, and it is stated
    as a divergence from the shipped proposer rather than hidden.
    """
    return json.dumps(
        {
            "cell_summaries": [render_cell_summary(row) for row in cell_summaries],
            "candidates": candidates,
        },
        sort_keys=True,
    )


@dataclass(frozen=True)
class Scenario:
    """One archived matrix, loaded once and shared by every episode on it."""

    scenario_id: str
    family: str
    root: Path
    grid: MatrixGrid
    label: Label
    offered: tuple[str, ...]
    #: ``loop._read_cell_summaries`` over the whole archive, in its own order.
    summaries: tuple[dict[str, Any], ...]
    #: :func:`archive_digest` of the archive, so a result can say which ground
    #: truth it was scored against. Empty only for a hand-built scenario.
    digest: str = ""

    @property
    def resolvers(self) -> frozenset[str]:
        return self.grid.resolution.resolvers

    @property
    def unresolvable(self) -> bool:
        return self.grid.resolution.baseline_failed and not self.resolvers

    def summaries_for(self, cell_names: Iterable[str]) -> list[dict[str, Any]]:
        """The archive's own summaries, filtered to the cells on an axis."""
        wanted = set(cell_names)
        return [dict(row) for row in self.summaries if row.get("cell_name") in wanted]


def archive_digest(root: str | Path) -> str:
    """SHA-256 over every ``trial_*/result.json`` in an archive, path and bytes.

    The identity of a ground truth. Two archives with the same directory name
    -- a different ``--corpus-root``, or the same path after a re-run or an
    edit -- are different answer keys, and a scenario id alone cannot tell them
    apart. The files hashed are exactly the ones the grid, the summaries and
    the labels are built from, keyed by their path relative to the archive so
    the digest does not depend on where the corpus is mounted.
    """
    root = Path(root)
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("result.json") if path.parent.name.startswith("trial_")
    )
    if not files:
        raise ValueError(f"{root}: no trial_*/result.json to take a digest of")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def strict_trial_results(cell: Path, scenario_id: str = "") -> list[dict[str, Any]]:
    """Every ``trial_*/result.json`` of a cell, or a ``ValueError`` naming each bad one.

    ``aorta.agent.state.read_trial_results`` skips a trial whose file is
    missing or does not parse, which is right for a live loop reading a cell
    still being written and wrong for an archive standing in for a re-run: a
    four-trial cell with one unreadable trial would pass as three that agree,
    and the dropped one may be the one that disagreed.
    """
    problems: list[str] = []
    indexed: list[tuple[int, dict[str, Any]]] = []
    for trial in sorted(p for p in cell.glob("trial_*") if p.is_dir()):
        path = trial / "result.json"
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            problems.append(f"{trial.name}/result.json: missing")
            continue
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"{trial.name}/result.json: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(doc, dict):
            problems.append(f"{trial.name}/result.json: not a JSON object")
            continue
        try:
            index = int(trial.name.removeprefix("trial_"))
        except ValueError:
            problems.append(f"{trial.name}: not a trial_<N> directory")
            continue
        indexed.append((index, doc))
    if problems:
        raise ValueError(
            f"{scenario_id or cell.name}: cell {cell} has trial(s) that cannot be read "
            f"({'; '.join(problems[:4])}); an archive with an unreadable trial cannot "
            f"say its trials agree"
        )
    return [doc for _, doc in sorted(indexed, key=lambda pair: pair[0])]


def load_scenario(
    scenario_id: str,
    root: str | Path,
    family: str,
    offered: Sequence[str] | None = None,
) -> Scenario:
    """Load one archive, refusing any the environment cannot answer from.

    Three refusals, each because the alternative is the environment inventing
    an outcome -- the judgement call ``event_reward`` is built to exclude:

    * **a name on the menu with no cell.** The scorer prices that silence at
      0.0 (``name_unmeasured``) in a single reply; across an episode the policy
      goes on to condition its next decision on it, so the silence compounds.
    * **a cell whose trials disagree.** Its verdict is a rate, and replaying it
      as one fixed answer would state a certainty the archive does not have.
    * **no failing baseline.** A passing ``none-none`` short-circuits before the
      proposer is ever called, so the scenario would harvest zero decisions.
    """
    root = Path(root)
    # Read strictly, and first: the grid reader below skips a trial it cannot
    # read, and a trial that cannot be read may be the one that disagrees.
    trials = {cell: strict_trial_results(cell, scenario_id) for cell in find_probe_cells(root)}
    grid = grid_from_matrix(root, scenario_id)
    menu = tuple(offered) if offered is not None else tuple(registered_mitigations())
    missing = sorted(set(menu) - set(grid.measured_mitigations))
    if missing:
        raise ValueError(
            f"{scenario_id}: the archive at {root} has no cell for "
            f"{len(missing)} offered mitigation(s) {missing[:4]}, so the "
            f"environment cannot answer every action it offers. Use an "
            f"archive that measured the whole menu, or shrink the menu -- but "
            f"do not let the environment invent a verdict."
        )
    baseline = next((c for c in trials if c.name == BASELINE_CELL), None)
    if baseline is None:
        raise ValueError(f"{scenario_id}: {root} has no {BASELINE_CELL} cell")
    # Per trial, through the same resolver the grid uses, so "the trials agree"
    # is a statement about the verdicts the scorer sees and not about whatever
    # string each trial happened to store.
    split = sorted(
        cell.name for cell, docs in trials.items()
        if len({label_run(doc).verdict for doc in docs}) > 1
    )
    if split:
        raise ValueError(
            f"{scenario_id}: {len(split)} cell(s) in {root} have trials that "
            f"disagree ({split[:4]}); a replayed archive can only stand in for "
            f"a re-run when every cell has one answer"
        )
    if not grid.resolution.baseline_failed:
        raise ValueError(
            f"{scenario_id}: the {BASELINE_CELL} cell in {root} did not fail, "
            f"so there is nothing to search for"
        )
    label = label_trials(trials[baseline], source=str(baseline))
    return Scenario(
        scenario_id=scenario_id,
        family=family,
        root=root,
        grid=grid,
        label=label,
        offered=menu,
        summaries=tuple(_read_cell_summaries(root)),
        digest=archive_digest(root),
    )


def load_corpus(
    root: str | Path | None = None,
    entries: Sequence[tuple[str, str, str]] = CORPUS,
    *,
    only: Sequence[str] = (),
) -> list[Scenario]:
    """The corpus under ``root``, optionally narrowed to ``only``.

    An id in ``only`` that is not in ``entries`` is an error rather than a
    silent drop: a typo would otherwise look like a result on a smaller corpus.
    """
    base = resolve_corpus_root(root)
    return [
        load_scenario(sid, base / archive, family)
        for sid, archive, family in _selected(entries, only)
    ]


def _selected(
    entries: Sequence[tuple[str, str, str]], only: Sequence[str]
) -> list[tuple[str, str, str]]:
    known = [sid for sid, _, _ in entries]
    unknown = [sid for sid in only if sid not in known]
    if unknown:
        raise SystemExit(
            f"scenario id(s) {unknown} are not in the corpus; available: {sorted(known)}"
        )
    return [entry for entry in entries if not only or entry[0] in only]


def corpus_digests(
    root: str | Path | None = None,
    entries: Sequence[tuple[str, str, str]] = CORPUS,
    *,
    only: Sequence[str] = (),
) -> dict[str, str]:
    """:func:`archive_digest` per scenario id, without loading the scenarios."""
    base = resolve_corpus_root(root)
    return {sid: archive_digest(base / archive) for sid, archive, _ in _selected(entries, only)}


@dataclass
class EpisodeStep:
    """One decision: what the policy was shown, what it emitted, what it meant.

    ``raw`` is the **wire text**, kept deliberately. A reply rebuilt from the
    log (``event_reward._step_to_raw``) cannot recreate a malformed reply, and
    a refused category never reaches the log at all, so ``malformed_reply`` is
    structurally zero on log-derived episodes. The harness generated the text,
    so it scores the text.
    """

    n: int
    prompt: str
    raw: str
    mitigation_axis_before: list[str]
    cells_before: int
    parsed: AgentStep | None = None
    stop: bool = False

    @property
    def unresolved_mitigations(self) -> list[str]:
        """What the shipped filter dropped from this reply (aorta#449)."""
        return list(self.parsed.unresolved_mitigations) if self.parsed else []


@dataclass
class Episode:
    """One search, advanced by the caller one step at a time.

    A state machine with :meth:`observe` and :meth:`act` rather than a
    ``run(proposer)`` method, because the caller has to hold a whole group of
    these mid-flight and issue one batched generation across all of them.
    """

    scenario: Scenario
    index: int
    policy: AgentPolicy
    run_dir: Path | None = None

    mitigation_axis: list[str] = field(default_factory=lambda: [BASELINE])
    cells_spent: int = 0
    iterations_completed: int = 0
    steps: list[EpisodeStep] = field(default_factory=list)

    done: bool = False
    outcome: str = "in_progress"
    recommended: str = ""
    winning_mitigation: str | None = None
    #: The classification ``event_reward`` scores, decided by the same rules
    #: ``event_reward.episode_from_log`` applies to a real log.
    terminal: str = "unobserved"
    terminal_why: str = "the episode has not ended"
    #: The prompt handed out by the last :meth:`observe`, carried so :meth:`act`
    #: can record what its reply answered.
    pending_prompt: str = ""
    _started: bool = False

    @property
    def tried_mitigations(self) -> list[str]:
        return [n for n in self.mitigation_axis if n != BASELINE]

    @property
    def remaining(self) -> list[str]:
        tried = set(self.tried_mitigations)
        return [n for n in self.scenario.offered if n not in tried and n != BASELINE]

    def _cells(self) -> list[str]:
        """The cells on the axis: one per mitigation, diagnostic ``none``.

        The archives carry a single diagnostic column, and ``run_agent_loop``
        does not grow the diagnostic axis, so the grid is the mitigation axis.
        """
        return [f"{m}-{BASELINE}" for m in self.mitigation_axis]

    def _log(self, kind: str, payload: dict[str, Any]) -> None:
        if self.run_dir is not None:
            append_log_event(self.run_dir, kind, payload)

    def _finish(self, outcome: str, recommended: str, terminal: str, why: str) -> None:
        self.done = True
        self.outcome, self.recommended = outcome, recommended
        self.terminal, self.terminal_why = terminal, why

    # -- the loop body, transcribed ---------------------------------------

    def observe(self) -> str | None:
        """Run the (archived) matrix and return the next prompt, or None.

        ``run_agent_loop``'s body between ``_execute_probe_matrix`` and
        ``proposer.propose``, with the executor replaced by an archive lookup.
        Returning None means the episode ended without the policy getting
        another turn.
        """
        if self.done:
            return None
        if not self._started:
            self._started = True
            self._log(
                "session_start",
                {
                    "ticket": f"{self.scenario.scenario_id}-{self.index:03d}",
                    "ticket_slug": f"{self.scenario.scenario_id}-{self.index:03d}",
                    "argv": [],
                    "symptom": None,
                    "llm_backend": "episode_env",
                },
            )

        names = self._cells()
        summaries = self.scenario.summaries_for(names)
        # What has been paid for: the cells that now exist.
        self.cells_spent = max(self.cells_spent, len(names))

        if _baseline_passed(summaries):
            # Unreachable on a corpus `load_scenario` accepted, and transcribed
            # anyway: it is the branch that decides an episode harvests zero
            # decisions, and a silent divergence here would be invisible.
            self._log("baseline_pass", {})
            self._finish(
                "baseline_pass",
                "Baseline cell (none-none) passed.",
                "other",
                "the baseline cell passed: there was nothing to search for",
            )
            return None

        winner = _find_winning_mitigation(summaries)
        if winner:
            self.winning_mitigation = winner
            self._log("converged", {"winning_mitigation": winner})
            self._finish(
                "converged",
                f"Re-run the repro with mitigation `{winner}` applied",
                "converged",
                f"cell {winner!r}-none passed",
            )
            return None

        try:
            self.policy.check_iteration_budget(self.iterations_completed)
        except PolicyViolation as exc:
            self._policy_stop(str(exc))
            return None

        return user_message(summaries, self.remaining)

    def _policy_stop(self, reason: str) -> None:
        # A budget or policy stop is not a statement about the search, which
        # is why `event_reward` withholds it as `other` -- the same rule
        # `episode_from_log` applies to the `policy_stop` record.
        self._log("policy_stop", {"reason": reason})
        self._finish("policy_stop", reason, "other", f"policy_stop: {reason}")

    def act(self, raw: str) -> None:
        """Consume one completion and advance the episode.

        The reply goes through the shipped parser and the shipped validator, in
        that order, exactly as ``run_agent_loop`` sends it. An unparseable reply
        becomes a stop (``llm._safe_stop``), not an exception, so a malformed
        reply ends this episode as it would in production rather than aborting
        a batched iteration holding the rest of the group.
        """
        if self.done:
            return
        step = _step_from_content(raw, self.remaining)
        record = EpisodeStep(
            n=len(self.steps) + 1,
            prompt=self.pending_prompt,
            raw=raw,
            mitigation_axis_before=list(self.mitigation_axis),
            cells_before=self.cells_spent,
            parsed=step,
            stop=bool(step.stop),
        )
        # The reply is kept even if validation refuses it, so the scorer still
        # prices it (as `malformed_reply`, since `admissible` refuses the same
        # category). The log does not see it: `run_agent_loop` validates BEFORE
        # it logs, so a refused step is never written, and the artifact and the
        # score disagree by one step in exactly this case. A test pins it.
        self.steps.append(record)
        try:
            step = self.policy.validate_step(step)
        except PolicyViolation as exc:
            self._policy_stop(str(exc))
            return
        record.parsed = step

        payload: dict[str, Any] = {
            "category": step.category,
            "hypothesis": step.hypothesis,
            "next_mitigations": step.next_mitigations,
            "confidence": step.confidence,
            "stop": step.stop,
            "stop_reason": step.stop_reason,
        }
        if step.unresolved_mitigations:
            payload["unresolved_mitigations"] = list(step.unresolved_mitigations)
        self._log("llm_step", payload)

        if step.stop or not step.next_mitigations:
            summaries = self.scenario.summaries_for(self._cells())
            outcome, recommended, resolved = _resolve_stop_outcome(step, summaries)
            stopped: dict[str, Any] = {"outcome": outcome, "stop_reason": resolved}
            if step.unresolved_mitigations:
                stopped["unresolved_mitigations"] = list(step.unresolved_mitigations)
            self._log("search_stopped", stopped)
            terminal, why = classify_stop(
                stop_reason=resolved,
                proposed_nothing=not step.next_mitigations,
                unresolvable=self.scenario.unresolvable,
                # Nothing from a step that stopped: its names were never run.
                resolver_named=any(
                    set(s.parsed.next_mitigations if s.parsed and not s.stop else [])
                    & self.scenario.resolvers
                    for s in self.steps
                ),
                offered=self.scenario.offered,
                tried=self.tried_mitigations,
                unresolved=step.unresolved_mitigations,
            )
            self._finish(outcome, recommended, terminal, why)
            return

        # Unreachable through `_step_from_content`, which has already filtered
        # to the remaining menu, and transcribed because `run_agent_loop` has it:
        # a proposer that bypassed the filter must not widen the search.
        disallowed = [m for m in step.next_mitigations if m not in self.scenario.offered]
        if disallowed:
            self._policy_stop(
                f"proposer returned mitigations outside the allowed candidate "
                f"set: {sorted(disallowed)}. Allowed: {sorted(self.scenario.offered)}."
            )
            return

        pending_approval = self.policy.pending_approvals(step.next_mitigations)
        if pending_approval:
            self._log("approval_required", {"mitigations": pending_approval})
            self._finish(
                "approval_required",
                f"Approval required for mitigations: {pending_approval}.",
                "other",
                "the run ended on approval_required",
            )
            return

        for mitigation in step.next_mitigations:
            if mitigation in self.mitigation_axis:
                continue
            self.mitigation_axis.append(mitigation)
            self._log("mitigation_tried", {"mitigation": mitigation})
        self.iterations_completed += 1
        self._log("iteration_complete", {"iteration": self.iterations_completed})

    # -- the artifact -----------------------------------------------------

    def as_logged(self) -> LoggedEpisode:
        """The episode in the shape ``event_reward.score_episode`` consumes.

        Built directly rather than by writing a log and reading it back; the
        difference is that ``LoggedStep.raw`` is the wire text here. Everything
        else -- the axis in force at each step, the cell count, the terminal --
        is the same value by the same rule, which a test checks against
        ``episode_from_log``.
        """
        return LoggedEpisode(
            run_dir=self.run_dir or Path("."),
            steps=[
                LoggedStep(
                    n=s.n,
                    raw=s.raw,
                    mitigation_axis_before=list(s.mitigation_axis_before),
                    stop=s.stop,
                )
                for s in self.steps
            ],
            terminal=self.terminal,
            terminal_why=self.terminal_why,
            cells=self.cells_spent,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario.scenario_id,
            "index": self.index,
            "outcome": self.outcome,
            "terminal": self.terminal,
            "terminal_why": self.terminal_why,
            "steps": len(self.steps),
            "cells": self.cells_spent,
            "winning_mitigation": self.winning_mitigation,
            "mitigation_axis": list(self.mitigation_axis),
        }


# ---------------------------------------------------------------------------
# lockstep driver
# ---------------------------------------------------------------------------


def open_episodes(
    scenario: Scenario,
    count: int,
    policy: AgentPolicy,
    *,
    log_root: Path | None = None,
    log_first: int = 0,
) -> list[Episode]:
    """``count`` fresh episodes on one scenario, the first ``log_first`` logged.

    Logging is sampled because the wire record already holds every completion.
    The sample is the *first* indices rather than a random subset so the logged
    set is the same across iterations and one episode's history can be followed.
    """
    out = []
    for index in range(count):
        run_dir = None
        if log_root is not None and index < log_first:
            run_dir = log_root / scenario.scenario_id / f"ep{index:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
        out.append(Episode(scenario=scenario, index=index, policy=policy, run_dir=run_dir))
    return out


def pending(episodes: Sequence[Episode]) -> list[tuple[Episode, str]]:
    """Every live episode paired with the prompt it is waiting on.

    Calling :meth:`Episode.observe` is what runs the (archived) matrix, so this
    is where convergence and the budget are decided; an episode that ends here
    simply does not appear in the result.
    """
    out: list[tuple[Episode, str]] = []
    for episode in episodes:
        prompt = episode.observe()
        if prompt is not None:
            episode.pending_prompt = prompt
            out.append((episode, prompt))
    return out


@dataclass(frozen=True)
class Sample:
    """One (prompt, completion, advantage) triple the policy gradient consumes.

    **The unit of the loss, and the one structural change multi-step needs.**
    With single completions every sample of a group answers the same prompt;
    an episode's step 3 answers a question step 1 never saw, so the prompt
    travels with the completion.

    ``advantage`` is the **episode's**, repeated on every step of it. That is
    REINFORCE's own arithmetic -- ``grad log P(trajectory) = sum_t grad log
    pi(a_t | s_t)`` -- so summing each step's log-probs under one advantage is
    the trajectory gradient.

    Lives here rather than in the trainer so the whole rollout path imports
    without ``torch`` and is testable on a CPU.
    """

    scenario_id: str
    prompt: str
    completion: str
    advantage: float
    episode: int = 0
    step: int = 1


def _parses(raw: str) -> bool:
    try:
        return isinstance(json.loads(raw), dict)
    except json.JSONDecodeError:
        return False


def score(episode: Episode) -> EventScore:
    """An episode's event score, with the baseline cell's label on every step.

    The baseline label on every step is not a simplification: a passing
    ``{m}-none`` cell ends the episode in :meth:`Episode.observe` before the
    policy gets another turn, so every cell on screen at every call the policy
    actually makes carries the baseline's verdict. A test pins that.
    """
    scenario = episode.scenario
    return score_episode(
        episode.as_logged(),
        scenario.grid,
        StepContext(offered_mitigations=frozenset(scenario.offered)),
        # Every step's label is passed; `score_episode` reads only the first,
        # because the evidence is paid once per episode.
        labels={s.n: scenario.label for s in episode.steps},
        points=POINTS,
    )


def rollout_scenario(
    scenario: Scenario,
    count: int,
    policy: AgentPolicy,
    generate: Callable[[list[str]], list[str]],
    *,
    advantage_fn: Callable[[list[float]], tuple[list[float], float, float]],
    log_root: Path | None = None,
    log_first: int = 0,
) -> tuple[dict[str, Any], list[Sample], list[dict[str, Any]]]:
    """Advance ``count`` episodes in lockstep, score them, and flatten.

    ``generate(list[str]) -> list[str]`` is injected rather than imported: it is
    the only part of a rollout that needs a GPU, so keeping it out makes the
    environment, the scoring, the flattening and the group statistics runnable
    on a CPU, and it is the seam where a different backend would attach.

    Every live episode of the group is advanced in one ``generate`` call, so an
    episode of depth *d* costs *d* batched calls rather than *d* sequential ones.
    A generator that returns the wrong number of completions is an error, not a
    silent ``zip`` truncation that would leave episodes waiting forever.
    """
    episodes = open_episodes(scenario, count, policy, log_root=log_root, log_first=log_first)
    depth = 0
    while True:
        live = pending(episodes)
        if not live:
            break
        depth += 1
        texts = generate([prompt for _, prompt in live])
        if len(texts) != len(live):
            raise ValueError(
                f"generate returned {len(texts)} completion(s) for {len(live)} prompt(s)"
            )
        for (episode, _), text in zip(live, texts, strict=True):
            episode.act(text)

    scores = [score(episode) for episode in episodes]
    rewards = [s.total for s in scores]
    adv, mean, sd = advantage_fn(rewards)

    samples: list[Sample] = []
    wire: list[dict[str, Any]] = []
    step_one: list[str] = []
    events: dict[str, int] = {}
    terminals: dict[str, int] = {}
    for index, (episode, fired, a, reward) in enumerate(
        zip(episodes, scores, adv, rewards, strict=True)
    ):
        terminals[episode.terminal] = terminals.get(episode.terminal, 0) + 1
        for event in fired.events:
            events[event.name] = events.get(event.name, 0) + 1
        for record in episode.steps:
            samples.append(
                Sample(
                    scenario_id=scenario.scenario_id,
                    prompt=record.prompt,
                    completion=record.raw,
                    advantage=a,
                    episode=index,
                    step=record.n,
                )
            )
            wire.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "episode": index,
                    "step": record.n,
                    "raw": record.raw,
                    "reward": reward,
                    "advantage": a,
                    "terminal": episode.terminal,
                    "episode_steps": len(episode.steps),
                    "cells": episode.cells_spent,
                }
            )
            if record.n == 1:
                step_one.append(record.raw)

    resolvers = set(scenario.resolvers)
    offered = list(scenario.offered)
    n = max(len(episodes), 1)

    def names_a_resolver(raw: str) -> bool:
        return bool(set(runnable_names(raw, offered)) & resolvers)

    group = {
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "n": len(episodes),
        "reward_mean": round(mean, 4),
        "reward_sd": round(sd, 4),
        "reward_spread": round(max(rewards) - min(rewards), 4) if rewards else 0.0,
        "advantage_spread": round(max(adv) - min(adv), 4) if adv else 0.0,
        # Distinct STEP-1 completions: step 1 is the same question for every
        # episode of the group, so this is the collapse check.
        "distinct_completions": len(set(step_one)),
        "steps_total": len(samples),
        "parsed_json": sum(_parses(s.completion) for s in samples),
        "rewards": [round(v, 4) for v in rewards],
        "max_depth": depth,
        "mean_steps": round(len(samples) / n, 3),
        "mean_cells": round(sum(e.cells_spent for e in episodes) / n, 3),
        "terminals": terminals,
        "events": events,
        # ⚠ Two rates, answering different questions. `step1` is the one to
        # compare across checkpoints: step 1 is the same prompt and menu for
        # every episode. `episode` is NOT comparable with it -- an episode
        # gets up to eight draws from a shrinking menu, so a high value there
        # is largely arithmetic. Reporting only the second would be the
        # easiest way to overstate what multi-step training does. Both are
        # None, not 0.0, on an unresolvable scenario, where there is no
        # resolver to name and 0.0 would read as failure.
        "step1_resolver_rate": (
            round(sum(names_a_resolver(raw) for raw in step_one) / max(len(step_one), 1), 4)
            if resolvers
            else None
        ),
        "episode_resolver_rate": (
            round(
                sum(any(names_a_resolver(s.raw) for s in e.steps) for e in episodes) / n, 4
            )
            if resolvers
            else None
        ),
        "converged_rate": round(sum(e.terminal == "converged" for e in episodes) / n, 4),
    }
    return group, samples, wire


def score_completion_compat(
    raw: str, scenario: Scenario, *, cells_added: int
) -> EventScore:
    """One reply scored as a single-step completion, with the loop's cell count.

    The control column for "does a constant get worse inside an episode": the
    same reply through ``event_reward.score_completion``, charged one baseline
    cell plus one per proposed name -- the accounting an episode uses -- so a
    difference between the two columns is the regime and not an offset.
    """
    return score_completion(
        raw,
        scenario.grid,
        StepContext(offered_mitigations=frozenset(scenario.offered)),
        cells_already_spent=1,
        cells_added=cells_added,
        label=scenario.label,
        points=POINTS,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _describe(root: Path) -> int:
    print(f"{'scenario':<24} {'cells':>5} {'menu':>5} {'family':<22} resolvers")
    refused = 0
    for scenario_id, archive, family in CORPUS:
        try:
            scenario = load_scenario(scenario_id, root / archive, family)
        except ValueError as exc:
            refused += 1
            print(f"{scenario_id:<24} REFUSED: {exc}")
            continue
        print(
            f"{scenario_id:<24} {len(scenario.grid.verdicts):>5} "
            f"{len(scenario.offered):>5} {family:<22} "
            f"{sorted(scenario.resolvers) or '(none -- unresolvable)'}"
        )
    return 1 if refused else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-root", default=None,
                        help=f"directory holding the archived matrices "
                             f"(default: ${CORPUS_ROOT_ENV})")
    parser.add_argument("--describe", action="store_true",
                        help="load every corpus scenario and print what it offers")
    args = parser.parse_args(argv)
    if args.describe:
        return _describe(resolve_corpus_root(args.corpus_root))
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
