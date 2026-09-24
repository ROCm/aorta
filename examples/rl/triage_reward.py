#!/usr/bin/env python3
"""Seam demonstration: a reward for triage classification, labelled by aorta.

The second reward identified alongside recipe synthesis. The task: hand the
model an archived probe run and ask it what happened -- did the bug reproduce
(``fail``), did the trial never validly run (``error``), or was it clean
(``pass``) -- and which detectors justify that. The label comes from aorta's own
verdict resolver, so the thing grading the model is the code the probe ships.

Same seam as ``recipe_reward.py``, same reason: a reward that reimplements the
precedence rules drifts from them, and a drifted label is worse than no label
because it still trains.

Two label sources
-----------------
``result.json`` from probe cells, and ``sanitizer_report.json`` from Waitcheck /
ConSan runs. Both answer the same question -- what happened, and what evidence
says so -- and ``--runs`` collects both from one tree.

The sanitizer source is the one that works on real data today: the six reports
under ``recipes/sanitizers/survey/reports/`` are the only archived labelled
failure evidence in the repository, and ``--runs recipes/sanitizers/survey``
scores them. The probe source has no corpus yet; what is missing there is *only*
the corpus, since the labelling, scoring and degenerate-policy check are tested
against synthetic ``result.json`` fixtures shaped like what
``SubprocessWorkload`` writes.

Labels are recomputed, not read
-------------------------------
Each fixture carries a ``verdict`` field, and this module ignores it. The label
is recomputed from the recorded detector IDs through
``aorta.probe.classifier.verdict``'s own ``partition_detectors`` and
``verdict_from_detectors``. Two reasons. First, it is the seam: if the
fail-over-error precedence changes, every label changes in the same commit.
Second, it detects corpus rot -- an archived run whose stored verdict disagrees
with today's rules is reported rather than silently trained on, and a run like
that is either a genuine rules change or a corrupted artifact. Either way the
reward should not quietly pick one.

Scoring
-------
Two terms, because either alone is trivially hackable:

``verdict``
    Exact match on ``pass``/``fail``/``error``. The fail-vs-error split is the
    one that matters operationally -- "the bug reproduced" versus "the trial
    never ran" -- and it is the distinction a plausible-sounding wrong answer
    most often gets backwards.

``attribution``
    F1 of the cited detector IDs against the fired ones. Without it, a policy
    that guesses the verdict from surface cues and invents a reason scores full
    marks; this is the "right answer, wrong reason" term.

The degenerate baseline
-----------------------
Probe corpora are skewed: most runs pass. So a policy that answers ``pass`` with
no detectors, always, scores well on verdict accuracy alone -- the classifier
analogue of the recipe reward's memorisation problem. ``--baselines`` scores
exactly that policy (and the always-``fail`` one) over the same fixtures, so the
number a real policy has to beat is stated rather than assumed. Report a
policy's reward against the baseline, never on its own.

Usage
-----

    python examples/rl/triage_reward.py               # fixtures + baselines
    python examples/rl/triage_reward.py --json
    python examples/rl/triage_reward.py --runs path/to/archived/runs
    python examples/rl/triage_reward.py --runs recipes/sanitizers/survey
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aorta.instrumentation.rocjitsu_sanitizers.models import SanitizerReport, Verdict
from aorta.probe.classifier.verdict import (
    VALID_VERDICTS,
    partition_detectors,
    verdict_from_detectors,
)

VERDICT_WEIGHT = 0.6
ATTRIBUTION_WEIGHT = 0.4

# Every verdict a corpus example may carry: the probe's three-way split plus the
# sanitizers' wider vocabulary. Taken from both enums rather than written out,
# so a new verdict upstream widens this automatically instead of making valid
# corpus rows look like corruption. The two spaces stay unmapped -- `warn` has
# no probe equivalent -- so this is a union, not a translation.
SANITIZER_VERDICTS = frozenset(v.value for v in Verdict) | frozenset(VALID_VERDICTS)


@dataclass
class Label:
    """Ground truth for one archived run, derived from aorta's resolver."""

    verdict: str
    failure_detectors: list[str] = field(default_factory=list)
    error_detectors: list[str] = field(default_factory=list)
    stored_verdict: str | None = None
    stale: bool = False
    source: str | None = None

    @property
    def cited_detectors(self) -> set[str]:
        """Every detector a correct answer should name."""
        return set(self.failure_detectors) | set(self.error_detectors)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "failure_detectors": self.failure_detectors,
            "error_detectors": self.error_detectors,
            "stored_verdict": self.stored_verdict,
            "stale": self.stale,
            "source": self.source,
        }


@dataclass
class Answer:
    """What the policy claims about a run."""

    verdict: str
    detectors: list[str] = field(default_factory=list)


@dataclass
class Score:
    verdict_correct: bool = False
    attribution_f1: float = 0.0
    reward: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict_correct": self.verdict_correct,
            "attribution_f1": round(self.attribution_f1, 4),
            "reward": round(self.reward, 4),
        }


def _detector_list(doc: dict[str, Any], key: str) -> list[str]:
    """``doc[key]`` as a list of detector IDs, or a refusal naming the shape.

    ``list(doc.get(key) or [])`` accepted anything iterable, and the shape that
    matters is a string: a writer that emitted ``"failure_detectors_fired":
    "tier1:sigsegv"`` instead of a one-element list produced
    ``['t', 'i', 'e', 'r', '1', ...]``, and every character went into
    ``partition_detectors`` as a detector ID. None are known, so they all sort
    to the failure side, the trial is labelled a reproduction, and fourteen
    fabricated IDs become the *ground truth* the attribution F1 is measured
    against -- a model naming the real detector scores zero against them.

    Nothing downstream can notice. The label is well-formed, the verdict is a
    legal verdict, and the only evidence is the ID list itself, which no longer
    resembles what the archive said. That is the corpus rot this module keeps
    refusing to let through quietly, so this raises rather than coercing, and
    the directory loaders turn it into a named skip beside their other ones.

    A missing key and a JSON ``null`` are still the empty list: absent is a
    real state for both fields -- a clean run fired no failure detectors -- and
    is what the archive means, not a shape it got wrong.
    """
    value = doc.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(
            f"{key} is a JSON {type(value).__name__}, not a list; "
            f"a bare string would be read one character per detector ID"
        )
    bad = [repr(item) for item in value if not isinstance(item, str)]
    if bad:
        raise TypeError(f"{key} holds non-string detector ID(s): {bad}")
    return list(value)


def label_run(doc: dict[str, Any], source: str | None = None) -> Label:
    """Recompute the verdict for one ``result.json`` through aorta's resolver.

    ``result.json`` stores failures and errors already partitioned, so the two
    lists are recombined and re-split rather than trusted: that routes the label
    through ``partition_detectors``, which owns which IDs mean "no valid
    observation". An archived run that recorded a detector on the wrong side of
    that line is corrected here, and flagged via ``stale``.

    With one exception, and it is upstream's own rule rather than ours.
    ``partition_detectors`` recognises exactly two error IDs -- ``tier1:timeout``
    and ``tier1:exec_failed`` -- and ``verdict.py`` says in its module docstring
    that ``error_detectors_fired`` "can still carry additional ``meta:``
    infra-error IDs, but those are written **directly** by a workload that
    bypasses :func:`resolve` entirely ... they are not produced by, and do not
    flow through, this resolver." So re-partitioning them asks the wrong
    authority: ``SubprocessWorkload`` writes ``meta:env_file_validation_failed``
    on the error side for a rejected ``probe.env``, where the subprocess never
    launched and the trial made no observation at all, and re-splitting moved
    it onto ``failure_detectors`` -- scoring a trial that never ran as a
    *reproduction*, marking the archive ``stale``, and feeding the wrong ground
    truth to the attribution term as well as the verdict one.

    The carve-out is **positional, not by prefix**, and that distinction is
    load-bearing: ``meta:`` is not a synonym for "infra error". ``resolve``
    synthesises ``meta:missing_pass_signal`` as a genuine *failure* signal and
    appends it to the failure list, so a rule keyed on the prefix would flip a
    real failure into an error -- the same defect in the opposite direction.
    What is preserved is where the producer *put* it, which is the judgement
    this function otherwise exists to re-derive and is the one case where the
    producer knows something the resolver does not.
    """
    stored_failures = _detector_list(doc, "failure_detectors_fired")
    stored_errors = _detector_list(doc, "error_detectors_fired")

    # Held back from the re-split rather than re-classified afterwards, so the
    # resolver is never asked about an ID it documents itself as not owning.
    bypassed = [d for d in stored_errors if d.startswith("meta:")]
    recorded = stored_failures + [d for d in stored_errors if d not in bypassed]
    failures, errors = partition_detectors(recorded)
    errors = errors + bypassed
    verdict = verdict_from_detectors(failures, errors)

    # `stale` has to separate three states, not two: the archive recorded no
    # verdict, it recorded one that agrees, or it recorded one that does not.
    # Normalising an invalid value to `None` collapsed the third into the first
    # -- `verdict: "fal"` read as *silent* rather than as *disagreeing*, and
    # silence is the one state this signal does not report. A rotted field is
    # exactly the corpus-rot the staleness flag exists to surface, so anything
    # that is not the recomputed verdict now counts as disagreement, whether it
    # is a different verdict or not a verdict at all.
    #
    # A present-but-null `verdict` stays in the absent case deliberately: JSON
    # null is how a writer records "no verdict", not a corrupted one.
    raw_stored = doc.get("verdict")
    if raw_stored is None:
        stored, stale = None, False
    else:
        # Carried through as text rather than dropped, so the STALE line names
        # what the archive actually held. `verdict != stored` covers both the
        # wrong-verdict and the not-a-verdict case, since `verdict` is always
        # one the resolver produced.
        stored = raw_stored if isinstance(raw_stored, str) else repr(raw_stored)
        stale = stored != verdict
    return Label(
        verdict=verdict,
        failure_detectors=failures,
        error_detectors=errors,
        stored_verdict=stored,
        stale=stale,
        source=source,
    )


def score_answer(answer: Answer, label: Label) -> Score:
    """Reward one answer against the ground truth.

    Attribution is F1 rather than exact-set-match so a partially right citation
    earns partial credit; exact match would make a 4-of-5 answer worth the same
    as a fabricated one. A ``pass`` run cites nothing, and an answer that also
    cites nothing is perfect attribution by convention -- scoring an empty-empty
    comparison as 0.0 would penalise the correct answer on the commonest class.
    """
    score = Score()
    score.verdict_correct = answer.verdict == label.verdict

    predicted, actual = set(answer.detectors), label.cited_detectors
    if not actual and not predicted:
        score.attribution_f1 = 1.0
    elif not actual or not predicted:
        score.attribution_f1 = 0.0
    else:
        overlap = len(predicted & actual)
        if overlap == 0:
            score.attribution_f1 = 0.0
        else:
            precision = overlap / len(predicted)
            recall = overlap / len(actual)
            score.attribution_f1 = 2 * precision * recall / (precision + recall)

    score.reward = (
        VERDICT_WEIGHT * (1.0 if score.verdict_correct else 0.0)
        + ATTRIBUTION_WEIGHT * score.attribution_f1
    )
    return score


def load_runs(root: Path) -> list[tuple[str, dict[str, Any]]]:
    """Every readable ``result.json`` under a directory of archived probe runs.

    Unreadable files are skipped *loudly*. Dropping them silently meant a
    half-corrupt archive still produced a mean reward that looked complete, with
    nothing to say it had been computed over fewer runs than the directory
    holds -- and the sanitizer-report loader beside this one already prints its
    skips, so the quiet one was the odd case rather than the convention.

    **Parsing is not the same as loading.** A file holding `[]` or `"fail"` is
    syntactically valid JSON, so it survived the decode and was appended, and
    then took down the whole `--runs` sweep from inside `label_run` --
    `doc.get` on a list is an `AttributeError`, which nothing here catches. One
    rotted file in a directory of hundreds therefore scored none of them. The
    root shape is checked at the seam where every other rejection is already
    reported, so it skips loudly like the rest rather than aborting.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(root.rglob("result.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipped {path}: unreadable ({exc})", file=sys.stderr)
            continue
        if not isinstance(doc, dict):
            print(f"  skipped {path}: root is a JSON {type(doc).__name__}, not "
                  f"an object, so it carries no detector lists to label",
                  file=sys.stderr)
            continue
        # The detector lists, checked here for the reason the root shape is:
        # this is the seam that reports rejections, and `label_run` is called
        # later over the whole batch with nothing between it and the sweep. A
        # malformed list is worse than an unreadable file, because it does not
        # raise -- see `_detector_list` -- so it has to be refused where a
        # refusal costs one run and gets named.
        try:
            for key in ("failure_detectors_fired", "error_detectors_fired"):
                _detector_list(doc, key)
        except TypeError as exc:
            print(f"  skipped {path}: malformed detector list ({exc})",
                  file=sys.stderr)
            continue
        out.append((str(path), doc))
    return out


def label_sanitizer_report(doc: dict[str, Any], source: str | None = None) -> Label:
    """Label one ``sanitizer_report.json`` through aorta's own report model.

    The second label source for the root-cause half, and the one the CIA tool
    fleet makes on-domain: Waitcheck and ConSan output is richer evidence than a
    detector ID, because a ``Finding`` names a code, a severity and often a
    kernel and code-object offset.

    The seam is stronger here than for ``result.json``. ``SanitizerReport``
    recomputes ``overall_verdict`` as the max-ranked check verdict in
    ``__post_init__``, and ``from_dict`` *raises* when the stored value
    contradicts that recomputation. So a rotted report cannot be silently
    trained on -- it fails to load, and the caller records why. The verdict
    vocabulary is the sanitizer's own (``pass``/``warn``/``fail``/
    ``not_checked``/``error``), which is wider than the probe's three-way split;
    the two label spaces are kept separate rather than mapped onto each other,
    because ``warn`` has no probe equivalent and inventing one would be a
    judgement the tools did not make.

    Cited evidence is the set of finding codes, namespaced by the sanitizer that
    produced them -- ``waitcheck:wait_hazard`` rather than ``wait_hazard`` -- so
    attribution reads the same way as a ``tier4:`` detector ID and cannot be
    confused with one.

    **Codes are cited only by a ``fail`` or ``warn`` report.** A finding is
    evidence that a sanitizer ran and saw something, so it cannot be the
    justification for a verdict that says none ran. See the comment on
    ``failures`` below for the mixed-verdict case that makes this more than a
    restatement.
    """
    report = SanitizerReport.from_dict(doc)
    codes: list[str] = []
    for check in report.checks:
        findings = list(check.findings)
        for kernel_result in check.kernel_results:
            findings.extend(kernel_result.findings)
        for finding in findings:
            code = f"{finding.sanitizer or check.sanitizer}:{finding.code}"
            if code not in codes:
                codes.append(code)

    verdict = report.overall_verdict.value
    # A sanitizer that did not run is an infra error, exactly as a probe trial
    # that never validly ran is: no observation was made, so there is nothing
    # to attribute.
    #
    # And nothing is what it now gets. Filing `codes` under `error_detectors`
    # looked like it mirrored the probe side, where `cited_detectors` is the
    # union of both lists -- but the two lists do not hold the same kind of
    # thing. On the probe side an error detector *is* the evidence of the
    # error: `tier1:exec_failed` is why the trial did not validly run. Here
    # the codes are findings, and a finding is evidence that a sanitizer ran
    # and saw something. `overall_verdict` ranks `error` above `warn`, so a
    # report whose overall verdict is `error` can still carry findings from a
    # *different* check that merely warned -- and those codes justify the
    # warning, not the error. Handing them to the scorer paid full attribution
    # credit for citing evidence of the wrong event, which is the "right
    # answer, wrong reason" case the attribution term exists to dock, arriving
    # through the ground truth instead of through the answer.
    #
    # So finding codes can only ever justify `fail` or `warn`, and an
    # `error`/`not_checked` report cites nothing -- which is what the sentence
    # above it always claimed.
    failures = codes if verdict in {"fail", "warn"} else []
    errors: list[str] = []
    return Label(
        verdict=verdict,
        failure_detectors=failures,
        error_detectors=errors,
        stored_verdict=verdict,
        stale=False,
        source=source,
    )


def load_sanitizer_reports(root: Path) -> list[tuple[str, Label]]:
    """Every loadable ``sanitizer_report.json`` under a directory.

    A report that fails aorta's own consistency check is skipped and named,
    never silently coerced: that is the corpus-rot signal for this label source.

    The root-shape check is the same one `load_runs` makes and is here for the
    same reason: `SanitizerReport.from_dict` on a list raises `AttributeError`,
    which the `ValueError`/`KeyError`/`TypeError` clause below does not catch,
    so a single non-object file aborted the sweep instead of being skipped.
    """
    out: list[tuple[str, Label]] = []
    for path in sorted(root.rglob("sanitizer_report.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipped {path}: unreadable ({exc})", file=sys.stderr)
            continue
        if not isinstance(doc, dict):
            print(f"  skipped {path}: root is a JSON {type(doc).__name__}, not "
                  f"an object, so it is not a sanitizer report",
                  file=sys.stderr)
            continue
        try:
            out.append((str(path), label_sanitizer_report(doc, source=str(path))))
        except (ValueError, KeyError, TypeError) as exc:
            print(f"  skipped {path}: rejected by aorta's report model ({exc})",
                  file=sys.stderr)
    return out


def load_corpus(
    path: Path, *, include_disagreements: bool = False
) -> list[tuple[str, Label, str]]:
    """Load a `build_corpus.py` triage JSONL as labels, with their families.

    The corpus stores the label rather than recomputing it from the report,
    because the report may not travel with it. The seam is not lost: the label
    was produced by ``label_sanitizer_report`` at build time, so it passed
    ``SanitizerReport.from_dict`` then. What is checked here instead is that the
    verdict is one this scorer knows, so a corpus written by a newer builder
    with a wider vocabulary fails loudly rather than scoring as a mismatch
    against every answer.

    **Rows whose observed verdict contradicts the committed baseline are skipped
    by default.** ``build_corpus.py`` deliberately emits them -- a disagreement
    is evidence of a tool defect and dropping it at build time would hide the
    one thing worth reporting -- and flags them ``ground_truth.agrees = false``
    for a consumer to act on. Nothing acted on it, so the advertised
    no-conversion path scored a policy against the *observed* verdict on exactly
    the scenarios where the observed verdict is known to be wrong: the corpus
    would teach the defect. Pass ``include_disagreements=True`` to score them
    anyway, which is the right setting when the question is how the tool behaves
    rather than what the right answer is.

    Rows with no baseline (``agrees`` is ``None``) are kept. Ungated is not the
    same as contradicted, and most of the corpus is ungated.
    """
    out: list[tuple[str, Label, str]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        # Same seam as the two directory loaders: a JSON list or scalar decodes
        # cleanly and then fails on `.get` with an `AttributeError` naming
        # neither the file nor the line. This one raises rather than skips,
        # because the corpus is the trained-on artifact -- a row it cannot read
        # is a build defect to fix, not a file to step over.
        if not isinstance(row, dict):
            raise ValueError(
                f"{path}:{line_number}: row is a JSON {type(row).__name__}, "
                f"not an object"
            )
        if row.get("kind") != "triage":
            continue
        if not include_disagreements:
            ground_truth = row.get("ground_truth") or {}
            if ground_truth.get("agrees") is False:
                continue
        stored = row["label"]
        verdict = stored["verdict"]
        if verdict not in SANITIZER_VERDICTS:
            raise ValueError(
                f"{path}:{line_number}: verdict {verdict!r} is outside this "
                f"scorer's vocabulary {sorted(SANITIZER_VERDICTS)}"
            )
        # The refusal `_detector_list` already makes on the `result.json` path.
        # `list()` read `"consan:1"` as eight one-character IDs and made them
        # the attribution ground truth. Ahead of the check below, which reads
        # the same two fields and would otherwise quote those characters back
        # as the row's citation.
        try:
            failure_detectors = _detector_list(stored, "failure_detectors")
            error_detectors = _detector_list(stored, "error_detectors")
        except TypeError as exc:
            raise ValueError(
                f"{path}:{line_number}: {exc}. A corpus row is ground truth, "
                "so it is refused rather than repaired; rebuild it with "
                "build_corpus.py."
            ) from exc
        stale = stored.get("stale")
        if stale is None:
            stale = False
        elif not isinstance(stale, bool):
            raise ValueError(
                f"{path}:{line_number}: stale is a JSON "
                f"{type(stale).__name__}, not a boolean, and `bool()` reads "
                f"{stale!r} as true; rebuild it with build_corpus.py."
            )
        # The same invariant `label_sanitizer_report` now holds, checked on the
        # way back in. Rows are what the scorer actually trains against, and a
        # corpus built before that fix carries finding codes under
        # `error_detectors` for an `error` report -- evidence of a warning
        # filed as the justification for an infra error. Reading it back
        # unexamined would let the defect outlive the commit that removed it,
        # in exactly the artifact nothing else inspects.
        #
        # Refused rather than dropped, matching the verdict check above: this
        # function already treats a row outside the scorer's vocabulary as a
        # corpus to rebuild rather than a row to repair, and `.jsonl` corpora
        # are generated artifacts that one `build_corpus.py` run replaces.
        # Silently emptying the list would score the run against a ground
        # truth the file does not contain.
        #
        # Both lists, not just `error_detectors`, though that is the one the
        # old code populated. The invariant is "cites nothing", and
        # `cited_detectors` is their union, so checking one half leaves a row
        # that reaches the scorer with the same miscitation through the other.
        if verdict in {"error", "not_checked"}:
            cited = [*failure_detectors, *error_detectors]
            if cited:
                raise ValueError(
                    f"{path}:{line_number}: a {verdict!r} report cites "
                    f"{cited} as evidence, but finding codes are evidence a "
                    "sanitizer ran and saw something, so they can only "
                    "justify 'fail' or 'warn'. This corpus predates that "
                    "rule; rebuild it with build_corpus.py."
                )
        out.append((
            row["example_id"],
            Label(
                verdict=verdict,
                failure_detectors=failure_detectors,
                error_detectors=error_detectors,
                stored_verdict=stored.get("stored_verdict"),
                stale=stale,
                source=row["example_id"],
            ),
            row.get("workload_family", "unknown"),
        ))
    return out


def score_policy(
    name: str,
    answer_for: Any,
    labelled: list[tuple[str, Label]],
) -> dict[str, Any]:
    """Mean reward for a callable policy over a labelled set."""
    scores = [score_answer(answer_for(label), label) for _, label in labelled]
    n = len(scores) or 1
    return {
        "policy": name,
        "runs": len(scores),
        "verdict_accuracy": round(sum(s.verdict_correct for s in scores) / n, 4),
        "mean_attribution_f1": round(sum(s.attribution_f1 for s in scores) / n, 4),
        "mean_reward": round(sum(s.reward for s in scores) / n, 4),
    }


# --------------------------------------------------------------------------- #
# Synthetic fixtures
#
# Shaped like what SubprocessWorkload writes (aorta/workloads/_subprocess.py):
# a `verdict`, the three partitioned detector lists, and the run's exit/timing
# fields.
#
# These are failure-heavy, which a real probe archive is not. That is
# deliberate -- the failure shapes are what the reward has to discriminate, and
# there is no point spending fixtures on the class a real corpus supplies in
# bulk. It does mean the printed baselines are read off *this* mix and not off a
# forecast of a real one: on these fixtures the always-`pass` floor is low and
# the always-`fail` floor is high, and both flip once the archive arrives. The
# baseline check is the invariant, not the numbers it currently prints.
# --------------------------------------------------------------------------- #

def _run(
    cell: str,
    verdict: str,
    failures: list[str],
    errors: list[str],
    warns: list[str] | None = None,
    exit_code: int = 0,
) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "exit_code": exit_code,
        "walltime_sec": 12.5,
        "peak_vram_mib": 4096,
        "argv": ["python", "repro.py"],
        "cell_name": cell,
        "trial_index": 0,
        "failure_detectors_fired": failures,
        "error_detectors_fired": errors,
        "warn_detectors_fired": warns or [],
        "capture": {},
        "tier_durations_ms": {"tier1": 3.0, "tier2": 1.0},
    }


FIXTURES: tuple[dict[str, Any], ...] = (
    _run("baseline-clean", "pass", [], []),
    _run("baseline-clean-2", "pass", [], []),
    _run("baseline-clean-3", "pass", [], []),
    _run("warn-only", "pass", [], [], warns=["tier3:vram_growth"]),
    _run("nonzero-exit", "fail", ["tier1:exit_nonzero"], [], exit_code=1),
    _run("segfault-and-reset", "fail",
         ["tier1:sigsegv", "tier3:amdgpu_reset"], [], exit_code=-11),
    _run("hang", "fail", ["tier2:hang"], [], exit_code=-9),
    # Only an infra signal fired: the trial never validly ran. This is the
    # fail-vs-error distinction the verdict term is really testing.
    _run("launch-failed", "error", [], ["tier1:exec_failed"], exit_code=127),
    _run("timeout-no-hang", "error", [], ["tier1:timeout"], exit_code=-9),
    # fail > error precedence: a genuine failure alongside an infra signal is a
    # reproduction, not a flake.
    _run("failure-plus-infra", "fail",
         ["tier1:sigabrt"], ["tier1:timeout"], exit_code=-6),
    # A required custom pattern that never fired -- a synthesised failure with
    # no underlying crash, which surface heuristics tend to read as a pass.
    _run("missing-pass-signal", "fail", ["meta:missing_pass_signal"], []),
    # --- The debugging vertical proper ------------------------------------- #
    # The cases above exercise the verdict precedence rules; these are the
    # failure shapes `aorta agent` is actually pointed at, one per autopsy
    # category the proposal contract enumerates. Detector IDs are the real
    # ones -- tier2_hang, tier3_kernel and tier4_patterns own them -- because a
    # fixture citing an ID that no tier can emit trains attribution against a
    # vocabulary that does not exist.
    #
    # rccl_hang: the collective times out and the monitor also sees no
    # progress, so two detectors justify one verdict and a correct answer must
    # cite both. This is the case attribution F1 exists for.
    _run("collective-timeout", "fail",
         ["tier2:hang", "tier4:collective_timeout", "tier4:nccl_rccl_error"],
         [], exit_code=-9),
    # illegal_mem: the page fault is the cause and the HIP error is the report
    # of it. Citing only the tier4 line is the plausible half-answer.
    _run("illegal-access", "fail",
         ["tier3:vm_l2_fault", "tier4:hip_error", "tier4:python_traceback"],
         [], exit_code=1),
    # thermal_throttle is a *failure* detector, not advisory -- only
    # tier3:vram_growth is a warn. A policy that reads "throttle" as a
    # performance note and answers pass is wrong here.
    _run("thermal-throttle", "fail", ["tier3:thermal_throttle"], []),
    # A fabric fault with a downstream collective error: the interconnect is
    # the story and the collective is the symptom.
    _run("xgmi-fault", "fail",
         ["tier3:xgmi_link_error", "tier4:collective_timeout"], [], exit_code=-9),
    # checkpoint_race: no crash, no signal, just wrong numbers.
    _run("nan-signature", "fail", ["tier4:nan_signature"], [], exit_code=0),
    # A reset after a segfault, with vram growth as a red herring: the warn
    # detector must not be cited as justification for the verdict.
    _run("reset-with-warn", "fail",
         ["tier1:sigsegv", "tier3:amdgpu_reset"], [],
         warns=["tier3:vram_growth"], exit_code=-11),
    # An SDMA timeout that the harness also recorded as an infra timeout:
    # fail > error again, but with a kernel-tier cause rather than a signal.
    _run("sdma-timeout", "fail",
         ["tier3:sdma_timeout"], ["tier1:timeout"], exit_code=-9),
)


def run_demo(
    as_json: bool,
    runs_root: Path | None,
    corpus: Path | None = None,
    *,
    include_disagreements: bool = False,
) -> int:
    families: dict[str, int] = {}
    if corpus is not None:
        rows = load_corpus(corpus, include_disagreements=include_disagreements)
        if not rows:
            print(f"no triage examples in {corpus}", file=sys.stderr)
            return 2
        labelled = [(src, label) for src, label, _ in rows]
        for _, _, family in rows:
            families[family] = families.get(family, 0) + 1
    elif runs_root is not None:
        loaded = load_runs(runs_root)
        labelled = [(src, label_run(doc, src)) for src, doc in loaded]
        # Sanitizer reports live alongside probe results in a real run tree, and
        # both are evidence for the same question, so one --runs sweep collects
        # both rather than making the caller know which kind they have.
        labelled += load_sanitizer_reports(runs_root)
        if not labelled:
            print(
                f"no result.json or sanitizer_report.json found under {runs_root}",
                file=sys.stderr,
            )
            return 2
    else:
        labelled = [
            (f"synthetic:{d['cell_name']}", label_run(d, f"synthetic:{d['cell_name']}"))
            for d in FIXTURES
        ]

    # Three policies over the same runs. An oracle to prove the scorer pays a
    # correct answer, a plausible-but-wrong one, and the degenerate constant.
    def oracle(label: Label) -> Answer:
        return Answer(label.verdict, sorted(label.cited_detectors))

    def right_verdict_wrong_reason(label: Label) -> Answer:
        return Answer(label.verdict, ["tier9:invented"] if label.cited_detectors else [])

    def always_pass(_: Label) -> Answer:
        return Answer("pass", [])

    def always_fail(label: Label) -> Answer:
        return Answer("fail", sorted(label.cited_detectors))

    policies = [
        score_policy("oracle", oracle, labelled),
        score_policy("right verdict, invented detectors", right_verdict_wrong_reason, labelled),
        score_policy("degenerate: always pass, no detectors", always_pass, labelled),
        score_policy("degenerate: always fail, correct detectors", always_fail, labelled),
    ]

    stale = [(src, lb) for src, lb in labelled if lb.stale]
    distribution: dict[str, int] = {}
    for _, lb in labelled:
        distribution[lb.verdict] = distribution.get(lb.verdict, 0) + 1

    if as_json:
        print(json.dumps({
            "runs": [{"source": s, "label": lb.as_dict()} for s, lb in labelled],
            "verdict_distribution": distribution,
            "workload_families": families,
            "policies": policies,
            "stale": [s for s, _ in stale],
        }, indent=2))
        return 0

    print("Seam demonstration: triage-classification reward, labelled by aorta.")
    print(f"reward = {VERDICT_WEIGHT} * verdict-correct + {ATTRIBUTION_WEIGHT} * attribution-F1\n")
    print(f"{len(labelled)} run(s); verdict distribution: {distribution}")
    if families:
        print(f"workload families: {families}")
    print()

    for source, label in labelled:
        cited = ", ".join(sorted(label.cited_detectors)) or "(none)"
        print(f"  {label.verdict:<5}  {source}")
        print(f"         detectors: {cited}")
        if label.stale:
            print(f"         STALE: stored verdict {label.stored_verdict!r} "
                  f"disagrees with today's rules")
    print()

    print("Policy scores over those runs:")
    for entry in policies:
        print(f"  {entry['policy']}")
        print(f"       verdict accuracy {entry['verdict_accuracy']:.3f}   "
              f"attribution F1 {entry['mean_attribution_f1']:.3f}   "
              f"reward {entry['mean_reward']:.3f}")
    print()

    pass_share = distribution.get("pass", 0) / (len(labelled) or 1)
    degenerate = next(p for p in policies if p["policy"].startswith("degenerate: always pass"))
    print(f"The floor to beat: {pass_share:.0%} of these runs are 'pass', so the")
    print(f"always-pass policy already earns {degenerate['mean_reward']:.3f} without reading")
    print("anything. A real policy's reward is only meaningful above that number,")
    print("which is why it is printed next to it and not on its own.")
    if stale:
        print(f"\n{len(stale)} run(s) carry a stored verdict that disagrees with the")
        print("current rules. They are scored, under the verdict recomputed here")
        print("rather than the one archived: the resolver is the label, and an old")
        print("run recorded before a detector moved sides is still good evidence.")
        print("The flag is so a reader can see which rows that applies to.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Reward triage classifications using aorta's own verdict resolver.",
    )
    ap.add_argument("--runs", type=Path, default=None,
                    help="directory of archived probe runs (result.json files); "
                         "omit to use the synthetic fixtures")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="triage.jsonl written by build_corpus.py; scores the "
                         "labelled corpus directly, with no conversion pass")
    ap.add_argument("--include-disagreements", action="store_true",
                    help="also score rows whose observed verdict contradicts "
                         "the committed baseline. Excluded by default: on those "
                         "scenarios the label is a known tool defect, so scoring "
                         "them rewards reproducing it")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args(argv)
    return run_demo(
        args.json, args.runs, args.corpus,
        include_disagreements=args.include_disagreements,
    )


if __name__ == "__main__":
    sys.exit(main())
