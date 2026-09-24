"""Watch's corpus: a log delta, joined to what the failure turned out to be.

The join is on ``job_id``, which is the directory name under the jobs root.
Two artifacts meet there:

* ``bundle/logs/watch.stderr.log`` -- written by ``write_bundle(job, job_dir,
  evidence or new_content[:4000], signal)``, so it is either the evidence lines
  Watch quoted or up to four thousand characters of the delta that alerted. This
  is the state a local-classifier tier would read.
* ``bundle/report.json`` -- written by ``aorta.cia.triage``, carrying the
  ``category`` an Autopsy reached after looking at the whole bundle. This is the
  label, and it is the right one precisely because it is *later*: it is what the
  failure turned out to be, not what Watch guessed at the time.

The events JSONL is read too, but only for the baseline. Its ``excerpt`` is
capped at 500 characters, which is too short to train on and is the reason the
bundle is preferred for the state.

**Two things about this corpus are worse than the plan assumed, and they are
reported rather than papered over.**

First, the vocabularies do not correspond. Watch answers with one of six
``WATCH_*`` slugs; Autopsy answers with one of eleven categories. Nothing in the
tree maps between them, so :data:`CATEGORY_TO_SIGNAL` below is new, and it is
lossy in both directions: four categories collapse onto ``WATCH_UNKNOWN_ERROR``,
and ``WATCH_LOSS_STALL`` has no category that reaches it at all. Every build
counts the collapse and says so, because an accuracy figure over a label
distribution that is mostly one slug is a majority-class score wearing a
macro-accuracy label.

The questions themselves come from ``aorta.cia.watch.watcher``, which is the tier
that asks them -- see :func:`watch_questions`. This module held its own copies
first, which is the defect the proposer builder had in the same place: a corpus
fitted against one wording and a tier asking another does not fail, it answers
slightly worse for a reason nobody would go looking for.

Second, there are no healthy states at full length. ``write_bundle`` runs only
after Watch alerts, so every bundle on disk is a failure. The only healthy
deltas recorded anywhere are the ``watchdog_ok`` excerpts, capped at 500
characters -- so a ``watch_healthy`` corpus built from what exists has a
positive class of up to 4000 characters and a negative class of at most 500, and
a classifier can score well on it by measuring length. The builder emits those
examples because half a corpus is a real starting point, and warns every time,
because a number from it is not a measurement of whether an encoder can read a
ROCm log.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from aorta.local_classifier.corpus.schema import (
    BuildResult,
    CorpusError,
    LabelledExample,
    iter_json_lines,
)
from aorta.local_classifier.predictor import Choice, Noul

#: How an eventual Autopsy category is read back as the signal Watch should have
#: emitted. New in this module, because nothing in the tree relates the two
#: vocabularies -- see this module's docstring for why that is a finding and not
#: a convenience.
#:
#: ``unknown`` and ``tooling_gap`` are deliberately absent, so an example whose
#: Autopsy landed on either is skipped rather than labelled. ``unknown`` means
#: the Autopsy could not tell, which is not ground truth about the log;
#: ``tooling_gap`` means the instrument never ran, so the bundle proves nothing
#: about the delta either way. Labelling both as ``WATCH_UNKNOWN_ERROR`` would
#: teach the classifier to emit an error slug for a healthy log whose sanitizer
#: happened to be missing.
CATEGORY_TO_SIGNAL: dict[str, str] = {
    "numeric_silent": "WATCH_NUMERIC_NAN",
    "rccl_hang": "WATCH_HANG",
    "oom_fragment": "WATCH_OOM",
    "thermal_throttle": "WATCH_THROUGHPUT_LOW",
    "perf_regression": "WATCH_THROUGHPUT_LOW",
    # The remaining four are real failures with no slug of their own. Watch's
    # vocabulary has one bucket for "clear error but does not fit above", and
    # this is it -- which is why the collapse is counted on every build.
    "illegal_mem": "WATCH_UNKNOWN_ERROR",
    "launch_error": "WATCH_UNKNOWN_ERROR",
    "checkpoint_race": "WATCH_UNKNOWN_ERROR",
    "gpu_race": "WATCH_UNKNOWN_ERROR",
}

#: Categories that mean "no conclusion about this log", so no label.
UNLABELLABLE_CATEGORIES: frozenset[str] = frozenset({"unknown", "tooling_gap", ""})

#: Signal slugs Watch emits about its own failures rather than about a job's.
#: A delta that could not be assessed says nothing about what the delta was.
_NON_VERDICT_SIGNALS: frozenset[str] = frozenset(
    {"WATCH_ASSESSMENT_FAILED", "WATCH_LOG_READ_FAILED"}
)

def watch_questions() -> tuple[Noul, Choice]:
    """The two questions Watch's local-classifier tier asks, from the tier's own definitions.

    This module used to hold its own copies, which was the defect the proposer
    builder had in the same place and for the same reason: a corpus labelled
    against one wording and a tier asking another does not fail, it answers
    slightly worse for a reason nobody would go looking for. ``watcher.py`` is
    canonical because the ``WatchAssessment`` signature's slug list is there.

    **The assemblers, not the five strings.** Track B splits into strings because
    its two sides legitimately differ in options -- the proposer offers
    ``PROBE_CATEGORIES`` and the router ``AUTOPSY_CATEGORIES`` -- so each side has
    to build its own. Watch's two sides want the identical object, and
    re-assembling one from parts here would be five chances to drop a gloss or
    reorder the slugs. Order is load-bearing rather than cosmetic:
    :class:`~aorta.local_classifier.predictor.ChoiceAnswer` re-reads a distribution in the
    order offered, and the ``criteria`` keys *are* the answer space, so a slug
    missing from them is a slug the model is never offered.

    Imported here rather than at module scope, following ``_dir_listing`` in
    ``corpus/log_finder.py``. ``aorta.cia.watch.watcher`` imports dspy, and
    reaching it eagerly would make the whole ``aorta.local_classifier`` package need the
    ``[cia]`` extra for the two builders that have no use for it.
    """
    try:
        from aorta.cia.watch.watcher import healthy_question, signal_question
    except ImportError as exc:
        raise CorpusError(
            "the Watch corpus labels against the questions Watch's own local-classifier tier "
            "asks, which needs the agents' extra: pip install 'amd-aorta[cia]'. "
            "Keeping a second copy of them here is what this import replaced.\n"
            f"(missing: {exc.name or 'dspy'})"
        ) from exc
    return healthy_question(), signal_question()


def watch_signals() -> tuple[str, ...]:
    """The slugs the signal choice offers, read off the question rather than beside it.

    Off ``signal_question().options`` and not off ``watcher.WATCH_SIGNALS``, even
    though today they are the same tuple. The question is what a row is labelled
    against, so anything validating a label has to read the set the question
    actually offered -- otherwise a slug could be dropped from the criteria and a
    label for it would still validate against the list next door.
    """
    return watch_questions()[1].options


#: What ``write_bundle`` calls the delta it persists, relative to the bundle.
_BUNDLE_DELTA = Path("logs") / "watch.stderr.log"

#: Below this, the state is a line or two and carries nothing to read. Bundles
#: this small happen: ``evidence`` can be a single quoted line.
_MIN_STATE_CHARS = 40


def build_watch_corpus(jobs_root: str | Path) -> BuildResult:
    """Build the Watch corpus from the job directories under *jobs_root*.

    *jobs_root* is what ``CIA_JOBS_ROOT`` names and what ``aorta.cia.triage``
    defaults to ``~/cia-jobs``. Every immediate subdirectory holding a
    ``job.json`` is a candidate.
    """
    root = Path(jobs_root).expanduser()
    result = BuildResult()
    if not root.is_dir():
        result.warn(f"no jobs root at {root}, so there is nothing to join on")
        return result

    # Resolved once, before the walk, and threaded down rather than read as a
    # module global. One import site means one place the extra can be missing, and
    # it means the questions on every row of a build are the same objects.
    healthy, signal = watch_questions()
    for job_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (job_dir / "job.json").is_file():
            continue
        result.scanned += 1
        _add_from_bundle(job_dir, result, healthy, signal)
        _add_healthy_from_events(job_dir, result, healthy)

    _warn_about_the_shape_of_it(result, signal.options)
    return result


def _add_from_bundle(
    job_dir: Path, result: BuildResult, healthy: Noul, signal_q: Choice
) -> None:
    """The alerting delta, labelled by the Autopsy that followed it."""
    delta_path = job_dir / "bundle" / _BUNDLE_DELTA
    state = _read_text(delta_path)
    if state is None:
        result.skip("no bundle: Watch never alerted on this job")
        return
    if len(state.strip()) < _MIN_STATE_CHARS:
        result.skip("bundle delta too short to read")
        return

    report = _read_json(job_dir / "bundle" / "report.json")
    if report is None:
        result.skip("bundle written but no Autopsy report, so no label")
        return
    category = str(report.get("category") or "")
    if category in UNLABELLABLE_CATEGORIES:
        result.skip(f"Autopsy reached {category or 'no category'}, which is not ground truth")
        return
    signal = CATEGORY_TO_SIGNAL.get(category)
    if signal is None:
        # A category outside AUTOPSY_CATEGORIES, or one added since this map was
        # written. Refusing to guess is the point: an unmapped category silently
        # bucketed into WATCH_UNKNOWN_ERROR is how a taxonomy drifts.
        result.skip(f"Autopsy category {category!r} has no Watch slug in CATEGORY_TO_SIGNAL")
        result.warn(
            f"category {category!r} is not in CATEGORY_TO_SIGNAL; either it is new or the "
            "map is stale. Extend the map deliberately rather than letting it default."
        )
        return

    baselines = _recorded_baselines(job_dir)
    source = str(delta_path)
    job_id = job_dir.name
    result.examples.append(
        LabelledExample(
            decision="watch_signal",
            state=state,
            question=signal_q,
            label=signal,
            join_key=job_id,
            source=source,
            baselines=baselines,
        )
    )
    # The same delta is also a labelled negative for the clean-gate: an Autopsy
    # that reached a real category is the observation that this delta was not
    # healthy.
    result.examples.append(
        LabelledExample(
            decision="watch_healthy",
            state=state,
            question=healthy,
            label="false",
            join_key=job_id,
            source=source,
            baselines=baselines,
        )
    )


def _add_healthy_from_events(job_dir: Path, result: BuildResult, healthy: Noul) -> None:
    """Healthy deltas, such as they are: the 500-character ``watchdog_ok`` excerpts.

    Emitted knowing they are truncated, and warned about on every build. Without
    them the corpus has no negative class at all and the clean-gate cannot be
    measured even in principle; with them it can be measured badly, which is a
    starting point a reader can be told about.
    """
    for event in iter_json_lines(job_dir / "events.jsonl"):
        if event.get("event_type") != "watchdog_ok":
            continue
        excerpt = str(event.get("excerpt") or "")
        # A healthy assessment quotes nothing, because ``evidence`` is 'none'
        # when nothing is wrong -- so most watchdog_ok events carry an empty
        # excerpt and are not states at all.
        if len(excerpt.strip()) < _MIN_STATE_CHARS:
            result.skip("watchdog_ok event carries no excerpt to use as a state")
            continue
        result.examples.append(
            LabelledExample(
                decision="watch_healthy",
                state=excerpt,
                question=healthy,
                label="true",
                join_key=f"{job_dir.name}:{event.get('event_id', '')}",
                source=str(job_dir / "events.jsonl"),
                baselines=(("dspy_signal", str(event.get("signal") or "")),),
            )
        )
        result.warn(
            "healthy examples come from watchdog_ok excerpts, which Watch caps at 500 "
            "characters, while unhealthy examples come from bundles of up to 4000. The "
            "two classes differ in length before they differ in content, so accuracy on "
            "watch_healthy from this corpus is not evidence about reading a log."
        )


def _recorded_baselines(job_dir: Path) -> tuple[tuple[str, str], ...]:
    """What the existing system answered for this job, from its own artifacts.

    Two baselines, and the second is the bar Phase 1 actually has to clear:

    * ``regex_signal`` -- the verdict ``sanitizer_assessment()`` produced, which
      is recoverable because it is the only path that writes ``confidence: 1.0``
      with a machine-readable sanitizer line as its evidence. Absent for a job
      whose log was not a sanitizer summary, which is most of them, and absence
      is recorded as such rather than as a wrong answer.
    * ``dspy_signal`` -- the slug the ReAct assessment emitted on the alerting
      poll. Read from the events file rather than recomputed, so it is the
      answer the model actually gave against the prompt as it was that day.
    """
    baselines: dict[str, str] = {}
    for event in iter_json_lines(job_dir / "events.jsonl"):
        signal = str(event.get("signal") or "")
        if not signal or signal in _NON_VERDICT_SIGNALS:
            continue
        if event.get("event_type") == "watchdog_alert":
            baselines["dspy_signal"] = signal
            # The self-reported confidence travels with the answer so the
            # baseline can be scored on Brier as well as on agreement. It is the
            # number Phase 1 is testing the replacement of -- ``should_alert``
            # compares it against 0.70 -- so measuring how calibrated it turns
            # out to be is half the argument for replacing it.
            baselines["dspy_confidence"] = str(event.get("confidence", ""))
        if float(event.get("confidence") or 0.0) == 1.0 and "[sanitizer]" in str(
            event.get("excerpt") or ""
        ):
            # ``sanitizer_assessment()`` is the only tier that pairs confidence
            # 1.0 with a quoted ``[sanitizer] ...`` line, so that pair is how its
            # verdict is told apart from the model's after the fact. Recovered
            # rather than recomputed because the regex has been edited since
            # these runs and the baseline has to be the answer it gave then.
            baselines["regex_signal"] = signal
    manifest_signal = _manifest_signal(job_dir)
    if manifest_signal and "dspy_signal" not in baselines:
        # The bundle's manifest records the signal Watch alerted on even when the
        # events file has been rotated away, so it is the fallback rather than
        # the primary: it says which slug, and not which tier produced it.
        baselines["dspy_signal"] = manifest_signal
    return tuple(sorted(baselines.items()))


def _manifest_signal(job_dir: Path) -> str:
    """``metadata.watch_signal`` from the bundle manifest, or ""."""
    path = job_dir / "bundle" / "manifest.yaml"
    if not path.is_file():
        return ""
    try:
        import yaml

        manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return ""
    if not isinstance(manifest, dict):
        return ""
    return str((manifest.get("metadata") or {}).get("watch_signal") or "")


def _warn_about_the_shape_of_it(
    result: BuildResult, offered: tuple[str, ...]
) -> None:
    """Say what the label distribution is, before anyone scores against it.

    *offered* is the slug set the signal question actually put on the table, taken
    from the question rather than from a list beside it -- see
    :func:`watch_signals`.
    """
    signals = [e.label for e in result.examples if e.decision == "watch_signal"]
    if signals:
        collapsed = sum(1 for label in signals if label == "WATCH_UNKNOWN_ERROR")
        if collapsed:
            result.warn(
                f"{collapsed} of {len(signals)} watch_signal labels are WATCH_UNKNOWN_ERROR, "
                "which four Autopsy categories collapse onto because Watch has no slug for "
                "them. Read macro accuracy on this, never plain accuracy."
            )
        unreachable = sorted(set(offered) - set(CATEGORY_TO_SIGNAL.values()))
        if unreachable:
            result.warn(
                f"no Autopsy category maps to {', '.join(unreachable)}, so those slugs can "
                "never appear as a label however many jobs are scanned. A classifier trained "
                "here cannot learn to emit them."
            )
    healthy = [e for e in result.examples if e.decision == "watch_healthy"]
    positives = sum(1 for e in healthy if e.label == "true")
    if healthy and not positives:
        result.warn(
            "every watch_healthy example is labelled false, because write_bundle only runs "
            "after an alert and no healthy delta is persisted at full length anywhere. The "
            "clean-gate cannot be measured from these artifacts alone."
        )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def iter_job_dirs(jobs_root: Path) -> Iterator[Path]:
    """Every job directory under *jobs_root*. Shared with the log-finder builder."""
    if not jobs_root.is_dir():
        return
    for job_dir in sorted(p for p in jobs_root.iterdir() if p.is_dir()):
        if (job_dir / "job.json").is_file():
            yield job_dir


__all__ = [
    "CATEGORY_TO_SIGNAL",
    "UNLABELLABLE_CATEGORIES",
    "build_watch_corpus",
    "iter_job_dirs",
    "watch_questions",
    "watch_signals",
]
