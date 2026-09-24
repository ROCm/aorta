"""The log finder's corpus: a directory listing, labelled by what Watch read.

Tier 3 of ``LogFinder.find`` spends an LLM call deciding which files in a job
directory are logs. Its prompt rules are already mechanical -- prefer a high
mtime, prefer ``log`` / ``stderr`` / ``out`` in the name, skip checkpoints, skip
anything under 100 bytes -- and tier 2 already implements most of them.

**So the labels here come from observed usefulness, not from the rules.** That
is the whole design constraint of this builder. A corpus labelled by the prompt's
own rules would teach the classifier the heuristic that is already in
``_scan_by_extension``, at which case replacing tier 3 buys a forward pass and
nothing else. A file is labelled useful when Watch is observed to have read bytes
out of it: it appears in the job record's ``watch_files`` *and* its byte cursor in
``watch_cursors.json`` advanced past zero. Everything else the listing contains
is labelled not-useful.

**The question is tier 3's own, imported rather than phrased here.** This module
used to carry its own wording of it, and used to attach one shared question
object to every file in a directory with the filename only in the join key. Both
halves of that were wrong and neither failed loudly. A fit made against a wording
the tier does not ask is calibrated against nothing while still returning a
probability; and identical states carrying identical questions with opposing
labels is a contradictory corpus rather than a hard one, which ``group_top1``
cannot rank because every row in the group scores the same. See
:func:`_useful_question`.

"Read bytes" is weaker than "read something that mattered", and that is a
limitation of the artifacts rather than a choice. Watch's events carry a
``source`` field, but it is always ``watch_files[0]`` regardless of which file
the assessed chunk came from, so no artifact attributes an alert to the file that
produced it. Strengthening the label means either recording per-file attribution
in the poll loop or labelling a sample by hand.

**A second finding, and it bounds how much of this corpus can exist.** Discovery
is only reached for a job whose record declares no ``log_path``: ``poll.py``
prefers a declared path outright, and ``aorta.cia.triage`` always sets one. So on
a machine whose jobs all came through triage, tier 3 has never run, and the
listings here are directories the LLM tier was never asked about. The builder
counts those separately and says so, because a corpus of examples from a code
path that never executes measures a decision nobody is making.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - the annotation must not cost an import
    from aorta.local_classifier.predictor import Noul

from aorta.local_classifier.corpus.schema import BuildResult, CorpusError, LabelledExample
from aorta.local_classifier.corpus.watch import iter_job_dirs

#: Below this many entries a directory is not a choice, so there is no decision
#: to label.
#:
#: A weak proxy for tier 3's own gate, deliberately. That gate is "the extension
#: scan returned nothing, or more than three files", and the extension set comes
#: from ``watch_config.yaml`` as it stood when the job ran, which no artifact
#: records. Reconstructing it from today's config would silently include or
#: exclude directories on the strength of a setting that has since changed, so
#: this errs toward including a directory and letting the reader see the count.
_MIN_LISTING_ENTRIES = 2


def build_log_finder_corpus(
    jobs_root: str | Path, *, listing: Callable[[Path], str] | None = None
) -> BuildResult:
    """Build the log-finder corpus from the job directories under *jobs_root*.

    One example per listed file. All of them carry the same state -- the listing
    the real tier 3 is shown -- and each carries its own question, naming its own
    file. That is exactly the shape Track C replaces the LLM call with: N
    independent nouls over one state, which is one forward pass, over the entries
    that are actually there, so no path can be named that the listing does not
    contain. It is also the only shape that can be scored: a directory's rows
    share a ``group``, and ``group_top1`` ranks them against each other, which
    needs them to be distinguishable questions.

    Args:
        jobs_root: Where the job directories are.
        listing: Injection point for tests; defaults to tier 3's own
            ``_dir_listing``. Same arrangement, and for the same reason, as
            ``evaluate(search=...)`` in ``aorta/chat/rag/eval.py``: it lets a
            test pin the labelling against a listing it wrote rather than
            against whatever is on disk.

            It no longer makes the builder runnable without the ``[cia]``
            extra, and that is deliberate rather than an oversight. The question
            comes from the tier now, so a build reaches ``aorta.cia.watch``
            whichever listing it is given -- and the alternative, a question
            phrased here so the builder could stand alone, is the defect this
            arrangement exists to remove. ``tests/cia/conftest.py`` already skips
            this whole directory when the extra is absent.
    """
    root = Path(jobs_root).expanduser()
    result = BuildResult()
    if not root.is_dir():
        result.warn(f"no jobs root at {root}, so there are no listings to label")
        return result
    render = listing or _dir_listing

    declared = 0
    for job_dir in iter_job_dirs(root):
        result.scanned += 1
        record = _read_job(job_dir)
        if record is None:
            result.skip("job.json unreadable")
            continue
        if _log_path_was_declared(record):
            # Counted rather than skipped silently: this is the finding above,
            # and its size is the interesting part.
            declared += 1
        _add_from_job(job_dir, record, result, render)

    if declared:
        result.warn(
            f"{declared} of {result.scanned} jobs declare a log_path, which poll.py prefers "
            "outright, so LogFinder's LLM tier was never consulted for them. Their listings "
            "are labelled from what Watch read, but the decision being modelled is one this "
            "machine's jobs do not reach."
        )
    _warn_about_the_shape_of_it(result)
    return result


def _add_from_job(
    job_dir: Path,
    record: dict,
    result: BuildResult,
    render: Callable[[Path], str],
) -> None:
    """Emit one example per listed file, or record why none were emitted."""
    listing = render(job_dir)
    entries = _listed_paths(listing)
    if len(entries) < _MIN_LISTING_ENTRIES:
        result.skip("listing holds fewer than two files, so the choice is not ambiguous")
        return

    read = _files_watch_read(job_dir, record)
    if not read:
        result.skip("Watch read no bytes from any file, so nothing is observed to be useful")
        return
    useful = {path for path in entries if path in read}
    if not useful:
        # Watch read files that the listing no longer contains -- a log outside
        # the job directory, or one rotated away. Labelling every listed file
        # not-useful would be a corpus of pure negatives for this directory.
        result.skip("the files Watch read are not in the current listing")
        return

    for path in entries:
        result.examples.append(
            LabelledExample(
                decision="log_finder_useful",
                state=listing,
                # One question per file, naming that file, because that is what
                # the tier asks. Every row here used to carry one shared question
                # object with the filename only in the join key -- the degenerate
                # shape the tier's own docstring rules out, since N questions
                # about one state are told apart by their text alone. Identical
                # states and identical questions with opposing labels is not a
                # hard corpus, it is a contradictory one, and ``group_top1``
                # could not rank it because every row scored the same.
                question=_useful_question(job_dir, path),
                label="true" if path in useful else "false",
                join_key=f"{job_dir.name}:{path}",
                # One directory is one decision, however many files it holds. That
                # is what lets ``group_top1`` score the ranking Track C actually
                # performs -- take ``max_files`` off the front of a sorted listing
                # -- rather than accuracy over files, which rewards answering false
                # to a directory of forty and watching none of them.
                group=job_dir.name,
                source=str(job_dir),
            )
        )


def _dir_listing(job_dir: Path) -> str:
    """The listing tier 3 is shown, produced by tier 3's own routine.

    Reused rather than reimplemented, for the reason ``_within_job`` gives about
    containment checks: two copies of this would be how the corpus comes to
    describe a listing the model never sees, and the differences would be exactly
    the fields the decision turns on -- the size and the mtime.

    Imported here rather than at module scope because
    ``aorta.cia.watch.log_finder`` imports dspy at *its* module scope, and
    reaching it eagerly would make the whole ``aorta.local_classifier`` package need the
    ``[cia]`` extra for the two builders that have no use for it.

    The listing is regenerated now, not at discovery time, so the sizes and
    mtimes are as of this build. Nothing persists the listing tier 3 was actually
    shown, so this is as close as the artifacts allow.
    """
    try:
        from aorta.cia.watch.log_finder import _dir_listing as render
    except ImportError as exc:
        raise CorpusError(
            "the log-finder corpus reads its listings through LogFinder's own "
            "_dir_listing, which needs the agents' extra: "
            "pip install 'amd-aorta[cia]'. Reimplementing the listing here would "
            "let the corpus describe a directory the model never sees.\n"
            f"(missing: {exc.name or 'dspy'})"
        ) from exc
    return render(job_dir)


def _useful_question(job_dir: Path, listed: str) -> Noul:
    """Tier 3's own question about one listed file, asked the way tier 3 asks it.

    Imported rather than phrased here, and that direction is the whole point.
    This module used to carry its own wording of the same question, so a
    temperature fitted on this corpus would have been fitted against a question
    the tier never asks. Nothing about that fails loudly: it answers slightly
    worse, for a reason nobody would go looking for. Track B found the same
    defect in the proposer and Phase 5 found it in chat, and the resolution is
    the same one -- the question belongs to the module that owns the decision,
    and the builder imports it.

    The label is imported too. It is part of the question text, so deriving it
    here would reintroduce the drift one level down: a corpus naming the file
    ``/jobs/cia-1/train.log`` and a tier naming it ``train.log`` are labelling
    two different questions however faithfully the template was shared.

    Imported inside the function for the reason ``_dir_listing`` is, and it is
    about the extra rather than the weight: ``aorta.cia.watch.log_finder``
    imports dspy at its own module scope, and reaching it eagerly would make the
    whole ``aorta.local_classifier`` package need the ``[cia]`` extra for the two builders
    that have no use for it.
    """
    try:
        from aorta.cia.watch.log_finder import listing_label, useful_question
    except ImportError as exc:
        raise CorpusError(
            "the log-finder corpus asks tier 3's own question, which lives with "
            "the tier and needs the agents' extra: "
            "pip install 'amd-aorta[cia]'. Rephrasing it here would label a "
            "corpus against a question the tier does not ask.\n"
            f"(missing: {exc.name or 'dspy'})"
        ) from exc
    return useful_question(listing_label(job_dir, listed))


def _listed_paths(listing: str) -> list[str]:
    """The paths in a ``_dir_listing`` block, in the order it emitted them.

    Parsed off the rendered text rather than re-walked, so the candidate set is
    exactly the one the question offers. A candidate set built by a second walk
    could contain a path the listing does not, which is the one property this
    whole track exists to remove.
    """
    paths: list[str] = []
    for line in listing.splitlines():
        path, separator, _ = line.partition("  size=")
        if separator and path and path not in paths:
            paths.append(path)
    return paths


def _files_watch_read(job_dir: Path, record: dict) -> set[str]:
    """Files Watch both selected and read bytes from.

    Both conditions, because either alone is wrong. ``watch_files`` alone
    includes a file Watch resolved and then found empty for the life of the job,
    which is not evidence that it was worth watching. A non-zero cursor alone
    would include a file whose cursor was left behind by an earlier run over a
    path that has since been re-resolved elsewhere.
    """
    selected = {str(path) for path in record.get("watch_files") or []}
    cursors = _read_json(job_dir / "watch_cursors.json") or {}
    advanced = set()
    for path, offset in cursors.items():
        try:
            if int(offset) > 0:
                advanced.add(str(path))
        except (TypeError, ValueError):
            continue
    return selected & advanced


def _log_path_was_declared(record: dict) -> bool:
    """Whether the job named its own log, which short-circuits discovery."""
    return bool(str(record.get("log_path") or "").strip())


def _read_job(job_dir: Path) -> dict | None:
    return _read_json(job_dir / "job.json")


def _read_json(path: Path) -> dict | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _warn_about_the_shape_of_it(result: BuildResult) -> None:
    labels = [e.label for e in result.examples]
    if not labels:
        return
    positives = labels.count("true")
    result.warn(
        f"{positives} of {len(labels)} listed files are labelled useful. A per-file noul over "
        "a listing is imbalanced by nature -- a job directory holds far more files than logs "
        "-- so this decision is scored on group_top1, whether the top-ranked file is one "
        "Watch read, and not on accuracy over files."
    )
    if positives == len(labels):
        result.warn(
            "every listed file is labelled useful, so there are no negatives to learn from. "
            "These listings hold only the files Watch was already reading."
        )


__all__ = ["build_log_finder_corpus"]
