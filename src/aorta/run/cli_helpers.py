"""Shared helpers between the ``aorta run`` CLI and library callers.

These keep ``cli/run.py`` a thin shell, per the B1 spec
(GitHub issue #148): *"the Click handler in `cli/run.py` is under
~30 lines and contains no `for trial in range(...)` loop"*.

Anything that's actually orchestration -- CSV parsing of CLI strings,
turning a ``list[TrialResult]`` into pass/fail counts -- lives here so
that:

* the Click handler stays at "parse args -> build RunRequest ->
  run_trials -> map to exit code", and
* B2's triage matrix runner (and any other programmatic caller) can
  reuse the exact same parsers/aggregators without going through
  Click.

Validation that the library API needs to enforce regardless of caller
(e.g. POSIX env-var name shape on ``extra_env`` keys) lives in
``aorta.run.dispatcher.run_trials`` so library callers that bypass
this module still get checked.
"""

import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass

from aorta.run.results import TrialResult


def configure_verbose_logging(verbose_count: int) -> None:
    """Wire a stderr StreamHandler onto the ``aorta`` logger when -v is set.

    Both ``aorta run`` and ``aorta triage run`` are silent by default --
    the ``aorta.*`` loggers exist but have no handlers, so INFO-level
    progress calls are dropped. Without this, a long matrix run prints
    nothing until the final ``Wrote matrix to ...`` line, leaving
    operators with no signal that anything is happening.

    Scope: only the ``aorta.*`` logger hierarchy. Workloads registered
    via the ``aorta.workloads`` entry-point group from sibling packages
    (e.g. ``<your_pkg>.workloads.<name>``) live OUTSIDE that hierarchy,
    so their logging is unaffected -- a separate handler on those
    packages' loggers (or the root logger) would be needed.

    ``verbose_count`` follows Click's ``count=True`` convention:

    * ``0``: tear-down / no-op. If a prior call in the same process
      installed verbosity, the marked handler is removed and the logger's
      ``level`` and ``propagate`` are restored to the values observed at
      install time (NOT unconditionally reset to NOTSET / True). An
      embedding application that had configured ``aorta`` itself before
      invoking the CLI gets its state back. If no prior install exists,
      the call is a true no-op.
    * ``1`` (``-v``): INFO -- per-trial / per-cell progress.
    * ``>=2`` (``-vv``): DEBUG -- aorta platform internals.

    Output goes to **stderr** so stdout stays clean for the existing
    ``click.echo`` lines (final pass/fail summary, "to rerun" hint, etc.)
    that callers may pipe into a parser. ``aorta_logger.propagate`` is
    set to ``False`` while verbose is active so a configured root logger
    (e.g. via ``logging.basicConfig()`` in an embedding app) does not
    also emit each record -- otherwise every progress line would print
    twice. Restored on teardown.

    Symmetric and idempotent: ``configure_verbose_logging(N)`` followed
    by ``configure_verbose_logging(0)`` returns the logger to its exact
    original state. CliRunner-based tests can invoke commands with and
    without ``-v`` in any order without leaking state.
    """
    aorta_logger = logging.getLogger("aorta")

    if verbose_count <= 0:
        # Tear down any handler we previously installed and restore the
        # exact (level, propagate) we observed at install time. Embedding
        # applications that had configured the aorta logger themselves
        # (e.g. ``logging.getLogger("aorta").setLevel(WARNING)``) get
        # their state back instead of an unconditional NOTSET reset.
        for h in [h for h in aorta_logger.handlers if getattr(h, "_aorta_verbose", False)]:
            aorta_logger.setLevel(getattr(h, "_aorta_prior_level", logging.NOTSET))
            aorta_logger.propagate = getattr(h, "_aorta_prior_propagate", True)
            aorta_logger.removeHandler(h)
        return

    level = logging.DEBUG if verbose_count >= 2 else logging.INFO
    existing = next(
        (h for h in aorta_logger.handlers if getattr(h, "_aorta_verbose", False)),
        None,
    )
    if existing is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
        # Snapshot prior (level, propagate) so teardown can restore exactly
        # what the embedding app configured. Capture BEFORE we mutate, and
        # store on the handler so the restore path is self-contained.
        handler._aorta_verbose = True  # type: ignore[attr-defined]
        handler._aorta_prior_level = aorta_logger.level  # type: ignore[attr-defined]
        handler._aorta_prior_propagate = aorta_logger.propagate  # type: ignore[attr-defined]
        aorta_logger.addHandler(handler)
        # Disable propagation so a configured root logger (e.g. via
        # ``logging.basicConfig()`` in an embedding app) does not also
        # emit the same record -- otherwise -v would print every progress
        # line twice. Restored on teardown.
        aorta_logger.propagate = False
    aorta_logger.setLevel(level)


def parse_csv(value: str) -> tuple[str, ...]:
    """Split a comma-separated CLI string into stripped, non-empty tokens.

    Empty input or input that's only commas/whitespace yields ``()``.
    """
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_mitigations(value: str) -> tuple[str, ...]:
    """Parse the ``--mitigations`` CSV.  Empty input means baseline.

    The default ``("none",)`` matches the documented CLI default; an
    empty string from a programmatic caller (e.g. an empty
    environment variable) gets the same baseline rather than silently
    becoming "no mitigations at all", which would diverge from the
    CLI surface.
    """
    parts = parse_csv(value)
    return parts if parts else ("none",)


def parse_extra_env(value: str) -> dict[str, str]:
    """Parse a ``--extra-env`` CSV like ``"KEY=VAL,KEY2=VAL2"``.

    Format errors (missing ``=``, empty key) raise ``ValueError`` --
    the CLI converts these to ``ClickException`` so the user sees a
    clean error instead of a Python traceback.

    Key *shape* (POSIX env-var name pattern) is intentionally NOT
    checked here: ``run_trials`` re-checks ``extra_env`` keys at the
    library entry-point so callers that bypass this parser (B2,
    direct programmatic users) can't slip an invalid key past it.
    """
    out: dict[str, str] = {}
    for pair in value.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid extra-env format: '{pair}'. Expected KEY=VALUE.")
        key, val = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid extra-env entry '{pair}': key is empty.")
        out[key] = val.strip()
    return out


@dataclass(frozen=True)
class RunSummary:
    """Pass/fail aggregation over a ``list[TrialResult]``.

    Surfaced separately from ``run_trials`` so both the CLI and
    programmatic callers (B2 cell-status reporting, future
    dashboards) compute pass/fail the same way and don't drift in
    "what counts as a failure".
    """

    total: int
    passed: int
    failed: int
    failed_trial_ids: tuple[str, ...]


def summarize_results(results: Iterable[TrialResult]) -> RunSummary:
    """Bucket ``TrialResult``s into passed / failed counts.

    A trial is "passed" iff ``exit_status == "ok"``.  Anything else
    (``workload_failed`` or ``infrastructure_failed``) is a failure
    from the runner's perspective; the distinction between those two
    is preserved on each individual ``TrialResult`` for callers that
    need it.
    """
    results_tuple = tuple(results)
    failed_ids = tuple(r.trial_id for r in results_tuple if r.exit_status != "ok")
    return RunSummary(
        total=len(results_tuple),
        passed=len(results_tuple) - len(failed_ids),
        failed=len(failed_ids),
        failed_trial_ids=failed_ids,
    )


__all__ = [
    "RunSummary",
    "configure_verbose_logging",
    "parse_csv",
    "parse_extra_env",
    "parse_mitigations",
    "summarize_results",
]
