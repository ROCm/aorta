"""Typed exceptions for ``aorta bundle`` (issue #196).

Kept in their own module so :mod:`aorta.bundle.writer`,
:mod:`aorta.bundle.cli`, and downstream Phase 3 (#188) integration
code can import them without pulling in the writer/tarball
machinery. Each subclass carries the operator-visible context
(path, ticket, etc.) so the CLI shim can render a useful
``ClickException`` message without re-deriving it.
"""

from __future__ import annotations

from pathlib import Path


class BundleError(Exception):
    """Base class for every error :mod:`aorta.bundle` raises.

    Catching :class:`BundleError` in the CLI shim covers all the
    documented failure modes (no ticket, missing run dir, empty run
    dir, operator-aborted review). Concrete subclasses each carry
    the structured context the message was rendered from so test
    assertions can match on the field rather than on substring
    soup.
    """


class RunDirNotFoundError(BundleError):
    """Raised when ``<run-dir>`` does not exist or is not a directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        super().__init__(
            f"aorta bundle: run dir {run_dir} does not exist or is not "
            "a directory. Pass the per-ticket leaf produced by 'aorta "
            "probe' (e.g. <probe-output>/<ticket>/)."
        )


class NoTicketError(BundleError):
    """Raised when the run dir resolves to ``_no_ticket_`` and no ``--ticket`` was passed.

    Mirrors rubric §3.B FR 3.1 for issue #188 Phase 3: a bundle with
    no ticket has nowhere to land downstream, so refuse early
    instead of writing a ``_no_ticket_-<timestamp>.tar.gz`` that
    nobody can route.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        super().__init__(
            f"aorta bundle: run dir {run_dir} resolves to '_no_ticket_'. "
            "Pass --ticket TICKET (or re-run 'aorta probe' with "
            "--ticket TICKET so the source tree carries one). Bundles "
            "without a real ticket have no routing target downstream."
        )


class EmptyRunDirError(BundleError):
    """Raised when the run dir has no ``trial_*/result.json`` artifacts.

    A directory with `aorta probe` shape but zero completed trials
    is almost always operator error -- pointing at the wrong path,
    forgetting the ``--ticket`` segment, or running before any cell
    finished. Refuse before writing an empty tarball.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        super().__init__(
            f"aorta bundle: run dir {run_dir} contains no "
            "'trial_*/result.json' artifacts. Did you forget to "
            "include the per-ticket segment, or run before any "
            "probe trial finished?"
        )


class BundleAbortedError(BundleError):
    """Raised when ``--review`` was passed and the operator answered ``n``.

    Carries the manifest summary that was shown so a CLI test can
    assert the operator was given the documented context before the
    abort.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        super().__init__(
            f"aorta bundle: review-pause aborted by operator (run dir "
            f"{run_dir}). No tarball was written."
        )
