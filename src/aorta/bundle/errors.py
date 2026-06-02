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


class UnsafeSymlinkError(BundleError):
    """Raised when a file under ``<run-dir>`` resolves outside the tree.

    ``aorta bundle`` produces a *shareable* artifact, so it must not
    silently pull bytes from outside the run dir. A symlink (or a
    symlinked parent component) whose resolved target escapes
    ``run_dir`` would dereference into an unrelated local file or a
    mounted-share path and bundle its contents -- breaking the trust
    boundary for the command. We refuse such entries instead of
    following them. In-tree symlinks (target stays inside
    ``run_dir``) are still followed.

    Carries the offending ``path`` and its resolved ``target`` so the
    CLI shim can name both in the operator-visible message and tests
    can assert on the fields.
    """

    def __init__(self, run_dir: Path, path: Path, target: Path) -> None:
        self.run_dir = run_dir
        self.path = path
        self.target = target
        super().__init__(
            f"aorta bundle: refusing to follow symlink {path} -> {target}: "
            f"its target is outside the run dir {run_dir}. Bundles must "
            "stay within the source tree to remain shareable. Remove the "
            "symlink or copy the real file into the run dir before bundling."
        )


class BundleAbortedError(BundleError):
    """Raised when ``--review`` was passed and the operator answered ``n``.

    Carries the source ``run_dir`` so the CLI can render a "no new
    tarball was written; run dir untouched" message. The rendered
    review summary is printed to ``stdout`` by the review callback
    BEFORE the prompt -- it is not stored on the exception (CLI tests
    assert against captured stdout, not against this class).
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        super().__init__(
            f"aorta bundle: review-pause aborted by operator (run dir "
            f"{run_dir}). No new tarball was written; any existing file "
            "at the output path was left untouched."
        )


class BundleIOError(BundleError):
    """Raised when a filesystem error escapes the staging / tarball pipeline.

    Wraps ``OSError`` from the redactor's ``scrub_file`` (and the
    underlying ``shutil.copyfile``), from ``tarfile.add`` /
    ``tarfile.open``, and from manifest writes so the CLI shim
    surfaces a clean ``ClickException`` instead of a Python
    traceback. The ``Redactor`` ABC docstring documents this wrap;
    the writer is the place that performs it.

    Carries the ``run_dir`` and the original ``OSError`` so a future
    test or ops tool can grade whether the failure was a permissions
    problem (``PermissionError``), out of disk (``ENOSPC``), or
    something else without parsing the rendered message.
    """

    def __init__(self, run_dir: Path, cause: OSError) -> None:
        self.run_dir = run_dir
        self.cause = cause
        super().__init__(
            f"aorta bundle: filesystem error while bundling {run_dir}: "
            f"{type(cause).__name__}: {cause}. No new tarball was written; "
            "any existing file at the output path was left untouched (the "
            "atomic write only replaces it on success)."
        )


class RedactionError(BundleError):
    """Raised when a redactor cannot guarantee an artifact was scrubbed.

    A :class:`~aorta.bundle.redactor.Redactor` that parses structured
    artifacts (``result.json``, ``host_env.json``) must fail *closed*
    when the artifact is corrupted/truncated: a raw
    ``json.JSONDecodeError`` would otherwise escape staging as an
    unhandled traceback, and -- worse -- partial parsing could let an
    unredacted artifact through. Failing with a typed ``BundleError``
    means the CLI shim renders a clean message and no bundle is written
    when scrubbing cannot be guaranteed.

    Carries the offending ``path`` and the underlying ``cause`` so the
    operator can name the bad file and tests can assert on the fields
    instead of substring-matching the rendered message.
    """

    def __init__(self, path: Path, cause: Exception) -> None:
        self.path = path
        self.cause = cause
        super().__init__(
            f"aorta bundle: cannot redact {path}: {type(cause).__name__}: "
            f"{cause}. Refusing to write a bundle whose scrubbing is not "
            "guaranteed -- fix or remove the corrupted artifact and re-run."
        )
