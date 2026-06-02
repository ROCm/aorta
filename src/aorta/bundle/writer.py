"""Bundle staging, manifest, and tarball writer for ``aorta bundle``.

Three layers, each independently testable:

1. :func:`resolve_ticket` -- pure function that maps ``(run_dir,
   --ticket)`` to the bundle ticket. Refuses ``_no_ticket_``.
2. :func:`stage_run_dir` -- copies every file under ``run_dir``
   through the redactor into a staging tree under
   ``<staging>/<bundle_name>/``, producing a :class:`Manifest`.
3. :func:`write_tarball` -- packs ``<staging>/<bundle_name>/`` into
   ``<output>.tar.gz`` (gzip-compressed, deterministic top-level
   directory).

:func:`bundle_run_dir` is the CLI's single entry point; it stitches
the three layers together with a ``TemporaryDirectory`` for
staging and an optional ``review_callback`` so the CLI can surface
the manifest before the tarball is written. The callback design
keeps the writer free of Click imports -- the CLI shim wires
:func:`click.confirm` in via the callback.

The redactor is :class:`IdentityRedactor` until Phase 3 of issue
#188 ships ``aorta.probe.redaction``. The function signature is
ready for the swap (``redactor=`` keyword) so #188 can wire its
implementation in without re-shaping any contract here.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import tarfile
import tempfile
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

from aorta.bundle._slug import NO_TICKET_SLUG, safe_slug
from aorta.bundle.errors import (
    BundleAbortedError,
    BundleIOError,
    EmptyRunDirError,
    NoTicketError,
    RunDirNotFoundError,
    UnsafeSymlinkError,
)
from aorta.bundle.manifest import MANIFEST_FILENAME, FileRecord, Manifest, _to_utc
from aorta.bundle.redactor import IdentityRedactor, Redactor

log = logging.getLogger(__name__)

#: Glob the writer uses to confirm a run dir actually came from
#: ``aorta probe`` (i.e. has at least one completed trial). The
#: pattern matches the artifact-tree contract documented at
#: ``docs/probe-188/usage.md`` (cell/trial_N/result.json).
_TRIAL_RESULT_GLOB = "*/trial_*/result.json"


def _aorta_version() -> str:
    """Return the installed ``aorta`` package version (best-effort)."""
    try:
        return _pkg_version("aorta")
    except PackageNotFoundError:
        return "unknown"
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _bundle_timestamp(now: _dt.datetime | None = None) -> str:
    """Filesystem-safe UTC timestamp for the bundle's directory + filename.

    ISO-8601 with colons replaced by dashes (Windows-friendly), to
    millisecond precision so two bundles written in quick succession
    (e.g. back-to-back CI invocations) do not collide on the same
    name -- a collision would also fail the staging
    ``mkdir(exist_ok=False)`` and abort the second run, so this is
    a correctness concern, not just a cosmetic one.

    The format is ``%Y-%m-%dT%H-%M-%S-<ms>`` with ``<ms>`` zero-padded
    to three digits (``microsecond // 1000``). Plain ``%f`` would give
    six digits, which is more entropy than we need and clutters the
    filename.

    A naive or non-UTC ``now`` is normalised through
    :func:`aorta.bundle.manifest._to_utc` so the embedded clock
    matches the ``Z`` convention in the manifest's ``created_at``.
    Test injection of a deterministic naive datetime previously
    rendered the caller's local clock as UTC silently -- the same
    trap Copilot caught on the manifest path.
    """
    now = _to_utc(now or _dt.datetime.now(_dt.timezone.utc))
    return f"{now.strftime('%Y-%m-%dT%H-%M-%S')}-{now.microsecond // 1000:03d}"


def resolve_ticket(run_dir: Path, ticket_flag: str | None) -> str:
    """Resolve the bundle's ticket per ``docs/probe-188/bundle.md``.

    Order of precedence:

    1. ``--ticket TICKET`` flag, if non-empty. The ticket is
       cross-checked against the basename of ``run_dir``; a
       mismatch is logged at WARNING level but does not refuse.
       Operators move artifact trees between machines (e.g. NFS
       handoff) and the basename is not authoritative when the
       operator overrides it.
    2. ``run_dir`` basename, if not ``_no_ticket_``.
    3. Otherwise, raise :class:`NoTicketError` -- the issue #196
       acceptance criterion 2 ("refuses without --ticket").

    Whitespace-only ``ticket_flag`` is treated as missing so
    ``--ticket ""`` does not silently produce an unsluggable
    bundle. The resolved value is returned verbatim (NOT slugged):
    the manifest records the operator-supplied ticket, while
    :func:`safe_slug` is applied separately for filesystem
    components.
    """
    basename = run_dir.name
    if ticket_flag is not None and ticket_flag.strip():
        ticket = ticket_flag.strip()
        if basename != safe_slug(ticket) and basename != NO_TICKET_SLUG:
            log.warning(
                "aorta bundle: --ticket %r does not match run-dir "
                "basename %r; proceeding with the flag value (operator "
                "override). The manifest will record %r.",
                ticket,
                basename,
                ticket,
            )
        if basename == NO_TICKET_SLUG:
            log.warning(
                "aorta bundle: run dir basename is '%s' but --ticket %r "
                "was passed; using the flag value. Re-run 'aorta probe' "
                "with --ticket %r so the source tree carries one too.",
                NO_TICKET_SLUG,
                ticket,
                ticket,
            )
        return ticket
    if basename == NO_TICKET_SLUG:
        raise NoTicketError(run_dir=run_dir)
    return basename


def _validate_run_dir(run_dir: Path) -> None:
    """Reject non-existent paths, non-directories, and empty trees.

    The writer treats an existing directory with at least one
    ``trial_*/result.json`` artifact as a valid probe run dir. The
    ``_TRIAL_RESULT_GLOB`` matches the documented per-cell layout
    in ``docs/probe-188/usage.md``; legitimate probe outputs always
    have at least one such file (the dispatcher writes
    ``result.json`` even on the exec-failed and timed-out paths --
    see ``src/aorta/workloads/_subprocess.py::run`` for the
    artifact-tree contract).
    """
    if not run_dir.exists() or not run_dir.is_dir():
        raise RunDirNotFoundError(run_dir=run_dir)
    if not any(run_dir.glob(_TRIAL_RESULT_GLOB)):
        raise EmptyRunDirError(run_dir=run_dir)


def _iter_source_files(run_dir: Path) -> list[Path]:
    """Walk ``run_dir`` and return every regular file in deterministic order.

    Order is sorted alphabetic on the relative POSIX path. That
    matters because:

    * Manifests round-trip byte-equivalently across hosts (no
      ``os.walk`` insertion-order surprises) -- two bundles of the
      same source tree on different hosts produce identical
      manifests modulo the timestamp / source path fields.
    * Tarball entry order is reproducible at the entry-list level
      (``tar -tzf <bundle>`` lists files in the same order
      everywhere). :func:`_scrub_tarinfo` additionally zeroes
      uid/gid, clears uname/gname, and pins mtime in the tar headers
      so the archive neither leaks the operator's workstation
      identity nor varies on those fields. The resulting **bytes**
      are still NOT bit-identical across hosts -- the outer gzip
      wrapper embeds its own header timestamp that ``tarfile`` does
      not expose -- so a consumer needing cryptographic equality
      should hash the manifest, not the tarball.

    Symlinks are followed via ``Path.is_file`` semantics ONLY when
    their resolved target stays inside ``run_dir``. A symlink (or a
    symlinked parent component) whose target escapes the tree raises
    :class:`UnsafeSymlinkError`: ``aorta bundle`` ships a shareable
    artifact, so it must not dereference a link pointing at an
    unrelated local file or a mounted-share path and pull those bytes
    in (the trust-boundary hole flagged in PR #199 review). In-tree
    symlinks are still followed so legitimate intra-run links work.

    Skips (TOP-LEVEL ONLY; a nested file with the same basename in a
    cell / trial directory is a legitimate artifact and IS bundled):

    * The ``flat_resume`` lockfile (``./.aorta-probe.lock``) -- it is
      a transient runtime artifact, not a deliverable.
    * ``./manifest.json`` left over from a prior bundle invocation
      that wrote into the source tree (defensive; the writer never
      does this today, but a downstream copy could).

    The skip set is matched against the file's relative POSIX path,
    not its basename -- a workload that emits ``cell/trial_0/manifest.json``
    or ``cell/.aorta-probe.lock`` (e.g. a script that wraps its own
    bundler-style output) gets those files bundled normally.
    """
    skipped_rel_paths = frozenset({".aorta-probe.lock", MANIFEST_FILENAME})
    root = run_dir.resolve()
    files: list[Path] = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        # Resolve the full path (dereferences symlinks at any
        # component) and refuse anything that lands outside the run
        # dir. A regular file always resolves within ``root``; only a
        # symlink escaping the tree trips this guard.
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise UnsafeSymlinkError(run_dir=run_dir, path=path, target=resolved)
        rel = path.relative_to(run_dir).as_posix()
        if rel in skipped_rel_paths:
            continue
        files.append(path)
    files.sort(key=lambda p: p.relative_to(run_dir).as_posix())
    return files


def stage_run_dir(
    run_dir: Path,
    staging_dir: Path,
    bundle_name: str,
    *,
    redactor: Redactor,
    ticket: str,
    aorta_version: str,
    now: _dt.datetime | None = None,
) -> Manifest:
    """Copy every file under ``run_dir`` through ``redactor`` into staging.

    Lays out the staging tree as
    ``<staging_dir>/<bundle_name>/<rel-path>/...`` so the tarball's
    top-level entry is ``<bundle_name>/`` (matching the layout
    documented in ``docs/probe-188/bundle.md``). Builds a
    :class:`Manifest` from the per-file :class:`RedactionCounts`
    and writes it to ``<staging_dir>/<bundle_name>/manifest.json``.

    Parent-dir responsibility: this function creates ONLY the
    bundle root (``<staging_dir>/<bundle_name>/``). The per-file
    ``dst.parent`` directories are created by the redactor's
    ``scrub_file`` implementation (see the :class:`Redactor` ABC
    docstring -- "Create the destination's parent directory if
    missing."). The split keeps the staging contract local to the
    redactor and avoids two layers of ``mkdir(exist_ok=True)``.

    Returns the in-memory manifest so the caller can show it under
    ``--review`` without re-reading from disk.
    """
    bundle_root = staging_dir / bundle_name
    bundle_root.mkdir(parents=True, exist_ok=False)

    records: list[FileRecord] = []
    for src in _iter_source_files(run_dir):
        rel = src.relative_to(run_dir)
        dst = bundle_root / rel
        counts = redactor.scrub_file(src, dst)
        records.append(
            FileRecord(
                path=rel.as_posix(),
                env_keys_removed=counts.env_keys_removed,
                paths_rewritten=counts.paths_rewritten,
                ips_rewritten=counts.ips_rewritten,
                bytes_in=counts.bytes_in,
                bytes_out=counts.bytes_out,
            )
        )

    manifest = Manifest.from_files(
        ticket=ticket,
        source_run_dir=run_dir,
        redactor_kind=redactor.kind,
        aorta_version=aorta_version,
        files=records,
        now=now,
    )
    (bundle_root / MANIFEST_FILENAME).write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def _scrub_tarinfo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    """Normalise ownership + mtime in tar headers for a shareable bundle.

    ``tarfile.add`` copies the staging file's uid/gid and -- because the
    files were just written by this process -- the operator's user/group
    names into the archive headers. A bundle is meant to be shared, so
    those headers would leak workstation identity (and make the archive
    non-reproducible). We zero uid/gid, clear uname/gname, and pin mtime
    to the epoch.

    The file MODE is left untouched on purpose: :class:`IdentityRedactor`
    preserves a ``0600`` ``probe.env`` (PR #199 review) and the tarball
    must carry that restrictive bit through to extraction.
    """
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    tarinfo.mtime = 0
    return tarinfo


def write_tarball(staging_dir: Path, bundle_name: str, output: Path) -> Path:
    """Pack ``<staging_dir>/<bundle_name>/`` into ``output.tar.gz``.

    Returns the absolute path of the written tarball. ``output`` is
    used as-is (no automatic ``.tar.gz`` suffix injection); the CLI
    layer is responsible for picking the final filename per the
    documented ``--output`` default. ``output.parent`` is created
    with ``exist_ok=True`` so a fresh checkout's ``./<ticket>...``
    default just works.

    Atomicity:

    * The tarball is written to a sibling ``<output>.partial`` first
      and renamed onto the final path with ``os.replace`` only after
      the gzip footer is flushed. Mid-write failures (``ENOSPC``,
      tarfile / gzip raising, a network FS dropping out) are cleaned up
      by unlinking ``<output>.partial`` (best-effort) and never produce a
      half-written ``output``.
    * If ``output`` already exists from a prior run, an in-flight
      failure leaves the OLD content intact -- ``os.replace`` only
      overwrites on success. The ``.partial`` sibling is unlinked
      either way so a retry does not race against a stale temp file.

    Tarball entries are added in the order :func:`_iter_source_files`
    produced (alphabetical on relative POSIX path), with the manifest
    last so consumers running ``tar -tzf`` see the data first and
    the index trailer last. ``tarfile.add`` is used with a
    deterministic ``arcname`` (so the tarball never carries the
    operator's ``--output`` parent path) and a ``filter=``
    (:func:`_scrub_tarinfo`) that strips ownership/identity metadata
    from the headers while preserving file modes.
    """
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle_root = staging_dir / bundle_name
    partial = output.with_name(output.name + ".partial")

    # Defensive: a stale .partial from a crashed previous run would
    # otherwise survive into the new try and confuse the os.replace
    # path (or, worse, get atomic-replaced into place if the next
    # write somehow succeeded against it).
    if partial.exists():
        partial.unlink()

    try:
        with tarfile.open(partial, "w:gz") as tar:
            # Walk the staged tree (NOT the source tree) so the
            # manifest written by stage_run_dir is included.
            rel_paths: list[Path] = []
            for path in bundle_root.rglob("*"):
                if path.is_file():
                    rel_paths.append(path.relative_to(bundle_root))
            # Sort with the TOP-LEVEL manifest.json last so ``tar -tzf``
            # reads it as the trailer (matches the docstring contract).
            # Key on the relative POSIX path, not the basename: a nested
            # ``*/manifest.json`` artifact must stay in alphabetical
            # position, only ``./manifest.json`` is the trailer.
            rel_paths.sort(
                key=lambda p: (p.as_posix() == MANIFEST_FILENAME, p.as_posix()),
            )
            for rel in rel_paths:
                tar.add(
                    str(bundle_root / rel),
                    arcname=f"{bundle_name}/{rel.as_posix()}",
                    filter=_scrub_tarinfo,
                )
        os.replace(partial, output)
    except BaseException:
        # Includes OSError (ENOSPC, EACCES, ...), KeyboardInterrupt,
        # and anything tarfile raises. The bundle_run_dir caller
        # wraps OSError into BundleIOError; here we just make sure
        # neither the partial nor a corrupted final file survives.
        if partial.exists():
            try:
                partial.unlink()
            except OSError:  # pragma: no cover - best-effort cleanup
                pass
        raise
    return output


def _default_output_path(bundle_name: str, output: Path | None) -> Path:
    """Resolve ``--output`` per the documented default.

    Issue #196's documented default is
    ``<safe_slug(ticket)>-<timestamp>.tar.gz`` in CWD (``bundle_name``
    is already slugified by the caller). When ``output`` is ``None``
    we apply that. When the
    operator passes a directory (``--output ./bundles``) we drop
    the default filename inside it. When they pass a file we use
    it verbatim, regardless of suffix -- ``aorta`` does not police
    the ``.tar.gz`` extension because operators sometimes have
    pipeline reasons for a non-default name.
    """
    if output is None:
        return Path.cwd() / f"{bundle_name}.tar.gz"
    if output.is_dir():
        return output / f"{bundle_name}.tar.gz"
    return output


def bundle_run_dir(
    run_dir: Path,
    *,
    ticket: str | None = None,
    output: Path | None = None,
    redaction_from: Path | None = None,
    redactor: Redactor | None = None,
    review_callback: Callable[[Manifest], bool] | None = None,
    now: _dt.datetime | None = None,
) -> Path:
    """End-to-end bundle write. CLI layer's only entry point.

    Validates the run dir, resolves the ticket, stages every file
    through the redactor into a temporary tree, writes the
    manifest, optionally pauses for ``review_callback`` (which the
    CLI wires to ``click.confirm``), and writes the tarball.

    Args:
        run_dir: ``<probe-output>/<ticket>/`` produced by
            ``aorta probe``.
        ticket: Optional override; otherwise inferred from
            ``run_dir`` basename. ``_no_ticket_`` raises
            :class:`NoTicketError` when no override is supplied.
        output: Where to write the tarball; defaults to
            ``./<bundle-name>.tar.gz``.
        redaction_from: Recipe path to load the ``redaction:``
            block from. Until Phase 3 of issue #188 ships
            ``aorta.probe.redaction``, the parameter is recorded in
            log output but NOT consumed -- the redactor stays
            :class:`IdentityRedactor`. The function signature is
            ready for #188 to wire the real loader in; that loader
            will own the ``<run-dir>/recipe.resolved.yaml`` fallback
            when it ships (today there is no fallback -- explicit
            paths only).
        redactor: Override the default :class:`IdentityRedactor`
            (test injection point + Phase 3 of #188 hand-off
            point).
        review_callback: Called with the staged :class:`Manifest`
            right before the tarball is written. Return ``True``
            to proceed, ``False`` to abort. Aborting raises
            :class:`BundleAbortedError`. ``None`` (default) skips
            the pause -- ``aorta bundle`` without ``--review`` is
            silent.
        now: Optional clock injection point for deterministic
            tests.

    Returns the absolute path of the written tarball.
    """
    run_dir = run_dir.resolve()
    _validate_run_dir(run_dir)
    resolved_ticket = resolve_ticket(run_dir, ticket)

    if redaction_from is not None:
        # Phase 3 of #188 will read the recipe's ``redaction:`` block
        # here and instantiate ``aorta.probe.redaction.RedactingRedactor``.
        # Until then we honor the flag's existence (operator can wire
        # it through pipelines today) but log that the scrubbers are
        # not yet active.
        log.info(
            "aorta bundle: --redaction-from %s ignored: "
            "aorta.probe.redaction is gated on issue #188 Phase 3 "
            "and the bundle will be written with the IdentityRedactor.",
            redaction_from,
        )

    effective_redactor = redactor if redactor is not None else IdentityRedactor()
    bundle_name = f"{safe_slug(resolved_ticket)}-{_bundle_timestamp(now)}"

    with tempfile.TemporaryDirectory(prefix="aorta-bundle-") as staging_str:
        staging = Path(staging_str)
        try:
            manifest = stage_run_dir(
                run_dir,
                staging,
                bundle_name,
                redactor=effective_redactor,
                ticket=resolved_ticket,
                aorta_version=_aorta_version(),
                now=now,
            )
        except OSError as exc:
            # Filesystem failure during staging (redactor scrub_file,
            # shutil.copyfile, or manifest write). The Redactor ABC
            # docstring promises BundleError-shaped failures here,
            # so wrap before letting Click see it. The original
            # OSError is preserved on .cause for ops tooling.
            raise BundleIOError(run_dir=run_dir, cause=exc) from exc
        if review_callback is not None and not review_callback(manifest):
            raise BundleAbortedError(run_dir=run_dir)
        try:
            return write_tarball(staging, bundle_name, _default_output_path(bundle_name, output))
        except OSError as exc:
            raise BundleIOError(run_dir=run_dir, cause=exc) from exc


__all__ = [
    "bundle_run_dir",
    "resolve_ticket",
    "stage_run_dir",
    "write_tarball",
]
