"""No-follow, fd-relative filesystem primitives for paths below ``--results-dir``.

The collector output tree (``<results_root>/<workload>/<trial>/<subdir>``) is
handed to the profiled command as a ``-d`` / ``-n`` target. Everything below the
operator's ``--results-dir`` is therefore payload-writable *during the run*, and
a pathname-based guard is inherently time-of-check/time-of-use: after
``resolve()`` and an ``is_symlink()`` walk return, ``rmtree`` / ``rglob`` /
retention reopen the path from ``/`` and a payload descendant that outlived its
process group (a ``setsid``'d child, a detached container) can have swapped an
ancestor for a symlink in the gap, redirecting the operation outside the tree.

The only race-free defence is to stop re-resolving pathnames: descend the tree
once with ``O_NOFOLLOW`` on every component below the trusted anchor, hold the
resulting directory file descriptors, and perform every subsequent stat / read /
write / unlink / rmdir *relative to those fds*. The kernel resolves a name under
a dir fd only within that inode, so a later swap of an ancestor pathname cannot
redirect it. This module is the shared leaf :mod:`aorta.run.collectors`,
:mod:`aorta.run.retention` and :mod:`aorta.run.dispatcher` build on; it depends
only on the stdlib so retention stays dependency-free.

The same reasoning covers the dispatcher's *own* output, not just the collector
artifacts. The per-workload directory sits below the same operator boundary, so
the trial record and the captured logs are written through a held fd with
:func:`open_write_nofollow` rather than by pathname -- otherwise a stale or
payload-planted link at the workload component silently redirects the audit
trail outside ``--results-dir``.

Two things are needed beyond ``O_NOFOLLOW``, because it is a narrower guarantee
than it looks:

* **It does not say the anchor is still the operator's directory.** A rename can
  move a *real* directory into the anchor's pathname, and the anchor's parent is
  necessarily opened by pathname (it lives above the operator's boundary and may
  hold the operator's own links). :class:`TrustedAnchor` therefore carries the
  ``(st_dev, st_ino)`` the anchor had before the first payload launched, and the
  descent refuses to continue when the directory it opened is a different inode.
* **It only guards the final component, and ``dir_fd`` only applies to a
  relative path.** So a "component" that is absolute, ``..``, or carries a
  separator silently converts a guarded descent into an unguarded one. Every
  name handed to a primitive here is therefore checked to be a single directory
  entry (:func:`_require_child_name`) before it reaches the kernel.

One property is also *not* obtainable from the descent: the type of a file can
change between the walk that selected it and the read that consumes it. A
regular file replaced by a FIFO turns a blocking ``open`` into an indefinite
hang, so :func:`secure_open_read` and :func:`open_write_nofollow` open
``O_NONBLOCK`` and validate the *descriptor* rather than trusting the earlier
``lstat``.

Everything here is POSIX-only (``O_NOFOLLOW`` / ``O_DIRECTORY`` / ``dir_fd``).
:data:`HAVE_FD_TRAVERSAL` is ``False`` on a platform that lacks them; callers
fall back to their previous lexical guards there. ``aorta probe`` is Linux-only
by design, so the fd path is the one that runs in practice.
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import logging
import os
import stat
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TextIO

log = logging.getLogger(__name__)

_O_NOFOLLOW: int = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY: int = getattr(os, "O_DIRECTORY", 0)
_O_NONBLOCK: int = getattr(os, "O_NONBLOCK", 0)

# Feature detection in the same spirit as the flags above, not an optional
# third-party dependency: ``fcntl`` is a CPython builtin extension present on
# every POSIX build and absent on Windows, so ``ImportError`` is the only way
# this can fail -- there is no half-installed state to fail differently.
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows, where the fd path is off anyway
    _fcntl = None  # type: ignore[assignment]

#: True when the platform exposes the primitives needed for race-free,
#: fd-relative traversal. False on Windows and anywhere ``O_NOFOLLOW`` /
#: ``O_DIRECTORY`` / ``dir_fd`` are unavailable, where callers keep their
#: lexical (``resolve()`` + ``is_symlink()``) guards.
HAVE_FD_TRAVERSAL: bool = bool(_O_NOFOLLOW and _O_DIRECTORY and os.open in os.supports_dir_fd)


class UnsafePathError(OSError):
    """A path could not be used without leaving the trusted tree.

    Covers a component that is a symlink, missing, or not a directory; a name
    that is not a single directory entry; an anchor that is no longer the frozen
    inode; and a file that turned into something other than a regular file
    between the walk and the open. It is an :class:`OSError` subclass so
    existing ``except OSError`` guards in the collector and retention paths keep
    failing closed without a new import. Callers that need to tell *absent* from
    *hostile* read ``errno`` (``ENOENT`` is the only benign one).
    """


@dataclasses.dataclass(frozen=True)
class TrustedAnchor:
    """A canonical directory path plus the inode it named before the payload ran.

    ``O_NOFOLLOW`` answers "is this component a symlink", which is not the same
    question as "is this still the operator's directory". Two swaps slip past a
    no-follow-only descent:

    * The anchor's *parent* is opened by pathname, above the operator's
      boundary, because it may legitimately hold the operator's own links. A
      payload that can write the grandparent can rename that parent aside and
      leave a symlink to a planted tree that contains an ordinary ``results``
      directory; the anchor then opens with no symlink in sight.
    * The anchor itself can be renamed aside and a *real* directory renamed
      into its place. Nothing in that pathname is a link, so every no-follow
      check passes.

    Both are defeated by naming the inode rather than the path: ``identity`` is
    the ``(st_dev, st_ino)`` pair captured before the first payload launches,
    and :func:`open_dir_nofollow` refuses to descend when the directory it
    opened is not that inode. A file descriptor would be the stronger pin, but
    the anchor has to cross the isolated-worker process boundary, and fds do
    not.

    ``identity`` is ``None`` for an anchor nobody froze -- a direct
    programmatic caller, or a platform where the stat failed. The descent then
    keeps its no-follow-only behaviour rather than inventing a pin.
    """

    path: Path
    identity: tuple[int, int] | None = None

    @classmethod
    def freeze(cls, path: Path | str) -> TrustedAnchor:
        """Capture ``path``'s current inode identity as the trust anchor.

        The caller must do this *before* any payload runs and after the
        directory exists, which for the dispatcher means after it has created
        the results tree. A stat failure yields an unpinned anchor rather than
        raising -- an anchor that cannot be pinned is still worth threading for
        its no-follow descent -- but it is logged at WARNING rather than
        swallowed: the guard silently drops to its weaker form, and the
        dispatcher creates the directory first precisely so this cannot happen.
        """
        anchor = Path(os.fspath(path))
        try:
            info = os.stat(anchor)
        except OSError as exc:
            log.warning(
                "could not pin the trust anchor %s (%s); path guards fall back "
                "to no-follow-only, which cannot detect the directory being "
                "renamed aside and replaced by a real one",
                anchor,
                exc,
            )
            return cls(anchor)
        return cls(anchor, (info.st_dev, info.st_ino))


def as_anchor(trusted_root: TrustedAnchor | Path | str) -> TrustedAnchor:
    """Coerce a path or an already-frozen anchor into a :class:`TrustedAnchor`.

    Lets every guard take either form, so a caller that has no frozen identity
    (a test, a direct programmatic caller) keeps passing a plain path.
    """
    if isinstance(trusted_root, TrustedAnchor):
        return trusted_root
    return TrustedAnchor(Path(os.fspath(trusted_root)))


def _require_child_name(name: str, where: str) -> str:
    """Refuse a ``name`` that is anything other than one entry of a directory.

    Every fd-relative primitive here promises to act *inside* the descriptor it
    is handed, but ``dir_fd`` alone does not deliver that: an absolute path
    makes :func:`os.open` ignore ``dir_fd`` outright, ``..`` walks above it, and
    ``O_NOFOLLOW`` only guards the **final** component, so a name carrying a
    separator has its intermediate components resolved with symlinks followed.
    Each of those turns a guarded descent into an unguarded one, so the name is
    checked before it reaches the kernel.

    Today's callers cannot trip this -- :func:`relative_components` decomposes a
    ``relative_to`` result and everything else forwards ``os.listdir`` entries,
    neither of which can produce such a name -- so this enforces the contract
    for the next caller rather than fixing a live escape. That is the point: the
    guarantee belongs to these helpers, not to the discipline of whoever calls
    them.

    Raises:
        UnsafePathError: the name is empty, ``.``, ``..``, absolute, or contains
            a path separator.
    """
    separators = [sep for sep in (os.sep, os.altsep) if sep]
    if (
        not name
        or name in (os.curdir, os.pardir)
        or os.path.isabs(name)
        or any(sep in name for sep in separators)
    ):
        raise UnsafePathError(
            errno.EINVAL,
            f"refusing to use {name!r} under {where!r}: expected a single "
            "directory entry name, not an empty, absolute, relative-traversal "
            "or multi-component path",
        )
    return name


def relative_components(trusted_root: Path, target: Path) -> list[str] | None:
    """The lexical path components of ``target`` below ``trusted_root``.

    Returns the component names to descend, or ``None`` when ``target`` is not
    lexically inside ``trusted_root`` or any component is ``..`` / empty. Neither
    path is touched on disk; this is pure lexical decomposition of two absolute
    paths the caller already holds (the anchor and the collector dir), so the fd
    walk can reject an escape before opening anything.
    """
    try:
        rel = target.relative_to(trusted_root)
    except ValueError:
        return None
    parts = [part for part in rel.parts if part not in ("", os.curdir)]
    if any(part == os.pardir for part in parts):
        return None
    return parts


@contextlib.contextmanager
def open_dir_nofollow(
    trusted_root: TrustedAnchor | Path | str,
    components: Sequence[str],
    *,
    create_missing: bool = False,
) -> Iterator[int]:
    """Yield a dir fd for ``trusted_root``/*components*, opened without following
    any symlink below ``trusted_root``.

    The anchor directory is opened first (see :func:`_open_anchor`) and, when
    the caller froze a :class:`TrustedAnchor`, checked to be the inode the
    operator's ``--results-dir`` named before the payload launched. Every
    component below it is then opened ``O_RDONLY | O_NOFOLLOW | O_DIRECTORY``
    relative to its parent fd, so a payload-owned symlink at any of them raises
    :class:`UnsafePathError` instead of redirecting the descent. All
    intermediate fds are closed as the walk proceeds; the yielded leaf fd is
    closed on context exit. The caller must not use the fd after the ``with``
    block.

    When ``create_missing`` is set, a component that does not exist is created
    with ``os.mkdir(dir_fd=...)`` and then opened -- the fd-relative equivalent
    of ``mkdir(parents=True)``, so a reset can build its trial subdirectory
    chain without ever following a symlink. A component that exists as a symlink
    or a non-directory still raises rather than being created over.

    Raises:
        UnsafePathError: a component is a symlink, is missing (and
            ``create_missing`` is false), is not a directory, is not a single
            directory entry name, or the anchor is no longer the frozen inode.
    """
    anchor = as_anchor(trusted_root)
    dir_fd = _open_anchor(anchor, create_missing)
    try:
        for name in components:
            child = _open_child_dir(dir_fd, name, anchor.path, create_missing)
            os.close(dir_fd)
            dir_fd = child
        yield dir_fd
    finally:
        os.close(dir_fd)


def _open_anchor(anchor: TrustedAnchor, create_missing: bool) -> int:
    """Open the trusted root itself and confirm it is the frozen inode.

    The anchor's **parent** is opened plainly (``O_RDONLY | O_DIRECTORY``): it
    sits *above* the operator's ``--results-dir``, so it may legitimately
    contain the operator's own symlinks (a ``/tmp`` that is really
    ``/private/tmp``, a mounted scratch path) that the dispatcher already
    resolved away. The anchor inode itself is the *first* payload-swappable
    component -- the payload can replace the results directory after launch --
    so it is opened ``O_NOFOLLOW`` relative to that parent fd.

    Opening the parent by pathname is what makes the identity check
    load-bearing rather than belt-and-braces: a payload that can write the
    grandparent can rename the parent aside and leave a symlink to a planted
    tree whose own ``results`` directory is perfectly ordinary, and no
    no-follow check below would notice. Comparing ``fstat`` against
    :attr:`TrustedAnchor.identity` also catches the reverse trick of renaming
    the anchor aside and moving a *real* directory into its pathname.

    Raises:
        UnsafePathError: the anchor is a symlink, is missing (and
            ``create_missing`` is false), or is not the frozen inode.
    """
    name = anchor.path.name
    if name:
        parent_fd = os.open(os.fspath(anchor.path.parent), os.O_RDONLY | _O_DIRECTORY)
        try:
            dir_fd = _open_child_dir(parent_fd, name, anchor.path, create_missing)
        finally:
            os.close(parent_fd)
    else:
        # Degenerate anchor (a filesystem root has no name): nothing above it to
        # descend from, so open it plainly and rely on the below-components.
        dir_fd = os.open(os.fspath(anchor.path), os.O_RDONLY | _O_DIRECTORY)
    if anchor.identity is None:
        return dir_fd
    try:
        info = os.fstat(dir_fd)
        if (info.st_dev, info.st_ino) != anchor.identity:
            raise UnsafePathError(
                errno.ESTALE,
                f"refusing to descend into {os.fspath(anchor.path)!r}: it is no "
                f"longer the directory frozen before the run "
                f"(expected inode {anchor.identity}, found "
                f"{(info.st_dev, info.st_ino)}), so it was renamed or replaced",
            )
    except BaseException:
        os.close(dir_fd)
        raise
    return dir_fd


def _open_child_dir(
    parent_fd: int, name: str, trusted_root: Path | str, create_missing: bool
) -> int:
    """Open (or optionally create) directory ``name`` under ``parent_fd``, no-follow.

    The single funnel every descent goes through (:func:`open_dir_nofollow`,
    :func:`open_dir_at`, :func:`_open_anchor`), so the child-name check lives
    here rather than being repeated at each entry point.
    """
    _require_child_name(name, os.fspath(trusted_root))
    try:
        return os.open(name, os.O_RDONLY | _O_NOFOLLOW | _O_DIRECTORY, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create_missing:
            raise UnsafePathError(
                errno.ENOENT,
                f"refusing to descend into {name!r} under {os.fspath(trusted_root)!r}: "
                "no such directory",
            ) from None
    except OSError as exc:
        # ELOOP: the component is a symlink and O_NOFOLLOW refused it. ENOTDIR:
        # it is a non-directory (a file, or a symlink to one). Both mean "cannot
        # descend here without leaving the trusted tree" -- fail closed.
        raise UnsafePathError(
            exc.errno,
            f"refusing to descend into {name!r} under {os.fspath(trusted_root)!r}: "
            f"{exc.strerror}",
        ) from exc
    # create_missing and the component was absent: make it, then open it. A
    # concurrent create losing the mkdir race is fine -- the open below still
    # gets the (real) directory, or refuses a symlink a payload planted instead.
    try:
        os.mkdir(name, dir_fd=parent_fd)
    except FileExistsError:
        pass
    try:
        return os.open(name, os.O_RDONLY | _O_NOFOLLOW | _O_DIRECTORY, dir_fd=parent_fd)
    except OSError as exc:
        raise UnsafePathError(
            exc.errno,
            f"refusing to descend into {name!r} under {os.fspath(trusted_root)!r}: "
            f"{exc.strerror}",
        ) from exc


@contextlib.contextmanager
def open_dir_at(base_fd: int, components: Sequence[str]) -> Iterator[int]:
    """Yield a dir fd for *components* under ``base_fd``, no-follow at every step.

    The short form of :func:`open_dir_nofollow` for a caller that already holds
    a verified directory fd and wants to reach a subdirectory of it. No anchor
    identity check is needed: ``base_fd`` names an inode directly, so the
    descent cannot be redirected by any pathname swap. Used to reopen one
    artifact's directory at a time so a large capture never holds more than a
    couple of descriptors.

    Raises:
        UnsafePathError: a component is a symlink, missing, not a directory, or
            not a single directory entry name.
    """
    dir_fd = os.dup(base_fd)
    try:
        for name in components:
            child = _open_child_dir(dir_fd, name, ".", False)
            os.close(dir_fd)
            dir_fd = child
        yield dir_fd
    finally:
        os.close(dir_fd)


def stat_at(dir_fd: int, name: str) -> os.stat_result | None:
    """``lstat`` ``name`` relative to ``dir_fd``, or ``None`` when it is missing.

    Never follows a final-component symlink (``follow_symlinks=False``), so the
    caller sees the link itself rather than its target.

    ``None`` means *absent*, which is not the same as *unusable*: a name that is
    not a single directory entry raises instead, because ``stat_at(fd, "..")``
    would report on the parent and read as a legitimate answer.

    Raises:
        UnsafePathError: ``name`` is not a single directory entry name.
    """
    _require_child_name(name, f"fd {dir_fd}")
    try:
        return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def remove_entry_at(dir_fd: int, name: str) -> tuple[int, int]:
    """Recursively remove ``name`` under ``dir_fd`` without following symlinks.

    A symlink or regular file is unlinked directly (never dereferenced). A real
    directory is emptied depth-first through fds opened ``O_NOFOLLOW`` relative
    to their parent, then ``rmdir``-ed. Because every descent holds a dir fd, a
    payload swap of an interior component after the walk began cannot redirect a
    later unlink outside the subtree.

    Returns ``(files_removed, bytes_freed)`` -- best effort sizes taken from an
    ``lstat`` before the unlink; a file that vanishes mid-walk contributes 0.

    Raises:
        UnsafePathError: ``name`` is not a single directory entry name. That
            matters most here of all the primitives in this module:
            ``remove_entry_at(fd, "..")`` would recursively empty the *parent*
            of the held directory.
        OSError: the entry could not be removed. Callers decide tolerance.
    """
    info = stat_at(dir_fd, name)
    if info is None:
        return (0, 0)
    if not stat.S_ISDIR(info.st_mode):
        size = info.st_size
        os.unlink(name, dir_fd=dir_fd)
        return (1, size)
    child_fd = os.open(name, os.O_RDONLY | _O_NOFOLLOW | _O_DIRECTORY, dir_fd=dir_fd)
    try:
        files, freed = _empty_dir(child_fd)
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=dir_fd)
    return (files, freed)


def _empty_dir(dir_fd: int) -> tuple[int, int]:
    """Remove every entry under ``dir_fd`` (leaving the dir itself)."""
    files = 0
    freed = 0
    for name in os.listdir(dir_fd):
        removed, bytes_freed = remove_entry_at(dir_fd, name)
        files += removed
        freed += bytes_freed
    return (files, freed)


def iter_regular_files(base_fd: int) -> Iterator[tuple[str, int, str, int]]:
    """Walk the tree under ``base_fd``, yielding one entry per regular file.

    Yields ``(relative_posix_path, parent_dir_fd, name, size)`` for every
    non-symlink regular file, never following a symlink and never descending
    into a symlinked directory. ``parent_dir_fd`` is a live directory fd the
    file sits directly in; the caller MUST act on it (``os.open`` /
    ``os.unlink`` / ``os.stat`` with ``dir_fd=parent_dir_fd``) before the
    generator advances, because the fd is closed once its directory is
    exhausted. Directories are visited top-down; the caller cannot rely on
    order beyond that.
    """
    yield from _walk(base_fd, "")


def _walk(dir_fd: int, prefix: str) -> Iterator[tuple[str, int, str, int]]:
    subdirs: list[str] = []
    for name in sorted(os.listdir(dir_fd)):
        info = stat_at(dir_fd, name)
        if info is None or stat.S_ISLNK(info.st_mode):
            continue
        rel = f"{prefix}{name}"
        if stat.S_ISDIR(info.st_mode):
            subdirs.append(name)
        elif stat.S_ISREG(info.st_mode):
            yield (rel, dir_fd, name, info.st_size)
    for name in subdirs:
        try:
            child_fd = os.open(
                name, os.O_RDONLY | _O_NOFOLLOW | _O_DIRECTORY, dir_fd=dir_fd
            )
        except OSError:
            continue
        try:
            yield from _walk(child_fd, f"{prefix}{name}/")
        finally:
            os.close(child_fd)


@contextlib.contextmanager
def secure_open_read(dir_fd: int, name: str, **text_kwargs: object) -> Iterator[TextIO]:
    """Open ``name`` under ``dir_fd`` read-only and no-follow as a text stream.

    Refuses a symlink at the final component (``O_NOFOLLOW``). ``text_kwargs``
    are forwarded to :func:`os.fdopen` (e.g. ``encoding="utf-8"``,
    ``newline=""``). The stream (and its fd) is closed on context exit.

    The open is ``O_NONBLOCK`` and the *opened descriptor* is then checked to be
    a regular file. A caller that saw a regular file in an earlier ``lstat``
    cannot rely on that here: a surviving payload process can unlink the file
    and ``mkfifo`` in its place, and a blocking ``O_RDONLY`` on a FIFO with no
    writer waits forever -- which would hang post-run summarization or the
    retention sweep rather than degrade it. Checking the fd rather than the name
    closes the window, because by then the descriptor is the thing we will read.
    ``O_NONBLOCK`` is cleared once the type is confirmed, so the stream behaves
    like any other file read.

    Raises:
        UnsafePathError: ``name`` is not a single directory entry name, or what
            was opened is not a regular file.
        OSError: the file could not be opened.
    """
    _require_child_name(name, f"fd {dir_fd}")
    fd = os.open(name, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK, dir_fd=dir_fd)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise UnsafePathError(
                errno.EINVAL,
                f"refusing to read {name!r}: it is not a regular file "
                f"(mode {stat.filemode(info.st_mode)}); it was replaced after "
                "the directory walk saw it",
            )
        _clear_nonblock(fd)
        stream = os.fdopen(fd, "r", **text_kwargs)  # type: ignore[arg-type]
    except BaseException:
        # Nothing owns the fd yet -- ``fdopen`` only adopts it on success -- so
        # a refusal here (or an interrupt) would otherwise leak it.
        os.close(fd)
        raise
    try:
        yield stream
    finally:
        stream.close()


def open_write_nofollow(dir_fd: int, name: str, **text_kwargs: object) -> TextIO:
    """Create/truncate ``name`` under ``dir_fd`` for writing, refusing a symlink.

    The write-side counterpart of :func:`secure_open_read`, and returns the
    stream rather than a context manager so a caller can keep the existing
    ``fh = open(...)`` shape (the dispatcher's log capture holds both handles for
    the length of a trial and has its own cleanup path for a partial open).

    ``O_NOFOLLOW`` refuses a symlink planted at the final component, so a
    payload cannot redirect a record write outside the tree by leaving a link
    where the file goes. ``O_CREAT`` without ``O_EXCL`` on purpose: a re-run
    legitimately overwrites its own trial record, and ``O_EXCL`` would break
    probe resume. ``O_NONBLOCK`` plus an ``S_ISREG`` check on the *descriptor*
    covers the write-side version of the FIFO trap -- opening a FIFO for writing
    blocks until a reader appears (or fails ``ENXIO`` non-blocking), which would
    hang the trial rather than fail it.

    The caller owns the returned stream and must close it.

    Raises:
        UnsafePathError: ``name`` is not a single directory entry name, or what
            was opened is not a regular file.
        OSError: the file could not be created or opened.
    """
    _require_child_name(name, f"fd {dir_fd}")
    fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _O_NOFOLLOW | _O_NONBLOCK,
        0o644,
        dir_fd=dir_fd,
    )
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise UnsafePathError(
                errno.EINVAL,
                f"refusing to write {name!r}: it is not a regular file "
                f"(mode {stat.filemode(info.st_mode)})",
            )
        _clear_nonblock(fd)
        return os.fdopen(fd, "w", **text_kwargs)  # type: ignore[arg-type,return-value]
    except BaseException:
        # ``fdopen`` only adopts the fd on success, so a refusal or an interrupt
        # in between would otherwise leak it.
        os.close(fd)
        raise


def _clear_nonblock(fd: int) -> None:
    """Drop ``O_NONBLOCK`` from ``fd``, best effort.

    Only reached once ``fd`` is known to be a regular file, where the flag is a
    no-op for reads on Linux. Cleared anyway so the handle handed to a parser is
    an ordinary blocking file, and suppressed rather than raised because failing
    to tidy a flag is not a reason to drop a readable artifact.
    """
    if _fcntl is None:
        return
    with contextlib.suppress(OSError):
        flags = _fcntl.fcntl(fd, _fcntl.F_GETFL)
        _fcntl.fcntl(fd, _fcntl.F_SETFL, flags & ~_O_NONBLOCK)


def prune_empty_dirs(base_fd: int) -> None:
    """Remove now-empty subdirectories under ``base_fd``, depth-first.

    The base directory itself is never removed. Symlinked directories are never
    descended or removed. Best-effort: a directory that cannot be removed (still
    non-empty, or a race) is left in place.
    """
    _prune(base_fd)


def _prune(dir_fd: int) -> None:
    for name in os.listdir(dir_fd):
        info = stat_at(dir_fd, name)
        if info is None or not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            continue
        try:
            child_fd = os.open(
                name, os.O_RDONLY | _O_NOFOLLOW | _O_DIRECTORY, dir_fd=dir_fd
            )
        except OSError:
            continue
        try:
            _prune(child_fd)
            is_empty = not os.listdir(child_fd)
        finally:
            os.close(child_fd)
        if is_empty:
            with contextlib.suppress(OSError):
                os.rmdir(name, dir_fd=dir_fd)


__all__ = [
    "HAVE_FD_TRAVERSAL",
    "TrustedAnchor",
    "UnsafePathError",
    "as_anchor",
    "iter_regular_files",
    "open_dir_at",
    "open_dir_nofollow",
    "open_write_nofollow",
    "prune_empty_dirs",
    "relative_components",
    "remove_entry_at",
    "secure_open_read",
    "stat_at",
]
