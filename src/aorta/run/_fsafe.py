"""No-follow, fd-relative filesystem primitives for collector path guards.

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
unlink / rmdir *relative to those fds*. The kernel resolves a name under a dir fd
only within that inode, so a later swap of an ancestor pathname cannot redirect
it. This module is the shared leaf both :mod:`aorta.run.collectors` and
:mod:`aorta.run.retention` build on; it depends only on the stdlib so retention
stays dependency-free.

Everything here is POSIX-only (``O_NOFOLLOW`` / ``O_DIRECTORY`` / ``dir_fd``).
:data:`HAVE_FD_TRAVERSAL` is ``False`` on a platform that lacks them; callers
fall back to their previous lexical guards there. ``aorta probe`` is Linux-only
by design, so the fd path is the one that runs in practice.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import stat
from collections.abc import Iterator, Sequence
from pathlib import Path

log = logging.getLogger(__name__)

_O_NOFOLLOW: int = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY: int = getattr(os, "O_DIRECTORY", 0)

#: True when the platform exposes the primitives needed for race-free,
#: fd-relative traversal. False on Windows and anywhere ``O_NOFOLLOW`` /
#: ``O_DIRECTORY`` / ``dir_fd`` are unavailable, where callers keep their
#: lexical (``resolve()`` + ``is_symlink()``) guards.
HAVE_FD_TRAVERSAL: bool = bool(_O_NOFOLLOW and _O_DIRECTORY and os.open in os.supports_dir_fd)


class UnsafePathError(OSError):
    """A component below the trusted root was a symlink, missing, or not a dir.

    Raised while descending with :func:`open_dir_nofollow`. It is an
    :class:`OSError` subclass so existing ``except OSError`` guards in the
    collector and retention paths keep failing closed without a new import.
    """


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
    trusted_root: Path | str,
    components: Sequence[str],
    *,
    create_missing: bool = False,
) -> Iterator[int]:
    """Yield a dir fd for ``trusted_root``/*components*, opened without following
    any symlink below ``trusted_root``.

    The **parent** of ``trusted_root`` is opened plainly (``O_RDONLY |
    O_DIRECTORY``): it sits *above* the operator's ``--results-dir``, so it is
    not payload-writable and may legitimately contain the operator's own
    symlinks (a ``/tmp`` that is really ``/private/tmp``, a mounted scratch
    path) that the dispatcher already resolved away. The ``--results-dir``
    inode itself is the *first* payload-swappable component -- the payload can
    replace the results directory with a symlink after launch -- so it, and
    every component below it, is opened ``O_RDONLY | O_NOFOLLOW | O_DIRECTORY``
    relative to its parent fd. A payload-owned symlink at any of them (including
    the anchor itself) raises :class:`UnsafePathError` instead of redirecting the
    descent. All intermediate fds are closed as the walk proceeds; the yielded
    leaf fd is closed on context exit. The caller must not use the fd after the
    ``with`` block.

    When ``create_missing`` is set, a component that does not exist is created
    with ``os.mkdir(dir_fd=...)`` and then opened -- the fd-relative equivalent
    of ``mkdir(parents=True)``, so a reset can build its trial subdirectory
    chain without ever following a symlink. A component that exists as a symlink
    or a non-directory still raises rather than being created over.

    Raises:
        UnsafePathError: a component is a symlink, is missing (and
            ``create_missing`` is false), or is not a directory.
    """
    anchor = Path(os.fspath(trusted_root))
    anchor_name = anchor.name
    if anchor_name:
        # Open the parent (above --results-dir, not payload territory) plainly,
        # then descend the results-dir inode itself no-follow so a swap of it is
        # caught like any component below it.
        base = os.fspath(anchor.parent)
        walk: list[str] = [anchor_name, *components]
    else:
        # Degenerate anchor (a filesystem root has no name): nothing above it to
        # anchor against, so open it plainly and rely on the below-components.
        base = os.fspath(trusted_root)
        walk = list(components)
    dir_fd = os.open(base, os.O_RDONLY | _O_DIRECTORY)
    try:
        for name in walk:
            child = _open_child_dir(dir_fd, name, trusted_root, create_missing)
            os.close(dir_fd)
            dir_fd = child
        yield dir_fd
    finally:
        os.close(dir_fd)


def _open_child_dir(
    parent_fd: int, name: str, trusted_root: Path | str, create_missing: bool
) -> int:
    """Open (or optionally create) directory ``name`` under ``parent_fd``, no-follow."""
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


def stat_at(dir_fd: int, name: str) -> os.stat_result | None:
    """``lstat`` ``name`` relative to ``dir_fd``, or ``None`` when it is missing.

    Never follows a final-component symlink (``follow_symlinks=False``), so the
    caller sees the link itself rather than its target.
    """
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
def secure_open_read(dir_fd: int, name: str, **text_kwargs: object) -> Iterator[object]:
    """Open ``name`` under ``dir_fd`` read-only and no-follow as a text stream.

    Refuses a symlink at the final component (``O_NOFOLLOW``). ``text_kwargs``
    are forwarded to :func:`os.fdopen` (e.g. ``encoding="utf-8"``,
    ``newline=""``). The stream (and its fd) is closed on context exit.
    """
    fd = os.open(name, os.O_RDONLY | _O_NOFOLLOW, dir_fd=dir_fd)
    stream = os.fdopen(fd, "r", **text_kwargs)  # type: ignore[arg-type]
    try:
        yield stream
    finally:
        stream.close()


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
    "UnsafePathError",
    "iter_regular_files",
    "open_dir_nofollow",
    "prune_empty_dirs",
    "relative_components",
    "remove_entry_at",
    "secure_open_read",
    "stat_at",
]
