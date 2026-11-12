"""Atomic writes (design Section 5 item 6).

Every durable file is written to a temporary file of its own beside it, `<name>.<writer>.partial`, flushed and
fsynced, then renamed over `<name>`, so writers of the same file never share a temporary file and the last rename
wins. A file that must never be replaced is linked to `<name>` instead, which fails if `<name>` exists. Readers
never open a `.partial` file.
"""

from __future__ import annotations

import errno
import json
import os
import secrets
import threading
from pathlib import Path
from typing import Any

from .ids import canonical_json

PARTIAL_SUFFIX = ".partial"


class PartialFileError(OSError):
    """A reader was pointed at a `.partial` file."""


def is_partial(path: str | Path) -> bool:
    return Path(path).name.endswith(PARTIAL_SUFFIX)


def partial_path(path: str | Path) -> Path:
    path = Path(path)
    if is_partial(path):
        raise PartialFileError(f"refusing to write through a .partial name: {path}")
    return path.with_name(path.name + PARTIAL_SUFFIX)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as exc:
        # Some filesystems do not support fsync on a directory; the rename is still atomic there.
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, data: bytes, *, replace: bool = True) -> Path:
    """With `replace=False` an existing `path` is never replaced: FileExistsError, and nothing is written."""
    path = Path(path)
    partial_path(path)  # refuses a .partial destination
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}-{secrets.token_hex(4)}{PARTIAL_SUFFIX}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
            tmp.unlink()
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: Any, *, canonical: bool = False, replace: bool = True) -> Path:
    """Canonical JSON (Section 5 item 2) when `canonical`, else sorted and indented for people."""
    if canonical:
        data = canonical_json(obj) + b"\n"
    else:
        data = (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    return atomic_write_bytes(path, data, replace=replace)


def read_bytes(path: str | Path) -> bytes:
    if is_partial(path):
        raise PartialFileError(f"readers never open .partial files: {path}")
    return Path(path).read_bytes()


def read_text(path: str | Path) -> str:
    return read_bytes(path).decode("utf-8")
