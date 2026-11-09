"""Atomic writes (design Section 5 item 6).

Every durable file is written as `<name>.partial`, flushed and fsynced, then renamed over `<name>`.
Readers never open a `.partial` file.
"""

from __future__ import annotations

import errno
import json
import os
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


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    tmp = partial_path(path)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: Any, *, canonical: bool = False) -> Path:
    """Canonical JSON (Section 5 item 2) when `canonical`, else sorted and indented for people."""
    if canonical:
        data = canonical_json(obj) + b"\n"
    else:
        data = (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    return atomic_write_bytes(path, data)


def read_bytes(path: str | Path) -> bytes:
    if is_partial(path):
        raise PartialFileError(f"readers never open .partial files: {path}")
    return Path(path).read_bytes()


def read_text(path: str | Path) -> str:
    return read_bytes(path).decode("utf-8")
