from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

_CURSOR_FILE = "watch_cursors.json"


def _cursor_path(job_dir: Path) -> Path:
    return job_dir / _CURSOR_FILE


def load_cursors(job_dir: Path) -> dict[str, int]:
    """Load byte-offset cursor state. Returns {} if not yet created."""
    p = _cursor_path(job_dir)
    if not p.is_file():
        return {}
    try:
        return {k: int(v) for k, v in json.loads(p.read_text()).items()}
    except Exception:
        return {}


def save_cursors(job_dir: Path, cursors: dict[str, int]) -> None:
    """Atomically write cursor state to disk."""
    p = _cursor_path(job_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({str(k): v for k, v in cursors.items()}, indent=2))
    os.replace(tmp, p)


def read_new_bytes(path: Path, cursor: int, max_bytes: int = 32768) -> tuple[str, int]:
    """Read up to max_bytes of new content from path starting at cursor.

    Returns (new_text, new_cursor).
    Resets cursor to 0 on file rotation (size shrank).
    Returns ("", cursor) if nothing new.
    """
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return "", cursor

    if size < cursor:
        # File was rotated or truncated — restart from beginning
        cursor = 0

    if size == cursor:
        return "", cursor

    to_read = min(max_bytes, size - cursor)
    with path.open("rb") as f:
        f.seek(cursor)
        raw = f.read(to_read)

    text = raw.decode("utf-8", errors="replace")
    return text, cursor + len(raw)
