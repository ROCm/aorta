"""Importing aorta's sanitizer backends for the stand-in adapters, read-only.

The adapters wrap aorta `main` at `d22f6116` (design Section 6.11): its `src` directory
is put on `sys.path` only if `aorta` is not importable already, and bytecode is never written there,
so a read-only checkout stays untouched. A backend imported from anywhere but that snapshot is refused,
because the checker's version names the snapshot's commit. A failure to import is reported, never
raised, so the adapter can answer `unsettled` instead of crashing its worker.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import threading
from pathlib import Path

_lock = threading.Lock()


def load(module: str, aorta_src: str | None):
    """Import `aorta.instrumentation.rocjitsu_sanitizers.<module>`; returns (module, None) or (None, why)."""
    name = f"aorta.instrumentation.rocjitsu_sanitizers.{module}"
    with _lock:
        before = sys.dont_write_bytecode
        try:
            if name in sys.modules:
                mod = sys.modules[name]
            else:
                if aorta_src and aorta_src not in sys.path and importlib.util.find_spec("aorta") is None:
                    sys.path.append(aorta_src)
                sys.dont_write_bytecode = True
                mod = importlib.import_module(name)
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"[:300]
        finally:
            sys.dont_write_bytecode = before
    where = Path(getattr(mod, "__file__", "") or "").resolve()
    if aorta_src and not where.is_relative_to(Path(aorta_src).resolve()):
        # the checker's version names the snapshot's commit, so a backend imported from elsewhere is not that checker
        return None, f"aorta is imported from {where.parent}, not from the snapshot {aorta_src}"[:300]
    return mod, None


def commit(aorta_src: str | None) -> str:
    """The commit recorded beside an archived snapshot (`<root>/COMMIT`), or 'unknown'."""
    if not aorta_src:
        return "unknown"
    marker = Path(aorta_src).parent / "COMMIT"
    try:
        return marker.read_text().split()[-1][:40]
    except (OSError, IndexError):
        return "unknown"
