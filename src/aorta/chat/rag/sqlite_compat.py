"""Make Chroma usable on distros whose Python ships an old sqlite3.

Chroma requires sqlite3 >= 3.35.0. CentOS Stream 9, RHEL 9 and several other
long-support distros ship 3.34.1, so ``import chromadb`` aborts on an otherwise
healthy machine. Upgrading the system library needs root and risks the package
manager, so the accepted fix is ``pysqlite3-binary``, a wheel with a current
sqlite statically linked in.

The swap has to happen before ``chromadb`` is first imported, which is why
``indexer.py`` and ``retriever.py`` call this at module import: Chroma imports
chromadb lazily inside its constructor, so by then it is too late.

This module is flow-independent -- both embedding providers store vectors in
Chroma -- and does nothing at all where the stdlib sqlite3 is new enough.
"""

from __future__ import annotations

import logging
import sqlite3
import sys

logger = logging.getLogger(__name__)

#: Chroma's floor, from chromadb/__init__.py.
MIN_SQLITE_VERSION = (3, 35, 0)

_INSTALL_HINT = (
    "Chroma needs sqlite3 >= 3.35.0 but this Python is linked against {found}, "
    "and the pysqlite3 fallback is not installed. Install the bundled build:\n"
    "  pip install pysqlite3-binary\n"
    'or, from the repo root:  pip install -e ".[sqlite]"\n'
    "No root or system sqlite upgrade is required -- the wheel carries its own "
    "copy. Common on CentOS Stream 9 and RHEL 9, which ship sqlite 3.34.1."
)


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".") if part.isdigit())


def ensure_modern_sqlite() -> None:
    """Point ``sqlite3`` at pysqlite3 when the stdlib build is too old.

    Idempotent and safe to call from several modules. Raises ``RuntimeError``
    with an actionable message when the stdlib build is too old and no
    fallback is available, in preference to Chroma's own error, which links to
    documentation rather than naming the package to install.
    """
    if _version_tuple(sqlite3.sqlite_version) >= MIN_SQLITE_VERSION:
        return

    try:
        import pysqlite3
    except ImportError as exc:
        raise RuntimeError(
            _INSTALL_HINT.format(found=sqlite3.sqlite_version)
        ) from exc

    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
    logger.info(
        "Replaced sqlite3 %s with pysqlite3 %s for Chroma.",
        sqlite3.sqlite_version,
        pysqlite3.sqlite_version,
    )
