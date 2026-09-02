"""Make sqlite-vec usable on distros whose Python ships an old or crippled sqlite3.

CentOS Stream 9, RHEL 9 and several other long-support distros ship sqlite
3.34.1. Upgrading the system library needs root and risks the package manager,
so the accepted fix is ``pysqlite3-binary``, a wheel with a current sqlite
statically linked in.

Dropping chromadb did not remove this class of problem, it raised it. Chroma
wanted 3.35.0; the KNN query shape ``retriever.py`` uses wants 3.41.0, and
sqlite-vec adds a second, independent requirement Chroma never had -- it is a
*loadable extension*, so the sqlite3 module must also have been built with
``--enable-loadable-sqlite-extensions``. Deleting this shim would break both
sets of users, and the same wheel fixes both.

Unlike the Chroma era there is no import to get ahead of: sqlite-vec is loaded
per connection, so both guards run where the connection is opened. This module
does nothing at all where the stdlib sqlite3 is new enough and can load
extensions.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from typing import Any

logger = logging.getLogger(__name__)

#: sqlite-vec's floor for the queries below it. Upstream tests as low as 3.31.1
#: (asg017/sqlite-vec#59) and documents >= 3.41 as the version where every query
#: shape works: before 3.41 SQLite does not pass a ``LIMIT`` down to a virtual
#: table as ``SQLITE_INDEX_CONSTRAINT_LIMIT``, so the ``LIMIT ?`` that bounds
#: ``retriever.py``'s vec0 KNN scan never reaches the extension and the search
#: fails rather than returning k neighbours. 3.41.0 sits above the 3.34.1 that
#: RHEL 9 and CentOS Stream 9 ship, which is why the swap below still matters.
MIN_SQLITE_VERSION = (3, 41, 0)

_INSTALL_HINT = (
    "sqlite-vec needs sqlite3 >= {needed} but this Python is linked against "
    "{found}, and the pysqlite3 fallback is not installed. Install the bundled "
    "build:\n"
    "  pip install 'amd-aorta[chat-sqlite]'\n"
    "or, if you manage dependencies yourself:  pip install pysqlite3-binary\n"
    "No root or system sqlite upgrade is required -- the wheel carries its own "
    "copy. Common on CentOS Stream 9 and RHEL 9, which ship sqlite 3.34.1."
)

_EXTENSION_HINT = (
    "sqlite-vec is a loadable sqlite extension, but this Python's sqlite3 "
    "cannot load one: it was built without --enable-loadable-sqlite-extensions, "
    "so Connection.enable_load_extension is unavailable. The sqlite version "
    "itself is fine; this is a separate build option. Install the bundled "
    "build, whose wheel enables it:\n"
    "  pip install 'amd-aorta[chat-sqlite]'\n"
    "or, if you manage dependencies yourself:  pip install pysqlite3-binary\n"
    "No root is required either way."
)

#: Set once the running sqlite3 has been shown to load extensions, and cleared
#: by a swap, since the answer belongs to the module that is now installed.
_extensions_checked = False


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".") if part.isdigit())


def _loads_extensions(module: Any) -> bool:
    """Whether *module* can load an extension, asked of the module itself.

    Separate from the version: ``--enable-loadable-sqlite-extensions`` is a
    build option, so a brand-new sqlite can be missing it and an older one have
    it. CPython omits the method entirely when the option is off; some builds
    keep it and raise on the call.
    """
    try:
        conn = module.connect(":memory:")
    except module.Error:  # pragma: no cover - a build that cannot open :memory:
        return False
    try:
        conn.enable_load_extension(True)
        conn.enable_load_extension(False)
    except (AttributeError, module.NotSupportedError):
        return False
    else:
        return True
    finally:
        conn.close()


def ensure_modern_sqlite() -> None:
    """Point ``sqlite3`` at pysqlite3 when the stdlib build is too old.

    Idempotent and safe to call from several modules. Raises ``RuntimeError``
    with an actionable message when the stdlib build is too old and no fallback
    is available, in preference to sqlite-vec's own error, which surfaces as a
    bare ``OperationalError`` from inside the extension.
    """
    global sqlite3, _extensions_checked

    # Two independent reasons to swap, and only the first used to be considered:
    # a *current* sqlite built without loadable-extension support is fine by
    # version and still unusable, so it returned here and left
    # ``ensure_loadable_extensions`` to raise the install hint forever -- even
    # once the user had followed it and installed the very build that fixes it.
    version_ok = _version_tuple(sqlite3.sqlite_version) >= MIN_SQLITE_VERSION
    if version_ok and (_extensions_checked or _loads_extensions(sqlite3)):
        return

    needed = ".".join(str(part) for part in MIN_SQLITE_VERSION)
    try:
        import pysqlite3
    except ImportError as exc:
        raise RuntimeError(
            _INSTALL_HINT.format(needed=needed, found=sqlite3.sqlite_version)
        ) from exc

    if _version_tuple(pysqlite3.sqlite_version) < MIN_SQLITE_VERSION:
        # Installed, but no newer. Naming the fallback's version rather than the
        # stdlib's is the actionable half: the user has the wheel and still
        # needs a different one.
        raise RuntimeError(
            _INSTALL_HINT.format(needed=needed, found=pysqlite3.sqlite_version)
        )

    if not _loads_extensions(pysqlite3):
        # New enough, but no better on the axis that sent us here. Swapping
        # would change nothing and would trade the precise "cannot load
        # extensions" message ``ensure_loadable_extensions`` is about to raise
        # for a version complaint that is not true.
        return

    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
    logger.info(
        "Replaced sqlite3 %s with pysqlite3 %s for sqlite-vec.",
        sqlite3.sqlite_version,
        pysqlite3.sqlite_version,
    )
    # Rebind the module global too, so the extension check below and any later
    # version check read the build callers will actually get from ``import
    # sqlite3``, not the stdlib one this module bound at its own import.
    sqlite3 = pysqlite3
    _extensions_checked = False


def ensure_loadable_extensions() -> None:
    """Fail early where sqlite3 cannot load an extension at all.

    The version check says nothing about this -- a current sqlite compiled
    without extension support is a distinct failure -- and sqlite-vec is unusable
    without it. Memoised, so the throwaway connection is opened at most once per
    interpreter.
    """
    global _extensions_checked

    if _extensions_checked:
        return

    conn = sqlite3.connect(":memory:")
    try:
        # CPython omits the method entirely when the build option is off;
        # some builds keep it and raise NotSupportedError on the call.
        conn.enable_load_extension(True)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.NotSupportedError) as exc:
        raise RuntimeError(_EXTENSION_HINT) from exc
    finally:
        conn.close()

    _extensions_checked = True
