"""Which directories of a codebase are worth reading, shared by rag consumers.

Both the vector indexer and the repo map walk the same tree and want the same
answer. They used to carry separate hard-coded lists, which drifted: the
indexer's missed `buck-out` and `.testvenv`, and the repo map's missed those plus
`experiments` and `misc`. One list, one place.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: Directories that carry no source worth answering questions about. Indexing
#: them is not merely wasteful: build output and vendored dependencies crowd out
#: real code in retrieval, because MMR sees thousands of near-duplicate chunks.
#: AORTA's own tree contributes `buck-out` (2.9k files) and `build`.
SKIP_DIRS = frozenset({
    # Dependencies and vendored trees
    "node_modules", "site-packages", "third-party", "third_party", "vendor",
    # Build and packaging output. Hidden ones such as `.eggs` need no entry --
    # the hidden-directory rule already covers them.
    "build", "dist", "buck-out", "target",
    # Caches and coverage
    "__pycache__", "htmlcov",
    # Environments whose name does not start with a dot
    "venv",
    # Deliberate editorial exclusions, kept from the original indexer list
    "experiments", "misc",
})

#: Hidden directories are tooling state, so they are skipped wholesale. These
#: are the exceptions: CI configuration is genuinely informative about how a
#: codebase is built and tested.
HIDDEN_DIRS_TO_KEEP = frozenset({".github"})


def is_virtualenv(path: Path) -> bool:
    """A directory holding ``pyvenv.cfg`` is a virtualenv root.

    Catches environments whatever they are called, which name matching alone
    does not: `.testvenv`, `env311`, and so on.
    """
    return (path / "pyvenv.cfg").is_file()


def should_skip_dir(path: Path) -> bool:
    """True when *path* is a directory not worth descending into."""
    name = path.name
    if name in SKIP_DIRS:
        return True
    if name.startswith(".") and name not in HIDDEN_DIRS_TO_KEEP:
        return True
    if is_virtualenv(path):
        logger.debug("Skipping virtualenv: %s", path)
        return True
    return False


def is_symlink(path: Path) -> bool:
    """True for a walked entry that must not be followed.

    The containment guards in this package vet the *entry point* of a traversal
    -- ``resolve_within`` resolves the path a caller asked for, so a link whose
    target sits outside the root is refused there. Neither that check nor
    :func:`prune_dirnames` re-resolves the individual files a walk then reaches,
    so a link found underneath an allowed root is read at wherever its target
    lives. With ``embedding_provider = "remote"`` that content is uploaded, and
    for a published index it is embedded verbatim despite the tracked-path
    allowlist passing on the link's own path.

    An unreadable entry counts as one to skip: the walk has nothing to gain from
    a path it cannot stat.
    """
    try:
        return path.is_symlink()
    except OSError:
        return True


def prune_dirnames(root: Path, dirnames: list[str]) -> None:
    """Drop unwanted directories from *dirnames*, editing it in place.

    In-place mutation is how ``os.walk`` is told not to descend, which matters
    for more than tidiness: a single ``.venv`` can hold tens of thousands of
    files, and filtering after the walk still pays to visit and stat every one.
    """
    dirnames[:] = sorted(
        name for name in dirnames if not should_skip_dir(root / name)
    )
