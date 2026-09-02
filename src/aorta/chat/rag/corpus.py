"""What goes into a chat index, and the guard on what must never go into a published one.

The published index is not an opaque vector blob. ``SqliteVecStore`` persists
each chunk's text verbatim, so the artifact CI uploads is a redistribution of
the source it was built from. That is harmless for MIT-licensed public
``ROCm/aorta`` and an IP incident for anything else, which is why
:func:`assert_public_tree` exists and why the build restricts itself to files
*tracked by git in the public repository* rather than to whatever happens to be
sitting in the working directory.

Those are two independent guards on purpose. The remote check catches the
workflow being pointed at the wrong repository; the tracked-file filter catches
an internal reproducer or a customer bundle dropped into an otherwise correct
checkout, which no remote check can see.
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

#: The one repository whose source may be embedded into a published index.
#: Matched against the checkout's ``origin`` remote after normalisation, so both
#: the SSH and HTTPS spellings resolve.
PUBLIC_REPO = "ROCm/aorta"

#: Subpaths of the repository that make up the published corpus, in order. The
#: package itself plus the prose that explains it; tests and CI configuration
#: are deliberately absent -- they answer almost no user question and would
#: crowd real code out of a fixed-k MMR retrieval.
PUBLISHED_SUBPATHS = ("src/aorta", "docs", "README.md")

#: How long ``git`` gets before the guard gives up. A hung git is a failed
#: guard, and a failed guard must not become a passed one.
_GIT_TIMEOUT = 60


class PublicTreeError(RuntimeError):
    """The tree being indexed is not the public repository.

    Raised, never warned. A published index built over internal or customer
    source cannot be recalled once it is a release asset.
    """


@dataclass
class Corpus:
    """A resolved corpus: one base tree, some subpaths, and an optional allowlist.

    ``source`` metadata is always relative to :attr:`base`, so a corpus spanning
    ``src/aorta`` and ``docs`` yields paths a user recognises
    (``src/aorta/cli/chat.py``) rather than two colliding sets of tree-relative
    ones.
    """

    base: Path
    subpaths: tuple[str, ...] = (".",)
    #: Tracked-file allowlist, or ``None`` for "index whatever the walk finds".
    #: Populated by :func:`tracked_files` for a published build.
    allowed: frozenset[str] | None = None
    roots_label: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        scope = "tracked files only" if self.allowed is not None else "all files"
        return f"{self.base} [{', '.join(self.subpaths)}] ({scope})"


#: Hosts whose ``owner/name`` path this guard is willing to read.
_GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})

#: ``user@host:owner/name`` -- git's SCP-like syntax, which is not a URL and so
#: is invisible to ``urlsplit``. The negative lookahead keeps ``host:/path``
#: (a genuine URL missing its scheme) out.
_SCP_REMOTE = re.compile(r"^(?:[^@/]+@)?(?P<host>[^:/]+):(?!/)(?P<path>.+)$")


def _normalise_remote(url: str) -> str:
    """Reduce a git remote URL to ``owner/name``.

    Handles ``git@github.com:owner/name.git``, ``https://github.com/owner/name``
    and the ``ssh://`` form, because the guard has to hold whichever spelling
    the runner's checkout happens to use.

    The host is matched as a host rather than found anywhere in the string. A
    substring test reduced ``https://evil.example/github.com/ROCm/aorta`` to
    ``ROCm/aorta``, so any remote at all could satisfy the publishable-index
    guard by carrying the right text in its path.
    """
    text = url.strip().removesuffix(".git")
    scp = _SCP_REMOTE.match(text)
    if scp:
        host, path = scp.group("host"), scp.group("path")
    else:
        split = urllib.parse.urlsplit(text)
        host, path = split.hostname or "", split.path
    if host.lower() in _GITHUB_HOSTS:
        return path.strip("/")
    # Not a GitHub remote: return it whole, so the caller reports what it really
    # found rather than a fragment that could coincide with the expected slug.
    return text.strip("/")


def _git(root: Path, *args: str) -> str:
    """Run a read-only git command in ``root``, or raise :class:`PublicTreeError`."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=True,
        )
    except FileNotFoundError as exc:
        raise PublicTreeError(
            "git is not on PATH, so the public-tree guard cannot verify what is "
            "about to be indexed. Refusing to build a publishable index."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise PublicTreeError(f"git {' '.join(args)} timed out in {root}") from exc
    except subprocess.CalledProcessError as exc:
        raise PublicTreeError(
            f"git {' '.join(args)} failed in {root}: {exc.stderr.strip() or exc}"
        ) from exc
    return completed.stdout


def repo_root(path: Path) -> Path:
    """The git work-tree root containing ``path``."""
    return Path(_git(path, "rev-parse", "--show-toplevel").strip())


def head_sha(path: Path) -> str:
    """Full SHA of HEAD, or ``""`` outside a git tree."""
    try:
        return _git(path, "rev-parse", "HEAD").strip()
    except PublicTreeError:
        return ""


def head_tag(path: Path) -> str:
    """The ``vX.Y.Z`` tag on HEAD, or ``""`` when HEAD is not tagged."""
    try:
        return _git(path, "tag", "--points-at", "HEAD", "v*").strip().splitlines()[0]
    except (PublicTreeError, IndexError):
        return ""


def assert_public_tree(root: Path, *, expected: str = PUBLIC_REPO) -> None:
    """Refuse unless ``root`` is a checkout of the public repository.

    Raises:
        PublicTreeError: If ``root`` is not a git tree, has no ``origin``, or
            ``origin`` does not point at *expected*.
    """
    top = repo_root(root)
    remote = _normalise_remote(_git(top, "remote", "get-url", "origin"))
    if remote.lower() != expected.lower():
        raise PublicTreeError(
            f"refusing to build a publishable index from {top}: its origin "
            f"remote resolves to {remote!r}, not {expected!r}.\n"
            "The index embeds source text verbatim, so publishing one built "
            "over an internal or customer tree would republish that source."
        )
    logger.info("Public-tree guard passed: %s -> %s", top, remote)


def tracked_files(root: Path, subpaths: tuple[str, ...]) -> frozenset[str]:
    """Repo-relative paths git tracks under ``subpaths``.

    This is the guard that a remote check cannot make. An internal reproducer or
    a customer bundle copied into a correct checkout is untracked, so it is
    absent from this set and cannot reach the index -- whatever the working
    directory looks like at build time.
    """
    top = repo_root(root)
    output = _git(top, "ls-files", "-z", "--", *subpaths)
    return frozenset(entry for entry in output.split("\0") if entry)


def published_corpus(root: Path, *, subpaths: tuple[str, ...] = PUBLISHED_SUBPATHS) -> Corpus:
    """The corpus for a CI-published index: public tree, tracked files only."""
    top = repo_root(root)
    assert_public_tree(top)
    present = tuple(sub for sub in subpaths if (top / sub).exists())
    missing = sorted(set(subpaths) - set(present))
    if missing:
        # Not fatal -- `docs/` could legitimately be renamed -- but a corpus
        # quietly missing half its prose produces an index that looks fine and
        # answers worse, so it belongs in the log the build leaves behind.
        logger.warning("Published corpus subpath(s) absent, skipping: %s", ", ".join(missing))
    if not present:
        raise PublicTreeError(f"none of the published corpus subpaths exist under {top}")
    return Corpus(
        base=top,
        subpaths=present,
        allowed=tracked_files(top, present),
        roots_label=present,
    )


def local_corpus(path: str | Path) -> Corpus:
    """The corpus for a local build: one tree, no tracking requirement.

    The default is the installed ``aorta`` package, which is real code the user
    demonstrably has. No public-tree guard here: a local index is never
    published, and an air-gapped user indexing their own checkout is the case
    Decision 21b exists to serve.
    """
    base = Path(path).resolve()
    if not base.exists():
        raise FileNotFoundError(f"corpus path does not exist: {base}")
    return Corpus(base=base, subpaths=(".",), allowed=None, roots_label=(str(base),))


def load_corpus(corpus: Corpus) -> list[Document]:
    """Load every indexable document in ``corpus``, deduplicated by source path.

    Subpaths may nest (``src/aorta`` under ``.``), so the same file can be
    reached twice; indexing it twice would put two identical vectors in front of
    a fixed-k retrieval.
    """
    from aorta.chat.rag.indexer import load_documents

    seen: dict[str, Document] = {}
    for sub in corpus.subpaths:
        target = corpus.base / sub if sub != "." else corpus.base
        include = None if corpus.allowed is None else corpus.allowed.__contains__
        if target.is_file():
            # A single-file subpath (README.md) never reaches os.walk.
            rel = str(target.relative_to(corpus.base))
            if include is not None and not include(rel):
                continue
            docs = load_documents(
                target.parent, include={rel}.__contains__, relative_to=corpus.base
            )
        else:
            docs = load_documents(target, include=include, relative_to=corpus.base)
        for doc in docs:
            seen.setdefault(doc.metadata["source"], doc)
    return [seen[source] for source in sorted(seen)]


def corpus_digest(
    documents: list[Document],
    *,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """A digest that changes exactly when a rebuild would produce a different index.

    Covers the corpus content and the parameters that turn it into vectors, so
    ``nightly.yml`` can skip re-uploading tens of megabytes on a night when only
    a test changed. Deliberately *not* keyed on the git SHA: most commits touch
    nothing indexable, and keying on the SHA would defeat the skip entirely.
    """
    digest = hashlib.sha256()
    digest.update(f"v1\0{embedding_model}\0{chunk_size}\0{chunk_overlap}\0".encode())
    for doc in sorted(documents, key=lambda d: d.metadata["source"]):
        body = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        digest.update(f"{doc.metadata['source']}\0{body}\0".encode())
    return digest.hexdigest()


__all__ = [
    "PUBLIC_REPO",
    "PUBLISHED_SUBPATHS",
    "Corpus",
    "PublicTreeError",
    "assert_public_tree",
    "corpus_digest",
    "head_sha",
    "head_tag",
    "load_corpus",
    "local_corpus",
    "published_corpus",
    "repo_root",
    "tracked_files",
]
