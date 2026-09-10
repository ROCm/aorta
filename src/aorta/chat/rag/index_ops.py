"""Build, fetch and side-load the chat index. No Click; ``cli/chat.py`` renders.

Three ways to get an index, in the order most users will want them:

* :func:`fetch_index` -- download the CI-published artifact matching the
  installed aorta. The default path, and the one that removes the indexer from
  the user's problem entirely.
* :func:`build_index` -- build from the corpus on disk. The air-gapped path, the
  developer path, and what CI itself runs.
* :func:`side_load` -- adopt an index staged by hand (Decision 21b). Index only;
  the embedding model is a documented pre-seed procedure rather than a second
  published artifact, and ``rag/embeddings/fastembed_bge.py`` is where that
  procedure gets printed.

**Why fetch resolves by installed version rather than always taking the newest.**
The two populations want different things. Someone who ran ``pip install
amd-aorta`` and got a released wheel wants the index for *that* release; a
nightly index would describe code they do not have. Someone on a
setuptools_scm dev version is on ``main`` and wants the rolling asset. Decision
18a plus "Release vs nightly": exact release version takes that release's asset,
a ``.devN+g<sha>`` version takes the rolling one and says how far off it is.
"""

from __future__ import annotations

import dataclasses
import http.client
import json
import logging
import os
import re
import shlex
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aorta.chat.config import settings
from aorta.chat.rag import corpus as corpus_mod
from aorta.chat.rag import manifest as manifest_mod
from aorta.chat.rag.embeddings.factory import get_provider

logger = logging.getLogger(__name__)

#: Where release assets live. Overridable so tests never touch the network and
#: so a customer behind a mirror can point at their own copy.
RELEASE_BASE_URL = "https://github.com/ROCm/aorta/releases/download"

#: Environment override for the above, for an internal mirror or an air-gapped
#: artifact server.
BASE_URL_ENV = "AORTA_CHAT_INDEX_BASE_URL"

#: One asset name for both channels; the tag distinguishes them. Publishing the
#: same name under every tag keeps the download URL a single template and keeps
#: a side-loaded file recognisable.
ASSET_NAME = "aorta-chat-index.sqlite"

#: Tag carrying the rolling asset built from ``main``. Shares ``nightly.yml``'s
#: existing pre-release rather than inventing a second rolling tag to prune.
ROLLING_TAG = "dev-wheels"

#: A version with no suffix: an exact release, so an exact release asset.
_RELEASE_VERSION = re.compile(r"\A\d+\.\d+\.\d+\Z")

#: setuptools_scm's local segment, e.g. ``0.2.2.dev122+g45edc3d.d20260810``.
_DEV_LOCAL = re.compile(r"\.dev(?P<distance>\d+)(?:\+g(?P<sha>[0-9a-f]{7,40}))?")

#: Budget for connecting and for the response headers. ``urlopen`` returns as
#: soon as the headers are in, so this covers everything up to the first byte
#: of the body -- which is the only phase an unreachable or blackholed host ever
#: reaches. One 300 s budget for both phases is what turned "this node cannot
#: see the release host" into five silent minutes on a 1 KB sidecar.
_CONNECT_TIMEOUT = 30

#: Budget for reading the body, once a server has answered. Kept generous
#: because the index is tens of megabytes and the link may be slow. It is a
#: per-``recv`` timeout rather than a total, so a transfer that keeps flowing
#: never approaches it.
_READ_TIMEOUT = 300

#: Read from the socket in blocks this size, so progress is reported against a
#: transfer rather than after it.
_DOWNLOAD_BLOCK = 256 * 1024

#: How much has to arrive between progress lines. Small enough that a slow link
#: still looks alive, large enough that a fast one does not scroll.
_PROGRESS_STEP = 4 * 1024 * 1024

#: Chunks embedded per write. Mirrors ``indexer._WRITE_BATCH``'s reasoning: a
#: real tree splits into ~15,000 chunks and embedding them in one call holds
#: every vector in memory at once.
_WRITE_BATCH = 500

#: Appended to every unreachable-host error. A node with no egress is the
#: expected case, not an anomaly (Decision 21b), and the manifest sidecar is
#: fetched first -- so the advice has to be on that error too, not only on the
#: index download's.
_NO_EGRESS_HINT = (
    "\nIf this node has no egress, stage the index and its .manifest.json "
    "elsewhere and side-load them:\n"
    "  aorta chat index fetch --from <file>"
)


class IndexFetchError(RuntimeError):
    """An index could not be downloaded, verified, or written."""


class IndexOverwriteError(IndexFetchError):
    """Refusing to replace an existing index that this command cannot give back.

    A subclass rather than a sibling so ``cli/chat.py``'s known-exception list
    needs no change to surface it. That list exists because these messages are
    the deliverable -- a refusal the user cannot act on gets worked around --
    and it already renders :class:`IndexFetchError` verbatim.
    """


@dataclass(frozen=True)
class IndexSource:
    """A resolved place to fetch the index from, and what to say about it."""

    tag: str
    channel: str
    base_url: str = RELEASE_BASE_URL
    #: Non-fatal things the user should hear, e.g. the SHA delta on ``main``.
    notes: tuple[str, ...] = ()

    @property
    def index_url(self) -> str:
        return f"{self.base_url}/{self.tag}/{ASSET_NAME}"

    @property
    def manifest_url(self) -> str:
        return self.index_url + manifest_mod.MANIFEST_SUFFIX

    @property
    def checksum_url(self) -> str:
        return self.index_url + manifest_mod.CHECKSUM_SUFFIX

    def describe(self) -> str:
        return f"{self.channel} channel, tag {self.tag}"


@dataclass
class BuildResult:
    """What a local or CI build produced."""

    index_path: Path
    manifest: manifest_mod.Manifest
    file_count: int
    chunk_count: int
    size_bytes: int
    seconds: float = 0.0
    corpus: str = ""


@dataclass
class FetchResult:
    """What a fetch or side-load installed."""

    index_path: Path
    manifest: manifest_mod.Manifest
    source: str
    warnings: list[str] = field(default_factory=list)
    #: What was known *before* the first request -- which asset was resolved and
    #: why. Separate from ``warnings`` because the caller has to be able to say
    #: it up front: these are the lines that explain a wait, and they used to be
    #: printed only once the wait had ended successfully.
    notes: list[str] = field(default_factory=list)
    #: What this install replaced, for a refresh that went ahead without asking.
    changes: list[str] = field(default_factory=list)
    #: Set when the published asset was already installed, so nothing was
    #: transferred and nothing was replaced.
    up_to_date: bool = False


def installed_version() -> str:
    """The installed ``amd-aorta`` version, or ``""`` when it cannot be read."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("amd-aorta")
    except PackageNotFoundError:  # pragma: no cover - running from a raw tree
        return ""


def base_url() -> str:
    """Release-asset base URL, honouring :data:`BASE_URL_ENV`."""
    return os.environ.get(BASE_URL_ENV, "").strip().rstrip("/") or RELEASE_BASE_URL


def resolve_source(version: str | None = None, installed: str | None = None) -> IndexSource:
    """Pick the asset to fetch (Decision 18a).

    Args:
        version: Explicit ``--version``. An exact ``X.Y.Z`` becomes that
            release's tag; anything else is passed through as a tag, so
            ``--version dev-wheels`` reaches the rolling asset by name.
        installed: Override the installed version; for tests.

    Returns:
        The resolved source, carrying any warning the caller should print.
    """
    if version:
        wanted = version.strip().lstrip("v")
        if _RELEASE_VERSION.match(wanted):
            return IndexSource(tag=f"v{wanted}", channel="release", base_url=base_url())
        return IndexSource(tag=version.strip(), channel="explicit", base_url=base_url())

    current = installed if installed is not None else installed_version()
    if _RELEASE_VERSION.match(current):
        return IndexSource(tag=f"v{current}", channel="release", base_url=base_url())

    notes = [
        f"This install is {current or 'an unreleased build'}, not a tagged release, so "
        f"the rolling '{ROLLING_TAG}' index built from main is the closest match."
    ]
    match = _DEV_LOCAL.search(current)
    if match:
        distance = match.group("distance")
        sha = match.group("sha")
        notes.append(
            f"It is {distance} commit(s) past the last release"
            + (f" at {sha}" if sha else "")
            + "; the manifest check below reports how far the index is from it."
        )
    return IndexSource(
        tag=ROLLING_TAG,
        channel="main (rolling)",
        base_url=base_url(),
        notes=tuple(notes),
    )


#: Prefix of the staging directory every install path writes through. Created
#: beside the destination so the final move is a rename on one filesystem.
_STAGING_PREFIX = ".aorta-index-"


# ── who built the index that is already there ─────────────────────────────

#: A CI-published index. ``corpus_roots`` holds the tracked subpaths
#: ``published_corpus`` was given, so every entry is repository-relative.
PROVENANCE_PUBLISHED = "published"

#: Built on this machine. ``local_corpus`` records the one absolute path it
#: walked, so a single absolute root is the signature.
PROVENANCE_LOCAL = "local"

#: A manifest that records no roots at all, from a builder predating the field.
PROVENANCE_UNKNOWN = "unknown"

#: A manifest whose ``corpus_roots`` is *present* and unusable, so nothing can
#: be concluded from it. Deliberately distinct from :data:`PROVENANCE_UNKNOWN`:
#: absent roots are a layout, an unusable value is a broken sidecar, and the
#: write guards treat the two differently.
PROVENANCE_INVALID = "invalid"


def _corpus_roots(manifest: manifest_mod.Manifest) -> list[str] | None:
    """``corpus_roots`` as a list of strings, or ``None`` when it is unusable.

    A hand-written or hand-edited sidecar -- the side-load path's ordinary case
    -- used to reach here carrying a scalar. Iterating a string yields
    characters and ``Path("/").is_absolute()`` is true, so ``"src/aorta"``
    classified as a local build and ``"docs"`` as a published one, both from
    nothing at all; ``[42]`` raised ``TypeError`` straight past the CLI's
    error guard. A field that decides whether a destructive overwrite is
    refused has to be validated rather than coerced.

    #463 added a field-type check to ``Manifest.from_dict``, which closes that
    route for every reader that parses strictly -- meaning everything that
    loads an index. It does not close it here: :func:`_read_destination` parses
    a write target with ``strict=False`` on purpose, because "this sidecar
    cannot be classified" and "this path has no manifest I can read" are
    different things to tell someone about a file that is about to be
    overwritten, and only this function can tell them apart. So a scalar still
    arrives, from the one caller that has to see it, and this still fails
    closed -- the cost of guessing is an overwrite.

    One reader, shared by the guards and by ``index status``'s payload, so the
    verdict and the reported value cannot disagree about what was readable.
    """
    roots = manifest.corpus_roots
    if not isinstance(roots, (list, tuple)):
        return None
    if not all(isinstance(root, str) for root in roots):
        return None
    return list(roots)


def _roots_provenance(roots: list[str] | tuple[str, ...]) -> str:
    """Classify a set of corpus roots by shape.

    ``published_corpus`` labels its roots with the repository-relative subpaths
    it was handed (``src/aorta``, ``docs``, ``README.md``); ``local_corpus``
    labels its single root with the absolute path it resolved. Reading the
    shape rather than matching the published list means renaming a published
    subpath does not silently reclassify every index.
    """
    if not roots:
        return PROVENANCE_UNKNOWN
    if any(Path(root).is_absolute() for root in roots):
        return PROVENANCE_LOCAL
    return PROVENANCE_PUBLISHED


def index_provenance(manifest: manifest_mod.Manifest) -> str:
    """Whether an index was published or built here, read off its manifest.

    Inferred rather than stored, because ``corpus_roots`` already distinguishes
    the two without a manifest schema bump. A recorded value that is not a list
    of paths yields :data:`PROVENANCE_INVALID` rather than a classification
    derived from iterating it -- see :func:`_corpus_roots`.
    """
    roots = _corpus_roots(manifest)
    if roots is None:
        return PROVENANCE_INVALID
    return _roots_provenance(roots)


def corpus_provenance(corpus: corpus_mod.Corpus) -> str:
    """The provenance an index built from ``corpus`` would record.

    The same rule applied to the corpus rather than to a finished manifest, so
    a guard can compare what is about to be built against what is already
    there instead of only inspecting the destination.
    """
    return _roots_provenance(corpus.roots_label or corpus.subpaths)


def _read_destination(index_path: str | Path) -> tuple[manifest_mod.Manifest | None, str]:
    """The sidecar at a write target, and why it could not be read.

    Three states rather than two. ``(None, "")`` is nothing there at all;
    ``(None, reason)`` is a path that exists with no manifest this build can
    read; ``(manifest, "")`` is an index that can be classified.

    :func:`_local_manifest` collapses the first two, which is right for the
    readers that only want a manifest and wrong for the write guards, where
    "there is nothing to overwrite" and "there is something unidentifiable to
    overwrite" are opposite verdicts.
    """
    target = Path(index_path)
    if not target.exists():
        # ``is_symlink`` as well, because ``exists()`` *follows* the link and so
        # answers about the target rather than the path the user named: a
        # dangling symlink reported "nothing here", the guards read that as a
        # first install, and ``replace()`` then consumed the link without
        # asking. A symlink is a deliberate object -- someone pointed it
        # somewhere -- and a broken one is the case where its owner is least
        # likely to notice it went. It is also unidentifiable by definition,
        # which is the state this function exists to distinguish, so it takes
        # the same refusal rather than a special one.
        if target.is_symlink():
            return None, f"{target} is a symlink to {os.readlink(target)}, which does not exist"
        return None, ""
    try:
        # ``strict=False``: the three states above are the whole product of this
        # function, and #463's field-type check collapses two of them. A sidecar
        # recording ``corpus_roots`` as a scalar is a *classification* failure,
        # answered below by ``index_provenance`` returning ``invalid`` and by
        # the guards' unclassifiable refusal; parsed strictly it becomes an
        # unreadable manifest instead, indistinguishable from a mistyped
        # ``--output`` at a file this tool never wrote. Both refuse, so nothing
        # was unsafe -- but the refusals say different things, only one of them
        # is true, and the build guard's exemption for an index this install
        # already refuses cannot be evaluated at all once the manifest it turns
        # on has stopped being readable. That exemption is what keeps `doctor`
        # says rebuild -> `build` refuses from being a dead end.
        #
        # Strictness is not lost, it is relocated to the reader that needs it:
        # everything that *loads* an index still parses strictly, and the values
        # this tolerance admits reach only ``index_provenance``, which validates
        # ``corpus_roots`` itself, and ``_unusable_reasons``/
        # ``_describe_tolerantly``, which are built to survive a field they
        # cannot render.
        #
        # ``advise=False``: this reason is embedded in a refusal that ends with
        # its own ``--force`` escape, so the parser's "Replace it with 'aorta
        # chat index fetch'" arrives inside the message refusing that very
        # fetch. Same loop as the published case in :func:`_parse_manifest`,
        # reached from the local side -- the caller knowing the origin is again
        # the one that should write the remedy.
        return manifest_mod.read_manifest(target, advise=False, strict=False), ""
    except (manifest_mod.ManifestError, UnicodeDecodeError, OSError) as exc:
        # ``read_manifest`` wraps its parse failures and not its *read* ones, so
        # a sidecar that is not UTF-8 -- a truncated multi-byte write, or a file
        # something else owns -- came out of here as UnicodeDecodeError and past
        # every caller. That is the same "cannot read it" answer as a bad parse,
        # arriving one layer lower, and this is the function whose whole job is
        # to name it. OSError for the same reason on a permission or I/O fault.
        logger.debug("No usable manifest beside %s: %s", target, exc)
        return None, str(exc) or type(exc).__name__


def _local_manifest(index_path: str | Path) -> manifest_mod.Manifest | None:
    """The installed index's sidecar, or ``None`` when there is not a usable one.

    Absent, unreadable and unparseable all collapse to ``None`` deliberately.
    Every caller here is *reporting* on the local index, and none of those
    reports is improved by a traceback out of a sidecar that is already broken
    -- the load path is where a bad manifest has to be fatal. The write guards
    use :func:`_read_destination` instead, because for them the collapse is the
    bug rather than the convenience.
    """
    return _read_destination(index_path)[0]


def _unusable_reasons(target: Path) -> list[str]:
    """Why this install cannot use the index at ``target``; empty when it can.

    One reader for "is what is on disk actually usable", shared by the build
    guard and by ``fetch``'s up-to-date shortcut, so a rebuild that is exempted
    and a refresh that is skipped cannot disagree about the same file.
    :func:`check_index` opens the store, so this is the question the first
    query asks rather than a second opinion derived from the manifest.

    Never raises, and the catch is deliberately wide. ``ManifestError`` alone
    was too narrow: local manifests are *deliberately* untyped, so a sidecar
    recording ``embedding_identity: 42`` makes ``check_index`` raise
    ``AttributeError`` while rendering the identity, and a non-string
    ``aorta_sha`` raises ``TypeError`` while slicing it -- both straight past a
    ``ManifestError`` handler and out of the write guard as a traceback. A
    malformed field is a reason this install cannot use the index, which is
    exactly what this function returns; it is not a reason to crash the guard
    that asked. Both callers have also read the manifest separately, so a
    sidecar that changed in between must not become a traceback either.
    """
    try:
        # Read the way :func:`_read_destination` reads, and handed to
        # ``check_index`` so it is not read twice. Strictly, a sidecar with one
        # wrong-typed field raises here and every such index reports "unusable"
        # -- which is the answer that *exempts* it from the build guard, so the
        # guard would have stopped refusing anything with a hand-edited
        # manifest. The question being asked is whether this install could
        # query the index, and one bad ``corpus_roots`` over a matching store
        # does not stop it.
        manifest = manifest_mod.read_manifest(target, strict=False)
        return list(check_index(target, strict=False, manifest=manifest).refusals)
    except manifest_mod.ManifestError as exc:
        return [str(exc)]
    except Exception as exc:  # noqa: BLE001 - a broken sidecar is an answer, not a crash
        logger.debug("Could not judge the index at %s: %r", target, exc)
        return [f"its manifest could not be interpreted: {type(exc).__name__}: {exc}"]


def _pasteable(command: str, dest: Path, corpus: corpus_mod.Corpus | None = None) -> str:
    """A refusal's remedy, carrying the flags that chose the index it is about.

    A remedy is only advice if pasting it acts on the index that was refused.
    These commands default ``--output`` to the configured cache and their
    corpus to the installed package, so a refusal raised over
    ``--output /tmp/scratch.sqlite`` printed a bare ``... --force`` that named
    a *different* index -- the user's real one, which they had not asked about
    and which the paste would then overwrite while leaving the refusal
    standing.

    Only the flags that differ from what the bare command would resolve are
    appended, so the common refusal stays the short line it was.
    """
    return command + _target_flags(dest, corpus)


def _target_flags(dest: Path, corpus: corpus_mod.Corpus | None = None) -> str:
    """The flags naming this destination, or ``""`` when they are the defaults.

    Separate from :func:`_pasteable` because the 404 remedy prints three
    commands in an aligned block and needs the suffix rather than each whole
    line.
    """
    parts = []
    if corpus is not None and corpus.base != settings.aorta_root:
        parts.append(f"--path {shlex.quote(str(corpus.base))}")
    if dest != settings.index_file:
        parts.append(f"--output {shlex.quote(str(dest))}")
    return "".join(f" {part}" for part in parts)


def _unclassifiable_error(target: Path, roots: object, command: str) -> IndexOverwriteError:
    """The refusal for a manifest whose ``corpus_roots`` cannot be read.

    Shared by both write guards, because neither of them may guess: the two
    provenances are protected differently and only one of them is something
    the network can give back. Names the type rather than echoing the value,
    which is third-party text of unbounded size.
    """
    return IndexOverwriteError(
        f"the index at {target} has a manifest this build cannot classify: "
        f"corpus_roots is a {type(roots).__name__}, not a list of paths, so "
        "whether the index was downloaded or built here cannot be read.\n"
        "A downloaded index costs a download to replace; a local build may not "
        "be reproducible at all. Refusing rather than guessing which one is "
        "there.\n"
        f"Overwrite it deliberately:  {command} --force"
    )


def _unidentifiable_error(target: Path, reason: str, command: str) -> IndexOverwriteError:
    """The refusal for an existing destination with no manifest to read.

    Every write path here produces its sidecars, so a path that exists without
    one is not an index this tool finished installing. It is a mistyped
    ``--output``, a file something else owns, or an index whose sidecars were
    lost -- and the first two are the ones an overwrite cannot give back.

    Names both, because the caller is the only one who can tell them apart, and
    the escape is one flag rather than a second diagnosis. The escape is worded
    as a deliberate replacement rather than a safe one: this branch covers an
    unrelated file *and* a local build whose sidecars were lost, and the second
    of those may be irreproducible, so nothing here can promise the overwrite
    costs nothing.
    """
    return IndexOverwriteError(
        f"there is already a file at {target}, and no manifest beside it that "
        f"this install can read ({reason}).\n"
        "A first install writes to a path that does not exist yet. This one "
        "exists, so something is there to lose, and without a manifest there "
        "is no way to tell an index from a file this tool never wrote.\n"
        "If the path is a typo, correcting it is the fix.\n"
        "If you know what is there and mean to replace it, say so:  "
        f"{command} --force"
    )


def _refuse_if_locally_built(dest: Path, *, force: bool, command: str) -> None:
    """Stop an incoming index from silently discarding one built on this machine.

    The two directions are not symmetric, which is why this is not a blanket
    guard. A *fetched* index is trivially re-fetchable, so replacing one costs
    a download; a *locally built* one may not be reproducible at all, because
    the tree it indexed may have moved or the node may have no egress to
    rebuild the weights. So the incoming-index paths (``fetch`` and its
    ``--from`` side-load, which would otherwise be the accidental way around
    this) guard against overwriting a local build, and only that.

    A manifest recording *no* roots is not protected: absent is a layout, from
    a builder predating the field, and refusing every such index would refuse
    every pre-field install's routine refresh. A manifest recording roots that
    cannot be read is refused -- that is a broken sidecar, not an old one, and
    it is the one case where guessing could discard the unrecoverable side.

    A destination with *no* readable manifest is refused on the same reasoning,
    one step earlier. Absent and unreadable used to collapse into "nothing
    there", so a mistyped ``--index`` at an existing file, or an index whose
    sidecar had been truncated, was overwritten without a word -- while the
    strictly better-informed case of a sidecar that reads but classifies badly
    was refused.
    """
    if force:
        return
    # Carries a non-default --output into every remedy below, so the line the
    # user pastes acts on the index they were refused rather than on the cache.
    remedy = _pasteable(command, dest)
    local, unreadable = _read_destination(dest)
    if unreadable:
        raise _unidentifiable_error(dest, unreadable, remedy)
    if local is None:
        return
    provenance = index_provenance(local)
    if provenance == PROVENANCE_INVALID:
        raise _unclassifiable_error(dest, local.corpus_roots, remedy)
    if provenance != PROVENANCE_LOCAL:
        return
    # Formatted from the validated reader rather than the raw attribute. The
    # branch above is what makes this a list of strings, and rendering the
    # field the guard just classified through a second, unchecked path is the
    # shape of the bug that validation exists to close.
    roots = _corpus_roots(local) or []
    # ``describe()`` and not the raw fields, but tolerantly: it slices
    # ``aorta_sha`` and formats ``dimensions``, and a *local* sidecar is
    # deliberately untyped, so `aorta_sha: 42` raised ``TypeError`` out of the
    # refusal itself. The classification above is unaffected -- it reads
    # ``corpus_roots``, which was validated -- so the index really is a local
    # build and really must be refused; only the courtesy line describing it
    # was unrenderable, and losing the refusal because its subtitle would not
    # format is the guard failing open over cosmetics.
    raise IndexOverwriteError(
        f"the index at {dest} was built on this machine, not downloaded.\n"
        f"  built as  {_describe_tolerantly(local)}\n"
        f"  corpus    {', '.join(roots)}\n"
        "Replacing it with the published index discards a build the network "
        "cannot give back -- the tree it indexed may have moved, and rebuilding "
        "needs the embedding weights again.\n"
        f"Pass --force to overwrite it:  {remedy} --force"
    )


def _describe_tolerantly(manifest: manifest_mod.Manifest) -> str:
    """``Manifest.describe()``, or a note that the manifest cannot be rendered.

    The manifests reaching the write guards are local sidecars, which are
    deliberately parsed without type checking so that the reports whose job is
    to say what is wrong with one can still read it. That tolerance has to
    extend to everything downstream of the read, and ``describe()`` is not
    written for it -- it slices and formats. This is the seam that keeps
    reappearing: the *guard's* decision never depends on these fields, so a
    field it does not consult must not be able to take the refusal down with
    it.
    """
    try:
        return manifest.describe()
    except Exception as exc:  # noqa: BLE001 - a summary is not worth a traceback
        logger.debug("Could not describe the manifest: %r", exc)
        return f"(its manifest cannot be summarised: {type(exc).__name__})"


def _refuse_if_published(target: Path, corpus: corpus_mod.Corpus, *, force: bool) -> None:
    """Stop a *narrower* build from silently downgrading a published index.

    The mirror of :func:`_refuse_if_locally_built`, and the mechanism behind
    the bad advice this guard exists to catch: ``doctor`` used to suggest a
    bare ``index build`` as a cache pre-warm, which defaults ``--output`` to
    the same path and its corpus to ``local_corpus`` -- so it replaced a
    published index covering ``src/aorta``, ``docs`` and ``README.md`` with one
    covering ``src/aorta`` alone, and said nothing.

    Three things are therefore exempt, and each of them is a case where the
    guard would refuse something that loses nothing:

    * **A build whose own corpus is the published one.** ``--public-only``
      produces the same shape it would be replacing, so this is a refresh, not
      a downgrade. This is what keeps ``nightly.yml`` and ``release.yml``
      running -- a guard that blocked them would stop the published index
      updating at all, which is worse than the defect being guarded against.
      It exempts the *provenance* question only; the manifest-less-destination
      refusal below is checked first and applies to ``--public-only`` too.
    * **An index this install cannot use.** For an embedding-identity
      mismatch, the manifest refusal and ``doctor`` both name ``index build``
      as the remedy, and it is the right one. Demanding ``--force`` on top
      would compose two individually-correct behaviours into a dead end --
      refused, told to rebuild, refused again -- on a user who is already
      stuck. Vectors that are not comparable to this install's queries cannot
      answer anything, and neither can a store that will not open, so in
      either case there is nothing to protect.

      Asked through :func:`check_index`, which reads the *store* and not only
      the manifest. Asking ``_validate_against_provider`` instead -- the
      manifest alone -- made this exemption narrower than the advice it exists
      to keep followable: an index whose ``.sqlite`` cannot be opened at all
      has an entirely valid manifest, so it was refused a rebuild while every
      reader that touches the file says to rebuild it. One reader for "can
      this install use what is there", shared with the load path.
    * **A destination with no published manifest**, which includes every first
      build and every local-over-local rebuild.

    A destination whose ``corpus_roots`` is present but unreadable is refused
    instead, once the exemptions above have not applied. A destination that
    exists with no readable manifest at all is refused *before any of them* --
    a mistyped ``--output`` reached ``replace()`` unopposed, because "no
    manifest" and "no file" were the same answer here. Ahead of the
    unusable-index exemption because that exemption's premise is that an index
    is what is there, and the whole point of this case is that nothing
    establishes that; ahead of ``--public-only`` because a typo is a typo on
    the CI path too, and the CI path never meets it (fresh workspace, no
    restored ``index-out/``).

    A *stale* index needs no exemption and deliberately does not get one.
    Source drift is a warning rather than a refusal, so it does not reach the
    branch below -- and on a remote embedding provider, where ``doctor`` names
    ``index build`` for drift because a fetch would land an asset that provider
    refuses, the index in front of the guard is already exempt for a different
    reason: either it is a local build (classified ``local``, and a local index
    is never protected from ``build``), or it is the published asset, which a
    remote provider refuses on its model and collection. Both paths permit the
    rebuild without widening anything. On a *local* provider a drift warning
    names ``index fetch``, and refusing a narrowing build there is the whole
    point of this guard.
    """
    if force:
        return
    # Both remedies below name a destination, and the build one also names a
    # corpus: this command defaults them to the cache and the installed
    # package, so a refusal over an explicit --path/--output that printed the
    # bare form sent the user at an index that was never in question.
    rebuild = _pasteable("aorta chat index build", target, corpus)
    refresh = _pasteable("aorta chat index fetch", target)
    local, unreadable = _read_destination(target)
    if unreadable:
        # Ahead of *every* exemption, including --public-only, and that ordering
        # is the fix for a hole the first version of it had: the CI exemption
        # returned before the destination was looked at, so
        # `build --public-only --output ~/notes.txt` replaced that file without
        # a word -- while the documented rule said any manifest-less path is
        # refused. It also sits ahead of the unusable-index exemption below,
        # which presumes an index is what is there; an unreadable manifest is
        # the case where that presumption is what is missing, so a mistyped
        # --output would otherwise be exempted for being unreadable enough.
        #
        # This costs the published build nothing. `nightly.yml` and
        # `release.yml` run on `ubuntu-latest` with a fresh workspace and never
        # restore `index-out/`, so the destination does not exist on their first
        # write and carries complete sidecars on every later one.
        raise _unidentifiable_error(target, unreadable, rebuild)
    if corpus_provenance(corpus) == PROVENANCE_PUBLISHED:
        return
    if local is None:
        return
    provenance = index_provenance(local)
    if provenance not in (PROVENANCE_PUBLISHED, PROVENANCE_INVALID):
        return
    unusable = _unusable_reasons(target)
    if unusable:
        # Ahead of both refusals, so an unusable index is exempt whether or not
        # its manifest is readable; either refusal would otherwise compose two
        # individually-correct behaviours into a dead end.
        #
        # Ahead of the unreadable-provenance refusal specifically, which looks
        # like a hole and is not: an unreadable manifest is either a published
        # index or a local one, and an *unusable* index reaches the same verdict
        # down both branches -- a refused published index is exempt by the rule
        # above, and a local index is never protected from `build` at all (the
        # early return above). So returning here is not a guess about which one
        # is on disk; it is the answer both possibilities give.
        logger.info(
            "The index at %s is not usable by this install, so rebuilding it "
            "loses nothing: %s",
            target,
            "; ".join(unusable),
        )
        return
    if provenance == PROVENANCE_INVALID:
        raise _unclassifiable_error(target, local.corpus_roots, rebuild)
    roots = _corpus_roots(local) or []
    raise IndexOverwriteError(
        f"the index at {target} covers the published corpus, and this build "
        "would not.\n"
        f"  there now  {local.describe()}\n"
        f"             corpus {', '.join(roots)}\n"
        f"  building   corpus {corpus.describe()}\n"
        "That replaces it with an index over a different corpus -- the two are "
        "above -- and one with no public-tree provenance, since only a "
        "--public-only build records the published root set. The default build "
        "corpus is the installed package alone, so it typically drops 'docs/' "
        "and 'README.md' as well.\n"
        f"Refresh the published one instead:  {refresh}\n"
        f"Or pass --force to build over it:   {rebuild} --force"
    )


# ── installing ────────────────────────────────────────────────────────────


def _install_staged(staged: Path, dest: Path) -> list[str]:
    """Move a finished index into place, keeping this machine's own collections.

    Every path here -- build, fetch, side-load -- produces the whole index
    somewhere else and then makes it live in one rename. That ordering is the
    crash safety: an interruption before the rename leaves the previous index
    and its sidecars exactly as they were, and one after it leaves an index
    whose sidecars have not caught up yet, which the load-time contents check
    refuses rather than answers from. What must never exist is a destination
    that is wrong but self-consistent, because nothing downstream can detect it.

    Sidecars are therefore the caller's *last* step, after this returns.
    """
    # Imported here, not at module scope: ``retriever`` pulls in langchain, and
    # this module is reachable from ``aorta --help``.
    from aorta.chat.rag.retriever import carry_over_collections

    carried = carry_over_collections(dest, staged)
    staged.replace(dest)
    return carried


# ── building ──────────────────────────────────────────────────────────────


def build_index(
    corpus: corpus_mod.Corpus | None = None,
    index_path: str | Path | None = None,
    *,
    force: bool = False,
) -> BuildResult:
    """Build an index plus its manifest from ``corpus``.

    Args:
        corpus: What to index. Defaults to the local corpus at
            ``settings.aorta_path``; CI passes
            :func:`~aorta.chat.rag.corpus.published_corpus`, whose tracked-file
            allowlist is the hard half of the public-tree guard.
        index_path: Where to write. Defaults to ``settings.index_file``.
        force: Build a narrower corpus over a published index. Without it that
            is refused -- see :func:`_refuse_if_published`, which lists what is
            exempt, including the ``--public-only`` build CI runs.
    """
    import time

    from aorta.chat.rag.indexer import split_documents
    from aorta.chat.rag.retriever import SqliteVecStore

    started = time.monotonic()
    corpus = corpus or corpus_mod.local_corpus(settings.aorta_path)
    target = Path(index_path) if index_path else settings.index_file
    target.parent.mkdir(parents=True, exist_ok=True)
    # Before the corpus load and the embedding pass, not after: a build takes
    # tens of minutes, and refusing at the end of one would be worse than not
    # refusing at all.
    _refuse_if_published(target, corpus, force=force)

    logger.info("Loading corpus: %s", corpus.describe())
    documents = corpus_mod.load_corpus(corpus)
    if not documents:
        raise FileNotFoundError(f"corpus is empty: {corpus.describe()}")
    logger.info("Loaded %d files; splitting ...", len(documents))
    chunks = split_documents(documents)
    logger.info("Created %d chunks.", len(chunks))

    provider = get_provider()
    logger.info("Embedding with %s ...", provider.describe())
    embeddings = provider.get_embeddings()
    collection = provider.collection_name()

    # Built beside the destination and moved in at the end. A build takes tens
    # of minutes and embeds into the store as it goes, so writing straight to
    # the configured path meant a Ctrl-C left a partly-populated index under
    # sidecars that still described the previous one -- an index that answers,
    # from a fraction of the corpus, with nothing on screen to say so.
    with tempfile.TemporaryDirectory(prefix=_STAGING_PREFIX, dir=target.parent) as staging_dir:
        staged = Path(staging_dir) / target.name
        store = SqliteVecStore(path=staged, embedding=embeddings, collection=collection)
        try:
            store.reset()
            for start in range(0, len(chunks), _WRITE_BATCH):
                store.add_documents(
                    chunks[start : start + _WRITE_BATCH], provider=provider.describe()
                )
            dimensions = store.dimension()
        finally:
            store.close()

        # Hashed before the move, because this records the index *as built* --
        # the bytes a publish would upload and ``sha256sum -c`` would check.
        # Anything this machine carries over afterwards is its own, and is why
        # the load-time check counts chunks instead of hashing the file.
        index_sha256 = manifest_mod.sha256_file(staged)
        _install_staged(staged, target)

    # The model the *selected* provider embeds with, not the local setting: a
    # remote build otherwise stamps BGE's name on OpenAI's vectors, and the
    # load-time check then compares that label against the same wrong setting
    # and agrees with itself.
    embedding_model = provider.model_id()
    digest = corpus_mod.corpus_digest(
        documents,
        embedding_model=embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    manifest = manifest_mod.Manifest(
        aorta_version=installed_version(),
        aorta_sha=corpus_mod.head_sha(corpus.base),
        aorta_tag=corpus_mod.head_tag(corpus.base),
        embedding_provider=provider.name,
        embedding_model=embedding_model,
        embedding_identity=provider.vector_identity(),
        dimensions=dimensions,
        collection=collection,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        index_sha256=index_sha256,
        store_version=_store_version(),
        built_at=manifest_mod.now_stamp(),
        corpus_digest=digest,
        corpus_roots=list(corpus.roots_label or corpus.subpaths),
        file_count=len(documents),
        chunk_count=len(chunks),
    )
    manifest_mod.write_manifest(target, manifest)
    return BuildResult(
        index_path=target,
        manifest=manifest,
        file_count=len(documents),
        chunk_count=len(chunks),
        size_bytes=target.stat().st_size,
        seconds=time.monotonic() - started,
        corpus=corpus.describe(),
    )


def _store_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("sqlite-vec")
    except PackageNotFoundError:  # pragma: no cover
        return ""


def compute_digest(corpus: corpus_mod.Corpus | None = None) -> tuple[str, int]:
    """Corpus digest and file count, without embedding anything.

    What ``nightly.yml`` compares against the published manifest to decide
    whether tonight's rebuild would produce anything new.
    """
    corpus = corpus or corpus_mod.local_corpus(settings.aorta_path)
    documents = corpus_mod.load_corpus(corpus)
    digest = corpus_mod.corpus_digest(
        documents,
        # Must be the same identity ``build_index`` digests with, or the
        # nightly comparison reports a rebuild is needed on every run.
        embedding_model=get_provider().model_id(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return digest, len(documents)


# ── validating what arrived ───────────────────────────────────────────────


def _no_collection_reason(target: Path, collection: str) -> str:
    """Why a readable store holds no chunks for the configured collection.

    Separated from the caller because the three states behind it want different
    sentences and only one of them is about the store's contents: a manifest
    whose index file is simply gone is a different problem from a store built
    against another provider, and telling the second user their file is missing
    would send them looking for it.
    """
    if not target.exists():
        return f"the manifest is here but the index file it describes is not ({target})"
    return (
        f"the index holds no {collection} collection, so this install cannot "
        "query it -- it was built by a different embedding provider, or the "
        "build did not reach the point of writing chunks"
    )


def check_index(
    index_path: str | Path | None = None,
    *,
    strict: bool = True,
    manifest: manifest_mod.Manifest | None = None,
) -> manifest_mod.ValidationReport:
    """Validate an on-disk index against the configured provider.

    ``strict`` raises on a refusal, which is what every load path wants. It is
    only ``False`` for ``doctor``, whose job is to report every problem at once
    rather than stop at the first.

    The index is opened, not merely stat-ed, so the manifest is checked against
    the contents rather than only against itself. That is what stops ``doctor``
    reporting a healthy index over one whose build was interrupted, where every
    field it could read came from the sidecar the interrupted build never got
    round to replacing.

    ``manifest`` lets a caller supply the sidecar it has already read, which is
    :func:`_unusable_reasons` asking this question of a manifest parsed
    tolerantly. Read here otherwise, and strictly, because every other caller
    is predicting what the load path will do and the load path parses strictly.
    """
    from aorta.chat.rag.retriever import IndexUnreadableError, collection_chunk_count

    target = Path(index_path) if index_path else settings.index_file
    provider = get_provider()
    if manifest is None:
        manifest = manifest_mod.read_manifest(target)

    unreadable = ""
    chunk_count: int | None = None
    try:
        chunk_count = collection_chunk_count(target, provider.collection_name())
    except IndexUnreadableError as exc:
        # Reported rather than raised, because ``doctor`` calls this to find out
        # what is wrong and must not be answered with a traceback. It still
        # becomes a refusal below.
        unreadable = str(exc)

    report = manifest_mod.validate(
        manifest,
        embedding_model=provider.model_id(),
        collection=provider.collection_name(),
        embedding_identity=provider.vector_identity(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        installed_version=installed_version(),
        chunk_count=chunk_count,
    )
    # A file nothing can open fails closed, whatever its manifest claims.
    #
    # This used to be gated on ``manifest.chunk_count``, on the reasoning that a
    # manifest predating the field asserts nothing and so has nothing to
    # contradict. That reads the wrong question. "Is the manifest's claim
    # contradicted" and "can this index be queried at all" are different, and
    # only the second one decides whether the index is usable -- the load path
    # (``retriever._check_manifest``) refuses an unreadable store outright, with
    # no such gate, so the gate made this reader *more permissive than the load
    # path it exists to predict*: a legacy manifest over filler bytes reported
    # no refusals here while the first query refused the same file. It also
    # defeated this function's own docstring, and PR #463 grew a second
    # store-reading helper in ``doctor`` to work around it.
    if unreadable:
        # Recorded on the report as well as refused, so ``doctor`` can tell this
        # apart from a provider mismatch without reading the sentence back.
        report.unreadable = unreadable
        # Two messages, because "describes 0 chunks" would be a claim the
        # manifest never made.
        claim = (
            f"the manifest describes {manifest.chunk_count} chunks, but the index "
            "could not be read to check"
            if manifest.chunk_count
            else "the index could not be read as a sqlite store"
        )
        report.refusals.append(f"contents: {claim} ({unreadable})")
    elif chunk_count is None:
        # A store that opened without holding the collection this install
        # queries. ``collection_chunk_count`` returns ``None`` here rather than
        # raising, and ``validate`` skips its count comparison when handed
        # ``None`` -- so between them this state produced *no refusals at all*,
        # and the two readers built on that silence drew the worst available
        # conclusion from it: the build guard exempted a rebuild, and ``fetch``
        # reported "already up to date" over an index that cannot answer a
        # single query. ``None`` is an ordinary answer from that helper and the
        # wrong one to treat as consent, because it covers three states and
        # every one of them is refused by the load path
        # (``retriever._open_store``, which names the collections that *are*
        # there): the sidecar outliving its store, a store some other
        # provider's collection was built into, and a build that stopped before
        # the chunk table existed.
        report.refusals.append(
            f"contents: {_no_collection_reason(target, provider.collection_name())}"
        )
    if strict:
        report.raise_if_refused(target)
    return report


# ── fetching ──────────────────────────────────────────────────────────────


def _widen_read_timeout(response: Any) -> None:
    """Give the body read the longer budget, now that the headers have arrived.

    The split the two constants describe is only expressible after the fact:
    ``urlopen`` takes one timeout and uses it for the connect and for every
    subsequent read, and reaching the socket is the only way to change it once
    the response exists.

    Best-effort on purpose. A stub response in a test has no socket to reach,
    and a CPython that moves the attribute would otherwise turn a cosmetic
    improvement into a failed download. Falling back leaves the body on the
    connect budget, which is a per-``recv`` timeout and so is still ample for a
    transfer that is actually flowing.

    The fallback is logged rather than silent. Reaching through
    ``response.fp.raw._sock`` is an implementation detail of the standard
    library, so the day it moves the only visible symptom would be downloads
    failing at 30 seconds again -- with nothing anywhere saying the split had
    stopped applying. Debug level: it is diagnostic for whoever is already
    looking at a timeout, and ``-v`` is what that person passes.
    """
    raw = getattr(getattr(response, "fp", None), "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is None:
        logger.debug(
            "No socket behind the response, so the body keeps the %ss connect "
            "timeout rather than the %ss read timeout.",
            _CONNECT_TIMEOUT,
            _READ_TIMEOUT,
        )
        return
    try:
        sock.settimeout(_READ_TIMEOUT)
    except (OSError, AttributeError) as exc:
        logger.debug(
            "Could not widen the read timeout to %ss (%s); the body keeps the %ss "
            "connect timeout.",
            _READ_TIMEOUT,
            exc,
            _CONNECT_TIMEOUT,
        )


def _human_bytes(count: int) -> str:
    return f"{count / (1024 * 1024):.1f} MB"


def _progress_line(done: int, total: int) -> str:
    """One transfer-progress line, with a percentage only when one is knowable."""
    if total > 0:
        return f"  {_human_bytes(done)} of {_human_bytes(total)} ({done * 100 // total}%)"
    return f"  {_human_bytes(done)}"


def _content_length(response: Any) -> int:
    """``Content-Length`` as an int, or 0 when the server did not send a usable one."""
    headers = getattr(response, "headers", None)
    raw = headers.get("Content-Length") if headers is not None else None
    try:
        return max(int(raw), 0)
    except (TypeError, ValueError):
        # Chunked transfer encoding sends none at all, and a malformed one must
        # cost a percentage rather than the download.
        return 0


def _copy_with_progress(response: Any, handle: Any, total: int) -> None:
    """Stream the body across, saying how far it has got as it goes.

    ``shutil.copyfileobj`` was doing this with no callback, so the one thing
    the user could see about a tens-of-megabytes transfer was that it had
    started.
    """
    done = 0
    report_at = _PROGRESS_STEP
    while True:
        block = response.read(_DOWNLOAD_BLOCK)
        if not block:
            break
        handle.write(block)
        done += len(block)
        if done >= report_at:
            logger.info("%s", _progress_line(done, total))
            report_at = done + _PROGRESS_STEP
    logger.info("%s", _progress_line(done, total))


def _download(url: str, target: Path, dest: Path | None = None) -> None:
    """Stream ``url`` to ``target``, or raise :class:`IndexFetchError`.

    ``target`` is the staging path; ``dest`` is where the install would land,
    and is carried only so the 404 remedy below can name it. The three lines it
    offers are ordinary invocations, so they resolve ``--output`` to the
    configured cache -- which is not where this fetch was going.
    """
    logger.info("Downloading %s", url)
    try:
        with urllib.request.urlopen(url, timeout=_CONNECT_TIMEOUT) as response:  # noqa: S310
            _widen_read_timeout(response)
            total = _content_length(response)
            with open(target, "wb") as handle:
                _copy_with_progress(response, handle, total)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # Labels lead so the block stays aligned whether or not the flags
            # are appended; the commands themselves are then variable-length.
            here = "" if dest is None else _target_flags(dest)
            raise IndexFetchError(
                f"no published index at {url} (HTTP 404).\n"
                "That release may predate the published index, or the asset may "
                "not have been built yet. Options:\n"
                f"  pick another release        aorta chat index fetch --version <X.Y.Z>{here}\n"
                f"  take the rolling main asset aorta chat index fetch "
                f"--version {ROLLING_TAG}{here}\n"
                f"  build locally instead       aorta chat index build{here}"
            ) from exc
        raise IndexFetchError(f"could not download {url}: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise IndexFetchError(f"could not reach {url}: {exc.reason}." + _NO_EGRESS_HINT) from exc
    except http.client.HTTPException as exc:
        # A response body that stops short of its Content-Length raises
        # IncompleteRead, which is not an OSError, so it would leave this
        # function as a traceback past the CLI's IndexFetchError guard -- and a
        # truncated transfer is the expected failure for tens of megabytes over
        # a flaky link, not an anomaly.
        raise IndexFetchError(
            f"the download of {url} ended early ({type(exc).__name__}: {exc}); "
            "nothing was installed"
        ) from exc
    except OSError as exc:
        # Names both sides rather than asserting a cause it cannot know: this
        # single handler covers a socket error mid-stream (never wrapped as a
        # URLError, because the request itself succeeded) as well as a full or
        # unwritable destination.
        raise IndexFetchError(f"could not transfer {url} to {target}: {exc}") from exc


def _download_text(url: str, dest: Path | None = None) -> str:
    """Fetch a small sidecar (manifest or checksum) as text, or raise
    :class:`IndexFetchError`.

    ``dest`` is where this command would write, carried only so the remedy
    below can name it. It offers ``index build``, which resolves ``--output``
    to the configured cache -- so a caller working on any other path was told
    to build over an index that had nothing to do with the failure.

    Fetched before the index itself, so every failure here has to arrive as
    ``IndexFetchError`` for the CLI to render it: this is the first thing
    ``aorta chat index fetch`` does on a node whose egress it knows nothing
    about.

    It is also the first thing the user waits on, which is why it logs. Without
    this line the only ``Downloading`` message in the module sat on the *third*
    request, so a node that could not reach the release host produced no output
    at all before it failed.
    """
    logger.info("Fetching %s", url)
    try:
        with urllib.request.urlopen(url, timeout=_CONNECT_TIMEOUT) as response:  # noqa: S310
            # Widened for the same reason the asset transfer is, and it was
            # missing here: ``urlopen``'s single timeout also governs every
            # read, so a sidecar served slowly by a loaded release host failed
            # on the connect budget rather than the read budget the two
            # constants advertise. A kilobyte rarely needs it -- but this is
            # the *first* request the command makes, so it is the one whose
            # failure a user reads as "fetch does not work here".
            _widen_read_timeout(response)
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise IndexFetchError(
            f"the published index has no companion sidecar at {url} "
            f"(HTTP {exc.code}).\n"
            "An index without its manifest and checksum cannot be verified "
            "against this install's embedding model, so it is not safe to use. "
            "Build locally instead:  "
            f"{'aorta chat index build' if dest is None else _pasteable('aorta chat index build', dest)}"
        ) from exc
    except urllib.error.URLError as exc:
        raise IndexFetchError(f"could not reach {url}: {exc.reason}" + _NO_EGRESS_HINT) from exc
    except (http.client.HTTPException, OSError) as exc:
        raise IndexFetchError(f"could not fetch {url}: {type(exc).__name__}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise IndexFetchError(f"the sidecar at {url} is not valid UTF-8 ({exc})") from exc


def _verify_checksum(path: Path, expected_line: str, url: str) -> str:
    """Compare ``path``'s SHA256 against a ``sha256sum``-format line."""
    expected = expected_line.strip().split()[0] if expected_line.strip() else ""
    actual = manifest_mod.sha256_file(path)
    if not expected:
        raise IndexFetchError(f"the checksum file at {url} is empty")
    if actual != expected:
        raise IndexFetchError(
            f"checksum mismatch on the downloaded index.\n"
            f"  expected {expected}\n"
            f"  got      {actual}\n"
            "The download is corrupt or was tampered with; it has not been installed."
        )
    return actual


def describe_target(source: IndexSource, dest: str | Path) -> list[str]:
    """The lines a caller should show *before* the first request.

    Every one of these is known before any network I/O and used to be printed
    only after the last request succeeded -- so a fetch that stalled, or that
    refused on the manifest, said nothing about what it had been trying to do.
    ``resolve_source`` even composes a note explaining the rolling asset on a
    dev install, and that note reached the user through ``FetchResult`` alone.

    Composed here rather than in the CLI so the wording is shared and testable,
    but *shown* by the caller: only the caller knows whether it has a terminal,
    a JSON stream or a log to write to. ``fetch_index`` therefore does not
    print this itself, and logs each individual request instead.
    """
    lines = [
        f"Fetching the published index ({source.describe()})",
        f"  from {source.index_url}",
        f"  into {dest}",
    ]
    lines.extend(f"  note: {note}" for note in source.notes)
    return lines


def _parse_manifest(text: str, source: IndexSource, dest: Path) -> manifest_mod.Manifest:
    """Parse a downloaded manifest, or raise :class:`IndexFetchError`.

    The schema and field-type checks belong here rather than after the install:
    a manifest this build cannot read must fail the fetch rather than land on
    disk and then be rejected by the first load, which reports a successful
    fetch and leaves a chat that no longer starts.

    **The remedy is this function's to write, and not the parser's.** The
    parsers wrapped here are handed a dict and cannot know whose manifest it is
    -- local sidecar, this download, or a side-loaded file all arrive
    identically and are fixed by three different things -- so a remedy composed
    down there is a remedy composed from the reader's configuration when the
    *origin* was the question. That is not hypothetical: #463 found the parser
    closing its type-mismatch and schema-version errors with "Replace it with
    'aorta chat index fetch'", which this wrapper carried verbatim into a fetch
    error, telling the user to re-run the download that produced the bad bytes.
    Retrying is deterministic -- same URL, same bytes, same failure.

    So the remedy here is the one a *published* origin admits, which is the
    thing worth saying plainly: nothing the user does to their own machine
    fixes it. Both commands offered therefore go somewhere else -- a different
    release, or their own corpus. This covers every parser reached below rather
    than the two sites that were reported, because they all arrive as one
    exception at one wrap; :func:`side_load` writes its own for the same reason
    from the other side.

    ``dest`` is required rather than defaulted, because a remedy naming the
    wrong index is the defect one layer along: both callers know the path they
    were asked about, and an optional parameter is how the bare ``index build``
    got into the sidecar 404s twice.
    """
    try:
        manifest = manifest_mod.Manifest.from_dict(json.loads(text), advise=False)
        _ensure_field_types(manifest)
        manifest_mod.ensure_supported_schema(
            manifest, f"the index at {source.index_url}", advise=False
        )
    except (ValueError, manifest_mod.ManifestError) as exc:
        raise IndexFetchError(
            f"the manifest at {source.manifest_url} is not usable: {exc}\n"
            "This is the published release's own metadata, so fetching it again "
            "downloads the same bytes -- there is no local fix. Pin a different "
            "release with 'aorta chat index fetch --version <tag>', or build "
            f"from your own corpus with '{_pasteable('aorta chat index build', dest)}'. "
            "Please report the broken release."
        ) from exc
    return manifest


#: What each declared annotation on :class:`~aorta.chat.rag.manifest.Manifest`
#: actually has to be. Read from the dataclass rather than listed field by
#: field, so a field added upstream is covered without a matching edit here.
_ANNOTATION_TYPES: dict[str, type | tuple[type, ...]] = {
    "str": str,
    "int": int,
    "list[str]": list,
}


def _ensure_field_types(manifest: manifest_mod.Manifest) -> None:
    """Reject a downloaded manifest whose fields are not the types they claim.

    ``Manifest.from_dict`` deliberately does not type-check: a *local* sidecar
    that is wrong should still be readable, because refusing to parse it would
    take away the reports whose whole job is to say what is wrong with it. The
    tolerant readers here (:func:`_corpus_roots`, :func:`_manifest_text`) exist
    for exactly that.

    A *downloaded* manifest is the other case. It is untrusted input arriving
    over the network, nothing later in the fetch is written to tolerate it, and
    the readers it reaches are not all here -- ``validate`` calls ``.split()``
    on ``embedding_identity``, so a published manifest carrying ``42`` there
    raised ``AttributeError`` past the ``IndexFetchError`` handling and out of
    the CLI as a traceback, rather than the refusal a bad manifest is supposed
    to produce.

    So the tolerance stops at the network boundary, which is the one place
    there is somewhere better to put the failure: this is the funnel that
    already turns "the published manifest is not usable" into a refusal naming
    its URL.
    """
    for field_def in dataclasses.fields(manifest):
        expected = _ANNOTATION_TYPES.get(str(field_def.type))
        if expected is None:
            continue
        value = getattr(manifest, field_def.name)
        # bool is an int subclass, and a manifest recording `chunk_count: true`
        # is malformed in exactly the way this is here to catch.
        if not isinstance(value, expected) or isinstance(value, bool):
            raise manifest_mod.ManifestError(
                f"{field_def.name} is a {type(value).__name__}, not {field_def.type}"
            )
        # The elements too, not just the container. Checking `list` alone let
        # `corpus_roots: [42]` install, and the *next* fetch then classified the
        # sidecar it had just written as unclassifiable and refused a routine
        # refresh -- a boundary check that admits a value the guard behind it
        # rejects is worse than none, because the damage surfaces one command
        # later and on a different command.
        if str(field_def.type) == "list[str]":
            for item in value:
                if not isinstance(item, str):
                    raise manifest_mod.ManifestError(
                        f"{field_def.name} contains a {type(item).__name__}, not str"
                    )


def _validate_against_provider(manifest: manifest_mod.Manifest) -> manifest_mod.ValidationReport:
    """Check an incoming manifest against what this install would query with."""
    provider = get_provider()
    return manifest_mod.validate(
        manifest,
        embedding_model=provider.model_id(),
        collection=provider.collection_name(),
        embedding_identity=provider.vector_identity(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        installed_version=installed_version(),
    )


def _manifest_text(value: object) -> str:
    """A manifest field as text, whatever type it actually holds.

    The same hole as :func:`_corpus_roots`, one field over.
    ``Manifest.from_dict`` type-checks nothing, so a hand-written sidecar can
    record ``null`` where a digest belongs -- and truncating that raised
    ``TypeError`` (``KeyError`` for a JSON object, which subscripts by key)
    straight out of the comparison whose entire job is to report on a manifest
    that looks wrong. Rendered before it is truncated, which is what
    ``index status``'s table already does with the same values.

    ``None`` becomes empty rather than ``"None"`` so the callers' own wording
    for a missing field still fires.

    Rendering rather than validating, deliberately, and the asymmetry with
    ``corpus_roots`` is the point: these fields are *reported*, and a reported
    field survives being odd -- a caller reads "corpus_digest: unknown -> abc"
    and knows what it is looking at. ``corpus_roots`` is *used*, to decide
    whether a destructive overwrite is refused, so it gets a validity verdict
    instead. A field whose only job is to be shown does not need one.
    """
    return "" if value is None else str(value)


def _refresh_notes(
    local: manifest_mod.Manifest | None, incoming: manifest_mod.Manifest
) -> list[str]:
    """What a refresh is about to change, field by field.

    A routine ``fetch`` over an older fetched index proceeds without asking --
    it is the documented way to refresh, and demanding ``--force`` every time
    would train people to always pass it -- so what it replaced is reported
    instead.

    **Derived from :data:`_SIDE_FIELDS`, which is what ``index status`` prints
    side by side.** It used to name three fields chosen by hand, and ``status``
    then took its verdict from this function while rendering a table of ten --
    so a sidecar differing only in ``embedding_model`` or ``dimensions``
    reported *up to date* above a table showing the mismatch, and the load path
    could refuse the same index. One list rather than two is the fix: a field
    added to the table is reported and judged without a second edit, which is
    the drift that produced the disagreement in the first place. ``index_sha256``
    is skipped because it is the identity the caller has already compared and
    "the hashes differ" is not news beside the fields that explain why.
    """
    if local is None:
        return []
    was_side, now_side = _side(local), _side(incoming)
    changes = []
    for name in _SIDE_FIELDS:
        if name == "index_sha256":
            continue
        was = _manifest_text(was_side[name])
        now = _manifest_text(now_side[name])
        if was == now:
            continue
        trim = _NOTE_TRIM.get(name)
        if trim:
            was, now = was[:trim], now[:trim]
        changes.append(f"{name}: {was or 'unknown'} -> {now or 'unknown'}")
    return changes


#: Fields whose full value is a digest, shortened in the change notes. Anything
#: absent from here is printed whole, because a truncated model name or
#: dimension count would hide the difference the line exists to show.
_NOTE_TRIM = {"aorta_sha": 7, "corpus_digest": 12}


def _is_same_index(local: manifest_mod.Manifest | None, incoming: manifest_mod.Manifest) -> bool:
    """Whether the installed index is byte-identical to the published one.

    ``index_sha256`` is exact content identity, so this is the one comparison
    that can skip the transfer outright. Both sides must actually carry one: a
    manifest predating the field would otherwise compare equal on ``""`` and
    skip a download it needed.
    """
    if local is None or not local.index_sha256 or not incoming.index_sha256:
        return False
    return local.index_sha256 == incoming.index_sha256


def fetch_index(
    version: str | None = None,
    index_path: str | Path | None = None,
    source: IndexSource | None = None,
    *,
    force: bool = False,
) -> FetchResult:
    """Download, verify, validate and install the published index.

    The order of the checks is load-bearing, and two of them moved.

    **Identity is checked on the manifest, before the asset is transferred.**
    That ~1 KB sidecar already carries ``embedding_model``, ``collection`` and
    ``embedding_identity``, so a mismatch is knowable the moment it arrives --
    yet it used to be validated only after the full index transfer and its
    checksum. A reporter with a misconfigured embedding provider therefore
    downloaded ~20 MB that was guaranteed to be discarded. Fixing the
    configuration stops that particular refusal; any later mismatch -- a model
    change, an endpoint change -- hits the same wasted transfer.

    **The asset is skipped entirely when the local sidecar already records the
    published ``index_sha256`` and the store it describes is usable.** A
    refresh that would change nothing is the common case, and it cost the whole
    download. Both halves are needed: the sidecar establishes that the right
    index was installed, and only opening the store establishes that it still
    is one.

    Checksum verification stays where it was, because it genuinely needs the
    bytes. Nothing is installed until it and the manifest/checksum agreement
    both pass.

    ``force`` overwrites a locally-built index, and re-downloads an asset that
    is already installed.
    """
    dest = Path(index_path) if index_path else settings.index_file
    source = source or resolve_source(version)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Free, and therefore first: refusing before any network I/O means a user
    # who is about to lose a local build is told so immediately rather than
    # after a download.
    _refuse_if_locally_built(dest, force=force, command="aorta chat index fetch")

    # The text is kept, not re-serialised from the dataclass at install time:
    # ``from_dict`` drops keys this version predates, and writing them back out
    # would strip a newer builder's fields from the sidecar this machine keeps.
    manifest_text = _download_text(source.manifest_url, dest)
    manifest = _parse_manifest(manifest_text, source, dest)
    report = _validate_against_provider(manifest)
    report.raise_if_refused(source.index_url)

    local = _local_manifest(dest)
    if not force and _is_same_index(local, manifest):
        # Matching ``index_sha256`` proves the *sidecar* is the published one.
        # It says nothing about the file beside it, which is written in place
        # by run indexing and can be damaged by anything else on the machine --
        # so a truncated store under an untouched manifest took this shortcut
        # and was told it was already up to date, by the one command that would
        # have repaired it.
        #
        # Asked through the same reader the build guard uses, which opens the
        # store. This PR already stopped ``check_index`` trusting the manifest
        # about the contents; a refresh that decided on the manifest alone
        # would have contradicted that from the other side.
        #
        # Deliberately the load path's notion of usable, not a schema probe:
        # this catches a store nothing can open, one holding no chunks for the
        # collection this install queries, and a row count the manifest
        # disagrees with -- and not the several ways a store can open, count
        # correctly and still be the wrong shape underneath (``doctor``
        # enumerates those). Matching the reader the *first query* uses is the
        # property that matters here -- "already up to date" must not mean
        # something the next query would refuse.
        unusable = _unusable_reasons(dest)
        if unusable:
            logger.info(
                "%s records the published index but this install cannot use it "
                "(%s); fetching the asset again.",
                dest,
                "; ".join(unusable),
            )
        else:
            # At debug, because ``up_to_date`` is on the result and the caller
            # renders it -- the CLI would otherwise print the same sentence twice.
            logger.debug("%s already holds the published index; skipping the asset.", dest)
            return FetchResult(
                index_path=dest,
                manifest=manifest,
                source=source.describe(),
                warnings=list(report.warnings),
                notes=list(source.notes),
                up_to_date=True,
            )
    changes = _refresh_notes(local, manifest)

    # Staged inside the destination directory so the final move is a rename on
    # one filesystem, and therefore atomic: a concurrent reader sees the old
    # index or the new one, never a half-written file.
    with tempfile.TemporaryDirectory(prefix=_STAGING_PREFIX, dir=dest.parent) as staging_dir:
        staged = Path(staging_dir) / ASSET_NAME
        checksum_line = _download_text(source.checksum_url, dest)
        _download(source.index_url, staged, dest)
        checksum = _verify_checksum(staged, checksum_line, source.checksum_url)
        if manifest.index_sha256 and manifest.index_sha256 != checksum:
            raise IndexFetchError(
                "the manifest and the checksum file disagree about the index:\n"
                f"  manifest says {manifest.index_sha256}\n"
                f"  .sha256 says  {checksum}\n"
                "The published set is inconsistent; nothing has been installed."
            )

        # Index first, sidecars after. Writing the sidecars first looked like
        # the fail-safe order -- and is, on a first install, where there is no
        # index for them to describe. On a re-fetch it is the dangerous one: an
        # interruption between the two leaves the *previous* index paired with
        # the incoming manifest, which is wrong and self-consistent, and no
        # field check can see it because every field it reads is the sidecar's.
        carried = _install_staged(staged, dest)
        # Checksum then manifest, matching ``write_manifest``: the manifest is
        # the file the load path gates on, so it is the last to appear.
        manifest_mod.checksum_path(dest).write_text(f"{checksum}  {dest.name}\n", encoding="utf-8")
        manifest_mod.manifest_path(dest).write_text(manifest_text, encoding="utf-8")

    return FetchResult(
        index_path=dest,
        manifest=manifest,
        source=source.describe(),
        warnings=[*report.warnings, *_carried_note(carried)],
        notes=list(source.notes),
        changes=changes,
    )


# ── comparing the two sides ───────────────────────────────────────────────

#: The verdicts :func:`compare_index` can reach. Named rather than inlined so
#: ``--json`` consumers have a closed set to switch on, and so the honest
#: distinction between "the same" and "nothing to compare against" is visible
#: in one place: ``nightly.yml``'s own comment makes the same point, that a
#: missing published manifest means *no baseline* and must never be reported as
#: up to date.
VERDICT_UP_TO_DATE = "up_to_date"
VERDICT_PUBLISHED_DIFFERS = "published_differs"
VERDICT_LOCALLY_BUILT = "locally_built"
VERDICT_INCOMPATIBLE = "incompatible"
VERDICT_NO_LOCAL_INDEX = "no_local_index"
VERDICT_UNREADABLE_LOCAL_INDEX = "unreadable_local_index"
VERDICT_NO_BASELINE = "no_baseline"

_VERDICT_SUMMARY = {
    VERDICT_UP_TO_DATE: "up to date; the local index is the published one",
    VERDICT_PUBLISHED_DIFFERS: "the published index differs from the local one",
    VERDICT_LOCALLY_BUILT: (
        "the local index was built here, so the published one is not a newer copy of it"
    ),
    VERDICT_INCOMPATIBLE: (
        "the published index is not usable by this install's embedding provider"
    ),
    VERDICT_NO_LOCAL_INDEX: "no local index; nothing has been installed yet",
    # Split out of ``no_local_index`` rather than reported as it: "nothing has
    # been installed yet" over a file that is sitting right there tells the one
    # person looking at this command the opposite of what is wrong.
    VERDICT_UNREADABLE_LOCAL_INDEX: (
        "a local index is installed, but no manifest beside it can be read, so "
        "it cannot be compared -- or loaded"
    ),
    VERDICT_NO_BASELINE: "no published baseline to compare against",
}


@dataclass
class IndexComparison:
    """The local index and the published one, side by side.

    Answers gap 4b's question -- "is the index in the cache the same as the
    remote one, and which is newer" -- from the two manifests alone, so it
    transfers ~1 KB rather than the index.
    """

    source: IndexSource
    index_path: Path
    local: manifest_mod.Manifest | None
    published: manifest_mod.Manifest | None
    verdict: str
    #: Why the published manifest could not be read, when it could not be.
    baseline_error: str = ""
    #: Why the *local* manifest could not be read, when a local index is there
    #: and it could not be. Reported for the same reason ``baseline_error`` is:
    #: the verdict names the state, and only this names the cause.
    local_error: str = ""
    #: Field-by-field differences, when both sides are readable.
    differences: list[str] = field(default_factory=list)
    provenance: str = PROVENANCE_UNKNOWN

    @property
    def summary(self) -> str:
        return _VERDICT_SUMMARY.get(self.verdict, self.verdict)

    @property
    def up_to_date(self) -> bool:
        return self.verdict == VERDICT_UP_TO_DATE


#: The manifest fields ``index status`` puts side by side. Enumerated so both
#: sides carry the same keys whether or not there is a manifest to fill them.
_SIDE_FIELDS = (
    "embedding_model",
    "embedding_identity",
    "built_at",
    "aorta_version",
    "aorta_sha",
    "corpus_digest",
    "corpus_roots",
    "index_sha256",
    "chunk_count",
    "dimensions",
)


def _side(manifest: manifest_mod.Manifest | None) -> dict[str, Any]:
    """One side of the comparison, as the fields gap 4b asks for.

    Every key is always present, with ``None`` for "there is no manifest on
    this side". ``--json`` is a contract, and returning ``{}`` made
    ``published`` an empty object in the no-baseline branch -- so a consumer
    indexing the documented shape got a ``KeyError`` on exactly the run it
    most needed to inspect, while the reachable branch it was written against
    carried all ten keys.

    ``corpus_roots`` is read through :func:`_corpus_roots`, so an unusable
    recorded value reports ``None`` rather than the characters of a string.
    """
    if manifest is None:
        return dict.fromkeys(_SIDE_FIELDS)
    return {
        "embedding_model": manifest.embedding_model,
        "embedding_identity": manifest.embedding_identity,
        "built_at": manifest.built_at,
        "aorta_version": manifest.aorta_version,
        "aorta_sha": manifest.aorta_sha,
        "corpus_digest": manifest.corpus_digest,
        "corpus_roots": _corpus_roots(manifest),
        "index_sha256": manifest.index_sha256,
        "chunk_count": manifest.chunk_count,
        "dimensions": manifest.dimensions,
    }


def compare_index(
    version: str | None = None,
    index_path: str | Path | None = None,
    source: IndexSource | None = None,
) -> IndexComparison:
    """Compare the installed index against the published one, read-only.

    Downloads the ~1 KB manifest and nothing else. This is the comparison
    ``nightly.yml``'s ``digest`` step already does in bash to decide whether to
    republish, so the capability is proven and load-bearing for the release
    process -- it was simply not exposed to the user who asked for it.

    Nothing here writes, and an unreachable host is a *result* rather than an
    error: "there is no baseline" is the honest answer, and the one thing it
    must not be rendered as is "up to date".
    """
    dest = Path(index_path) if index_path else settings.index_file
    source = source or resolve_source(version)
    local, unreadable = _read_destination(dest)

    published: manifest_mod.Manifest | None = None
    baseline_error = ""
    try:
        # ``dest`` carried in so a sidecar 404 recommends building the index the
        # user asked about with ``--index``, not the cache. ``status --index``
        # and ``build --output`` are the same path under two flag names, and the
        # remedy has to be spelled in the second one.
        published = _parse_manifest(_download_text(source.manifest_url, dest), source, dest)
    except IndexFetchError as exc:
        baseline_error = str(exc)

    if published is None:
        verdict = VERDICT_NO_BASELINE
    elif unreadable:
        # Ahead of the compatibility check, which reads the *published*
        # manifest: when both hold, the local read failure is the one the user
        # can act on, and it is the only one of the two this command exits
        # non-zero for. Ordered the other way, an install with an unreadable
        # sidecar *and* an incompatible baseline reported ``incompatible`` and
        # exited 0 -- so the state this verdict was added for was reported as
        # the state it was added to be distinguished from, and a script gating
        # on the exit code saw success. Absent-local stays *after*
        # compatibility, unchanged: there is nothing local to have failed to
        # read, and a baseline this install could not use is then the more
        # useful answer than "you have no index".
        verdict = VERDICT_UNREADABLE_LOCAL_INDEX
    elif _validate_against_provider(published).refusals:
        # Checked before the content comparison: an index this install cannot
        # query is not made usable by being newer.
        verdict = VERDICT_INCOMPATIBLE
    elif local is None:
        verdict = VERDICT_NO_LOCAL_INDEX
    elif _is_same_index(local, published) and not _refresh_notes(local, published):
        # Both halves, because ``status`` prints the differences beside the
        # verdict and the two must not disagree in the same output. The hash is
        # the right predicate for ``fetch``, where the question is "would the
        # transfer change these bytes" and the answer is no -- but ``status``
        # answers "is my index the published one", and a sidecar carrying the
        # published ``index_sha256`` under a different ``aorta_sha`` or
        # ``corpus_digest`` reported *up to date* while the table below it
        # listed those very fields as differing. A reader shown a contradiction
        # believes the shorter half.
        #
        # Asked through ``_refresh_notes`` rather than by comparing a second
        # list of fields here, because that function *is* what the command
        # prints: a field added to it is then covered by this verdict without
        # anyone remembering to add it twice, which is how the two would drift
        # apart again.
        verdict = VERDICT_UP_TO_DATE
    elif index_provenance(local) == PROVENANCE_LOCAL:
        # Deliberately not a recency claim. ``built_at`` is wall-clock from
        # whoever built it, so a local index can carry a later timestamp while
        # indexing older source -- and it is the one case where "just fetch"
        # is the wrong advice, because the fetch would discard it.
        verdict = VERDICT_LOCALLY_BUILT
    else:
        verdict = VERDICT_PUBLISHED_DIFFERS

    return IndexComparison(
        source=source,
        index_path=dest,
        local=local,
        published=published,
        verdict=verdict,
        baseline_error=baseline_error,
        local_error=unreadable,
        differences=_refresh_notes(local, published) if published is not None else [],
        provenance=index_provenance(local) if local is not None else PROVENANCE_UNKNOWN,
    )


def comparison_to_dict(comparison: IndexComparison) -> dict[str, Any]:
    """``index status --json``, matching the other index subcommands."""
    return {
        "verdict": comparison.verdict,
        "summary": comparison.summary,
        "up_to_date": comparison.up_to_date,
        # Named explicitly because a dev install resolves to the rolling tag:
        # without this the verdict does not say what it compared against.
        "compared_against": {
            "source": comparison.source.describe(),
            "manifest_url": comparison.source.manifest_url,
        },
        "local": {
            "index_path": str(comparison.index_path),
            "provenance": comparison.provenance,
            "error": comparison.local_error,
            **_side(comparison.local),
        },
        "published": _side(comparison.published),
        "differences": comparison.differences,
        "baseline_error": comparison.baseline_error,
    }


def side_load(
    staged: str | Path, index_path: str | Path | None = None, *, force: bool = False
) -> FetchResult:
    """Adopt a locally staged index (Decision 21b's ``--from``).

    The manifest sidecar is required, not optional: side-loading is the path an
    air-gapped user takes, and it is the path where a mismatched index is most
    likely, because the file was carried by hand from somewhere else.

    It is also the third write path, so it carries ``fetch``'s guard: without
    it, ``--from`` would be the way to overwrite a local build by accident.
    """
    origin = Path(staged).expanduser().resolve()
    if origin.is_dir():
        candidate = origin / ASSET_NAME
        if not candidate.exists():
            raise IndexFetchError(
                f"{origin} is a directory and holds no {ASSET_NAME}. Pass the "
                "index file itself, or stage it under that name."
            )
        origin = candidate
    if not origin.exists():
        raise IndexFetchError(f"no index at {origin}")

    dest = Path(index_path) if index_path else settings.index_file
    if origin == dest.resolve():
        raise IndexFetchError(f"{origin} is already the configured index path")
    # ``shlex.quote``, because this string is printed as a command to run and
    # ``origin`` is a user-supplied path: a space or an apostrophe (a
    # ``/home/o'brien`` home directory is enough) otherwise yields advice that
    # does not parse in the shell it is pasted into. Quoting is a no-op for an
    # ordinary path, so the usual message is unchanged.
    _refuse_if_locally_built(
        dest,
        force=force,
        command=f"aorta chat index fetch --from {shlex.quote(str(origin))}",
    )

    manifest_source = manifest_mod.manifest_path(origin)
    if not manifest_source.exists():
        raise IndexFetchError(
            f"no manifest beside {origin}. Stage {manifest_source.name} alongside "
            "the index -- it is what records the embedding model, and without it "
            "a mismatched index cannot be distinguished from a correct one."
        )
    # Held to the same field-type check as a downloaded manifest, and for a
    # stronger reason: this sidecar was hand-carried, so it is the one most
    # likely to have been hand-edited. ``read_manifest`` wraps a bad parse but
    # not a bad *read*, and neither wraps a wrong-typed field -- so a non-UTF-8
    # file escaped as ``UnicodeDecodeError`` and ``embedding_identity: 42``
    # escaped as ``AttributeError`` out of the provider validation below, both
    # past the ``IndexFetchError`` the CLI knows how to print.
    try:
        manifest = manifest_mod.read_manifest(origin, advise=False)
        _ensure_field_types(manifest)
    except (manifest_mod.ManifestError, UnicodeDecodeError, ValueError) as exc:
        # Its own remedy, for the same reason :func:`_parse_manifest` writes
        # one: the bytes are local but they are not aorta's, so neither a fetch
        # nor a build addresses them -- the file that has to change is the one
        # the user carried in. Naming the sidecar rather than the index, since
        # that is the half that failed and they are two files.
        raise IndexFetchError(
            f"the manifest beside {origin} is not usable: {exc}\n"
            f"Re-stage {manifest_source.name} from wherever the index was "
            "built; it travelled with the index and did not survive the trip "
            "intact, so re-running this command on the same pair fails the "
            "same way."
        ) from exc
    checksum = manifest_mod.sha256_file(origin)
    if manifest.index_sha256 and manifest.index_sha256 != checksum:
        raise IndexFetchError(
            f"{origin} does not match the SHA256 in its manifest.\n"
            f"  manifest {manifest.index_sha256}\n"
            f"  file     {checksum}\n"
            "The staged copy is incomplete or corrupt."
        )

    report = _validate_against_provider(manifest)
    report.raise_if_refused(origin)

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Copied through staging for the same reason the download is: ``copy2``
    # straight onto ``dest`` is not atomic, and an interrupted one leaves a
    # truncated index under the sidecars of the index it was replacing.
    with tempfile.TemporaryDirectory(prefix=_STAGING_PREFIX, dir=dest.parent) as staging_dir:
        staged = Path(staging_dir) / dest.name
        shutil.copy2(origin, staged)
        carried = _install_staged(staged, dest)
    manifest_mod.write_manifest(dest, manifest)
    return FetchResult(
        index_path=dest,
        manifest=manifest,
        source=f"side-loaded from {origin}",
        warnings=[*report.warnings, *_carried_note(carried)],
    )


def _carried_note(carried: list[str]) -> list[str]:
    """Say so when local collections were preserved, rather than only logging it."""
    if not carried:
        return []
    return [f"kept this machine's own collection(s) across the install: {', '.join(carried)}"]


__all__ = [
    "ASSET_NAME",
    "BASE_URL_ENV",
    "RELEASE_BASE_URL",
    "ROLLING_TAG",
    "PROVENANCE_INVALID",
    "PROVENANCE_LOCAL",
    "PROVENANCE_PUBLISHED",
    "PROVENANCE_UNKNOWN",
    "VERDICT_INCOMPATIBLE",
    "VERDICT_LOCALLY_BUILT",
    "VERDICT_NO_BASELINE",
    "VERDICT_NO_LOCAL_INDEX",
    "VERDICT_PUBLISHED_DIFFERS",
    "VERDICT_UNREADABLE_LOCAL_INDEX",
    "VERDICT_UP_TO_DATE",
    "BuildResult",
    "FetchResult",
    "IndexComparison",
    "IndexFetchError",
    "IndexOverwriteError",
    "IndexSource",
    "build_index",
    "check_index",
    "compare_index",
    "comparison_to_dict",
    "compute_digest",
    "corpus_provenance",
    "describe_target",
    "fetch_index",
    "index_provenance",
    "resolve_source",
    "side_load",
]
