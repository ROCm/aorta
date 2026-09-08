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

import contextlib
import http.client
import json
import logging
import os
import re
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
    """The published index could not be downloaded or verified."""


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
) -> BuildResult:
    """Build an index plus its manifest from ``corpus``.

    Args:
        corpus: What to index. Defaults to the local corpus at
            ``settings.aorta_path``; CI passes
            :func:`~aorta.chat.rag.corpus.published_corpus`, whose tracked-file
            allowlist is the hard half of the public-tree guard.
        index_path: Where to write. Defaults to ``settings.index_file``.
    """
    import time

    from aorta.chat.rag.indexer import split_documents
    from aorta.chat.rag.retriever import SqliteVecStore

    started = time.monotonic()
    corpus = corpus or corpus_mod.local_corpus(settings.aorta_path)
    target = Path(index_path) if index_path else settings.index_file
    target.parent.mkdir(parents=True, exist_ok=True)

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


def check_index(
    index_path: str | Path | None = None,
    *,
    strict: bool = True,
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
    """
    from aorta.chat.rag.retriever import IndexUnreadableError, collection_chunk_count

    target = Path(index_path) if index_path else settings.index_file
    provider = get_provider()
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
    # A manifest that claims contents over a file nothing can open fails
    # closed. Gated on the claim: a manifest from a builder that predates
    # ``chunk_count`` asserts nothing here, so there is nothing to contradict.
    if unreadable and manifest.chunk_count:
        report.refusals.append(
            f"contents: the manifest describes {manifest.chunk_count} chunks, but the "
            f"index could not be read to check ({unreadable})"
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
    """
    raw = getattr(getattr(response, "fp", None), "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is None:
        return
    with contextlib.suppress(OSError, AttributeError):
        sock.settimeout(_READ_TIMEOUT)


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


def _download(url: str, target: Path) -> None:
    """Stream ``url`` to ``target``, or raise :class:`IndexFetchError`."""
    logger.info("Downloading %s", url)
    try:
        with urllib.request.urlopen(url, timeout=_CONNECT_TIMEOUT) as response:  # noqa: S310
            _widen_read_timeout(response)
            total = _content_length(response)
            with open(target, "wb") as handle:
                _copy_with_progress(response, handle, total)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise IndexFetchError(
                f"no published index at {url} (HTTP 404).\n"
                "That release may predate the published index, or the asset may "
                "not have been built yet. Options:\n"
                "  aorta chat index fetch --version <X.Y.Z>   pick another release\n"
                f"  aorta chat index fetch --version {ROLLING_TAG}   take the rolling "
                "main asset\n"
                "  aorta chat index build                     build locally instead"
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


def _download_text(url: str) -> str:
    """Fetch a small sidecar (manifest or checksum) as text, or raise
    :class:`IndexFetchError`.

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
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise IndexFetchError(
            f"the published index has no companion sidecar at {url} "
            f"(HTTP {exc.code}).\n"
            "An index without its manifest and checksum cannot be verified "
            "against this install's embedding model, so it is not safe to use. "
            "Build locally instead:  aorta chat index build"
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


def _parse_manifest(text: str, source: IndexSource) -> manifest_mod.Manifest:
    """Parse a downloaded manifest, or raise :class:`IndexFetchError`.

    The schema check belongs here rather than after the install: a schema this
    build cannot read must fail the fetch rather than land on disk and then be
    rejected by the first load, which reports a successful fetch and leaves a
    chat that no longer starts.
    """
    try:
        manifest = manifest_mod.Manifest.from_dict(json.loads(text))
        manifest_mod.ensure_supported_schema(manifest, f"the index at {source.index_url}")
    except (ValueError, manifest_mod.ManifestError) as exc:
        raise IndexFetchError(
            f"the manifest at {source.manifest_url} is not usable: {exc}"
        ) from exc
    return manifest


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


def _local_manifest(index_path: str | Path) -> manifest_mod.Manifest | None:
    """The installed index's sidecar, or ``None`` when there is not a usable one.

    Absent, unreadable and unparseable all collapse to ``None`` deliberately.
    Every caller is deciding what to do *about* the local index, and none of
    those decisions is improved by a traceback out of a sidecar that is already
    broken -- the load path is where a bad manifest has to be fatal.
    """
    target = Path(index_path)
    if not target.exists():
        return None
    try:
        return manifest_mod.read_manifest(target)
    except manifest_mod.ManifestError as exc:
        logger.debug("No usable manifest beside %s: %s", target, exc)
        return None


def _refresh_notes(
    local: manifest_mod.Manifest | None, incoming: manifest_mod.Manifest
) -> list[str]:
    """What a refresh is about to change, field by field.

    A routine ``fetch`` over an older fetched index proceeds without asking --
    it is the documented way to refresh, and demanding ``--force`` every time
    would train people to always pass it -- so what it replaced is reported
    instead.
    """
    if local is None:
        return []
    changes = []
    for label, was, now in (
        ("built_at", local.built_at, incoming.built_at),
        ("aorta_sha", local.aorta_sha[:7], incoming.aorta_sha[:7]),
        ("corpus_digest", local.corpus_digest[:12], incoming.corpus_digest[:12]),
    ):
        if was != now:
            changes.append(f"{label}: {was or 'unknown'} -> {now or 'unknown'}")
    return changes


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
    published ``index_sha256``.** A refresh that would change nothing is the
    common case, and it cost the whole download.

    Checksum verification stays where it was, because it genuinely needs the
    bytes. Nothing is installed until it and the manifest/checksum agreement
    both pass.
    """
    dest = Path(index_path) if index_path else settings.index_file
    source = source or resolve_source(version)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # The text is kept, not re-serialised from the dataclass at install time:
    # ``from_dict`` drops keys this version predates, and writing them back out
    # would strip a newer builder's fields from the sidecar this machine keeps.
    manifest_text = _download_text(source.manifest_url)
    manifest = _parse_manifest(manifest_text, source)
    report = _validate_against_provider(manifest)
    report.raise_if_refused(source.index_url)

    local = _local_manifest(dest)
    if _is_same_index(local, manifest):
        logger.info("Already up to date; the published asset was not downloaded.")
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
        checksum_line = _download_text(source.checksum_url)
        _download(source.index_url, staged)
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


def side_load(staged: str | Path, index_path: str | Path | None = None) -> FetchResult:
    """Adopt a locally staged index (Decision 21b's ``--from``).

    The manifest sidecar is required, not optional: side-loading is the path an
    air-gapped user takes, and it is the path where a mismatched index is most
    likely, because the file was carried by hand from somewhere else.
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

    manifest_source = manifest_mod.manifest_path(origin)
    if not manifest_source.exists():
        raise IndexFetchError(
            f"no manifest beside {origin}. Stage {manifest_source.name} alongside "
            "the index -- it is what records the embedding model, and without it "
            "a mismatched index cannot be distinguished from a correct one."
        )
    manifest = manifest_mod.read_manifest(origin)
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
    "BuildResult",
    "FetchResult",
    "IndexFetchError",
    "IndexSource",
    "build_index",
    "check_index",
    "compute_digest",
    "describe_target",
    "fetch_index",
    "resolve_source",
    "side_load",
]
