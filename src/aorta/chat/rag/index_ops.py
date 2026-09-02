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

#: How long a single HTTP request gets. The index is tens of megabytes, so this
#: is generous, but an unbounded download inside a CLI command is a hang.
_HTTP_TIMEOUT = 300

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

    store = SqliteVecStore(path=target, embedding=embeddings, collection=collection)
    try:
        store.reset()
        for start in range(0, len(chunks), _WRITE_BATCH):
            store.add_documents(chunks[start : start + _WRITE_BATCH], provider=provider.describe())
        dimensions = store.dimension()
    finally:
        store.close()

    digest = corpus_mod.corpus_digest(
        documents,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    manifest = manifest_mod.Manifest(
        aorta_version=installed_version(),
        aorta_sha=corpus_mod.head_sha(corpus.base),
        aorta_tag=corpus_mod.head_tag(corpus.base),
        embedding_provider=provider.name,
        embedding_model=settings.embedding_model,
        dimensions=dimensions,
        collection=collection,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        index_sha256=manifest_mod.sha256_file(target),
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
        embedding_model=settings.embedding_model,
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
    """
    target = Path(index_path) if index_path else settings.index_file
    provider = get_provider()
    report = manifest_mod.validate(
        manifest_mod.read_manifest(target),
        embedding_model=settings.embedding_model,
        collection=provider.collection_name(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        installed_version=installed_version(),
    )
    if strict:
        report.raise_if_refused(target)
    return report


# ── fetching ──────────────────────────────────────────────────────────────


def _download(url: str, target: Path) -> None:
    """Stream ``url`` to ``target``, or raise :class:`IndexFetchError`."""
    logger.info("Downloading %s", url)
    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as response:  # noqa: S310
            with open(target, "wb") as handle:
                shutil.copyfileobj(response, handle)
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
    """
    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as response:  # noqa: S310
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


def fetch_index(
    version: str | None = None,
    index_path: str | Path | None = None,
    source: IndexSource | None = None,
) -> FetchResult:
    """Download, verify, validate and install the published index.

    Verification order matters: checksum first, so a corrupt download never
    reaches the manifest parser, then manifest validation, so a mismatched index
    never reaches the destination path. Nothing is installed until both pass.
    """
    dest = Path(index_path) if index_path else settings.index_file
    source = source or resolve_source(version)
    provider = get_provider()
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Staged inside the destination directory so the final move is a rename on
    # one filesystem, and therefore atomic: a concurrent reader sees the old
    # index or the new one, never a half-written file.
    with tempfile.TemporaryDirectory(prefix=".aorta-index-", dir=dest.parent) as staging_dir:
        staged = Path(staging_dir) / ASSET_NAME
        manifest_text = _download_text(source.manifest_url)
        checksum_line = _download_text(source.checksum_url)
        _download(source.index_url, staged)
        checksum = _verify_checksum(staged, checksum_line, source.checksum_url)

        try:
            manifest = manifest_mod.Manifest.from_dict(json.loads(manifest_text))
        except (ValueError, manifest_mod.ManifestError) as exc:
            raise IndexFetchError(
                f"the manifest at {source.manifest_url} is not usable: {exc}"
            ) from exc
        if manifest.index_sha256 and manifest.index_sha256 != checksum:
            raise IndexFetchError(
                "the manifest and the checksum file disagree about the index:\n"
                f"  manifest says {manifest.index_sha256}\n"
                f"  .sha256 says  {checksum}\n"
                "The published set is inconsistent; nothing has been installed."
            )

        report = manifest_mod.validate(
            manifest,
            embedding_model=settings.embedding_model,
            collection=provider.collection_name(),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            installed_version=installed_version(),
        )
        report.raise_if_refused(source.index_url)

        # Sidecars first: a reader that finds an index without a manifest
        # refuses, so an install interrupted here fails safe rather than leaving
        # an unverifiable index in the configured path.
        manifest_mod.manifest_path(dest).write_text(manifest_text, encoding="utf-8")
        manifest_mod.checksum_path(dest).write_text(f"{checksum}  {dest.name}\n", encoding="utf-8")
        staged.replace(dest)

    return FetchResult(
        index_path=dest,
        manifest=manifest,
        source=source.describe(),
        warnings=[*source.notes, *report.warnings],
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

    provider = get_provider()
    report = manifest_mod.validate(
        manifest,
        embedding_model=settings.embedding_model,
        collection=provider.collection_name(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        installed_version=installed_version(),
    )
    report.raise_if_refused(origin)

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origin, dest)
    manifest_mod.write_manifest(dest, manifest)
    return FetchResult(
        index_path=dest,
        manifest=manifest,
        source=f"side-loaded from {origin}",
        warnings=list(report.warnings),
    )


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
    "fetch_index",
    "resolve_source",
    "side_load",
]
