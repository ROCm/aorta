"""The manifest that travels with a chat index, and the checks it makes possible.

Decision 20a. An index is valid for exactly one tuple of (embedding model,
dimensions, chunk params, store version, source commit), and the failure mode
when that tuple is wrong is the reason this module exists: a mismatched index
does not raise. It answers, fluently, from vectors that were never comparable
to the query's. For a debugging assistant that is worse than an outage, because
the user has no signal that anything is wrong.

So the policy is asymmetric on purpose:

* **Source drift warns.** An index built two weeks and forty commits ago is
  still mostly right, and refusing would leave the user with nothing.
* **An embedding-model or dimension mismatch refuses.** There is no partially
  correct answer available, and a warning printed above a confident answer is a
  warning nobody reads.

The manifest is a sidecar JSON file next to the ``.sqlite``, not a table inside
it, so CI can publish and a user can inspect it without loading sqlite-vec.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Bumped when a field is removed or its meaning changes. A reader tolerates a
#: manifest carrying *extra* keys (a newer builder adding a field must not
#: strand an older client) but refuses one it cannot interpret at all.
SCHEMA_VERSION = 1

#: The on-disk store this manifest describes. Recorded rather than assumed
#: because the format changed once already (Chroma -> sqlite-vec) and a
#: published artifact outlives the decision that produced it.
STORE_NAME = "sqlite-vec"

#: Suffixes of the two files published alongside the index. Both are derived
#: from the index filename so a side-loaded set stays self-describing after a
#: user renames it.
MANIFEST_SUFFIX = ".manifest.json"
CHECKSUM_SUFFIX = ".sha256"

#: Read/written in 1 MiB blocks: the index is tens of megabytes, and hashing it
#: must not hold a second copy in memory on a node that is already tight.
_HASH_BLOCK = 1024 * 1024


class ManifestError(RuntimeError):
    """The manifest is missing, unreadable, or not a manifest."""


class IndexMismatchError(RuntimeError):
    """The index cannot answer queries from this install's embedding provider.

    Raised rather than logged. See the module docstring: the alternative to
    raising is a plausible wrong answer.
    """


def manifest_path(index_path: str | Path) -> Path:
    """Sidecar manifest path for an index file."""
    return Path(str(index_path) + MANIFEST_SUFFIX)


def checksum_path(index_path: str | Path) -> Path:
    """Sidecar SHA256 path for an index file."""
    return Path(str(index_path) + CHECKSUM_SUFFIX)


def sha256_file(path: str | Path) -> str:
    """Hex SHA256 of a file, read incrementally."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Manifest:
    """What a published index says about itself.

    ``index_sha256`` covers the ``.sqlite`` only. The manifest cannot hash
    itself, which is why the published set also carries a ``.sha256`` file: the
    checksum is verified against the download before the manifest is trusted.
    """

    aorta_version: str
    aorta_sha: str
    embedding_provider: str
    embedding_model: str
    dimensions: int
    collection: str
    chunk_size: int
    chunk_overlap: int
    index_sha256: str
    schema_version: int = SCHEMA_VERSION
    aorta_tag: str = ""
    store: str = STORE_NAME
    store_version: str = ""
    built_at: str = ""
    corpus_digest: str = ""
    corpus_roots: list[str] = field(default_factory=list)
    file_count: int = 0
    chunk_count: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Manifest:
        """Build from a parsed manifest, dropping keys this version predates.

        Forward-tolerant by design: a newer CI job adding a field must not make
        its index unreadable to an older client, since the two are published and
        installed independently.
        """
        if not isinstance(raw, dict):
            raise ManifestError(f"manifest is a {type(raw).__name__}, not an object")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            logger.debug("Ignoring unknown manifest key(s): %s", ", ".join(unknown))
        try:
            return cls(**{key: value for key, value in raw.items() if key in known})
        except TypeError as exc:
            raise ManifestError(f"manifest is missing required field(s): {exc}") from exc

    def describe(self) -> str:
        """One line for ``doctor`` and for a warning that has to name the index."""
        source = self.aorta_tag or (self.aorta_sha[:7] if self.aorta_sha else "unknown")
        return (
            f"aorta {self.aorta_version} ({source}), {self.embedding_model} "
            f"@ {self.dimensions}d, chunks {self.chunk_size}/{self.chunk_overlap}, "
            f"built {self.built_at or 'unknown'}"
        )


def now_stamp() -> str:
    """Build timestamp, UTC and second-resolution."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_manifest(index_path: str | Path, manifest: Manifest) -> Path:
    """Write the sidecar manifest and the sidecar checksum, returning the former."""
    target = manifest_path(index_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.to_json(), encoding="utf-8")
    # Two spaces is sha256sum's own format, so `sha256sum -c` works on the
    # published file without reformatting.
    checksum_path(index_path).write_text(
        f"{manifest.index_sha256}  {Path(index_path).name}\n", encoding="utf-8"
    )
    return target


def read_manifest(index_path: str | Path) -> Manifest:
    """Read the sidecar manifest for an index file.

    Raises:
        ManifestError: If it is absent, unparseable, or from a schema this
            version cannot interpret.
    """
    path = manifest_path(index_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(
            f"no manifest beside the index at {index_path}.\n"
            "An index without one cannot be checked against this install's "
            "embedding model, which is the check that stops a silently "
            "mismatched index answering from the wrong vectors. Re-fetch with "
            "'aorta chat index fetch', or rebuild with 'aorta chat index build'."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not read the manifest at {path}: {exc}") from exc

    manifest = Manifest.from_dict(raw)
    if manifest.schema_version > SCHEMA_VERSION:
        raise ManifestError(
            f"the index at {index_path} carries manifest schema version "
            f"{manifest.schema_version}, but this aorta understands "
            f"{SCHEMA_VERSION}. Upgrade aorta, or rebuild the index locally "
            "with 'aorta chat index build'."
        )
    return manifest


@dataclass
class ValidationReport:
    """Outcome of checking a manifest against the configured provider."""

    manifest: Manifest
    warnings: list[str] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.refusals and not self.warnings

    def raise_if_refused(self, index_path: str | Path) -> None:
        if self.refusals:
            raise IndexMismatchError(_refusal_text(index_path, self.refusals, self.manifest))


def _refusal_text(index_path: str | Path, refusals: list[str], manifest: Manifest) -> str:
    """Compose the refusal. Its job is to be impossible to skim past.

    It leads with the consequence rather than the mismatch, because the
    mismatch is not self-evidently serious to someone who just wants an answer,
    and it ends with three concrete commands, because a refusal the user cannot
    act on gets worked around.
    """
    rule = "=" * 72
    lines = [
        "",
        rule,
        f"REFUSING to query the chat index at {index_path}",
        rule,
        "",
        "It was not built by the embedding provider this install queries with, so",
        "every retrieval would compare vectors that are not comparable.",
        "",
        "This would not error. You would get confident answers assembled from the",
        "wrong chunks, with nothing on screen to say so.",
        "",
        *(f"  - {line}" for line in refusals),
        "",
        f"Index built as: {manifest.describe()}",
        "",
        "Resolve it by one of:",
        "  aorta chat index fetch     download the index matching this install",
        "  aorta chat index build     rebuild locally with the configured provider",
        "  aorta chat doctor          show what the two sides currently disagree on",
        "",
    ]
    return "\n".join(lines)


def validate(
    manifest: Manifest,
    *,
    embedding_model: str,
    collection: str,
    dimensions: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    installed_version: str = "",
    installed_sha: str = "",
) -> ValidationReport:
    """Check a manifest against what this install would query with.

    ``dimensions`` is optional because the caller may not know it before the
    model is loaded, and loading it costs a 65 MB download on a cold cache. The
    model and collection checks are the load-bearing ones; the dimension check
    is a second, cheaper net for a model whose dimensions changed under a
    stable name.
    """
    report = ValidationReport(manifest=manifest)

    if manifest.embedding_model != embedding_model:
        report.refusals.append(
            f"embedding model: index was built with {manifest.embedding_model!r}, "
            f"this install queries with {embedding_model!r}"
        )
    if manifest.collection != collection:
        report.refusals.append(
            f"collection: index holds {manifest.collection!r}, this install reads "
            f"{collection!r}"
        )
    if dimensions is not None and manifest.dimensions != dimensions:
        report.refusals.append(
            f"dimensions: index was built at {manifest.dimensions}, this install "
            f"produces {dimensions}"
        )

    if manifest.store != STORE_NAME:
        report.refusals.append(
            f"store format: index is {manifest.store!r}, this aorta reads {STORE_NAME!r}"
        )

    # Chunk params do not invalidate an index -- the vectors are still the same
    # model's, and the chunker only ran at build time -- but a mismatch means
    # retrieved spans are not the size the prompt budget was tuned for.
    if chunk_size is not None and manifest.chunk_size != chunk_size:
        report.warnings.append(
            f"chunk size: index was built at {manifest.chunk_size}, configured "
            f"value is {chunk_size}; retrieved spans will not be the size this "
            "install expects"
        )
    if chunk_overlap is not None and manifest.chunk_overlap != chunk_overlap:
        report.warnings.append(
            f"chunk overlap: index was built at {manifest.chunk_overlap}, "
            f"configured value is {chunk_overlap}"
        )

    if installed_version and manifest.aorta_version != installed_version:
        report.warnings.append(
            f"source drift: index was built from aorta {manifest.aorta_version}"
            f"{f' ({manifest.aorta_sha[:7]})' if manifest.aorta_sha else ''}, this "
            f"install is {installed_version}. Answers may cite code that has "
            "since changed; refresh with 'aorta chat index fetch'"
        )
    elif installed_sha and manifest.aorta_sha and not manifest.aorta_sha.startswith(installed_sha):
        report.warnings.append(
            f"source drift: index was built at {manifest.aorta_sha[:7]}, this "
            f"install reports {installed_sha}. Refresh with 'aorta chat index fetch'"
        )

    return report


__all__ = [
    "CHECKSUM_SUFFIX",
    "MANIFEST_SUFFIX",
    "SCHEMA_VERSION",
    "STORE_NAME",
    "IndexMismatchError",
    "Manifest",
    "ManifestError",
    "ValidationReport",
    "checksum_path",
    "manifest_path",
    "now_stamp",
    "read_manifest",
    "sha256_file",
    "validate",
    "write_manifest",
]
