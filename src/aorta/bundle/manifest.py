"""Manifest dataclasses + JSON shape for ``aorta bundle`` (issue #196).

The manifest lives at ``<bundle-name>/manifest.json`` inside every
bundle and is the single authoritative record of:

* what was redacted (per-file counts),
* what was bundled (relative paths inside the tarball),
* who wrote the bundle (aorta version, redactor kind, and the source
  run dir's leaf name only -- never its absolute path, which would
  leak operator/customer path details off the source machine).

Schema is versioned via :data:`MANIFEST_SCHEMA_VERSION`. Phase 3 of
issue #188 will not need a bump -- it only fills in the existing
``RedactionCounts`` fields with non-zero values.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Filename used inside both the staging tree and the tarball.
MANIFEST_FILENAME = "manifest.json"

#: Bumped only on a non-additive change. Adding fields with
#: documented defaults does NOT bump this. The reader in
#: :func:`Manifest.from_json` rejects schema versions it does not
#: understand; current readers are inside
#: ``tests/bundle/test_writer.py`` only, but a downstream tool that
#: parses the manifest can pin the same constant.
MANIFEST_SCHEMA_VERSION = 1


def _to_utc(now: _dt.datetime) -> _dt.datetime:
    """Normalise an arbitrary datetime to UTC.

    Bundle artifacts ALWAYS label timestamps as UTC ("Z" suffix in
    manifests; no suffix but UTC-convention in filenames). Callers
    (tests, future Phase 3 hooks) may inject a naive or
    non-UTC-aware ``now`` -- we treat a naive datetime as already
    being in UTC (matches ``datetime.now(timezone.utc)``'s shape
    once tz is stripped) and convert any aware datetime to UTC
    before formatting. The alternative (silently slapping ``Z`` on
    a local-time naive datetime) is the bug Copilot caught on PR
    #199.
    """
    if now.tzinfo is None:
        return now.replace(tzinfo=_dt.timezone.utc)
    return now.astimezone(_dt.timezone.utc)


@dataclass(frozen=True)
class FileRecord:
    """One row of ``manifest.json::files``.

    ``path`` is relative to the bundle's top-level directory (which
    is also the tarball's top-level entry). Forward slashes
    regardless of host OS so a manifest written on Windows is
    readable on Linux without normalisation.
    """

    path: str
    env_keys_removed: int = 0
    paths_rewritten: int = 0
    ips_rewritten: int = 0
    bytes_in: int = 0
    bytes_out: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class Manifest:
    """``manifest.json`` shape -- see ``docs/probe-188/bundle.md``.

    ``redaction_applied`` is the boolean roll-up of the per-file
    counts (it's ``True`` iff at least one ``FileRecord`` carries a
    non-zero env / path / ip count). Recorded explicitly so a
    consumer does not have to scan every record to know whether
    scrubbers ran -- the field is the index, not the source of
    truth. The actual roll-up is computed in :meth:`from_files`
    so callers cannot accidentally mark a bundle as "redaction
    applied" while submitting zero-count records.
    """

    schema_version: int
    ticket: str
    created_at: str
    aorta_version: str
    #: Provenance ONLY: the leaf directory name of the source run dir
    #: (its basename), never the absolute path. A bundle is a
    #: shareable artifact, so recording ``/home/<user>/<customer>/...``
    #: would leak workstation usernames, mount points, and customer
    #: directory names off the source machine (PR #199 security
    #: review). The leaf name is the per-ticket segment, which is
    #: already non-sensitive (it equals ``ticket`` in normal probe
    #: output).
    source_run_dir: str
    redaction_applied: bool
    redactor_kind: str
    files: tuple[FileRecord, ...] = field(default_factory=tuple)

    @classmethod
    def from_files(
        cls,
        ticket: str,
        source_run_dir: Path,
        redactor_kind: str,
        aorta_version: str,
        files: list[FileRecord],
        now: _dt.datetime | None = None,
    ) -> Manifest:
        """Build a :class:`Manifest` and derive ``redaction_applied``.

        ``now`` is optional so tests can pin a deterministic
        timestamp; default uses ``datetime.now(timezone.utc)``. A
        non-UTC or naive ``now`` is normalised by :func:`_to_utc`
        before formatting so the ``Z`` suffix in the on-disk
        ``created_at`` always reflects the real UTC clock (not
        the caller's local time silently labelled UTC).

        The timestamp is rendered in ISO-8601 with the ``Z`` suffix
        so downstream tools can parse it with
        ``datetime.fromisoformat`` (Python 3.11+) or any standard
        ISO-8601 parser.
        """
        now = _to_utc(now or _dt.datetime.now(_dt.timezone.utc))
        applied = any(f.env_keys_removed or f.paths_rewritten or f.ips_rewritten for f in files)
        return cls(
            schema_version=MANIFEST_SCHEMA_VERSION,
            ticket=ticket,
            created_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            aorta_version=aorta_version,
            # Basename only -- NEVER the absolute path. See the
            # ``source_run_dir`` field docstring for the rationale.
            source_run_dir=source_run_dir.name,
            redaction_applied=applied,
            redactor_kind=redactor_kind,
            files=tuple(files),
        )

    def to_json(self) -> str:
        doc: dict[str, object] = {
            "schema_version": self.schema_version,
            "ticket": self.ticket,
            "created_at": self.created_at,
            "aorta_version": self.aorta_version,
            "source_run_dir": self.source_run_dir,
            "redaction_applied": self.redaction_applied,
            "redactor_kind": self.redactor_kind,
            "files": [f.to_dict() for f in self.files],
        }
        return json.dumps(doc, indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, raw: str) -> Manifest:
        """Round-trip the manifest from its on-disk JSON form.

        Used by tests + downstream tooling. Rejects unknown schema
        versions up front so a future bump cannot be silently mis-read.
        """
        doc = json.loads(raw)
        version = doc.get("schema_version")
        if version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"manifest.json: unsupported schema_version {version!r}; "
                f"this build understands version {MANIFEST_SCHEMA_VERSION}"
            )
        files = tuple(
            FileRecord(
                path=str(f["path"]),
                env_keys_removed=int(f.get("env_keys_removed", 0)),
                paths_rewritten=int(f.get("paths_rewritten", 0)),
                ips_rewritten=int(f.get("ips_rewritten", 0)),
                bytes_in=int(f.get("bytes_in", 0)),
                bytes_out=int(f.get("bytes_out", 0)),
            )
            for f in doc.get("files", [])
        )
        return cls(
            schema_version=version,
            ticket=str(doc["ticket"]),
            created_at=str(doc["created_at"]),
            aorta_version=str(doc["aorta_version"]),
            source_run_dir=str(doc["source_run_dir"]),
            redaction_applied=bool(doc.get("redaction_applied", False)),
            redactor_kind=str(doc.get("redactor_kind", "identity")),
            files=files,
        )

    def total_bytes_in(self) -> int:
        return sum(f.bytes_in for f in self.files)

    def total_bytes_out(self) -> int:
        return sum(f.bytes_out for f in self.files)


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "FileRecord",
    "Manifest",
]
