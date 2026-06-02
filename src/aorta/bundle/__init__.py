"""``aorta.bundle`` -- shareable artifact bundler for ``aorta probe`` runs.

Public surface (issue #196):

* :func:`bundle_run_dir` -- programmatic entry point. Stages a probe
  run directory through the configured :class:`Redactor`, writes a
  ``manifest.json``, and packs everything into a single ``.tar.gz``.
  Returns the absolute path of the bundle.
* :class:`Manifest`, :class:`FileRecord` -- typed view of the
  ``manifest.json`` shape so downstream tooling does not have to
  re-derive it.
* :class:`Redactor`, :class:`IdentityRedactor`, :class:`RedactionCounts`
  -- the redactor contract. ``aorta probe`` Phase 3 (issue #188) ships
  a real implementation; until then bundles use the
  :class:`IdentityRedactor` (no scrubbing, zero counts).
* :class:`BundleError` and its subclasses -- typed exceptions the CLI
  layer maps to ``click.ClickException``.
* :data:`MANIFEST_SCHEMA_VERSION`, :data:`MANIFEST_FILENAME` --
  constants other modules can reference without duplicating the
  literals.

The CLI lives in :mod:`aorta.cli.bundle` and is a thin Click shim
around :func:`bundle_run_dir` (mirrors the ``aorta probe`` /
``aorta run`` discipline).
"""

from __future__ import annotations

from aorta.bundle.errors import (
    BundleAbortedError,
    BundleError,
    BundleIOError,
    EmptyRunDirError,
    NoTicketError,
    RunDirNotFoundError,
    UnsafeSymlinkError,
)
from aorta.bundle.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    FileRecord,
    Manifest,
)
from aorta.bundle.redactor import IdentityRedactor, RedactionCounts, Redactor
from aorta.bundle.writer import bundle_run_dir, resolve_ticket

__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "BundleAbortedError",
    "BundleError",
    "BundleIOError",
    "EmptyRunDirError",
    "FileRecord",
    "IdentityRedactor",
    "Manifest",
    "NoTicketError",
    "RedactionCounts",
    "Redactor",
    "RunDirNotFoundError",
    "UnsafeSymlinkError",
    "bundle_run_dir",
    "resolve_ticket",
]
