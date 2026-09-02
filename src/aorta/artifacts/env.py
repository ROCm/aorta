"""Typed reader for ``env.json``, the environment snapshot.

Producer: :func:`aorta.instrumentation.environment.capture_to` (and the
``aorta env probe`` CLI), serialising
:class:`aorta.instrumentation.environment.EnvSnapshot`. The full schema is
documented in ``docs/env-probe.md``; that snapshot carries several dozen
top-level blocks and this reader models only the handful a triage consumer
reasons over -- the ROCm version plus how much of the probe actually
succeeded. Everything else stays reachable through :attr:`EnvArtifact.raw`
or :meth:`EnvArtifact.block`.

The probe writes the same shape under more than one filename depending on
how it was invoked (``env.json`` from the CLI and from a per-environment
capture, ``host_env.json`` for a triage run's one host-level capture), so
callers pass an explicit path rather than a directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aorta.artifacts._common import (
    FieldReader,
    HasMissingFields,
    SchemaStatus,
    classify_dotted_schema,
    load_json_object,
)

#: Filenames the env probe writes. ``env.json`` is the CLI default and the
#: per-environment capture; ``host_env.json`` is the triage run's host-level one.
ENV_FILENAME = "env.json"
HOST_ENV_FILENAME = "host_env.json"

#: The ``schema_version`` major this reader was written against. Minor bumps
#: are additive by the probe's own documented policy, so only the major
#: participates in the compatibility check -- see ``classify_dotted_schema``.
ENV_SCHEMA_MAJOR = 1


@dataclass(frozen=True)
class EnvArtifact(HasMissingFields):
    """A parsed ``env.json`` / ``host_env.json``.

    ``partial`` and ``partial_reasons`` are the probe's own honesty signal:
    ``partial`` is true when at least one sub-probe fell back, and each reason
    names the field and the cause. They are worth propagating, because an
    environment fact being ``null`` because the probe could not run is a
    different claim from it being ``null`` because the thing is not installed.

    Every modelled field is ``None`` when the snapshot did not carry it in a
    readable form, and its name then appears in :attr:`missing_fields`. In
    particular an absent ``partial`` does not read as "the probe was clean".
    """

    schema_version: Any
    schema_status: SchemaStatus
    schema_note: str | None
    captured_at: str | None
    partial: bool | None
    partial_reasons: tuple[str, ...] | None
    rocm: Mapping[str, Any] | None
    rocm_version: str | None
    missing_fields: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def block(self, name: str) -> Mapping[str, Any] | None:
        """Return an unmodelled top-level block, or ``None``.

        The escape hatch for the blocks this reader does not type -- the GEMM
        library identities, the driver block, the catalogs. Returns the raw
        mapping as the probe wrote it.

        ``None`` covers both an absent block and one the probe wrote as
        something other than an object, and -- unlike the typed accessors --
        neither case is recorded in :attr:`missing_fields`. A caller that has
        to tell the two apart reads :attr:`raw` directly.
        """
        value = self.raw.get(name)
        return value if isinstance(value, Mapping) else None


def parse_env(doc: Mapping[str, Any], source_path: Path | None = None) -> EnvArtifact:
    """Parse an already-loaded env-snapshot document."""
    reader = FieldReader(doc)
    schema_status, schema_note = classify_dotted_schema(doc.get("schema_version"), ENV_SCHEMA_MAJOR)
    captured_at = reader.string("captured_at")
    partial = reader.boolean("partial")
    partial_reasons = reader.string_tuple("partial_reasons")
    rocm = reader.mapping("rocm")

    # ``rocm.version`` is nullable by contract: the probe records null when it
    # found no version file on an install it did locate. Absent-vs-null is
    # therefore worth keeping apart, and the nested key gets its own dotted
    # entry in ``missing_fields`` so a caller can see which half failed.
    if rocm is None:
        rocm_version = None
        reader.record_missing("rocm.version")
    else:
        nested = FieldReader(rocm)
        rocm_version = nested.nullable_string("version")
        if "version" in nested.missing:
            reader.record_missing("rocm.version")

    return EnvArtifact(
        schema_version=doc.get("schema_version"),
        schema_status=schema_status,
        schema_note=schema_note,
        captured_at=captured_at,
        partial=partial,
        partial_reasons=partial_reasons,
        rocm=rocm,
        rocm_version=rocm_version,
        missing_fields=reader.missing,
        raw=dict(doc),
        source_path=source_path,
    )


def read_env(path: Path | str) -> EnvArtifact:
    """Read and parse an ``env.json`` / ``host_env.json`` file.

    Raises :class:`~aorta.artifacts.ArtifactReadError` when the file is
    unreadable or is not a JSON object. Everything else is reported through
    ``missing_fields`` / ``schema_status`` instead of raising.
    """
    resolved = Path(path)
    doc = load_json_object(resolved)
    return parse_env(doc, source_path=resolved)


__all__ = [
    "ENV_FILENAME",
    "ENV_SCHEMA_MAJOR",
    "HOST_ENV_FILENAME",
    "EnvArtifact",
    "parse_env",
    "read_env",
]
