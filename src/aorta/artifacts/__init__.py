"""Typed readers for aorta's own run artifacts: ``matrix.json`` and ``env.json``.

**Status: internal.** This package is not a supported public API yet. Its
scope is exactly what aorta's own assistant tooling needs today, and names
may change without a deprecation cycle. It nonetheless lives in core rather
than behind an optional extra, so an out-of-tree consumer can
``import aorta.artifacts`` from a base ``pip install amd-aorta``; promoting
it to a supported surface is then a docstring change rather than a move. For
the same reason it depends on the standard library only.

Why the readers exist. Both artifacts are consumed by tools that decide
whether a GPU failure reproduced, and both shapes are pinned loosely.
``matrix.json`` publishes whatever ``asdict(CellStats)`` happens to produce,
and ``failure_rate`` reaches it only because ``write_matrix_json`` re-adds it
by hand -- it is a ``@property``, so ``asdict`` drops it. Hand-rolled parsing
of that tends to look like ``float(cell.get("failure_rate") or 0.0)``, which
turns a dropped key into a confident "every cell passed": a real reproduction
reported as a clean run, with nothing raised anywhere.

So the rule these readers hold to, and the one worth preserving in any later
change: **a field that was absent or unreadable is never presented as zero.**
It is ``None``, and its name is listed in ``missing_fields``. ``None`` is not
quietly comparable -- ``cell.failure_rate > 0`` raises rather than answering
"no" -- and a caller that would rather fail up front can call ``require()``::

    matrix = read_matrix(run_dir / MATRIX_FILENAME)
    for cell in matrix.cells:
        cell.require("failure_rate", "failed_count", "exit_status_counts")

Unknown ``schema_version`` values are reported, not rejected: each artifact
carries ``schema_status`` (``supported`` / ``newer`` / ``older`` / ``unknown``)
and a human-readable ``schema_note``. Refusing an unrecognised version would
make the readers useless against the artifacts already sitting in ticket
attachments.

``result.json`` is deliberately not modelled -- nothing consumes it.
"""

from __future__ import annotations

from aorta.artifacts._common import (
    ArtifactError,
    ArtifactReadError,
    MissingFieldError,
    SchemaStatus,
)
from aorta.artifacts.env import (
    ENV_FILENAME,
    ENV_SCHEMA_MAJOR,
    HOST_ENV_FILENAME,
    EnvArtifact,
    parse_env,
    read_env,
)
from aorta.artifacts.matrix import (
    MATRIX_FILENAME,
    MATRIX_SCHEMA_VERSION,
    FailureHint,
    MatrixArtifact,
    MatrixCell,
    parse_matrix,
    parse_matrix_cell,
    read_matrix,
)

__all__ = [
    "ENV_FILENAME",
    "ENV_SCHEMA_MAJOR",
    "HOST_ENV_FILENAME",
    "MATRIX_FILENAME",
    "MATRIX_SCHEMA_VERSION",
    "ArtifactError",
    "ArtifactReadError",
    "EnvArtifact",
    "FailureHint",
    "MatrixArtifact",
    "MatrixCell",
    "MissingFieldError",
    "SchemaStatus",
    "parse_env",
    "parse_matrix",
    "parse_matrix_cell",
    "read_env",
    "read_matrix",
]
