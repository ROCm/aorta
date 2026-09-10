"""Typed reader for ``matrix.json``, the per-cell triage matrix.

Producer: :func:`aorta.triage.output.write_matrix_json`, serialising
:class:`aorta.triage.matrix.CellStats`. That producer's per-cell shape is
whatever ``asdict(CellStats)`` yields, plus a handful of ``@property``
values re-added by hand -- ``failure_rate`` among them. This reader models
only the fields a failure-triage consumer reasons over; everything else is
still reachable through :attr:`MatrixCell.raw`.

Nothing here classifies a run. Deciding what a failing cell *means* is the
consumer's call, and deliberately stays out of this module.
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
    classify_integer_schema,
    load_json_object,
)

#: Conventional filename, both in a triage run directory and inside a bundle.
MATRIX_FILENAME = "matrix.json"

#: The ``schema_version`` this reader was written against. ``write_matrix_json``
#: emits a hardcoded integer; unlike env.json there is no changelog behind it,
#: so any other value is reported rather than assumed additive.
MATRIX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FailureHint:
    """One deduplicated failure hint from a cell's trials.

    aorta emits ``[text, trials]`` pairs, where ``trials`` counts how many of
    the cell's trials produced that exact hint. Bare strings are also accepted
    -- they turn up in hand-written fixtures -- and leave ``trials`` as
    ``None`` rather than inventing a count of one.
    """

    text: str
    trials: int | None


@dataclass(frozen=True)
class MatrixCell(HasMissingFields):
    """One entry of ``matrix.json::cells``.

    Every field is ``None`` when the artifact did not carry it in a readable
    form, and its name then appears in :attr:`missing_fields`. That is the
    whole point of the type: ``cell.failure_rate > 0`` raises ``TypeError``
    on an absent field instead of quietly evaluating to ``False``, which is
    what a hand-rolled ``float(cell.get("failure_rate") or 0.0)`` would do.

    ``error`` is the exception to the shape, because ``null`` is its normal
    value: a cell that ran carries ``error: null``. Use ``"error" in
    missing_fields`` to tell "the run reported no cell-level error" from "this
    artifact never had the key".
    """

    name: str | None
    mitigations: tuple[str, ...] | None
    trials: int | None
    passed_count: int | None
    failed_count: int | None
    failure_rate: float | None
    failure_hints: tuple[FailureHint, ...] | None
    exit_status_counts: Mapping[str, int] | None
    error: str | None
    missing_fields: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def workload_failed_count(self) -> int | None:
        """Trials whose ``exit_status`` was ``workload_failed``.

        ``None`` only when the histogram itself is unavailable. When the
        histogram is present but lacks the key the answer really is zero:
        the producer counts every trial into it, so an absent status did not
        occur.
        """
        if self.exit_status_counts is None:
            return None
        return self.exit_status_counts.get("workload_failed", 0)


@dataclass(frozen=True)
class MatrixArtifact(HasMissingFields):
    """A parsed ``matrix.json``.

    :attr:`missing_fields` covers the top level only; per-cell gaps live on
    each :class:`MatrixCell` and are summarised by :meth:`missing_cell_fields`.
    """

    schema_version: Any
    schema_status: SchemaStatus
    schema_note: str | None
    workload: str | None
    ticket: str | None
    run_timestamp: str | None
    steps_per_trial: int | None
    trials_per_cell: int | None
    cells: tuple[MatrixCell, ...]
    missing_fields: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def cell(self, name: str) -> MatrixCell | None:
        """Return the cell called *name*, or ``None`` if the matrix has no such cell."""
        for entry in self.cells:
            if entry.name == name:
                return entry
        return None

    def missing_cell_fields(self) -> dict[str, tuple[str, ...]]:
        """Map each incomplete cell to the fields it did not carry.

        The schema-drift check in reader form: an empty result means every
        cell carried every modelled field. Cells whose own ``name`` is
        unreadable are keyed by their index, since there is nothing else to
        call them.
        """
        gaps: dict[str, tuple[str, ...]] = {}
        for index, entry in enumerate(self.cells):
            if entry.missing_fields:
                gaps[entry.name if entry.name is not None else f"<cell {index}>"] = (
                    entry.missing_fields
                )
        return gaps


def _parse_failure_hints(value: Any) -> tuple[FailureHint, ...] | None:
    if not isinstance(value, list):
        return None
    hints: list[FailureHint] = []
    for entry in value:
        if isinstance(entry, str):
            hints.append(FailureHint(text=entry, trials=None))
            continue
        # asdict() keeps the (text, count) tuple, which json.dumps writes as a
        # two-element array.
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], int)
            and not isinstance(entry[1], bool)
        ):
            hints.append(FailureHint(text=entry[0], trials=entry[1]))
            continue
        return None
    return tuple(hints)


def parse_matrix_cell(doc: Mapping[str, Any]) -> MatrixCell:
    """Parse one ``cells[]`` entry."""
    reader = FieldReader(doc)
    name = reader.string("name")
    mitigations = reader.string_tuple("mitigations")
    trials = reader.integer("trials")
    passed_count = reader.integer("passed_count")
    failed_count = reader.integer("failed_count")
    failure_rate = reader.number("failure_rate")
    exit_status_counts = reader.count_map("exit_status_counts")
    error = reader.nullable_string("error")

    if "failure_hints" in doc:
        failure_hints = _parse_failure_hints(doc["failure_hints"])
        if failure_hints is None:
            reader.record_missing("failure_hints")
    else:
        failure_hints = None
        reader.record_missing("failure_hints")

    return MatrixCell(
        name=name,
        mitigations=mitigations,
        trials=trials,
        passed_count=passed_count,
        failed_count=failed_count,
        failure_rate=failure_rate,
        failure_hints=failure_hints,
        exit_status_counts=exit_status_counts,
        error=error,
        missing_fields=reader.missing,
        raw=dict(doc),
    )


def parse_matrix(doc: Mapping[str, Any], source_path: Path | None = None) -> MatrixArtifact:
    """Parse an already-loaded ``matrix.json`` document."""
    reader = FieldReader(doc)
    schema_status, schema_note = classify_integer_schema(
        doc.get("schema_version"), MATRIX_SCHEMA_VERSION
    )
    if schema_status == "unknown":
        # Same reason as ``env.parse_env``: an unreadable schema version is a
        # modelled field this reader could not get, so the no-argument
        # ``require()`` must not pass over it.
        reader.record_missing("schema_version")
    workload = reader.string("workload")
    # ``ticket`` is legitimately null on an untracked run, unlike the rest.
    ticket = reader.nullable_string("ticket")
    run_timestamp = reader.string("run_timestamp")
    steps_per_trial = reader.integer("steps_per_trial")
    trials_per_cell = reader.integer("trials_per_cell")

    raw_cells = doc.get("cells")
    if isinstance(raw_cells, list):
        cells = tuple(parse_matrix_cell(c) for c in raw_cells if isinstance(c, Mapping))
        if len(cells) != len(raw_cells):
            # A non-object entry means the list is not the shape we think it
            # is; say so rather than reporting a shorter matrix than was run.
            reader.record_missing("cells")
    else:
        cells = ()
        reader.record_missing("cells")

    return MatrixArtifact(
        schema_version=doc.get("schema_version"),
        schema_status=schema_status,
        schema_note=schema_note,
        workload=workload,
        ticket=ticket,
        run_timestamp=run_timestamp,
        steps_per_trial=steps_per_trial,
        trials_per_cell=trials_per_cell,
        cells=cells,
        missing_fields=reader.missing,
        raw=dict(doc),
        source_path=source_path,
    )


def read_matrix(path: Path | str) -> MatrixArtifact:
    """Read and parse a ``matrix.json`` file.

    Raises :class:`~aorta.artifacts.ArtifactReadError` when the file is
    unreadable or is not a JSON object -- that is an absent artifact, not a
    partial one. Anything short of that is reported through
    ``missing_fields`` / ``schema_status`` instead of raising.
    """
    resolved = Path(path)
    doc = load_json_object(resolved)
    return parse_matrix(doc, source_path=resolved)


__all__ = [
    "MATRIX_FILENAME",
    "MATRIX_SCHEMA_VERSION",
    "FailureHint",
    "MatrixArtifact",
    "MatrixCell",
    "parse_matrix",
    "parse_matrix_cell",
    "read_matrix",
]
