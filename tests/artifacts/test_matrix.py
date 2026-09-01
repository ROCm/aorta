"""Tests for the ``matrix.json`` reader.

The theme running through most of these: a field that was not in the
artifact must not arrive at the caller as a zero. Several of them assert a
``TypeError`` on purpose -- that is the designed behaviour, because a
consumer comparing an absent ``failure_rate`` should be stopped rather than
told "no failures".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.artifacts import (
    MATRIX_SCHEMA_VERSION,
    ArtifactReadError,
    MissingFieldError,
    parse_matrix,
    parse_matrix_cell,
    read_matrix,
)


def _cell(**overrides) -> dict:
    """A complete cell as ``write_matrix_json`` publishes it, minus the fields
    this reader does not model."""
    doc = {
        "name": "bf16_tf32",
        "mitigations": ["shampoo_precond_bf16"],
        "trials": 16,
        "passed_count": 4,
        "failed_count": 12,
        "failure_rate": 0.75,
        "failure_hints": [["residual NaN in Shampoo preconditioner", 12]],
        "exit_status_counts": {"ok": 4, "workload_failed": 12},
        "error": None,
    }
    doc.update(overrides)
    return doc


def _cell_without(*keys: str, **overrides) -> dict:
    doc = _cell(**overrides)
    for key in keys:
        del doc[key]
    return doc


def _matrix(cells=None, **overrides) -> dict:
    doc = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "workload": "recom_repro",
        "ticket": "RECOM-1",
        "run_timestamp": "2026-01-01T00-00-00",
        "steps_per_trial": 1000,
        "trials_per_cell": 16,
        "cells": [_cell()] if cells is None else cells,
    }
    doc.update(overrides)
    return doc


# ---- the happy path -------------------------------------------------------


def test_parses_the_documented_field_contract():
    matrix = parse_matrix(_matrix())

    assert matrix.workload == "recom_repro"
    assert matrix.ticket == "RECOM-1"
    assert matrix.run_timestamp == "2026-01-01T00-00-00"
    assert matrix.steps_per_trial == 1000
    assert matrix.trials_per_cell == 16
    assert matrix.missing_fields == ()

    cell = matrix.cells[0]
    assert cell.name == "bf16_tf32"
    assert cell.mitigations == ("shampoo_precond_bf16",)
    assert cell.failed_count == 12
    assert cell.failure_rate == pytest.approx(0.75)
    assert cell.workload_failed_count == 12
    assert cell.error is None
    assert cell.missing_fields == ()


def test_cell_lookup_by_name():
    matrix = parse_matrix(_matrix(cells=[_cell(name="a"), _cell(name="b")]))

    assert matrix.cell("b") is matrix.cells[1]
    assert matrix.cell("nope") is None


def test_raw_preserves_unmodelled_keys():
    matrix = parse_matrix(_matrix(cells=[_cell(step_time_source="per_step")]))

    assert matrix.cells[0].raw["step_time_source"] == "per_step"


# ---- absence is not zero --------------------------------------------------


def test_absent_failure_rate_is_none_and_recorded():
    cell = parse_matrix_cell(_cell_without("failure_rate"))

    assert cell.failure_rate is None
    assert "failure_rate" in cell.missing_fields


def test_absent_failure_rate_refuses_the_repro_comparison():
    """The hazard this module exists for.

    ``write_matrix_json`` re-adds ``failure_rate`` by hand because it is a
    ``@property``. If that line were ever lost, a consumer doing
    ``float(cell.get("failure_rate") or 0.0) > 0`` would read every cell as
    clean and report a real reproduction as a passing run. Reading through
    this type, the same comparison raises.
    """
    cell = parse_matrix_cell(_cell_without("failure_rate"))

    with pytest.raises(TypeError):
        _ = cell.failure_rate > 0


def test_present_zero_is_a_real_zero():
    cell = parse_matrix_cell(_cell(failure_rate=0.0, failed_count=0))

    assert cell.failure_rate == 0.0
    assert cell.failed_count == 0
    assert cell.missing_fields == ()
    assert (cell.failure_rate > 0) is False


def test_null_failure_rate_counts_as_unreadable():
    cell = parse_matrix_cell(_cell(failure_rate=None))

    assert cell.failure_rate is None
    assert "failure_rate" in cell.missing_fields


def test_string_failure_rate_is_not_coerced():
    cell = parse_matrix_cell(_cell(failure_rate="0.75"))

    assert cell.failure_rate is None
    assert "failure_rate" in cell.missing_fields


def test_bool_is_not_a_count():
    cell = parse_matrix_cell(_cell(failed_count=True))

    assert cell.failed_count is None
    assert "failed_count" in cell.missing_fields


def test_absent_top_level_fields_are_recorded():
    doc = _matrix()
    del doc["steps_per_trial"]
    del doc["trials_per_cell"]

    matrix = parse_matrix(doc)

    assert matrix.steps_per_trial is None
    assert matrix.trials_per_cell is None
    assert set(matrix.missing_fields) == {"steps_per_trial", "trials_per_cell"}


def test_null_ticket_is_a_value_not_a_gap():
    """An untracked run legitimately carries ``ticket: null``."""
    matrix = parse_matrix(_matrix(ticket=None))

    assert matrix.ticket is None
    assert "ticket" not in matrix.missing_fields


def test_errored_cell_keeps_its_error_message():
    """A whole-cell error zeroes every count, so the message is the only thing
    separating "nothing failed" from "nothing ran"."""
    cell = parse_matrix_cell(
        _cell(
            trials=0, passed_count=0, failed_count=0, failure_rate=0.0, error="docker pull failed"
        )
    )

    assert cell.failed_count == 0
    assert cell.error == "docker pull failed"
    assert "error" not in cell.missing_fields


def test_absent_error_key_is_distinguishable_from_explicit_null():
    with_null = parse_matrix_cell(_cell(error=None))
    without = parse_matrix_cell(_cell_without("error"))

    assert with_null.error is None and "error" not in with_null.missing_fields
    assert without.error is None and "error" in without.missing_fields


# ---- require() ------------------------------------------------------------


def test_require_raises_naming_the_absent_fields():
    cell = parse_matrix_cell(_cell_without("failure_rate", "failed_count"))

    with pytest.raises(MissingFieldError) as excinfo:
        cell.require("failure_rate", "failed_count", "exit_status_counts")

    assert set(excinfo.value.fields) == {"failure_rate", "failed_count"}
    assert "failure_rate" in str(excinfo.value)


def test_require_ignores_fields_the_caller_did_not_ask_for():
    cell = parse_matrix_cell(_cell_without("mitigations"))

    cell.require("failure_rate", "failed_count")


def test_require_with_no_arguments_demands_everything():
    complete = parse_matrix_cell(_cell())
    complete.require()

    with pytest.raises(MissingFieldError):
        parse_matrix_cell(_cell_without("name")).require()


# ---- failure_hints --------------------------------------------------------


def test_failure_hints_pair_shape_carries_the_trial_count():
    cell = parse_matrix_cell(_cell(failure_hints=[["nan detected", 3]]))

    assert [(h.text, h.trials) for h in cell.failure_hints] == [("nan detected", 3)]


def test_failure_hints_bare_string_shape_has_no_count():
    """aorta emits pairs; bare strings turn up in hand-written fixtures and
    are worth reading rather than rejecting."""
    cell = parse_matrix_cell(_cell(failure_hints=["nan detected"]))

    assert [(h.text, h.trials) for h in cell.failure_hints] == [("nan detected", None)]


def test_empty_hint_list_is_not_the_same_as_no_hint_field():
    empty = parse_matrix_cell(_cell(failure_hints=[]))
    absent = parse_matrix_cell(_cell_without("failure_hints"))

    assert empty.failure_hints == ()
    assert "failure_hints" not in empty.missing_fields
    assert absent.failure_hints is None
    assert "failure_hints" in absent.missing_fields


def test_malformed_hint_entry_makes_the_field_unreadable():
    cell = parse_matrix_cell(_cell(failure_hints=[{"hint": "nan"}]))

    assert cell.failure_hints is None
    assert "failure_hints" in cell.missing_fields


# ---- exit_status_counts ---------------------------------------------------


def test_workload_failed_is_zero_when_the_histogram_omits_the_status():
    """The producer counts every trial into the histogram, so a status that is
    not in it did not occur -- that is a real zero, not an absence."""
    cell = parse_matrix_cell(_cell(exit_status_counts={"ok": 16}))

    assert cell.workload_failed_count == 0


def test_workload_failed_is_unknown_when_the_histogram_is_absent():
    cell = parse_matrix_cell(_cell_without("exit_status_counts"))

    assert cell.workload_failed_count is None
    with pytest.raises(TypeError):
        _ = cell.workload_failed_count > 0


def test_non_integer_histogram_value_makes_the_field_unreadable():
    cell = parse_matrix_cell(_cell(exit_status_counts={"ok": "four"}))

    assert cell.exit_status_counts is None
    assert "exit_status_counts" in cell.missing_fields


def test_non_string_mitigation_makes_the_list_unreadable():
    """Dropping the bad element would understate what the cell applied."""
    cell = parse_matrix_cell(_cell(mitigations=["tf32_off", 7]))

    assert cell.mitigations is None
    assert "mitigations" in cell.missing_fields


# ---- cell-level drift summary --------------------------------------------


def test_missing_cell_fields_is_empty_for_a_complete_matrix():
    assert parse_matrix(_matrix()).missing_cell_fields() == {}


def test_missing_cell_fields_names_each_incomplete_cell():
    matrix = parse_matrix(_matrix(cells=[_cell(name="a"), _cell_without("failure_rate", name="b")]))

    assert matrix.missing_cell_fields() == {"b": ("failure_rate",)}


def test_unnamed_cell_is_keyed_by_index():
    matrix = parse_matrix(_matrix(cells=[_cell_without("name")]))

    assert list(matrix.missing_cell_fields()) == ["<cell 0>"]


def test_absent_cells_list_is_recorded_not_silently_empty():
    doc = _matrix()
    del doc["cells"]

    matrix = parse_matrix(doc)

    assert matrix.cells == ()
    assert "cells" in matrix.missing_fields


def test_non_object_cell_entry_flags_the_list():
    matrix = parse_matrix(_matrix(cells=[_cell(), "surprise"]))

    assert len(matrix.cells) == 1
    assert "cells" in matrix.missing_fields


# ---- schema_version -------------------------------------------------------


def test_schema_version_matching_the_producer_is_supported():
    matrix = parse_matrix(_matrix())

    assert matrix.schema_status == "supported"
    assert matrix.schema_note is None


def test_newer_schema_version_is_read_anyway_but_flagged():
    matrix = parse_matrix(_matrix(schema_version=MATRIX_SCHEMA_VERSION + 1))

    assert matrix.schema_status == "newer"
    assert "newer" in matrix.schema_note
    # Tolerance is the point: the fields we do recognise still parse.
    assert matrix.cells[0].failure_rate == pytest.approx(0.75)


def test_older_schema_version_is_flagged_as_older():
    matrix = parse_matrix(_matrix(schema_version=0))

    assert matrix.schema_status == "older"


@pytest.mark.parametrize("value", ["1", 1.0, None, True])
def test_non_integer_schema_version_is_unknown(value):
    doc = _matrix()
    doc["schema_version"] = value

    assert parse_matrix(doc).schema_status == "unknown"


def test_absent_schema_version_is_unknown():
    doc = _matrix()
    del doc["schema_version"]

    matrix = parse_matrix(doc)

    assert matrix.schema_status == "unknown"
    assert matrix.schema_version is None


# ---- file-level errors ----------------------------------------------------


def test_read_matrix_round_trips_a_file(tmp_path: Path):
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(_matrix()), encoding="utf-8")

    matrix = read_matrix(path)

    assert matrix.source_path == path
    assert matrix.cells[0].failed_count == 12


def test_read_matrix_raises_when_the_file_is_absent(tmp_path: Path):
    with pytest.raises(ArtifactReadError):
        read_matrix(tmp_path / "nope.json")


def test_read_matrix_raises_on_invalid_json(tmp_path: Path):
    path = tmp_path / "matrix.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ArtifactReadError):
        read_matrix(path)


def test_read_matrix_raises_when_the_document_is_not_an_object(tmp_path: Path):
    path = tmp_path / "matrix.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ArtifactReadError):
        read_matrix(path)
