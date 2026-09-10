"""Tests for the ``env.json`` reader.

Same rule as the matrix reader: an absent ``partial`` must not read as "the
probe was clean", and an absent ``rocm`` block must not read as "no ROCm".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.artifacts import (
    ENV_SCHEMA_MAJOR,
    ArtifactReadError,
    MissingFieldError,
    parse_env,
    read_env,
)


def _env(**overrides) -> dict:
    doc = {
        "schema_version": "1.16",
        "captured_at": "2026-01-01T00:00:00Z",
        "rocm": {"version": "7.2.0", "version_dev": None, "kmd_version": "6.16.13"},
        "hipblaslt": {"package_version": "1.4.0"},
        "partial": False,
        "partial_reasons": [],
    }
    doc.update(overrides)
    return doc


def _env_without(*keys: str, **overrides) -> dict:
    doc = _env(**overrides)
    for key in keys:
        del doc[key]
    return doc


# ---- the happy path -------------------------------------------------------


def test_parses_the_documented_field_contract():
    env = parse_env(_env())

    assert env.schema_status == "supported"
    assert env.captured_at == "2026-01-01T00:00:00Z"
    assert env.rocm_version == "7.2.0"
    assert env.partial is False
    assert env.partial_reasons == ()
    assert env.missing_fields == ()


def test_partial_reasons_are_preserved_in_order():
    env = parse_env(_env(partial=True, partial_reasons=["system_health: rdhc absent", "b"]))

    assert env.partial is True
    assert env.partial_reasons == ("system_health: rdhc absent", "b")
    assert len(env.partial_reasons) == 2


def test_block_reaches_an_unmodelled_top_level_block():
    env = parse_env(_env())

    assert env.block("hipblaslt") == {"package_version": "1.4.0"}
    assert env.block("nics") is None


# ---- absence is not a clean probe ----------------------------------------


def test_absent_partial_is_unknown_not_false():
    env = parse_env(_env_without("partial"))

    assert env.partial is None
    assert "partial" in env.missing_fields
    # A caller writing ``if env.partial:`` would silently conclude "clean",
    # so the strict path exists for exactly this field.
    with pytest.raises(MissingFieldError):
        env.require("partial")


def test_absent_partial_reasons_is_not_an_empty_list():
    absent = parse_env(_env_without("partial_reasons"))
    empty = parse_env(_env(partial_reasons=[]))

    assert absent.partial_reasons is None
    assert "partial_reasons" in absent.missing_fields
    assert empty.partial_reasons == ()
    assert "partial_reasons" not in empty.missing_fields


def test_non_boolean_partial_is_unreadable():
    env = parse_env(_env(partial="yes"))

    assert env.partial is None
    assert "partial" in env.missing_fields


# ---- rocm.version --------------------------------------------------------


def test_null_rocm_version_is_a_value_the_probe_reports():
    """The probe records ``null`` when it located an install but no version
    file, which is a finding rather than a gap in the artifact."""
    env = parse_env(_env(rocm={"version": None, "kmd_version": "6.16.13"}))

    assert env.rocm_version is None
    assert "rocm.version" not in env.missing_fields
    assert env.rocm == {"version": None, "kmd_version": "6.16.13"}


def test_rocm_block_without_a_version_key_records_the_dotted_path():
    env = parse_env(_env(rocm={"kmd_version": "6.16.13"}))

    assert env.rocm_version is None
    assert "rocm.version" in env.missing_fields
    assert "rocm" not in env.missing_fields


def test_absent_rocm_block_records_both_levels():
    env = parse_env(_env_without("rocm"))

    assert env.rocm is None
    assert env.rocm_version is None
    assert set(env.missing_fields) == {"rocm", "rocm.version"}


def test_non_object_rocm_block_is_unreadable():
    env = parse_env(_env(rocm="7.2.0"))

    assert env.rocm is None
    assert "rocm" in env.missing_fields


# ---- schema_version -------------------------------------------------------


@pytest.mark.parametrize("version", ["1.0", "1.11", "1.16", "1.17", "1.20"])
def test_any_minor_within_the_known_major_is_supported(version):
    """The probe bumps its minor for additive changes, so pinning the exact
    minor would report routine bumps as a compatibility problem."""
    env = parse_env(_env(schema_version=version))

    assert env.schema_status == "supported"
    assert env.schema_note is None


def test_newer_major_is_read_anyway_but_flagged():
    env = parse_env(_env(schema_version=f"{ENV_SCHEMA_MAJOR + 1}.0"))

    assert env.schema_status == "newer"
    assert "newer major" in env.schema_note
    assert env.rocm_version == "7.2.0"


def test_older_major_is_flagged_as_older():
    env = parse_env(_env(schema_version="0.9"))

    assert env.schema_status == "older"


@pytest.mark.parametrize("value", [1, None, "", "v1.16"])
def test_unparseable_schema_version_is_unknown(value):
    env = parse_env(_env(schema_version=value))

    assert env.schema_status == "unknown"


@pytest.mark.parametrize("value", ["1", "1.", "1.bad", "1.2.3", "1.16.0", " 1.16"])
def test_a_value_that_is_not_major_dot_minor_is_unknown(value):
    """The whole value has to parse, not just the part before the first dot.

    Splitting once read the major and ignored whatever followed, so "1",
    "1.bad" and "1.2.3" were all reported as a supported major 1 -- a corrupt
    schema presented as compatible, which is the one answer this must not give.
    """
    env = parse_env(_env(schema_version=value))

    assert env.schema_status == "unknown"
    assert "MAJOR.MINOR" in env.schema_note


@pytest.mark.parametrize("value", ["1", "1.bad", "1.2.3"])
def test_a_malformed_schema_version_is_also_a_missing_field(value):
    """Unknown has to reach ``require()``, or strict callers still see a pass."""
    env = parse_env(_env(schema_version=value))

    assert "schema_version" in env.missing_fields
    with pytest.raises(MissingFieldError, match="schema_version"):
        env.require()


def test_absent_schema_version_is_unknown():
    env = parse_env(_env_without("schema_version"))

    assert env.schema_status == "unknown"
    assert env.schema_version is None


@pytest.mark.parametrize("value", [1, None, "", "v1.16"])
def test_an_unreadable_schema_version_is_a_missing_field(value):
    """``require()`` with no arguments promises every modelled field.

    Classifying the value as unknown without recording it told a strict caller
    the artifact was complete while it held no usable schema version at all.
    """
    env = parse_env(_env(schema_version=value))

    assert "schema_version" in env.missing_fields
    with pytest.raises(MissingFieldError, match="schema_version"):
        env.require()


def test_an_absent_schema_version_is_a_missing_field():
    env = parse_env(_env_without("schema_version"))

    assert "schema_version" in env.missing_fields
    with pytest.raises(MissingFieldError):
        env.require("schema_version")


@pytest.mark.parametrize("value", ["1.16", "2.0", "0.9"])
def test_a_readable_schema_version_is_not_missing_even_when_it_disagrees(value):
    """"newer" and "older" are values that were read, not values that were not."""
    env = parse_env(_env(schema_version=value))

    assert "schema_version" not in env.missing_fields


# ---- file-level errors ----------------------------------------------------


def test_read_env_round_trips_a_file(tmp_path: Path):
    path = tmp_path / "env.json"
    path.write_text(json.dumps(_env()), encoding="utf-8")

    env = read_env(path)

    assert env.source_path == path
    assert env.rocm_version == "7.2.0"


def test_read_env_raises_when_the_file_is_absent(tmp_path: Path):
    with pytest.raises(ArtifactReadError):
        read_env(tmp_path / "nope.json")


def test_read_env_raises_on_invalid_json(tmp_path: Path):
    path = tmp_path / "env.json"
    path.write_text("nope", encoding="utf-8")

    with pytest.raises(ArtifactReadError):
        read_env(path)


def test_read_env_raises_on_invalid_utf8(tmp_path: Path):
    """A snapshot cut mid-character is unreadable, not a different exception.

    ``UnicodeDecodeError`` is a ``ValueError`` rather than an ``OSError``, so a
    reader that converts only the latter lets it past ``ArtifactError``
    altogether and the tolerant callers in :mod:`aorta.chat` -- which catch
    ``ArtifactReadError`` precisely to survive a run killed mid-write -- abort
    on the case they were written for.
    """
    path = tmp_path / "env.json"
    # Cut inside the three bytes of an em dash: valid UTF-8 up to the cut, then
    # a truncated sequence, which is what a killed write actually leaves.
    encoded = json.dumps(_env(captured_at="2026\u201401"), ensure_ascii=False).encode("utf-8")
    path.write_bytes(encoded[: encoded.index(b"\xe2\x80\x94") + 2])

    with pytest.raises(ArtifactReadError):
        read_env(path)
