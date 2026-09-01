"""Manifest round-tripping and the warn/refuse split of Decision 20a.

The asymmetry under test is the whole design. Source drift warns, because an
index forty commits old is still mostly right and refusing would leave the user
with nothing. An embedding-model or dimension mismatch refuses, because there is
no partially correct answer available -- the vectors are not comparable, so
retrieval returns confident nonsense with nothing on screen to say so.

The refusal *text* is asserted, not just the exception type. A refusal the user
skims past or works around has failed, and its wording is the only thing
standing between them and a plausible wrong answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.chat.rag import manifest as manifest_mod
from aorta.chat.rag.manifest import (
    SCHEMA_VERSION,
    IndexMismatchError,
    Manifest,
    ManifestError,
    checksum_path,
    manifest_path,
    read_manifest,
    sha256_file,
    validate,
    write_manifest,
)

MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "aorta_fastembed_baai_bge_small_en_v1_5"


def _manifest(**overrides) -> Manifest:
    values = {
        "aorta_version": "0.2.1",
        "aorta_sha": "45edc3d1122334455667788990011223344556677",
        "aorta_tag": "v0.2.1",
        "embedding_provider": "local",
        "embedding_model": MODEL,
        "dimensions": 384,
        "collection": COLLECTION,
        "chunk_size": 512,
        "chunk_overlap": 50,
        "index_sha256": "0" * 64,
        "store_version": "0.1.6",
        "built_at": "2026-09-01T00:00:00+00:00",
        "corpus_digest": "abc123",
        "file_count": 700,
        "chunk_count": 15343,
    }
    values.update(overrides)
    return Manifest(**values)


@pytest.fixture()
def index_file(tmp_path: Path) -> Path:
    path = tmp_path / "index.sqlite"
    path.write_bytes(b"not really sqlite, but it hashes")
    return path


class TestRoundTrip:
    def test_writing_produces_a_manifest_and_a_checksum_sidecar(self, index_file: Path):
        written = write_manifest(index_file, _manifest())
        assert written == manifest_path(index_file)
        assert written.exists()
        assert checksum_path(index_file).exists()

    def test_the_checksum_file_is_in_sha256sum_format(self, index_file: Path):
        """So `sha256sum -c` works on the published file without reformatting."""
        write_manifest(index_file, _manifest(index_sha256="deadbeef"))
        assert checksum_path(index_file).read_text() == f"deadbeef  {index_file.name}\n"

    def test_reading_returns_what_was_written(self, index_file: Path):
        write_manifest(index_file, _manifest())
        found = read_manifest(index_file)
        assert found.embedding_model == MODEL
        assert found.dimensions == 384
        assert found.collection == COLLECTION
        assert found.chunk_size == 512

    def test_sha256_matches_the_stdlib(self, index_file: Path):
        import hashlib

        assert sha256_file(index_file) == hashlib.sha256(index_file.read_bytes()).hexdigest()

    def test_describe_names_the_source_and_the_model(self):
        text = _manifest().describe()
        assert "v0.2.1" in text
        assert MODEL in text
        assert "384d" in text


class TestReadFailures:
    def test_a_missing_manifest_says_why_it_matters(self, index_file: Path):
        with pytest.raises(ManifestError) as exc:
            read_manifest(index_file)
        message = str(exc.value)
        assert "no manifest" in message
        assert "embedding model" in message
        assert "aorta chat index fetch" in message

    def test_malformed_json_is_reported_with_its_path(self, index_file: Path):
        manifest_path(index_file).write_text("{not json", encoding="utf-8")
        with pytest.raises(ManifestError, match="could not read the manifest"):
            read_manifest(index_file)

    def test_a_missing_required_field_is_named(self, index_file: Path):
        manifest_path(index_file).write_text(json.dumps({"aorta_version": "1.0"}))
        with pytest.raises(ManifestError, match="missing required field"):
            read_manifest(index_file)

    def test_a_newer_schema_is_refused_with_an_upgrade_hint(self, index_file: Path):
        raw = json.loads(_manifest().to_json())
        raw["schema_version"] = SCHEMA_VERSION + 1
        manifest_path(index_file).write_text(json.dumps(raw))
        with pytest.raises(ManifestError) as exc:
            read_manifest(index_file)
        assert "Upgrade aorta" in str(exc.value)

    def test_an_unknown_extra_key_is_tolerated(self, index_file: Path):
        """A newer builder adding a field must not strand an older client.

        The two are published and installed independently, so forward
        tolerance is the difference between a rolling asset that keeps working
        and one that breaks every user on the previous release.
        """
        raw = json.loads(_manifest().to_json())
        raw["future_field"] = "whatever"
        manifest_path(index_file).write_text(json.dumps(raw))
        assert read_manifest(index_file).embedding_model == MODEL


class TestRefusals:
    def _validate(self, **overrides):
        return validate(
            _manifest(**overrides),
            embedding_model=MODEL,
            collection=COLLECTION,
            dimensions=384,
            chunk_size=512,
            chunk_overlap=50,
        )

    def test_a_matching_manifest_is_clean(self):
        report = self._validate()
        assert report.refusals == []
        assert report.warnings == []
        assert report.ok

    def test_a_different_embedding_model_refuses(self):
        report = self._validate(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        assert report.refusals
        assert "embedding model" in report.refusals[0]

    def test_a_different_collection_refuses(self):
        report = self._validate(collection="aorta")
        assert any("collection" in line for line in report.refusals)

    def test_a_different_dimension_refuses(self):
        report = self._validate(dimensions=1536)
        assert any("dimensions" in line for line in report.refusals)

    def test_a_different_store_format_refuses(self):
        """The format changed once already (Chroma -> sqlite-vec)."""
        report = self._validate(store="chroma")
        assert any("store format" in line for line in report.refusals)

    def test_same_dimensions_but_a_different_model_still_refuses(self):
        """The exact trap this exists for.

        BGE-small on torch and BGE-small on quantised ONNX are both 384
        dimensions, so a dimension check alone lets one index answer the other's
        queries. The model check is what closes it.
        """
        report = validate(
            _manifest(embedding_model="BAAI/bge-small-en-v1.5-torch", collection="aorta"),
            embedding_model=MODEL,
            collection=COLLECTION,
            dimensions=384,
        )
        assert report.refusals


class TestRefusalText:
    """The message is the deliverable, so it is asserted like one."""

    def _refusal(self) -> str:
        report = validate(
            _manifest(embedding_model="other/model", collection="aorta_other"),
            embedding_model=MODEL,
            collection=COLLECTION,
        )
        with pytest.raises(IndexMismatchError) as exc:
            report.raise_if_refused("/home/u/.cache/aorta/chat/index.sqlite")
        return str(exc.value)

    def test_it_leads_with_a_refusal_and_names_the_file(self):
        assert "REFUSING" in self._refusal()
        assert "/home/u/.cache/aorta/chat/index.sqlite" in self._refusal()

    def test_it_states_the_consequence_not_just_the_mismatch(self):
        """A mismatch is not self-evidently serious to someone wanting an answer."""
        text = self._refusal()
        assert "would not error" in text
        assert "wrong" in text

    def test_it_names_both_sides_of_the_disagreement(self):
        text = self._refusal()
        assert "other/model" in text
        assert MODEL in text

    def test_it_ends_with_commands_the_user_can_run(self):
        """A refusal nobody can act on gets worked around instead of fixed."""
        text = self._refusal()
        assert "aorta chat index fetch" in text
        assert "aorta chat index build" in text
        assert "aorta chat doctor" in text


class TestWarnings:
    def test_version_drift_warns_rather_than_refuses(self):
        report = validate(
            _manifest(),
            embedding_model=MODEL,
            collection=COLLECTION,
            installed_version="0.2.2.dev122+g9b20106",
        )
        assert report.refusals == []
        assert any("source drift" in line for line in report.warnings)
        assert any("0.2.1" in line for line in report.warnings)

    def test_the_drift_warning_says_how_to_refresh(self):
        report = validate(
            _manifest(),
            embedding_model=MODEL,
            collection=COLLECTION,
            installed_version="0.3.0",
        )
        assert any("aorta chat index fetch" in line for line in report.warnings)

    def test_an_identical_version_does_not_warn(self):
        report = validate(
            _manifest(),
            embedding_model=MODEL,
            collection=COLLECTION,
            installed_version="0.2.1",
        )
        assert report.warnings == []

    def test_chunk_size_drift_warns_because_spans_change_size(self):
        report = validate(
            _manifest(),
            embedding_model=MODEL,
            collection=COLLECTION,
            chunk_size=1024,
        )
        assert report.refusals == []
        assert any("chunk size" in line for line in report.warnings)

    def test_a_sha_delta_warns_when_the_version_string_matches(self):
        """A dev install and the rolling asset can share a version and differ."""
        report = validate(
            _manifest(aorta_version="0.2.2.dev122+g45edc3d"),
            embedding_model=MODEL,
            collection=COLLECTION,
            installed_version="0.2.2.dev122+g45edc3d",
            installed_sha="9b20106",
        )
        assert any("source drift" in line for line in report.warnings)


class TestNowStamp:
    def test_it_is_utc_and_second_resolution(self):
        stamp = manifest_mod.now_stamp()
        assert stamp.endswith("+00:00")
        assert "." not in stamp
