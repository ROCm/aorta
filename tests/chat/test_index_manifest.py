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

    @pytest.mark.parametrize("value", ["1", None, 1.5, [1], True])
    def test_a_non_integer_schema_is_refused_rather_than_compared(self, index_file, value):
        """``schema_version > SCHEMA_VERSION`` against a string raises TypeError.

        That escaped every caller, all of which handle only ``ManifestError``,
        so a malformed sidecar surfaced as an unhandled crash instead of the
        refusal the reader is supposed to produce.
        """
        raw = json.loads(_manifest().to_json())
        raw["schema_version"] = value
        manifest_path(index_file).write_text(json.dumps(raw))

        with pytest.raises(ManifestError, match="non-integer schema version"):
            read_manifest(index_file)

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


class TestTheEmbeddingIdentityRefusal:
    """Same model name, different endpoint, is a different vector space.

    ``text-embedding-3-small`` means whatever the configured OpenAI-compatible
    API says it means. Keying only on the model let a gateway switch keep the
    old vectors and query them with the new endpoint's -- every field check
    passed and retrieval returned plausible nonsense.
    """

    def _validate(self, manifest_identity: str, install_identity: str):
        return validate(
            _manifest(embedding_identity=manifest_identity),
            embedding_model=MODEL,
            collection=COLLECTION,
            embedding_identity=install_identity,
        )

    def test_a_different_endpoint_refuses(self):
        report = self._validate(f"https://a.example/v1\n{MODEL}", f"https://b.example/v1\n{MODEL}")

        assert any("embedding identity" in line for line in report.refusals)

    def test_the_refusal_names_both_endpoints(self):
        """Two opaque collection digests are not something an operator can act on."""
        report = self._validate(f"https://a.example/v1\n{MODEL}", f"https://b.example/v1\n{MODEL}")
        line = next(line for line in report.refusals if "embedding identity" in line)

        assert "https://a.example/v1" in line
        assert "https://b.example/v1" in line
        assert "\n" not in line

    def test_the_same_identity_is_clean(self):
        report = self._validate(f"https://a.example/v1\n{MODEL}", f"https://a.example/v1\n{MODEL}")

        assert report.refusals == []

    def test_a_manifest_predating_the_field_is_not_refused_on_it(self):
        """It has no identity to disagree with, and the collection check covers it.

        Refusing on an empty value would reject every manifest written before
        the field existed for a reason that is not true of them.
        """
        report = self._validate("", f"https://a.example/v1\n{MODEL}")

        assert not any("embedding identity" in line for line in report.refusals)


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

    def test_a_remote_embedder_is_not_told_to_fetch(self, monkeypatch):
        """The impossible remedy, offered first, is what sends people to a workaround.

        CI publishes one index asset and builds it with the local embedder, so
        "the index matching this install" does not exist for a remote provider
        and never will. Following that line gets a second refusal with different
        wording, from which the reasonable conclusion is that chat is broken.
        """
        monkeypatch.setattr(manifest_mod, "_configured_embedding_provider", lambda: "remote")
        text = self._refusal()
        commands = [line for line in text.splitlines() if line.startswith("  aorta")]
        assert not any("index fetch" in line for line in commands)
        assert any("index build" in line for line in commands)


class TestRemedyLines:
    """Which commands a mismatch is resolved by depends on the embedding provider."""

    def test_a_local_provider_leads_with_fetch(self):
        lines = manifest_mod.remedy_lines("local")
        assert lines[0].strip().startswith("aorta chat index fetch")

    def test_a_remote_provider_explains_the_absence_rather_than_hiding_it(self):
        """Otherwise the user goes looking for the command the docs mention."""
        text = "\n".join(manifest_mod.remedy_lines("remote"))
        assert "is not offered here" in text
        assert "AORTA_CHAT_EMBEDDING_PROVIDER=local" in text

    def test_doctor_is_not_told_to_run_doctor(self):
        text = "\n".join(manifest_mod.remedy_lines("local", include_doctor=False))
        assert "aorta chat doctor" not in text

    def test_it_reads_the_configured_provider_when_not_given_one(self, monkeypatch):
        from aorta.chat.config import settings

        monkeypatch.setattr(settings, "embedding_provider", "remote")
        assert "is not offered here" in "\n".join(manifest_mod.remedy_lines())

    def test_an_empty_provider_setting_is_treated_as_local(self, monkeypatch):
        """A setting that resolves to no provider falls back to the shipped default.

        Which is also the one the published index is built with, so the fetch
        remedy is the right guess when there is nothing to read.
        """
        from aorta.chat.config import settings

        monkeypatch.setattr(settings, "embedding_provider", "")
        assert manifest_mod._configured_embedding_provider() == "local"

    def test_every_alias_of_the_local_provider_still_gets_the_fetch_remedy(self, monkeypatch):
        """``onnx`` and ``fastembed`` are spellings of ``local``, not remote providers.

        Comparing ``settings.embedding_provider`` as a raw string handed an
        ordinary local install the remote remedy: it withholds ``index fetch``,
        the one command that fixes its index, and tells it to set the provider
        it is already on. Discovered from the factory rather than listed here,
        so a new alias is covered without editing this test.
        """
        from aorta.chat.config import settings
        from aorta.chat.rag.embeddings import factory

        aliases = [name for name, target in factory._ALIASES.items() if target == "local"]
        assert aliases, "the factory no longer aliases anything to the local provider"
        for alias in aliases:
            monkeypatch.setattr(settings, "embedding_provider", alias)
            assert manifest_mod._configured_embedding_provider() == "local", alias
            lines = manifest_mod.remedy_lines()
            assert lines[0].strip().startswith("aorta chat index fetch"), alias
            assert "is not offered here" not in "\n".join(lines), alias


class TestRefreshCommand:
    """The one-command form, for the messages that are prose rather than a report."""

    def test_a_local_provider_is_told_to_fetch(self):
        assert manifest_mod._refresh_command("local") == "aorta chat index fetch"

    def test_a_remote_provider_is_told_to_build(self):
        assert manifest_mod._refresh_command("remote") == "aorta chat index build"

    def test_it_agrees_with_the_block_form(self):
        """Two independent answers to "is a fetch worth suggesting" would drift.

        Read off the command lines rather than the whole block: the remote
        block names ``index fetch`` in prose precisely to say it is not on
        offer, so a substring search over all of it answers the wrong question.
        """
        for provider in ("local", "remote"):
            commands = [
                line for line in manifest_mod.remedy_lines(provider) if line.startswith("  aorta")
            ]
            offered = any("index fetch" in line for line in commands)
            assert offered == (manifest_mod._refresh_command(provider) == "aorta chat index fetch")

    def test_a_missing_manifest_names_a_command_the_provider_can_run(self, monkeypatch, tmp_path):
        """The message doctor prints for an index nobody can verify.

        It reaches a remote install through ``doctor``'s "cannot be verified"
        branch and through the query-time refusal, so a hardcoded fetch here is
        the same impossible remedy in two more places.
        """
        monkeypatch.setattr(manifest_mod, "_configured_embedding_provider", lambda: "remote")
        with pytest.raises(manifest_mod.ManifestError) as excinfo:
            manifest_mod.read_manifest(tmp_path / "index.sqlite")
        assert "no manifest beside" in str(excinfo.value)
        assert "index fetch" not in str(excinfo.value)
        assert "aorta chat index build" in str(excinfo.value)

    def test_a_malformed_schema_version_names_one_too(self, monkeypatch):
        monkeypatch.setattr(manifest_mod, "_configured_embedding_provider", lambda: "local")
        with pytest.raises(manifest_mod.ManifestError) as excinfo:
            manifest_mod.ensure_supported_schema(_manifest(schema_version="1"), "the index")
        assert "aorta chat index fetch" in str(excinfo.value)


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

    def test_the_drift_warning_does_not_name_fetch_on_a_remote_provider(self, monkeypatch):
        """The index still works here, so the advice must be a command that runs.

        A fetch under a remote embedder is refused rather than stale, so
        answering "your index is a little old" with it trades a warning the
        user could act on for an error they cannot.
        """
        monkeypatch.setattr(manifest_mod, "_configured_embedding_provider", lambda: "remote")
        report = validate(
            _manifest(),
            embedding_model=MODEL,
            collection=COLLECTION,
            installed_version="0.3.0",
        )
        drift = [line for line in report.warnings if "source drift" in line]
        assert drift
        assert not any("index fetch" in line for line in drift)
        assert any("aorta chat index build" in line for line in drift)

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
