"""Version resolution, download verification, and ``--from`` side-loading.

No test here touches the network: ``urllib.request.urlopen`` is replaced by a
fake server serving an in-memory asset set. The point is the decision logic and
the verification order, not HTTP.

Verification order is load-bearing and asserted directly. Checksum first, so a
corrupt download never reaches the manifest parser; manifest validation second,
so a mismatched index never reaches the configured path. Nothing is installed
until both pass -- a partial install would leave a broken index where the reader
expects a good one.
"""

from __future__ import annotations

import http.client
import io
import json
import urllib.error
from pathlib import Path

import pytest

from aorta.chat.config import settings
from aorta.chat.rag import index_ops
from aorta.chat.rag import manifest as manifest_mod
from aorta.chat.rag.embeddings.base import build_collection_name
from aorta.chat.rag.embeddings.fastembed_bge import LOCAL_COLLECTION_PREFIX
from aorta.chat.rag.index_ops import (
    ASSET_NAME,
    ROLLING_TAG,
    IndexFetchError,
    fetch_index,
    resolve_source,
    side_load,
)

MODEL = "BAAI/bge-small-en-v1.5"
# Derived, not spelled out: the name carries a digest of the embedding
# identity, and a fixture that hardcoded it would make every manifest here
# refuse for the wrong reason the next time the identity gains a component.
COLLECTION = build_collection_name(LOCAL_COLLECTION_PREFIX, MODEL)
BODY = b"pretend this is a 48 MB sqlite-vec index" * 16


@pytest.fixture(autouse=True)
def local_provider(monkeypatch):
    """Pin the provider side of every comparison, so the manifest is the variable."""
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(settings, "embedding_model", MODEL)
    monkeypatch.setattr(settings, "chunk_size", 512)
    monkeypatch.setattr(settings, "chunk_overlap", 50)


def _manifest(**overrides) -> manifest_mod.Manifest:
    import hashlib

    values = {
        "aorta_version": "0.2.1",
        "aorta_sha": "45edc3d" + "0" * 33,
        "aorta_tag": "v0.2.1",
        "embedding_provider": "local",
        "embedding_model": MODEL,
        "dimensions": 384,
        "collection": COLLECTION,
        "chunk_size": 512,
        "chunk_overlap": 50,
        "index_sha256": hashlib.sha256(BODY).hexdigest(),
        "built_at": "2026-09-01T00:00:00+00:00",
        "corpus_digest": "abc123",
    }
    values.update(overrides)
    return manifest_mod.Manifest(**values)


class _FakeServer:
    """Serves an in-memory asset set, and records what was asked for."""

    def __init__(self, assets: dict[str, bytes]) -> None:
        self.assets = assets
        self.requested: list[str] = []

    def urlopen(self, url, timeout=None):  # noqa: ARG002 - signature match
        self.requested.append(url)
        for suffix, payload in self.assets.items():
            if url.endswith(suffix):
                return _FakeResponse(payload)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _FailingResponse(_FakeResponse):
    """Opens cleanly, then fails partway through its body.

    The interesting half of the failure surface: the request has already
    succeeded, so nothing here arrives as the ``URLError`` that ``urlopen``
    wraps a connection failure in.
    """

    def __init__(self, payload: bytes, error: BaseException) -> None:
        super().__init__(payload)
        self._error = error

    def read(self, *args):  # noqa: ARG002 - signature match
        raise self._error


@pytest.fixture()
def server(monkeypatch):
    """A published asset set that validates cleanly, unless a test perturbs it."""
    import hashlib

    manifest = _manifest()
    assets = {
        ASSET_NAME: BODY,
        ASSET_NAME + manifest_mod.MANIFEST_SUFFIX: manifest.to_json().encode(),
        ASSET_NAME
        + manifest_mod.CHECKSUM_SUFFIX: f"{hashlib.sha256(BODY).hexdigest()}  {ASSET_NAME}\n".encode(),
    }
    fake = _FakeServer(assets)
    monkeypatch.setattr(index_ops.urllib.request, "urlopen", fake.urlopen)
    return fake


def _reserialise(server: _FakeServer, manifest: manifest_mod.Manifest) -> None:
    server.assets[ASSET_NAME + manifest_mod.MANIFEST_SUFFIX] = manifest.to_json().encode()


class TestVersionResolution:
    """Decision 18a. Match the index to the installed version, not to 'newest'."""

    def test_an_exact_release_takes_that_releases_asset(self):
        source = resolve_source(installed="0.2.1")
        assert source.tag == "v0.2.1"
        assert source.channel == "release"
        assert source.notes == ()

    def test_a_dev_version_takes_the_rolling_main_asset(self):
        """A released wheel wants its own release; a dev install is on main."""
        source = resolve_source(installed="0.2.2.dev122+g45edc3d")
        assert source.tag == ROLLING_TAG
        assert "main" in source.channel

    def test_a_dev_version_warns_with_the_distance_and_the_sha(self):
        notes = " ".join(resolve_source(installed="0.2.2.dev122+g45edc3d.d20260810").notes)
        assert "0.2.2.dev122+g45edc3d.d20260810" in notes
        assert "122 commit" in notes
        assert "45edc3d" in notes

    def test_an_rc_nightly_version_is_not_mistaken_for_a_release(self):
        """nightly.yml stamps X.Y.ZrcYYYYMMDD, which is not a released tag."""
        assert resolve_source(installed="0.2.2rc20260901").tag == ROLLING_TAG

    def test_an_unknown_version_still_resolves_to_the_rolling_asset(self):
        source = resolve_source(installed="")
        assert source.tag == ROLLING_TAG
        assert "unreleased" in " ".join(source.notes)

    def test_an_explicit_version_overrides_the_installed_one(self):
        source = resolve_source("0.1.9", installed="0.2.2.dev1+gabc1234")
        assert source.tag == "v0.1.9"
        assert source.channel == "release"

    def test_a_leading_v_is_accepted(self):
        assert resolve_source("v0.1.9", installed="0.2.1").tag == "v0.1.9"

    def test_a_non_version_override_is_passed_through_as_a_tag(self):
        """So `--version dev-wheels` reaches the rolling asset by name."""
        source = resolve_source(ROLLING_TAG, installed="0.2.1")
        assert source.tag == ROLLING_TAG
        assert source.channel == "explicit"

    def test_the_base_url_can_be_pointed_at_a_mirror(self, monkeypatch):
        """An internal mirror or an air-gapped artifact server."""
        monkeypatch.setenv(index_ops.BASE_URL_ENV, "https://mirror.invalid/aorta/")
        source = resolve_source(installed="0.2.1")
        assert source.index_url == f"https://mirror.invalid/aorta/v0.2.1/{ASSET_NAME}"

    def test_the_sidecar_urls_derive_from_the_index_url(self):
        source = resolve_source(installed="0.2.1")
        assert source.manifest_url == source.index_url + manifest_mod.MANIFEST_SUFFIX
        assert source.checksum_url == source.index_url + manifest_mod.CHECKSUM_SUFFIX


class TestFetch:
    def test_a_clean_fetch_installs_the_index_and_both_sidecars(self, server, tmp_path: Path):
        dest = tmp_path / "cache" / "index.sqlite"
        result = fetch_index(version="0.2.1", index_path=dest)

        assert dest.read_bytes() == BODY
        assert manifest_mod.manifest_path(dest).exists()
        assert manifest_mod.checksum_path(dest).exists()
        assert result.manifest.embedding_model == MODEL

    def test_the_installed_manifest_reads_back(self, server, tmp_path: Path):
        dest = tmp_path / "index.sqlite"
        fetch_index(version="0.2.1", index_path=dest)
        assert manifest_mod.read_manifest(dest).collection == COLLECTION

    def test_no_temporary_files_are_left_behind(self, server, tmp_path: Path):
        dest = tmp_path / "cache" / "index.sqlite"
        fetch_index(version="0.2.1", index_path=dest)
        assert sorted(p.name for p in dest.parent.iterdir()) == [
            "index.sqlite",
            "index.sqlite.manifest.json",
            "index.sqlite.sha256",
        ]

    def test_it_replaces_an_existing_index(self, server, tmp_path: Path):
        dest = tmp_path / "index.sqlite"
        dest.write_bytes(b"stale")
        fetch_index(version="0.2.1", index_path=dest)
        assert dest.read_bytes() == BODY

    def test_the_source_and_warnings_are_reported_back(self, server, tmp_path: Path):
        result = fetch_index(
            index_path=tmp_path / "i.sqlite",
            source=resolve_source(installed="0.2.2.dev122+g45edc3d"),
        )
        assert ROLLING_TAG in result.source
        assert any("122 commit" in warning for warning in result.warnings)


class TestFetchFailures:
    def test_a_missing_asset_names_the_alternatives(self, monkeypatch, tmp_path: Path):
        fake = _FakeServer({})
        monkeypatch.setattr(index_ops.urllib.request, "urlopen", fake.urlopen)
        with pytest.raises(IndexFetchError) as exc:
            fetch_index(version="0.1.0", index_path=tmp_path / "i.sqlite")
        message = str(exc.value)
        assert "aorta chat index build" in message

    def test_a_missing_manifest_refuses_rather_than_using_the_index(self, server, tmp_path: Path):
        """An unverifiable index is not a usable one."""
        del server.assets[ASSET_NAME + manifest_mod.MANIFEST_SUFFIX]
        with pytest.raises(IndexFetchError) as exc:
            fetch_index(version="0.2.1", index_path=tmp_path / "i.sqlite")
        assert "not safe to use" in str(exc.value)

    def test_a_corrupt_download_is_caught_by_the_checksum(self, server, tmp_path: Path):
        server.assets[ASSET_NAME] = BODY + b"truncated-or-tampered"
        dest = tmp_path / "i.sqlite"
        with pytest.raises(IndexFetchError) as exc:
            fetch_index(version="0.2.1", index_path=dest)
        assert "checksum mismatch" in str(exc.value)
        assert not dest.exists()

    def test_a_manifest_disagreeing_with_the_checksum_file_is_refused(self, server, tmp_path: Path):
        _reserialise(server, _manifest(index_sha256="f" * 64))
        with pytest.raises(IndexFetchError, match="disagree about the index"):
            fetch_index(version="0.2.1", index_path=tmp_path / "i.sqlite")

    def test_a_model_mismatch_refuses_and_installs_nothing(self, server, tmp_path: Path):
        """The refusal has to arrive before the file lands, not after."""
        _reserialise(server, _manifest(embedding_model="other/model"))
        dest = tmp_path / "i.sqlite"
        with pytest.raises(manifest_mod.IndexMismatchError) as exc:
            fetch_index(version="0.2.1", index_path=dest)
        assert "REFUSING" in str(exc.value)
        assert not dest.exists()
        assert not manifest_mod.manifest_path(dest).exists()

    def test_an_unreachable_host_points_at_side_loading(self, monkeypatch, tmp_path: Path):
        """The air-gapped user's next move, in the error they actually get."""

        def _refuse(url, timeout=None):
            raise urllib.error.URLError("Network is unreachable")

        monkeypatch.setattr(index_ops.urllib.request, "urlopen", _refuse)
        with pytest.raises(IndexFetchError) as exc:
            fetch_index(version="0.2.1", index_path=tmp_path / "i.sqlite")
        assert "index fetch --from" in str(exc.value)

    def test_a_garbled_manifest_is_reported_as_such(self, server, tmp_path: Path):
        server.assets[ASSET_NAME + manifest_mod.MANIFEST_SUFFIX] = b"{not json"
        with pytest.raises(IndexFetchError, match="not usable"):
            fetch_index(version="0.2.1", index_path=tmp_path / "i.sqlite")


class TestTheFetchedSchemaIsChecked:
    """``read_manifest`` refused a future schema; the download path did not.

    So an older client reported a successful fetch, installed the files, and
    then had every load refuse the manifest it had just written -- a fetch that
    "worked" followed by a chat that no longer starts.
    """

    def _serve_schema(self, server, value) -> None:
        raw = json.loads(_manifest().to_json())
        raw["schema_version"] = value
        server.assets[ASSET_NAME + manifest_mod.MANIFEST_SUFFIX] = json.dumps(raw).encode()

    def test_a_newer_schema_refuses_and_installs_nothing(self, server, tmp_path: Path):
        self._serve_schema(server, manifest_mod.SCHEMA_VERSION + 1)
        dest = tmp_path / "i.sqlite"

        with pytest.raises(IndexFetchError) as exc:
            fetch_index(version="0.2.1", index_path=dest)

        assert "Upgrade aorta" in str(exc.value)
        assert not dest.exists()
        assert not manifest_mod.manifest_path(dest).exists()
        assert not manifest_mod.checksum_path(dest).exists()

    @pytest.mark.parametrize("value", ["1", None, 1.5, [1]])
    def test_a_non_integer_schema_is_an_index_fetch_error(self, server, tmp_path, value):
        """Not a ``TypeError`` out of a comparison the caller never guarded."""
        self._serve_schema(server, value)
        dest = tmp_path / "i.sqlite"

        with pytest.raises(IndexFetchError, match="non-integer schema version"):
            fetch_index(version="0.2.1", index_path=dest)

        assert not dest.exists()

    def test_an_older_schema_is_still_accepted(self, server, tmp_path: Path):
        """Forward tolerance runs one way only; an older sidecar still parses."""
        self._serve_schema(server, manifest_mod.SCHEMA_VERSION)
        dest = tmp_path / "i.sqlite"

        assert fetch_index(version="0.2.1", index_path=dest).index_path == dest
        assert dest.exists()


class TestTransferFailures:
    """Failures after the request succeeded, which is where the gaps were.

    ``urlopen`` wraps a *connection* failure in ``URLError``; a body that stops
    short or a socket that resets mid-stream does not go through that path, so
    each one needs a handler of its own or it leaves ``index_ops`` unconverted
    and the CLI, which only knows :class:`IndexFetchError`, prints a traceback.
    """

    @staticmethod
    def _failing_asset(monkeypatch, server, error: BaseException) -> None:
        """Serve the index body from a response that raises ``error`` mid-read."""
        real = server.urlopen

        def _urlopen(url, timeout=None):
            if url.endswith(ASSET_NAME):
                return _FailingResponse(BODY, error)
            return real(url, timeout=timeout)

        monkeypatch.setattr(index_ops.urllib.request, "urlopen", _urlopen)

    def test_a_body_that_stops_short_is_reported_and_installs_nothing(
        self, server, tmp_path: Path, monkeypatch
    ):
        """``IncompleteRead`` is an ``HTTPException``, not an ``OSError``."""
        self._failing_asset(
            monkeypatch, server, http.client.IncompleteRead(BODY[:16], len(BODY) - 16)
        )
        dest = tmp_path / "i.sqlite"
        with pytest.raises(IndexFetchError, match="ended early"):
            fetch_index(version="0.2.1", index_path=dest)
        assert not dest.exists()

    def test_a_reset_mid_stream_does_not_blame_the_destination(
        self, server, tmp_path: Path, monkeypatch
    ):
        """Calling a stalled transfer a write failure sends the operator to the disk.

        The node with no egress is the expected case here, so the one message
        that must not appear is the one naming the destination as the cause.
        """
        self._failing_asset(
            monkeypatch, server, ConnectionResetError(104, "Connection reset by peer")
        )
        with pytest.raises(IndexFetchError) as exc:
            fetch_index(version="0.2.1", index_path=tmp_path / "i.sqlite")
        assert "could not write" not in str(exc.value)
        assert "Connection reset by peer" in str(exc.value)

    def test_a_sidecar_reset_is_reported_as_a_fetch_error(
        self, server, tmp_path: Path, monkeypatch
    ):
        """The sidecar is fetched first, so it fails first."""

        def _urlopen(url, timeout=None):  # noqa: ARG001 - signature match
            return _FailingResponse(b"", ConnectionResetError(104, "Connection reset by peer"))

        monkeypatch.setattr(index_ops.urllib.request, "urlopen", _urlopen)
        with pytest.raises(IndexFetchError, match="Connection reset by peer"):
            fetch_index(version="0.2.1", index_path=tmp_path / "i.sqlite")

    def test_a_sidecar_that_is_not_utf8_is_reported_as_a_fetch_error(self, server, tmp_path: Path):
        server.assets[ASSET_NAME + manifest_mod.MANIFEST_SUFFIX] = b"\xff\xfe{}"
        with pytest.raises(IndexFetchError, match="not valid UTF-8"):
            fetch_index(version="0.2.1", index_path=tmp_path / "i.sqlite")


class TestSideLoad:
    """Decision 21b: index only. The model is a documented pre-seed, not an asset."""

    @pytest.fixture()
    def staged(self, tmp_path: Path) -> Path:
        staging = tmp_path / "usb"
        staging.mkdir()
        index = staging / ASSET_NAME
        index.write_bytes(BODY)
        manifest_mod.write_manifest(index, _manifest())
        return index

    def test_a_staged_index_is_adopted(self, staged: Path, tmp_path: Path):
        dest = tmp_path / "cache" / "index.sqlite"
        result = side_load(staged, index_path=dest)
        assert dest.read_bytes() == BODY
        assert manifest_mod.read_manifest(dest).embedding_model == MODEL
        assert str(staged) in result.source

    def test_a_directory_is_accepted_if_it_holds_the_asset(self, staged: Path, tmp_path: Path):
        dest = tmp_path / "index.sqlite"
        side_load(staged.parent, index_path=dest)
        assert dest.exists()

    def test_a_directory_without_the_asset_says_what_to_stage(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(IndexFetchError) as exc:
            side_load(empty, index_path=tmp_path / "i.sqlite")
        assert ASSET_NAME in str(exc.value)

    def test_a_missing_file_is_reported(self, tmp_path: Path):
        with pytest.raises(IndexFetchError, match="no index at"):
            side_load(tmp_path / "absent.sqlite", index_path=tmp_path / "i.sqlite")

    def test_a_staged_index_without_a_manifest_is_refused(self, tmp_path: Path):
        """Side-loading is where a mismatch is most likely, not least.

        The file was carried by hand from somewhere else, so the manifest is
        required rather than optional.
        """
        lone = tmp_path / ASSET_NAME
        lone.write_bytes(BODY)
        with pytest.raises(IndexFetchError) as exc:
            side_load(lone, index_path=tmp_path / "cache" / "i.sqlite")
        message = str(exc.value)
        assert "no manifest beside" in message
        assert "embedding model" in message

    def test_a_truncated_staged_copy_is_caught(self, staged: Path, tmp_path: Path):
        staged.write_bytes(BODY[: len(BODY) // 2])
        with pytest.raises(IndexFetchError, match="incomplete or corrupt"):
            side_load(staged, index_path=tmp_path / "i.sqlite")

    def test_a_model_mismatch_refuses(self, staged: Path, tmp_path: Path):
        manifest_mod.write_manifest(staged, _manifest(embedding_model="other/model"))
        dest = tmp_path / "i.sqlite"
        with pytest.raises(manifest_mod.IndexMismatchError):
            side_load(staged, index_path=dest)
        assert not dest.exists()

    def test_side_loading_onto_itself_is_refused(self, staged: Path):
        with pytest.raises(IndexFetchError, match="already the configured index path"):
            side_load(staged, index_path=staged)

    def test_it_needs_no_network_at_all(self, staged: Path, tmp_path: Path, no_network):
        """The whole point: this is the path with no egress."""
        side_load(staged, index_path=tmp_path / "i.sqlite")
