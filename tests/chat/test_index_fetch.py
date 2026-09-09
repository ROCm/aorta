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
import shlex
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
#: Rows in the published asset's chunk table, and therefore in its manifest.
CHUNKS = 3


def _published_asset(chunks: int = CHUNKS) -> bytes:
    """The published index, as a real sqlite store rather than filler bytes.

    ``fetch`` now opens the file before deciding a refresh would change nothing,
    so an asset of opaque bytes would make every install here look damaged and
    the skip unreachable. Only the chunk table is built: nothing in this module
    queries vectors, and ``collection_chunk_count`` -- the reader behind that
    decision -- counts the table with raw sqlite and no embedding provider.
    """
    import sqlite3
    import tempfile

    with tempfile.TemporaryDirectory() as staging:
        path = Path(staging) / ASSET_NAME
        conn = sqlite3.connect(path)
        try:
            conn.execute(f'CREATE TABLE "chunks_{COLLECTION}" (id INTEGER PRIMARY KEY, text TEXT)')
            conn.executemany(
                f'INSERT INTO "chunks_{COLLECTION}" (text) VALUES (?)',
                [(f"chunk {n}",) for n in range(chunks)],
            )
            conn.commit()
        finally:
            conn.close()
        return path.read_bytes()


BODY = _published_asset()


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
        "chunk_count": CHUNKS,
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
        # An older *fetched* index, sidecar included -- which is what every
        # install path leaves behind, and what a routine refresh runs over.
        manifest_mod.write_manifest(dest, _manifest(index_sha256="", corpus_roots=["docs"]))
        fetch_index(version="0.2.1", index_path=dest)
        assert dest.read_bytes() == BODY

    def test_the_source_and_notes_are_reported_back(self, server, tmp_path: Path):
        """The dev-version note is a ``note``, not a ``warning``.

        Nothing is wrong with a dev install taking the rolling asset, and the
        note's real job is to be shown *before* the download rather than after
        it -- which is what ``notes`` exists to make possible.
        """
        result = fetch_index(
            index_path=tmp_path / "i.sqlite",
            source=resolve_source(installed="0.2.2.dev122+g45edc3d"),
        )
        assert ROLLING_TAG in result.source
        assert any("122 commit" in note for note in result.notes)


class TestTheWaitIsExplainedBeforeItStarts:
    """The reported symptom was no output at all, for minutes.

    Three requests in series, and the only ``Downloading`` line sat on the
    third; meanwhile everything the command echoed ran after the last one
    returned. So a node that could not reach the release host produced nothing
    whatsoever before it failed, and a fetch that was going to be refused on
    the manifest still said nothing about which asset it had been reaching for.
    """

    def test_the_target_is_describable_without_touching_the_network(self, no_network):
        source = resolve_source(installed="0.2.1")
        lines = "\n".join(index_ops.describe_target(source, "/cache/index.sqlite"))

        assert "v0.2.1" in lines
        assert source.index_url in lines
        assert "/cache/index.sqlite" in lines

    def test_a_dev_installs_note_is_carried_up_front(self):
        """It explains the rolling tag, so it is worth nothing after the fact."""
        source = resolve_source(installed="0.2.2.dev122+g45edc3d")
        lines = "\n".join(index_ops.describe_target(source, "/cache/i.sqlite"))

        assert ROLLING_TAG in lines
        assert "122 commit" in lines

    def test_every_request_says_what_it_is_contacting(self, server, tmp_path: Path, caplog):
        """Including the two sidecars, which had no logging at all."""
        with caplog.at_level("INFO", logger=index_ops.__name__):
            fetch_index(version="0.2.1", index_path=tmp_path / "i.sqlite")

        logged = "\n".join(record.getMessage() for record in caplog.records)
        for suffix in (
            manifest_mod.MANIFEST_SUFFIX,
            manifest_mod.CHECKSUM_SUFFIX,
        ):
            assert ASSET_NAME + suffix in logged, f"no log line for the {suffix} request"

    def test_the_asset_transfer_reports_progress(self, server, tmp_path: Path, caplog):
        """``shutil.copyfileobj`` had no callback, so tens of megabytes were silent."""
        with caplog.at_level("INFO", logger=index_ops.__name__):
            fetch_index(version="0.2.1", index_path=tmp_path / "i.sqlite")

        assert any("MB" in record.getMessage() for record in caplog.records)


class TestTheTimeoutsAreSplit:
    """One 300 s budget for connect *and* read is five silent minutes per request.

    A blackholed host only ever reaches the connect phase, so that phase is
    what has to fail fast; the body then wants the generous budget, because the
    index is tens of megabytes over a link that may be slow.
    """

    def test_the_connect_budget_is_much_shorter_than_the_read_budget(self):
        assert index_ops._CONNECT_TIMEOUT < index_ops._READ_TIMEOUT

    def test_a_sidecar_request_uses_the_connect_budget(self, server, tmp_path: Path, monkeypatch):
        """A 1 KB sidecar that has not answered in 30 s is not going to."""
        seen: list[float | None] = []
        real = server.urlopen

        def _urlopen(url, timeout=None):
            seen.append(timeout)
            return real(url, timeout=timeout)

        monkeypatch.setattr(index_ops.urllib.request, "urlopen", _urlopen)
        fetch_index(version="0.2.1", index_path=tmp_path / "i.sqlite")

        assert seen and set(seen) == {index_ops._CONNECT_TIMEOUT}

    def test_the_body_read_is_widened_once_the_headers_are_in(self):
        """The split is only expressible after ``urlopen`` returns."""

        class _Sock:
            timeout = None

            def settimeout(self, value):
                self.timeout = value

        class _Response:
            def __init__(self, sock):
                self.fp = type("Fp", (), {"raw": type("Raw", (), {"_sock": sock})()})()

        sock = _Sock()
        index_ops._widen_read_timeout(_Response(sock))
        assert sock.timeout == index_ops._READ_TIMEOUT

    def test_a_response_with_no_reachable_socket_is_left_alone(self):
        """Best-effort: a stub response must not turn into a failed download."""
        index_ops._widen_read_timeout(io.BytesIO(b"body"))

    def test_every_request_widens_its_body_read_not_only_the_asset(
        self, server, tmp_path: Path, monkeypatch
    ):
        """The sidecar reads were left on the connect budget.

        ``urlopen``'s single timeout governs every subsequent read, so the two
        sidecar bodies kept the 30 s connect budget while only the asset was
        widened -- and the sidecars are the *first* requests the command makes,
        so theirs is the failure a user reads as "fetch does not work here".
        """
        widened: dict[str, list[float]] = {}
        real = server.urlopen

        def _urlopen(url, timeout=None):
            response = real(url, timeout=timeout)
            calls: list[float] = []
            widened[url] = calls
            sock = type("Sock", (), {"settimeout": lambda _self, value: calls.append(value)})()
            response.fp = type("Fp", (), {"raw": type("Raw", (), {"_sock": sock})()})()
            return response

        monkeypatch.setattr(index_ops.urllib.request, "urlopen", _urlopen)
        fetch_index(version="0.2.1", index_path=tmp_path / "i.sqlite")

        assert len(widened) == 3, "manifest, checksum and asset"
        assert all(calls == [index_ops._READ_TIMEOUT] for calls in widened.values()), widened

    def test_falling_back_to_the_shorter_budget_is_logged_not_silent(self, caplog):
        """The attribute chain is a standard-library detail, so its loss must be visible.

        ``response.fp.raw._sock`` is not API. The day it moves, the only
        symptom would be downloads failing at 30 s again with nothing anywhere
        saying the split had stopped applying.
        """
        import logging

        with caplog.at_level(logging.DEBUG, logger=index_ops.logger.name):
            index_ops._widen_read_timeout(io.BytesIO(b"body"))

        assert "connect timeout" in caplog.text
        assert str(index_ops._READ_TIMEOUT) in caplog.text


class TestIdentityIsCheckedBeforeTheAsset:
    """The ~1 KB manifest answers "is this usable" before the ~20 MB transfer.

    It was downloaded first and validated last, so a reporter whose embedding
    provider was misconfigured paid for the whole asset and its checksum before
    being told the result was unusable. Fixing that configuration stops this
    particular refusal; a later model or endpoint change hits the same waste.
    """

    def test_a_mismatch_never_requests_the_asset(self, server, tmp_path: Path):
        _reserialise(server, _manifest(embedding_model="other/model"))

        with pytest.raises(manifest_mod.IndexMismatchError):
            fetch_index(version="0.2.1", index_path=tmp_path / "i.sqlite")

        assert not any(url.endswith(ASSET_NAME) for url in server.requested), (
            f"the index asset was transferred anyway: {server.requested}"
        )

    def test_a_future_schema_never_requests_the_asset(self, server, tmp_path: Path):
        raw = json.loads(_manifest().to_json())
        raw["schema_version"] = manifest_mod.SCHEMA_VERSION + 1
        server.assets[ASSET_NAME + manifest_mod.MANIFEST_SUFFIX] = json.dumps(raw).encode()

        with pytest.raises(IndexFetchError, match="Upgrade aorta"):
            fetch_index(version="0.2.1", index_path=tmp_path / "i.sqlite")

        assert not any(url.endswith(ASSET_NAME) for url in server.requested)

    def test_a_clean_fetch_still_requests_all_three(self, server, tmp_path: Path):
        """Failing fast must not turn into fetching less than it needs."""
        fetch_index(version="0.2.1", index_path=tmp_path / "i.sqlite")

        assert sum(url.endswith(ASSET_NAME) for url in server.requested) == 1
        assert any(url.endswith(manifest_mod.CHECKSUM_SUFFIX) for url in server.requested)

    def test_the_installed_sidecar_keeps_a_newer_builders_extra_keys(self, server, tmp_path: Path):
        """Forward tolerance is the reason the downloaded text is what lands.

        ``Manifest.from_dict`` drops keys this version predates, so
        re-serialising the dataclass would quietly strip them from the sidecar
        this machine then keeps.
        """
        raw = json.loads(_manifest().to_json())
        raw["some_future_field"] = "keep me"
        server.assets[ASSET_NAME + manifest_mod.MANIFEST_SUFFIX] = json.dumps(raw).encode()
        dest = tmp_path / "i.sqlite"

        fetch_index(version="0.2.1", index_path=dest)

        installed = json.loads(manifest_mod.manifest_path(dest).read_text(encoding="utf-8"))
        assert installed.get("some_future_field") == "keep me"


class TestAlreadyUpToDate:
    """A refresh that would change nothing cost the whole transfer."""

    def test_a_matching_local_sidecar_skips_the_asset(self, server, tmp_path: Path):
        dest = tmp_path / "i.sqlite"
        fetch_index(version="0.2.1", index_path=dest)
        server.requested.clear()

        result = fetch_index(version="0.2.1", index_path=dest)

        assert result.up_to_date is True
        assert not any(url.endswith(ASSET_NAME) for url in server.requested)
        # Only the manifest: the checksum file is the asset's, so it is not
        # worth a request when the asset is not being fetched.
        assert all(url.endswith(manifest_mod.MANIFEST_SUFFIX) for url in server.requested)
        assert dest.read_bytes() == BODY

    def test_a_differing_published_index_is_installed_and_says_what_changed(
        self, server, tmp_path: Path
    ):
        import hashlib

        dest = tmp_path / "i.sqlite"
        fetch_index(version="0.2.1", index_path=dest)

        newer = BODY + b" plus a commit"
        digest = hashlib.sha256(newer).hexdigest()
        server.assets[ASSET_NAME] = newer
        server.assets[ASSET_NAME + manifest_mod.CHECKSUM_SUFFIX] = (
            f"{digest}  {ASSET_NAME}\n".encode()
        )
        _reserialise(
            server,
            _manifest(
                index_sha256=digest,
                aorta_sha="99beef0" + "0" * 33,
                corpus_digest="def456",
                built_at="2026-09-05T00:00:00+00:00",
            ),
        )

        result = fetch_index(version="0.2.1", index_path=dest)

        assert result.up_to_date is False
        assert dest.read_bytes() == newer
        reported = " ".join(result.changes)
        assert "corpus_digest" in reported
        assert "aorta_sha" in reported
        assert "built_at" in reported

    def test_a_matching_sidecar_over_a_damaged_store_is_not_up_to_date(
        self, server, tmp_path: Path
    ):
        """The sidecar says which index was installed, not that it still is one.

        ``check_index`` was already changed on this branch to open the store
        rather than take the manifest's word for the contents. Deciding a
        refresh on the manifest alone contradicted that from the other side:
        the one command that would repair a damaged index reported it as
        already current and did nothing.
        """
        dest = tmp_path / "i.sqlite"
        fetch_index(version="0.2.1", index_path=dest)
        # The sidecars are left exactly as the install wrote them; only the
        # store is damaged, which is the pairing the shortcut used to trust.
        dest.write_bytes(b"not a database any more")
        server.requested.clear()

        result = fetch_index(version="0.2.1", index_path=dest)

        assert result.up_to_date is False
        assert any(url.endswith(ASSET_NAME) for url in server.requested)
        assert dest.read_bytes() == BODY, "the damaged store must be repaired"
        assert index_ops.check_index(dest, strict=True).refusals == []

    def test_a_matching_sidecar_over_a_store_missing_its_chunks_is_not_up_to_date(
        self, server, tmp_path: Path
    ):
        """The other half of the same reader: readable, but not what is claimed.

        A store that opens fine and holds a different number of chunks than the
        manifest describes is the interrupted-install pairing, and it is
        exactly what ``check_index`` was taught to catch.
        """
        dest = tmp_path / "i.sqlite"
        fetch_index(version="0.2.1", index_path=dest)
        dest.write_bytes(_published_asset(chunks=CHUNKS + 2))
        server.requested.clear()

        assert fetch_index(version="0.2.1", index_path=dest).up_to_date is False
        assert any(url.endswith(ASSET_NAME) for url in server.requested)

    def test_a_local_index_with_no_sidecar_is_refused_rather_than_assumed(
        self, server, tmp_path: Path
    ):
        """An index nothing describes is not evidence that it is up to date.

        It is not evidence that it is safe to replace either, which is why this
        is a refusal rather than a silent re-fetch: the same file could be a
        local build whose sidecar was lost, and nothing here can tell.
        """
        dest = tmp_path / "i.sqlite"
        dest.write_bytes(BODY)

        with pytest.raises(index_ops.IndexOverwriteError, match="no manifest beside it"):
            fetch_index(version="0.2.1", index_path=dest)

    def test_forcing_over_a_sidecar_less_index_installs_rather_than_skips(
        self, server, tmp_path: Path
    ):
        """``--force`` is the escape the refusal names, and it must transfer.

        Reporting *up to date* here would be the original defect wearing a
        flag: there is no sidecar to have matched anything.
        """
        dest = tmp_path / "i.sqlite"
        dest.write_bytes(BODY)

        result = fetch_index(version="0.2.1", index_path=dest, force=True)

        assert result.up_to_date is False
        assert any(url.endswith(ASSET_NAME) for url in server.requested)
        assert manifest_mod.read_manifest(dest).collection == COLLECTION

    def test_a_manifest_predating_index_sha256_does_not_compare_equal_on_nothing(
        self, server, tmp_path: Path
    ):
        """Two empty digests must not read as a match and skip a needed download."""
        dest = tmp_path / "i.sqlite"
        dest.write_bytes(b"stale")
        manifest_mod.write_manifest(dest, _manifest(index_sha256=""))
        _reserialise(server, _manifest(index_sha256=""))

        assert fetch_index(version="0.2.1", index_path=dest).up_to_date is False
        assert dest.read_bytes() == BODY

    @pytest.mark.parametrize(
        "value",
        [None, 42, {"sha": "abc"}, ["abc"]],
        ids=["null", "int", "object", "list"],
    )
    @pytest.mark.parametrize("field_name", ["aorta_sha", "corpus_digest", "built_at"])
    def test_a_non_string_digest_is_rendered_rather_than_raised(self, field_name, value):
        """The ``corpus_roots`` hole, one field over, pinned at the renderer.

        Truncating a field that is not a string raised ``TypeError`` --
        ``KeyError`` for a JSON object, which subscripts by key -- out of the
        comparison whose entire job is to report on a manifest that looks
        wrong, and past the CLI's error guard. Both ``fetch``'s change report
        and ``index status``'s differences come from this one function, so
        pinning it here covers both.

        Driven with constructed manifests rather than a sidecar on disk,
        deliberately. Whether ``Manifest.from_dict`` *admits* a null digest is
        the parser's question, and PR #463 is tightening it to refuse one; what
        this test is about is the renderer downstream surviving a field it did
        not expect, whichever way the parser lands. Routed through the parser,
        it would silently change verdict on a merge it has no conflict with.

        ``built_at`` is the control rather than a regression case: it was never
        truncated, so it survived all four values before this fix and is here
        to pin that rendering did not break the field that already worked.
        """
        changes = index_ops._refresh_notes(_manifest(**{field_name: value}), _manifest())

        assert any(change.startswith(field_name) for change in changes), changes

    def test_an_empty_digest_is_reported_as_unknown_rather_than_as_nothing(
        self, server, tmp_path: Path
    ):
        """``index status`` reads the same notes, so it had the same traceback.

        Empty rather than non-string, because that is the shape that reaches
        this path through a real sidecar both before and after #463's parser
        change -- and it is the case the renderer's ``None`` handling exists
        for: blank becomes the caller's own word for a missing field, not the
        string ``"None"``.
        """
        dest = tmp_path / "i.sqlite"
        dest.write_bytes(b"stale")
        manifest_mod.write_manifest(dest, _manifest(index_sha256="d" * 64, aorta_sha=""))

        comparison = index_ops.compare_index(version="0.2.1", index_path=dest)

        assert comparison.verdict == index_ops.VERDICT_PUBLISHED_DIFFERS
        reported = " ".join(comparison.differences)
        assert "aorta_sha: unknown ->" in reported, comparison.differences
        assert "None" not in reported


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

    def test_a_refused_fetch_leaves_an_index_that_was_already_there(self, server, tmp_path: Path):
        """The promise on a destination that is not empty.

        The test above fetches onto a fresh path, so it pins that nothing is
        created. What ``rag-index.md`` promises is the stronger claim a reader
        is actually deciding on -- that a refusal costs them the download and
        nothing else -- and that is the mirror of
        ``test_it_replaces_an_existing_index``: the staging directory sits
        beside the destination and installing is a rename, so a refusal raised
        before ``_install_staged`` must leave both files exactly as they were.
        """
        _reserialise(server, _manifest(embedding_model="other/model"))
        dest = tmp_path / "index.sqlite"
        dest.write_bytes(b"the index the user already had")
        existing = manifest_mod.manifest_path(dest)
        existing.write_text('{"schema_version": 1}', encoding="utf-8")

        with pytest.raises(manifest_mod.IndexMismatchError):
            fetch_index(version="0.2.1", index_path=dest)

        assert dest.read_bytes() == b"the index the user already had"
        assert existing.read_text(encoding="utf-8") == '{"schema_version": 1}'
        assert sorted(p.name for p in dest.parent.iterdir()) == [
            "index.sqlite",
            "index.sqlite.manifest.json",
        ], "a refused fetch should not leave staging behind either"

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
        """Not a ``TypeError`` out of a comparison the caller never guarded.

        Matched on this module's own wrapper rather than on the sentence
        ``manifest.py`` raises underneath it. Which of the two layers refuses a
        non-integer ``schema_version`` -- the field type check or the supported
        -schema check -- is that module's business, and PR #463 is moving it
        from the second to the first. What ``fetch`` owes the caller either way
        is an ``IndexFetchError`` that names the manifest it could not use, and
        no partial install.
        """
        self._serve_schema(server, value)
        dest = tmp_path / "i.sqlite"

        with pytest.raises(IndexFetchError) as exc:
            fetch_index(version="0.2.1", index_path=dest)

        message = str(exc.value)
        assert "is not usable" in message
        assert "schema" in message
        assert ASSET_NAME + manifest_mod.MANIFEST_SUFFIX in message
        assert not dest.exists()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("embedding_identity", 42),
            ("embedding_model", None),
            ("collection", ["c"]),
            ("corpus_digest", 7),
            ("corpus_roots", "src/aorta"),
            ("dimensions", "384"),
            ("chunk_count", True),
        ],
    )
    def test_a_published_field_of_the_wrong_type_is_refused_not_raised(
        self, server, tmp_path: Path, field, value
    ):
        """The tolerance for a wrong-typed field stops at the network boundary.

        A *local* sidecar that records nonsense must stay readable, because the
        reports whose job is to say what is wrong with it have to be able to
        read it -- so ``from_dict`` type-checks nothing and this module's
        readers absorb it. A *downloaded* one has nowhere better to fail than
        here, and not everything it reaches is in this module: ``validate``
        calls ``.split()`` on ``embedding_identity``, so a published manifest
        carrying ``42`` there came out of the CLI as an ``AttributeError``
        traceback rather than as the refusal a bad manifest is supposed to be.

        Parametrised across the types rather than pinned to that one field,
        because the hole was the missing check and not the field that exposed
        it.
        """
        raw = json.loads(
            server.assets[ASSET_NAME + manifest_mod.MANIFEST_SUFFIX].decode("utf-8")
        )
        raw[field] = value
        server.assets[ASSET_NAME + manifest_mod.MANIFEST_SUFFIX] = json.dumps(raw).encode()
        dest = tmp_path / "i.sqlite"

        with pytest.raises(IndexFetchError) as exc:
            fetch_index(version="0.2.1", index_path=dest)

        message = str(exc.value)
        assert "is not usable" in message
        assert field in message
        assert not dest.exists(), "a manifest this build cannot read must install nothing"

    def test_the_404_options_also_name_the_destination_that_was_asked_for(
        self, server, tmp_path: Path, monkeypatch
    ):
        """The fourth site with the same defect, found by sweeping for it.

        Copilot named three refusals whose remedy dropped a non-default
        ``--output``. This one is the same class one layer down: a missing
        asset offers three ordinary invocations, and an ordinary invocation
        installs to the cache -- so a reader who had asked for a different path
        gets three lines that all act somewhere else.
        """
        monkeypatch.setattr(settings, "index_path", str(tmp_path / "cache.sqlite"))
        del server.assets[ASSET_NAME]
        dest = tmp_path / "elsewhere.sqlite"

        with pytest.raises(IndexFetchError) as exc:
            fetch_index(version="0.2.1", index_path=dest)

        flag = f"--output {shlex.quote(str(dest))}"
        options = [
            line for line in str(exc.value).splitlines() if "aorta chat index" in line
        ]
        assert len(options) == 3, options
        for line in options:
            assert flag in line, f"option acts on the wrong index: {line!r}"

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


class TestProvenanceIsReadOffTheManifest:
    """Which side built an index decides which guard applies to it.

    Inferred rather than stored: a schema field would be clearer and would
    cost a manifest version bump, and the two corpora already label their roots
    differently.
    """

    def test_published_roots_are_repository_relative(self):
        manifest = _manifest(corpus_roots=["src/aorta", "docs", "README.md"])
        assert index_ops.index_provenance(manifest) == index_ops.PROVENANCE_PUBLISHED

    def test_a_local_build_records_one_absolute_path(self):
        manifest = _manifest(corpus_roots=["/home/someone/src/aorta"])
        assert index_ops.index_provenance(manifest) == index_ops.PROVENANCE_LOCAL

    def test_a_manifest_predating_the_field_is_unknown_rather_than_guessed(self):
        """Neither guard fires without positive evidence of what it protects."""
        assert index_ops.index_provenance(_manifest()) == index_ops.PROVENANCE_UNKNOWN

    @pytest.mark.parametrize(
        "roots",
        ["src/aorta", "docs", None, 7, {"src/aorta": 1}, ["src/aorta", 42]],
        ids=["str-with-slash", "bare-str", "null", "int", "object", "mixed-list"],
    )
    def test_a_recorded_value_that_is_not_a_list_of_paths_is_invalid(self, roots):
        """``Manifest.from_dict`` type-checks nothing, so this has to.

        Iterating a string yields characters and ``Path("/").is_absolute()`` is
        true, so ``"src/aorta"`` classified as a *local build* and ``"docs"``
        as a *published* one -- both answers derived from nothing -- while
        ``[42]`` raised ``TypeError`` straight past the CLI's error guard.
        """
        manifest = manifest_mod.Manifest.from_dict(
            {**json.loads(_manifest().to_json()), "corpus_roots": roots}
        )

        assert index_ops.index_provenance(manifest) == index_ops.PROVENANCE_INVALID

    def test_the_status_payload_reports_an_unusable_value_as_null(self):
        """Not as the characters of a string, which is what ``list()`` gave."""
        manifest = manifest_mod.Manifest.from_dict(
            {**json.loads(_manifest().to_json()), "corpus_roots": "src/aorta"}
        )

        assert index_ops._side(manifest)["corpus_roots"] is None

    def test_the_two_sides_of_the_payload_declare_one_key_set(self):
        """The invariant, not the instance: a new field is covered for free."""
        assert set(index_ops._side(_manifest())) == set(index_ops._SIDE_FIELDS)
        assert set(index_ops._side(None)) == set(index_ops._SIDE_FIELDS)

    def test_a_renamed_published_subpath_is_still_published(self):
        """Read the shape, not the list, so renaming `docs/` reclassifies nothing."""
        manifest = _manifest(corpus_roots=["src/aorta", "documentation"])
        assert index_ops.index_provenance(manifest) == index_ops.PROVENANCE_PUBLISHED

    def test_the_same_rule_classifies_a_corpus_before_it_is_built(self):
        """So `build` can compare what it is about to make against what is there."""
        from aorta.chat.rag import corpus as corpus_mod

        published = corpus_mod.Corpus(
            base=Path("/repo"), subpaths=("src/aorta",), roots_label=corpus_mod.PUBLISHED_SUBPATHS
        )
        assert index_ops.corpus_provenance(published) == index_ops.PROVENANCE_PUBLISHED

        local = corpus_mod.Corpus(base=Path("/repo"), subpaths=(".",), roots_label=("/repo",))
        assert index_ops.corpus_provenance(local) == index_ops.PROVENANCE_LOCAL


class TestFetchWillNotSilentlyDiscardALocalBuild:
    """The two directions are not symmetric, so neither is a blanket guard.

    A fetched index costs a download to replace. A locally built one may not be
    reproducible at all -- the tree it indexed may have moved, and an
    air-gapped node cannot re-download the weights. So the incoming-index paths
    guard against overwriting a local build, and only that.
    """

    @staticmethod
    def _install_local_build(dest: Path) -> None:
        dest.write_bytes(b"an index built here")
        manifest_mod.write_manifest(
            dest,
            _manifest(
                corpus_roots=[str(dest.parent / "checkout")],
                index_sha256=manifest_mod.sha256_file(dest),
            ),
        )

    def test_fetch_over_a_local_build_refuses_and_names_the_flag(self, server, tmp_path: Path):
        dest = tmp_path / "i.sqlite"
        self._install_local_build(dest)

        with pytest.raises(index_ops.IndexOverwriteError) as exc:
            fetch_index(version="0.2.1", index_path=dest)

        message = str(exc.value)
        assert "built on this machine" in message
        assert "--force" in message
        assert dest.read_bytes() == b"an index built here"

    def test_it_refuses_before_any_request(self, server, tmp_path: Path):
        """The local manifest is free to read, so the refusal costs no network."""
        dest = tmp_path / "i.sqlite"
        self._install_local_build(dest)

        with pytest.raises(index_ops.IndexOverwriteError):
            fetch_index(version="0.2.1", index_path=dest)

        assert server.requested == []

    def test_force_overwrites_it(self, server, tmp_path: Path):
        dest = tmp_path / "i.sqlite"
        self._install_local_build(dest)

        fetch_index(version="0.2.1", index_path=dest, force=True)

        assert dest.read_bytes() == BODY

    def test_force_also_re_downloads_an_identical_asset(self, server, tmp_path: Path):
        """Otherwise the up-to-date short-circuit would swallow the override."""
        dest = tmp_path / "i.sqlite"
        fetch_index(version="0.2.1", index_path=dest)
        server.requested.clear()

        assert fetch_index(version="0.2.1", index_path=dest, force=True).up_to_date is False
        assert any(url.endswith(ASSET_NAME) for url in server.requested)

    def test_a_routine_refresh_of_a_fetched_index_is_not_refused(self, server, tmp_path: Path):
        """`fetch` is the documented way to refresh; demanding --force would
        train people to always pass it."""
        dest = tmp_path / "i.sqlite"
        dest.write_bytes(b"stale")
        manifest_mod.write_manifest(
            dest, _manifest(corpus_roots=["src/aorta", "docs", "README.md"], index_sha256="0" * 64)
        )

        assert fetch_index(version="0.2.1", index_path=dest).index_path == dest
        assert dest.read_bytes() == BODY

    def test_an_unclassifiable_manifest_is_refused_rather_than_guessed(
        self, server, tmp_path: Path
    ):
        """The guard must not be reachable around by a broken sidecar.

        A hand-carried manifest recording ``corpus_roots`` as a scalar used to
        be classified by iterating it: ``"docs"`` came out *published*, so the
        fetch went ahead and overwrote whatever was there, and ``[42]`` left a
        ``TypeError`` traceback. Neither answer is available, so this refuses.
        """
        dest = tmp_path / "i.sqlite"
        dest.write_bytes(b"an index of unknown provenance")
        raw = json.loads(_manifest(index_sha256="0" * 64).to_json())
        raw["corpus_roots"] = "docs"
        manifest_mod.manifest_path(dest).write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(index_ops.IndexOverwriteError, match="cannot classify"):
            fetch_index(version="0.2.1", index_path=dest)

        assert dest.read_bytes() == b"an index of unknown provenance"
        assert server.requested == [], "it should refuse before any request"
        assert fetch_index(version="0.2.1", index_path=dest, force=True).index_path == dest

    def test_a_truncated_sidecar_is_refused_rather_than_read_as_absence(
        self, server, tmp_path: Path
    ):
        """Unreadable and absent used to be the same answer, and they are not.

        A half-written sidecar makes the index beside it unclassifiable, which
        is the state the guard above already refuses when the *value* is
        unreadable. Collapsing it into "there is nothing here" refused the
        better-informed case and waved through the worse one.
        """
        dest = tmp_path / "i.sqlite"
        self._install_local_build(dest)
        manifest_mod.manifest_path(dest).write_text('{"aorta_version": "0.2', encoding="utf-8")

        with pytest.raises(index_ops.IndexOverwriteError) as exc:
            fetch_index(version="0.2.1", index_path=dest)

        assert "no manifest beside it" in str(exc.value)
        assert dest.read_bytes() == b"an index built here"
        assert server.requested == [], "it should refuse before any request"

    def test_a_first_install_is_still_a_first_install(self, server, tmp_path: Path):
        """The whole point of the distinction: a path that does not exist."""
        dest = tmp_path / "nested" / "i.sqlite"

        assert fetch_index(version="0.2.1", index_path=dest).index_path == dest
        assert dest.read_bytes() == BODY

    def test_side_load_carries_the_same_guard(self, tmp_path: Path):
        """Or `--from` becomes the way around it by accident."""
        staging = tmp_path / "usb"
        staging.mkdir()
        origin = staging / ASSET_NAME
        origin.write_bytes(BODY)
        manifest_mod.write_manifest(origin, _manifest())

        dest = tmp_path / "i.sqlite"
        self._install_local_build(dest)

        with pytest.raises(index_ops.IndexOverwriteError, match="built on this machine"):
            side_load(origin, index_path=dest)

        assert side_load(origin, index_path=dest, force=True).index_path == dest

    def test_side_load_also_refuses_a_manifest_less_destination(self, tmp_path: Path):
        """Both guards read the destination the same way, so both had the hole."""
        staging = tmp_path / "usb"
        staging.mkdir()
        origin = staging / ASSET_NAME
        origin.write_bytes(BODY)
        manifest_mod.write_manifest(origin, _manifest())

        dest = tmp_path / "notes.txt"
        dest.write_text("a year of notes", encoding="utf-8")

        with pytest.raises(index_ops.IndexOverwriteError, match="no manifest beside it"):
            side_load(origin, index_path=dest)

        assert dest.read_text(encoding="utf-8") == "a year of notes"
        assert side_load(origin, index_path=dest, force=True).index_path == dest

    def test_the_remedy_names_the_index_that_was_refused(
        self, server, tmp_path: Path, monkeypatch
    ):
        """``fetch --force`` without ``--output`` fetches over a different index.

        The refused index stays refused and the cache -- which the user did not
        mention and may well have wanted -- is replaced instead. The flag is
        carried only when it differs from what the bare command resolves to, so
        the ordinary refusal over the cache keeps its short line.
        """
        monkeypatch.setattr(settings, "index_path", str(tmp_path / "cache.sqlite"))
        dest = tmp_path / "elsewhere" / "i.sqlite"
        dest.parent.mkdir()
        self._install_local_build(dest)

        with pytest.raises(index_ops.IndexOverwriteError) as exc:
            fetch_index(version="0.2.1", index_path=dest)

        message = str(exc.value)
        assert f"aorta chat index fetch --output {shlex.quote(str(dest))} --force" in message

        monkeypatch.setattr(settings, "index_path", str(dest))
        with pytest.raises(index_ops.IndexOverwriteError) as plain:
            fetch_index(version="0.2.1", index_path=dest)
        assert "--output" not in str(plain.value)

    def test_the_side_load_remedy_keeps_both_the_source_and_the_destination(
        self, tmp_path: Path, monkeypatch
    ):
        """``--from`` survived the refusal; the destination beside it did not.

        Pasting the advertised line therefore side-loaded the staged file over
        the configured cache -- an install the user never asked for, from a
        refusal about a different path entirely.
        """
        monkeypatch.setattr(settings, "index_path", str(tmp_path / "cache.sqlite"))
        staging = tmp_path / "usb"
        staging.mkdir()
        origin = staging / ASSET_NAME
        origin.write_bytes(BODY)
        manifest_mod.write_manifest(origin, _manifest())

        dest = tmp_path / "elsewhere" / "i.sqlite"
        dest.parent.mkdir()
        self._install_local_build(dest)

        with pytest.raises(index_ops.IndexOverwriteError) as exc:
            side_load(origin, index_path=dest)

        remedy = (
            f"aorta chat index fetch --from {shlex.quote(str(origin))} "
            f"--output {shlex.quote(str(dest))} --force"
        )
        assert remedy in str(exc.value)


class TestCompareIndex:
    """Gap 4b: "is the cached index the same as the remote one, and which is newer".

    Answerable from the two manifests alone, which is what `nightly.yml`
    already does in bash to decide whether to republish -- so the capability
    was proven and load-bearing for the release process while the CLI did not
    expose it.
    """

    def test_an_identical_pair_is_up_to_date(self, server, tmp_path: Path):
        dest = tmp_path / "i.sqlite"
        fetch_index(version="0.2.1", index_path=dest)

        comparison = index_ops.compare_index(version="0.2.1", index_path=dest)

        assert comparison.verdict == index_ops.VERDICT_UP_TO_DATE
        assert comparison.up_to_date is True

    def test_it_transfers_only_the_manifest(self, server, tmp_path: Path):
        """The whole point of comparing manifests rather than indexes."""
        dest = tmp_path / "i.sqlite"
        fetch_index(version="0.2.1", index_path=dest)
        server.requested.clear()

        index_ops.compare_index(version="0.2.1", index_path=dest)

        assert server.requested
        assert all(url.endswith(manifest_mod.MANIFEST_SUFFIX) for url in server.requested)

    def test_it_writes_nothing(self, server, tmp_path: Path):
        """Read-only, so it is safe to suggest to someone who is already stuck."""
        dest = tmp_path / "cache" / "i.sqlite"
        dest.parent.mkdir(parents=True)

        index_ops.compare_index(version="0.2.1", index_path=dest)

        assert list(dest.parent.iterdir()) == []

    def test_an_unreachable_baseline_is_not_reported_as_up_to_date(
        self, monkeypatch, tmp_path: Path
    ):
        """`nightly.yml`'s own comment: no baseline must not read as no change."""

        def _refuse(url, timeout=None):  # noqa: ARG001 - signature match
            raise urllib.error.URLError("Network is unreachable")

        monkeypatch.setattr(index_ops.urllib.request, "urlopen", _refuse)
        dest = tmp_path / "i.sqlite"
        dest.write_bytes(BODY)
        manifest_mod.write_manifest(dest, _manifest())

        comparison = index_ops.compare_index(version="0.2.1", index_path=dest)

        assert comparison.verdict == index_ops.VERDICT_NO_BASELINE
        assert comparison.up_to_date is False
        assert "Network is unreachable" in comparison.baseline_error

    def test_a_missing_published_manifest_is_no_baseline_rather_than_an_exception(
        self, server, tmp_path: Path
    ):
        del server.assets[ASSET_NAME + manifest_mod.MANIFEST_SUFFIX]

        comparison = index_ops.compare_index(version="0.2.1", index_path=tmp_path / "i.sqlite")

        assert comparison.verdict == index_ops.VERDICT_NO_BASELINE

    def test_no_local_index_is_its_own_verdict(self, server, tmp_path: Path):
        comparison = index_ops.compare_index(version="0.2.1", index_path=tmp_path / "absent")

        assert comparison.verdict == index_ops.VERDICT_NO_LOCAL_INDEX
        assert comparison.local is None
        assert comparison.published is not None
        assert comparison.local_error == ""

    def test_an_unreadable_sidecar_is_not_reported_as_no_local_index(
        self, server, tmp_path: Path
    ):
        """The verdict said nothing had been installed, over a file sitting there.

        Both states read the manifest and got ``None``, so ``index status``
        told the one person looking at it the opposite of what was wrong.
        """
        dest = tmp_path / "i.sqlite"
        dest.write_bytes(BODY)
        manifest_mod.manifest_path(dest).write_text('{"aorta_version": "0.2', encoding="utf-8")

        comparison = index_ops.compare_index(version="0.2.1", index_path=dest)

        assert comparison.verdict == index_ops.VERDICT_UNREADABLE_LOCAL_INDEX
        assert comparison.local_error, "the cause must be reported, not only the state"
        payload = index_ops.comparison_to_dict(comparison)
        assert payload["local"]["error"] == comparison.local_error
        assert payload["up_to_date"] is False

    def test_a_differing_published_index_lists_the_fields(self, server, tmp_path: Path):
        dest = tmp_path / "i.sqlite"
        dest.write_bytes(b"stale")
        manifest_mod.write_manifest(
            dest,
            _manifest(
                corpus_roots=["src/aorta", "docs", "README.md"],
                index_sha256="0" * 64,
                corpus_digest="old123",
                aorta_sha="0ldsha0" + "0" * 33,
            ),
        )

        comparison = index_ops.compare_index(version="0.2.1", index_path=dest)

        assert comparison.verdict == index_ops.VERDICT_PUBLISHED_DIFFERS
        reported = " ".join(comparison.differences)
        assert "corpus_digest" in reported
        assert "aorta_sha" in reported

    def test_a_locally_built_index_is_not_called_stale(self, server, tmp_path: Path):
        """ "Newer" is the wrong frame here: fetching would discard it.

        ``built_at`` is wall-clock from whoever built the index, so a local one
        can carry a later timestamp over *older* source. This verdict says what
        is true -- the two are not versions of each other.
        """
        dest = tmp_path / "i.sqlite"
        dest.write_bytes(b"built here")
        manifest_mod.write_manifest(
            dest, _manifest(corpus_roots=["/home/dev/aorta"], index_sha256="1" * 64)
        )

        comparison = index_ops.compare_index(version="0.2.1", index_path=dest)

        assert comparison.verdict == index_ops.VERDICT_LOCALLY_BUILT
        assert comparison.provenance == index_ops.PROVENANCE_LOCAL

    def test_an_unusable_published_index_says_so_rather_than_offering_it(
        self, server, tmp_path: Path
    ):
        """Being newer does not make an index this install cannot query useful."""
        _reserialise(server, _manifest(embedding_model="other/model"))

        comparison = index_ops.compare_index(version="0.2.1", index_path=tmp_path / "i.sqlite")

        assert comparison.verdict == index_ops.VERDICT_INCOMPATIBLE

    def test_the_payload_names_which_asset_it_compared_against(self, server, tmp_path: Path):
        """A dev install resolves to the rolling tag, so the verdict is
        ambiguous without this."""
        comparison = index_ops.compare_index(
            index_path=tmp_path / "i.sqlite",
            source=resolve_source(installed="0.2.2.dev122+g45edc3d"),
        )
        payload = index_ops.comparison_to_dict(comparison)

        assert ROLLING_TAG in payload["compared_against"]["source"]
        assert ROLLING_TAG in payload["compared_against"]["manifest_url"]

    def test_every_verdict_has_a_sentence(self):
        """A raw enum value reaching the user would be the register's own complaint."""
        for verdict in (
            index_ops.VERDICT_UP_TO_DATE,
            index_ops.VERDICT_PUBLISHED_DIFFERS,
            index_ops.VERDICT_LOCALLY_BUILT,
            index_ops.VERDICT_INCOMPATIBLE,
            index_ops.VERDICT_NO_LOCAL_INDEX,
            index_ops.VERDICT_NO_BASELINE,
        ):
            assert index_ops._VERDICT_SUMMARY[verdict] != verdict

    def test_the_payload_carries_every_field_the_register_asked_for(self, server, tmp_path: Path):
        dest = tmp_path / "i.sqlite"
        fetch_index(version="0.2.1", index_path=dest)

        payload = index_ops.comparison_to_dict(
            index_ops.compare_index(version="0.2.1", index_path=dest)
        )

        for side in ("local", "published"):
            for key in (
                "embedding_model",
                "embedding_identity",
                "built_at",
                "aorta_sha",
                "corpus_digest",
                "index_sha256",
            ):
                assert key in payload[side], f"{side} is missing {key}"


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

    def test_the_suggested_command_survives_an_awkward_path(self, tmp_path: Path):
        """The refusal prints a command to run, and the path is the user's.

        ``--from /home/o'brien/staged index.sqlite`` pasted back into a shell
        does not parse -- unbalanced quote, then word splitting. Same class as
        the pre-seed procedure #463 fixed; the remedy here is only the shell
        half, since nothing interpolates into a Python literal.
        """
        import shlex

        staging = tmp_path / "o'brien dir"
        staging.mkdir()
        origin = staging / ASSET_NAME
        origin.write_bytes(BODY)
        manifest_mod.write_manifest(origin, _manifest())

        dest = tmp_path / "cache" / "index.sqlite"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"an index built here")
        manifest_mod.write_manifest(
            dest,
            _manifest(
                corpus_roots=[str(tmp_path / "checkout")],
                index_sha256=manifest_mod.sha256_file(dest),
            ),
        )

        with pytest.raises(index_ops.IndexOverwriteError) as exc:
            side_load(origin, index_path=dest)

        suggested = next(
            line for line in str(exc.value).splitlines() if "--from" in line
        ).split(":", 1)[1]
        # The whole point: it parses, and both paths come back as the ones we
        # passed in. ``--output`` is here because the destination is not the
        # configured cache -- a remedy that dropped it would side-load over an
        # index the user never named while leaving this refusal standing.
        assert shlex.split(suggested) == [
            "aorta",
            "chat",
            "index",
            "fetch",
            "--from",
            str(origin.resolve()),
            "--output",
            str(dest),
            "--force",
        ]

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
