"""What survives when an index install stops half way.

A build embeds the whole corpus into the store as it goes and takes tens of
minutes, so being interrupted is an ordinary event rather than an edge case.
The property every test here defends is narrow and specific: at no point may
the configured path hold an index that is *wrong but self-consistent*, because
that is the one state nothing downstream can detect. A missing manifest is
refused; a manifest that disagrees with its index is refused; what must never
happen is an index whose sidecars confidently describe contents it does not
have, which reads as healthy all the way through ``doctor``.

So each install path produces the index somewhere else and makes it live in one
rename, and writes its sidecars only afterwards.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from aorta.chat.config import settings
from aorta.chat.rag import index_ops
from aorta.chat.rag import manifest as manifest_mod
from aorta.chat.rag.corpus import local_corpus
from aorta.chat.rag.retriever import SqliteVecStore, collection_chunk_count

COLLECTION = "aorta_fake"
RUN_COLLECTION = f"{COLLECTION}_runs"


class _Fake(Embeddings):
    """Deterministic 8-dimension bag-of-words, so no model is downloaded."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        vector = [0.0] * 8
        for index, _token in enumerate(text.split()):
            vector[index % 8] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else [1.0] + [0.0] * 7


class _Provider:
    name = "local"

    def get_embeddings(self):
        return _Fake()

    def collection_name(self):
        return COLLECTION

    def model_id(self):
        return settings.embedding_model

    def describe(self):
        return "fake 8d embeddings"


@pytest.fixture(autouse=True)
def provider(monkeypatch):
    """Pin the provider on both sides: the builder's and the reader's."""
    from aorta.chat.rag import retriever

    monkeypatch.setattr(settings, "embedding_model", "fake/model")
    monkeypatch.setattr(index_ops, "get_provider", _Provider)
    monkeypatch.setattr(retriever, "get_provider", _Provider)


@pytest.fixture()
def corpus_root(tmp_path: Path) -> Path:
    """A small indexable tree. Grown by :func:`_grow` to change the chunk count."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "alpha.py").write_text("def alpha():\n    return 'first module'\n", encoding="utf-8")
    (root / "beta.md").write_text("# Beta\n\nSecond document.\n", encoding="utf-8")
    return root


def _grow(root: Path) -> None:
    """Add files, so a rebuild has a different chunk count than the last one."""
    for name in ("gamma.py", "delta.py", "epsilon.md"):
        (root / name).write_text(f"# {name}\n\nmore indexable text here\n", encoding="utf-8")


def _build(corpus_root: Path, target: Path):
    return index_ops.build_index(local_corpus(corpus_root), index_path=target)


def _sidecars(target: Path) -> tuple[str, str]:
    return (
        manifest_mod.manifest_path(target).read_text(encoding="utf-8"),
        manifest_mod.checksum_path(target).read_text(encoding="utf-8"),
    )


def _add_run_collection(target: Path, *, texts=("run one failed", "run two passed")) -> None:
    """Write the per-user run collection into the same file, as ``rag/runs`` does."""
    store = SqliteVecStore(path=target, embedding=_Fake(), collection=RUN_COLLECTION)
    try:
        store.reset()
        store.add_documents(
            [Document(page_content=text, metadata={"source": text}) for text in texts]
        )
    finally:
        store.close()


class TestInterruptedBuild:
    """A build killed part way must cost the user nothing they already had."""

    def test_the_previous_index_and_its_sidecars_stay_usable(
        self, corpus_root: Path, tmp_path: Path, monkeypatch
    ):
        target = tmp_path / "cache" / "index.sqlite"
        first = _build(corpus_root, target)
        before_bytes = target.read_bytes()
        before_sidecars = _sidecars(target)

        _grow(corpus_root)

        def _interrupt(self, *args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(SqliteVecStore, "add_documents", _interrupt)
        with pytest.raises(KeyboardInterrupt):
            _build(corpus_root, target)

        assert target.read_bytes() == before_bytes
        assert _sidecars(target) == before_sidecars
        # Not merely present: still accepted by the load path it has to serve.
        assert index_ops.check_index(target, strict=True).refusals == []
        assert collection_chunk_count(target, COLLECTION) == first.chunk_count

    def test_it_leaves_no_staging_directory_behind(
        self, corpus_root: Path, tmp_path: Path, monkeypatch
    ):
        target = tmp_path / "cache" / "index.sqlite"
        _build(corpus_root, target)

        def _interrupt(self, *args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(SqliteVecStore, "add_documents", _interrupt)
        with pytest.raises(KeyboardInterrupt):
            _build(corpus_root, target)

        assert sorted(p.name for p in target.parent.iterdir()) == [
            "index.sqlite",
            "index.sqlite.manifest.json",
            "index.sqlite.sha256",
        ]

    def test_a_first_ever_build_that_is_interrupted_installs_nothing(
        self, corpus_root: Path, tmp_path: Path, monkeypatch
    ):
        """With no previous index there is nothing to preserve, so leave no index."""
        target = tmp_path / "cache" / "index.sqlite"

        def _interrupt(self, *args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(SqliteVecStore, "add_documents", _interrupt)
        with pytest.raises(KeyboardInterrupt):
            _build(corpus_root, target)

        assert not target.exists()
        assert not manifest_mod.manifest_path(target).exists()


class TestSidecarsComeLast:
    """The window between the index landing and its sidecars must be detectable."""

    def test_a_build_that_dies_before_its_sidecars_is_refused_not_answered(
        self, corpus_root: Path, tmp_path: Path, monkeypatch
    ):
        target = tmp_path / "index.sqlite"
        first = _build(corpus_root, target)
        _grow(corpus_root)

        def _die(*args, **kwargs):
            raise RuntimeError("killed before the sidecars were written")

        monkeypatch.setattr(index_ops.manifest_mod, "write_manifest", _die)
        with pytest.raises(RuntimeError):
            _build(corpus_root, target)

        # The new index is in place; the sidecars still describe the old one.
        installed = collection_chunk_count(target, COLLECTION)
        assert installed != first.chunk_count
        assert manifest_mod.read_manifest(target).chunk_count == first.chunk_count

        # That state is exactly the one that used to read as healthy.
        report = index_ops.check_index(target, strict=False)
        assert any("contents" in refusal for refusal in report.refusals)
        with pytest.raises(manifest_mod.IndexMismatchError):
            index_ops.check_index(target, strict=True)


class TestPartialIndexUnderAStaleManifest:
    """The failure this whole change exists for, reproduced without timing.

    An in-place build that stopped part way left the destination holding a
    fraction of its chunks while both sidecars still described the previous,
    complete build -- self-consistent, and wrong.
    """

    @staticmethod
    def _truncate_in_place(target: Path, keep: int) -> None:
        """Drop all but ``keep`` chunks, leaving the sidecars untouched."""
        conn = sqlite3.connect(target)
        try:
            conn.execute(f'DELETE FROM "chunks_{COLLECTION}" WHERE id > ?', (keep,))
            conn.commit()
        finally:
            conn.close()

    def test_the_load_path_refuses_it(self, corpus_root: Path, tmp_path: Path):
        from aorta.chat.rag import retriever

        target = tmp_path / "index.sqlite"
        _build(corpus_root, target)
        self._truncate_in_place(target, keep=1)

        with pytest.raises(manifest_mod.IndexMismatchError) as exc:
            retriever._check_manifest(target, _Provider())
        assert "REFUSING" in str(exc.value)

    def test_doctor_reports_it_rather_than_ok(self, corpus_root: Path, tmp_path: Path, monkeypatch):
        """``doctor`` validated the manifest's existence, so it said [ ok ]."""
        from aorta.chat.doctor import FAIL, run_checks

        target = tmp_path / "index.sqlite"
        _build(corpus_root, target)
        self._truncate_in_place(target, keep=1)
        monkeypatch.setattr(settings, "index_path", str(target))

        checks = {check.name: check for check in run_checks(backend=False).checks}
        assert checks["index manifest"].status == FAIL
        assert "chunks" in checks["index manifest"].hint

    def test_an_index_matching_its_manifest_is_still_accepted(
        self, corpus_root: Path, tmp_path: Path
    ):
        """The check must not fire on the healthy case it shares a code path with."""
        target = tmp_path / "index.sqlite"
        _build(corpus_root, target)
        assert index_ops.check_index(target, strict=True).refusals == []

    def test_a_manifest_predating_the_field_is_not_refused(self, corpus_root: Path, tmp_path: Path):
        """``chunk_count`` of 0 means an older builder, not an empty index."""
        target = tmp_path / "index.sqlite"
        built = _build(corpus_root, target)
        manifest_mod.write_manifest(
            target, manifest_mod.Manifest(**{**vars(built.manifest), "chunk_count": 0})
        )
        assert index_ops.check_index(target, strict=True).refusals == []


class TestAnUnreadableIndexIsNotSkipped:
    """"Cannot be read" must not reach the same place as "is not there".

    Returning "no evidence" for both would let a file nothing can open pass the
    contents check by being too damaged to contradict its own manifest.
    """

    @staticmethod
    def _clobber(target: Path) -> None:
        """Leave the sidecars, replace the index with something that is not one."""
        target.write_bytes(b"not a sqlite database, not even close")

    def test_an_absent_collection_reports_no_evidence(self, corpus_root: Path, tmp_path: Path):
        """Not indexed by this provider yet is a state, not a failure."""
        from aorta.chat.rag.retriever import collection_chunk_count

        target = tmp_path / "index.sqlite"
        _build(corpus_root, target)
        assert collection_chunk_count(target, "aorta_some_other_provider") is None

    def test_an_absent_file_reports_no_evidence(self, tmp_path: Path):
        from aorta.chat.rag.retriever import collection_chunk_count

        assert collection_chunk_count(tmp_path / "nothing.sqlite", COLLECTION) is None

    def test_an_unreadable_file_raises_rather_than_returning_none(
        self, corpus_root: Path, tmp_path: Path
    ):
        from aorta.chat.rag.retriever import IndexUnreadableError, collection_chunk_count

        target = tmp_path / "index.sqlite"
        _build(corpus_root, target)
        self._clobber(target)
        with pytest.raises(IndexUnreadableError):
            collection_chunk_count(target, COLLECTION)

    def test_the_load_path_refuses_it_with_something_to_do(
        self, corpus_root: Path, tmp_path: Path
    ):
        from aorta.chat.rag import retriever

        target = tmp_path / "index.sqlite"
        _build(corpus_root, target)
        self._clobber(target)
        with pytest.raises(manifest_mod.IndexMismatchError) as exc:
            retriever._check_manifest(target, _Provider())
        assert "aorta chat index fetch" in str(exc.value)

    def test_doctor_reports_it_rather_than_raising(
        self, corpus_root: Path, tmp_path: Path, monkeypatch
    ):
        from aorta.chat.doctor import FAIL, run_checks

        target = tmp_path / "index.sqlite"
        _build(corpus_root, target)
        self._clobber(target)
        monkeypatch.setattr(settings, "index_path", str(target))

        checks = {check.name: check for check in run_checks(backend=False).checks}
        assert checks["index manifest"].status == FAIL
        assert "could not be read" in checks["index manifest"].hint


class TestTheLocalRunCollectionSurvives:
    """Run artifacts are per-user data in the same file, on their own cadence.

    ``rag/runs`` documents the two collections as independently refreshable, so
    an install that silently dropped one of them broke run retrieval with
    nothing on screen to say so.
    """

    def test_a_rebuild_keeps_it(self, corpus_root: Path, tmp_path: Path):
        target = tmp_path / "index.sqlite"
        _build(corpus_root, target)
        _add_run_collection(target)
        _grow(corpus_root)

        _build(corpus_root, target)

        assert collection_chunk_count(target, RUN_COLLECTION) == 2

    def test_it_stays_queryable_and_not_merely_present(self, corpus_root: Path, tmp_path: Path):
        target = tmp_path / "index.sqlite"
        _build(corpus_root, target)
        _add_run_collection(target)
        _build(corpus_root, target)

        store = SqliteVecStore(path=target, embedding=_Fake(), collection=RUN_COLLECTION)
        try:
            assert store.collection_exists()
            assert store.dimension() == 8
            # Content, not ranking: the fake embedder's ordering is not the
            # property under test, but that the vectors and their chunk rows
            # both came across and still join.
            hits = store.max_marginal_relevance_search("run one failed", k=2, fetch_k=4)
            assert {hit.page_content for hit in hits} == {"run one failed", "run two passed"}
        finally:
            store.close()

    def test_a_side_load_keeps_it(self, corpus_root: Path, tmp_path: Path):
        """The air-gapped update path replaces the file just as a fetch does."""
        staged_dir = tmp_path / "usb"
        staged_dir.mkdir()
        staged = staged_dir / index_ops.ASSET_NAME
        _grow(corpus_root)
        _build(corpus_root, staged)

        target = tmp_path / "cache" / "index.sqlite"
        target.parent.mkdir(parents=True)
        _build(corpus_root, target)
        _add_run_collection(target)

        result = index_ops.side_load(staged, index_path=target)

        assert collection_chunk_count(target, RUN_COLLECTION) == 2
        assert any("kept this machine" in warning for warning in result.warnings)

    def test_the_incoming_source_collection_still_wins(self, corpus_root: Path, tmp_path: Path):
        """Preserving private data must not resurrect the index being replaced."""
        staged_dir = tmp_path / "usb"
        staged_dir.mkdir()
        staged = staged_dir / index_ops.ASSET_NAME
        _grow(corpus_root)
        incoming = _build(corpus_root, staged)

        target = tmp_path / "index.sqlite"
        (small := tmp_path / "small").mkdir()
        (small / "only.py").write_text("x = 1\n", encoding="utf-8")
        _build(small, target)

        index_ops.side_load(staged, index_path=target)

        assert collection_chunk_count(target, COLLECTION) == incoming.chunk_count

    def test_building_it_does_not_make_the_index_look_corrupt(
        self, corpus_root: Path, tmp_path: Path
    ):
        """Why the load check counts chunks instead of hashing the file.

        The run collection shares the ``.sqlite``, so writing it changes the
        file's bytes. A whole-file digest would therefore refuse every index
        whose owner had built run retrieval -- a supported, documented state.
        """
        target = tmp_path / "index.sqlite"
        built = _build(corpus_root, target)
        _add_run_collection(target)

        assert manifest_mod.sha256_file(target) != built.manifest.index_sha256
        assert index_ops.check_index(target, strict=True).refusals == []


class TestTheCarryOverIsAllOrNothing:
    """The staged file is renamed into place whatever happens here.

    So a copy that fails half way must leave no trace of the collection it was
    copying, or the new index ships with tables that have no registry row and
    read as a missing collection.
    """

    def test_a_failure_part_way_leaves_no_partial_collection(
        self, corpus_root: Path, tmp_path: Path
    ):
        from aorta.chat.rag.retriever import carry_over_collections

        old = tmp_path / "old.sqlite"
        _build(corpus_root, old)
        _add_run_collection(old)
        # Drop the vector table but leave the registry row, so the copy gets
        # part way -- chunk table created and filled -- and then fails.
        # sqlite_vec has to be loaded to drop a vec0 table at all.
        import sqlite_vec

        conn = sqlite3.connect(old)
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute(f'DROP TABLE "vec_{RUN_COLLECTION}"')
            conn.commit()
        finally:
            conn.close()

        staged = tmp_path / "staged.sqlite"
        _build(corpus_root, staged)

        assert carry_over_collections(old, staged) == []
        assert collection_chunk_count(staged, RUN_COLLECTION) is None
        # And the index it was copying from is untouched.
        assert collection_chunk_count(old, COLLECTION) is not None

    def test_an_unreadable_source_does_not_fail_the_install(
        self, corpus_root: Path, tmp_path: Path
    ):
        """The incoming index is already verified; losing nothing is the worse trade."""
        staged_dir = tmp_path / "usb"
        staged_dir.mkdir()
        staged = staged_dir / index_ops.ASSET_NAME
        _build(corpus_root, staged)

        target = tmp_path / "cache" / "index.sqlite"
        target.parent.mkdir(parents=True)
        _build(corpus_root, target)
        target.write_bytes(b"not a database")

        result = index_ops.side_load(staged, index_path=target)

        assert result.index_path == target
        assert collection_chunk_count(target, COLLECTION) is not None


class TestInterruptedSideLoad:
    def test_a_copy_that_dies_part_way_leaves_the_previous_index_usable(
        self, corpus_root: Path, tmp_path: Path, monkeypatch
    ):
        staged_dir = tmp_path / "usb"
        staged_dir.mkdir()
        staged = staged_dir / index_ops.ASSET_NAME
        _grow(corpus_root)
        _build(corpus_root, staged)

        target = tmp_path / "cache" / "index.sqlite"
        target.parent.mkdir(parents=True)
        first = _build(corpus_root, target)
        before_bytes = target.read_bytes()
        before_sidecars = _sidecars(target)

        def _die(*args, **kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr(index_ops.shutil, "copy2", _die)
        with pytest.raises(OSError, match="no space left"):
            index_ops.side_load(staged, index_path=target)

        assert target.read_bytes() == before_bytes
        assert _sidecars(target) == before_sidecars
        assert collection_chunk_count(target, COLLECTION) == first.chunk_count
        assert sorted(p.name for p in target.parent.iterdir()) == [
            "index.sqlite",
            "index.sqlite.manifest.json",
            "index.sqlite.sha256",
        ]
