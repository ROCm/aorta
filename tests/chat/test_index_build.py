"""Local and published index builds, and the two guards on what gets published.

The published index is a redistribution of source text -- ``SqliteVecStore``
persists each chunk verbatim -- so the build has two independent guards, and the
independence is the point:

* :func:`assert_public_tree` catches the *workflow* being pointed at the wrong
  repository.
* the tracked-file allowlist catches an internal reproducer or a customer bundle
  sitting in an otherwise correct checkout, which no remote check can see.

Also covered: the corpus digest, which is what lets ``nightly.yml`` skip
re-uploading tens of megabytes on a night when nothing indexable changed.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pytest
from langchain_core.documents import Document

from aorta.chat.config import settings
from aorta.chat.rag import corpus as corpus_mod
from aorta.chat.rag.corpus import (
    PUBLISHED_SUBPATHS,
    PublicTreeError,
    corpus_digest,
    load_corpus,
    local_corpus,
    published_corpus,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A miniature ROCm/aorta checkout: real git, real remote, real tracked set."""
    root = tmp_path / "aorta"
    (root / "src" / "aorta" / "cli").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "src" / "aorta" / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    (root / "src" / "aorta" / "cli" / "chat.py").write_text(
        "def chat():\n    'the assistant entry point'\n", encoding="utf-8"
    )
    (root / "docs" / "usage.md").write_text("# Usage\n\nRun aorta sweep.\n", encoding="utf-8")
    (root / "README.md").write_text("# aorta\n", encoding="utf-8")

    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "T")
    _git(root, "remote", "add", "origin", "https://github.com/ROCm/aorta.git")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    return root


class TestPublicTreeGuard:
    def test_the_public_remote_passes(self, repo: Path):
        corpus_mod.assert_public_tree(repo)

    @pytest.mark.parametrize(
        "url",
        [
            "git@github.com:ROCm/aorta.git",
            "https://github.com/ROCm/aorta",
            "ssh://git@github.com/ROCm/aorta.git",
        ],
    )
    def test_every_remote_spelling_resolves(self, repo: Path, url: str):
        """Which spelling a runner's checkout uses is not our choice."""
        _git(repo, "remote", "set-url", "origin", url)
        corpus_mod.assert_public_tree(repo)

    def test_an_internal_remote_refuses(self, repo: Path):
        _git(repo, "remote", "set-url", "origin", "git@github.com:example-internal-org/aorta.git")
        with pytest.raises(PublicTreeError) as exc:
            corpus_mod.assert_public_tree(repo)
        message = str(exc.value)
        assert "example-internal-org/aorta" in message
        assert "verbatim" in message

    def test_a_lookalike_remote_refuses(self, repo: Path):
        """``ROCm/aorta-fork`` must not pass a prefix check."""
        _git(repo, "remote", "set-url", "origin", "https://github.com/ROCm/aorta-fork.git")
        with pytest.raises(PublicTreeError):
            corpus_mod.assert_public_tree(repo)

    @pytest.mark.parametrize(
        "url",
        [
            "https://evil.example/github.com/ROCm/aorta.git",
            "https://github.com.evil.example/ROCm/aorta.git",
            "git@evil.example:github.com/ROCm/aorta.git",
        ],
    )
    def test_a_remote_that_only_contains_the_host_refuses(self, repo: Path, url: str):
        """The host has to be the host, not a substring anywhere in the URL.

        Searching for ``github.com/`` reduced every one of these to
        ``ROCm/aorta``, so any remote at all could satisfy the guard by
        carrying the right text somewhere in its path.
        """
        _git(repo, "remote", "set-url", "origin", url)
        with pytest.raises(PublicTreeError):
            corpus_mod.assert_public_tree(repo)

    def test_a_tree_that_is_not_a_git_checkout_refuses(self, tmp_path: Path):
        """Absence of evidence is not a pass. The guard fails closed."""
        (tmp_path / "loose").mkdir()
        with pytest.raises(PublicTreeError):
            corpus_mod.assert_public_tree(tmp_path / "loose")

    def test_no_origin_remote_refuses(self, repo: Path):
        _git(repo, "remote", "remove", "origin")
        with pytest.raises(PublicTreeError):
            corpus_mod.assert_public_tree(repo)


class TestTrackedFileFilter:
    def test_only_tracked_files_are_indexed(self, repo: Path):
        """The guard a remote check cannot make.

        A correct checkout of the correct repository can still have an internal
        reproducer sitting in it -- a developer's scratch copy, a customer
        bundle unpacked for debugging. It is untracked, so it is absent from the
        corpus whatever the working directory looks like.
        """
        leaked = repo / "src" / "aorta" / "customer_repro.py"
        leaked.write_text("# NDA: acme corp allreduce hang\nSECRET = 1\n", encoding="utf-8")
        assert leaked.exists()

        sources = {doc.metadata["source"] for doc in load_corpus(published_corpus(repo))}
        assert "src/aorta/customer_repro.py" not in sources
        assert "src/aorta/cli/chat.py" in sources

    def test_a_local_build_does_index_untracked_files(self, repo: Path):
        """A local index is never published, so the filter would only hurt.

        An air-gapped user indexing their own working tree is the case Decision
        21b exists to serve.
        """
        scratch = repo / "src" / "aorta" / "scratch.py"
        scratch.write_text("# work in progress\n", encoding="utf-8")
        sources = {doc.metadata["source"] for doc in load_corpus(local_corpus(repo / "src"))}
        assert "aorta/scratch.py" in sources

    def test_the_allowlist_is_the_repos_tracked_set(self, repo: Path):
        corpus = published_corpus(repo)
        assert corpus.allowed is not None
        assert "src/aorta/cli/chat.py" in corpus.allowed
        assert "docs/usage.md" in corpus.allowed


class TestPublishedCorpus:
    def test_sources_are_repository_relative(self, repo: Path):
        """So a corpus spanning src/ and docs/ yields paths a user recognises."""
        sources = {doc.metadata["source"] for doc in load_corpus(published_corpus(repo))}
        assert sources == {
            "README.md",
            "docs/usage.md",
            "src/aorta/__init__.py",
            "src/aorta/cli/chat.py",
        }

    def test_a_single_file_subpath_is_picked_up(self, repo: Path):
        """README.md is a file, not a directory, so os.walk never reaches it."""
        sources = {doc.metadata["source"] for doc in load_corpus(published_corpus(repo))}
        assert "README.md" in sources

    def test_an_absent_subpath_is_skipped_rather_than_fatal(self, repo: Path):
        corpus = published_corpus(repo, subpaths=("src/aorta", "not-here"))
        assert corpus.subpaths == ("src/aorta",)

    def test_no_subpaths_at_all_is_fatal(self, repo: Path):
        with pytest.raises(PublicTreeError, match="none of the published corpus subpaths"):
            published_corpus(repo, subpaths=("nope",))

    def test_nesting_does_not_index_a_file_twice(self, repo: Path):
        """Two identical vectors would halve what a fixed-k retrieval reaches."""
        corpus = published_corpus(repo, subpaths=(".", "src/aorta"))
        sources = [doc.metadata["source"] for doc in load_corpus(corpus)]
        assert len(sources) == len(set(sources))

    def test_the_default_subpaths_are_code_and_prose_only(self):
        assert PUBLISHED_SUBPATHS == ("src/aorta", "docs", "README.md")


class TestCorpusDigest:
    def _docs(self, *pairs: tuple[str, str]) -> list[Document]:
        return [Document(page_content=body, metadata={"source": src}) for src, body in pairs]

    def _digest(self, docs, **overrides) -> str:
        params = {"embedding_model": "m", "chunk_size": 512, "chunk_overlap": 50}
        params.update(overrides)
        return corpus_digest(docs, **params)

    def test_the_same_corpus_digests_the_same(self):
        docs = self._docs(("a.py", "x = 1"), ("b.py", "y = 2"))
        assert self._digest(docs) == self._digest(docs)

    def test_order_does_not_matter(self):
        """Two runners may walk in different orders; the artifact is the same."""
        forward = self._docs(("a.py", "x = 1"), ("b.py", "y = 2"))
        assert self._digest(forward) == self._digest(list(reversed(forward)))

    def test_changed_content_changes_the_digest(self):
        assert self._digest(self._docs(("a.py", "x = 1"))) != self._digest(
            self._docs(("a.py", "x = 2"))
        )

    def test_a_renamed_file_changes_the_digest(self):
        assert self._digest(self._docs(("a.py", "x = 1"))) != self._digest(
            self._docs(("b.py", "x = 1"))
        )

    def test_a_new_file_changes_the_digest(self):
        assert self._digest(self._docs(("a.py", "x = 1"))) != self._digest(
            self._docs(("a.py", "x = 1"), ("b.py", "y = 2"))
        )

    @pytest.mark.parametrize(
        "override",
        [{"embedding_model": "other"}, {"chunk_size": 1024}, {"chunk_overlap": 0}],
    )
    def test_the_parameters_that_shape_vectors_are_covered(self, override):
        """An unchanged corpus with different chunking is a different artifact."""
        docs = self._docs(("a.py", "x = 1"))
        assert self._digest(docs) != self._digest(docs, **override)

    def test_it_is_not_keyed_on_the_git_sha(self, repo: Path):
        """Most commits touch nothing indexable; keying on the SHA kills the skip.

        A test-only commit must leave the digest alone, or the nightly re-uploads
        an identical index every night.
        """
        before = corpus_digest(
            load_corpus(published_corpus(repo)),
            embedding_model="m",
            chunk_size=512,
            chunk_overlap=50,
        )
        (repo / "test_extra.py").write_text("def test_x(): pass\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "test only")
        after = corpus_digest(
            load_corpus(published_corpus(repo)),
            embedding_model="m",
            chunk_size=512,
            chunk_overlap=50,
        )
        assert before == after


class TestLocalCorpus:
    def test_it_defaults_to_no_allowlist(self, repo: Path):
        assert local_corpus(repo).allowed is None

    def test_a_missing_path_is_reported(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="corpus path does not exist"):
            local_corpus(tmp_path / "absent")


class TestBuildIndex:
    """The full build, with the real store and a deterministic fake embedder."""

    def test_it_writes_an_index_and_a_manifest_that_agree(self, repo: Path, tmp_path, monkeypatch):
        from aorta.chat.rag import index_ops
        from aorta.chat.rag import manifest as manifest_mod

        _install_fake_embedder(monkeypatch)
        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        target = tmp_path / "out" / "index.sqlite"

        result = index_ops.build_index(published_corpus(repo), index_path=target)

        assert target.exists()
        assert result.chunk_count > 0
        assert result.file_count == 4
        found = manifest_mod.read_manifest(target)
        assert found.embedding_model == "fake/model"
        assert found.dimensions == 8
        assert found.chunk_count == result.chunk_count
        assert found.index_sha256 == manifest_mod.sha256_file(target)
        assert found.corpus_roots == list(PUBLISHED_SUBPATHS)

    def test_the_manifest_records_the_source_commit(self, repo: Path, tmp_path, monkeypatch):
        from aorta.chat.rag import index_ops
        from aorta.chat.rag import manifest as manifest_mod

        _install_fake_embedder(monkeypatch)
        target = tmp_path / "index.sqlite"
        index_ops.build_index(published_corpus(repo), index_path=target)
        found = manifest_mod.read_manifest(target)
        assert len(found.aorta_sha) == 40

    def test_the_built_index_validates_against_the_provider_that_built_it(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """The round trip that matters: build then load must not refuse."""
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        target = tmp_path / "index.sqlite"
        index_ops.build_index(published_corpus(repo), index_path=target)
        report = index_ops.check_index(target, strict=True)
        assert report.refusals == []

    def test_an_empty_corpus_is_reported_rather_than_silently_published(
        self, tmp_path: Path, monkeypatch
    ):
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="corpus is empty"):
            index_ops.build_index(local_corpus(empty), index_path=tmp_path / "i.sqlite")

    def test_compute_digest_matches_what_the_build_records(self, repo: Path, tmp_path, monkeypatch):
        """CI compares the two, so a drift between them would break the skip."""
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        target = tmp_path / "index.sqlite"
        result = index_ops.build_index(published_corpus(repo), index_path=target)
        digest, files = index_ops.compute_digest(published_corpus(repo))
        assert digest == result.manifest.corpus_digest
        assert files == result.file_count


class TestBuildWillNotSilentlyDowngradeAFetchedIndex:
    """A local build covers less than the published one, and used to say nothing.

    This is the mechanism behind the bad advice the guard exists to catch:
    ``index build`` defaults ``--output`` to the live index and its corpus to
    ``local_corpus``, so following a suggestion to "build" replaced an index
    covering ``src/aorta``, ``docs`` and ``README.md`` with one covering a
    single tree.
    """

    @staticmethod
    def _install_published(target: Path, monkeypatch, **overrides) -> str:
        """Put a *readable* index at ``target`` that looks like a CI-published one.

        A real sqlite store holding the collection the manifest names, not
        filler bytes. The guard asks :func:`check_index` whether the index is
        usable, and an index nothing can open is exempt from it -- so a fixture
        of filler bytes made every refusal test here pass through the
        exemption rather than through the corpus comparison it was written to
        exercise. The bytes were never load-bearing; being openable is.

        Returns the digest of the file it wrote, so a caller can assert the
        index was or was not replaced without depending on its contents.
        """
        import sqlite3

        from aorta.chat.rag import manifest as manifest_mod

        target.parent.mkdir(parents=True, exist_ok=True)
        collection = overrides.get("collection", "aorta_fake")
        chunks = overrides.get("chunk_count", 0)
        conn = sqlite3.connect(target)
        try:
            conn.execute(f'CREATE TABLE "chunks_{collection}" (id INTEGER, text TEXT)')
            conn.executemany(
                f'INSERT INTO "chunks_{collection}" VALUES (?, ?)',
                [(i, "the published index") for i in range(chunks)],
            )
            conn.commit()
        finally:
            # Closed rather than left to the context manager, which commits but
            # does not close -- and the file is about to be digested and moved.
            conn.close()
        values = {
            "aorta_version": "0.2.1",
            "aorta_sha": "a" * 40,
            "embedding_provider": "local",
            "embedding_model": "fake/model",
            "dimensions": 8,
            "collection": "aorta_fake",
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "index_sha256": manifest_mod.sha256_file(target),
            "corpus_roots": list(PUBLISHED_SUBPATHS),
        }
        values.update(overrides)
        manifest_mod.write_manifest(target, manifest_mod.Manifest(**values))
        return manifest_mod.sha256_file(target)

    def test_it_refuses_and_names_what_would_be_lost(self, repo: Path, tmp_path, monkeypatch):
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        target = tmp_path / "cache" / "index.sqlite"
        digest = self._install_published(target, monkeypatch)

        with pytest.raises(index_ops.IndexOverwriteError) as exc:
            index_ops.build_index(local_corpus(repo), index_path=target)

        message = str(exc.value)
        assert "docs/" in message and "README.md" in message
        assert "aorta chat index fetch" in message
        assert "--force" in message
        assert index_ops.manifest_mod.sha256_file(target) == digest, "the index must be untouched"

    def test_a_published_build_over_a_published_index_is_a_refresh_not_a_downgrade(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """The guard is about the corpus narrowing, not about the path being occupied.

        `--public-only` produces the same shape it would replace, so refusing
        it protects nothing -- and refusing it is how a guard keyed on the
        destination alone would break `nightly.yml` and `release.yml` the first
        time their `index-out/` was restored or re-used, which stops the
        published index updating at all.
        """
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        target = tmp_path / "cache" / "index.sqlite"
        digest = self._install_published(target, monkeypatch)

        result = index_ops.build_index(published_corpus(repo), index_path=target)

        assert result.manifest.corpus_roots == list(PUBLISHED_SUBPATHS)
        assert index_ops.manifest_mod.sha256_file(target) != digest, "the index must be replaced"

    def test_force_builds_over_it(self, repo: Path, tmp_path, monkeypatch):
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        target = tmp_path / "cache" / "index.sqlite"
        digest = self._install_published(target, monkeypatch)

        assert index_ops.build_index(local_corpus(repo), index_path=target, force=True)
        assert index_ops.manifest_mod.sha256_file(target) != digest, "the index must be replaced"

    def test_a_refused_index_is_exempt_so_the_advice_stays_followable(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """The cross-PR interaction, which neither diff shows on its own.

        ``doctor`` and the manifest refusal both name ``aorta chat index build``
        as the remedy for an embedding-identity mismatch, and that is the right
        remedy. A refused index is still an existing index at the destination,
        so a guard keyed on presence alone would compose the two into a dead
        end: refused, told to rebuild, refused again -- landing on a user who
        is already stuck.

        There is also nothing to protect. Vectors that are not comparable to
        this install's queries cannot answer anything, so rebuilding over them
        is not destructive. The guard is for a *usable* published index.
        """
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        target = tmp_path / "cache" / "index.sqlite"
        # Built by a different embedding model: exactly what `validate` refuses.
        self._install_published(target, monkeypatch, embedding_model="some/other-model")

        assert index_ops.check_index(target, strict=False).refusals, (
            "the fixture must be an index this install would actually refuse"
        )

        result = index_ops.build_index(local_corpus(repo), index_path=target)

        assert result.index_path == target
        assert index_ops.check_index(target, strict=True).refusals == []

    def test_an_index_whose_store_cannot_be_read_is_exempt_too(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """The exemption is "cannot be used", not "the manifest is refused".

        PR #463 added a FAIL branch for an index whose ``.sqlite`` cannot be
        opened, and it names ``aorta chat index build`` as the remedy. Such an
        index has an entirely *valid* manifest, so a guard asking only
        ``manifest.validate`` refused the rebuild that every reader touching
        the file was recommending -- the same two-step dead end as the refused
        case, one layer down. Asked through ``check_index``, which reads the
        store, so this guard and the load path agree on what "usable" means.
        """
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        target = tmp_path / "cache" / "index.sqlite"
        self._install_published(target, monkeypatch)
        # Leave the sidecars, replace the index with something that is not one.
        target.write_bytes(b"not a sqlite database, not even close" * 40)

        report = index_ops.check_index(target, strict=False)
        assert report.refusals, "the fixture must be an index this install cannot read"
        assert not index_ops._validate_against_provider(
            index_ops.manifest_mod.read_manifest(target)
        ).refusals, "and its manifest must still be perfectly valid, which is the point"

        assert index_ops.build_index(local_corpus(repo), index_path=target)
        assert index_ops.check_index(target, strict=True).refusals == []

    def test_a_legacy_manifest_does_not_hide_an_unreadable_store(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """``chunk_count`` of 0 must not buy an unreadable index a clean bill of health.

        ``check_index`` used to suppress its unreadable-file refusal unless the
        manifest claimed a chunk count, on the reasoning that a manifest
        predating the field asserts nothing to contradict. But "is the claim
        contradicted" is not "can this be queried", and the load path refuses
        such a file with no gate at all -- so this reader was *more permissive
        than the load path it exists to predict*.
        """
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        target = tmp_path / "cache" / "index.sqlite"
        self._install_published(target, monkeypatch, chunk_count=0)
        target.write_bytes(b"not a sqlite database, not even close" * 40)

        assert index_ops.manifest_mod.read_manifest(target).chunk_count == 0
        refusals = index_ops.check_index(target, strict=False).refusals

        assert refusals, "an unreadable store fails closed whatever the manifest claims"
        # Worded for a manifest that made no claim, rather than "describes 0 chunks".
        assert "could not be read as a sqlite store" in " ".join(refusals)
        assert "describes 0 chunks" not in " ".join(refusals)
        # And the guard follows it, so #463's advice stays followable here too.
        assert index_ops.build_index(local_corpus(repo), index_path=target)

    def test_a_stale_index_on_a_remote_provider_needs_no_new_exemption(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """The third state #463 names, which the guard already permitted.

        Source drift is a *warning*, so it never reaches the usability
        exemption -- and it does not need to. On a remote embedding provider
        ``doctor`` names ``index build`` for drift because a fetch would land
        the published asset, which that provider refuses. The index in front of
        the guard is therefore already exempt for a different reason: it is a
        local build, classified ``local``, and a local index is never protected
        from ``build``. Pinned so a later widening of the exemption cannot be
        justified by this case.
        """
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        target = tmp_path / "cache" / "index.sqlite"
        # A local build: one absolute root, which is the local signature.
        self._install_published(target, monkeypatch, corpus_roots=[str(tmp_path / "checkout")])

        local = index_ops.manifest_mod.read_manifest(target)
        assert index_ops.index_provenance(local) == index_ops.PROVENANCE_LOCAL
        assert not index_ops.check_index(target, strict=False).refusals, (
            "the fixture must be healthy, so the exemption cannot be what permits it"
        )

        assert index_ops.build_index(local_corpus(repo), index_path=target)

    def test_a_healthy_published_index_is_still_refused(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """The control for the three exemptions above: the guard still guards.

        Widening "refused" to "unusable" must not make the guard permissive on
        the case it exists for -- a healthy published index that a bare build
        would narrow.
        """
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        target = tmp_path / "cache" / "index.sqlite"
        self._install_published(target, monkeypatch)

        assert not index_ops.check_index(target, strict=False).refusals
        with pytest.raises(index_ops.IndexOverwriteError):
            index_ops.build_index(local_corpus(repo), index_path=target)

    def test_a_refused_index_stays_exempt_even_when_its_provenance_is_unreadable(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """The exemption is checked *before* the unclassifiable refusal, on purpose.

        It looks like a hole in the "unreadable provenance is refused" rule and
        it is not. An unreadable ``corpus_roots`` means the index is either a
        published one or a local one, and a *refused* index reaches the same
        verdict down both branches: a refused published index is exempt by the
        rule above, and a local index is never protected from ``build`` at all.
        So proceeding is not a guess about which is on disk -- it is what both
        possibilities agree on.

        Reordering the two would also break the cross-PR interaction the
        exemption exists for: ``doctor`` and the manifest refusal both name
        ``aorta chat index build`` as the remedy, and a user whose sidecar is
        *also* hand-edited would be refused, told to rebuild, refused again.
        """
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        target = tmp_path / "cache" / "index.sqlite"
        self._install_published(target, monkeypatch, embedding_model="some/other-model")
        path = index_ops.manifest_mod.manifest_path(target)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["corpus_roots"] = "src/aorta"
        path.write_text(json.dumps(raw), encoding="utf-8")

        assert index_ops.index_provenance(index_ops.manifest_mod.read_manifest(target)) == (
            index_ops.PROVENANCE_INVALID
        ), "the fixture must be unclassifiable as well as refused"
        assert index_ops.check_index(target, strict=False).refusals

        result = index_ops.build_index(local_corpus(repo), index_path=target)

        assert result.index_path == target
        assert index_ops.check_index(target, strict=True).refusals == []

    def test_an_unclassifiable_manifest_is_refused_rather_than_guessed(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """The guard reads ``corpus_roots``, which nothing type-checks.

        A sidecar recording it as the string ``"src/aorta"`` classified as a
        *local* build -- because iterating the string yields ``"/"``, and
        ``Path("/").is_absolute()`` is true -- so this guard returned early and
        the published index was overwritten anyway. The mirror spelling
        ``"docs"`` classified as published. Both are answers from nothing.
        """
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        target = tmp_path / "cache" / "index.sqlite"
        digest = self._install_published(target, monkeypatch)
        path = index_ops.manifest_mod.manifest_path(target)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["corpus_roots"] = "src/aorta"
        path.write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(index_ops.IndexOverwriteError) as exc:
            index_ops.build_index(local_corpus(repo), index_path=target)

        assert "cannot classify" in str(exc.value)
        assert "corpus_roots is a str" in str(exc.value)
        assert index_ops.manifest_mod.sha256_file(target) == digest, "the index must be untouched"
        assert index_ops.build_index(local_corpus(repo), index_path=target, force=True)

    def test_the_refusal_does_not_claim_a_loss_that_did_not_happen(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """``build --path <a checkout>`` does cover ``docs/`` and ``README.md``.

        The message asserted "no 'docs/' or 'README.md' coverage"
        unconditionally, which is true of the default corpus (the installed
        package alone) and false for any build pointed at a full checkout --
        an error message stating a fact the invocation disproves.
        """
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        target = tmp_path / "cache" / "index.sqlite"
        self._install_published(target, monkeypatch)

        corpus = local_corpus(repo)
        covered = {
            str(document.metadata.get("source", "")) for document in corpus_mod.load_corpus(corpus)
        }
        assert any("README" in source for source in covered), "fixture must cover README.md"

        with pytest.raises(index_ops.IndexOverwriteError) as exc:
            index_ops.build_index(corpus, index_path=target)

        message = str(exc.value)
        assert "no 'docs/' or 'README.md' coverage" not in message
        # What is always true of a corpus reaching this branch, and both sides.
        assert "no public-tree provenance" in message
        assert str(repo) in message

    def test_a_mistyped_output_over_someone_elses_file_is_refused(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """The gap under the guard rather than in it: no manifest, no opinion.

        Both guards used to open with "read the sidecar; if there is not one,
        return", which made a path that exists with nothing beside it
        indistinguishable from a path that does not exist -- so
        ``--output ~/notes.txt`` reached ``replace()`` unopposed while the
        strictly better-informed case of a sidecar that reads but classifies
        badly was refused.
        """
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        target = tmp_path / "notes.txt"
        target.write_text("a year of notes", encoding="utf-8")

        with pytest.raises(index_ops.IndexOverwriteError) as exc:
            index_ops.build_index(local_corpus(repo), index_path=target)

        message = str(exc.value)
        assert "no manifest beside it" in message
        assert "typo" in message
        assert "--force" in message
        assert target.read_text(encoding="utf-8") == "a year of notes", "the file must survive"

    def test_force_builds_over_a_manifest_less_destination(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """The escape the refusal names has to work, or it is not an escape."""
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        target = tmp_path / "notes.txt"
        target.write_text("a year of notes", encoding="utf-8")

        assert index_ops.build_index(local_corpus(repo), index_path=target, force=True)
        assert index_ops.check_index(target, strict=True).refusals == []

    def test_a_published_build_over_a_manifest_less_destination_still_proceeds(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """``nightly.yml`` and ``release.yml`` run the same command repeatedly.

        Their second run lands on an ``index-out/`` that a restored cache, or
        the previous run in the same job, has already populated -- and an
        interrupted first run leaves the index there with no sidecar. The
        ``--public-only`` exemption is ahead of this refusal for that reason,
        and a guard that tripped here would stop the published index updating
        at all.
        """
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        target = tmp_path / "index-out" / "aorta-chat-index.sqlite"
        target.parent.mkdir()
        target.write_bytes(b"an interrupted first run left this behind")

        assert index_ops.build_index(published_corpus(repo), index_path=target)

    def test_the_remedy_names_the_index_that_was_refused(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """A remedy that acts on a different index is worse than none at all.

        Both lines default ``--output`` to the cache and their corpus to the
        installed package, so a refusal raised over an explicit ``--path`` and
        ``--output`` printed a bare ``aorta chat index build --force`` -- which
        rebuilds the *user's real* index, over a corpus they did not name, and
        leaves the one they were refused exactly as it was. Pasting the advice
        did damage and did not resolve the refusal.
        """
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        monkeypatch.setattr(settings, "index_path", str(tmp_path / "cache" / "default.sqlite"))
        target = tmp_path / "elsewhere" / "index.sqlite"
        self._install_published(target, monkeypatch)

        with pytest.raises(index_ops.IndexOverwriteError) as exc:
            index_ops.build_index(local_corpus(repo), index_path=target)

        message = str(exc.value)
        for line in message.splitlines():
            if not line.strip().startswith("aorta chat index"):
                continue
            assert str(target) in line, f"remedy targets the wrong index: {line!r}"
        assert f"--output {shlex.quote(str(target))}" in message
        assert f"--path {shlex.quote(str(repo))}" in message
        assert "aorta chat index build --force\n" not in message

    def test_the_remedy_stays_short_when_the_defaults_are_what_ran(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """The flags are carried because they differ, not as decoration.

        A refusal over the configured cache is the common one, and spelling
        out the two flags a bare command already resolves to would make every
        such message longer for no reader.
        """
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        target = tmp_path / "cache" / "index.sqlite"
        monkeypatch.setattr(settings, "index_path", str(target))
        monkeypatch.setattr(settings, "aorta_path", str(repo))
        self._install_published(target, monkeypatch)

        with pytest.raises(index_ops.IndexOverwriteError) as exc:
            index_ops.build_index(local_corpus(repo), index_path=target)

        message = str(exc.value)
        assert "--output" not in message
        assert "--path" not in message
        assert "aorta chat index build --force" in message

    def test_a_local_build_over_a_local_build_needs_nothing(
        self, repo: Path, tmp_path, monkeypatch
    ):
        """Rebuilding your own index is the ordinary developer loop."""
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        target = tmp_path / "index.sqlite"
        index_ops.build_index(local_corpus(repo), index_path=target)

        assert index_ops.build_index(local_corpus(repo), index_path=target)

    def test_a_fresh_destination_needs_nothing(self, repo: Path, tmp_path, monkeypatch):
        """What CI does: `--output` into a directory it just created.

        A guard that tripped here would stop the published index updating at
        all, which is worse than the defect it is guarding against.
        """
        from aorta.chat.rag import index_ops

        _install_fake_embedder(monkeypatch)
        out = tmp_path / "index-out"
        out.mkdir()

        result = index_ops.build_index(
            published_corpus(repo), index_path=out / "aorta-chat-index.sqlite"
        )

        assert result.index_path.exists()


def _install_fake_embedder(monkeypatch) -> None:
    """Replace the provider with a deterministic 8-dimension bag-of-words model.

    Real embeddings would mean a 65 MB download and a minute of CPU per test;
    what these tests are about is the plumbing around the vectors.
    """
    from langchain_core.embeddings import Embeddings

    from aorta.chat.rag import index_ops

    class _Fake(Embeddings):
        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text):
            vector = [0.0] * 8
            for token in text.split():
                vector[hash(token) % 8] += 1.0
            norm = sum(value * value for value in vector) ** 0.5
            return [value / norm for value in vector] if norm else [1.0] + [0.0] * 7

    class _Provider:
        name = "local"

        def get_embeddings(self):
            return _Fake()

        def collection_name(self):
            return "aorta_fake"

        def model_id(self):
            # Read from settings, as the real local provider does, so the
            # manifest assertions still see the model the test configured.
            from aorta.chat.config import settings

            return settings.embedding_model

        def vector_identity(self):
            from aorta.chat.config import settings

            return settings.embedding_model

        def describe(self):
            return "fake 8d embeddings"

    monkeypatch.setattr(index_ops, "get_provider", _Provider)
