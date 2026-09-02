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

        def describe(self):
            return "fake 8d embeddings"

    monkeypatch.setattr(index_ops, "get_provider", _Provider)
