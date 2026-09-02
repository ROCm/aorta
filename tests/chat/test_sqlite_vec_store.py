"""Round-trips through the sqlite-vec store that replaced Chroma.

Everything here runs against a real ``.sqlite`` file in ``tmp_path`` with a fake
deterministic embedder, so the tests exercise the actual vec0 tables and the
actual SQL without downloading a model or reaching the network. The embedder is
a bag of words over a fixed vocabulary: which document is nearest to which query
is then something the test can state, rather than something it has to trust.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from aorta.chat.rag import indexer as indexer_module
from aorta.chat.rag import retriever as retriever_module
from aorta.chat.rag.indexer import index_codebase
from aorta.chat.rag.retriever import SqliteVecStore, get_retriever, reset_caches, search_docs

VOCABULARY = ["gpu", "kernel", "sqlite", "vector", "config", "timeout"]

DOCUMENTS = [
    Document(
        page_content="the gpu kernel launches once per dispatch",
        metadata={"source": "src/launch.py", "start_line": 1, "end_line": 4},
    ),
    Document(
        page_content="sqlite vector search over the index file",
        metadata={"source": "src/store.py", "start_line": 10, "end_line": 20},
    ),
    Document(
        page_content="config timeout defaults to sixty seconds",
        metadata={"source": "src/config.py", "start_line": 7, "end_line": 7},
    ),
    Document(
        page_content="a gpu kernel and a second gpu kernel, nearly the same text",
        metadata={"source": "src/launch_again.py", "start_line": 1, "end_line": 2},
    ),
]


class BagOfWordsEmbeddings(Embeddings):
    """Counts vocabulary hits. No model, no network, and stable across runs."""

    def __init__(self, vocabulary: list[str] | None = None) -> None:
        self.vocabulary = vocabulary if vocabulary is not None else VOCABULARY

    def _embed(self, text: str) -> list[float]:
        words = re.findall(r"[a-z]+", text.lower())
        return [float(words.count(term)) for term in self.vocabulary]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _build(path: Path, collection: str = "aorta", documents=None) -> SqliteVecStore:
    store = SqliteVecStore(path=path, embedding=BagOfWordsEmbeddings(), collection=collection)
    store.reset()
    store.add_documents(list(DOCUMENTS if documents is None else documents))
    return store


@pytest.fixture()
def index_file(tmp_path: Path) -> Path:
    return tmp_path / "index.sqlite"


@pytest.fixture(autouse=True)
def clear_retriever_caches():
    reset_caches()
    yield
    reset_caches()


class TestRoundTrip:
    def test_the_index_is_one_file(self, index_file: Path):
        store = _build(index_file)
        store.close()
        assert index_file.is_file()

    def test_the_nearest_document_comes_back_first(self, index_file: Path):
        store = _build(index_file)
        results = store.similarity_search("gpu kernel", k=1)
        assert results[0].metadata["source"] == "src/launch.py"

    def test_a_different_query_reaches_a_different_document(self, index_file: Path):
        store = _build(index_file)
        results = store.similarity_search("config timeout", k=1)
        assert results[0].metadata["source"] == "src/config.py"

    def test_metadata_survives_the_round_trip(self, index_file: Path):
        store = _build(index_file)
        found = store.similarity_search("sqlite vector", k=1)[0]
        assert found.metadata == {
            "source": "src/store.py",
            "start_line": 10,
            "end_line": 20,
        }
        assert found.page_content == "sqlite vector search over the index file"

    def test_the_store_reopens_from_disk(self, index_file: Path):
        _build(index_file).close()
        reopened = SqliteVecStore(
            path=index_file, embedding=BagOfWordsEmbeddings(), collection="aorta"
        )
        assert reopened.collection_exists()
        assert reopened.similarity_search("gpu kernel", k=1)[0].metadata["source"] == (
            "src/launch.py"
        )

    def test_reindexing_replaces_rather_than_appends(self, index_file: Path):
        """A rebuild that stacked a second copy would halve a fixed-k retrieval."""
        _build(index_file).close()
        store = _build(index_file)
        sources = [doc.metadata["source"] for doc in store.similarity_search("gpu", k=99)]
        assert len(sources) == len(DOCUMENTS)
        assert len(set(sources)) == len(DOCUMENTS)


class TestCollectionPartitioning:
    def test_two_providers_coexist_in_one_file(self, index_file: Path):
        """Different dimensions, same file: the collection is in the table name."""
        _build(index_file, collection="aorta").close()
        other = SqliteVecStore(
            path=index_file,
            embedding=BagOfWordsEmbeddings(["gpu", "sqlite"]),
            collection="aorta_remote_tiny",
        )
        other.add_documents(list(DOCUMENTS))
        assert other.dimension() == 2
        assert sorted(other.collection_names()) == ["aorta", "aorta_remote_tiny"]
        other.close()

        original = SqliteVecStore(
            path=index_file, embedding=BagOfWordsEmbeddings(), collection="aorta"
        )
        assert original.dimension() == len(VOCABULARY)
        assert original.similarity_search("gpu kernel", k=1)[0].metadata["source"] == (
            "src/launch.py"
        )

    def test_an_unknown_collection_is_not_reported_as_present(self, index_file: Path):
        _build(index_file, collection="aorta").close()
        absent = SqliteVecStore(
            path=index_file, embedding=BagOfWordsEmbeddings(), collection="absent"
        )
        assert not absent.collection_exists()

    def test_a_collection_name_that_is_not_an_identifier_is_refused(self, index_file: Path):
        """It is interpolated into a table name, so it cannot be bound."""
        with pytest.raises(ValueError, match="identifier"):
            SqliteVecStore(
                path=index_file,
                embedding=BagOfWordsEmbeddings(),
                collection='aorta"; DROP TABLE chunks_aorta; --',
            )


class TestDimensionMismatch:
    def test_querying_with_the_wrong_width_raises_rather_than_guesses(self, index_file: Path):
        """The failure a provider switch produces, which used to be silent noise."""
        _build(index_file).close()
        wrong = SqliteVecStore(
            path=index_file,
            embedding=BagOfWordsEmbeddings(["gpu", "sqlite"]),
            collection="aorta",
        )
        with pytest.raises(ValueError) as excinfo:
            wrong.similarity_search("gpu kernel", k=1)
        message = str(excinfo.value)
        assert "2-dimension" in message
        assert "6 dimensions" in message
        assert "AORTA_CHAT_EMBEDDING_PROVIDER" in message

    def test_a_collection_that_was_never_written_says_so(self, index_file: Path):
        _build(index_file, collection="aorta").close()
        absent = SqliteVecStore(
            path=index_file, embedding=BagOfWordsEmbeddings(), collection="absent"
        )
        with pytest.raises(ValueError, match="not present"):
            absent.dimension()


class TestMaximalMarginalRelevance:
    def test_it_returns_exactly_k_results(self, index_file: Path):
        store = _build(index_file)
        results = store.max_marginal_relevance_search("gpu kernel", k=3, fetch_k=4, lambda_mult=0.7)
        assert len(results) == 3

    def test_it_cannot_return_a_document_twice(self, index_file: Path):
        store = _build(index_file)
        results = store.max_marginal_relevance_search("gpu kernel", k=4, fetch_k=4, lambda_mult=0.7)
        assert len({doc.metadata["source"] for doc in results}) == 4

    def test_it_still_leads_with_the_most_relevant_document(self, index_file: Path):
        store = _build(index_file)
        results = store.max_marginal_relevance_search("gpu kernel", k=3, fetch_k=4, lambda_mult=0.7)
        assert results[0].metadata["source"] == "src/launch.py"

    def test_pure_relevance_keeps_the_near_duplicate(self, index_file: Path):
        """launch_again.py embeds to the same direction as launch.py."""
        store = _build(index_file)
        results = store.max_marginal_relevance_search("gpu kernel", k=2, fetch_k=4, lambda_mult=1.0)
        assert [doc.metadata["source"] for doc in results] == [
            "src/launch.py",
            "src/launch_again.py",
        ]

    def test_a_low_lambda_mult_drops_the_near_duplicate_for_a_new_topic(self, index_file: Path):
        """Which is the whole reason retrieval asks for MMR rather than KNN."""
        store = _build(index_file)
        results = store.max_marginal_relevance_search("gpu kernel", k=2, fetch_k=4, lambda_mult=0.1)
        sources = [doc.metadata["source"] for doc in results]
        assert sources[0] == "src/launch.py"
        # The two unrelated documents are equidistant, so which of them fills the
        # second slot is arbitrary; that the near-duplicate lost it is the point.
        assert "src/launch_again.py" not in sources

    def test_it_cannot_exceed_what_the_corpus_holds(self, index_file: Path):
        store = _build(index_file)
        results = store.max_marginal_relevance_search(
            "gpu kernel", k=99, fetch_k=99, lambda_mult=0.7
        )
        assert len(results) == len(DOCUMENTS)


def _fake_provider() -> SimpleNamespace:
    return SimpleNamespace(
        get_embeddings=BagOfWordsEmbeddings,
        collection_name=lambda: "aorta",
        describe=lambda: "bag-of-words test embeddings",
    )


def _patch_settings(monkeypatch, index_file: Path) -> None:
    monkeypatch.setattr(
        retriever_module,
        "settings",
        SimpleNamespace(index_file=index_file, retriever_k=3, retriever_fetch_k=4),
    )
    monkeypatch.setattr(retriever_module, "get_provider", _fake_provider)


class TestModuleLevelApi:
    def test_get_retriever_answers_invoke_with_documents(self, monkeypatch, index_file: Path):
        """The contract graph/nodes.py depends on, and which must not have moved."""
        _build(index_file).close()
        _patch_settings(monkeypatch, index_file)

        documents = get_retriever().invoke("gpu kernel")
        assert len(documents) == 3
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].metadata["source"] == "src/launch.py"

    def test_search_docs_honours_the_k_it_is_given(self, monkeypatch, index_file: Path):
        """tools/search.py picks k per call, unlike the fixed-k retriever."""
        _build(index_file).close()
        _patch_settings(monkeypatch, index_file)

        assert len(search_docs("gpu kernel", k=2)) == 2

    def test_a_missing_index_file_names_how_to_build_one(self, monkeypatch, index_file: Path):
        """graph/nodes.py catches FileNotFoundError to say the index is missing."""
        _patch_settings(monkeypatch, index_file)

        with pytest.raises(FileNotFoundError) as excinfo:
            get_retriever()
        message = str(excinfo.value)
        assert str(index_file) in message
        assert "aorta chat index fetch" in message
        assert "aorta chat index build" in message

    def test_a_file_without_this_collection_names_the_provider(self, monkeypatch, index_file: Path):
        """The diagnosable half of switching EMBEDDING_PROVIDER without re-indexing."""
        _build(index_file, collection="aorta_remote_something").close()
        _patch_settings(monkeypatch, index_file)

        with pytest.raises(FileNotFoundError) as excinfo:
            get_retriever()
        message = str(excinfo.value)
        assert "'aorta'" in message
        assert "aorta_remote_something" in message
        assert "bag-of-words test embeddings" in message


class TestIndexCodebase:
    def test_a_walked_tree_becomes_a_retrievable_index(self, monkeypatch, tmp_path: Path):
        """The whole Phase 2b path: walk, split, embed, write, read back."""
        codebase = tmp_path / "repo"
        (codebase / "src").mkdir(parents=True)
        (codebase / "src" / "launch.py").write_text(
            "# the gpu kernel launches once per dispatch\n", encoding="utf-8"
        )
        (codebase / "src" / "config.py").write_text(
            "# config timeout defaults to sixty seconds\n", encoding="utf-8"
        )

        index_file = tmp_path / "cache" / "index.sqlite"
        monkeypatch.setattr(
            indexer_module,
            "settings",
            SimpleNamespace(index_file=index_file, chunk_size=512, chunk_overlap=50),
        )
        monkeypatch.setattr(indexer_module, "get_provider", _fake_provider)

        store = index_codebase(codebase)
        store.close()

        # The parent directory is created for us; nothing pre-made it here.
        assert index_file.is_file()

        _patch_settings(monkeypatch, index_file)
        documents = get_retriever().invoke("config timeout")
        assert documents[0].metadata["source"] == "src/config.py"
