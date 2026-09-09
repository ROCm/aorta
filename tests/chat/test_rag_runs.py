"""The run-artifact RAG collection (Decision 11b).

Runs against a real ``.sqlite`` file with a fake deterministic embedder, so the
vec0 tables, the registry table and the actual SQL are exercised without a model
download or a network call.

What makes this collection worth its own module, and its own tests, is that it
must be **separate from the source collection in every way that matters**:
rebuilding one leaves the other intact, the two never share a table, and a
provider switch is a named error rather than a dimension crash from inside the
extension. It is also per-user data that must never be published, so the naming
is what an index-publishing job will have to select on.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings

from aorta.chat.config import configure, reset_settings, settings
from aorta.chat.rag import runs as runs_rag
from aorta.chat.rag.embeddings.base import MAX_COLLECTION_NAME, identity_digest
from aorta.chat.rag.retriever import SqliteVecStore

VOCABULARY = ["nan", "loss", "rocm", "tf32", "hang", "memory", "triton"]


class BagOfWordsEmbeddings(Embeddings):
    """Counts vocabulary hits. No model, no network, stable across runs."""

    def _embed(self, text: str) -> list[float]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return [float(words.count(term)) for term in VOCABULARY]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeProvider:
    """Stands in for the embedding provider, which owns the collection name."""

    name = "local"

    def __init__(self, collection: str = "aorta") -> None:
        self._collection = collection

    def get_embeddings(self) -> Embeddings:
        return BagOfWordsEmbeddings()

    def collection_name(self) -> str:
        return self._collection

    def model_id(self) -> str:
        return "fake/model"

    def vector_identity(self) -> str:
        return "fake/model"

    def describe(self) -> str:
        return f"fake provider ({self._collection})"


def _matrix(cells: list[dict]) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "workload": "hrx",
            "ticket": "AORTA-99",
            "run_timestamp": "2026-09-01T00:00:00Z",
            "steps_per_trial": 5,
            "trials_per_cell": 4,
            "cells": cells,
        }
    )


def _cell(name: str, hint: str, rate: float = 0.75) -> dict:
    return {
        "name": name,
        "mitigations": [],
        "trials": 4,
        "passed_count": 1,
        "failed_count": 3,
        "failure_rate": rate,
        "failure_hints": [[hint, 3]],
        "exit_status_counts": {"workload_failed": 3},
        "error": None,
    }


@pytest.fixture()
def wired(tmp_path: Path, monkeypatch):
    """A run root plus an index file, with the fake provider installed."""
    root = tmp_path / "runs"
    nan_run = root / "run_nan"
    nan_run.mkdir(parents=True)
    (nan_run / "matrix.json").write_text(_matrix([_cell("none-none", "nan detected in loss")]))
    (nan_run / "env.json").write_text(
        json.dumps(
            {
                "schema_version": "1.17",
                "captured_at": "2026-09-01T00:00:00Z",
                "rocm": {"version": "7.0.1"},
                "partial": True,
                "partial_reasons": ["triton: not importable"],
            }
        )
    )
    hang_run = root / "run_hang"
    hang_run.mkdir(parents=True)
    (hang_run / "matrix.json").write_text(_matrix([_cell("tf32_off-none", "rccl hang detected")]))

    index = tmp_path / "index.sqlite"
    reset_settings()
    configure(runs_path=str(root), index_path=str(index))

    provider = FakeProvider()
    monkeypatch.setattr(runs_rag, "get_provider", lambda: provider)
    runs_rag.reset_caches()
    yield SimpleNamespace(root=root, index=index, provider=provider)
    runs_rag.reset_caches()
    reset_settings()


class TestTheCliLifecycle:
    """``aorta chat index runs`` is the only route to this collection.

    Before it existed, ``index_run_artifacts()`` had no caller outside the
    ``python -c`` snippet in its own missing-collection message -- so the
    collection that ``search_run_artifacts`` and the tool prompts advertise
    could not be built by any documented command.
    """

    def test_it_builds_the_collection_and_reports_the_count(self, wired):
        from click.testing import CliRunner

        from aorta.chat.rag import retriever
        from aorta.cli.chat import chat

        # The CLI resolves the provider through the retriever's factory too.
        result = CliRunner().invoke(chat, ["index", "runs", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload.get("collection") == "aorta_runs"
        assert payload.get("chunks", 0) > 0
        assert retriever.collection_chunk_count(wired.index, "aorta_runs") == payload["chunks"]

    def test_it_leaves_the_source_collection_alone(self, wired):
        """The independent cadence is the entire reason the two are separate."""
        from click.testing import CliRunner

        from aorta.chat.rag import retriever
        from aorta.cli.chat import chat

        source = SqliteVecStore(
            path=wired.index,
            embedding=wired.provider.get_embeddings(),
            collection=wired.provider.collection_name(),
        )
        try:
            source.reset()
            source.add_texts(["rocm source text"], [{"source": "a.py"}])
        finally:
            source.close()

        assert CliRunner().invoke(chat, ["index", "runs"]).exit_code == 0
        assert retriever.collection_chunk_count(wired.index, "aorta") == 1

    def test_an_empty_run_root_says_so_rather_than_reporting_success(
        self, wired, tmp_path: Path
    ):
        from click.testing import CliRunner

        from aorta.cli.chat import chat

        empty = tmp_path / "no-runs"
        empty.mkdir()
        result = CliRunner().invoke(chat, ["index", "runs", "--path", str(empty)])
        assert result.exit_code == 0, result.output
        assert "no run artifacts were found" in result.output

    def test_a_missing_run_root_is_a_sentence_not_a_traceback(self, wired, tmp_path: Path):
        from click.testing import CliRunner

        from aorta.cli.chat import chat

        result = CliRunner().invoke(chat, ["index", "runs", "--path", str(tmp_path / "absent")])
        assert result.exit_code != 0
        assert "Run root does not exist" in result.output

    def test_the_missing_collection_message_names_the_command(self):
        """It used to hand the user a ``python -c`` snippet instead."""
        assert "aorta chat index runs" in runs_rag._MISSING_COLLECTION


class TestCollectionNaming:
    def test_the_run_collection_is_distinct_from_the_source_one(self, wired):
        """Same file, different tables: vec0 fixes a table's dimension at CREATE."""
        assert runs_rag.run_collection_name() == "aorta_runs"
        assert runs_rag.run_collection_name() != wired.provider.collection_name()

    def test_the_name_stays_a_bare_identifier_under_a_remote_provider(self, monkeypatch):
        """``SqliteVecStore`` interpolates the collection into table names."""
        monkeypatch.setattr(
            runs_rag, "get_provider", lambda: FakeProvider("aorta_remote_text_embedding_3_small")
        )
        name = runs_rag.run_collection_name()
        assert re.fullmatch(r"[A-Za-z0-9_]+", name)
        assert name.endswith(runs_rag.RUN_COLLECTION_SUFFIX)


class TestTheRunCollectionSeparatesModelsByName:
    """The name is the *only* thing keeping two models' run vectors apart.

    Source retrieval has a second line of defence: the manifest sidecar records
    the embedding model and refuses a mismatch before the first query. This
    collection has no sidecar. ``_get_store`` opens ``run_collection_name()``
    and queries whatever is under it, comparing nothing -- so two models that
    landed on one name and happen to share a dimension read each other's
    vectors and answer normally.

    Every other test in this file that reads a collection name substitutes
    ``FakeProvider``, which hard-codes its own, so none of them touch the real
    naming and a change to it could not fail them. These use the actual
    factory: naming never builds an embedding model, so there is no download.
    """

    @pytest.fixture(autouse=True)
    def _remote_provider(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "remote")
        monkeypatch.setattr(settings, "remote_embedding_base_url", "")

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            # Slug identically: the punctuation is folded to the same run of _.
            ("foo/bar", "foo-bar"),
            # Slug identically for the first 63 characters.
            ("m" * 70 + "-alpha", "m" * 70 + "-beta"),
        ],
    )
    def test_models_that_slug_alike_get_different_run_collections(
        self, monkeypatch, left: str, right: str
    ):
        monkeypatch.setattr(settings, "remote_embedding_model", left)
        at_left = runs_rag.run_collection_name()
        monkeypatch.setattr(settings, "remote_embedding_model", right)

        assert runs_rag.run_collection_name() != at_left

    def test_switching_endpoint_switches_the_run_collection_too(self, monkeypatch):
        """The endpoint is half a remote identity, and run data is per-user.

        A model name means nothing without the API serving it, so pointing the
        same model at a second gateway has to move this collection as well --
        otherwise the previous gateway's run vectors answer the new one's
        queries.
        """
        monkeypatch.setattr(settings, "remote_embedding_model", "text-embedding-3-small")
        monkeypatch.setattr(settings, "remote_embedding_base_url", "https://a.example/v1")
        at_a = runs_rag.run_collection_name()
        monkeypatch.setattr(settings, "remote_embedding_base_url", "https://b.example/v1")

        assert runs_rag.run_collection_name() != at_a

    def test_it_is_still_a_bare_identifier_after_the_suffix(self, monkeypatch):
        """The suffix lands after the digest, on a name already at the cap.

        Shape only. Whether the result is *within* the cap is a separate
        question, and the answer is currently no -- see the xfail below.
        """
        monkeypatch.setattr(settings, "remote_embedding_model", "q" * 200)
        name = runs_rag.run_collection_name()

        assert re.fullmatch(r"[A-Za-z0-9_]+", name)
        assert name.endswith(runs_rag.RUN_COLLECTION_SUFFIX)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "#476: run_collection_name() appends _runs to a name "
            "build_collection_name() has already filled to MAX_COLLECTION_NAME, "
            "so the composed name runs up to 68 characters against a cap of 63"
        ),
    )
    @pytest.mark.parametrize(
        "model",
        [
            # Ordinary configuration, not a synthetic edge: slugs to 38
            # characters, which is over the 36 the remote prefix leaves once the
            # digest and ``_runs`` are both accounted for. Measures 65.
            "sentence-transformers/all-MiniLM-L6-v2",
            # The extreme end, where the slug is truncated to the cap first.
            # Measures 68.
            "q" * 200,
        ],
    )
    def test_the_suffix_does_not_push_the_name_over_the_cap(self, monkeypatch, model: str):
        """The cap is the whole point of reserving room for the digest.

        ``build_collection_name`` reserves for its own suffix and returns a name
        at exactly :data:`MAX_COLLECTION_NAME`; ``run_collection_name`` then
        concatenates ``_runs`` onto it, and nothing re-checks. Each function
        keeps its own contract and the composition breaks it.

        Marked ``xfail(strict=True)`` rather than deleted or inverted: asserting
        the current 68-character result would pin a defect as a contract, and
        deleting it would lose the only statement of what the cap means for this
        collection. Strict so that fixing #476 turns this into a failure that
        has to be acknowledged, instead of a silent xpass.

        This is only half the contract. The other half -- that the fix must not
        buy those characters back out of the digest -- is
        ``test_the_whole_digest_reaches_the_run_name`` below, which is kept
        outside this ``xfail`` so it cannot be masked by it.
        """
        monkeypatch.setattr(settings, "remote_embedding_model", model)
        name = runs_rag.run_collection_name()

        assert len(name) <= MAX_COLLECTION_NAME, (
            f"{len(name)} characters, cap is {MAX_COLLECTION_NAME}: {name}"
        )

    @pytest.mark.parametrize(
        "model",
        [
            # Short: nothing is under truncation pressure, so this leg fails
            # only if the digest stops being appended at all.
            "text-embedding-3-small",
            # Past the threshold, where fixing #476 has to take characters from
            # somewhere. These are the legs that say where it must not take them.
            "sentence-transformers/all-MiniLM-L6-v2",
            "q" * 200,
        ],
    )
    def test_the_whole_digest_reaches_the_run_name(self, monkeypatch, model: str):
        """The cap and the complete digest have to hold *at once*.

        ``test_the_digest_survives_the_cap_under_every_shipped_prefix`` states
        that pair for the collection ``build_collection_name`` returns. Only
        half of it was ever carried through to the composed run name, and the
        half left off is the load-bearing one: the shortest way to satisfy the
        cap is to cut characters off the end, and the end is the digest.

        A #476 fix along the lines of ``base[:58] + "_runs"`` measures 63, so it
        would clear the cap while keeping three of the eight digest characters.
        What #422 shipped is collision *resistance*, not injectivity -- eight
        hex characters are 32 bits, and #447 is open precisely because that is
        searchable. Truncating to three leaves 12 bits: 4096 buckets, with the
        birthday point around 80 distinct model ids, which a model zoo reaches
        by accident rather than by search. So the cost of taking the characters
        from here is not a new class of bug, it is #447 made ordinary.

        **Deliberately not folded into the xfail above.** ``xfail`` swallows a
        failure, so a digest-truncating fix would clear the cap, leave that test
        still failing, and be reported as an expected failure -- exactly the
        silence this is meant to break. Kept separate, it passes today and turns
        red the moment the tail is shortened, whether or not the ``xfail``
        marker has been removed yet.
        """
        monkeypatch.setattr(settings, "remote_embedding_model", model)
        digest = identity_digest(runs_rag.get_provider().vector_identity())
        name = runs_rag.run_collection_name()

        tail = f"_{digest}{runs_rag.RUN_COLLECTION_SUFFIX}"
        assert name.endswith(tail), f"expected the name to end in {tail!r}, got {name!r}"


class TestIndexing:
    def test_a_matrix_yields_a_run_document_plus_one_per_cell(self, wired):
        docs = runs_rag.collect_run_documents(wired.root)
        scopes = [d.metadata.get("scope") for d in docs if d.metadata["artifact_kind"] == "matrix"]
        # Two runs, each with one cell: two run-level docs and two cell docs.
        assert sorted(scopes) == ["cell", "cell", "run", "run"]

    def test_every_cell_chunk_carries_its_run_header(self, wired):
        """Retrieved alone, "failed 3 of 4" without the workload is not an answer."""
        docs = runs_rag.collect_run_documents(wired.root)
        cells = [d for d in docs if d.metadata.get("scope") == "cell"]
        assert cells
        for doc in cells:
            assert "workload: hrx" in doc.page_content
            assert "cell:" in doc.page_content

    def test_metadata_records_the_source_relative_to_the_run_root(self, wired):
        docs = runs_rag.collect_run_documents(wired.root)
        sources = {d.metadata["source"] for d in docs}
        assert "run_nan/matrix.json" in sources
        assert "run_nan/env.json" in sources

    def test_an_unreadable_artifact_is_skipped_not_fatal(self, wired, caplog):
        """One killed job must not cost the user every other run's index."""
        (wired.root / "broken").mkdir()
        (wired.root / "broken" / "matrix.json").write_text('{"cells": [')
        docs = runs_rag.collect_run_documents(wired.root)
        assert docs
        assert any("run_nan/matrix.json" == d.metadata["source"] for d in docs)

    def test_an_artifact_cut_mid_character_is_skipped_too(self, wired):
        """The same killed job, one byte earlier in the multi-byte sequence.

        A truncated UTF-8 sequence surfaces as ``UnicodeDecodeError``, which is
        a ``ValueError``: unless the reader converts it to ``ArtifactReadError``
        it escapes the ``except`` here and the one bad file costs the user every
        other run's index -- the outcome this walk exists to prevent.
        """
        (wired.root / "broken").mkdir()
        doc = json.dumps({"cells": [{"name": "fp32\u2014off"}]}, ensure_ascii=False)
        encoded = doc.encode("utf-8")
        (wired.root / "broken" / "matrix.json").write_bytes(
            encoded[: encoded.index(b"\xe2\x80\x94") + 2]
        )
        docs = runs_rag.collect_run_documents(wired.root)
        assert docs
        assert any("run_nan/matrix.json" == d.metadata["source"] for d in docs)

    def test_indexing_then_searching_finds_the_right_run(self, wired):
        runs_rag.index_run_artifacts(wired.root)
        hits = runs_rag.search_run_docs("nan in the loss", k=3)
        assert hits
        assert "nan detected in loss" in hits[0].page_content

    def test_search_distinguishes_the_two_runs(self, wired):
        runs_rag.index_run_artifacts(wired.root)
        hang = runs_rag.search_run_docs("rccl hang", k=3)
        assert any("rccl hang detected" in doc.page_content for doc in hang)

    def test_env_snapshots_are_searchable_too(self, wired):
        runs_rag.index_run_artifacts(wired.root)
        hits = runs_rag.search_run_docs("triton rocm", k=5)
        assert any(doc.metadata["artifact_kind"] == "env" for doc in hits)

    def test_reindexing_replaces_rather_than_stacks(self, wired):
        """Append would put a second copy of every chunk behind the first."""
        store = runs_rag.index_run_artifacts(wired.root)
        first = len(runs_rag.collect_run_documents(wired.root))
        store.close()
        runs_rag.reset_caches()
        runs_rag.index_run_artifacts(wired.root)
        rows = (
            SqliteVecStore(
                path=wired.index,
                embedding=BagOfWordsEmbeddings(),
                collection=runs_rag.run_collection_name(),
            )
            ._connection()
            .execute(f'SELECT COUNT(*) FROM "chunks_{runs_rag.run_collection_name()}"')
            .fetchone()[0]
        )
        assert rows == first

    def test_an_emptied_run_root_leaves_no_stale_cells_behind(self, wired):
        """Otherwise a deleted run stays retrievable as though it were current.

        Re-indexing an empty root drops the collection outright, so the next
        search reports it as absent rather than returning nothing -- either way
        the stale cells are unreachable, which is the property that matters.
        """
        runs_rag.index_run_artifacts(wired.root)
        assert runs_rag.search_run_docs("nan", k=3)

        for path in wired.root.rglob("*.json"):
            path.unlink()
        runs_rag.reset_caches()
        runs_rag.index_run_artifacts(wired.root)
        runs_rag.reset_caches()

        with pytest.raises(runs_rag.RunCollectionMissingError):
            runs_rag.search_run_docs("nan", k=3)

    def test_the_missing_collection_message_also_names_an_empty_run_root(self, wired):
        """Someone who has just run the indexer must not be told to run it again."""
        with pytest.raises(runs_rag.RunCollectionMissingError) as exc:
            runs_rag.search_run_docs("anything", k=1)
        message = str(exc.value)
        assert "no artifacts in it" in message
        assert "AORTA_CHAT_RUNS_PATH" in message
        assert str(wired.root) in message


class TestSeparationFromTheSourceCollection:
    def test_rebuilding_the_run_collection_leaves_the_source_one_intact(self, wired):
        """The whole reason the two are separate: different refresh cadence."""
        from langchain_core.documents import Document

        source = SqliteVecStore(
            path=wired.index,
            embedding=BagOfWordsEmbeddings(),
            collection="aorta",
        )
        source.reset()
        source.add_documents(
            [Document(page_content="rocm kernel launch path", metadata={"source": "src/a.py"})]
        )

        runs_rag.index_run_artifacts(wired.root)

        assert source.collection_exists()
        assert source.similarity_search("rocm", k=1)[0].metadata["source"] == "src/a.py"

    def test_both_collections_appear_in_the_registry(self, wired):
        from langchain_core.documents import Document

        source = SqliteVecStore(
            path=wired.index, embedding=BagOfWordsEmbeddings(), collection="aorta"
        )
        source.reset()
        source.add_documents([Document(page_content="rocm", metadata={})])
        runs_rag.index_run_artifacts(wired.root)
        assert set(source.collection_names()) >= {"aorta", "aorta_runs"}


class TestARebuildThatFailsLeavesTheOldCollectionUsable:
    """The collection was dropped before the first vector existed.

    So a remote embedding provider that went down part way through -- or a
    Ctrl-C -- destroyed a working collection and left an empty one, which is
    the opposite of the interruption safety docs/chat/rag-index.md promises.
    Embedding now happens in a scratch database that is swapped in one
    transaction, so a failure costs the rebuild and nothing else.
    """

    @staticmethod
    def _fail_after(provider, n_batches: int):
        """An embedder that dies once it has embedded ``n_batches`` batches."""
        embeddings = BagOfWordsEmbeddings()
        calls = {"n": 0}

        class _Flaky(Embeddings):
            def embed_documents(self, texts):
                calls["n"] += 1
                if calls["n"] > n_batches:
                    raise RuntimeError("remote embedding endpoint went away")
                return embeddings.embed_documents(texts)

            def embed_query(self, text):
                return embeddings.embed_query(text)

        provider.get_embeddings = _Flaky
        return calls

    def test_the_previous_collection_still_answers(self, wired, monkeypatch):
        runs_rag.index_run_artifacts(wired.root)
        before = runs_rag.search_run_docs("nan", k=1)
        assert before

        self._fail_after(wired.provider, 0)
        with pytest.raises(RuntimeError, match="went away"):
            runs_rag.index_run_artifacts(wired.root)

        runs_rag.reset_caches()
        wired.provider.get_embeddings = BagOfWordsEmbeddings
        after = runs_rag.search_run_docs("nan", k=1)

        assert after[0].page_content == before[0].page_content

    def test_a_failure_part_way_through_does_not_half_write_it(self, wired, monkeypatch):
        """A partial rebuild is the worse outcome: it answers, from a subset."""
        monkeypatch.setattr(runs_rag, "_WRITE_BATCH", 1)
        runs_rag.index_run_artifacts(wired.root)
        store = SqliteVecStore(
            path=wired.index,
            embedding=BagOfWordsEmbeddings(),
            collection=runs_rag.run_collection_name(),
        )
        full = store.similarity_search("nan loss rocm hang", k=50)

        self._fail_after(wired.provider, 1)
        with pytest.raises(RuntimeError, match="went away"):
            runs_rag.index_run_artifacts(wired.root)

        runs_rag.reset_caches()
        store = SqliteVecStore(
            path=wired.index,
            embedding=BagOfWordsEmbeddings(),
            collection=runs_rag.run_collection_name(),
        )
        assert len(store.similarity_search("nan loss rocm hang", k=50)) == len(full)

    def test_it_leaves_no_staging_directory_behind(self, wired):
        self._fail_after(wired.provider, 0)
        with pytest.raises(RuntimeError, match="went away"):
            runs_rag.index_run_artifacts(wired.root)

        leftovers = list(wired.index.parent.glob(runs_rag._STAGING_PREFIX + "*"))
        assert leftovers == []

    def test_a_successful_rebuild_still_replaces_the_contents(self, wired):
        """Swapping must not turn the rebuild into an append."""
        runs_rag.index_run_artifacts(wired.root)
        runs_rag.reset_caches()
        first = len(
            SqliteVecStore(
                path=wired.index,
                embedding=BagOfWordsEmbeddings(),
                collection=runs_rag.run_collection_name(),
            ).similarity_search("nan loss rocm hang", k=100)
        )

        runs_rag.index_run_artifacts(wired.root)
        runs_rag.reset_caches()
        second = len(
            SqliteVecStore(
                path=wired.index,
                embedding=BagOfWordsEmbeddings(),
                collection=runs_rag.run_collection_name(),
            ).similarity_search("nan loss rocm hang", k=100)
        )

        assert second == first

    def test_the_source_collection_survives_a_failed_run_rebuild(self, wired):
        from langchain_core.documents import Document

        source = SqliteVecStore(
            path=wired.index, embedding=BagOfWordsEmbeddings(), collection="aorta"
        )
        source.reset()
        source.add_documents(
            [Document(page_content="rocm kernel launch path", metadata={"source": "src/a.py"})]
        )

        self._fail_after(wired.provider, 0)
        with pytest.raises(RuntimeError, match="went away"):
            runs_rag.index_run_artifacts(wired.root)

        assert source.similarity_search("rocm", k=1)[0].metadata["source"] == "src/a.py"


class TestMissingCollection:
    def test_searching_before_indexing_names_how_to_build_it(self, wired):
        """The command, not the ``python -c`` snippet it used to print.

        Naming an internal function was the only instruction available while
        no CLI command built this collection; ``aorta chat index runs`` is now
        the documented route, so that is what the error has to point at.
        """
        with pytest.raises(runs_rag.RunCollectionMissingError) as exc:
            runs_rag.search_run_docs("anything", k=1)
        assert "aorta chat index runs" in str(exc.value)

    def test_the_error_says_the_collection_is_never_published(self, wired):
        """Per-user data. An index-publishing job must not ship this collection."""
        with pytest.raises(runs_rag.RunCollectionMissingError) as exc:
            runs_rag.search_run_docs("anything", k=1)
        assert "never part of a published index" in str(exc.value)

    def test_an_absent_run_root_is_refused_with_its_path(self, wired):
        with pytest.raises(FileNotFoundError) as exc:
            runs_rag.index_run_artifacts(wired.root / "nope")
        assert "nope" in str(exc.value)

    def test_the_tool_turns_a_missing_collection_into_an_error_string(self, wired):
        """A raised exception here would abort the whole graph run."""
        from aorta.chat.tools.artifacts import search_run_artifacts

        out = search_run_artifacts.invoke({"query": "nan"})
        assert out.startswith("Error:")
        assert "aorta chat index runs" in out

    def test_the_tool_returns_results_once_indexed(self, wired):
        from aorta.chat.tools.artifacts import search_run_artifacts

        runs_rag.index_run_artifacts(wired.root)
        out = search_run_artifacts.invoke({"query": "nan in the loss"})
        assert "nan detected in loss" in out
        assert "matrix" in out

    def test_a_non_positive_k_is_rejected_as_a_string(self, wired):
        from aorta.chat.tools.artifacts import search_run_artifacts

        assert search_run_artifacts.invoke({"query": "x", "k": 0}).startswith("Error:")
