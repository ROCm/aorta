"""The torch-free embedding provider, and the two things it must not lose.

Decision 19a replaced ``sentence-transformers`` with ``fastembed``. Two
properties of the old path have to survive the swap, and one new property has to
arrive with it:

1. **The BGE query instruction.** BGE is trained asymmetrically. fastembed's own
   ``query_embed()`` applies no prefix, so without this the query side of every
   retrieval changes -- silently, since nothing fails and every other test still
   passes.
2. **Newline collapse**, for the same reason.
3. **A collection name that encodes the model.** This is new and it is the point
   of the swap. The deleted torch provider emitted 384-dimension vectors from
   the same model family, so an index it built would pass ``SqliteVecStore``'s
   dimension check and then answer from vectors that were never comparable.

Nothing here loads a model or touches the network: ``fastembed.TextEmbedding``
is constructed lazily and every test either stubs it or asks only for names.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from aorta.chat.config import settings
from aorta.chat.rag.embeddings import fastembed_bge
from aorta.chat.rag.embeddings.fastembed_bge import (
    BGE_QUERY_INSTRUCTION,
    DEFAULT_MODEL,
    LOCAL_COLLECTION_PREFIX,
    PRE_SEED_PROCEDURE,
    FastembedBgeEmbeddings,
    FastembedBgeProvider,
    ModelUnavailableError,
)


class _FakeModel:
    """Stands in for ``fastembed.TextEmbedding``, recording what it was asked."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts):
        batch = list(texts)
        self.calls.append(batch)
        # fastembed yields numpy arrays; the wrapper only needs .tolist().
        return [_FakeVector([float(index)] * 384) for index, _ in enumerate(batch)]


class _FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


@pytest.fixture()
def stubbed(monkeypatch) -> tuple[FastembedBgeEmbeddings, _FakeModel]:
    """Embeddings whose model is a stub, so nothing downloads."""
    model = _FakeModel()
    embeddings = FastembedBgeEmbeddings(model_name=DEFAULT_MODEL)
    monkeypatch.setattr(embeddings, "_model", model)
    return embeddings, model


class TestQueryInstruction:
    def test_the_prefix_matches_langchain_communitys_default_verbatim(self):
        """Any drift changes query vectors against every index built to date."""
        assert BGE_QUERY_INSTRUCTION == "Represent this question for searching relevant passages: "

    def test_a_query_is_prefixed(self, stubbed):
        embeddings, model = stubbed
        embeddings.embed_query("how do I profile?")
        assert model.calls == [[BGE_QUERY_INSTRUCTION + "how do I profile?"]]

    def test_newlines_in_a_query_become_spaces(self, stubbed):
        embeddings, model = stubbed
        embeddings.embed_query("line one\nline two")
        assert model.calls == [[BGE_QUERY_INSTRUCTION + "line one line two"]]

    def test_a_query_returns_one_vector_not_a_list_of_them(self, stubbed):
        embeddings, _ = stubbed
        vector = embeddings.embed_query("q")
        assert isinstance(vector, list)
        assert len(vector) == 384
        assert isinstance(vector[0], float)


class TestDocuments:
    def test_documents_are_not_prefixed(self, stubbed):
        """Asymmetric by design: the prefix belongs on queries only."""
        embeddings, model = stubbed
        embeddings.embed_documents(["def run(): ..."])
        assert model.calls == [["def run(): ..."]]

    def test_newlines_in_documents_become_spaces(self, stubbed):
        embeddings, model = stubbed
        embeddings.embed_documents(["def run():\n    return 1\n"])
        assert model.calls == [["def run():     return 1 "]]

    def test_every_document_is_handled(self, stubbed):
        embeddings, model = stubbed
        vectors = embeddings.embed_documents(["a\nb", "c\nd"])
        assert model.calls == [["a b", "c d"]]
        assert len(vectors) == 2


class TestLazyLoading:
    def test_construction_loads_no_model(self, monkeypatch):
        """``doctor`` and collection lookups must not cost a 65 MB download."""
        called = []
        monkeypatch.setattr(
            fastembed_bge, "_text_embedding", lambda *a, **k: called.append(a) or _FakeModel()
        )
        FastembedBgeEmbeddings()
        assert called == []

    def test_the_model_is_built_once_and_reused(self, monkeypatch):
        builds = []

        def _build(*args, **kwargs):
            builds.append(args)
            return _FakeModel()

        monkeypatch.setattr(fastembed_bge, "_text_embedding", _build)
        embeddings = FastembedBgeEmbeddings()
        embeddings.embed_query("a")
        embeddings.embed_query("b")
        assert len(builds) == 1


class TestCollectionName:
    def test_the_name_encodes_the_model(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_model", DEFAULT_MODEL)
        name = FastembedBgeProvider().collection_name()
        assert name == "aorta_fastembed_baai_bge_small_en_v1_5"
        assert name.startswith(LOCAL_COLLECTION_PREFIX)

    def test_it_is_not_the_pre_swap_name(self, monkeypatch):
        """The old provider owned the bare name ``aorta`` at the same 384 dims.

        Reusing it would let a torch-built index load here and answer from
        vectors that are not comparable, because the dimension check -- the only
        other guard the store has -- would pass.
        """
        monkeypatch.setattr(settings, "embedding_model", DEFAULT_MODEL)
        assert FastembedBgeProvider().collection_name() != "aorta"

    def test_switching_model_switches_collection(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_model", DEFAULT_MODEL)
        small = FastembedBgeProvider().collection_name()
        monkeypatch.setattr(settings, "embedding_model", "BAAI/bge-base-en-v1.5")
        assert FastembedBgeProvider().collection_name() != small

    def test_the_name_is_a_bare_identifier(self, monkeypatch):
        """It becomes part of a table name, so the store rejects anything else."""
        from aorta.chat.rag.retriever import _SAFE_COLLECTION

        monkeypatch.setattr(settings, "embedding_model", "Weird/Model.v2-x")
        assert _SAFE_COLLECTION.match(FastembedBgeProvider().collection_name())

    def test_it_never_collides_with_a_remote_collection(self, monkeypatch):
        from aorta.chat.rag.embeddings.remote_api import RemoteApiProvider

        monkeypatch.setattr(settings, "embedding_model", DEFAULT_MODEL)
        monkeypatch.setattr(settings, "remote_embedding_model", DEFAULT_MODEL)
        assert FastembedBgeProvider().collection_name() != RemoteApiProvider().collection_name()


class TestSourceRepoLookupIsFailSoft:
    """``_source_repo`` reads a *private* fastembed API, so it must not raise.

    ``_text_embedding`` calls ``model_is_cached`` -- and so ``_source_repo`` --
    from inside its own ``except`` handler. An exception from the registry walk
    would replace the download failure being explained with an unrelated one,
    and the operator would never see ``PRE_SEED_PROCEDURE``.

    A fake ``fastembed`` module is installed rather than the real one, so this
    runs identically with and without the chat extra present.
    """

    @staticmethod
    def _install_fake(monkeypatch, text_embedding) -> None:
        module = types.ModuleType("fastembed")
        module.TextEmbedding = text_embedding
        monkeypatch.setitem(sys.modules, "fastembed", module)

    def test_a_registry_that_cannot_be_read_falls_back_to_the_model_id(self, monkeypatch):
        class Renamed:
            """``_list_supported_models`` gone, as a private API may do."""

            @staticmethod
            def _list_supported_models():
                raise AttributeError("_list_supported_models")

        self._install_fake(monkeypatch, Renamed)
        assert fastembed_bge._source_repo(DEFAULT_MODEL) == DEFAULT_MODEL

    def test_a_description_without_sources_falls_back_too(self, monkeypatch):
        """Only ``.hf`` was guarded before; ``.sources`` itself can go as well."""

        class NoSources:
            @staticmethod
            def _list_supported_models():
                return [types.SimpleNamespace(model=DEFAULT_MODEL)]

        self._install_fake(monkeypatch, NoSources)
        assert fastembed_bge._source_repo(DEFAULT_MODEL) == DEFAULT_MODEL

    def test_a_readable_registry_is_still_honoured(self, monkeypatch):
        """The fallback must not swallow the answer it exists to protect."""

        class Healthy:
            @staticmethod
            def _list_supported_models():
                return [
                    types.SimpleNamespace(
                        model=DEFAULT_MODEL,
                        sources=types.SimpleNamespace(hf="qdrant/bge-small-en-v1.5-onnx-q"),
                    )
                ]

        self._install_fake(monkeypatch, Healthy)
        assert fastembed_bge._source_repo(DEFAULT_MODEL) == "qdrant/bge-small-en-v1.5-onnx-q"


class TestModelCacheProbe:
    def test_an_empty_cache_reports_absent(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        assert not fastembed_bge.model_is_cached(DEFAULT_MODEL)

    def test_weights_under_the_source_repo_name_count(self, monkeypatch, tmp_path: Path):
        """fastembed downloads from its own re-host, not from the model id.

        ``BAAI/bge-small-en-v1.5`` arrives from
        ``qdrant/bge-small-en-v1.5-onnx-q``, so that is the directory name the
        cache actually carries.
        """
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        weights = (
            tmp_path
            / "models--qdrant--bge-small-en-v1.5-onnx-q"
            / "snapshots"
            / "abc"
            / "model_optimized.onnx"
        )
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"\x00")
        assert fastembed_bge.model_is_cached(DEFAULT_MODEL)

    def test_a_hub_seeded_cache_is_recognised_too(self, monkeypatch, tmp_path: Path):
        """Plain huggingface_hub writes under $HF_HOME/hub, not beside it."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        weights = tmp_path / "hub" / "models--qdrant--bge-small-en-v1.5-onnx-q" / "m.onnx"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"\x00")
        assert fastembed_bge.model_is_cached(DEFAULT_MODEL)

    def test_a_directory_without_weights_does_not_count(self, monkeypatch, tmp_path: Path):
        """A half-finished download must not read as a warm cache."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        (tmp_path / f"models--{DEFAULT_MODEL.replace('/', '--')}").mkdir(parents=True)
        assert not fastembed_bge.model_is_cached(DEFAULT_MODEL)

    def test_hf_home_is_honoured(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "elsewhere"))
        assert fastembed_bge.model_cache_dir() == tmp_path / "elsewhere"

    def test_without_hf_home_it_is_aortas_own_cache_not_fastembeds_tmpdir(
        self, monkeypatch, tmp_path: Path
    ):
        """fastembed's default is /tmp/fastembed_cache: wiped on reboot, shared.

        Verified against fastembed 0.8.0 rather than assumed. Leaving it there
        would cost a 65 MB re-download after every reboot and put the weights in
        a directory other users on a shared node can write.
        """
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.setattr(settings, "model_cache_path", str(tmp_path / "models"))
        resolved = fastembed_bge.model_cache_dir()
        assert resolved == tmp_path / "models"
        assert "fastembed_cache" not in str(resolved)

    def test_the_resolved_cache_dir_is_what_gets_passed_to_fastembed(
        self, monkeypatch, tmp_path: Path
    ):
        """The probe and the download must agree, or doctor lies."""
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        seen = {}

        def _capture(model, cache_dir):
            seen["cache_dir"] = cache_dir
            return _FakeModel()

        monkeypatch.setattr(fastembed_bge, "_text_embedding", _capture)
        FastembedBgeEmbeddings().embed_query("q")
        assert seen["cache_dir"] == tmp_path / "hf"

    def test_describe_model_state_reports_the_offline_flag(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        state = fastembed_bge.describe_model_state(DEFAULT_MODEL)
        assert state["offline"] is True
        assert state["cached"] is False
        assert state["cache_dir"] == str(tmp_path)


class TestDownloadFailureCarriesThePreSeedProcedure:
    """Decision 21b's mitigation: the docs arrive where the user is stuck.

    An air-gapped user is blocked twice -- no published index and no model -- and
    only the first has a flag. The second surfaces as a HuggingFace connection
    error from inside fastembed, which reads as a bug in aorta rather than as
    "pre-seed a cache", so it is re-raised with the procedure attached.
    """

    def test_a_download_failure_names_hf_home_and_hf_hub_offline(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        monkeypatch.setattr(fastembed_bge, "model_is_cached", lambda model=None: False)

        import fastembed as fastembed_pkg

        def _explode(**kwargs):
            raise ConnectionError("HTTPSConnectionPool(host='huggingface.co'): timed out")

        monkeypatch.setattr(fastembed_pkg, "TextEmbedding", _explode)

        with pytest.raises(ModelUnavailableError) as exc:
            fastembed_bge._text_embedding(DEFAULT_MODEL, None)

        message = str(exc.value)
        assert "HF_HOME" in message
        assert "HF_HUB_OFFLINE=1" in message
        assert str(tmp_path) in message
        # The real cause stays visible; the advice is added, not substituted.
        assert "ConnectionError" in message
        assert "timed out" in message

    def test_it_also_points_at_side_loading_the_index(self):
        """The two blockers are separate, so the message names both fixes."""
        assert "index fetch --from" in PRE_SEED_PROCEDURE

    def test_a_failure_with_the_model_present_is_not_dressed_up_as_an_air_gap(
        self, monkeypatch, tmp_path
    ):
        """A real bug must surface as itself.

        Weights on disk and a construction failure means something else is
        wrong -- a corrupt file, an onnxruntime mismatch -- and telling the user
        to pre-seed a cache they already have would bury it.
        """
        monkeypatch.setattr(fastembed_bge, "model_is_cached", lambda model=None: True)

        import fastembed as fastembed_pkg

        def _explode(**kwargs):
            raise RuntimeError("onnxruntime: unsupported opset")

        monkeypatch.setattr(fastembed_pkg, "TextEmbedding", _explode)

        with pytest.raises(RuntimeError) as exc:
            fastembed_bge._text_embedding(DEFAULT_MODEL, None)
        assert not isinstance(exc.value, ModelUnavailableError)
        assert "unsupported opset" in str(exc.value)


class TestNoTorch:
    def test_the_provider_module_imports_nothing_from_torch(self):
        """The whole point of Decision 19a, asserted rather than assumed."""
        import ast
        import inspect

        source = inspect.getsource(fastembed_bge)
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "torch" not in imported
        assert "sentence_transformers" not in imported

    def test_the_deleted_torch_module_is_gone(self):
        """It was the sole source of the CUDA-on-AMD hazard."""
        from importlib.util import find_spec

        assert find_spec("aorta.chat.rag.embeddings.local_bge") is None
