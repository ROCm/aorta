"""``aorta chat doctor``: report everything, raise nothing, and name the fix.

Two properties matter more than the individual checks.

First, **every check runs even when an earlier one failed.** A user whose chat
session just broke wants the whole list, not the first item on it, and a doctor
that dies partway through has failed at its only job.

Second, **the missing-model check carries the pre-seed procedure.** Decision 21b
publishes the index but not the embedding weights, so an air-gapped user is
blocked twice and discovers the second blocker as a HuggingFace connection
error. The procedure has to appear where they are stuck, because nobody reads
docs at that moment.
"""

from __future__ import annotations

import ast
import shlex
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from aorta.chat import doctor
from aorta.chat.config import settings
from aorta.chat.doctor import FAIL, OK, SKIP, WARN, run_checks

MODEL = "BAAI/bge-small-en-v1.5"


def _by_name(report, name: str):
    found = [check for check in report.checks if check.name == name]
    assert found, f"no check named {name!r}; got {[c.name for c in report.checks]}"
    return found[0]


class FixedWidthEmbeddings(Embeddings):
    """384 dimensions -- what the fixture manifest claims -- with no model.

    The doctor never embeds anything, so the vectors only have to exist at the
    width the store was registered at. Deterministic and offline for the same
    reason as ``test_sqlite_vec_store``'s bag-of-words embedder.
    """

    def _embed(self, text: str) -> list[float]:
        return [float(len(text))] * 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _vec_connection(index: Path):
    """A connection that can write ``vec0`` tables, for damaging a built store.

    ``DROP``/``DELETE`` against the virtual table need the extension loaded;
    the doctor deliberately reads without it, so the damage side has to bring
    its own.
    """
    import sqlite3

    import sqlite_vec

    conn = sqlite3.connect(index)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _fill_index(index: Path, collection: str, rows: int = 3) -> None:
    """Write a store retrieval could actually read, holding ``rows`` chunks.

    Built by the real :class:`SqliteVecStore` rather than by hand-rolled DDL,
    which is the whole point. "Healthy" means an index this install could
    query, so a fixture that invents its own schema can only ever prove the
    probe agrees with the fixture: the previous one wrote a lone chunk table
    with a ``text`` column and no registry beside it, which the probe called
    healthy and ``_get_vectorstore`` rejected outright as a missing
    collection. Writing it with the real writer means the schema under test is
    the schema retrieval expects.

    ``rows=0`` empties the tables afterwards rather than skipping the build, so
    "opens and holds nothing for this install" keeps a real store's schema
    instead of becoming a second, weaker kind of damage.
    """
    from aorta.chat.rag.retriever import SqliteVecStore

    store = SqliteVecStore(path=index, embedding=FixedWidthEmbeddings(), collection=collection)
    try:
        store.add_texts([f"chunk {n}" for n in range(max(rows, 1))], provider="local")
    finally:
        store.close()

    if rows == 0:
        conn = _vec_connection(index)
        try:
            conn.execute(f'DELETE FROM "chunks_{collection}"')
            conn.execute(f'DELETE FROM "vec_{collection}"')
            conn.commit()
        finally:
            conn.close()


#: Ways a partial copy or an interrupted build breaks the readable-collection
#: contract while leaving the non-empty chunk table intact. Every one of these
#: read as healthy until the probe checked more than the chunk count.
STORE_DAMAGE = ("registry-table", "registry-row", "chunk-columns", "vec-table", "vector-rows")


def _break_store(index: Path, collection: str, how: str) -> None:
    """Damage one part of the readable-collection contract, leaving the rest."""
    from aorta.chat.rag.retriever import _REGISTRY_TABLE

    conn = _vec_connection(index)
    try:
        if how == "registry-table":
            conn.execute(f'DROP TABLE "{_REGISTRY_TABLE}"')
        elif how == "registry-row":
            conn.execute(f'DELETE FROM "{_REGISTRY_TABLE}" WHERE collection = ?', (collection,))
        elif how == "chunk-columns":
            # The exact shape the old fixture built: same row count, no column
            # retrieval can select.
            rows = conn.execute(f'SELECT COUNT(*) FROM "chunks_{collection}"').fetchone()[0]
            conn.execute(f'DROP TABLE "chunks_{collection}"')
            conn.execute(f'CREATE TABLE "chunks_{collection}" (id INTEGER PRIMARY KEY, text TEXT)')
            conn.executemany(
                f'INSERT INTO "chunks_{collection}" (text) VALUES (?)',
                [(f"chunk {n}",) for n in range(rows)],
            )
        elif how == "vec-table":
            conn.execute(f'DROP TABLE "vec_{collection}"')
        elif how == "vector-rows":
            conn.execute(
                f'DELETE FROM "vec_{collection}" WHERE rowid = '
                f'(SELECT MIN(rowid) FROM "vec_{collection}")'
            )
        else:  # pragma: no cover - guards the parametrise list against typos
            raise AssertionError(f"unknown damage {how!r}")
        conn.commit()
    finally:
        conn.close()


def _write_index(
    monkeypatch, tmp_path: Path, *, readable: bool = True, rows: int = 3, **overrides
) -> Path:
    """Install an index and a matching manifest, and point the settings at it.

    ``readable=False`` leaves filler bytes where the sqlite store should be --
    the state a truncated copy or a clobbered file leaves behind, and the one
    whose manifest still describes an index that is no longer under it.
    ``rows=0`` is the readable half of the same problem: a store that opens and
    holds nothing for this install to retrieve.
    """
    from aorta.chat.rag import manifest as manifest_mod
    from aorta.chat.rag.embeddings.factory import get_provider

    index = tmp_path / "index.sqlite"
    if readable:
        _fill_index(index, get_provider().collection_name(), rows=rows)
    else:
        index.write_bytes(b"x" * 2048)
    monkeypatch.setattr(settings, "index_path", str(index))
    values = {
        "aorta_version": doctor._dist_version("amd-aorta"),
        "aorta_sha": "a" * 40,
        "embedding_provider": "local",
        "embedding_model": MODEL,
        "dimensions": 384,
        "collection": get_provider().collection_name(),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "index_sha256": manifest_mod.sha256_file(index),
    }
    values.update(overrides)
    manifest_mod.write_manifest(index, manifest_mod.Manifest(**values))
    return index


@pytest.fixture(autouse=True)
def offline_but_quiet(monkeypatch, tmp_path: Path):
    """No network probes and no LLM preflight; each test opts into what it needs.

    ``index_path`` is pinned at an absent file rather than left alone because
    the embedding-cache hint is now conditioned on index health, so a developer
    who happens to have a real index in ``~/.cache`` would otherwise get a
    different report from CI.
    """
    monkeypatch.setattr(doctor, "_probe_huggingface", lambda: True)
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(settings, "embedding_model", MODEL)
    monkeypatch.setattr(settings, "index_path", str(tmp_path / "absent.sqlite"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))


class TestStructure:
    def test_it_reports_rather_than_raises(self):
        report = run_checks(backend=False)
        assert report.checks

    def test_python_and_aorta_are_always_reported(self):
        report = run_checks(backend=False)
        assert _by_name(report, "python").status == OK
        assert _by_name(report, "aorta").detail

    def test_the_backend_check_can_be_skipped(self):
        """It needs the network, so it is optional rather than mandatory."""
        assert _by_name(run_checks(backend=False), "llm backend").status == SKIP

    def test_chat_cli_is_reported_as_a_required_extra(self):
        """It is installed in this environment, so it must read as present."""
        assert _by_name(run_checks(backend=False), "extra chat-cli").status == OK

    def test_an_uninstalled_optional_extra_is_a_fact_not_a_finding(self, monkeypatch):
        monkeypatch.setitem(doctor._EXTRA_MODULES, "chat-imaginary", (("no_such_module", "nope"),))
        report = run_checks(backend=False)
        assert _by_name(report, "extra chat-imaginary").status == SKIP
        assert not [c for c in report.checks if c.name.startswith("extra ") and c.status == FAIL]

    def test_a_missing_required_extra_fails_with_an_install_command(self, monkeypatch):
        monkeypatch.setitem(doctor._EXTRA_MODULES, "chat-cli", (("no_such_module", "nope"),))
        check = _by_name(run_checks(backend=False), "extra chat-cli")
        assert check.status == FAIL
        assert "pip install 'amd-aorta[chat-cli]'" in check.hint

    def test_a_check_that_explodes_becomes_a_finding_not_a_traceback(self, monkeypatch):
        """A doctor must not die on its own diagnostics."""

        def _explode(report):
            raise RuntimeError("the probe itself is broken")

        monkeypatch.setattr(doctor, "_check_sqlite", _explode)
        report = run_checks(backend=False)
        assert _by_name(report, "sqlite").status == FAIL
        # And the checks after it still ran.
        assert _by_name(report, "embedding provider")

    def test_a_raising_store_probe_is_reported_and_the_run_continues(
        self, monkeypatch, tmp_path: Path
    ):
        """Why ``_index_is_healthy``'s last line is outside its own guard.

        Review asked for that line to be wrapped, on the grounds that the rest
        of the function is defence in depth. The reason it is not: the guard
        above it turns *a damaged index* into ``False``, and a raise out of
        ``_store_defect`` is not that -- it means this module's imports moved,
        which the probe deliberately lets escape rather than report as a defect
        in every index. Swallowing it would put a quiet health verdict over an
        index nobody checked, which is the defect shape this PR closes.

        The concern behind the comment is real and is already answered one
        level up: ``run_checks`` catches per check, so the raise becomes a
        ``fail`` line naming the exception and every later check still runs.
        Pinned here so the containment this argument rests on is a tested
        property rather than an assertion in a docstring.
        """
        _write_index(monkeypatch, tmp_path)

        def _explode(index_file):
            raise ImportError("no module named 'aorta.chat.rag.retriever'")

        monkeypatch.setattr(doctor, "_store_defect", _explode)
        report = run_checks(backend=False)

        failures = [check for check in report.checks if check.status == FAIL]
        assert any("ImportError" in check.detail for check in failures), failures
        # Loud, and contained: the report is still a whole report.
        assert _by_name(report, "llm tool mode")
        assert _by_name(report, "llm backend")

    def test_a_raising_health_probe_is_not_softened_into_a_cold_cache_warning(
        self, monkeypatch, tmp_path: Path
    ):
        """The same property on the ``_index_is_healthy`` path specifically.

        This is the caller review's comment was about. With a cold cache,
        ``_check_embedding_model`` asks ``_index_is_healthy`` whether to soften
        its wording, so a raise from the last line escapes *that* check rather
        than ``_check_index``. It has to stay a reported failure: wrapping the
        line to return ``False`` would produce an ordinary cold-cache warning
        and no trace of the exception anywhere in the report, which is a report
        that has quietly stopped checking something.
        """
        _write_index(monkeypatch, tmp_path)

        def _explode(index_file):
            raise ImportError("no module named 'aorta.chat.rag.retriever'")

        monkeypatch.setattr(doctor, "_store_defect", _explode)
        report = run_checks(backend=False)

        assert any(
            check.status == FAIL and "ImportError" in check.detail for check in report.checks
        ), report.checks
        # It raised before it could add its line, which is the loud outcome.
        assert not [check for check in report.checks if check.name == "embedding model cache"]


class TestEmbeddingModelCache:
    def _seed(self, tmp_path: Path) -> None:
        weights = (
            tmp_path
            / "hf"
            / "hub"
            / "models--qdrant--bge-small-en-v1.5-onnx-q"
            / "model_optimized.onnx"
        )
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"\x00")

    def test_a_warm_cache_is_ok(self, tmp_path: Path):
        self._seed(tmp_path)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert check.status == OK
        assert MODEL in check.detail

    def test_a_cold_cache_with_egress_and_no_index_is_only_a_warning(self, monkeypatch):
        """It will download itself on first use, so this is information."""
        monkeypatch.setattr(doctor, "_probe_huggingface", lambda: True)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert check.status == WARN
        assert "65 MB" in check.hint

    def test_a_cold_cache_beside_a_healthy_index_is_not_a_warning(
        self, monkeypatch, tmp_path: Path
    ):
        """``index fetch`` never warms the cache, so the happy path landed here.

        A warning that fires on every correctly completed setup teaches people
        to skim the one command that also reports the fatal mismatches.
        """
        _write_index(monkeypatch, tmp_path)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert check.status == SKIP
        assert "Nothing to do" in check.hint

    def test_a_legacy_manifest_over_an_unreadable_index_is_not_healthy(
        self, monkeypatch, tmp_path: Path
    ):
        """Saying "nothing to do" over an unopenable index is worse than over-warning.

        A manifest predating ``chunk_count`` claims no contents, so
        ``check_index`` has nothing to contradict and records no refusal --
        while the query path refuses the same file unconditionally. Reading
        only the refusal list would tell a user whose index is unusable that
        their setup is fine, and withhold the remedy.
        """
        _write_index(monkeypatch, tmp_path, readable=False)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert check.status == WARN
        assert "Nothing to do" not in check.hint

    def test_an_index_holding_nothing_for_this_install_is_not_healthy(
        self, monkeypatch, tmp_path: Path
    ):
        """An interrupted build leaves the manifest over a store with no chunks."""
        _write_index(monkeypatch, tmp_path, rows=0)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert check.status == WARN

    def test_the_pre_warm_advice_never_says_index_build(self, monkeypatch, tmp_path: Path):
        """It would overwrite a fetched index with a code-only one, silently.

        ``index build`` defaults ``--output`` to the index already installed and
        its corpus to ``src/aorta`` alone, so following it drops ``docs/`` and
        ``README.md`` out of retrieval to download a file that downloads itself.
        """
        without_index = _by_name(run_checks(backend=False), "embedding model cache").hint
        _write_index(monkeypatch, tmp_path)
        with_index = _by_name(run_checks(backend=False), "embedding model cache").hint

        for hint in (without_index, with_index):
            assert "index build" not in hint
            assert "TextEmbedding" in hint

    def test_the_pre_warm_command_is_the_one_the_procedure_uses(self, monkeypatch):
        """Two copies of an invocation drift; this is what stops them."""
        from aorta.chat.rag.embeddings.fastembed_bge import (
            PRE_SEED_PROCEDURE,
            describe_model_state,
        )

        monkeypatch.setattr(doctor, "_probe_huggingface", lambda: True)
        hint = _by_name(run_checks(backend=False), "embedding model cache").hint
        command = next(line.strip() for line in hint.splitlines() if "TextEmbedding" in line)

        # Same invocation, different cache_dir: the procedure seeds a machine
        # that has egress, this one seeds the machine being diagnosed. Both
        # halves are pinned whole rather than by a shared prefix, because the
        # prefix stops before ``cache_dir`` -- the argument the two differ on
        # and the one a drift would drop.
        state = describe_model_state()
        assert command == doctor._warm_command(state["model"], state["cache_dir"])
        assert MODEL in command
        seeded = doctor._warm_command(MODEL, "/tmp/aorta-model-cache")
        assert seeded in PRE_SEED_PROCEDURE.format(model=MODEL, cache="/tmp/cache")

    def test_the_pre_warm_command_survives_a_quote_in_the_cache_path(
        self, monkeypatch, tmp_path: Path
    ):
        """A remedy that will not parse is the failure this whole check exists to avoid.

        ``cache_dir`` follows ``HF_HOME`` and the model name is a setting, so
        neither is safe to interpolate into a single-quoted shell argument: an
        apostrophe in the path closed it early.
        """
        cache = tmp_path / "o'brien"
        monkeypatch.setenv("HF_HOME", str(cache))
        hint = _by_name(run_checks(backend=False), "embedding model cache").hint
        command = next(line.strip() for line in hint.splitlines() if "TextEmbedding" in line)

        # `shlex.split` raises on an unterminated quote, so this is the check.
        argv = shlex.split(command)
        assert argv[:2] == ["python", "-c"]
        # And what the shell would hand python has to be python, with the path
        # carried through whole rather than truncated at the apostrophe.
        ast.parse(argv[2])
        assert str(cache) in argv[2]

    def test_a_quote_in_the_model_name_is_escaped_too(self):
        """The other interpolated value, from the same untrusted place: settings."""
        command = doctor._warm_command("evil/model\"'; rm -rf /", "/tmp/cache")
        argv = shlex.split(command)
        assert argv[:2] == ["python", "-c"]
        ast.parse(argv[2])
        assert "rm -rf" not in " ".join(argv[:2])

    def test_a_cold_cache_with_no_egress_fails_and_prints_the_procedure(self, monkeypatch):
        monkeypatch.setattr(doctor, "_probe_huggingface", lambda: False)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert check.status == FAIL
        assert "HF_HOME" in check.procedure
        assert "HF_HUB_OFFLINE=1" in check.procedure

    def test_the_procedure_names_the_directory_to_populate(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(doctor, "_probe_huggingface", lambda: False)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert str(tmp_path / "hf") in check.procedure

    def test_the_procedure_also_covers_the_index_which_is_the_other_blocker(self, monkeypatch):
        """The two are separate artifacts, so one procedure would be half an answer."""
        monkeypatch.setattr(doctor, "_probe_huggingface", lambda: False)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert "index fetch --from" in check.procedure

    def test_a_remote_provider_needs_no_local_weights(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "remote")
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert check.status == SKIP

    def test_an_unknown_provider_is_reported_rather_than_raised(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "sbert")
        check = _by_name(run_checks(backend=False), "embedding provider")
        assert check.status == FAIL
        assert "unknown embedding provider" in check.detail


class TestIndexChecks:
    def test_an_absent_index_fails_with_both_ways_to_get_one(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(settings, "index_path", str(tmp_path / "absent.sqlite"))
        check = _by_name(run_checks(backend=False), "chat index")
        assert check.status == FAIL
        assert "aorta chat index fetch" in check.hint
        assert "aorta chat index build" in check.hint
        # It is already running; suggesting it back is noise.
        assert "aorta chat doctor" not in check.hint

    def test_the_absent_index_remedy_does_not_offer_fetch_on_a_remote_provider(
        self, monkeypatch, tmp_path: Path
    ):
        """A fresh remote install is the likeliest way to reach this branch.

        ``fetch_index`` validates the published manifest against the configured
        provider before installing anything, and the published asset is built
        with the local embedder -- so the user who has done nothing wrong yet
        would be led straight into a refusal.
        """
        from aorta.chat.rag import manifest as manifest_mod

        monkeypatch.setattr(settings, "index_path", str(tmp_path / "absent.sqlite"))
        monkeypatch.setattr(manifest_mod, "_configured_embedding_provider", lambda: "remote")
        check = _by_name(run_checks(backend=False), "chat index")
        assert check.status == FAIL
        commands = [line for line in check.hint.splitlines() if line.startswith("  aorta")]
        assert not any("index fetch" in line for line in commands)
        assert any("index build" in line for line in commands)
        assert "AORTA_CHAT_EMBEDDING_PROVIDER=local" in check.hint

    def test_the_absent_and_refused_remedies_are_the_same_list(self, monkeypatch, tmp_path: Path):
        """Two hand-written lists are how one of them keeps the impossible command."""
        _write_index(monkeypatch, tmp_path, embedding_model="other/model")
        refused = _by_name(run_checks(backend=False), "index manifest").procedure
        monkeypatch.setattr(settings, "index_path", str(tmp_path / "absent.sqlite"))
        absent = _by_name(run_checks(backend=False), "chat index").hint

        assert absent
        assert absent in refused

    def test_an_index_without_a_manifest_warns(self, monkeypatch, tmp_path: Path):
        index = tmp_path / "index.sqlite"
        index.write_bytes(b"x" * 1024)
        monkeypatch.setattr(settings, "index_path", str(index))
        report = run_checks(backend=False)
        assert _by_name(report, "chat index").status == OK
        assert _by_name(report, "index manifest").status == WARN

    def test_a_matching_manifest_is_ok(self, monkeypatch, tmp_path: Path):
        index = _write_index(monkeypatch, tmp_path)
        check = _by_name(run_checks(backend=False), "index manifest")
        assert check.status == OK
        assert MODEL in check.detail
        assert index.exists()

    def test_a_mismatched_manifest_fails_and_says_why_it_matters(self, monkeypatch, tmp_path: Path):
        """The report has to convey that this is not cosmetic."""
        _write_index(monkeypatch, tmp_path, embedding_model="other/model")
        check = _by_name(run_checks(backend=False), "index manifest")
        assert check.status == FAIL
        assert "queries are refused" in check.detail
        assert "not comparable" in check.procedure

    def test_the_mismatch_remedy_offers_fetch_on_a_local_provider(
        self, monkeypatch, tmp_path: Path
    ):
        _write_index(monkeypatch, tmp_path, embedding_model="other/model")
        check = _by_name(run_checks(backend=False), "index manifest")
        assert "aorta chat index fetch" in check.procedure
        # It is already running; suggesting it back is noise.
        assert "aorta chat doctor" not in check.procedure

    def test_the_mismatch_remedy_does_not_lead_with_fetch_on_a_remote_provider(
        self, monkeypatch, tmp_path: Path
    ):
        """CI publishes one asset, built locally, so a fetch would refuse in turn."""
        from aorta.chat.rag import manifest as manifest_mod

        _write_index(monkeypatch, tmp_path, embedding_model="other/model")
        # Only the remedy's view of the provider, not the factory's: building a
        # real remote provider needs an endpoint and a key, which is a different
        # check's problem.
        monkeypatch.setattr(manifest_mod, "_configured_embedding_provider", lambda: "remote")
        check = _by_name(run_checks(backend=False), "index manifest")
        commands = [line for line in check.procedure.splitlines() if line.startswith("  aorta")]
        assert commands == ["  aorta chat index build     embed the corpus with the configured"]
        assert "AORTA_CHAT_EMBEDDING_PROVIDER=local" in check.procedure

    def test_a_manifest_over_an_unreadable_store_is_not_reported_as_matching(
        self, monkeypatch, tmp_path: Path
    ):
        """The query path refuses this file, so the report must not call it a match.

        ``check_index`` only turns an unopenable index into a refusal when the
        manifest claims a chunk count to contradict, so a manifest predating
        that field left this state reported as ``ok``.
        """
        _write_index(monkeypatch, tmp_path)
        assert _by_name(run_checks(backend=False), "index manifest").status == OK

        _write_index(monkeypatch, tmp_path, readable=False)
        check = _by_name(run_checks(backend=False), "index manifest")
        assert check.status == FAIL
        assert "cannot read" in check.detail
        assert "aorta chat index build" in check.procedure

    def test_the_healthy_fixture_is_a_store_retrieval_can_actually_read(
        self, monkeypatch, tmp_path: Path
    ):
        """The probe and the fixture must not agree with each other and nobody else.

        This is the assertion whose absence let the previous fixture stand. It
        built a chunk table with a ``text`` column and no registry or vector
        table beside it; the probe counted the rows, called the index healthy,
        and ``_get_vectorstore`` rejected the same file outright as a missing
        collection. Every test asserting ``healthy`` was therefore asserting it
        of a store no query could answer from. Reading the fixture back through
        the real class is what keeps "healthy" meaning queryable, and what will
        fail if the store's schema moves under the probe.
        """
        from aorta.chat.rag.embeddings.factory import get_provider
        from aorta.chat.rag.retriever import SqliteVecStore

        index = _write_index(monkeypatch, tmp_path)
        assert doctor._store_defect(index) == ""

        store = SqliteVecStore(
            path=index,
            embedding=FixedWidthEmbeddings(),
            collection=get_provider().collection_name(),
        )
        try:
            assert store.collection_exists()
            assert len(store.similarity_search("chunk", k=3)) == 3
        finally:
            store.close()

    def test_the_vector_count_comes_from_a_table_sqlite_vec_still_writes(
        self, monkeypatch, tmp_path: Path
    ):
        """Pin the one assumption the parity check makes about somebody else's schema.

        Counting ``vec_<collection>`` itself needs the extension loaded, and
        the probe deliberately reads without it, so the row count comes from
        vec0's ``_rowids`` shadow table -- an ordinary table, and sqlite-vec's
        layout rather than ours. The probe treats its absence as "cannot tell"
        instead of as a defect, because a renamed shadow table would otherwise
        make every healthy index report as unreadable. That fail-open is only
        safe if a rename is loud somewhere, and this is where: it fails here,
        at the assumption, rather than silently narrowing the check in the
        field.
        """
        import sqlite3

        from aorta.chat.rag.embeddings.factory import get_provider

        index = _write_index(monkeypatch, tmp_path)
        collection = get_provider().collection_name()

        conn = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
        try:
            tables = {
                name
                for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            assert f"vec_{collection}_rowids" in tables, sorted(tables)
            chunks = conn.execute(f'SELECT COUNT(*) FROM "chunks_{collection}"').fetchone()[0]
            vectors = conn.execute(f'SELECT COUNT(*) FROM "vec_{collection}_rowids"').fetchone()[0]
            assert vectors == chunks == 3
        finally:
            conn.close()

    @pytest.mark.parametrize("how", STORE_DAMAGE)
    def test_a_partially_copied_store_is_not_reported_as_matching(
        self, monkeypatch, tmp_path: Path, how
    ):
        """A non-empty chunk table is necessary for a query, not sufficient for one.

        Each of these leaves the chunk table this probe used to be satisfied by
        and removes something the read path also needs -- the registry, the row
        in it, the columns retrieval selects, the vector table, or the vectors
        themselves. All of them answer nothing, and all of them were reported
        as a matching index.
        """
        from aorta.chat.rag.embeddings.factory import get_provider

        index = _write_index(monkeypatch, tmp_path)
        assert doctor._store_defect(index) == ""

        _break_store(index, get_provider().collection_name(), how)
        assert doctor._store_defect(index)

        report = run_checks(backend=False)
        assert _by_name(report, "index manifest").status == FAIL
        assert "cannot read" in _by_name(report, "index manifest").detail
        # And the other reader of the same helper moves with it.
        assert _by_name(report, "embedding model cache").status != SKIP

    @pytest.mark.parametrize("how", STORE_DAMAGE)
    def test_the_damaged_states_are_ones_the_manifest_check_cannot_see(
        self, monkeypatch, tmp_path: Path, how
    ):
        """Otherwise the test above would pass without the store probe existing.

        The manifest is a sidecar: it is not re-read from the store, and
        ``validate`` never hashes the ``.sqlite``. A legacy manifest claims no
        chunk count either, so nothing in ``check_index`` contradicts any of
        this damage -- which is exactly why the probe has to.
        """
        from aorta.chat.rag.embeddings.factory import get_provider
        from aorta.chat.rag.index_ops import check_index

        index = _write_index(monkeypatch, tmp_path)
        _break_store(index, get_provider().collection_name(), how)
        assert not check_index(index, strict=False).refusals

    @pytest.mark.parametrize("state", ["healthy", "unreadable", "empty"])
    def test_both_readers_of_index_health_agree(self, monkeypatch, tmp_path: Path, state):
        """One report must not call the index fine on one line and unusable on another.

        The cold-cache hint gates on index health and the manifest line reports
        it. They were two separate reads of the same question, which is how the
        manifest line came to say ``ok`` over a file the hint had already
        concluded was unusable; they share one helper for that reason.
        """
        _write_index(
            monkeypatch,
            tmp_path,
            readable=state != "unreadable",
            rows=0 if state == "empty" else 3,
        )
        report = run_checks(backend=False)
        queryable = _by_name(report, "index manifest").status == OK
        assert queryable == (state == "healthy")
        assert queryable == (_by_name(report, "embedding model cache").status == SKIP)

    def test_version_drift_warns_rather_than_failing(self, monkeypatch, tmp_path: Path):
        _write_index(monkeypatch, tmp_path, aorta_version="0.0.1")
        check = _by_name(run_checks(backend=False), "index manifest")
        assert check.status == WARN
        assert any("source drift" in line for line in check.hint.splitlines())


class TestToolMode:
    """The setup-time signal for a failure that otherwise only shows as a dead query.

    ``text`` is the shipped default and a reasoning model cannot drive it: it
    writes its working to a separate channel and returns empty content where the
    ``ACTION:`` line should be, so every action-routed question spends billed
    rounds on a reply the protocol cannot read. The user-facing give-up message
    points here, so this check has to name both the problem and the setting that
    fixes it.

    What happens *after* the empty reply is deliberately not asserted as a
    single outcome -- see
    ``test_the_warning_names_the_cost_rather_than_predicting_one_outcome``.
    """

    def test_the_resolved_mode_is_always_reported(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_tool_mode", "native")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == OK
        assert "native" in check.detail

    def test_native_on_a_stock_vllm_says_what_the_endpoint_has_to_accept(self, monkeypatch):
        """Found by sweeping this check for the shape review flagged elsewhere in it.

        ``native`` had the report's other advice-free green line, and it is the
        worse of the two: it is the mode with an endpoint requirement, nothing
        in the report tests that requirement -- ``_check_backend`` asks for
        ``/health``, which a server that rejects ``tools`` answers normally --
        and ``native`` on a stock local vLLM is a configuration that cannot
        answer a single action-routed question.
        """
        monkeypatch.setattr(settings, "llm_tool_mode", "native")
        monkeypatch.setattr(settings, "llm_provider", "vllm")
        monkeypatch.setattr(settings, "vllm_model", "Qwen/Qwen2.5-Coder-7B-Instruct")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == OK
        assert "/health" in check.hint
        assert "--enable-auto-tool-choice" in check.hint

    def test_native_on_a_remote_gateway_names_the_gateway_requirement_instead(self, monkeypatch):
        """The remedy is the provider's, the same way it is in text mode."""
        monkeypatch.setattr(settings, "llm_tool_mode", "native")
        monkeypatch.setattr(settings, "llm_provider", "openai")
        monkeypatch.setattr(settings, "remote_llm_model", "gpt-4o")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert "gateway" in check.hint
        assert "--enable-auto-tool-choice" not in check.hint

    def test_every_reported_tool_mode_carries_a_hint(self, monkeypatch):
        """The invariant behind both fixes: no green tool-mode line without advice.

        Asserted as a sweep rather than per branch, because a hint on three
        branches out of five is how the gap got there in the first place.
        """
        monkeypatch.setattr(settings, "llm_provider", "vllm")
        for mode, model in (
            ("native", "Qwen/Qwen2.5-Coder-7B-Instruct"),
            ("native", ""),
            ("text", "Qwen/Qwen2.5-Coder-7B-Instruct"),
            ("text", "openai/gpt-oss-20b"),
            ("text", ""),
        ):
            monkeypatch.setattr(settings, "llm_tool_mode", mode)
            monkeypatch.setattr(settings, "vllm_model", model)
            check = _by_name(run_checks(backend=False), "llm tool mode")
            assert check.hint, (mode, model)

    def test_text_on_a_local_vllm_is_ok_but_still_costs_the_native_flags(self, monkeypatch):
        """A stock vLLM drives text mode, and needs two server flags for native."""
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "vllm")
        monkeypatch.setattr(settings, "vllm_model", "Qwen/Qwen2.5-Coder-7B-Instruct")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == OK
        assert "Qwen/Qwen2.5-Coder-7B-Instruct" in check.detail
        assert "--enable-auto-tool-choice" in check.hint

    def test_a_locally_served_reasoning_model_is_warned_about_too(self, monkeypatch):
        """The reasoning channel belongs to the model, not to the endpoint.

        Reading only ``remote_llm_model`` left gpt-oss on a local vLLM -- the
        configuration ``docs/chat/providers.md`` measures at 0 parseable
        actions in 8 rounds -- with no signal at all, on the provider that is
        the shipped default.
        """
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "vllm")
        monkeypatch.setattr(settings, "vllm_model", "openai/gpt-oss-20b")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == WARN
        assert 'llm_tool_mode = "native"' in check.hint
        # Not the remote remedy: a stock vLLM has to be restarted for native.
        assert "--enable-auto-tool-choice" in check.hint
        assert "gateway" not in check.hint

    def test_a_provider_with_no_model_setting_reads_no_name(self, monkeypatch):
        """Guessing which setting holds it would invent one; the backend check reports it."""
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "not-a-provider")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == OK
        assert "no model name to check" in check.detail

    def test_the_line_with_no_model_name_still_carries_the_fix(self, monkeypatch):
        """It was the one green line in the report offering no route to a fix.

        The user it belongs to is already misconfigured -- an empty model
        setting, or a provider no backend is registered for -- and the tool-mode
        line names the symptom they will hit first. The hint does not depend on
        the model name, so there is nothing to withhold: every other branch of
        this check carries the ``native`` pointer, and leaving this one silent
        made the table's only advice-free cell the one that needed it.
        """
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "not-a-provider")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert 'llm_tool_mode = "native"' in check.hint

    def test_an_empty_model_setting_on_a_known_provider_gets_that_native_note(self, monkeypatch):
        """The provider is known even when the model is not, so the cost is too."""
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "vllm")
        monkeypatch.setattr(settings, "vllm_model", "")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == OK
        assert 'llm_tool_mode = "native"' in check.hint
        assert "--enable-auto-tool-choice" in check.hint

    def test_every_text_mode_branch_carries_the_native_pointer(self, monkeypatch):
        """The property, swept across all three ``text`` outcomes at once.

        A hint on two branches out of three is how the gap this closes got
        there, so it is asserted as an invariant rather than per branch.
        """
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "openai")
        for model in ("", "prod-chat-deployment", "o3-mini"):
            monkeypatch.setattr(settings, "remote_llm_model", model)
            check = _by_name(run_checks(backend=False), "llm tool mode")
            assert 'llm_tool_mode = "native"' in check.hint, model

    def test_text_with_a_reasoning_model_warns_and_names_the_setting(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "openai")
        monkeypatch.setattr(settings, "remote_llm_model", "GPT-oss-20B")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == WARN
        assert "GPT-oss-20B" in check.detail
        assert 'llm_tool_mode = "native"' in check.hint
        assert "empty content" in check.hint

    def test_the_warning_names_the_cost_rather_than_predicting_one_outcome(self, monkeypatch):
        """The hint may not claim the question ends up unanswered.

        It used to: "the act loop re-prompts until it gives up and the question
        is answered with nothing". #464 makes that false -- it escalates to
        ``native`` once on this exact signature, and adds a labelled
        retrieval-only fallback when neither protocol answers -- so the same
        ``text`` setting can now end in a good answer one round late, a degraded
        answer, or nothing. Which one is not readable from
        ``settings.llm_tool_mode``: the escalation only moves the built-in
        default, never a mode the user set, and this check cannot tell those
        apart.

        What is true either way, and what the user can act on, is that the round
        is spent and wasted. So the hint states the cost and gives the outcome
        as a disjunction. Asserted here so a later tightening back to one
        prediction fails rather than quietly re-contradicting #464, whichever
        of the two lands first.
        """
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "openai")
        monkeypatch.setattr(settings, "remote_llm_model", "o3-mini")
        hint = _by_name(run_checks(backend=False), "llm tool mode").hint

        assert "billed rounds" in hint
        assert "late, degraded, or not at all" in hint
        # The claims #464 falsifies. "gives up" and "answered with nothing"
        # describe one of three outcomes as though it were the only one.
        for overstatement in ("gives up", "answered with nothing", "re-prompts until"):
            assert overstatement not in hint, overstatement

    @pytest.mark.parametrize("model", ["gpt-oss-120b", "o3-mini", "deepseek-r1", "Qwen/QwQ-32B"])
    def test_the_reasoning_models_it_recognises(self, monkeypatch, model):
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "litellm")
        monkeypatch.setattr(settings, "remote_llm_model", model)
        assert _by_name(run_checks(backend=False), "llm tool mode").status == WARN

    @pytest.mark.parametrize(
        "model",
        [
            # ``gpt-4o`` and ``gpt-4o-mini`` are the pair worth pinning: they
            # sit one character from ``o1``-``o4`` and are the most widely
            # deployed non-reasoning models the pattern has to clear.
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-3.5-turbo-0125",
            "claude-sonnet-4",
            "llama-3.3-70b",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "deepseek-ai/DeepSeek-V3",
            "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
            "tiiuae/Falcon3-10B-Instruct",
            "01-ai/Yi-1.5-34B-Chat",
            "ibm-granite/granite-3.1-8b-instruct",
            "CohereForAI/c4ai-command-r-plus",
            "command-r7b-12-2024",
        ],
    )
    def test_a_model_that_can_drive_text_mode_is_not_warned_about(self, monkeypatch, model):
        """A warning on every correct setup is worth less than no warning.

        Review asked whether the pattern false-positives in real use, so the
        list is real published names rather than invented ones -- the families
        a gateway or a vLLM server actually serves. It clears all of them.

        The shape it cannot clear is a bespoke deployment ending in a revision
        suffix (``internal-llama-70b-r1``), which is why the pattern only ever
        chooses the wording: see ``_REASONING_MODEL_PATTERN``.
        """
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "openai")
        monkeypatch.setattr(settings, "remote_llm_model", model)
        assert _by_name(run_checks(backend=False), "llm tool mode").status == OK

    def test_it_still_points_at_native_for_a_deployment_name_it_cannot_read(self, monkeypatch):
        """A gateway can call a deployment anything, so the OK line has to carry the fix too."""
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "openai")
        monkeypatch.setattr(settings, "remote_llm_model", "prod-chat-deployment")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == OK
        assert 'llm_tool_mode = "native"' in check.hint

    def test_an_unknown_mode_fails_before_it_raises_mid_query(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_tool_mode", "function_calling")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == FAIL
        assert "function_calling" in check.detail


class TestBackendCheck:
    """The check calls ``probe``, not ``preflight``.

    Which is the whole point: ``preflight`` is allowed to be permissive, so a
    diagnostic built on it reported ``ok`` for the most likely failure. The
    reachability behaviour itself lives in test_backend_reachability.py; these
    pin the report shape against a stand-in backend.
    """

    def test_an_unreachable_backend_fails_with_the_underlying_error(self, monkeypatch):
        from aorta.chat.inference.providers import factory

        class _Dead:
            name = "vllm"

            async def probe(self, timeout=None):
                raise ConnectionError("connection refused")

            def describe(self):
                return "vllm at http://localhost:8000/v1"

        monkeypatch.setattr(factory, "get_backend", lambda *a, **k: _Dead())
        check = _by_name(run_checks(backend=True), "llm backend")
        assert check.status == FAIL
        assert "connection refused" in check.hint

    def test_an_unexpected_exception_type_is_named_in_the_hint(self, monkeypatch):
        """Only BackendUnreachableError's message stands on its own."""
        from aorta.chat.inference.providers import factory

        class _Weird:
            name = "vllm"

            async def probe(self, timeout=None):
                raise ConnectionError("connection refused")

            def describe(self):
                return "vllm at http://localhost:8000/v1"

        monkeypatch.setattr(factory, "get_backend", lambda *a, **k: _Weird())
        assert _by_name(run_checks(backend=True), "llm backend").hint.startswith("ConnectionError:")

    def test_a_permissive_preflight_is_not_what_gets_called(self, monkeypatch):
        """The false positive, pinned: a backend that starts anyway is still a FAIL."""
        from aorta.chat.inference.providers import factory

        class _StartsAnyway:
            name = "vllm"

            async def preflight(self):
                return None  # what the local backend does after waiting 300s

            async def probe(self, timeout=None):
                raise ConnectionError("connection refused")

            def describe(self):
                return "vllm at http://localhost:8000/v1"

        monkeypatch.setattr(factory, "get_backend", lambda *a, **k: _StartsAnyway())
        assert _by_name(run_checks(backend=True), "llm backend").status == FAIL

    def test_a_healthy_backend_is_ok(self, monkeypatch):
        from aorta.chat.inference.providers import factory

        class _Alive:
            name = "vllm"

            async def probe(self, timeout=None):
                return None

            def describe(self):
                return "vllm at http://localhost:8000/v1"

        monkeypatch.setattr(factory, "get_backend", lambda *a, **k: _Alive())
        assert _by_name(run_checks(backend=True), "llm backend").status == OK

    def test_the_probe_is_given_a_diagnostics_budget(self, monkeypatch):
        """Five minutes is a session's patience, not a waiting operator's."""
        from aorta.chat.inference.providers import factory

        budgets: list[float | None] = []

        class _Recording:
            name = "vllm"

            async def probe(self, timeout=None):
                budgets.append(timeout)

            def describe(self):
                return "vllm at http://localhost:8000/v1"

        monkeypatch.setattr(factory, "get_backend", lambda *a, **k: _Recording())
        run_checks(backend=True)
        assert budgets and budgets[0] is not None and budgets[0] <= 10
