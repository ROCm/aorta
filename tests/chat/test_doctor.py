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
import re
import shlex
from contextlib import suppress
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


#: Ways a partial copy or an interrupted build breaks the *structure* the
#: readable-collection contract needs, while leaving the non-empty chunk table
#: intact. Every one of these read as healthy until the probe checked more than
#: the chunk count. These are about tables, rows and the registry, so they are
#: not derivable from the read path's column list -- unlike the value states
#: below, which are.
STRUCTURAL_DAMAGE = (
    "registry-table",
    "registry-row",
    "chunk-columns",
    "vec-table",
    "vector-rows",
    "vector-rowids",
    "registry-width",
    "registry-width-text",
)

#: What each column ``_knn`` selects has to hold for a query to survive it, as
#: the damage states that violate it.
#:
#: Hand-maintaining the state list is what failed twice. Review found the
#: ``metadata`` states missing in one round and the ``content`` states missing
#: in the next -- adjacent columns of the same ``SELECT``, the same omission
#: both times, because the list was written beside the read path rather than
#: taken from it. A third round of adding a state would buy the same fix a
#: third time. So the states are keyed by column and the columns come from
#: ``_read_path_columns()``: an oracle that is right by construction rather
#: than by having enumerated enough.
#:
#: An empty tuple is a claim, not a gap, and needs the reason with it.
COLUMN_DAMAGE = {
    "content": ("content-null", "content-not-text"),
    "metadata": ("metadata-not-json", "metadata-not-an-object"),
    # No value state, because none is reachable: ``vec0`` validates on write.
    # Measured against the extension rather than assumed -- a text value, a
    # short blob and an integer are each refused with "invalid"/"Dimension
    # mismatch", and a NULL is a silent no-op that leaves the stored vector
    # in place. So this column cannot hold a value ``_deserialise`` chokes on,
    # and the ways it does defeat a query -- no vector table, missing rows,
    # rowids that do not line up -- are the structural states above.
    "embedding": (),
}


def _read_path_columns() -> tuple[str, ...]:
    """The columns the read path selects, read out of the query it runs.

    Taken from ``_knn``'s own SQL so the two cannot drift: a column added to
    that ``SELECT`` arrives here without anyone updating a list. Reading the
    source rather than an exported constant keeps this inside the test --
    ``retriever`` is not this PR's to change, and a constant it did not ask for
    would be a second thing to keep in step.

    Reading source with a regex is only safe because it fails closed. It wants
    the outer ``SELECT``, and matches it rather than the KNN subquery because
    that one is indented inside its string literal. If a reformat ever changed
    that, the match either fails the identifier assertion below or yields
    ``rowid``/``distance``, which ``COLUMN_DAMAGE`` does not describe -- so
    ``_derive_store_damage`` raises. Every way this can misread the query is
    therefore loud; none of them silently sweeps the wrong set.
    """
    import inspect

    from aorta.chat.rag import retriever

    source = inspect.getsource(retriever.SqliteVecStore._knn)
    match = re.search(r'"SELECT (.+?) FROM', source)
    assert match, (
        "could not find the read path's SELECT in _knn -- this oracle derives "
        "its damage states from that query and cannot vouch for anything without it"
    )
    columns = tuple(part.strip().split(".")[-1] for part in match.group(1).split(","))
    assert all(column.isidentifier() for column in columns), columns
    return columns


def _derive_store_damage() -> tuple[str, ...]:
    """``STRUCTURAL_DAMAGE`` plus a value state per column the read path reads.

    Fails closed. A column ``COLUMN_DAMAGE`` has never heard of is one this
    oracle has no states for, so it raises here rather than sweeping the ones
    it happens to know and reporting agreement it did not establish -- the
    failure mode being fixed, where the probe and the oracle agreed with each
    other while both ignored a column every query reads.
    """
    columns = _read_path_columns()
    undescribed = [column for column in columns if column not in COLUMN_DAMAGE]
    if undescribed:
        raise AssertionError(
            f"_knn now selects {', '.join(undescribed)}, which COLUMN_DAMAGE does "
            "not describe. Add the damage states for it -- or an empty tuple with "
            "the reason none is reachable -- because an undescribed column is one "
            "this oracle cannot vouch for, and every test below would otherwise "
            "pass without covering it."
        )
    return STRUCTURAL_DAMAGE + tuple(state for column in columns for state in COLUMN_DAMAGE[column])


#: Every way the store under a matching manifest can stop answering. Both the
#: probe's per-state tests and ``TestStoreProbeAgreesWithTheReadPath`` read
#: this, which is the point: the probe models the read path rather than asking
#: it (see ``doctor._store_defect``), so the only thing keeping the model
#: honest is a set neither side gets to curate.
STORE_DAMAGE = _derive_store_damage()


def _break_store(index: Path, collection: str, how: str) -> None:
    """Damage one part of the readable-collection contract, leaving the rest."""
    from aorta.chat.rag.retriever import _REGISTRY_TABLE

    conn = _vec_connection(index)
    try:
        if how.startswith("content-"):
            # ``content TEXT NOT NULL`` means a store aorta wrote cannot reach
            # these, so the state is the one the finding described: a store
            # carried in by hand, or rebuilt by another tool, whose column is
            # nullable and untyped. Recreated rather than updated in place for
            # both reasons -- NOT NULL would refuse the NULL, and TEXT affinity
            # would quietly convert the integer to '123' and hide the state.
            # Ids and metadata are carried over so the join and the vectors
            # still line up: only the value ``Document`` is handed is wrong.
            rows = conn.execute(f'SELECT id, metadata FROM "chunks_{collection}"').fetchall()
            conn.execute(f'DROP TABLE "chunks_{collection}"')
            conn.execute(
                f'CREATE TABLE "chunks_{collection}" '
                "(id INTEGER PRIMARY KEY, content, metadata TEXT NOT NULL)"
            )
            value = None if how.endswith("null") else 123
            conn.executemany(
                f'INSERT INTO "chunks_{collection}" (id, content, metadata) VALUES (?, ?, ?)',
                [(id_, value, metadata) for id_, metadata in rows],
            )
            conn.commit()
            return
        if how.startswith("metadata-"):
            # The chunk rows are all present and the vectors match them; only
            # what ``_knn`` hands to ``json.loads`` is wrong. ``Document``
            # additionally needs a mapping, so valid non-object JSON is its own
            # state rather than a variation of the first.
            value = "not json at all" if how.endswith("not-json") else "123"
            conn.execute(f'UPDATE "chunks_{collection}" SET metadata = ?', (value,))
            conn.commit()
            return
        if how.startswith("registry-width"):
            # Everything else stays intact: the collection is registered, the
            # columns and vector table are there and the rows match. Only the
            # width `_knn` compares the query vector against is wrong, which is
            # enough to refuse every search.
            width = "'wide'" if how.endswith("text") else "999"
            conn.execute(
                f'UPDATE "{_REGISTRY_TABLE}" SET dimension = {width} WHERE collection = ?',
                (collection,),
            )
            conn.commit()
            return
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
        elif how == "vector-rowids":
            moved = conn.execute(f'SELECT rowid, embedding FROM "vec_{collection}"').fetchall()
            conn.execute(f'DELETE FROM "vec_{collection}"')
            conn.executemany(
                f'INSERT INTO "vec_{collection}" (rowid, embedding) VALUES (?, ?)',
                [(rowid + 10_000, embedding) for rowid, embedding in moved],
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


def _probe(index: Path) -> str:
    """``_store_defect`` with the width its callers pass it.

    Read back off the manifest rather than written as a literal, so this cannot
    quietly stop matching the fixture -- and so it is the same number
    production passes, which is the whole subject of the width check.
    """
    from aorta.chat.rag import manifest as manifest_mod

    return doctor._store_defect(index, manifest_mod.read_manifest(index).dimensions)


def _read_path_answers(index: Path) -> str:
    """Whether a real retrieval can answer out of ``index``, and why not if it cannot.

    The oracle for ``TestStoreProbeAgreesWithTheReadPath``. Deliberately the
    genuine article -- ``SqliteVecStore.similarity_search``, sqlite-vec loaded,
    the real join -- because a hand-rolled approximation of the read path is
    exactly what ``_store_defect`` already is, and a second one would only
    prove the two approximations agree with each other.

    Returns "" when a query comes back with rows, and the failure otherwise. An
    empty result counts as answering: the read path did not refuse the index,
    and the probe being stricter about that is a documented one-way asymmetry.
    """
    from aorta.chat.rag.embeddings.factory import get_provider
    from aorta.chat.rag.retriever import SqliteVecStore

    store = SqliteVecStore(
        path=index, embedding=FixedWidthEmbeddings(), collection=get_provider().collection_name()
    )
    try:
        store.similarity_search("chunk", k=3)
        return ""
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        with suppress(Exception):
            store.close()


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

        def _explode(index_file, dimensions):
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

        def _explode(index_file, dimensions):
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
        monkeypatch.setattr(settings, "remote_embedding_api_key", "sk-test")
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert check.status == SKIP

    def test_a_remote_provider_without_a_key_fails_the_provider_check(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "remote")
        monkeypatch.setattr(settings, "remote_embedding_api_key", "")
        check = _by_name(run_checks(backend=False), "embedding provider")
        assert check.status == FAIL
        assert "remote_embedding_api_key is not set" in check.detail

    def test_an_unknown_provider_is_reported_rather_than_raised(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "sbert")
        check = _by_name(run_checks(backend=False), "embedding provider")
        assert check.status == FAIL
        assert "unknown embedding provider" in check.detail

    def test_the_unknown_provider_row_carries_a_route_to_a_fix(self, monkeypatch):
        """It was a FAIL with an empty hint -- the complaint this PR started from.

        This row already caught the factory's ``ValueError`` rather than
        letting the name resolve to local, so it was the caller that had the
        distinction right. What it did not have was a remedy, and there was one
        available once ``remedy_lines`` learned the state.
        """
        monkeypatch.setattr(settings, "embedding_provider", "sbert")
        check = _by_name(run_checks(backend=False), "embedding provider")
        assert check.hint, "a FAIL naming a misconfigured setting with nothing to do about it"
        assert 'embedding_provider = "local"' in check.hint
        assert "AORTA_CHAT_EMBEDDING_PROVIDER=local" in check.hint

    def test_the_two_rows_do_not_disagree_about_the_remedy(self, monkeypatch, tmp_path: Path):
        """One text, so a future edit cannot move one row and not the other.

        Both rows fire together for this install -- the provider is unknown
        *and* there is no index -- and a reader seeing two different remedies
        for one typo would reasonably conclude they need both.
        """
        monkeypatch.setattr(settings, "embedding_provider", "sbert")
        monkeypatch.setattr(settings, "index_path", str(tmp_path / "absent.sqlite"))
        report = run_checks(backend=False)
        provider_row = _by_name(report, "embedding provider")
        index_row = _by_name(report, "chat index")

        assert provider_row.status == FAIL and index_row.status == FAIL
        assert provider_row.hint == index_row.hint

    def test_neither_row_offers_a_command_that_dies_on_the_same_setting(
        self, monkeypatch, tmp_path
    ):
        """The finding itself: two commands offered, both fatal on resolution."""
        monkeypatch.setattr(settings, "embedding_provider", "sbert")
        monkeypatch.setattr(settings, "index_path", str(tmp_path / "absent.sqlite"))
        report = run_checks(backend=False)

        for name in ("embedding provider", "chat index"):
            row = _by_name(report, name)
            for text in (row.hint, row.procedure, row.detail):
                for path, _, offered in _commands_named_in(text or ""):
                    assert not (offered and path[:1] == ["index"]), (
                        f"{name} offers 'aorta chat {' '.join(path)}', which cannot "
                        "resolve the provider it needs"
                    )


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
        # A key as well, or the remedy answers a different question: with an
        # empty one the build is the impossible command and the block leads
        # with the switch back to local instead.
        monkeypatch.setattr(settings, "remote_embedding_api_key", "sk-test")
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
        monkeypatch.setattr(settings, "remote_embedding_api_key", "sk-test")
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
        assert _probe(index) == ""

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
        assert _probe(index) == ""

        _break_store(index, get_provider().collection_name(), how)
        assert _probe(index)

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


class TestNativeModeOnAReasoningModel:
    """The one mode-and-model pair where the fallback advice is the dead end.

    ``_NATIVE_MODE_HINT`` ends by naming ``text`` as the mode to try if the
    endpoint turns out to reject tools. For a reasoning model that is the
    failure the ``text`` branch of this same check warns about, so following it
    walks straight into what the check exists to prevent -- and the module has
    the model name on this path already.
    """

    def _native(self, monkeypatch, model: str) -> None:
        monkeypatch.setattr(settings, "llm_tool_mode", "native")
        monkeypatch.setattr(settings, "llm_provider", "openai")
        monkeypatch.setattr(settings, "remote_llm_model", model)

    @pytest.mark.parametrize("model", ["gpt-oss-120b", "DeepSeek-R1", "o3-mini", "QwQ-32B"])
    def test_a_reasoning_model_is_not_sent_to_text(self, monkeypatch, model):
        """The assertion the review asked for, over the detector's own vocabulary."""
        self._native(monkeypatch, model)
        check = _by_name(run_checks(backend=False), "llm tool mode")

        assert check.status == OK
        assert '"text" is the mode to try' not in check.hint
        assert 'cannot drive "text" mode' in check.hint

    @pytest.mark.parametrize("model", ["gpt-4o-mini", "claude-sonnet-4-5", "llama-3.3-70b"])
    def test_an_ordinary_model_keeps_the_fallback(self, monkeypatch, model):
        """The advice is right for everything else, so it must not be dropped wholesale."""
        self._native(monkeypatch, model)
        check = _by_name(run_checks(backend=False), "llm tool mode")

        assert '"text" is the mode to try' in check.hint

    def test_the_two_branches_do_not_contradict_each_other(self, monkeypatch):
        """Read the same model through both modes; neither may recommend the other.

        This is the property under the finding. Whichever mode a reasoning
        model is configured in, the report must not route it to the one that
        cannot parse its replies.
        """
        self._native(monkeypatch, "gpt-oss-120b")
        native = _by_name(run_checks(backend=False), "llm tool mode")

        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        text = _by_name(run_checks(backend=False), "llm tool mode")

        assert text.status == WARN
        assert '"text" is the mode to try' not in native.hint
        # The text branch may still point at native, which is the working one.
        assert "native" in text.hint

    def test_the_endpoint_note_survives_the_reasoning_wording(self, monkeypatch):
        """The provider-specific note is appended to both, not just the default one.

        ``native`` on a stock vLLM needs two server flags, and that is more
        relevant for a reasoning model, not less -- the endpoint is the only
        thing left to fix.
        """
        monkeypatch.setattr(settings, "llm_tool_mode", "native")
        monkeypatch.setattr(settings, "llm_provider", "vllm")
        monkeypatch.setattr(settings, "vllm_model", "deepseek-r1-distill-qwen-32b")
        check = _by_name(run_checks(backend=False), "llm tool mode")

        assert "--enable-auto-tool-choice" in check.hint
        assert 'cannot drive "text" mode' in check.hint

    def test_a_native_line_with_no_model_name_keeps_the_general_advice(self, monkeypatch):
        """No name is not evidence of a reasoning model, and must not be read as one."""
        monkeypatch.setattr(settings, "llm_tool_mode", "native")
        monkeypatch.setattr(settings, "llm_provider", "openai")
        monkeypatch.setattr(settings, "remote_llm_model", "")
        check = _by_name(run_checks(backend=False), "llm tool mode")

        assert '"text" is the mode to try' in check.hint


class TestStoreProbeAgreesWithTheReadPath:
    """Hold ``_store_defect`` to the thing it models, instead of to a checklist.

    Three review rounds found the same shape in that probe: a condition the
    read path enforces and the probe did not -- the chunk table, then the rest
    of the collection contract, then the width inside the registry. Patching
    the third one is not a fix for that pattern, because the pattern is that
    the probe *derives* queryability from conditions someone thought of while
    the read path *is* queryability.

    Asking the read path in production is not available at a doctor's price
    (sqlite-vec must be loaded, the connection is read-write, and a real query
    needs a real query vector -- see ``doctor._store_defect``). A test has none
    of those constraints, so the oracle lives here: for every state in
    ``STORE_DAMAGE``, run a genuine ``similarity_search`` and require the probe
    to have already said so.

    This is not decoration. Writing it is what turned up the non-integer
    registry width, which no review round had named.

    **And then it missed twice, in the same place, which is why the states are
    derived now.** The mechanism above was never wrong; its *input set* was.
    Review found no ``metadata`` state in one round and no ``content`` state in
    the next -- two columns of one ``SELECT``, the same omission both times,
    because the list of states was maintained beside the read path instead of
    taken from it. An oracle whose completeness rests on a hand-written list
    inherits that list's blind spots and reports agreement anyway, which is the
    worst failure available to a thing whose whole job is catching the probe
    out. So ``STORE_DAMAGE`` is now ``STRUCTURAL_DAMAGE`` plus a state per
    column ``_knn`` selects, and a column nobody has described raises rather
    than being skipped: see ``_derive_store_damage``.
    """

    @pytest.mark.parametrize("how", STORE_DAMAGE)
    def test_no_state_that_defeats_a_query_reads_as_healthy(self, monkeypatch, tmp_path: Path, how):
        """The direction that matters: a false ``healthy`` withholds the remedy.

        Getting this wrong tells someone whose index cannot answer that there
        is nothing to do, which is worse than the over-warning this probe
        replaced -- that at least erred noisily.
        """
        from aorta.chat.rag.embeddings.factory import get_provider

        index = _write_index(monkeypatch, tmp_path)
        _break_store(index, get_provider().collection_name(), how)

        failure = _read_path_answers(index)
        defect = _probe(index)
        if failure:
            assert defect, (
                f"retrieval refuses a {how} store with {failure!r}, and the probe called it healthy"
            )
        else:
            # Not skipped: a new damage state that leaves queries working is a
            # claim about the read path, and it should have to be made out loud
            # here rather than passing this test by not exercising it.
            assert how in {"vector-rows", "vector-rowids"}, (
                f"{how} leaves retrieval working, so this test asserts nothing about it; "
                "add it to the strictness test's expected list if that is intended"
            )

    def test_every_column_the_read_path_reads_has_its_damage_described(self):
        """The property that replaces remembering: no undescribed column.

        ``_derive_store_damage`` raises on a column ``COLUMN_DAMAGE`` has never
        heard of, so a column added to ``_knn``'s ``SELECT`` cannot arrive
        without states. That guard runs at import, which makes it loud but not
        legible -- this states the contract so it reads as intended rather than
        as an accident, and covers the direction the guard cannot: a described
        column the read path has stopped selecting, whose states would go on
        being swept while proving nothing.
        """
        columns = set(_read_path_columns())
        assert columns, "the read path selects nothing, which cannot be right"
        assert columns == set(COLUMN_DAMAGE), (
            f"the read path selects {sorted(columns)} and COLUMN_DAMAGE describes "
            f"{sorted(COLUMN_DAMAGE)}; the two have to be the same set, since a "
            "column missing here is unswept and one left over here is swept for "
            "nothing"
        )
        assert set(_derive_store_damage()) >= set(STRUCTURAL_DAMAGE)

    def test_the_columns_the_probe_requires_are_the_columns_the_read_path_reads(self):
        """The probe's own required-column list, held to the same source.

        ``_collection_schema_defect`` checks that ``id``, ``content`` and
        ``metadata`` exist. That list is hand-written too, and is the next one
        that would drift for exactly the reasons the state list did -- so pin
        it against the read path rather than against itself. ``id`` is expected
        to be absent from the ``SELECT``: it is the join key
        (``c.id = m.rowid``), read by the query without being selected by it,
        so its presence in the probe's list is correct and this test says why
        instead of tripping over it.
        """
        import inspect

        source = inspect.getsource(doctor._collection_schema_defect)
        required = re.search(r"for column in \((.+?)\) if column not in columns", source)
        assert required, "could not find the probe's required-column list"
        listed = {name.strip().strip("\"'") for name in required.group(1).split(",")}

        selected = set(_read_path_columns())
        assert listed >= selected - {"embedding"}, (
            f"the read path selects {sorted(selected)} off the chunk table but the "
            f"probe only requires {sorted(listed)} to exist"
        )
        assert listed - selected == {"id"}, (
            f"the probe requires {sorted(listed - selected)} which the read path does "
            "not select; only the join key id is expected to be in that position"
        )

    def test_a_healthy_store_satisfies_both(self, monkeypatch, tmp_path: Path):
        """The other end: neither side may be vacuously strict."""
        index = _write_index(monkeypatch, tmp_path)
        assert _read_path_answers(index) == ""
        assert _probe(index) == ""

    def test_the_only_state_the_probe_is_stricter_about_is_row_parity(
        self, monkeypatch, tmp_path: Path
    ):
        """The one deliberate disagreement, pinned so it stays the only one.

        A chunk with no vector beside it is unreachable through an inner join,
        but the rows that do have vectors still answer -- so retrieval does not
        refuse the index and the probe does. That is the safe direction (a
        warning over a half-built index) and it is why the agreement above is
        one-way rather than an equality. Enumerated here so that a second
        divergence has to be argued for rather than absorbed.
        """
        from aorta.chat.rag.embeddings.factory import get_provider

        stricter = []
        for how in STORE_DAMAGE:
            (tmp_path / how).mkdir()
            index = _write_index(monkeypatch, tmp_path / how)
            _break_store(index, get_provider().collection_name(), how)
            if not _read_path_answers(index) and _probe(index):
                stricter.append(how)

        assert stricter == ["vector-rows", "vector-rowids"]

    def test_metadata_that_is_not_a_json_object_is_caught(self, monkeypatch, tmp_path: Path):
        """The reported case: the column exists, its values defeat the read path.

        ``_knn`` calls ``json.loads`` on every metadata value it selects and
        hands the result to ``Document(metadata=...)``. The chunk count, the
        columns, the registry width and the vectors all still look right, so
        this reached the end of the probe and came back clean -- and the cache
        row then said "Nothing to do" for an index where every matching
        retrieval raises.
        """
        from aorta.chat.rag.embeddings.factory import get_provider

        for how in ("metadata-not-json", "metadata-not-an-object"):
            (tmp_path / how).mkdir()
            index = _write_index(monkeypatch, tmp_path / how)
            _break_store(index, get_provider().collection_name(), how)

            defect = _probe(index)
            assert "not a JSON object" in defect, how
            # Counted, not just detected: one bad row out of many is still a
            # query that raises, and which row is chosen by the query vector.
            assert defect.split()[0].isdigit(), how

    def test_content_that_is_not_text_is_caught(self, monkeypatch, tmp_path: Path):
        """The metadata finding's neighbour, one column over in the same SELECT.

        ``_knn`` hands ``content`` to ``Document(page_content=...)``, which
        requires a string, so a NULL or a number raises there exactly as bad
        metadata does one argument later. This is the state that showed the
        damage table was the problem rather than any one missing check: the
        metadata round had just been closed, and the same omission was sitting
        in the adjacent column.
        """
        from aorta.chat.rag.embeddings.factory import get_provider

        for how in ("content-null", "content-not-text"):
            (tmp_path / how).mkdir()
            index = _write_index(monkeypatch, tmp_path / how)
            _break_store(index, get_provider().collection_name(), how)

            defect = _probe(index)
            assert "no text in the content column" in defect, how
            assert defect.split()[0].isdigit(), how

    def test_a_blob_in_the_content_column_is_not_a_defect(self, monkeypatch, tmp_path: Path):
        """The line between "not text" and "unreadable", measured not guessed.

        ``Document`` decodes a BLOB and accepts it, so flagging one would fail
        a store that retrieves perfectly well -- the over-warning this probe
        was written to replace. This is why the check asks ``typeof`` for the
        two types that survive rather than testing for ``text``.
        """
        from aorta.chat.rag.embeddings.factory import get_provider

        index = _write_index(monkeypatch, tmp_path)
        collection = get_provider().collection_name()
        conn = _vec_connection(index)
        try:
            conn.execute(f'UPDATE "chunks_{collection}" SET content = CAST(content AS BLOB)')
            conn.commit()
        finally:
            conn.close()

        assert _probe(index) == ""
        assert not _read_path_answers(index)

    def test_the_metadata_check_falls_back_rather_than_passing(self, monkeypatch, tmp_path: Path):
        """A SQLite without JSON1 must not turn the check into a no-op.

        The predicate is duplicated in Python for that build, so the two are
        pinned against each other here: same verdict for every value the SQL
        branch classifies.
        """
        conn = _vec_connection(_write_index(monkeypatch, tmp_path))
        try:
            for value, is_object in (
                ('{"source": "a.py"}', True),
                ("not json at all", False),
                ("123", False),
                ('"a string"', False),
                ("[]", False),
                (None, False),
            ):
                sql = conn.execute(
                    "SELECT CASE WHEN metadata IS NOT NULL AND json_valid(metadata) "
                    "THEN json_type(metadata) ELSE 'null' END = 'object' "
                    "FROM (SELECT ? AS metadata)",
                    (value,),
                ).fetchone()[0]
                python = isinstance(doctor._loads_or_none(value), dict)
                assert bool(sql) == python == is_object, value
        finally:
            conn.close()

    def test_the_registry_width_is_checked_against_the_manifest(self, monkeypatch, tmp_path: Path):
        """The reported case: 999 in the registry over a 384-dimension index.

        Every other part of the contract holds -- registered collection, the
        columns, the vector table, matching row counts -- so this is the state
        that reaches the end of the probe and used to come back clean.
        """
        from aorta.chat.rag.embeddings.factory import get_provider

        index = _write_index(monkeypatch, tmp_path)
        _break_store(index, get_provider().collection_name(), "registry-width")

        defect = _probe(index)
        assert "999" in defect and "384" in defect
        # And the real reason, in the report's own words rather than sqlite's.
        assert "refuses every 384-dimension search" in defect

    def test_a_width_that_is_not_a_number_is_caught_too(self, monkeypatch, tmp_path: Path):
        """Found by the oracle above, not by review.

        ``_knn`` coerces the registry value with ``int()`` before it compares
        anything, so a width of ``'wide'`` raises just as surely as a
        mismatched one -- and a check written only for the reported case would
        have been the fourth round of this.
        """
        from aorta.chat.rag.embeddings.factory import get_provider

        index = _write_index(monkeypatch, tmp_path)
        _break_store(index, get_provider().collection_name(), "registry-width-text")

        assert "not a number" in _probe(index)

    def test_an_unknown_manifest_width_still_checks_what_it_can(self, monkeypatch, tmp_path: Path):
        """``0`` means "the manifest does not say", which is not "anything goes".

        The comparison is skipped, because there is nothing to compare against.
        The coercion is not, because ``_knn`` performs it either way.
        """
        from aorta.chat.rag.embeddings.factory import get_provider

        collection = get_provider().collection_name()
        index = _write_index(monkeypatch, tmp_path)

        _break_store(index, collection, "registry-width")
        assert doctor._store_defect(index, 0) == ""
        _break_store(index, collection, "registry-width-text")
        assert doctor._store_defect(index, 0)

    def test_row_id_alignment_is_checked_not_just_row_count(self, monkeypatch, tmp_path: Path):
        from aorta.chat.rag.embeddings.factory import get_provider

        index = _write_index(monkeypatch, tmp_path)
        _break_store(index, get_provider().collection_name(), "vector-rowids")
        assert "row IDs disagree" in _probe(index)


class TestRemoteEmbeddingProfile:
    """The migration gap raised on #462, which lands here rather than there.

    ``config.write_profile`` runs only on ``config init``, so flipping the
    profile templates to local embeddings cannot reach a ``chat.toml`` that
    already exists. Anyone who ran ``config init`` with a remote-LLM profile
    before that change keeps ``embedding_provider = "remote"`` on disk and gets
    no signal at all: the published index cannot be read that way, so
    ``index fetch`` refuses, and this PR's ``remedy_lines`` work stops
    *offering* the fetch without ever saying that flipping to local is what
    brings it back.

    The hard part is not saying it. It is not saying it to the install that
    chose a remote embedder on purpose and built an index to match -- which is
    correct, and for which "flip to local" is wrong advice. So the trigger is
    index health, never the provider alone.
    """

    def _remote(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "embedding_provider", "remote")
        monkeypatch.setattr(settings, "remote_embedding_api_key", "sk-test")
        monkeypatch.setattr(settings, "remote_embedding_model", "text-embedding-3-small")
        monkeypatch.setattr(settings, "remote_embedding_base_url", "")

    def test_a_remote_profile_with_no_index_is_told_what_to_change(self, monkeypatch):
        """The stuck state: a fetch that refuses, and nothing naming the one line."""
        self._remote(monkeypatch)
        check = _by_name(run_checks(backend=False), "embedding profile")

        assert check.status == WARN
        assert "remote embeddings" in check.detail
        # The reviewer's two requirements: name the incompatibility, name the edit.
        assert "published" in check.hint
        assert 'embedding_provider = "local"' in check.hint
        assert 'embedding_provider = "local"' in check.procedure
        assert "AORTA_CHAT_EMBEDDING_PROVIDER=local" in check.procedure

    def test_it_names_the_profile_rewrite_as_the_alternative(self, monkeypatch):
        """``config init --force`` fixes it, and nobody knows to run it."""
        self._remote(monkeypatch)
        check = _by_name(run_checks(backend=False), "embedding profile")

        assert "aorta chat config init --force" in check.procedure
        # And says what it costs, since it discards hand edits and re-prompts.
        assert "discards" in check.procedure

    def test_it_says_where_the_setting_probably_came_from(self, monkeypatch):
        """Migration advice, not a scolding: the template wrote it, not the user.

        This is the sentence that makes the check worth a row of its own.
        ``remedy_lines`` can name the setting; only this can say where it came
        from and that nothing on any normal path will rewrite it.

        This wording was merge-order-neutral while #462 was unmerged: it said
        the setting persists "whatever the current template writes", claiming
        nothing about which way they read, because either tense would have been
        wrong in one of the two trees a reader might have checked out. #462 is
        now an ancestor of this branch, so the hedge has nothing left to
        protect against and the definite statement is the more useful one --
        pinned by ``test_the_claim_about_the_templates_matches_the_templates``.

        What survives the hedge is the part that was never about merge order:
        an existing ``chat.toml`` is not rewritten, so a profile written by an
        older install still carries the setting however the templates read
        today. That is the population this row exists for, and it grew rather
        than shrank when the templates changed.
        """
        self._remote(monkeypatch)
        procedure = _by_name(run_checks(backend=False), "embedding profile").procedure

        assert "where the setting usually comes from" in procedure
        assert "write_profile runs" in procedure
        assert "edited or regenerated" in procedure
        assert "written by an older install" in procedure

    def test_the_claim_about_the_templates_matches_the_templates(self, monkeypatch):
        """The fact the old wording got wrong, read rather than asserted.

        The procedure now states as fact that no template selects a remote
        embedder, which is only safe while it is true, and the templates are a
        sibling PR's to change. So read them and hold the sentence to what they
        say, in both directions: dropping the claim fails this too, because a
        tree where it is true should say so.

        The read has to reach the assertion to be worth anything. The version
        of this test written before the rebase computed the same list and spent
        it on the failure message, asserting only ``"current template" in
        procedure`` -- which the ``config init --force ... from the current
        template`` remedy line satisfies by itself. It passed against a
        procedure claiming every template still wrote ``remote``, on a tree
        where none did, which is the whole fact it was named for. A fixture or
        a read that does not feed an assertion is decoration.
        """
        from aorta.chat import config as config_mod

        remote = sorted(
            name
            for name, template in config_mod.PROFILE_TEMPLATES.items()
            if template.get("embedding_provider") == "remote"
        )
        self._remote(monkeypatch)
        procedure = _by_name(run_checks(backend=False), "embedding profile").procedure
        flattened = " ".join(procedure.split())

        claims_none_do = "no profile template selects a remote embedder any more" in flattened
        assert claims_none_do == (not remote), (
            f"the procedure claims no template selects a remote embedder: "
            f"{claims_none_do}; templates that actually do: {remote or 'none'}"
        )

    def test_it_does_not_fire_on_a_remote_profile_with_an_index_to_match(
        self, monkeypatch, tmp_path: Path
    ):
        """The test this check exists to pass, and the one that makes it safe.

        A deliberate remote embedder with an index built by that embedder is a
        correct setup. Telling it to flip to local would discard a working
        index and be wrong on the facts. Conditioning on the provider alone
        would do exactly that -- and a warning on a correct setup is the defect
        this whole PR exists to remove, so recreating one here would be
        self-defeating.
        """
        self._remote(monkeypatch)
        from aorta.chat.rag.embeddings.factory import get_provider

        _write_index(monkeypatch, tmp_path, embedding_model=get_provider().model_id())

        report = run_checks(backend=False)
        assert _by_name(report, "index manifest").status == OK
        assert not [check for check in report.checks if check.name == "embedding profile"]

    def test_it_does_not_fire_on_a_local_profile(self, monkeypatch, tmp_path: Path):
        """Nothing to migrate; ``index fetch`` is offered and works."""
        _write_index(monkeypatch, tmp_path)
        report = run_checks(backend=False)
        assert not [check for check in report.checks if check.name == "embedding profile"]

    def test_a_remote_profile_over_a_refused_index_is_told_too(self, monkeypatch, tmp_path: Path):
        """An index present but built by something else is the same dead end."""
        self._remote(monkeypatch)
        _write_index(monkeypatch, tmp_path, embedding_model="somebody/else")

        report = run_checks(backend=False)
        assert _by_name(report, "index manifest").status == FAIL
        assert _by_name(report, "embedding profile").status == WARN

    @pytest.mark.parametrize("how", STORE_DAMAGE)
    def test_a_remote_profile_over_a_clobbered_store_is_told_too(
        self, monkeypatch, tmp_path: Path, how
    ):
        """Index health is the trigger, so every unusable state reaches it.

        The message points at the index checks rather than claiming the
        provider is the whole fault, because for these states a rebuild is the
        proportionate fix and flipping to local is merely *a* fix.
        """
        self._remote(monkeypatch)
        from aorta.chat.rag.embeddings.factory import get_provider

        collection = get_provider().collection_name()
        _write_index(monkeypatch, tmp_path, embedding_model=get_provider().model_id())
        _break_store(settings.index_file, collection, how)

        check = _by_name(run_checks(backend=False), "embedding profile")
        assert check.status == WARN
        assert "index checks below" in check.procedure

    def test_it_composes_with_the_remedies_rather_than_repeating_them(self, monkeypatch):
        """Two halves of one diagnosis, on two rows, saying different things.

        ``remedy_lines`` explains why the fetch is absent from the remedy list;
        this explains where the setting came from and what to change. Rendering
        the first draft showed both reciting the CI-publishes-one-asset
        argument forty lines apart, which is how a report teaches people to
        skim it -- the failure this batch exists to fix. So the split is
        asserted, not left to whoever reads the two strings next.
        """
        self._remote(monkeypatch)
        report = run_checks(backend=False)

        index = _by_name(report, "chat index")
        assert index.status == FAIL
        # The index row still owns the fetch argument, and still withholds the
        # fetch itself.
        assert "aorta chat index fetch     download" not in index.hint
        assert "no published asset can match a remote one" in index.hint

        profile = _by_name(report, "embedding profile")
        assert "no published asset can match a remote one" not in profile.procedure
        assert "CI publishes" not in profile.procedure
        # What each row uniquely carries.
        assert "write_profile runs" not in index.hint
        assert 'embedding_provider = "local"' in profile.hint

    def test_a_keyless_remote_profile_still_gets_the_migration_advice(self, monkeypatch):
        """The population this whole check was written for, and the one it nearly lost.

        No profile template has ever prompted for ``remote_embedding_api_key``
        and it defaults to empty with no fallback to the chat key, so a
        pre-#462 remote profile is keyless almost by definition. Reporting the
        provider failure and returning -- which is what the first version of
        the buildability check did -- silenced this row for exactly them, and
        left them reading "set the key" over a problem the key does not fix.
        """
        self._remote(monkeypatch)
        monkeypatch.setattr(settings, "remote_embedding_api_key", "")
        report = run_checks(backend=False)

        assert _by_name(report, "embedding provider").status == FAIL
        profile = _by_name(report, "embedding profile")
        assert profile.status == WARN
        assert 'embedding_provider = "local"' in profile.procedure

    def test_a_keyless_profile_is_not_told_to_build_the_index_locally(self, monkeypatch):
        """The remedy the missing key guarantees will fail.

        Review raised the same gap from #462's side. Every chunk of the corpus
        goes through the embeddings API, so with no key ``index build`` fails
        on the first one -- it is the worst possible thing to recommend here,
        because it looks like work and is not.
        """
        self._remote(monkeypatch)
        monkeypatch.setattr(settings, "remote_embedding_api_key", "")
        procedure = _by_name(run_checks(backend=False), "embedding profile").procedure

        assert "keep it and build the index" not in procedure
        assert "would fail on the first chunk" in procedure
        # And says why the key was never there, so "not set" does not read as
        # something the user did.
        assert "no profile template" in procedure

    def test_a_keyed_remote_profile_is_still_told_it_can_build(self, monkeypatch):
        """The other branch: with a key, building locally is a real option."""
        self._remote(monkeypatch)
        report = run_checks(backend=False)

        assert _by_name(report, "embedding provider").status == OK
        procedure = _by_name(report, "embedding profile").procedure
        assert "keep it and build the index" in procedure
        assert "would fail on the first chunk" not in procedure

    def test_the_two_closing_paragraphs_are_mutually_exclusive(self, monkeypatch):
        """One tail or the other, never both and never neither.

        They give opposite advice about the same command, so a change that
        concatenated them would be worse than either alone.
        """
        for key in ("sk-test", ""):
            self._remote(monkeypatch)
            monkeypatch.setattr(settings, "remote_embedding_api_key", key)
            procedure = _by_name(run_checks(backend=False), "embedding profile").procedure
            builds = "keep it and build the index" in procedure
            needs_key = "would fail on the first chunk" in procedure
            assert builds != needs_key, (key, procedure)

    def test_a_raising_index_probe_leaves_the_provider_row_alone(self, monkeypatch, tmp_path: Path):
        """Measured, not assumed: letting it escape put two rows under one name.

        ``run_checks`` labels a raising check with the label it was registered
        under, which for this one is ``embedding provider`` -- so a broken
        sqlite-vec produced an ``ok`` provider row *and* a ``fail`` provider
        row in the same report, blaming the embedder for a sqlite fault.
        ``_check_index`` reports that exception a row later under its own name,
        so abstaining here loses no information and costs one wrong label.
        """
        self._remote(monkeypatch)
        from aorta.chat.rag.embeddings.factory import get_provider

        _write_index(monkeypatch, tmp_path, embedding_model=get_provider().model_id())

        def _explode(*args, **kwargs):
            raise ImportError("sqlite-vec extension not loadable")

        monkeypatch.setattr(doctor, "_store_defect", _explode)
        report = run_checks(backend=False)

        provider_rows = [check for check in report.checks if check.name == "embedding provider"]
        assert [check.status for check in provider_rows] == [OK]
        assert not [check for check in report.checks if check.name == "embedding profile"]
        # Reported once, under the name it belongs to.
        assert [check.status for check in report.checks if check.name == "chat index"] == [OK, FAIL]

    def test_it_is_a_warning_because_the_profile_is_valid_not_broken(self, monkeypatch):
        """Which is why this is not ``config validate``'s job.

        A remote embedding profile is valid and incompatible with the published
        index. Those are different statements, and only the second one is a
        doctor's to make.
        """
        self._remote(monkeypatch)
        assert _by_name(run_checks(backend=False), "embedding profile").status == WARN
        # The provider itself works, and says so on its own line.
        assert _by_name(run_checks(backend=False), "embedding provider").status == OK


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

    def test_the_cost_is_not_described_as_billed_on_a_self_hosted_model(self, monkeypatch):
        """The same branch fires for a locally served reasoning model.

        Its own docstring says so -- the channel is the model's, not the
        endpoint's -- so a stock vLLM serving ``DeepSeek-R1`` reaches this
        warning, and there is nobody billing it. "billed rounds" made the
        newly added local-vLLM diagnosis read as somebody else's problem.
        """
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "vllm")
        monkeypatch.setattr(settings, "vllm_model", "deepseek-ai/DeepSeek-R1")
        check = _by_name(run_checks(backend=False), "llm tool mode")

        assert check.status == WARN
        assert "inference rounds" in check.hint
        assert "billed rounds" not in check.hint

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

        assert "inference rounds" in hint
        assert "wasted" in hint
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


#: The model CI publishes the one index asset with. An install that queries
#: with anything else cannot read it, which is what makes the fetch remedy
#: conditional on more than the provider.
DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"


def _fetch_would_be_accepted() -> bool:
    """Whether ``aorta chat index fetch`` would install, given resolved settings.

    Runs the comparison ``fetch_index`` runs -- ``validate`` of the published
    manifest against this install's provider identity, then the refusal gate --
    rather than asserting anything about it. Offline: no asset is downloaded,
    the published manifest is described from the default-model provider, and
    ``validate`` only reads fields.

    Note what is deliberately *not* consulted: ``chunk_size`` and
    ``chunk_overlap``. Drift in those is a warning in ``validate`` and
    ``fetch_index`` installs through it (`manifest.py:376`), so treating them
    as blocking here would make this oracle demand that a working fetch be
    withheld.
    """
    from aorta.chat.config import settings
    from aorta.chat.rag import manifest as manifest_mod
    from aorta.chat.rag.embeddings.factory import get_provider

    resolved_model = settings.embedding_model
    resolved_provider = settings.embedding_provider
    try:
        settings.embedding_model = DEFAULT_LOCAL_MODEL
        settings.embedding_provider = "local"
        publisher = get_provider()
        published = manifest_mod.Manifest.from_dict(
            {
                "schema_version": manifest_mod.SCHEMA_VERSION,
                "embedding_model": publisher.model_id(),
                "embedding_provider": "local",
                "embedding_identity": publisher.vector_identity(),
                "collection": publisher.collection_name(),
                "index_sha256": "0" * 64,
                "dimensions": 384,
                "chunk_size": 512,
                "chunk_overlap": 50,
                "aorta_version": "0.2.1",
                "aorta_sha": "",
                "chunks": 10,
            }
        )
    finally:
        # Restored from what was actually set, not from a module-level record
        # of it: this runs inside a monkeypatched state and a second source of
        # truth for it is one that can disagree.
        settings.embedding_model = resolved_model
        settings.embedding_provider = resolved_provider

    try:
        provider = get_provider()
    except Exception:
        # ``fetch_index`` resolves the provider before it reads anything, so a
        # provider that cannot be built is a fetch that cannot run. Failing
        # closed here rather than raising keeps the tier's verdict a verdict.
        return False
    report = manifest_mod.validate(
        published,
        embedding_model=provider.model_id(),
        collection=provider.collection_name(),
        embedding_identity=provider.vector_identity(),
    )
    return not report.refusals


def _build_would_start() -> bool:
    """Whether ``aorta chat index build`` would get past its provider, offline.

    A local build always can -- the weights download if absent, and a missing
    ``fastembed`` is reported by the extras row with a command that does run. A
    remote build cannot until its client builds, which is the precondition
    ``RemoteApiProvider.get_embeddings`` enforces before sending anything.
    """
    from aorta.chat.rag import manifest as manifest_mod

    # Asked before the provider is classified, because
    # ``_configured_embedding_provider`` answers "local" for a name it could
    # not resolve -- which would score an unbuildable provider as a build that
    # starts, the exact reading this state exists to reject.
    if manifest_mod._unknown_embedding_provider():
        return False
    if manifest_mod._configured_embedding_provider() == "local":
        return True
    return not manifest_mod._remote_embedder_error()


#: What each index command's outcome is asked of. Keyed by the command path the
#: extractor produces, so an arm that starts offering a third index command
#: fails the completeness assertion below rather than going unchecked.
_OUTCOME_ORACLES = {
    ("index", "fetch"): _fetch_would_be_accepted,
    ("index", "build"): _build_would_start,
}

# The placeholders the report's commands are allowed to contain, and a real
# value for each. A new one makes the sweep below fail rather than skip: the
# point is that every command is parsed, so an unregistered placeholder has to
# be a deliberate decision by whoever added it.
_PLACEHOLDERS = {"<name>": "openai"}


def _commands_named_in(text: str) -> list[tuple[list[str], list[str], bool]]:
    """Every ``aorta chat ...`` invocation in ``text``, as ``(path, args, offered)``.

    The report prints commands with their explanation on the same line, so
    tokens are consumed greedily through the command path and then only while
    they are options or an option's value. Prose does not start with ``--``,
    which is where each command ends.

    ``offered`` distinguishes the two ways the report names a command, because
    they carry different promises. A command that begins its own line is being
    offered to be typed, so it has to run exactly as printed. A command quoted
    inside a sentence may be referring to a mechanism rather than proposing it
    -- the migration procedure explains what ``'aorta chat config init'`` did
    to profiles in the past -- so it must name something real, but need not
    carry every option a run would need.
    """
    import re

    from aorta.cli.chat import chat as chat_group

    found = []
    for match in re.finditer(r"aorta chat ([^\n']*)", text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        offered = text[line_start : match.start()].strip() == ""
        tokens = match.group(1).split()
        command, path = chat_group, []
        while tokens and not tokens[0].startswith("-"):
            nxt = command.get_command(None, tokens[0]) if hasattr(command, "get_command") else None
            if nxt is None:
                break
            path.append(tokens.pop(0))
            command = nxt
        if not path:
            continue
        args = []
        params = {opt: p for p in getattr(command, "params", []) for opt in p.opts}
        while tokens and tokens[0].startswith("--"):
            option = tokens.pop(0)
            args.append(option)
            param = params.get(option)
            if param is not None and not getattr(param, "is_flag", False) and tokens:
                raw = tokens.pop(0)
                args.append(_PLACEHOLDERS.get(raw, raw))
        found.append((path, args, offered))
    return found


def _resolve(path: list[str]):
    """The Click command at ``path``, and the group holding it."""
    from aorta.cli.chat import chat as chat_group

    command, parent = chat_group, None
    for part in path:
        parent = command
        command = command.get_command(None, part)
    return command, parent


class TestEveryCommandTheReportNamesCanRun:
    """The closed-set answer to "is this advice followable?".

    Five separate findings in this area were each a command that could not run
    in the state that printed it: ``index fetch`` offered to a remote embedder,
    ``index build`` offered on an empty ``remote_embedding_api_key``, ``config
    init --force`` printed without the ``--profile`` Click requires, ``index
    fetch`` offered under a customised ``embedding_model``, and *both* index
    commands offered under an ``embedding_provider`` the factory does not have.
    Every one was found by hand, one review round apart. So the question is
    asked here of every arm at once instead, and asked of the code that would
    reject the line rather than of a reader's judgement.

    Two tiers, because there are two ways a command fails. Click's parser
    rejects a line that is not a valid invocation -- the ``--profile`` case.
    It cannot reject one that parses and is then refused at runtime, which is
    the fourth case and was invisible to the first version of this sweep. That
    is what ``test_every_offered_index_command_would_be_accepted`` is for.
    """

    #: The resolved states the sweep runs under. Each entry forks at least one
    #: arm. ``embedding_model`` is here because leaving it out is what let a
    #: real finding through: the parser tier can never reject ``index fetch``,
    #: so a state whose only defect is that the fetch would be *refused* is
    #: invisible without both the state and the outcome tier below.
    STATES = (
        ("local", "", "text", "gpt-4o", DEFAULT_LOCAL_MODEL),
        ("remote", "", "native", "gpt-oss-120b", DEFAULT_LOCAL_MODEL),
        ("remote", "sk-test", "native", "gpt-4o", DEFAULT_LOCAL_MODEL),
        ("local", "", "sideways", "gpt-4o", DEFAULT_LOCAL_MODEL),
        ("local", "", "text", "gpt-4o", "BAAI/bge-base-en-v1.5"),
        # Whitespace-padded default. Reads as the published model and is not
        # one: every identity the fetch is validated against is built from
        # this string verbatim.
        ("local", "", "text", "gpt-4o", "  BAAI/bge-small-en-v1.5  "),
        # A provider aorta does not have. Neither index command can start, so
        # this is the state that proves the tier can reject *both* oracles
        # rather than only choosing between them.
        ("sbert", "", "text", "gpt-4o", DEFAULT_LOCAL_MODEL),
    )

    def _texts(self, monkeypatch, tmp_path: Path) -> list[str]:
        """Every hint and procedure the report can produce, across resolved states.

        Driven rather than read off the source, so advice composed by f-string
        is included, and swept across the settings that select different arms:
        the embedding provider, the embedding key, and the tool mode each fork
        the wording, and the arms that differ are exactly the ones at issue.
        """
        from aorta.chat.rag import manifest as manifest_mod

        texts = []
        for provider, key, mode, model, embedding_model in self.STATES:
            # ``_configured_embedding_provider`` is deliberately *not* stubbed.
            # It used to be, for determinism, and that stub is what hid the
            # unresolvable-provider arm: it answered "sbert" where the real
            # resolver answers "local", so the sweep scored a state that
            # cannot occur -- a remote provider named sbert -- and never saw
            # the local arm the setting actually reaches. The resolver is one
            # of the things under test here, so it runs.
            monkeypatch.setattr(settings, "embedding_provider", provider)
            monkeypatch.setattr(settings, "remote_embedding_api_key", key)
            monkeypatch.setattr(settings, "llm_tool_mode", mode)
            monkeypatch.setattr(settings, "vllm_model", model)
            monkeypatch.setattr(settings, "remote_llm_model", model)
            monkeypatch.setattr(settings, "embedding_model", embedding_model)
            monkeypatch.setattr(settings, "index_path", str(tmp_path / "absent.sqlite"))
            for check in run_checks(backend=False).checks:
                texts += [check.hint, check.procedure, check.detail]
            texts += manifest_mod.remedy_lines()
            texts.append(manifest_mod._refresh_advice())
        return [t for t in texts if t]

    def test_every_offered_command_parses_as_printed(self, monkeypatch, tmp_path: Path):
        from click import Context

        offered_seen = 0
        for text in self._texts(monkeypatch, tmp_path):
            for path, args, offered in _commands_named_in(text):
                command, parent = _resolve(path)
                assert command is not None, path
                if not offered:
                    continue
                offered_seen += 1
                # A group prints help rather than doing the thing, so offering
                # one is advice that does not act -- and it is what a command
                # name broken across a line wrap looks like to the extractor.
                assert not hasattr(command, "get_command"), f"{path} is a group, not a command"
                with Context(parent) as ctx:
                    # Parses and validates required options without running the
                    # callback: a missing required option raises here, which is
                    # what a user typing the printed line would get.
                    command.make_context(" ".join(path), list(args), parent=ctx)
        assert offered_seen > 8, f"only {offered_seen} offered commands; the extractor has broken"

    def test_a_quoted_mention_may_omit_options_but_must_name_something_real(
        self, monkeypatch, tmp_path: Path
    ):
        """The weaker promise, still checked -- a renamed command breaks both tiers."""
        from click import Context, MissingParameter

        quoted = 0
        for text in self._texts(monkeypatch, tmp_path):
            for path, args, offered in _commands_named_in(text):
                if offered:
                    continue
                command, parent = _resolve(path)
                assert command is not None, path
                quoted += 1
                if hasattr(command, "get_command"):
                    continue  # A bare group named in prose is a noun, not advice.
                try:
                    with Context(parent) as ctx:
                        command.make_context(" ".join(path), list(args), parent=ctx)
                except MissingParameter:
                    pass  # Allowed here, and only here.
        assert quoted, "no quoted mentions found; the extractor has broken"

    def test_every_offered_index_command_would_be_accepted(self, monkeypatch, tmp_path: Path):
        """The tier the parser cannot provide, and the hole a real finding fell through.

        Parsing answers "is this a command", not "will it work". ``aorta chat
        index fetch`` parses in every state, so the sweep's first tier could
        never have rejected it -- and a customised ``embedding_model`` makes
        ``fetch_index`` refuse the published asset on the embedding model, the
        collection and the embedding identity. That arm was offered anyway
        until this round, and no amount of extra parsing would have found it.

        So each index command the report *offers* is also asked of the code
        that would run it, per resolved state. This is the same question the
        third column of the enumeration asks, made mechanical.
        """
        for path, args, offered in self._offered_index_commands(monkeypatch, tmp_path):
            assert not args, f"{path} offered with unexpected options {args}"
            assert offered
            oracle = _OUTCOME_ORACLES[tuple(path)]
            assert oracle(), (
                f"the report offers 'aorta chat {' '.join(path)}' in a state where it "
                f"cannot succeed (embedding_provider={settings.embedding_provider!r}, "
                f"embedding_model={settings.embedding_model!r}, "
                f"key={'set' if settings.remote_embedding_api_key else 'empty'})"
            )

    def _offered_index_commands(self, monkeypatch, tmp_path: Path):
        """Offered commands that have an outcome oracle, with the state still applied.

        Yields inside the state loop rather than collecting first, because the
        oracles read ``settings`` and a collected list would be scored against
        whichever state happened to be last.
        """
        from aorta.chat.rag import manifest as manifest_mod

        for provider, key, mode, model, embedding_model in self.STATES:
            # ``_configured_embedding_provider`` is deliberately *not* stubbed.
            # It used to be, for determinism, and that stub is what hid the
            # unresolvable-provider arm: it answered "sbert" where the real
            # resolver answers "local", so the sweep scored a state that
            # cannot occur -- a remote provider named sbert -- and never saw
            # the local arm the setting actually reaches. The resolver is one
            # of the things under test here, so it runs.
            monkeypatch.setattr(settings, "embedding_provider", provider)
            monkeypatch.setattr(settings, "remote_embedding_api_key", key)
            monkeypatch.setattr(settings, "llm_tool_mode", mode)
            monkeypatch.setattr(settings, "vllm_model", model)
            monkeypatch.setattr(settings, "remote_llm_model", model)
            monkeypatch.setattr(settings, "embedding_model", embedding_model)
            monkeypatch.setattr(settings, "index_path", str(tmp_path / "absent.sqlite"))

            texts = []
            for check in run_checks(backend=False).checks:
                texts += [check.hint, check.procedure, check.detail]
            texts += manifest_mod.remedy_lines()
            texts.append(manifest_mod._refresh_advice())
            for text in texts:
                for path, args, offered in _commands_named_in(text or ""):
                    if offered and tuple(path) in _OUTCOME_ORACLES:
                        yield path, args, offered

    def test_both_oracles_refuse_an_unresolvable_provider(self, monkeypatch):
        """The oracles' own contract, pinned where the sweep cannot pin it.

        In the sweep both are only reached through an arm that offers the
        command, and the fetch oracle happens to be scored first -- so the
        build oracle's guard would go unexercised by any mutation of the
        production code. It is not redundant: ``_configured_embedding_provider``
        answers "local" for a name it could not resolve, so without the guard
        the build oracle would score an unbuildable provider as a build that
        starts, and a future arm that offers ``index build`` alone would pass
        unchecked.
        """
        monkeypatch.setattr(settings, "embedding_provider", "sbert")
        assert not _fetch_would_be_accepted()
        assert not _build_would_start()

    def test_every_index_command_the_report_offers_has_an_oracle(self, monkeypatch, tmp_path: Path):
        """Completeness, so the tier above cannot be satisfied by checking nothing.

        An arm that starts offering a third index command -- ``index eval``,
        say -- must fail here rather than pass unexamined.
        """
        offered = set()
        for text in self._texts(monkeypatch, tmp_path):
            for path, _, is_offered in _commands_named_in(text):
                if is_offered and path[:1] == ["index"]:
                    offered.add(tuple(path))
        assert offered, "no index commands offered anywhere; the extractor has broken"
        assert offered <= set(_OUTCOME_ORACLES), (
            f"no outcome oracle for {offered - set(_OUTCOME_ORACLES)}"
        )

    def test_the_sweep_would_catch_a_missing_required_option(self):
        """Without this the sweep could pass by never finding a failure to catch.

        ``config init`` requires ``--profile``, and the migration procedure
        offered it without one until this round. Pinning the rejection keeps
        the strict tier honest about what it rules out.
        """
        from click import Context, UsageError

        config, _ = _resolve(["config"])
        init, _ = _resolve(["config", "init"])
        with Context(config) as ctx, pytest.raises(UsageError):
            init.make_context("init", ["--force"], parent=ctx)

    def test_the_migration_procedure_offers_the_profile_option(self):
        """The finding itself, pinned where it was printed."""
        offered = [
            args
            for path, args, is_offered in _commands_named_in(doctor._REMOTE_EMBEDDING_MIGRATION)
            if path == ["config", "init"] and is_offered
        ]
        assert offered, "the migration procedure no longer offers 'config init'"
        assert all("--profile" in args for args in offered)
