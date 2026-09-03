"""Retriever over the persisted sqlite-vec index, and the store it reads.

``SqliteVecStore`` lives here rather than in ``indexer.py`` because this is the
module every query path already reaches through -- ``graph.nodes`` and
``tools.search`` both import it -- and Phase 4 moves index building out behind
``aorta chat index build``. The reader must not have to import the builder.
"""

from __future__ import annotations

import json
import logging
import math
import re
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from aorta.chat.config import settings
from aorta.chat.rag.embeddings.base import EmbeddingProvider
from aorta.chat.rag.embeddings.factory import get_provider
from aorta.chat.rag.sqlite_compat import ensure_loadable_extensions, ensure_modern_sqlite

if TYPE_CHECKING:  # ``sqlite3`` is imported per call, after the guards run.
    import sqlite3

logger = logging.getLogger(__name__)

#: One row per collection, recording the dimension its vectors were written at.
#: sqlite-vec fixes a vec0 table's dimension in its CREATE statement, so two
#: embedding providers cannot share one table and the collection has to be part
#: of the *table names* rather than a column. This registry is what turns a
#: provider switch into a named error instead of an OperationalError raised from
#: inside the extension, and it is how the reader discovers a collection exists.
_REGISTRY_TABLE = "aorta_collections"

#: Table names are interpolated, not bound, so the collection has to be an
#: identifier and not merely a string. Both providers already produce this shape.
_SAFE_COLLECTION = re.compile(r"\A[A-Za-z0-9_]+\Z")

_MISSING_INDEX = (
    "No chat index at {path}.\n"
    "  aorta chat index fetch     download the prebuilt index for this version\n"
    "  aorta chat index build     build one from the code on this machine\n"
    "  aorta chat doctor          check what else is missing first"
)

_MISSING_COLLECTION = (
    "The chat index at {path} holds no collection named {collection!r}, which is "
    "what the configured embedding provider reads ({provider}). Either re-index "
    "with this provider, or set AORTA_CHAT_EMBEDDING_PROVIDER back to the one "
    "that built the index. Collections present: {present}."
)


def _serialise(vector: list[float]) -> bytes:
    """Pack a vector the way sqlite-vec reads it: little-endian float32."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _deserialise(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _mmr(
    query_vector: list[float],
    candidates: list[list[float]],
    k: int,
    lambda_mult: float,
) -> list[int]:
    """Greedy maximal marginal relevance, returning candidate indices best-first.

    sqlite-vec ranks by distance alone, so the diversity half of what Chroma's
    ``search_type="mmr"`` did has to happen here. Brute force is the right shape:
    fetch_k is 30 by default and the vectors are already in memory, so this is
    microseconds and costs the chat-cli extra no numpy dependency.
    """
    if k <= 0 or not candidates:
        return []

    query = _normalise(query_vector)
    vectors = [_normalise(vector) for vector in candidates]
    relevance = [_dot(query, vector) for vector in vectors]

    selected = [max(range(len(vectors)), key=relevance.__getitem__)]
    while len(selected) < min(k, len(vectors)):
        best_index = -1
        best_score = -math.inf
        for index, vector in enumerate(vectors):
            if index in selected:
                continue
            redundancy = max(_dot(vector, vectors[chosen]) for chosen in selected)
            score = lambda_mult * relevance[index] - (1.0 - lambda_mult) * redundancy
            if score > best_score:
                best_index = index
                best_score = score
        selected.append(best_index)
    return selected


class SqliteVecStore(VectorStore):
    """Chunk rows in a plain table, vectors in a vec0 table, joined on rowid.

    Subclasses ``VectorStore`` so ``as_retriever(search_type="mmr")`` keeps
    working untouched: ``VectorStoreRetriever`` dispatches straight to
    :meth:`max_marginal_relevance_search`, and ``graph.nodes`` still gets back a
    list of ``Document`` from ``.invoke()``.
    """

    def __init__(
        self,
        path: str | Path,
        embedding: Embeddings,
        collection: str,
    ) -> None:
        if not _SAFE_COLLECTION.match(collection):
            raise ValueError(
                f"collection name {collection!r} is not a bare identifier; it "
                "becomes part of a table name and cannot be bound as a parameter"
            )
        self._path = Path(path)
        self._embedding = embedding
        self._collection = collection
        self._chunks_table = f"chunks_{collection}"
        self._vec_table = f"vec_{collection}"
        self._conn: sqlite3.Connection | None = None

    # -- connection -------------------------------------------------------

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def _connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn

        # Both guards belong here rather than at module import: sqlite-vec is
        # loaded per connection, so nothing is decided before this point.
        ensure_modern_sqlite()
        ensure_loadable_extensions()

        # Imported here, not at module scope, so the name resolves to whatever
        # ensure_modern_sqlite just put in sys.modules.
        import sqlite3

        import sqlite_vec

        # check_same_thread=False because the graph may run a retrieval on a
        # worker thread; the connection is read-mostly and used one call at a
        # time, so sqlite's own serialisation is enough.
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- schema -----------------------------------------------------------

    def _ensure_registry(self) -> sqlite3.Connection:
        conn = self._connection()
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{_REGISTRY_TABLE}" ('
            "collection TEXT PRIMARY KEY, "
            "dimension INTEGER NOT NULL, "
            "provider TEXT NOT NULL DEFAULT '')"
        )
        return conn

    def _ensure_schema(self, dimension: int, provider: str = "") -> None:
        conn = self._ensure_registry()
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{self._chunks_table}" ('
            "id INTEGER PRIMARY KEY, "
            "content TEXT NOT NULL, "
            "metadata TEXT NOT NULL)"
        )
        conn.execute(
            f'CREATE VIRTUAL TABLE IF NOT EXISTS "{self._vec_table}" '
            f"USING vec0(embedding float[{dimension}])"
        )
        conn.execute(
            f'INSERT INTO "{_REGISTRY_TABLE}" (collection, dimension, provider) '
            "VALUES (?, ?, ?) ON CONFLICT(collection) DO UPDATE SET "
            "dimension = excluded.dimension, provider = excluded.provider",
            (self._collection, dimension, provider),
        )
        conn.commit()

    def reset(self) -> None:
        """Drop this collection, leaving the rest of the file alone.

        Indexing is a rebuild, not an append: re-running it against a live file
        would otherwise stack a second copy of every chunk behind the first and
        halve what a fixed-k retrieval can reach. The next write recreates the
        tables, at whatever dimension the provider turns out to emit.
        """
        conn = self._ensure_registry()
        conn.execute(f'DROP TABLE IF EXISTS "{self._chunks_table}"')
        conn.execute(f'DROP TABLE IF EXISTS "{self._vec_table}"')
        conn.execute(
            f'DELETE FROM "{_REGISTRY_TABLE}" WHERE collection = ?',
            (self._collection,),
        )
        conn.commit()

    def _registry_exists(self) -> bool:
        """Whether anything has ever been written to this file.

        Read paths ask before querying the registry, so opening a fresh or
        foreign ``.sqlite`` reports a missing collection rather than raising
        ``no such table`` -- and never creates a table just by looking.
        """
        row = (
            self._connection()
            .execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
                (_REGISTRY_TABLE,),
            )
            .fetchone()
        )
        return row is not None

    def collection_exists(self) -> bool:
        if not self._registry_exists():
            return False
        found = (
            self._connection()
            .execute(
                f'SELECT 1 FROM "{_REGISTRY_TABLE}" WHERE collection = ?',
                (self._collection,),
            )
            .fetchone()
        )
        return found is not None

    def collection_names(self) -> list[str]:
        if not self._registry_exists():
            return []
        return [
            name
            for (name,) in self._connection().execute(
                f'SELECT collection FROM "{_REGISTRY_TABLE}" ORDER BY collection'
            )
        ]

    def dimension(self) -> int:
        row = None
        if self._registry_exists():
            row = (
                self._connection()
                .execute(
                    f'SELECT dimension FROM "{_REGISTRY_TABLE}" WHERE collection = ?',
                    (self._collection,),
                )
                .fetchone()
            )
        if row is None:
            raise ValueError(f"collection {self._collection!r} is not present in {self._path}")
        return int(row[0])

    # -- writing ----------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        contents = list(texts)
        if not contents:
            return []
        metadata_list = list(metadatas or [{} for _ in contents])

        vectors = self._embedding.embed_documents(contents)
        self._ensure_schema(len(vectors[0]), kwargs.get("provider", ""))

        conn = self._connection()
        ids: list[str] = []
        for content, metadata, vector in zip(contents, metadata_list, vectors, strict=True):
            cursor = conn.execute(
                f'INSERT INTO "{self._chunks_table}" (content, metadata) VALUES (?, ?)',
                (content, json.dumps(metadata or {})),
            )
            rowid = cursor.lastrowid
            conn.execute(
                f'INSERT INTO "{self._vec_table}" (rowid, embedding) VALUES (?, ?)',
                (rowid, _serialise(vector)),
            )
            ids.append(str(rowid))
        conn.commit()
        return ids

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> SqliteVecStore:
        path = kwargs.pop("path")
        collection = kwargs.pop("collection")
        store = cls(path=path, embedding=embedding, collection=collection)
        store.add_texts(texts, metadatas, **kwargs)
        return store

    # -- reading ----------------------------------------------------------

    def _knn(self, vector: list[float], fetch_k: int) -> list[tuple[Document, list[float]]]:
        expected = self.dimension()
        if len(vector) != expected:
            raise ValueError(
                f"the configured embedding provider produced a {len(vector)}-"
                f"dimension query vector, but collection {self._collection!r} in "
                f"{self._path} was built at {expected} dimensions. Re-index with "
                "this provider, or switch AORTA_CHAT_EMBEDDING_PROVIDER back to "
                "the one that built it."
            )

        conn = self._connection()
        # LIMIT rather than the older `k = ?` constraint: it needs sqlite 3.41,
        # which is what MIN_SQLITE_VERSION is pinned to. The KNN scan is its own
        # subquery so the bound reaches vec0 rather than the join.
        rows = conn.execute(
            "SELECT c.content, c.metadata, m.embedding FROM ("
            f'  SELECT rowid, distance, embedding FROM "{self._vec_table}"'
            "  WHERE embedding MATCH ? LIMIT ?"
            f') m JOIN "{self._chunks_table}" c ON c.id = m.rowid'
            " ORDER BY m.distance",
            (_serialise(vector), fetch_k),
        ).fetchall()

        return [
            (
                Document(page_content=content, metadata=json.loads(metadata)),
                _deserialise(embedding),
            )
            for content, metadata, embedding in rows
        ]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        vector = self._embedding.embed_query(query)
        return [document for document, _ in self._knn(vector, k)]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        vector = self._embedding.embed_query(query)
        candidates = self._knn(vector, max(fetch_k, k))
        picked = _mmr(
            vector,
            [candidate_vector for _, candidate_vector in candidates],
            k,
            lambda_mult,
        )
        return [candidates[index][0] for index in picked]


def carry_over_collections(source: Path, dest: Path) -> list[str]:
    """Copy collections present in ``source`` but absent from ``dest``.

    Installing a published index replaces the whole ``.sqlite``, and the
    per-user run collection lives in that same file (``rag/runs.py``) with its
    own refresh cadence. Without this, every fetch silently deleted run
    retrieval until the user rebuilt it -- and nothing said so.

    Only collections ``dest`` lacks are copied, so the incoming source
    collection always wins: the point is to preserve private data, not to
    resurrect the index being replaced. Returns the names copied.
    """
    if not source.exists() or not dest.exists():
        return []

    ensure_modern_sqlite()
    ensure_loadable_extensions()

    import sqlite3

    import sqlite_vec

    conn = sqlite3.connect(dest)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("ATTACH DATABASE ? AS incoming", (str(source),))
        try:
            copied = _copy_missing_collections(conn)
        finally:
            conn.execute("DETACH DATABASE incoming")
    except sqlite3.DatabaseError:
        # A source file that is not a usable store has nothing to preserve, and
        # failing the install over it would be worse than the data loss this
        # function exists to prevent: the new index is already verified.
        logger.warning("Could not read %s to preserve its private collections", source)
        return []
    finally:
        conn.close()
    if copied:
        logger.info("Preserved local collection(s) across the install: %s", ", ".join(copied))
    return copied


def _copy_missing_collections(conn: sqlite3.Connection) -> list[str]:
    """Copy every ``incoming`` collection the main database does not have."""
    if not _has_registry(conn, "incoming"):
        return []
    theirs = {
        name: (int(dimension), provider)
        for name, dimension, provider in conn.execute(
            f'SELECT collection, dimension, provider FROM incoming."{_REGISTRY_TABLE}"'
        )
    }
    if not theirs:
        return []
    mine = (
        {name for (name,) in conn.execute(f'SELECT collection FROM main."{_REGISTRY_TABLE}"')}
        if _has_registry(conn, "main")
        else set()
    )

    copied = []
    for name in sorted(set(theirs) - mine):
        # Names come from the registry of a file this install wrote, but they
        # are interpolated into table names, so they get the same check a
        # constructed store gets rather than being trusted for their origin.
        if not _SAFE_COLLECTION.match(name):
            logger.warning("Skipping collection %r: not a bare identifier", name)
            continue
        dimension, provider = theirs[name]
        chunks, vectors = f"chunks_{name}", f"vec_{name}"
        conn.execute(
            f'CREATE TABLE main."{chunks}" '
            "(id INTEGER PRIMARY KEY, content TEXT NOT NULL, metadata TEXT NOT NULL)"
        )
        conn.execute(
            f'CREATE VIRTUAL TABLE main."{vectors}" USING vec0(embedding float[{dimension}])'
        )
        conn.execute(
            f'INSERT INTO main."{chunks}" (id, content, metadata) '
            f'SELECT id, content, metadata FROM incoming."{chunks}"'
        )
        # Row by row rather than INSERT..SELECT: vec0 is a virtual table, and
        # the rowid has to be carried across explicitly because it is the join
        # key back to the chunk text.
        for rowid, embedding in conn.execute(
            f'SELECT rowid, embedding FROM incoming."{vectors}"'
        ).fetchall():
            conn.execute(
                f'INSERT INTO main."{vectors}" (rowid, embedding) VALUES (?, ?)',
                (rowid, embedding),
            )
        conn.execute(
            f'INSERT INTO main."{_REGISTRY_TABLE}" (collection, dimension, provider) '
            "VALUES (?, ?, ?)",
            (name, dimension, provider),
        )
        copied.append(name)
    conn.commit()
    return copied


def _has_registry(conn: sqlite3.Connection, schema: str) -> bool:
    row = conn.execute(
        f"SELECT name FROM {schema}.sqlite_master WHERE type = 'table' AND name = ?",
        (_REGISTRY_TABLE,),
    ).fetchone()
    return row is not None


_retriever_cache: VectorStoreRetriever | None = None
_vectorstore_cache: SqliteVecStore | None = None


def reset_caches() -> None:
    """Drop the cached vectorstore and retriever.

    Both caches are tied to one embedding provider, so they must be cleared
    together when the configured provider changes within a process.
    """
    global _retriever_cache, _vectorstore_cache
    if isinstance(_vectorstore_cache, SqliteVecStore):
        _vectorstore_cache.close()
    _retriever_cache = None
    _vectorstore_cache = None


def _check_manifest(
    index_file: Path,
    provider: EmbeddingProvider,
    chunk_count: int | None = None,
) -> None:
    """Enforce Decision 20a before the first query, not after it.

    Warn on source drift, refuse on an embedding-model or dimension mismatch.
    The refusal is the point: the store's own dimension check only fires when
    the numbers differ, and two BGE-small variants at 384 dimensions would sail
    straight past it and answer from vectors that were never comparable.

    An index whose manifest is absent, unreadable or from a schema this install
    does not know is refused rather than warned about. The warning it used to
    get was the same hole by another name: without a usable manifest there is
    nothing to compare the model against, so two same-dimension indexes from
    different models are exactly the case that sails through -- the one this
    function exists to stop. Every path that builds or fetches an index writes
    the sidecar, so a missing one is a copied or hand-moved index rather than a
    normal state, and rebuilding or re-fetching is the fix. ``aorta chat
    doctor`` reports it without raising.

    ``chunk_count`` is the collection's live row count, which is how a manifest
    that describes different contents than the file holds gets caught. It is
    deliberately not a hash of the ``.sqlite``: the private run collection
    shares that file and rewrites it on its own cadence, so a whole-file digest
    stops matching the moment the user builds run retrieval.
    """
    from aorta.chat.rag import manifest as manifest_mod

    try:
        found = manifest_mod.read_manifest(index_file)
    except manifest_mod.ManifestError as exc:
        raise manifest_mod.IndexMismatchError(str(exc)) from exc

    report = manifest_mod.validate(
        found,
        embedding_model=provider.model_id(),
        collection=provider.collection_name(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        installed_version=_installed_version(),
        chunk_count=chunk_count,
    )
    for warning in report.warnings:
        logger.warning("chat index: %s", warning)
    report.raise_if_refused(index_file)


def _installed_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("amd-aorta")
    except PackageNotFoundError:  # pragma: no cover - raw source tree
        return ""


def collection_chunk_count(path: Path, collection: str) -> int | None:
    """Rows in ``collection``'s chunk table, or ``None`` when unreadable.

    ``None`` means "no evidence" rather than "empty", so a caller skips the
    contents check instead of refusing: a file that is not a store, or holds no
    such collection, has a better error waiting for it further down.

    Deliberately raw sqlite rather than a :class:`SqliteVecStore`. The chunk
    table is an ordinary table, so counting it needs neither the sqlite-vec
    extension nor an embedding provider -- and requiring a provider would mean
    ``aorta chat doctor`` downloaded a 65 MB model to count rows.
    """
    import sqlite3

    if not path.exists() or not _SAFE_COLLECTION.match(collection):
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(f'SELECT COUNT(*) FROM "chunks_{collection}"').fetchone()
        return int(row[0])
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _get_vectorstore() -> SqliteVecStore:
    """Return the cached sqlite-vec store, opening it on first call."""
    global _vectorstore_cache
    if _vectorstore_cache is not None:
        return _vectorstore_cache

    index_file = settings.index_file
    if not index_file.exists():
        raise FileNotFoundError(_MISSING_INDEX.format(path=index_file))

    # One provider decides both halves: embeddings and the collection they were
    # written to. Mixing them would query the wrong vector dimension.
    provider = get_provider()
    store = SqliteVecStore(
        path=index_file,
        embedding=provider.get_embeddings(),
        collection=provider.collection_name(),
    )
    # The manifest check still runs first, and on an absent collection it runs
    # exactly as it did before the count existed: a model mismatch renames the
    # collection, and its refusal names the cause far better than "no such
    # collection" would.
    _check_manifest(
        index_file, provider, collection_chunk_count(index_file, provider.collection_name())
    )
    if not store.collection_exists():
        present = ", ".join(store.collection_names()) or "none"
        store.close()
        raise FileNotFoundError(
            _MISSING_COLLECTION.format(
                path=index_file,
                collection=provider.collection_name(),
                provider=provider.describe(),
                present=present,
            )
        )
    _vectorstore_cache = store
    return _vectorstore_cache


def get_retriever(k: int | None = None) -> VectorStoreRetriever:
    """Return a retriever over the persisted AORTA sqlite-vec collection.

    Uses MMR (Maximal Marginal Relevance) to balance relevance and diversity.
    Results are cached after the first call.
    """
    global _retriever_cache
    if _retriever_cache is not None:
        return _retriever_cache

    k = k or settings.retriever_k
    fetch_k = settings.retriever_fetch_k
    _retriever_cache = _get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.7},
    )
    return _retriever_cache


def search_docs(query: str, k: int) -> list:
    """Search the sqlite-vec collection for exactly k results using MMR.

    Unlike the cached retriever (which has a fixed k), this function queries
    the vectorstore directly so callers can request any k at call time.
    """
    fetch_k = max(settings.retriever_fetch_k, k * 2)
    return _get_vectorstore().max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=0.7
    )
