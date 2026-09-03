"""The run-artifact RAG collection (Decision 11b).

A *second* collection in the same index file, holding this user's own run
outcomes rather than aorta's source. The two are separate on purpose:

* **Different refresh cadence.** Source changes when the package is upgraded;
  run artifacts change every time the operator runs a sweep. Rebuilding one
  must not rebuild the other, so this module owns its own reset and its own
  cache.
* **Different provenance, and this is the load-bearing one.** The docs/source
  collection is public and is what Phase 4 publishes from CI. This one is
  per-user data that can contain customer hostnames, filesystem layouts and
  environment variables. **It must never be built or shipped by CI.** Nothing
  here reads the public tree, and the index-publishing job must select the
  source collection by name rather than copying the whole file.

The store is :class:`~aorta.chat.rag.retriever.SqliteVecStore`, which records
each collection's vector dimension in its registry table and keys its tables on
the collection name. So the two collections coexist in one file, a provider
switch is a named error rather than a dimension crash, and this module's only
obligation is to pick a name that cannot collide -- hence the suffix on the
provider's own collection name.

Chunking is per artifact rather than per fixed window: a matrix cell and an
environment snapshot are already the units a question is asked about, and
splitting them mid-block would strand a failure count from the cell it belongs
to.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from aorta.artifacts import read_env, read_matrix
from aorta.chat.config import settings
from aorta.chat.rag.embeddings.factory import get_provider
from aorta.chat.rag.retriever import SqliteVecStore
from aorta.chat.runs import (
    ArtifactReadError,
    iter_artifacts,
    render_env,
    render_matrix,
    render_matrix_cell,
)

logger = logging.getLogger(__name__)

#: Appended to the provider's collection name. Keeps the run collection distinct
#: from the source one under every provider, and keeps the whole name a bare
#: identifier, which ``SqliteVecStore`` requires because it interpolates the
#: collection into table names.
RUN_COLLECTION_SUFFIX = "_runs"

#: Chunks written per call, matching the source indexer. A run root holds far
#: fewer chunks than the codebase, but the remote embedding provider still sends
#: one request per batch.
_WRITE_BATCH = 200

#: Covers both ways this state is reached, because they are indistinguishable
#: once ``reset()`` has dropped the collection: never indexed, or indexed from a
#: run root that held nothing. Telling someone who has just run the indexer to
#: "build one" without naming the second cause would send them in a circle.
_MISSING_COLLECTION = (
    "No run-artifact collection in {path}. Either it was never built, or it was "
    "built from a run root with no artifacts in it.\n"
    "  build:  aorta chat index runs\n"
    "  check:  the run root is {runs_root} (set AORTA_CHAT_RUNS_PATH to change "
    "it), and must contain matrix.json, env.json or host_env.json.\n"
    "Run artifacts are per-user data, so this collection is always built "
    "locally and is never part of a published index."
)


class RunCollectionMissingError(RuntimeError):
    """The index file has no run-artifact collection yet."""


def run_collection_name() -> str:
    """Collection this provider's run artifacts live in."""
    return f"{get_provider().collection_name()}{RUN_COLLECTION_SUFFIX}"


def _documents_for(path: Path, kind: str, root: Path) -> list[Document]:
    """Render one artifact into the documents that represent it.

    A matrix becomes a run-level document plus one per cell, because "which
    cell failed" and "what was this run" are different questions and a single
    blob answers neither well. An env snapshot is one document: it is already
    reduced to the handful of fields a triage question turns on.
    """
    try:
        source = str(path.relative_to(root))
    except ValueError:
        source = str(path)

    base = {"source": source, "artifact_kind": kind, "run_dir": str(path.parent)}

    if kind == "env":
        return [Document(page_content=render_env(read_env(path)), metadata=dict(base))]

    matrix = read_matrix(path)
    header = render_matrix(matrix, include_cells=False)
    docs = [Document(page_content=header, metadata={**base, "scope": "run"})]
    for cell in matrix.cells:
        # The run header rides along with every cell chunk: retrieved alone, a
        # cell block says "failed 3 of 5" without naming the workload or the
        # ticket it belongs to, which is not a usable answer.
        docs.append(
            Document(
                page_content=f"{header}\n\n{render_matrix_cell(cell)}",
                metadata={**base, "scope": "cell", "cell": cell.name or "?"},
            )
        )
    return docs


def collect_run_documents(runs_path: str | Path | None = None) -> list[Document]:
    """Render every readable run artifact under the run root into documents.

    An unreadable artifact is logged and skipped rather than aborting: a run
    root routinely holds one truncated ``matrix.json`` from a job that was
    killed, and that must not cost the user every other run's index.
    """
    root = Path(runs_path).resolve() if runs_path else settings.runs_root
    docs: list[Document] = []
    for path, kind in iter_artifacts(root):
        try:
            docs.extend(_documents_for(path, kind, root))
        except ArtifactReadError as exc:
            logger.warning("Skipping unreadable run artifact %s: %s", path, exc)
    return docs


def index_run_artifacts(runs_path: str | Path | None = None) -> SqliteVecStore:
    """Rebuild the run-artifact collection from the run root.

    Returns the populated store. Refreshing is a full rebuild of this
    collection only -- the source collection in the same file is untouched,
    which is the whole reason the two are separate.
    """
    root = Path(runs_path).resolve() if runs_path else settings.runs_root
    if not root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {root}")

    docs = collect_run_documents(root)
    provider = get_provider()
    collection = run_collection_name()

    index_file = settings.index_file
    index_file.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteVecStore(
        path=index_file,
        embedding=provider.get_embeddings(),
        collection=collection,
    )
    # Reset even with nothing to write: a run root that was emptied should
    # leave no stale cells behind to be retrieved as current.
    store.reset()
    if not docs:
        logger.warning("No run artifacts found under %s; collection is now empty.", root)
        return store

    for start in range(0, len(docs), _WRITE_BATCH):
        store.add_documents(docs[start : start + _WRITE_BATCH], provider=provider.describe())
    logger.info(
        "Indexed %d run-artifact chunks from %s into %s (collection %s)",
        len(docs),
        root,
        index_file,
        collection,
    )
    return store


_store_cache: SqliteVecStore | None = None


def reset_caches() -> None:
    """Drop the cached run store. Paired with the retriever's own reset."""
    global _store_cache
    if _store_cache is not None:
        _store_cache.close()
    _store_cache = None


def _get_store() -> SqliteVecStore:
    global _store_cache
    if _store_cache is not None:
        return _store_cache

    index_file = settings.index_file
    if not index_file.exists():
        raise RunCollectionMissingError(
            _MISSING_COLLECTION.format(path=index_file, runs_root=settings.runs_root)
        )

    provider = get_provider()
    store = SqliteVecStore(
        path=index_file,
        embedding=provider.get_embeddings(),
        collection=run_collection_name(),
    )
    if not store.collection_exists():
        store.close()
        raise RunCollectionMissingError(
            _MISSING_COLLECTION.format(path=index_file, runs_root=settings.runs_root)
        )
    _store_cache = store
    return _store_cache


def search_run_docs(query: str, k: int) -> list[Document]:
    """Search the run-artifact collection for exactly *k* results."""
    fetch_k = max(settings.retriever_fetch_k, k * 2)
    return _get_store().max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=0.7
    )


__all__ = [
    "RUN_COLLECTION_SUFFIX",
    "RunCollectionMissingError",
    "collect_run_documents",
    "index_run_artifacts",
    "reset_caches",
    "run_collection_name",
    "search_run_docs",
]
