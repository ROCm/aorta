"""Index the AORTA codebase into sqlite-vec using language-aware splitting."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from aorta.chat.config import settings
from aorta.chat.rag.embeddings.factory import get_provider
from aorta.chat.rag.retriever import SqliteVecStore
from aorta.chat.rag.walk import prune_dirnames

logger = logging.getLogger(__name__)

EXTENSION_LANG_MAP: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".go": Language.GO,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".rs": Language.RUST,
    ".rb": Language.RUBY,
    ".md": Language.MARKDOWN,
    ".rst": Language.RST,
}

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".java",
    ".cpp", ".c", ".h", ".rs", ".rb", ".sh", ".bash",
    ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
    ".md", ".rst", ".txt",
}

#: Chunks embedded and written per call. A real AORTA tree splits into ~15,000
#: chunks, and embedding them in one call holds every vector in memory at once;
#: the remote provider would also send them all as a single request body.
_WRITE_BATCH = 500



def _get_splitter(ext: str) -> RecursiveCharacterTextSplitter:
    lang = EXTENSION_LANG_MAP.get(ext)
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def _load_documents(
    codebase_path: Path,
    include: Callable[[str], bool] | None = None,
    relative_to: Path | None = None,
) -> list[Document]:
    """Walk the codebase and load files as LangChain Documents.

    Args:
        codebase_path: Directory to walk.
        include: Optional predicate over the ``source`` metadata value. The
            published-index build passes the checkout's tracked-file set here,
            which is what makes "public tree only" a filter rather than a
            convention (see :mod:`aorta.chat.rag.corpus`).
        relative_to: Base the ``source`` metadata is computed against, when it
            differs from the walk root -- so a corpus assembled from several
            subdirectories of one tree yields non-colliding, tree-relative
            source paths.
    """
    base = relative_to or codebase_path
    docs: list[Document] = []
    for dirpath, dirnames, filenames in os.walk(codebase_path):
        root = Path(dirpath)
        prune_dirnames(root, dirnames)

        for filename in sorted(filenames):
            fpath = root / filename
            if fpath.suffix not in CODE_EXTENSIONS:
                continue

            rel = str(fpath.relative_to(base))
            if include is not None and not include(rel):
                continue

            try:
                size = fpath.stat().st_size
            except OSError:
                logger.warning("Skipping unreadable file: %s", fpath)
                continue
            if size > 50_000:
                logger.debug(
                    "Skipping large file: %s (%d bytes)", fpath, size
                )
                continue

            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.warning("Skipping unreadable file: %s", fpath)
                continue

            lines = text.splitlines()
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": rel,
                        "start_line": 1,
                        "end_line": len(lines),
                        "extension": fpath.suffix,
                    },
                )
            )
    return docs


def _split_documents(docs: list[Document]) -> list[Document]:
    """Split documents using language-aware splitters."""
    all_chunks: list[Document] = []
    for doc in docs:
        ext = doc.metadata.get("extension", "")
        splitter = _get_splitter(ext)
        chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        all_chunks.extend(chunks)
    return all_chunks


#: Public aliases for the two halves of corpus loading. ``rag/corpus.py`` and
#: ``rag/index_ops.py`` build the published index from the same walk and the same
#: splitter as a local ``index_codebase()`` run -- an artifact built by a
#: different chunker than the one the manifest advertises would be undetectable.
load_documents = _load_documents
split_documents = _split_documents


def index_codebase(
    codebase_path: str | Path | None = None,
) -> SqliteVecStore:
    """Index the AORTA codebase into the sqlite-vec index file.

    Args:
        codebase_path: Override path; defaults to settings.aorta_path.

    Returns:
        The populated sqlite-vec store.
    """
    root = Path(codebase_path or settings.aorta_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Codebase path does not exist: {root}")

    logger.info("Loading documents from %s ...", root)
    docs = _load_documents(root)
    logger.info("Loaded %d files.", len(docs))

    logger.info("Splitting into chunks ...")
    chunks = _split_documents(docs)
    logger.info("Created %d chunks.", len(chunks))

    provider = get_provider()
    logger.info("Building embeddings with %s ...", provider.describe())
    embeddings = provider.get_embeddings()

    index_file = settings.index_file
    index_file.parent.mkdir(parents=True, exist_ok=True)

    collection = provider.collection_name()
    store = SqliteVecStore(
        path=index_file,
        embedding=embeddings,
        collection=collection,
    )
    store.reset()
    for start in range(0, len(chunks), _WRITE_BATCH):
        store.add_documents(chunks[start : start + _WRITE_BATCH], provider=provider.describe())
    logger.info(
        "Indexed %d chunks into %s (collection %s)",
        len(chunks),
        index_file,
        collection,
    )
    return store
