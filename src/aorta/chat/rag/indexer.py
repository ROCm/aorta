"""Index the AORTA codebase into ChromaDB using tree-sitter or fallback splitting."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from aorta.chat.config import settings
from aorta.chat.rag.embeddings.factory import get_provider
from aorta.chat.rag.sqlite_compat import ensure_modern_sqlite
from aorta.chat.rag.walk import prune_dirnames

# ``langchain_chroma`` imports ``chromadb`` at module import time, where the
# deprecated ``langchain_community.vectorstores.Chroma`` imported it lazily
# inside its constructor. The sqlite swap therefore has to happen before the
# import below, not merely before the first Chroma() call.
ensure_modern_sqlite()

from langchain_chroma import Chroma  # noqa: E402

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


def _load_documents(codebase_path: Path) -> list[Document]:
    """Walk the codebase and load files as LangChain Documents."""
    docs: list[Document] = []
    for dirpath, dirnames, filenames in os.walk(codebase_path):
        root = Path(dirpath)
        prune_dirnames(root, dirnames)

        for filename in sorted(filenames):
            fpath = root / filename
            if fpath.suffix not in CODE_EXTENSIONS:
                continue

            rel = str(fpath.relative_to(codebase_path))

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


def index_codebase(
    codebase_path: str | Path | None = None,
) -> Chroma:
    """Index the AORTA codebase into ChromaDB.

    Args:
        codebase_path: Override path; defaults to settings.aorta_path.

    Returns:
        The populated Chroma vector store.
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

    chroma_dir = Path(settings.chroma_path)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    collection = provider.collection_name()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
        collection_name=collection,
    )
    logger.info(
        "Indexed %d chunks into ChromaDB at %s (collection %s)",
        len(chunks),
        chroma_dir,
        collection,
    )
    return vectorstore
