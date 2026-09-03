"""Traversals must not follow a symlink out of the tree they were pointed at.

The containment guard these modules share, ``resolve_within``, vets the path a
caller *named*: it resolves it and refuses one that lands outside the root, so a
symlink handed to ``read_file`` is already caught. What it cannot vet is the
files a traversal then discovers for itself, and four of them existed -- the
corpus loader, ``grep_code``, the repo map and the run-artifact walk -- each
reading whatever a link underneath an allowed root pointed at.

Two consequences, depending on the flow: with ``embedding_provider = "remote"``
the target's bytes are uploaded to the embeddings API, and a ``--public-only``
build embeds them verbatim while both tracked-path guards pass, because the
link's *own* path is the tracked one.

The wider question of whether these tools should sit on ``aorta.run._fsafe``
instead is #430; this is the exposure that issue leaves unanalysed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def tree_with_escape(tmp_path: Path) -> tuple[Path, Path]:
    """A corpus root holding a link to a secret that lives outside it."""
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.py"
    secret.write_text("SECRET_TOKEN = 'leaked-from-outside-the-root'\n", encoding="utf-8")

    root = tmp_path / "root"
    root.mkdir()
    (root / "real.py").write_text("x = 1\n", encoding="utf-8")
    (root / "link.py").symlink_to(secret)
    return root, secret


class TestCorpusLoader:
    def test_a_symlinked_file_is_not_indexed(self, tree_with_escape):
        from aorta.chat.rag.indexer import _load_documents

        root, _ = tree_with_escape
        docs = _load_documents(root)

        sources = {doc.metadata["source"] for doc in docs}
        assert "real.py" in sources
        assert "link.py" not in sources
        assert not any("leaked-from-outside-the-root" in doc.page_content for doc in docs)


class TestGrepCode:
    def test_a_symlinked_file_is_not_searched(self, tree_with_escape):
        from aorta.chat.tools.search import grep_code

        root, _ = tree_with_escape
        with patch("aorta.chat.tools.search.settings") as mock_settings:
            mock_settings.aorta_root = root
            mock_settings.search_tool_k = 10
            result = grep_code.invoke({"pattern": "SECRET_TOKEN", "path": "."})

        assert "leaked-from-outside-the-root" not in result
        assert "link.py" not in result


class TestRepoMap:
    def test_a_symlinked_directory_is_not_descended(self, tmp_path: Path):
        from aorta.chat.rag.repo_map import _build_tree

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.py").write_text("def leaked(): ...\n", encoding="utf-8")

        root = tmp_path / "root"
        root.mkdir()
        (root / "real.py").write_text("def kept(): ...\n", encoding="utf-8")
        (root / "escape").symlink_to(outside, target_is_directory=True)

        lines = "\n".join(_build_tree(root))

        assert "real.py" in lines
        assert "escape" not in lines
        assert "leaked" not in lines

    def test_a_directory_cycle_terminates(self, tmp_path: Path):
        """A link to an ancestor recursed until the interpreter stopped it."""
        from aorta.chat.rag.repo_map import _build_tree

        root = tmp_path / "root"
        (root / "pkg").mkdir(parents=True)
        (root / "pkg" / "loop").symlink_to(root, target_is_directory=True)

        assert "loop" not in "\n".join(_build_tree(root))


class TestRunArtifactWalk:
    def test_a_symlinked_artifact_is_not_discovered(self, tmp_path: Path):
        """``iter_artifacts`` already refused linked directories, not files."""
        from aorta.chat.runs import iter_artifacts

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "matrix.json").write_text('{"cells": []}', encoding="utf-8")

        root = tmp_path / "runs"
        (root / "real_run").mkdir(parents=True)
        (root / "real_run" / "matrix.json").write_text('{"cells": []}', encoding="utf-8")
        (root / "linked_run").mkdir()
        (root / "linked_run" / "matrix.json").symlink_to(outside / "matrix.json")

        found = {path for path, _ in iter_artifacts(root)}

        assert root / "real_run" / "matrix.json" in found
        assert root / "linked_run" / "matrix.json" not in found
