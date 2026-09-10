"""Which directories the indexer walks, and which it refuses to descend into.

Regression cover for a real incident: indexing AORTA produced 134,843 chunks
because `buck-out` and a `.testvenv` were walked, so 88% of the vector store was
build output and a virtualenv. Pruning brought it to 15,343 chunks over 672
files. Retrieval quality depends on this more than on speed -- MMR crowds out
real code when thousands of near-duplicate vendored chunks are present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aorta.chat.rag.indexer import _load_documents
from aorta.chat.rag.walk import HIDDEN_DIRS_TO_KEEP, SKIP_DIRS, is_virtualenv


def _write(path: Path, text: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sources(root: Path) -> set[str]:
    return {doc.metadata["source"] for doc in _load_documents(root)}


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """A codebase with one real module and a representative pile of noise."""
    _write(tmp_path / "src" / "real.py")
    _write(tmp_path / "README.md", "# real\n")

    _write(tmp_path / "buck-out" / "gen" / "artifact.py")
    _write(tmp_path / "build" / "lib" / "copy.py")
    _write(tmp_path / "dist" / "pkg.py")
    _write(tmp_path / "node_modules" / "dep" / "index.js")
    _write(tmp_path / "third-party" / "vendored.py")
    _write(tmp_path / "__pycache__" / "cached.py")
    _write(tmp_path / "experiments" / "scratch.py")
    _write(tmp_path / "misc" / "notes.md")

    _write(tmp_path / ".git" / "hooks" / "hook.py")
    _write(tmp_path / ".mypy_cache" / "cache.json", "{}\n")
    _write(tmp_path / ".ruff_cache" / "cache.json", "{}\n")
    _write(tmp_path / ".pytest_cache" / "v.json", "{}\n")
    _write(tmp_path / ".sanitizer-nightly" / "run.md", "# run\n")

    # A virtualenv whose name is in no denylist -- AORTA's real tree has one.
    _write(tmp_path / ".testvenv" / "pyvenv.cfg", "home = /usr\n")
    _write(tmp_path / ".testvenv" / "lib" / "dep.py")
    _write(tmp_path / "env311" / "pyvenv.cfg", "home = /usr\n")
    _write(tmp_path / "env311" / "lib" / "dep.py")

    _write(tmp_path / ".github" / "workflows" / "ci.yml", "on: push\n")
    return tmp_path


class TestPruning:
    def test_real_source_is_indexed(self, tree: Path):
        assert "src/real.py" in _sources(tree)
        assert "README.md" in _sources(tree)

    def test_ci_config_is_kept_despite_being_hidden(self, tree: Path):
        assert ".github/workflows/ci.yml" in _sources(tree)

    @pytest.mark.parametrize(
        "unwanted",
        [
            "buck-out/gen/artifact.py",
            "build/lib/copy.py",
            "dist/pkg.py",
            "node_modules/dep/index.js",
            "third-party/vendored.py",
            "__pycache__/cached.py",
            "experiments/scratch.py",
            "misc/notes.md",
        ],
    )
    def test_named_junk_directories_are_skipped(self, tree: Path, unwanted: str):
        assert unwanted not in _sources(tree)

    @pytest.mark.parametrize(
        "unwanted",
        [
            ".git/hooks/hook.py",
            ".mypy_cache/cache.json",
            ".ruff_cache/cache.json",
            ".pytest_cache/v.json",
            ".sanitizer-nightly/run.md",
        ],
    )
    def test_hidden_directories_are_skipped(self, tree: Path, unwanted: str):
        """One rule covers every cache and state directory, named or not."""
        assert unwanted not in _sources(tree)

    def test_a_virtualenv_is_skipped_whatever_it_is_called(self, tree: Path):
        sources = _sources(tree)
        assert ".testvenv/lib/dep.py" not in sources
        assert "env311/lib/dep.py" not in sources

    def test_only_the_intended_files_survive(self, tree: Path):
        """Asserted as an exact set, so new noise cannot creep in unnoticed."""
        assert _sources(tree) == {
            "src/real.py",
            "README.md",
            ".github/workflows/ci.yml",
        }


class TestVirtualenvDetection:
    def test_a_directory_with_pyvenv_cfg_is_a_virtualenv(self, tmp_path: Path):
        _write(tmp_path / "anyname" / "pyvenv.cfg", "home = /usr\n")
        assert is_virtualenv(tmp_path / "anyname")

    def test_an_ordinary_package_is_not(self, tmp_path: Path):
        _write(tmp_path / "src" / "__init__.py", "")
        assert not is_virtualenv(tmp_path / "src")

    def test_a_missing_directory_is_not(self, tmp_path: Path):
        assert not is_virtualenv(tmp_path / "absent")


class TestSkipConstants:
    def test_the_directories_that_caused_the_incident_are_covered(self):
        """`buck-out` was the single largest contributor, at 2,961 files."""
        assert "buck-out" in SKIP_DIRS
        assert "build" in SKIP_DIRS

    def test_site_packages_is_skipped_by_name_too(self):
        """A venv missing pyvenv.cfg still must not leak dependencies in."""
        assert "site-packages" in SKIP_DIRS

    def test_dot_prefixed_names_do_not_belong_in_skip_dirs(self):
        """Hidden directories are handled by one rule; listing them would rot."""
        assert not [name for name in SKIP_DIRS if name.startswith(".")]

    def test_kept_hidden_directories_are_hidden(self):
        assert all(name.startswith(".") for name in HIDDEN_DIRS_TO_KEEP)
