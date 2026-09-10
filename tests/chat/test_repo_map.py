"""Tests for repo_map functions using temporary directories."""

from __future__ import annotations

from unittest.mock import patch

from aorta.chat.rag.repo_map import _build_tree, _extract_python_signatures, generate_repo_map


class TestExtractPythonSignatures:
    def test_extracts_function(self, tmp_path):
        py_file = tmp_path / "mod.py"
        py_file.write_text("def greet(name, greeting):\n    pass\n", encoding="utf-8")
        sigs = _extract_python_signatures(py_file)
        assert any("def greet(name, greeting)" in s for s in sigs)

    def test_extracts_class_with_methods(self, tmp_path):
        py_file = tmp_path / "mod.py"
        py_file.write_text(
            "class Foo:\n"
            "    def bar(self):\n"
            "        pass\n"
            "    def baz(self, x):\n"
            "        pass\n",
            encoding="utf-8",
        )
        sigs = _extract_python_signatures(py_file)
        assert any("class Foo" in s for s in sigs)
        assert any("bar" in s and "baz" in s for s in sigs)

    def test_handles_syntax_error(self, tmp_path):
        py_file = tmp_path / "bad.py"
        py_file.write_text("def broken(\n", encoding="utf-8")
        sigs = _extract_python_signatures(py_file)
        assert sigs == []

    def test_async_function(self, tmp_path):
        py_file = tmp_path / "async_mod.py"
        py_file.write_text("async def fetch(url):\n    pass\n", encoding="utf-8")
        sigs = _extract_python_signatures(py_file)
        assert any("def fetch(url)" in s for s in sigs)

    def test_empty_file(self, tmp_path):
        py_file = tmp_path / "empty.py"
        py_file.write_text("", encoding="utf-8")
        sigs = _extract_python_signatures(py_file)
        assert sigs == []


class TestBuildTree:
    def test_builds_tree_with_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("hello\n", encoding="utf-8")
        lines = _build_tree(tmp_path)
        text = "\n".join(lines)
        assert "a.py" in text
        assert "b.txt" in text

    def test_skips_pycache(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "foo.pyc").write_bytes(b"\x00")
        (tmp_path / "main.py").write_text("pass\n", encoding="utf-8")
        lines = _build_tree(tmp_path)
        text = "\n".join(lines)
        assert "__pycache__" not in text

    def test_skips_git(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
        (tmp_path / "main.py").write_text("pass\n", encoding="utf-8")
        lines = _build_tree(tmp_path)
        text = "\n".join(lines)
        assert ".git" not in text

    def test_includes_subdirectories(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "app.py").write_text("def run():\n    pass\n", encoding="utf-8")
        lines = _build_tree(tmp_path)
        text = "\n".join(lines)
        assert "src/" in text
        assert "app.py" in text


class TestGenerateRepoMap:
    @patch("aorta.chat.rag.repo_map.settings")
    def test_generates_map(self, mock_settings, tmp_path):
        mock_settings.aorta_path = str(tmp_path)
        mock_settings.repo_map_path = str(tmp_path / "output" / "map.md")
        (tmp_path / "main.py").write_text("def hello():\n    pass\n", encoding="utf-8")
        result = generate_repo_map(codebase_path=tmp_path)
        assert "main.py" in result
        assert "hello" in result

    @patch("aorta.chat.rag.repo_map.settings")
    def test_writes_to_disk(self, mock_settings, tmp_path):
        output = tmp_path / "output" / "map.md"
        mock_settings.aorta_path = str(tmp_path)
        mock_settings.repo_map_path = str(output)
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        generate_repo_map(codebase_path=tmp_path)
        assert output.exists()
        assert "a.py" in output.read_text(encoding="utf-8")

    @patch("aorta.chat.rag.repo_map.settings")
    def test_nonexistent_path_raises(self, mock_settings, tmp_path):
        mock_settings.aorta_path = str(tmp_path / "nope")
        mock_settings.repo_map_path = str(tmp_path / "map.md")
        import pytest
        with pytest.raises(FileNotFoundError):
            generate_repo_map(codebase_path=tmp_path / "nope")
