"""Tests for list_files and read_file tools using temporary directories."""

from __future__ import annotations

from unittest.mock import patch

from aorta.chat.tools.files import list_files, read_file


class TestListFiles:
    @patch("aorta.chat.tools.files.settings")
    def test_lists_root_files(self, mock_settings, fake_aorta_dir):
        mock_settings.aorta_root = fake_aorta_dir
        result = list_files.invoke({"path": "."})
        assert "README.md" in result
        assert "config.yaml" in result
        assert "src/" in result

    @patch("aorta.chat.tools.files.settings")
    def test_ignores_pycache(self, mock_settings, fake_aorta_dir):
        mock_settings.aorta_root = fake_aorta_dir
        result = list_files.invoke({"path": "."})
        assert "__pycache__" not in result

    @patch("aorta.chat.tools.files.settings")
    def test_ignores_git(self, mock_settings, fake_aorta_dir):
        mock_settings.aorta_root = fake_aorta_dir
        result = list_files.invoke({"path": "."})
        assert ".git" not in result

    @patch("aorta.chat.tools.files.settings")
    def test_nonexistent_path(self, mock_settings, fake_aorta_dir):
        mock_settings.aorta_root = fake_aorta_dir
        result = list_files.invoke({"path": "does_not_exist"})
        assert "Error" in result
        assert "does not exist" in result

    @patch("aorta.chat.tools.files.settings")
    def test_lists_subdirectory(self, mock_settings, fake_aorta_dir):
        mock_settings.aorta_root = fake_aorta_dir
        result = list_files.invoke({"path": "src"})
        assert "__init__.py" in result
        assert "main.py" in result

    @patch("aorta.chat.tools.files.settings")
    def test_file_not_dir_error(self, mock_settings, fake_aorta_dir):
        mock_settings.aorta_root = fake_aorta_dir
        result = list_files.invoke({"path": "README.md"})
        assert "not a directory" in result

    @patch("aorta.chat.tools.files.settings")
    def test_empty_directory(self, mock_settings, fake_aorta_dir):
        mock_settings.aorta_root = fake_aorta_dir
        (fake_aorta_dir / "empty_dir").mkdir()
        result = list_files.invoke({"path": "empty_dir"})
        assert result == "(empty directory)"


class TestReadFile:
    @patch("aorta.chat.tools.files.settings")
    def test_reads_file_content(self, mock_settings, fake_aorta_dir):
        mock_settings.aorta_root = fake_aorta_dir
        result = read_file.invoke({"file_path": "config.yaml"})
        assert "key: value" in result

    @patch("aorta.chat.tools.files.settings")
    def test_reads_python_file(self, mock_settings, fake_aorta_dir):
        mock_settings.aorta_root = fake_aorta_dir
        result = read_file.invoke({"file_path": "src/main.py"})
        assert "def main():" in result

    @patch("aorta.chat.tools.files.settings")
    def test_nonexistent_file(self, mock_settings, fake_aorta_dir):
        mock_settings.aorta_root = fake_aorta_dir
        result = read_file.invoke({"file_path": "no_such_file.txt"})
        assert "Error" in result
        assert "does not exist" in result

    @patch("aorta.chat.tools.files.settings")
    def test_truncates_large_file(self, mock_settings, fake_aorta_dir):
        mock_settings.aorta_root = fake_aorta_dir
        big_file = fake_aorta_dir / "big.py"
        big_file.write_text("x" * 10000, encoding="utf-8")
        result = read_file.invoke({"file_path": "big.py"})
        assert "truncated" in result
        assert "10000 total chars" in result

    @patch("aorta.chat.tools.files.settings")
    def test_directory_not_file_error(self, mock_settings, fake_aorta_dir):
        mock_settings.aorta_root = fake_aorta_dir
        result = read_file.invoke({"file_path": "src"})
        assert "not a file" in result

    @patch("aorta.chat.tools.files.settings")
    def test_traversal_blocked(self, mock_settings, fake_aorta_dir):
        mock_settings.aorta_root = fake_aorta_dir
        result = read_file.invoke({"file_path": "../../etc/passwd"})
        assert "Error" in result or "escapes" in result


class TestSandboxEscapes:
    """The guard on both tools is what stands between a prompt injection and the disk."""

    @patch("aorta.chat.tools.files.settings")
    def test_read_file_refuses_a_sibling_sharing_the_root_prefix(self, mock_settings, tmp_path):
        root = tmp_path / "aorta"
        root.mkdir()
        (tmp_path / "aorta-old").mkdir()
        (tmp_path / "aorta-old" / "secrets.txt").write_text("api_key=hunter2\n", encoding="utf-8")
        mock_settings.aorta_root = root
        result = read_file.invoke({"file_path": "../aorta-old/secrets.txt"})
        assert "escapes the AORTA root" in result
        assert "hunter2" not in result

    @patch("aorta.chat.tools.files.settings")
    def test_list_files_refuses_a_sibling_sharing_the_root_prefix(self, mock_settings, tmp_path):
        root = tmp_path / "aorta"
        root.mkdir()
        (tmp_path / "aorta-old").mkdir()
        (tmp_path / "aorta-old" / "secrets.txt").write_text("api_key=hunter2\n", encoding="utf-8")
        mock_settings.aorta_root = root
        result = list_files.invoke({"path": "../aorta-old"})
        assert "escapes the AORTA root" in result
        assert "secrets.txt" not in result

    @patch("aorta.chat.tools.files.settings")
    def test_read_file_refuses_an_absolute_path(self, mock_settings, fake_aorta_dir):
        mock_settings.aorta_root = fake_aorta_dir
        assert "escapes the AORTA root" in read_file.invoke({"file_path": "/etc/passwd"})

    @patch("aorta.chat.tools.files.settings")
    def test_read_file_refuses_a_symlink_out_of_the_root(self, mock_settings, tmp_path):
        root = tmp_path / "aorta"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secrets.txt").write_text("api_key=hunter2\n", encoding="utf-8")
        (root / "link.txt").symlink_to(outside / "secrets.txt")
        mock_settings.aorta_root = root
        result = read_file.invoke({"file_path": "link.txt"})
        assert "escapes the AORTA root" in result
        assert "hunter2" not in result

    @patch("aorta.chat.tools.files.settings")
    def test_list_files_refuses_a_symlinked_directory_out_of_the_root(
        self, mock_settings, tmp_path
    ):
        root = tmp_path / "aorta"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secrets.txt").write_text("api_key=hunter2\n", encoding="utf-8")
        (root / "link").symlink_to(outside, target_is_directory=True)
        mock_settings.aorta_root = root
        assert "escapes the AORTA root" in list_files.invoke({"path": "link"})

    @patch("aorta.chat.tools.files.settings")
    def test_a_child_sharing_the_root_prefix_is_still_readable(self, mock_settings, tmp_path):
        """Refusing ``<root>-old`` must not also refuse ``<root>/<root name>-old``."""
        root = tmp_path / "aorta"
        (root / "aorta-old").mkdir(parents=True)
        (root / "aorta-old" / "notes.md").write_text("# kept\n", encoding="utf-8")
        mock_settings.aorta_root = root
        assert "aorta-old/notes.md" in list_files.invoke({"path": "aorta-old"})
        assert "# kept" in read_file.invoke({"file_path": "aorta-old/notes.md"})
