"""Tests for Settings class from aorta/chat/config.py."""

from __future__ import annotations

from pathlib import Path


class TestSettings:
    def test_defaults(self):
        from aorta.chat.config import Settings
        s = Settings()
        assert s.vllm_base_url == "http://localhost:8000/v1"
        assert "python" in s.allowed_commands
        assert s.max_retry_iterations == 3
        assert s.chunk_size == 512

    def test_aorta_root_is_path(self):
        from aorta.chat.config import Settings
        s = Settings()
        assert isinstance(s.aorta_root, Path)

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AORTA_PATH", str(tmp_path))
        monkeypatch.setenv("VLLM_BASE_URL", "http://other:9000/v1")
        from aorta.chat.config import Settings
        s = Settings()
        assert s.aorta_path == str(tmp_path)
        assert s.vllm_base_url == "http://other:9000/v1"

    def test_command_timeout_default(self):
        from aorta.chat.config import Settings
        s = Settings()
        assert s.command_timeout == 60
