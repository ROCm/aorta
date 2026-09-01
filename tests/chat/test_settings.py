"""``aorta.chat.config`` -- defaults, anchoring, laziness, and precedence.

The three things the move changed are the three things worth pinning, because
each of them fails silently rather than loudly:

* a default that resolves into ``site-packages`` only breaks on an installed
  wheel on a read-only share, not in a source checkout;
* an import-time ``Settings()`` only breaks ``aorta --help`` once someone's
  environment is malformed;
* an unprefixed env var only collides once the tool runs inside somebody else's
  job script.
"""

from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from aorta.chat.config import ENV_PREFIX, SECRET_FIELDS, ConfigFileError, Settings


class TestDefaults:
    def test_defaults(self):
        s = Settings()
        assert s.vllm_base_url == "http://localhost:8000/v1"
        assert "python" in s.allowed_commands
        assert s.max_retry_iterations == 3
        assert s.chunk_size == 512
        assert s.command_timeout == 60

    def test_aorta_root_is_path(self):
        assert isinstance(Settings().aorta_root, Path)

    def test_index_default_is_a_single_sqlite_file(self):
        """Not a Chroma directory: one file to checksum, side-load, or delete."""
        assert Settings().index_file.suffix == ".sqlite"


class TestDefaultsAreAnchoredOnTheUser:
    """Nothing writable may resolve inside the installed package.

    ``config/settings.py`` anchored the vector store, the repo map and the
    ``.env`` file on ``Path(__file__).parent.parent``, which in a wheel is
    ``site-packages/aorta/``. A tool that writes into its own install directory
    cannot be pip-upgraded cleanly, and on a shared node the directory is not
    even writable.
    """

    @pytest.mark.parametrize("field", ["index_path", "repo_map_path"])
    def test_writable_defaults_avoid_site_packages(self, field):
        value = Path(getattr(Settings(), field)).resolve()
        for key in ("purelib", "platlib"):
            site_packages = Path(sysconfig.get_paths()[key]).resolve()
            assert site_packages not in value.parents, f"{field} resolves into {key}"

    @pytest.mark.parametrize("field", ["index_path", "repo_map_path"])
    def test_writable_defaults_follow_xdg_cache_home(self, field, monkeypatch, tmp_path):
        """Resolved per construction, not baked in at module import."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        value = Path(getattr(Settings(), field))
        assert tmp_path in value.parents

    def test_the_corpus_default_is_the_installed_package(self):
        """``aorta_path`` is the one default that may point at the install.

        It is the RAG corpus and is only ever read, so pointing it at real code
        the user demonstrably has is the useful default.
        """
        assert Settings().aorta_root == Path(__file__).resolve().parents[2] / "src/aorta"


class TestPrecedence:
    def test_env_override_needs_the_prefix(self, monkeypatch, tmp_path):
        monkeypatch.setenv(f"{ENV_PREFIX}AORTA_PATH", str(tmp_path))
        monkeypatch.setenv(f"{ENV_PREFIX}VLLM_BASE_URL", "http://other:9000/v1")
        s = Settings()
        assert s.aorta_path == str(tmp_path)
        assert s.vllm_base_url == "http://other:9000/v1"

    def test_a_bare_unprefixed_name_is_ignored(self, monkeypatch):
        """The whole point of the prefix: ``CHUNK_SIZE`` belongs to nobody.

        A public package that reads bare names collides with whatever else the
        user's job script exports.
        """
        monkeypatch.setenv("CHUNK_SIZE", "9999")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        s = Settings()
        assert s.chunk_size == 512
        assert s.llm_provider == "vllm"

    def test_the_profile_file_beats_the_built_in_default(self, chat_profile):
        chat_profile.write_text("chunk_size = 777\n", encoding="utf-8")
        assert Settings().chunk_size == 777

    def test_the_environment_beats_the_profile_file(self, chat_profile, monkeypatch):
        chat_profile.write_text("chunk_size = 777\n", encoding="utf-8")
        monkeypatch.setenv(f"{ENV_PREFIX}CHUNK_SIZE", "888")
        assert Settings().chunk_size == 888

    def test_a_constructor_argument_beats_the_environment(self, monkeypatch):
        """How ``aorta chat --llm-provider`` outranks a configured value."""
        monkeypatch.setenv(f"{ENV_PREFIX}LLM_PROVIDER", "openai")
        assert Settings(llm_provider="litellm").llm_provider == "litellm"

    def test_an_unknown_profile_key_is_ignored_at_load_time(self, chat_profile):
        """A profile written by a newer aorta must not stop an older one.

        ``aorta chat config validate`` is where the key gets reported.
        """
        chat_profile.write_text('chroma_path = "/gone"\nchunk_size = 64\n', encoding="utf-8")
        assert Settings().chunk_size == 64

    def test_a_malformed_profile_raises_a_named_error(self, chat_profile):
        chat_profile.write_text("this is not = = toml\n", encoding="utf-8")
        with pytest.raises(ConfigFileError) as exc:
            Settings()
        assert str(chat_profile) in str(exc.value)
        assert "config init" in str(exc.value)


class TestLaziness:
    def test_importing_the_module_does_not_build_settings(self, monkeypatch):
        """``settings = Settings()`` at module scope would break `aorta --help`.

        A malformed environment has to fail at command dispatch, not at import,
        because ``aorta.cli`` reaches this module for every chat invocation and
        ``aorta chat --help`` must work on a broken configuration.
        """
        probe = (
            "import aorta.chat.config as c;"
            "assert c._cached is None, 'settings built at import time';"
            "print('ok')"
        )
        env_key = f"{ENV_PREFIX}CHUNK_SIZE"
        out = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env={**dict(__import__("os").environ), env_key: "not-an-integer"},
        )
        assert out.returncode == 0, out.stderr
        assert "ok" in out.stdout

    def test_the_proxy_forwards_reads_and_writes(self):
        from aorta.chat import config

        config.reset_settings()
        assert config.settings.chunk_size == 512
        config.settings.chunk_size = 99
        assert config.get_settings().chunk_size == 99
        config.reset_settings()


class TestSecretFields:
    def test_every_secret_field_exists_on_settings(self):
        """A renamed field must not silently stop being masked."""
        assert SECRET_FIELDS <= set(Settings.model_fields)

    def test_every_api_key_field_is_declared_secret(self):
        """Catches a new credential field that forgets to join the set."""
        declared = {name for name in Settings.model_fields if name.endswith("api_key")}
        assert declared == set(SECRET_FIELDS)
