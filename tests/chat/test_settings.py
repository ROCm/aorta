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
from pydantic import ValidationError

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

    @pytest.mark.parametrize("field", ["index_path", "repo_map_path", "model_cache_path"])
    def test_writable_defaults_avoid_site_packages(self, field):
        value = Path(getattr(Settings(), field)).resolve()
        for key in ("purelib", "platlib"):
            site_packages = Path(sysconfig.get_paths()[key]).resolve()
            assert site_packages not in value.parents, f"{field} resolves into {key}"

    @pytest.mark.parametrize("field", ["index_path", "repo_map_path", "model_cache_path"])
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


class TestBlankEmbeddingModelIsRejected:
    """An empty embedding model name is refused where it enters, not downstream.

    It used to be accepted, and then reasoned about as though it were a model
    name by every consumer in turn: ``doctor`` printed a pre-warm command
    calling ``TextEmbedding("")``, which raises ``Model  is not supported in
    TextEmbedding``; ``manifest.remedy_lines`` offered ``index build`` on both
    the local and the remote arm, and the build resolves the same empty name
    and fails before the first chunk; ``collection_name()`` hashed ``""`` into
    a plausible-looking collection and ``describe()`` rendered ``local BGE
    embeddings ( on onnxruntime)``.

    Four consumers, one bad value. These tests pin the decision at the single
    place that makes all four unreachable, so a fifth consumer added later
    inherits the guarantee instead of the bug.
    """

    @pytest.mark.parametrize("field", ["embedding_model", "remote_embedding_model"])
    @pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
    def test_a_blank_model_name_is_refused(self, field, blank):
        """Both providers' model settings, and whitespace-only as well as empty.

        Whitespace-only is in here because a name of spaces is exactly as
        unusable as an empty one while looking non-empty to every ``if not
        model`` in the tree -- accepting it would move the defect rather than
        fix it. Both fields, because the remote arm has the same hole: a blank
        ``remote_embedding_model`` was offered ``index build`` too, which sends
        every chunk of the corpus to an embeddings API with no model named.
        """
        with pytest.raises(ValidationError) as exc:
            Settings(**{field: blank})
        assert field in str(exc.value)
        assert "must name an embedding model" in str(exc.value)

    def test_the_message_says_what_to_do_instead(self):
        """A rejection a user cannot act on just moves the dead end earlier."""
        with pytest.raises(ValidationError) as exc:
            Settings(embedding_model="")
        message = str(exc.value)
        assert "Remove the setting to take the default" in message
        assert "index build" in message and "index fetch" in message

    def test_it_refuses_the_value_from_the_environment_too(self, monkeypatch):
        """The constructor is not the only door; ``AORTA_CHAT_*`` is the common one."""
        monkeypatch.setenv(f"{ENV_PREFIX}EMBEDDING_MODEL", "")
        with pytest.raises(ValidationError):
            Settings()

    def test_a_model_name_is_not_stripped(self):
        """Rejecting a name with stray whitespace is the job; rewriting one is not.

        ``model_id()`` and ``vector_identity()`` return this setting verbatim
        and ``collection_name()`` hashes it, so stripping would silently change
        an install's embedding identity and mismatch the index it already
        built. ``manifest._custom_local_model`` compares unstripped for the
        same reason.
        """
        assert Settings(embedding_model=" BAAI/bge-small-en-v1.5 ").embedding_model == (
            " BAAI/bge-small-en-v1.5 "
        )

    def test_an_ordinary_custom_model_still_loads(self):
        """The guard is against blankness, not against configuring the model."""
        assert Settings(embedding_model="BAAI/bge-base-en-v1.5").embedding_model == (
            "BAAI/bge-base-en-v1.5"
        )

    def test_unset_still_takes_the_shipped_default(self):
        """ "Remove the setting" is the advice the message gives, so it has to work."""
        assert Settings().embedding_model == Settings.model_fields["embedding_model"].default

    def test_the_pre_warm_command_can_no_longer_be_built_on_a_blank_name(self, monkeypatch):
        """The leak site that opened this, tied back to the boundary that closes it.

        Asserted through the real settings rather than a monkeypatched
        attribute: what the fix guarantees is that no *configuration* produces
        a blank model, and going through ``Settings`` is the only way to test
        that claim. Monkeypatching ``settings.embedding_model`` bypasses
        validation and would pass either way.
        """
        monkeypatch.setenv(f"{ENV_PREFIX}EMBEDDING_MODEL", "")
        with pytest.raises(ValidationError):
            Settings()

        # And the command really is unusable for the value that used to reach
        # it -- the other half of the claim, which is why the guard is worth
        # having at all.
        from fastembed import TextEmbedding

        with pytest.raises(ValueError, match="not supported"):
            TextEmbedding("")


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
