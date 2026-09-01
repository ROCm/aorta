"""``aorta chat config init|show|validate``, and the two guards Decision 9b owes.

Writing an API key into a predictable path inside a tool whose day job is
collecting diagnostic bundles is only safe with both guards in place:
``config show`` masks the key, and ``aorta bundle`` refuses to package the file.
The likeliest leak is a customer pasting their own config into a support ticket,
so the masking is tested as a contract, not as formatting.
"""

from __future__ import annotations

import stat

import pytest
import tomllib
from click.testing import CliRunner

from aorta.chat import config
from aorta.cli.chat import _CONFIG_PROFILES, chat


class TestProfileTemplates:
    def test_the_click_choice_matches_the_templates(self):
        """The Choice list is hard-coded because decorators run at import time.

        Reading ``PROFILE_TEMPLATES`` in the decorator would import
        pydantic-settings on every ``aorta --help``, so the duplication is
        deliberate and this is the guard on it.
        """
        assert sorted(_CONFIG_PROFILES) == sorted(config.PROFILE_TEMPLATES)

    def test_every_template_prompts_for_something(self):
        assert set(config.PROFILE_PROMPTS) == set(config.PROFILE_TEMPLATES)

    @pytest.mark.parametrize("name", sorted(config.PROFILE_TEMPLATES))
    def test_every_template_only_sets_real_fields(self, name):
        """A template key that no longer exists would be silently dropped."""
        unknown = set(config.PROFILE_TEMPLATES[name]) - set(config.Settings.model_fields)
        assert not unknown, f"{name}: {sorted(unknown)}"

    @pytest.mark.parametrize("name", sorted(config.PROFILE_PROMPTS))
    def test_every_prompted_field_is_a_real_field(self, name):
        unknown = set(config.PROFILE_PROMPTS[name]) - set(config.Settings.model_fields)
        assert not unknown, f"{name}: {sorted(unknown)}"

    @pytest.mark.parametrize("name", sorted(config.PROFILE_TEMPLATES))
    def test_every_template_validates(self, name, chat_profile):
        """A shipped template that fails validation is a broken `config init`."""
        config.write_profile(dict(config.PROFILE_TEMPLATES[name]), chat_profile)
        assert config.validate_profile(chat_profile) == []


class TestConfigInit:
    def test_writes_a_parseable_profile(self, chat_profile):
        result = CliRunner().invoke(
            chat, ["config", "init", "--profile", "local-vllm", "--no-input"]
        )
        assert result.exit_code == 0, result.output
        assert chat_profile.exists()
        loaded = tomllib.loads(chat_profile.read_text(encoding="utf-8"))
        assert loaded["llm_provider"] == "vllm"

    def test_the_file_is_owner_only(self, chat_profile):
        """Decision 9b: the key lives at rest, so 0600 is the minimum bar."""
        CliRunner().invoke(chat, ["config", "init", "--profile", "openai", "--no-input"])
        mode = stat.S_IMODE(chat_profile.stat().st_mode)
        assert mode == 0o600, oct(mode)

    def test_an_existing_profile_is_not_clobbered(self, chat_profile):
        chat_profile.write_text("chunk_size = 1\n", encoding="utf-8")
        result = CliRunner().invoke(chat, ["config", "init", "--profile", "openai", "--no-input"])
        assert result.exit_code != 0
        assert "--force" in result.output
        assert chat_profile.read_text(encoding="utf-8") == "chunk_size = 1\n"

    def test_force_overwrites(self, chat_profile):
        chat_profile.write_text("chunk_size = 1\n", encoding="utf-8")
        result = CliRunner().invoke(
            chat, ["config", "init", "--profile", "openai", "--no-input", "--force"]
        )
        assert result.exit_code == 0, result.output
        assert "chunk_size" not in chat_profile.read_text(encoding="utf-8")

    def test_a_pre_existing_permissive_file_is_tightened(self, chat_profile):
        """O_CREAT's mode is ignored when the file already exists.

        A previous run under a loose umask, or a careless editor, leaves the
        profile at 0644; overwriting it must not preserve that.
        """
        chat_profile.write_text("chunk_size = 1\n", encoding="utf-8")
        chat_profile.chmod(0o644)
        config.write_profile({"chunk_size": 2}, chat_profile)
        assert stat.S_IMODE(chat_profile.stat().st_mode) == 0o600

    def test_prompted_answers_land_in_the_file(self, chat_profile):
        result = CliRunner().invoke(
            chat,
            ["config", "init", "--profile", "openai"],
            input="gpt-4o\nsk-typed-at-the-prompt\n",
        )
        assert result.exit_code == 0, result.output
        loaded = tomllib.loads(chat_profile.read_text(encoding="utf-8"))
        assert loaded["remote_llm_model"] == "gpt-4o"
        assert loaded["remote_llm_api_key"] == "sk-typed-at-the-prompt"

    def test_the_key_is_not_echoed_while_being_typed(self, chat_profile):
        result = CliRunner().invoke(
            chat,
            ["config", "init", "--profile", "openai"],
            input="gpt-4o\nsk-should-not-appear\n",
        )
        assert result.exit_code == 0, result.output
        assert "sk-should-not-appear" not in result.output


class TestConfigShow:
    def test_the_key_is_masked_by_default(self, chat_profile, monkeypatch):
        monkeypatch.setenv("AORTA_CHAT_REMOTE_LLM_API_KEY", "sk-abcdefgh12345678")
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "show"])
        assert result.exit_code == 0, result.output
        assert "sk-abcdefgh12345678" not in result.output
        # Enough tail to tell two keys apart in a support conversation.
        assert "5678" in result.output

    def test_reveal_prints_it_in_full(self, chat_profile, monkeypatch):
        monkeypatch.setenv("AORTA_CHAT_REMOTE_LLM_API_KEY", "sk-abcdefgh12345678")
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "show", "--reveal"])
        assert result.exit_code == 0, result.output
        assert "sk-abcdefgh12345678" in result.output

    def test_json_output_is_masked_too(self, chat_profile, monkeypatch):
        """--json is the form most likely to be pasted somewhere."""
        monkeypatch.setenv("AORTA_CHAT_REMOTE_LLM_API_KEY", "sk-abcdefgh12345678")
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "show", "--json"])
        assert result.exit_code == 0, result.output
        assert "sk-abcdefgh12345678" not in result.output

    @pytest.mark.parametrize("field", sorted(config.SECRET_FIELDS))
    def test_every_secret_field_is_masked(self, field, chat_profile, monkeypatch):
        secret = "sk-uniquesecretvalue9876"
        monkeypatch.setenv(f"AORTA_CHAT_{field.upper()}", secret)
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "show"])
        assert secret not in result.output

    def test_a_short_key_is_fully_starred(self):
        """No tail hint below nine characters, or the mask leaks most of it."""
        assert config.mask("sk-short") == "********"
        assert config.mask("") == ""


class TestConfigValidate:
    def test_a_healthy_profile_exits_zero(self, chat_profile):
        config.write_profile({"chunk_size": 128}, chat_profile)
        result = CliRunner().invoke(chat, ["config", "validate"])
        assert result.exit_code == 0, result.output
        assert "OK" in result.output

    def test_a_missing_profile_is_reported_not_crashed(self, chat_profile):
        result = CliRunner().invoke(chat, ["config", "validate"])
        assert result.exit_code == 1
        assert "config init" in result.output

    def test_a_dead_key_is_reported(self, chat_profile):
        chat_profile.write_text('chroma_path = "/gone"\n', encoding="utf-8")
        result = CliRunner().invoke(chat, ["config", "validate"])
        assert result.exit_code == 1
        assert "chroma_path" in result.output

    def test_a_credential_at_a_permissive_mode_is_reported(self, chat_profile):
        """A real finding on a shared node, not a style note."""
        chat_profile.write_text('remote_llm_api_key = "sk-live"\n', encoding="utf-8")
        chat_profile.chmod(0o644)
        result = CliRunner().invoke(chat, ["config", "validate"])
        assert result.exit_code == 1
        assert "0644" in result.output
        assert "chmod 600" in result.output

    def test_a_permissive_profile_with_no_credential_is_fine(self, chat_profile):
        """Only the presence of a key makes the mode a problem."""
        chat_profile.write_text("chunk_size = 64\n", encoding="utf-8")
        chat_profile.chmod(0o644)
        assert config.validate_profile(chat_profile) == []
