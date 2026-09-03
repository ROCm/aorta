"""``aorta chat config init|show|validate``, and the two guards Decision 9b owes.

Writing an API key into a predictable path inside a tool whose day job is
collecting diagnostic bundles is only safe with both guards in place:
``config show`` masks the key, and ``aorta bundle`` refuses to package the file.
The likeliest leak is a customer pasting their own config into a support ticket,
so the masking is tested as a contract, not as formatting.
"""

from __future__ import annotations

import os
import stat
from unittest.mock import patch

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

    def test_a_permissive_existing_profile_is_tightened_before_the_key_lands(
        self, chat_profile
    ):
        """``O_CREAT``'s mode applies only on create, so an existing file kept its.

        A key written into a 0644 profile and chmodded afterwards is readable by
        every other user on the box for the length of the write. The mode has to
        be set on the descriptor first, which is what makes the guarantee in
        ``write_profile``'s docstring true rather than aspirational.
        """
        from aorta.chat.config import write_profile

        chat_profile.parent.mkdir(parents=True, exist_ok=True)
        chat_profile.write_text("chunk_size = 1\n", encoding="utf-8")
        chat_profile.chmod(0o644)

        observed: list[int] = []
        real_open = open

        def _watching_open(fd, *args, **kwargs):
            # The mode as it stands at the moment the writable handle is made,
            # which is the instant before any byte of the key is written.
            observed.append(stat.S_IMODE(os.fstat(fd).st_mode))
            return real_open(fd, *args, **kwargs)

        with patch("aorta.chat.config.open", _watching_open):
            write_profile({"remote_llm_api_key": "sk-secret"}, path=chat_profile)

        assert observed == [0o600], [oct(m) for m in observed]
        assert stat.S_IMODE(chat_profile.stat().st_mode) == 0o600

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


class TestExtraHeaderValuesAreMasked:
    """``config show`` printed both ``*_extra_headers`` maps verbatim.

    Those values are handed straight to the HTTP client as request headers, so
    a gateway credential lives there as readily as in ``remote_llm_api_key``:
    Azure API Management reads ``Ocp-Apim-Subscription-Key``, Azure OpenAI reads
    ``api-key``, Anthropic reads ``x-api-key``. ``effective_settings`` documents
    that credentials are masked without ``--reveal``, so printing them was a
    promise the code did not keep -- on exactly the profile shape an enterprise
    user has.
    """

    @pytest.mark.parametrize("field", sorted(config.SECRET_MAPPING_FIELDS))
    @pytest.mark.parametrize(
        "header",
        ["Ocp-Apim-Subscription-Key", "api-key", "x-api-key", "Authorization"],
    )
    def test_the_value_is_masked(self, field, header, chat_profile, monkeypatch):
        secret = "gw-abcdefgh12345678"
        monkeypatch.setenv(f"AORTA_CHAT_{field.upper()}", f"{header}={secret}")
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "show"])
        assert result.exit_code == 0, result.output
        assert secret not in result.output
        # The header name stays legible -- it is what makes the output useful,
        # and describe_auth() already reports names in the clear.
        assert header in result.output

    def test_json_output_is_masked_too(self, chat_profile, monkeypatch):
        """--json is the form most likely to be pasted into a ticket."""
        secret = "gw-abcdefgh12345678"
        monkeypatch.setenv(
            "AORTA_CHAT_REMOTE_LLM_EXTRA_HEADERS", f"x-api-key={secret}"
        )
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "show", "--json"])
        assert result.exit_code == 0, result.output
        assert secret not in result.output

    def test_reveal_still_prints_it(self, chat_profile, monkeypatch):
        secret = "gw-abcdefgh12345678"
        monkeypatch.setenv(
            "AORTA_CHAT_REMOTE_LLM_EXTRA_HEADERS", f"x-api-key={secret}"
        )
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "show", "--reveal"])
        assert result.exit_code == 0, result.output
        assert secret in result.output

    def test_a_non_secret_extra_is_masked_as_well(self, chat_profile, monkeypatch):
        """No name-based guessing: we cannot know which header a gateway reads.

        Masking an attribution value such as ``user=alice`` costs nothing --
        ``config show`` is a diagnostic and ``--reveal`` exists -- while a
        denylist of "credential-looking" header names fails open on the one
        convention nobody enumerated.
        """
        monkeypatch.setenv("AORTA_CHAT_REMOTE_LLM_EXTRA_HEADERS", "user=alice-in-eng")
        config.reset_settings()
        values = config.effective_settings()
        assert values.get("remote_llm_extra_headers") == {"user": config.mask("alice-in-eng")}

    def test_the_masking_notice_mentions_headers(self, chat_profile, monkeypatch):
        """The footer said "API keys are masked", which undersold what it does."""
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "show"])
        assert result.exit_code == 0, result.output
        assert "extra-header" in result.output

    def test_every_header_map_on_settings_is_registered(self):
        """A third header field must not be able to arrive unmasked.

        The membership list and the fields it is supposed to cover are two
        spellings of one rule, so the agreement is pinned rather than assumed:
        adding ``remote_rerank_extra_headers`` and forgetting
        SECRET_MAPPING_FIELDS fails here instead of in a support ticket.
        """
        header_fields = {
            name for name in config.Settings.model_fields if name.endswith("_extra_headers")
        }
        assert header_fields == set(config.SECRET_MAPPING_FIELDS)

    def test_the_two_credential_lists_do_not_overlap(self):
        """Scalars are masked whole, maps per value; a field is one or the other."""
        assert not (config.SECRET_FIELDS & config.SECRET_MAPPING_FIELDS)


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

    @pytest.mark.parametrize("field", sorted(config.SECRET_MAPPING_FIELDS))
    def test_a_credential_in_an_extra_header_needs_0600_too(self, field, chat_profile):
        """A profile whose only secret is a gateway header is just as sensitive.

        The mode check keyed on SECRET_FIELDS alone, so this profile was
        reported healthy at 0644 -- the check honouring the field name rather
        than the secret.
        """
        chat_profile.write_text(
            f'{field} = {{ "x-api-key" = "gw-live" }}\n', encoding="utf-8"
        )
        chat_profile.chmod(0o644)
        problems = config.validate_profile(chat_profile)
        assert any("0644" in p and "chmod 600" in p for p in problems), problems

    @pytest.mark.parametrize("field", sorted(config.SECRET_MAPPING_FIELDS))
    def test_an_empty_extra_header_map_does_not_demand_0600(self, field, chat_profile):
        """Presence of the key, not the field being declared, is the trigger."""
        chat_profile.write_text(f"{field} = {{}}\n", encoding="utf-8")
        chat_profile.chmod(0o644)
        assert config.validate_profile(chat_profile) == []
