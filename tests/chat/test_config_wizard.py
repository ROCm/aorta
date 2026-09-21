"""``aorta chat config init|show|validate``, and the two guards Decision 9b owes.

Writing an API key into a predictable path inside a tool whose day job is
collecting diagnostic bundles is only safe with both guards in place:
``config show`` masks the key, and ``aorta bundle`` refuses to package the file.
The likeliest leak is a customer pasting their own config into a support ticket,
so the masking is tested as a contract, not as formatting.
"""

from __future__ import annotations

import contextlib
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
import tomllib
from click.testing import CliRunner
from pydantic import ValidationError

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


def _unreachable_remote_embedding_fields(
    template: dict[str, object], prompted: tuple[str, ...]
) -> list[str]:
    """Settings a remote-embedding template needs but neither sets nor asks for.

    Empty for a template that does not select remote embeddings at all: the
    fields are read by nothing then.

    A template key only counts as set when it carries a value. ``PROFILE_TEMPLATES``
    already uses ``""`` as a placeholder for a field the wizard is expected to fill
    (``openai-compatible``'s ``remote_llm_base_url``), so counting bare presence
    would let ``remote_embedding_api_key = ""`` satisfy a guard whose whole point
    is that the key must be reachable. A placeholder is only honest when the
    wizard asks for it, which the ``prompted`` half covers.
    """
    if template.get("embedding_provider") != "remote":
        return []
    collected = {key for key, value in template.items() if value} | set(prompted)
    return sorted({"remote_embedding_base_url", "remote_embedding_api_key"} - collected)


class TestNoTemplateOptsIntoRemoteEmbeddings:
    """Four templates coupled the embedding provider to the chat provider.

    Choosing a remote *chat* model also selected a remote *embedder*, and CI
    publishes the index asset under default settings -- so ``index fetch``
    refused the asset outright when ``config init`` ran first, and the manifest
    refused every query when it ran second. There was no ordering of the two
    documented commands that worked, on four of the five profiles.

    The templates now all choose the local embedder. These pin the parts of that
    which a later edit could quietly undo.
    """

    @pytest.mark.parametrize("name", sorted(config.PROFILE_TEMPLATES))
    def test_the_embedding_choice_is_written_down(self, name):
        """Explicit in the template, not inherited from the field default.

        The profile file is what a user reads to find out what was decided for
        them, so the decision belongs in it.
        """
        assert "embedding_provider" in config.PROFILE_TEMPLATES[name]

    @pytest.mark.parametrize("name", sorted(config.PROFILE_TEMPLATES))
    def test_every_template_matches_the_provider_the_published_index_uses(self, name):
        """The published asset is built with defaults, so agreeing with the
        default is exactly what makes ``index fetch`` usable.

        Asserting against the field default rather than the literal ``"local"``
        keeps this true if the default itself ever moves: what breaks fetch is
        the two disagreeing, not the particular value.
        """
        default = config.Settings.model_fields["embedding_provider"].default
        assert config.PROFILE_TEMPLATES[name].get("embedding_provider") == default

    def test_no_template_selects_remote_embeddings_it_cannot_reach(self):
        """The durable half of the guard, and what closes #443 and gap 1a.

        Stated as a rule rather than as "they are all local" so it keeps
        meaning if a later template deliberately picks remote: what must never
        ship is a template that selects a remote embedder without an endpoint
        and a key.
        """
        for name in sorted(config.PROFILE_TEMPLATES):
            missing = _unreachable_remote_embedding_fields(
                config.PROFILE_TEMPLATES[name], config.PROFILE_PROMPTS[name]
            )
            assert not missing, f"{name}: remote embeddings but no {missing}"

    def test_the_rule_catches_the_shape_that_was_reported(self):
        """Proves the guard above is not vacuously green while every template is local.

        This is the ``azure-apim`` template as it shipped: an APIM subscription
        header for embeddings, no base URL -- which resolves to
        ``api.openai.com``, so the corpus was addressed to OpenAI with a header
        it does not read -- and no key ever collected. It failed closed only
        because of the unrelated missing-key check in ``remote_api.py``.
        """
        missing = _unreachable_remote_embedding_fields(
            {
                "llm_provider": "openai",
                "remote_llm_auth_header": "Ocp-Apim-Subscription-Key",
                "embedding_provider": "remote",
                "remote_embedding_auth_header": "Ocp-Apim-Subscription-Key",
            },
            ("remote_llm_base_url", "remote_llm_model", "remote_llm_api_key"),
        )
        assert missing == ["remote_embedding_api_key", "remote_embedding_base_url"]

    def test_an_empty_placeholder_nobody_prompts_for_is_not_a_setting(self):
        """The decoy: the shape a loose guard would wave through.

        Both fields are present, so a presence-only check reports nothing
        missing -- while ``remote_api.py`` raises on the empty key and the empty
        base URL resolves to ``api.openai.com``. Only the prompted placeholder
        is legitimate, and that one is left out of the expectation.
        """
        missing = _unreachable_remote_embedding_fields(
            {
                "embedding_provider": "remote",
                "remote_embedding_base_url": "",
                "remote_embedding_api_key": "",
            },
            ("remote_embedding_base_url",),
        )
        assert missing == ["remote_embedding_api_key"]

    @pytest.mark.parametrize("name", sorted(config.PROFILE_TEMPLATES))
    def test_a_local_template_carries_no_remote_embedding_settings(self, name):
        """A leftover ``remote_embedding_*`` key is inert but reads as intent.

        ``openai`` shipped ``remote_embedding_model`` and ``azure-apim`` a
        ``remote_embedding_auth_header``, both unread once the provider is
        local. Left in the written profile they suggest a remote embedder is
        configured and working.
        """
        template = config.PROFILE_TEMPLATES[name]
        if template.get("embedding_provider") != "local":
            pytest.skip(f"{name} does not select local embeddings")
        dead = [key for key in template if key.startswith("remote_embedding_")]
        assert not dead, f"{name}: {sorted(dead)}"


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

    def test_a_permissive_existing_profile_is_tightened_before_the_key_lands(self, chat_profile):
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

    @pytest.mark.parametrize("name", sorted(config.PROFILE_TEMPLATES))
    def test_it_says_what_it_decided_about_embeddings(self, name, chat_profile):
        """The wizard asks about the chat model and nothing else.

        The embedding provider is therefore chosen on the user's behalf, and it
        is the setting that decides whether the published index can be used --
        so leaving them to ``cat`` the file for it is how a first-time setup
        ends in a refused query with no idea which command caused it.
        """
        result = CliRunner().invoke(chat, ["config", "init", "--profile", name, "--no-input"])
        assert result.exit_code == 0, result.output
        assert "Embeddings: local" in result.output
        assert "index fetch" in result.output

    def test_it_names_the_build_when_a_template_picks_a_remote_embedder(
        self, monkeypatch, chat_profile
    ):
        """The other arm, by the route no shipped template takes today.

        Every template is local, so only an environment override
        (``test_the_advice_follows_the_environment_and_not_the_template``)
        reaches this branch on a stock install. Pinning the template route as
        well is what keeps the message naming the right command on the day one
        deliberately picks remote -- the case
        ``test_no_template_selects_remote_embeddings_it_cannot_reach`` is
        written to keep allowing.

        The key is set because ``index build`` is only the right advice once it
        can run; without one the message names the missing key instead, which is
        ``test_a_remote_provider_with_no_key_is_not_sent_to_index_build``.
        """
        monkeypatch.setitem(
            config.PROFILE_TEMPLATES,
            "openai",
            {**config.PROFILE_TEMPLATES["openai"], "embedding_provider": "remote"},
        )
        monkeypatch.setenv("AORTA_CHAT_REMOTE_EMBEDDING_API_KEY", "sk-test")
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "init", "--profile", "openai", "--no-input"])
        assert result.exit_code == 0, result.output
        assert "Embeddings: remote" in result.output
        assert "index build" in result.output
        assert "Embeddings: local" not in result.output

    def test_the_advice_follows_the_environment_and_not_the_template(
        self, monkeypatch, chat_profile
    ):
        """``AORTA_CHAT_*`` outranks the file ``config init`` has just written.

        Reporting the template would tell a user with the override exported
        that local embeddings were configured and send them to ``index fetch``,
        which then refuses the published asset -- advice their own environment
        does not let them follow, which is the failure this command was changed
        to stop rather than to reproduce.

        A key is exported alongside the provider so this stays a test about
        *which* command is named. Without one the arm below fires instead, and
        the reason it is a separate test is that the keyless combination is the
        common one here, not an edge: the profile just written is local and
        carries no remote key.
        """
        monkeypatch.setenv("AORTA_CHAT_EMBEDDING_PROVIDER", "remote")
        monkeypatch.setenv("AORTA_CHAT_REMOTE_EMBEDDING_API_KEY", "sk-test")
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "init", "--profile", "openai", "--no-input"])
        assert result.exit_code == 0, result.output
        assert "Embeddings: remote" in result.output
        assert "index build" in result.output
        assert "index fetch" not in result.output

    def test_a_remote_provider_with_no_key_is_not_sent_to_index_build(
        self, monkeypatch, chat_profile
    ):
        """``index build`` cannot run without a key, so it must not be named.

        ``RemoteApiProvider.get_embeddings()`` raises on an empty
        ``remote_embedding_api_key`` before it sends anything, so the command is
        known to fail from the settings alone -- the same class of unfollowable
        advice this whole path exists to remove, one case over.

        This is the *reachable* combination rather than a constructed one: the
        profile ``config init`` has just written is local and carries no remote
        key, so exporting the provider alone lands here. Both sibling tests
        above had to be given a key once this arm existed, which is how reachable
        it is.
        """
        monkeypatch.setenv("AORTA_CHAT_EMBEDDING_PROVIDER", "remote")
        monkeypatch.delenv("AORTA_CHAT_REMOTE_EMBEDDING_API_KEY", raising=False)
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "init", "--profile", "openai", "--no-input"])
        assert result.exit_code == 0, result.output
        assert "Embeddings: remote" in result.output
        assert "AORTA_CHAT_REMOTE_EMBEDDING_API_KEY" in result.output
        # Neither index command is offered, because neither can succeed yet.
        assert "index build" not in result.output
        assert "index fetch" not in result.output

    def test_it_says_where_the_override_came_from(self, monkeypatch, chat_profile):
        """Otherwise the line contradicts the file it was just told was written."""
        monkeypatch.setenv("AORTA_CHAT_EMBEDDING_PROVIDER", "remote")
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "init", "--profile", "openai", "--no-input"])
        assert result.exit_code == 0, result.output
        assert "AORTA_CHAT_EMBEDDING_PROVIDER" in result.output
        assert 'embedding_provider = "local"' in chat_profile.read_text(encoding="utf-8")

    def test_a_hand_set_local_model_does_not_get_promised_fetch_either(
        self, monkeypatch, chat_profile
    ):
        """The provider alone is not the test the manifest applies.

        ``rag/manifest.validate`` refuses on the model name and on the
        embedding identity, which for the local provider *is* the model name --
        so a local install on another model is refused exactly like a remote
        one, and must not be sent to ``index fetch``.
        """
        monkeypatch.setenv("AORTA_CHAT_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "init", "--profile", "openai", "--no-input"])
        assert result.exit_code == 0, result.output
        assert "Embeddings: local" in result.output
        assert "bge-large-en-v1.5" in result.output
        assert "index build" in result.output
        assert "index fetch" not in result.output

    @pytest.mark.parametrize(
        ("variable", "value"),
        [("AORTA_CHAT_CHUNK_SIZE", "1024"), ("AORTA_CHAT_CHUNK_OVERLAP", "999")],
    )
    def test_chunk_drift_does_not_withdraw_the_fetch(
        self, monkeypatch, chat_profile, variable, value
    ):
        """Chunk parameters are not part of the compatibility decision.

        ``manifest.validate`` reports them as warnings, not refusals, and
        ``fetch_index`` gates on ``raise_if_refused`` alone -- verified by
        driving ``fetch_index`` against a published manifest built at other chunk
        values, which installs and warns. So a fetch stays followable here, and
        withdrawing it would send a reader to ``index build`` -- a full local
        re-embed -- to avoid a warning about span sizes.
        """
        monkeypatch.setenv(variable, value)
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "init", "--profile", "openai", "--no-input"])
        assert result.exit_code == 0, result.output
        assert "index fetch" in result.output
        assert "index build" not in result.output

    @pytest.mark.parametrize(
        "spelling",
        sorted(
            name
            for name, flow in config.EMBEDDING_PROVIDER_FLOWS.items()
            if flow == "local" and name != "local"
        ),
    )
    def test_an_accepted_spelling_of_local_is_still_local(
        self, spelling, monkeypatch, chat_profile
    ):
        """``onnx`` and ``fastembed`` resolve to the local flow in the factory.

        Reading the setting as an opaque string would tell these users the
        published index does not match theirs and send them to a build they do
        not need -- the same unfollowable advice, pointed the other way.
        """
        monkeypatch.setenv("AORTA_CHAT_EMBEDDING_PROVIDER", spelling)
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "init", "--profile", "openai", "--no-input"])
        assert result.exit_code == 0, result.output
        assert "index fetch" in result.output
        assert "index build" not in result.output

    def test_the_provider_table_matches_the_factory(self):
        """Duplicated because importing the factory would pull in langchain_core.

        ``config init`` has no other reason to load it, so the resolution table
        is repeated rather than imported -- the same arrangement as
        ``_CONFIG_PROFILES``, and this is the guard on it. Checked as an
        equality over the whole table, not just the local half: a provider the
        factory grew and this copy did not would otherwise be reported as a
        name aorta does not know.
        """
        from aorta.chat.rag.embeddings import factory

        assert {
            **{name: name for name in factory._PROVIDERS},
            **factory._ALIASES,
        } == config.EMBEDDING_PROVIDER_FLOWS

    def test_an_unknown_provider_is_named_rather_than_guessed_at(
        self, monkeypatch, chat_profile
    ):
        """Neither command works, so neither may be recommended.

        The factory raises on a name it does not recognise, so reporting a
        typo'd provider as though it were remote and sending the user to
        ``index build`` is the same unfollowable advice one layer down.
        """
        monkeypatch.setenv("AORTA_CHAT_EMBEDDING_PROVIDER", "remvote")
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "init", "--profile", "openai", "--no-input"])
        assert result.exit_code == 0, result.output
        assert "remvote" in result.output
        assert "index build" not in result.output
        assert "index fetch" not in result.output

    def test_a_broken_environment_does_not_traceback_over_a_written_profile(
        self, monkeypatch, chat_profile
    ):
        """The file is already on disk by the time the settings are merged.

        An unrelated bad ``AORTA_CHAT_*`` value used to be invisible here,
        because the command never built ``Settings``. Now that it does, it must
        report the problem rather than exit non-zero over a write that worked.
        """
        monkeypatch.setenv("AORTA_CHAT_LLM_TIMEOUT", "not-a-number")
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "init", "--profile", "openai", "--no-input"])
        assert result.exit_code == 0, result.output
        assert chat_profile.exists()
        assert "config validate" in result.output

    @pytest.mark.parametrize(
        "value",
        [
            "Authorization: Bearer sk-SUPERSECRET-abcd1234",
            '{"Authorization": "Bearer sk-SUPERSECRET-abcd1234"',
        ],
        ids=["comma-form", "json-form"],
    )
    def test_a_rejected_extra_header_is_not_echoed_back(self, value, monkeypatch, chat_profile):
        """A malformed header map must not print the credential it carries.

        ``str(ValidationError)`` embeds pydantic's ``input_value``, so both
        spellings of a bad value reached the terminal with the key intact --
        from the one command a user runs immediately after typing one in, and
        for a field ``config show`` masks. The field name still has to appear,
        or the line is not actionable.
        """
        monkeypatch.setenv("AORTA_CHAT_REMOTE_EMBEDDING_EXTRA_HEADERS", value)
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "init", "--profile", "openai", "--no-input"])
        assert result.exit_code == 0, result.output
        assert "SUPERSECRET" not in result.output
        assert "input_value" not in result.output
        assert "remote_embedding_extra_headers" in result.output

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
        monkeypatch.setenv("AORTA_CHAT_REMOTE_LLM_EXTRA_HEADERS", f"x-api-key={secret}")
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "show", "--json"])
        assert result.exit_code == 0, result.output
        assert secret not in result.output

    def test_reveal_still_prints_it(self, chat_profile, monkeypatch):
        secret = "gw-abcdefgh12345678"
        monkeypatch.setenv("AORTA_CHAT_REMOTE_LLM_EXTRA_HEADERS", f"x-api-key={secret}")
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
        chat_profile.write_text(f'{field} = {{ "x-api-key" = "gw-live" }}\n', encoding="utf-8")
        chat_profile.chmod(0o644)
        problems = config.validate_profile(chat_profile)
        assert any("0644" in p and "chmod 600" in p for p in problems), problems

    @pytest.mark.parametrize("field", sorted(config.SECRET_MAPPING_FIELDS))
    def test_an_empty_extra_header_map_does_not_demand_0600(self, field, chat_profile):
        """Presence of the key, not the field being declared, is the trigger."""
        chat_profile.write_text(f"{field} = {{}}\n", encoding="utf-8")
        chat_profile.chmod(0o644)
        assert config.validate_profile(chat_profile) == []

    @pytest.mark.parametrize(
        "value",
        [
            "Authorization: Bearer sk-SUPERSECRET-abcd1234",
            '{"Authorization": "Bearer sk-SUPERSECRET-abcd1234"',
        ],
        ids=["comma-form", "json-form"],
    )
    def test_a_rejected_extra_header_is_not_echoed_back(self, value, monkeypatch, chat_profile):
        """The command ``config init`` points at must not leak what it warned about.

        ``config init`` answers a rejected credential with "sort the environment
        out first: aorta chat config validate". This is that command, and it
        interpolated the same raw ``ValidationError`` -- so following the advice
        printed the key. The field name still has to appear, or the report is
        not actionable.
        """
        config.write_profile({"chunk_size": 128}, chat_profile)
        monkeypatch.setenv("AORTA_CHAT_REMOTE_EMBEDDING_EXTRA_HEADERS", value)
        config.reset_settings()
        result = CliRunner().invoke(chat, ["config", "validate"])
        assert result.exit_code == 1, result.output
        assert "SUPERSECRET" not in result.output
        assert "input_value" not in result.output
        assert "remote_embedding_extra_headers" in result.output

    def test_the_validator_names_the_position_not_the_value(self, chat_profile):
        """The message is rendered by consumers this module does not control.

        ``ValidationError`` is a ``ValueError``, so the two ``str(exc)``
        handlers in ``cli/chat.py`` catch this one too. Keeping the rejected
        pair out of the message is what stops it surfacing there; the position
        is what a user counts commas to find anyway.
        """
        with pytest.raises(ValidationError) as caught:
            config.Settings(
                remote_embedding_extra_headers="user=alice,Authorization: Bearer sk-SECRET"
            )
        rendered = str(caught.value)
        assert "extra header #2 is missing '='" in rendered
        assert "sk-SECRET" not in rendered.split("input_value=")[0]


def _fenced_commands(text: str) -> list[str]:
    """``aorta ...`` lines inside shell code fences, in order.

    Fences are paired by toggling on *any* ``` line rather than by matching a
    language, because pairing only ``` ```bash ``` openers desynchronises on the
    first ``` ```toml ``` block and silently drops most of the page -- the first
    draft of this helper checked 4 of the 9 commands and passed.
    """
    commands: list[str] = []
    inside, language = False, ""
    for line in text.splitlines():
        if line.startswith("```"):
            inside, language = (False, "") if inside else (True, line[3:].strip())
            continue
        if inside and language in ("bash", "sh", "console", ""):
            stripped = line.split("#")[0].strip()
            if stripped.startswith("aorta "):
                commands.append(stripped)
    return commands


#: The pages this PR owns. A fenced command is an instruction, so it is held to
#: the strict reading: it must exist and it must parse as printed.
_DOC_PAGES = ("docs/chat/configuration.md", "docs/chat/providers.md")


class TestEveryCommandTheseDocsPrintCanRun:
    """A command in a fenced block is something a reader will paste.

    Six findings in this batch were advice naming a command that could not run
    in the state that printed it, and the round that prompted this one asked for
    ``index fetch --force`` -- a flag that is correct only *after* a sibling PR
    lands and would be rejected by Click until then. Prose can be checked by a
    reader; a flag that does not exist yet cannot, so it is asked of Click's own
    parser here.

    Deliberately narrow. The equivalent gate over *report* text is #463's
    ``TestEveryCommandTheReportNamesCanRun``, which owns the extractor for
    commands named in prose; duplicating that here would leave two copies to
    diverge. This checks the fenced instructions on the two pages this PR owns.
    """

    def _commands(self) -> list[tuple[str, str]]:
        root = Path(__file__).resolve().parents[2]
        found = []
        for page in _DOC_PAGES:
            text = (root / page).read_text(encoding="utf-8")
            found += [(page, command) for command in _fenced_commands(text)]
        return found

    def test_the_extractor_still_finds_the_commands(self):
        """Without this the sweep below passes by finding nothing to check.

        The count is the load-bearing part: the fence-pairing bug this helper
        documents made the sweep pass while reading a quarter of the page.
        """
        commands = self._commands()
        assert len(commands) >= 9, f"only {len(commands)} fenced commands; the extractor has broken"

    def test_every_fenced_command_parses_as_printed(self):
        from click import Context

        from aorta.cli import main as cli

        for page, command in self._commands():
            tokens = command.split()
            assert tokens[0] == "aorta"
            remaining = tokens[1:]
            node, parent = cli, None
            with contextlib.ExitStack() as stack:
                # Descend while the next token names a subcommand of this group.
                # Splitting on "looks like an option" instead would drop an
                # option's *value* and turn a valid line into a parse error.
                while remaining and hasattr(node, "get_command"):
                    ctx = stack.enter_context(Context(node, info_name=command, parent=parent))
                    resolved = node.get_command(ctx, remaining[0])
                    if resolved is None:
                        raise AssertionError(
                            f"{page}: '{command}' -- {remaining[0]!r} is not a command"
                        )
                    node, parent = resolved, ctx
                    remaining = remaining[1:]
                assert not hasattr(node, "get_command"), (
                    f"{page}: '{command}' names a group, which prints help rather "
                    "than doing anything"
                )
                # Parses and validates required options without running the
                # callback, so an unknown flag or a missing required option fails
                # here exactly as it would for a reader who pasted the line.
                node.make_context(command, remaining, parent=parent)

    def test_the_sweep_would_reject_a_flag_that_does_not_exist(self):
        """Keeps the strict tier honest: without this the sweep above is vacuous.

        Originally asked with ``--force`` on ``index fetch``, the flag #465 was
        about to add. That made the control expire the moment #465 landed, which
        is the wrong failure -- it says nothing about whether the sweep still
        rejects anything. A flag no one will ever add keeps the control about
        the parser rather than about the schedule.
        """
        from click import Context, NoSuchOption

        from aorta.cli import main as cli

        with Context(cli) as root_ctx:
            chat_group = cli.get_command(root_ctx, "chat")
            with Context(chat_group, parent=root_ctx) as chat_ctx:
                index_group = chat_group.get_command(chat_ctx, "index")
                with Context(index_group, parent=chat_ctx) as index_ctx:
                    fetch = index_group.get_command(index_ctx, "fetch")
                    with pytest.raises(NoSuchOption):
                        fetch.make_context("fetch", ["--not-a-real-flag"], parent=index_ctx)

    def test_the_docs_do_not_promise_the_force_flag_yet(self):
        """The finding itself, pinned where it was raised.

        The going-back procedure is refused by #465's guard because step 3 built
        the index locally. The fix describes that refusal and lets it name its
        own flag, rather than printing a flag this Click does not accept.
        """
        root = Path(__file__).resolve().parents[2]
        text = (root / "docs/chat/configuration.md").read_text(encoding="utf-8")
        assert "index fetch --force" not in text
        assert "the refusal names the flag" in text
