"""Options given before the subcommand name have to reach the subcommand.

``aorta chat`` carries ``--llm-provider``, ``--llm-model``, ``--no-redact`` and
``--no-wait`` on the group. The group callback returned as soon as it saw a
subcommand, so every one of them parsed cleanly and was then discarded:
``aorta chat --llm-provider openai ask q`` used the profile's provider, and
``ui`` -- which has no duplicate options of its own -- could not be given them
at all while still advertising them in ``--help``.
"""

from __future__ import annotations

import importlib.util

import pytest
from click.testing import CliRunner

from aorta.cli.chat import chat

_CHAT_AVAILABLE = importlib.util.find_spec("langchain_core") is not None

pytestmark = pytest.mark.skipif(not _CHAT_AVAILABLE, reason="amd-aorta[chat-cli] not installed")


class _Dispatched(list):
    """The recorded ``_dispatch`` calls, with a readable accessor.

    ``only()`` fails with "not called" rather than an ``IndexError`` when a
    regression stops the subcommand dispatching at all, which is one of the
    ways this could break.
    """

    def only(self) -> dict:
        assert len(self) == 1, f"expected one _dispatch call, got {len(self)}"
        return self[0]


@pytest.fixture()
def dispatched(monkeypatch) -> _Dispatched:
    """Capture what ``_dispatch`` was called with, without running a query."""
    from aorta.cli import chat as chat_cli

    calls = _Dispatched()

    def _record(query, as_json, plain, llm_provider, llm_model, no_wait, no_redact, verbose):
        calls.append(
            {
                "query": query,
                "as_json": as_json,
                "plain": plain,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "no_wait": no_wait,
                "no_redact": no_redact,
                "verbose": verbose,
            }
        )

    monkeypatch.setattr(chat_cli, "_dispatch", _record)
    return calls


class TestAskInheritsGroupOptions:
    def test_a_provider_before_the_subcommand_is_applied(self, dispatched):
        result = CliRunner().invoke(chat, ["--llm-provider", "openai", "ask", "q"])
        assert result.exit_code == 0, result.output
        assert dispatched.only()["llm_provider"] == "openai"

    def test_a_model_before_the_subcommand_is_applied(self, dispatched):
        CliRunner().invoke(chat, ["--llm-model", "gpt-4o", "ask", "q"])
        assert dispatched.only()["llm_model"] == "gpt-4o"

    @pytest.mark.parametrize("flag", ["--no-redact", "--no-wait", "--json", "--plain", "-v"])
    def test_every_group_flag_reaches_the_subcommand(self, dispatched, flag):
        CliRunner().invoke(chat, [flag, "ask", "q"])
        assert any(dispatched.only().values())

    def test_the_option_after_the_subcommand_still_wins(self, dispatched):
        """The more specific of the two positions is the one the user meant."""
        CliRunner().invoke(
            chat, ["--llm-provider", "openai", "ask", "--llm-provider", "vllm", "q"]
        )
        assert dispatched.only()["llm_provider"] == "vllm"

    def test_the_subcommand_alone_is_unaffected(self, dispatched):
        CliRunner().invoke(chat, ["ask", "--llm-provider", "openai", "q"])
        assert dispatched.only()["llm_provider"] == "openai"

    def test_no_options_anywhere_stays_at_the_defaults(self, dispatched):
        """A profile's own values must not be overridden by a phantom flag."""
        CliRunner().invoke(chat, ["ask", "q"])
        assert dispatched.only() == {
            "query": "q",
            "as_json": False,
            "plain": False,
            "llm_provider": None,
            "llm_model": None,
            "no_wait": False,
            "no_redact": False,
            "verbose": False,
        }


class TestTheRealEntryPointGetsThemToo:
    """Every test above invokes ``chat`` directly, which hides the bug.

    Invoked that way, ``chat`` *is* the root context, so ``find_root().obj``
    happened to be the ``_GroupOptions`` this module is about. Under the real
    ``aorta`` entry point the root is the top-level group, whose ``obj`` is
    something else entirely -- so the lookup missed and every group option was
    silently dropped for the invocation form users actually type.
    """

    def test_a_provider_survives_the_top_level_group(self, dispatched):
        from aorta.cli import main

        result = CliRunner().invoke(main, ["chat", "--llm-provider", "openai", "ask", "q"])

        assert result.exit_code == 0, result.output
        assert dispatched.only()["llm_provider"] == "openai"

    @pytest.mark.parametrize(
        ("flag", "field"),
        [
            ("--no-redact", "no_redact"),
            ("--no-wait", "no_wait"),
            ("--json", "as_json"),
            ("--plain", "plain"),
            ("-v", "verbose"),
        ],
    )
    def test_every_group_flag_survives_the_top_level_group(self, dispatched, flag, field):
        from aorta.cli import main

        CliRunner().invoke(main, ["chat", flag, "ask", "q"])

        assert dispatched.only()[field] is True

    def test_the_option_after_the_subcommand_still_wins(self, dispatched):
        from aorta.cli import main

        CliRunner().invoke(
            main, ["chat", "--llm-provider", "openai", "ask", "--llm-provider", "vllm", "q"]
        )

        assert dispatched.only()["llm_provider"] == "vllm"

    def test_no_options_anywhere_stays_at_the_defaults(self, dispatched):
        """The walk must not pick up an unrelated ``obj`` from an outer context."""
        from aorta.cli import main

        CliRunner().invoke(main, ["chat", "ask", "q"])

        assert dispatched.only()["llm_provider"] is None
        assert dispatched.only()["no_redact"] is False


class TestUiReceivesThemThroughTheEnvironment:
    """The Chainlit child is a fresh interpreter, so overrides travel as env."""

    @staticmethod
    def _env_for(argv: list[str]) -> dict[str, str]:
        """Invoke ``ui`` and return the environment it handed the child.

        ``subprocess`` and ``importlib.util`` are imported inside the command
        body, so the patches land on the modules themselves rather than on
        names in ``cli.chat``. Chainlit is stubbed because the ``chat-ui``
        extra is a separate install, and what is under test here is the
        environment handed over rather than the server that receives it.
        """
        import importlib.util
        import subprocess
        from types import SimpleNamespace

        captured: dict[str, dict[str, str]] = {}

        def _call(cmd, env=None):  # noqa: ARG001 - signature match
            captured["env"] = env or {}
            return 0

        real_find_spec = importlib.util.find_spec

        def _find_spec(name, *args, **kwargs):
            if name == "chainlit":
                return SimpleNamespace(origin="/stub/chainlit/__init__.py")
            if name == "aorta.chat.ui.app":
                return SimpleNamespace(origin="/stub/aorta/chat/ui/app.py")
            return real_find_spec(name, *args, **kwargs)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(subprocess, "call", _call)
            monkeypatch.setattr(importlib.util, "find_spec", _find_spec)
            result = CliRunner().invoke(chat, argv)
        assert result.exit_code == 0, result.output
        return captured["env"]

    def test_a_provider_is_passed_to_the_child(self):
        env = self._env_for(["--llm-provider", "openai", "ui"])
        assert env.get("AORTA_CHAT_LLM_PROVIDER") == "openai"

    def test_no_redact_is_passed_to_the_child(self):
        env = self._env_for(["--no-redact", "ui"])
        assert env.get("AORTA_CHAT_REDACT") == "false"

    def test_a_model_lands_in_the_resolved_providers_field(self):
        """``--llm-model`` does not say whose model it is; the provider decides."""
        env = self._env_for(["--llm-provider", "openai", "--llm-model", "gpt-4o", "ui"])
        assert env.get("AORTA_CHAT_REMOTE_LLM_MODEL") == "gpt-4o"
        assert "AORTA_CHAT_VLLM_MODEL" not in env

    def test_a_vllm_model_lands_in_the_vllm_field(self):
        env = self._env_for(["--llm-provider", "vllm", "--llm-model", "my/local", "ui"])
        assert env.get("AORTA_CHAT_VLLM_MODEL") == "my/local"

    def test_no_wait_and_verbose_travel_as_their_own_variables(self):
        from aorta.cli.chat import UI_NO_WAIT_ENV, UI_VERBOSE_ENV

        env = self._env_for(["--no-wait", "-v", "ui"])
        assert env[UI_NO_WAIT_ENV] == "1"
        assert env[UI_VERBOSE_ENV] == "1"

    def test_nothing_is_injected_when_no_options_were_given(self):
        env = self._env_for(["ui"])
        from aorta.cli.chat import UI_NO_WAIT_ENV, UI_VERBOSE_ENV

        for name in (
            "AORTA_CHAT_LLM_PROVIDER",
            "AORTA_CHAT_REDACT",
            UI_NO_WAIT_ENV,
            UI_VERBOSE_ENV,
        ):
            assert name not in env

    def test_the_childs_own_environment_is_preserved(self, monkeypatch):
        """Replacing rather than extending it would strip HF_HOME and friends."""
        monkeypatch.setenv("HF_HOME", "/shared/hf-cache")
        env = self._env_for(["--no-redact", "ui"])
        assert env.get("HF_HOME") == "/shared/hf-cache"


class TestOptionsTheUiCannotHonour:
    """Refused out loud, rather than accepted and dropped."""

    @pytest.mark.parametrize("flag", ["--json", "--plain"])
    def test_an_output_mode_flag_is_refused_rather_than_ignored(self, flag):
        result = CliRunner().invoke(chat, [flag, "ui"])
        assert result.exit_code != 0
        assert "web server" in result.output


class TestTheEnvNamesDoNotDrift:
    def test_the_cli_and_the_chat_package_agree(self):
        """``cli/chat.py`` may not import the chat package at module scope.

        So the names are written twice, the same way ``_LLM_PROVIDERS`` is, and
        this is the test that keeps the two copies honest.
        """
        from aorta.chat import config
        from aorta.cli import chat as chat_cli

        assert chat_cli.UI_NO_WAIT_ENV == config.UI_NO_WAIT_ENV
        assert chat_cli.UI_VERBOSE_ENV == config.UI_VERBOSE_ENV
