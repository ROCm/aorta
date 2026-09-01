"""CLI surface tests for ``aorta chat``.

Tests the shim contract only, in the same spirit as ``test_bench.py``: the
command is registered, its flags parse, and a missing extra produces a sentence
rather than a traceback. Nothing here invokes the agent.
"""

from __future__ import annotations

import importlib.util
from collections import namedtuple

import pytest
from click.testing import CliRunner

from aorta.cli import main
from aorta.cli.chat import chat

_CHAT_AVAILABLE = importlib.util.find_spec("langchain_core") is not None


def test_chat_help_exits_zero() -> None:
    result = CliRunner().invoke(chat, ["--help"])
    assert result.exit_code == 0, result.output
    for subcommand in ("ask", "ui"):
        assert subcommand in result.output


def test_chat_wired_under_main() -> None:
    """Validates the ``aorta/cli/__init__.py`` wiring."""
    result = CliRunner().invoke(main, ["chat", "--help"])
    assert result.exit_code == 0, result.output


def test_json_and_plain_are_mutually_exclusive() -> None:
    result = CliRunner().invoke(chat, ["ask", "--json", "--plain", "hello"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_llm_provider_choice_is_validated() -> None:
    """The Choice list is hard-coded, so a typo must still be rejected."""
    result = CliRunner().invoke(chat, ["ask", "--llm-provider", "nope", "hello"])
    assert result.exit_code != 0
    assert "nope" in result.output


def test_llm_provider_choice_matches_the_provider_factory() -> None:
    """The hard-coded ``click.Choice`` must not drift from the registry.

    It cannot be derived: decorators run at import time, and enumerating the
    factory there would import langchain on every ``aorta --help``. So the
    duplication is deliberate and this is the guard on it.
    """
    if not _CHAT_AVAILABLE:
        pytest.skip("amd-aorta[chat-cli] not installed")
    from aorta.chat.inference.providers.factory import available_providers
    from aorta.cli.chat import _LLM_PROVIDERS

    assert tuple(sorted(_LLM_PROVIDERS)) == available_providers()


@pytest.mark.skipif(_CHAT_AVAILABLE, reason="chat extra installed -- hint path not active")
def test_missing_extra_shows_an_install_hint() -> None:
    """Without the extra, ``aorta chat ask`` advises rather than traces back."""
    result = CliRunner().invoke(chat, ["ask", "hello"])
    assert result.exit_code != 0
    assert "amd-aorta[chat-cli]" in result.output


@pytest.mark.skipif(
    importlib.util.find_spec("chainlit") is not None,
    reason="chainlit installed -- neither error path is active",
)
def test_ui_names_the_python_ceiling_rather_than_looping(monkeypatch) -> None:
    """A py3.14 user must not be told to install an extra that installs nothing.

    Chainlit declares ``Requires-Python < 3.14``, so on 3.14
    ``pip install amd-aorta[chat-ui]`` succeeds and resolves to nothing. Advising
    it would send the user round a loop.
    """
    from aorta.cli import chat as cli_chat

    # sys.version_info is not constructible, and a bare tuple would not carry
    # the .major/.minor the message formats.
    fake = namedtuple("version_info", "major minor micro releaselevel serial")
    monkeypatch.setattr(cli_chat.sys, "version_info", fake(3, 14, 0, "final", 0))
    result = CliRunner().invoke(chat, ["ui"])
    assert result.exit_code != 0
    assert "Chainlit does not support Python 3.14" in result.output
    assert "amd-aorta[chat-ui]" not in result.output
