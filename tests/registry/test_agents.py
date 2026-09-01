"""Tests for the agents registry."""

import logging

import click
import pytest

from aorta.registry.agents import BUILTIN_AGENTS, get_agent, load_agents
from aorta.registry.errors import (
    RegistryCollisionError,
    RegistryError,
    UnknownAgentError,
)


@click.command(name="plugin-agent")
def _plugin_agent() -> None:
    """A third-party agent."""


def test_load_agents_includes_builtins(fake_agent_eps):
    fake_agent_eps([])
    result = load_agents()
    assert "mitigate" in result
    assert result["mitigate"].source_package == "aorta"
    assert isinstance(result["mitigate"].command, click.Command)


def test_builtin_targets_are_module_attr_strings():
    # Built-ins register directly here rather than through the entry-point
    # group, which is reserved for third parties. Targets stay strings so
    # `aorta.registry` never imports `aorta.cli`.
    for name, target in BUILTIN_AGENTS.items():
        assert ":" in target, f"built-in agent {name!r} target is not 'module:attr'"


def test_get_agent_returns_the_click_command():
    entry = get_agent("mitigate")
    assert entry.name == "mitigate"
    assert entry.command.name == "mitigate"


def test_get_agent_unknown_raises_with_helpful_message():
    with pytest.raises(UnknownAgentError) as exc:
        get_agent("not_a_real_agent")
    msg = str(exc.value)
    assert "available:" in msg
    assert "plugin" in msg
    # str() must NOT wrap the message in quotes (KeyError's default repr).
    assert not msg.startswith("'") and not msg.endswith("'")


def test_load_agents_discovers_plugin(fake_agent_eps):
    fake_agent_eps([("autopsy", _plugin_agent, "fake_plugin")])
    result = load_agents()
    assert result["autopsy"].command is _plugin_agent
    assert result["autopsy"].source_package == "fake_plugin"


def test_collision_plugin_vs_builtin_raises(fake_agent_eps):
    fake_agent_eps([("mitigate", _plugin_agent, "plugin_x")])
    with pytest.raises(RegistryCollisionError, match="aorta.*plugin_x"):
        load_agents()


def test_collision_between_plugins_raises(fake_agent_eps):
    fake_agent_eps([
        ("autopsy", _plugin_agent, "plugin_a"),
        ("autopsy", _plugin_agent, "plugin_b"),
    ])
    with pytest.raises(RegistryCollisionError, match="plugin_a.*plugin_b"):
        load_agents()


def test_plugin_that_fails_to_load_is_logged_and_skipped(fake_agent_eps, caplog):
    # One broken third-party package must not take the whole namespace down.
    fake_agent_eps([
        ("broken", ImportError("no such module"), "plugin_bad"),
        ("autopsy", _plugin_agent, "plugin_good"),
    ])
    with caplog.at_level(logging.WARNING, logger="aorta.registry.agents"):
        result = load_agents()
    assert "broken" not in result
    assert "autopsy" in result
    assert "mitigate" in result
    assert "broken" in caplog.text and "plugin_bad" in caplog.text


def test_plugin_that_is_not_a_command_is_logged_and_skipped(fake_agent_eps, caplog):
    fake_agent_eps([("bogus", lambda: None, "plugin_bad")])
    with caplog.at_level(logging.WARNING, logger="aorta.registry.agents"):
        result = load_agents()
    assert "bogus" not in result
    assert "click.Command" in caplog.text


def test_malformed_builtin_target_raises(monkeypatch):
    # A broken built-in is an aorta bug, not a plugin problem: it must surface
    # rather than be silently skipped like a plugin load failure.
    monkeypatch.setitem(BUILTIN_AGENTS, "busted", "aorta.cli.agent_mitigate")
    with pytest.raises(RegistryError, match="module:attr"):
        load_agents()


def test_builtin_that_is_not_a_command_raises(monkeypatch):
    monkeypatch.setitem(BUILTIN_AGENTS, "busted", "aorta.cli.agent_mitigate:COMMAND_LABEL")
    with pytest.raises(RegistryError, match="click.Command"):
        load_agents()
