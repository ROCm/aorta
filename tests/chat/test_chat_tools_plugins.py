"""The ``aorta.chat_tools`` extension point.

Mirrors ``tests/registry/test_agents.py``, because the failure semantics are
deliberately the same three: a plugin that cannot load is logged and skipped, a
plugin of the wrong type is logged and skipped, and a name collision raises.
The asymmetry is the point -- one broken third-party package must not take the
assistant's whole tool surface down, but silently shadowing
``run_terminal_command`` with someone else's idea of it is not a thing to
recover from quietly.

The fourth case is specific to this registry: the entry-point name and the
tool's own ``name`` are two strings in two files and only the second reaches the
model, so a mismatch is a tool that fails on every call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.tools import tool

from aorta.chat.plugins import BUILTIN_CHAT_TOOLS, enabled_builtins, load_chat_tools
from aorta.registry.errors import RegistryCollisionError, RegistryError


@tool
def read_fabric_counters(node: str = "") -> str:
    """Read the fabric performance counters for a node."""
    return f"counters for {node}"


@tool
def list_files(path: str = ".") -> str:
    """A third-party tool that collides with a built-in."""
    return path


@dataclass
class _FakeDist:
    name: str


@dataclass
class _FakeEntryPoint:
    name: str
    payload: Any  # tests pass non-BaseTool payloads to exercise validation
    dist: _FakeDist | None

    def load(self):
        # An exception payload simulates an entry point whose import blows up.
        if isinstance(self.payload, BaseException):
            raise self.payload
        return self.payload


@pytest.fixture()
def fake_chat_tool_eps(monkeypatch):
    """Install fake entry points: fake_chat_tool_eps([(name, payload, dist), ...])."""

    def _install(specs):
        eps = [
            _FakeEntryPoint(
                name=name,
                payload=payload,
                dist=_FakeDist(name=dist) if dist is not None else None,
            )
            for name, payload, dist in specs
        ]
        monkeypatch.setattr("aorta.chat.plugins.entry_points", lambda group: eps)

    return _install


def test_builtins_load_with_no_plugins_installed(fake_chat_tool_eps):
    fake_chat_tool_eps([])
    registry = load_chat_tools()
    assert set(registry) == set(BUILTIN_CHAT_TOOLS)
    assert all(entry.source_package == "aorta" for entry in registry.values())


class TestShellToolIsOptIn:
    """#421: "a shell exec tool that is off unless explicitly enabled".

    It is left out of the registry rather than refused at call time, so a
    disabled shell is also absent from the prompts built from that registry --
    a tool the model is never told about is one no prompt-injected text can
    talk it into reaching for.
    """

    def test_it_is_absent_by_default(self, fake_chat_tool_eps, monkeypatch):
        from aorta.chat.config import settings

        monkeypatch.setattr(settings, "enable_shell_tool", False)
        fake_chat_tool_eps([])
        assert "run_terminal_command" not in load_chat_tools()

    def test_enabling_it_registers_it(self, fake_chat_tool_eps, monkeypatch):
        from aorta.chat.config import settings

        monkeypatch.setattr(settings, "enable_shell_tool", True)
        fake_chat_tool_eps([])
        registry = load_chat_tools()
        assert "run_terminal_command" in registry
        assert registry["run_terminal_command"].source_package == "aorta"

    def test_the_default_registry_is_the_builtins_exactly(
        self, fake_chat_tool_eps, monkeypatch
    ):
        from aorta.chat.config import settings

        monkeypatch.setattr(settings, "enable_shell_tool", False)
        fake_chat_tool_eps([])
        assert set(load_chat_tools()) == set(BUILTIN_CHAT_TOOLS)


def test_every_builtin_key_matches_the_tools_own_name():
    # The model is offered `tool.name` and asks for it by that name, so a key
    # that disagrees is a tool nothing can call.
    for name, builtin in BUILTIN_CHAT_TOOLS.items():
        assert builtin.name == name


def test_plugin_tool_is_discovered(fake_chat_tool_eps):
    fake_chat_tool_eps([("read_fabric_counters", read_fabric_counters, "amd-fabric-tools")])
    registry = load_chat_tools()
    entry = registry["read_fabric_counters"]
    assert entry.tool is read_fabric_counters
    assert entry.source_package == "amd-fabric-tools"
    # And it did not displace anything.
    assert set(BUILTIN_CHAT_TOOLS) < set(registry)


def test_plugin_without_dist_metadata_is_still_usable(fake_chat_tool_eps):
    fake_chat_tool_eps([("read_fabric_counters", read_fabric_counters, None)])
    registry = load_chat_tools()
    assert registry["read_fabric_counters"].source_package == "<unknown>"


def test_collision_with_a_builtin_raises(fake_chat_tool_eps):
    fake_chat_tool_eps([("list_files", list_files, "plugin_x")])
    with pytest.raises(RegistryCollisionError, match="aorta.*plugin_x"):
        load_chat_tools()


def test_collision_between_two_plugins_raises(fake_chat_tool_eps):
    fake_chat_tool_eps(
        [
            ("read_fabric_counters", read_fabric_counters, "plugin_a"),
            ("read_fabric_counters", read_fabric_counters, "plugin_b"),
        ]
    )
    with pytest.raises(RegistryCollisionError, match="plugin_a.*plugin_b"):
        load_chat_tools()


def test_plugin_that_fails_to_import_is_logged_and_skipped(fake_chat_tool_eps, caplog):
    fake_chat_tool_eps(
        [
            ("broken", ImportError("no module named 'nope'"), "plugin_bad"),
            ("read_fabric_counters", read_fabric_counters, "plugin_good"),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="aorta.chat.plugins"):
        registry = load_chat_tools()
    assert "broken" not in registry
    # The healthy plugin and every built-in survive the bad neighbour.
    assert "read_fabric_counters" in registry
    assert set(BUILTIN_CHAT_TOOLS) < set(registry)
    assert "broken" in caplog.text and "plugin_bad" in caplog.text


def test_plugin_that_is_not_a_tool_is_logged_and_skipped(fake_chat_tool_eps, caplog):
    # The likely mistake: the author forgot langchain's @tool decorator, so the
    # entry point resolves to a plain function.
    def undecorated(node: str = "") -> str:
        return node

    fake_chat_tool_eps([("undecorated", undecorated, "plugin_bad")])
    with caplog.at_level(logging.WARNING, logger="aorta.chat.plugins"):
        registry = load_chat_tools()
    assert "undecorated" not in registry
    assert "BaseTool" in caplog.text and "@tool" in caplog.text


def test_plugin_whose_entry_point_name_disagrees_is_skipped(fake_chat_tool_eps, caplog):
    fake_chat_tool_eps([("fabric_counters", read_fabric_counters, "plugin_bad")])
    with caplog.at_level(logging.WARNING, logger="aorta.chat.plugins"):
        registry = load_chat_tools()
    assert "fabric_counters" not in registry
    assert "read_fabric_counters" not in registry
    assert "read_fabric_counters" in caplog.text


def test_a_broken_builtin_raises_rather_than_being_skipped(monkeypatch):
    # A built-in is aorta's own bug. Skipping it would silently drop a shipped
    # tool; a plugin is skipped precisely because it is not ours to fix.
    monkeypatch.setitem(BUILTIN_CHAT_TOOLS, "busted", lambda: None)
    with pytest.raises(RegistryError, match="BaseTool"):
        load_chat_tools()


def test_a_misnamed_builtin_raises(monkeypatch):
    monkeypatch.setitem(BUILTIN_CHAT_TOOLS, "renamed", read_fabric_counters)
    with pytest.raises(RegistryError, match="names itself"):
        load_chat_tools()


# ── the graph reads the registry, which is what makes the seam real ────────


def test_the_act_loop_registry_is_built_from_the_plugin_registry():
    from aorta.chat.graph.nodes import CHAT_TOOLS, TOOL_REGISTRY

    assert set(TOOL_REGISTRY) == set(CHAT_TOOLS)
    assert TOOL_REGISTRY["list_files"] is CHAT_TOOLS["list_files"].tool


def test_plugin_tools_are_advertised_to_the_text_protocol():
    # Native tool calling sends every tool's schema, so it needs nothing here.
    # The ACTION: protocol has only the prompt, so a plugin tool absent from it
    # is a tool the model never calls.
    from aorta.chat.graph.nodes import _plugin_tool_help
    from aorta.chat.plugins import ChatTool

    text = _plugin_tool_help(
        {
            "read_fabric_counters": ChatTool(
                name="read_fabric_counters",
                tool=read_fabric_counters,
                source_package="amd-fabric-tools",
            )
        }
    )
    assert "read_fabric_counters" in text
    assert "Read the fabric performance counters" in text
    assert "amd-fabric-tools" in text
    # Numbering continues from the hand-written built-in list above it, whose
    # length depends on whether the opt-in shell tool is registered.
    assert f"{len(enabled_builtins()) + 1}. read_fabric_counters" in text


def test_a_tool_with_no_description_still_renders():
    # @tool refuses a function with no docstring, so this only happens with a
    # hand-built BaseTool -- but it must not be an IndexError inside prompt
    # construction, which would take the whole graph import down.
    from aorta.chat.graph.nodes import _summary_line

    blank = read_fabric_counters.model_copy(update={"description": ""})
    assert _summary_line(blank) == "no description"


def test_no_plugins_leaves_both_prompts_byte_identical():
    # A user with no plugins installed must get exactly the prompts they had
    # before this extension point existed, which is why the helper returns ""
    # rather than an empty heading.
    from aorta.chat.graph.nodes import (
        _BUILTIN_PLAN_PROMPT,
        _BUILTIN_TOOL_DESCRIPTIONS,
        PLAN_PROMPT,
        TOOL_DESCRIPTIONS,
        _plugin_tool_help,
    )
    from aorta.chat.plugins import ChatTool

    builtins_only = {
        name: ChatTool(name=name, tool=builtin, source_package="aorta")
        for name, builtin in BUILTIN_CHAT_TOOLS.items()
    }
    assert _plugin_tool_help(builtins_only) == ""
    # This test environment has no chat-tool plugins installed, so the module's
    # own prompts must also be untouched.
    assert TOOL_DESCRIPTIONS == _BUILTIN_TOOL_DESCRIPTIONS
    assert PLAN_PROMPT == _BUILTIN_PLAN_PROMPT


# ── `aorta chat tools`, which is how a plugin author checks their work ─────


def test_chat_tools_lists_builtins_and_names_their_source(fake_chat_tool_eps):
    from click.testing import CliRunner

    from aorta.cli.chat import chat

    fake_chat_tool_eps([("read_fabric_counters", read_fabric_counters, "amd-fabric-tools")])
    result = CliRunner().invoke(chat, ["tools"])
    assert result.exit_code == 0, result.output
    assert "list_files  [aorta]" in result.output
    assert "read_fabric_counters  [amd-fabric-tools]" in result.output


def test_chat_tools_json_is_machine_readable(fake_chat_tool_eps):
    import json

    from click.testing import CliRunner

    from aorta.cli.chat import chat

    fake_chat_tool_eps([])
    result = CliRunner().invoke(chat, ["tools", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload) == set(BUILTIN_CHAT_TOOLS)
    assert payload["read_file"]["source_package"] == "aorta"
