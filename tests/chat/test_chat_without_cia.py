"""`aorta chat` works without the agents installed.

``aorta.chat.plugins`` imported ``chat.tools.cluster`` at module scope, and from
there the chain runs ``cia.triage`` → ``cia.watch.poll`` → ``cia.watch.watcher``
→ ``import dspy``. DSPy is in the ``[cia]`` extra and in neither ``chat-cli``
nor ``chat-ui``, so ``pip install amd-aorta[chat-ui]`` followed by ``aorta chat``
raised ``ModuleNotFoundError: dspy`` before the assistant started.

``tests/cli/test_chat_boundaries.py`` protects this property in the other
direction — the agents run without the chat extras — measured in a fresh
interpreter for the same reason: ``sys.modules`` is already dirty by the time
pytest gets here.

Chat without the agents is a shipped configuration, not a fault: an assistant
over the codebase and past runs, minus the five tools that submit cluster jobs.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from aorta.chat.plugins import _DIAGNOSTIC_TOOL_NAMES, load_chat_tools

#: Run in a subprocess so the absence of dspy is real rather than patched over
#: an already-imported module.
_PROBE = """
import builtins, json, logging
logging.disable(logging.CRITICAL)
_real = builtins.__import__
def _no_dspy(name, *args, **kwargs):
    if name == "dspy" or name.startswith("dspy."):
        raise ModuleNotFoundError("No module named 'dspy'")
    return _real(name, *args, **kwargs)
builtins.__import__ = _no_dspy

import aorta.chat.plugins as plugins
registry = plugins.load_chat_tools()
print(json.dumps(sorted(registry)))
"""


@pytest.fixture(scope="module")
def tools_without_dspy() -> list[str]:
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True
    )
    assert out.returncode == 0, (
        "importing aorta.chat.plugins without dspy failed:\n" + out.stderr[-1500:]
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


class TestChatStartsWithoutTheAgents:
    def test_the_plugin_registry_loads(self, tools_without_dspy):
        assert tools_without_dspy, "no tools registered at all"

    def test_the_codebase_tools_are_all_there(self, tools_without_dspy):
        for name in (
            "list_files",
            "read_file",
            "search_code",
            "grep_code",
            "search_repo_map",
            "list_runs",
            "read_run_matrix",
            "read_run_env",
            "search_run_artifacts",
        ):
            assert name in tools_without_dspy, name

    @pytest.mark.parametrize("name", _DIAGNOSTIC_TOOL_NAMES)
    def test_the_diagnostic_tools_are_absent(self, name, tools_without_dspy):
        """Absent, not broken: offering one that cannot run is worse."""
        assert name not in tools_without_dspy


class TestWithTheAgentsInstalled:
    def test_every_diagnostic_tool_is_registered(self):
        registry = load_chat_tools()
        for name in _DIAGNOSTIC_TOOL_NAMES:
            assert name in registry, name

    def test_they_reach_the_act_registry(self):
        from aorta.chat.graph.nodes import TOOL_REGISTRY

        for name in _DIAGNOSTIC_TOOL_NAMES:
            assert name in TOOL_REGISTRY, name

    def test_and_the_prompt_describes_them(self):
        """A tool the text protocol cannot name is one it never calls."""
        from aorta.chat.graph.nodes import TOOL_DESCRIPTIONS, TOOL_REGISTRY

        missing = [n for n in TOOL_REGISTRY if f"{n}(" not in TOOL_DESCRIPTIONS]
        assert not missing, missing


def test_it_says_why_they_are_missing():
    """Five tools quietly absent from a catalogue is a bug report waiting."""
    probe = _PROBE.replace("logging.disable(logging.CRITICAL)", "").replace(
        "print(json.dumps(sorted(registry)))", "pass"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-800:]
    assert "amd-aorta[cia]" in out.stderr, out.stderr[-500:]
