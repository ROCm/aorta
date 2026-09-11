"""The agents reach a model through the configuration the rest of chat uses.

``docs/chat/providers.md`` is explicit: a model comes from ``get_chat_llm()``,
selected by ``llm_provider`` and configured through ``~/.config/aorta/chat.toml``
or ``AORTA_CHAT_*``. The agents had their own surface -- DSPy plus
LITELLM_API_BASE / LITELLM_API_KEY / LITELLM_MODEL -- defaulting to
``http://localhost:4000`` with the key ``dummy``.

So a user who had run ``aorta chat config init --profile openai`` had chat
talking to their provider and a Watch and Autopsy talking to a proxy that was
not running, with nothing to say the two disagreed.

The settings are read; the provider layer is not. ``aorta.chat.config`` needs
pydantic and stdlib and none of the chat extras, so the agents still run on a
base install -- which ``tests/cli/test_chat_boundaries.py`` measures.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

PROBE = """
import json, os, sys
from aorta.cia import llm as m
built = {}
# build_lm returns a RedactingLM, so that is what has to be intercepted.
m.RedactingLM = lambda **kw: built.update(kw) or object()
try:
    m.build_lm()
    print(json.dumps({"base": built.get("api_base"), "model": built.get("model"),
                      "key": built.get("api_key")}))
except m.ProviderNotConfigured as exc:
    print(json.dumps({"refused": str(exc)}))
"""


def _resolve(env: dict[str, str]) -> dict:
    """Build an LM in a fresh interpreter, since settings are a singleton."""
    import json
    import os

    clean = {k: v for k, v in os.environ.items() if not k.startswith(("AORTA_CHAT_", "LITELLM_"))}
    clean.update(env)
    out = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, env=clean
    )
    assert out.returncode == 0, out.stderr[-800:]
    return json.loads(out.stdout.strip().splitlines()[-1])


class TestConfiguringChatConfiguresTheAgents:
    def test_a_configured_vllm_endpoint_is_the_one_used(self):
        got = _resolve(
            {
                "AORTA_CHAT_VLLM_BASE_URL": "http://proxy.example:4000/v1",
                "AORTA_CHAT_VLLM_MODEL": "qwen3-35b",
                "AORTA_CHAT_VLLM_API_KEY": "sk-real",
            }
        )
        assert got["base"] == "http://proxy.example:4000/v1"
        assert got["model"] == "openai/qwen3-35b"
        assert got["key"] == "sk-real"

    def test_it_never_reaches_the_proxy_this_package_used_to_assume(self):
        """localhost:4000 with key "dummy" was the old silent default."""
        got = _resolve({"AORTA_CHAT_VLLM_BASE_URL": "http://elsewhere:9/v1"})
        assert "4000" not in (got["base"] or "")
        assert got["key"] != "dummy"

    def test_a_remote_provider_is_read_from_its_own_fields(self):
        got = _resolve(
            {
                "AORTA_CHAT_LLM_PROVIDER": "openai",
                "AORTA_CHAT_REMOTE_LLM_MODEL": "gpt-4o-mini",
                "AORTA_CHAT_REMOTE_LLM_API_KEY": "sk-remote",
            }
        )
        assert got["model"] == "openai/gpt-4o-mini"
        assert got["key"] == "sk-remote"


class TestDeploymentsOnTheOlderVariables:
    def test_they_still_work(self):
        """Reading chat's defaults instead would silently move their endpoint."""
        got = _resolve({"LITELLM_API_BASE": "http://legacy:4000", "LITELLM_MODEL": "old"})
        assert got["base"] == "http://legacy:4000"
        assert got["model"] == "openai/old"

    def test_configured_chat_outranks_them(self):
        got = _resolve(
            {
                "LITELLM_API_BASE": "http://legacy:4000",
                "AORTA_CHAT_VLLM_BASE_URL": "http://configured:4000/v1",
            }
        )
        assert got["base"] == "http://configured:4000/v1"


class TestWithNothingConfigured:
    def test_it_lands_where_an_unconfigured_chat_lands(self):
        """Failing the same way as chat beats failing a second way elsewhere."""
        from aorta.chat.config import Settings

        got = _resolve({"HOME": "/tmp/aorta-no-such-home"})
        assert got["base"] == Settings.model_fields["vllm_base_url"].default


def test_reading_the_settings_does_not_drag_in_the_chat_extras():
    """The whole reason the agents are a separate package."""
    probe = (
        "import sys, json;"
        "from aorta.cia.llm import chat_provider;"
        "chat_provider();"
        "bad=('langchain','langgraph','chainlit','chromadb','fastembed','torch');"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in bad)))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip().endswith("[]"), out.stdout


def test_the_module_does_not_invent_its_own_provider_surface():
    """LITELLM_* survives only as the older spelling, and says so.

    Asserted against the code with its docstrings dropped: they name the old
    endpoint deliberately, to record what was removed and why.
    """
    import ast
    import inspect

    from aorta.cia import llm

    tree = ast.parse(inspect.getsource(llm))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                body.pop(0)
    code = ast.unparse(tree)

    assert "localhost:4000" not in code
    assert "'dummy'" not in code and '"dummy"' not in code
    assert "predate" in inspect.getsource(llm), "the older variables should say so"
