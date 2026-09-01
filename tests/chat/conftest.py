"""Shared fixtures for the ``aorta chat`` test suite."""

from __future__ import annotations

import importlib.util
import os
import socket
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# These tests exercise real langchain/langgraph objects, so they need the
# chat-cli extra. A base install (pyyaml + click) is a supported and common
# configuration -- it is what `pip install amd-aorta` gives a customer -- so the
# directory skips itself rather than erroring at collection. The import-boundary
# tests deliberately live in tests/cli/ instead, because those are pure AST and
# must run here too.
if importlib.util.find_spec("langchain_core") is None:  # pragma: no cover
    collect_ignore_glob = ["test_*.py"]

# Settings that steer control flow are pinned before anything reads them.
# Environment outranks the profile file, so this makes the suite independent of
# whatever the developer has configured locally. Without it, a profile
# containing llm_tool_mode = "native" -- the setting a reasoning model needs, so
# a likely one to have -- sends act_node down the native path in tests written
# for the text protocol. A test that wants the other value monkeypatches the
# settings object directly.
os.environ["AORTA_CHAT_LLM_TOOL_MODE"] = "text"

import httpx  # noqa: E402
import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_chat_config(monkeypatch, tmp_path_factory):
    """Point XDG at empty tmp dirs so no test reads the developer's own profile.

    Autouse and unconditional. The profile file is a real credential store on a
    developer machine, and a suite that reads it both leaks its values into
    assertion output and fails differently for every person who runs it. Also
    drops the cached settings on the way in and out, since laziness means the
    first test to touch ``settings`` would otherwise pin the values for the
    rest of the session.
    """
    from aorta.chat import config

    root = tmp_path_factory.mktemp("xdg")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(root / "cache"))
    config.reset_settings()
    yield
    config.reset_settings()


@pytest.fixture()
def chat_profile(isolated_chat_config) -> Path:
    """Path of the isolated profile file, with its parent created.

    The file itself is not written -- a test that wants content writes it, and
    a test that wants "no profile" gets that for free.
    """
    from aorta._user_paths import chat_config_path

    path = chat_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def fake_aorta_dir(tmp_path: Path) -> Path:
    """Create a temporary AORTA-like codebase directory."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "main.py").write_text(
        "def main():\n    print('hello')\n\n" "class App:\n    def run(self):\n        pass\n",
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text("key: value\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Test Project\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.pyc").write_bytes(b"\x00")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def mock_settings(fake_aorta_dir: Path, tmp_path: Path):
    """Patch settings to use the fake AORTA directory."""
    with patch("aorta.chat.config.settings") as mock_s:
        mock_s.aorta_path = str(fake_aorta_dir)
        mock_s.aorta_root = fake_aorta_dir
        mock_s.index_path = str(tmp_path / "index.sqlite")
        mock_s.repo_map_path = str(tmp_path / "repo_map.md")
        mock_s.embedding_model = "BAAI/bge-small-en-v1.5"
        mock_s.chunk_size = 512
        mock_s.chunk_overlap = 50
        mock_s.allowed_commands = [
            "python",
            "pytest",
            "make",
            "pip",
            "grep",
            "wc",
            "head",
            "tail",
            "cat",
            "ls",
            "find",
        ]
        mock_s.command_timeout = 10
        mock_s.max_retry_iterations = 3
        mock_s.max_act_rounds = 5
        mock_s.max_act_rounds_search = 8
        mock_s.llm_provider = "vllm"
        mock_s.vllm_base_url = "http://localhost:8000/v1"
        mock_s.vllm_model = "test-model"
        mock_s.vllm_api_key = "EMPTY"
        mock_s.remote_llm_model = "gpt-4o-mini"
        mock_s.remote_llm_api_key = ""
        mock_s.remote_llm_base_url = ""
        mock_s.remote_llm_auth_header = ""
        mock_s.remote_llm_extra_headers = {}
        mock_s.llm_max_tokens = None
        mock_s.llm_timeout = 120.0
        mock_s.llm_max_retries = 2
        mock_s.embedding_provider = "local"
        mock_s.remote_embedding_model = "text-embedding-3-small"
        mock_s.remote_embedding_api_key = ""
        mock_s.remote_embedding_base_url = ""
        mock_s.remote_embedding_auth_header = ""
        mock_s.remote_embedding_extra_headers = {}
        yield mock_s


class NetworkUsedError(RuntimeError):
    """Raised when a test reaches for the network, which no test here may do."""


@pytest.fixture()
def no_network(monkeypatch):
    """Turn every outbound connection attempt into a :class:`NetworkUsedError`.

    Both httpx and the raw socket layer are covered, so a code path that
    contacts a provider fails immediately and visibly instead of hanging on a
    real connect or, worse, succeeding against a live endpoint.
    """

    def _refuse(*args, **kwargs):
        raise NetworkUsedError("a test tried to open a network connection")

    monkeypatch.setattr(httpx.AsyncClient, "send", _refuse)
    monkeypatch.setattr(httpx.Client, "send", _refuse)
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


def make_fake_llm(responses: list[str]) -> MagicMock:
    """Create a mock LLM that returns preset AIMessage responses in order."""
    replies = iter([AIMessage(content=r) for r in responses])

    def _next_reply(*args, **kwargs) -> AIMessage:
        try:
            return next(replies)
        except StopIteration:
            raise AssertionError(
                f"the node invoked this mock LLM more than the {len(responses)} "
                "scripted times; add the missing response"
            ) from None

    mock = MagicMock()
    mock.ainvoke = AsyncMock(side_effect=_next_reply)
    return mock


def make_llm_sequence(*llms: MagicMock) -> Callable[..., MagicMock]:
    """Return a ``_get_llm`` side effect handing out *llms* in node order.

    A bare ``iter()`` here reports exhaustion as ``RuntimeError: coroutine
    raised StopIteration``, which points at asyncio rather than at the test
    being one chat model short of the graph's node count -- the exact trap that
    silently broke these tests when the router, plan and answer nodes landed.
    """
    pending = iter(llms)

    def _next_llm(**kwargs) -> MagicMock:
        try:
            return next(pending)
        except StopIteration:
            raise AssertionError(
                f"the graph asked for more than the {len(llms)} scripted chat "
                "models; update the test's node sequence"
            ) from None

    return _next_llm


@pytest.fixture()
def fake_retriever():
    """Return a mock retriever that yields fixed documents."""
    mock = MagicMock()
    mock.invoke.return_value = [
        Document(
            page_content="def run_scenario(name):\n    pass",
            metadata={"source": "src/runner.py", "start_line": 1, "end_line": 2},
        ),
        Document(
            page_content="CONFIG = {'timeout': 30}",
            metadata={"source": "config/defaults.py", "start_line": 5, "end_line": 5},
        ),
    ]
    return mock
