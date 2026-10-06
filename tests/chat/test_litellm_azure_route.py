"""A litellm profile at an Azure-style gateway reaches the deployment path (#556).

``aorta chat ask`` failed with ``NotFoundError: Resource not found`` against a
gateway that serves only ``/openai/deployments/<model>/chat/completions
?api-version=<version>``. The profile named a bare ``gpt-4o-mini``, so LiteLLM
treated the base URL as an OpenAI endpoint and posted to ``/chat/completions``.
Only an ``azure/`` model makes LiteLLM build the deployment path, and there was
no setting for the ``api-version`` it needs -- nor any signal before the first
query, because preflight makes no call.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import tomllib
from click.testing import CliRunner

from aorta.chat import config
from aorta.chat.config import settings
from aorta.cli.chat import chat

_needs_litellm = pytest.mark.skipif(
    importlib.util.find_spec("litellm") is None
    or importlib.util.find_spec("langchain_litellm") is None,
    reason="requires the chat-all extra",
)

GATEWAY = "https://gateway.example.com"


@pytest.fixture()
def azure_gateway(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "litellm")
    monkeypatch.setattr(settings, "remote_llm_model", "azure/gpt-4o-mini")
    monkeypatch.setattr(settings, "remote_llm_base_url", GATEWAY)
    monkeypatch.setattr(settings, "remote_llm_api_key", "sk-not-a-real-key")
    monkeypatch.setattr(settings, "remote_llm_api_version", "2024-10-21")
    monkeypatch.setattr(settings, "remote_llm_auth_header", "")
    monkeypatch.setattr(settings, "remote_llm_extra_headers", {})
    monkeypatch.setattr(settings, "llm_max_tokens", None)
    monkeypatch.setattr(settings, "llm_timeout", 30.0)
    monkeypatch.setattr(settings, "llm_max_retries", 0)


def _backend():
    from aorta.chat.inference.providers.remote_litellm import RemoteLiteLLMBackend

    return RemoteLiteLLMBackend()


@_needs_litellm
class TestTheApiVersionIsSent:
    def test_it_reaches_litellm_through_model_kwargs(self, azure_gateway, no_network):
        llm = _backend().get_chat_model(streaming=False)
        assert llm.model_kwargs["api_version"] == "2024-10-21"

    def test_an_empty_one_is_left_to_litellm(self, azure_gateway, monkeypatch, no_network):
        monkeypatch.setattr(settings, "remote_llm_api_version", "  ")
        llm = _backend().get_chat_model(streaming=False)
        assert "api_version" not in (llm.model_kwargs or {})

    def test_it_is_not_sent_to_a_model_that_is_not_azure(
        self, azure_gateway, monkeypatch, no_network
    ):
        """A leftover setting must not follow a switch to another provider."""
        monkeypatch.setattr(settings, "remote_llm_model", "anthropic/claude-example")
        llm = _backend().get_chat_model(streaming=False)
        assert "api_version" not in (llm.model_kwargs or {})

    def test_it_travels_beside_the_gateway_header(self, azure_gateway, monkeypatch, no_network):
        monkeypatch.setattr(settings, "remote_llm_auth_header", "Ocp-Apim-Subscription-Key")
        llm = _backend().get_chat_model(streaming=False)
        assert llm.model_kwargs == {
            "extra_headers": {"Ocp-Apim-Subscription-Key": "sk-not-a-real-key"},
            "api_version": "2024-10-21",
        }


@_needs_litellm
class TestAPrefixWithNoDeploymentIsRefused:
    @pytest.mark.parametrize("model", ["azure/", "azure/  ", " azure/"])
    async def test_preflight_names_the_missing_deployment(self, azure_gateway, monkeypatch, model):
        monkeypatch.setattr(settings, "remote_llm_model", model)
        with pytest.raises(ValueError, match="names no Azure deployment"):
            await _backend().preflight()

    def test_building_the_model_refuses_it_too(self, azure_gateway, monkeypatch):
        monkeypatch.setattr(settings, "remote_llm_model", "azure/")
        with pytest.raises(ValueError, match="names no Azure deployment"):
            _backend().get_chat_model(streaming=False)


@_needs_litellm
class TestTheRouteIsCheckedWithoutACall:
    def test_a_bare_model_behind_a_base_url_is_the_reported_shape(
        self, azure_gateway, monkeypatch, no_network
    ):
        monkeypatch.setattr(settings, "remote_llm_model", "gpt-4o-mini")
        warning = _backend().route_warning()
        assert warning is not None
        assert f"{GATEWAY}/chat/completions" in warning
        assert "azure/<deployment>" in warning

    def test_an_unknown_bare_name_is_flagged_without_printing(
        self, azure_gateway, monkeypatch, capsys, no_network
    ):
        """litellm.get_llm_provider prints a banner for this; --json would carry it."""
        monkeypatch.setattr(settings, "remote_llm_model", "my-deployment")
        assert _backend().route_warning() is not None
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize(
        "model", ["azure/gpt-4o-mini", "openai/gpt-oss-20b", "anthropic/claude-example"]
    )
    def test_a_provider_prefix_is_not_flagged(self, azure_gateway, monkeypatch, model):
        monkeypatch.setattr(settings, "remote_llm_model", model)
        assert _backend().route_warning() is None

    def test_no_base_url_leaves_routing_to_litellm(self, azure_gateway, monkeypatch):
        monkeypatch.setattr(settings, "remote_llm_model", "claude-example")
        monkeypatch.setattr(settings, "remote_llm_base_url", "")
        assert _backend().route_warning() is None

    @pytest.mark.parametrize("suffix", ["/openai", "/openai/"])
    def test_a_base_url_ending_in_openai_is_flagged(self, azure_gateway, monkeypatch, suffix):
        monkeypatch.setattr(settings, "remote_llm_base_url", GATEWAY + suffix)
        warning = _backend().route_warning()
        assert warning is not None
        assert "Drop the trailing /openai" in warning

    async def test_preflight_logs_it(self, azure_gateway, monkeypatch, caplog):
        monkeypatch.setattr(settings, "remote_llm_model", "gpt-4o-mini")
        with caplog.at_level(logging.WARNING):
            await _backend().preflight()
        assert any("/chat/completions" in r.getMessage() for r in caplog.records)


class _Capture(BaseHTTPRequestHandler):
    """Records the request line and answers like any chat completions server."""

    paths: list[str] = []

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
        self.rfile.read(int(self.headers.get("content-length", 0)))
        type(self).paths.append(self.path)
        body = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def capture_server():
    _Capture.paths = []
    server = HTTPServer(("127.0.0.1", 0), _Capture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", _Capture.paths
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@_needs_litellm
class TestTheRequestPathLiteLLMBuilds:
    """End to end through ChatLiteLLM: the path is LiteLLM's decision, not ours."""

    def test_the_azure_model_reaches_the_deployment_path(
        self, azure_gateway, monkeypatch, capture_server
    ):
        base, paths = capture_server
        monkeypatch.setattr(settings, "remote_llm_base_url", base)
        _backend().get_chat_model(streaming=False).invoke("hi")
        assert paths == ["/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-10-21"]

    def test_the_bare_model_is_what_the_gateway_404d(
        self, azure_gateway, monkeypatch, capture_server
    ):
        base, paths = capture_server
        monkeypatch.setattr(settings, "remote_llm_base_url", base)
        monkeypatch.setattr(settings, "remote_llm_model", "gpt-4o-mini")
        _backend().get_chat_model(streaming=False).invoke("hi")
        assert paths == ["/chat/completions"]


class TestDoctorReportsTheRoute:
    def _check(self, monkeypatch, backend):
        from aorta.chat.doctor import run_checks
        from aorta.chat.inference.providers import factory

        monkeypatch.setattr(factory, "get_backend", lambda *a, **k: backend)
        return next(c for c in run_checks(backend=True).checks if c.name == "llm backend")

    def test_a_route_warning_is_a_warn_with_the_reason(self, monkeypatch):
        from aorta.chat.doctor import WARN

        class _Misrouted:
            name = "litellm"

            async def probe(self, timeout=None):
                return None

            def describe(self):
                return "remote LiteLLM -- gpt-4o-mini via https://gateway.example.com"

            def route_warning(self):
                return "posts to https://gateway.example.com/chat/completions"

        check = self._check(monkeypatch, _Misrouted())
        assert check.status == WARN
        assert "/chat/completions" in check.hint

    def test_a_backend_without_the_hook_is_ok(self, monkeypatch):
        from aorta.chat.doctor import OK

        class _Plain:
            name = "vllm"

            async def probe(self, timeout=None):
                return None

            def describe(self):
                return "vllm at http://localhost:8000/v1"

        assert self._check(monkeypatch, _Plain()).status == OK

    def test_a_hook_that_raises_does_not_take_the_report_down(self, monkeypatch):
        from aorta.chat.doctor import OK

        class _Broken:
            name = "litellm"

            async def probe(self, timeout=None):
                return None

            def describe(self):
                return "remote LiteLLM"

            def route_warning(self):
                raise RuntimeError("advisory check broke")

        assert self._check(monkeypatch, _Broken()).status == OK


class TestTheAzureOpenAIProfile:
    def test_it_routes_through_litellm_with_the_azure_prefix(self):
        template = config.PROFILE_TEMPLATES["azure-openai"]
        assert template["llm_provider"] == "litellm"
        assert template["remote_llm_model"].startswith("azure/")

    def test_it_asks_for_the_api_version(self):
        assert "remote_llm_api_version" in config.PROFILE_PROMPTS["azure-openai"]

    def test_prompted_answers_land_in_the_file(self, chat_profile):
        result = CliRunner().invoke(
            chat,
            ["config", "init", "--profile", "azure-openai"],
            input=f"{GATEWAY}\nazure/gpt-4o-mini\n2024-10-21\nsk-typed\n",
        )
        assert result.exit_code == 0, result.output
        loaded = tomllib.loads(chat_profile.read_text(encoding="utf-8"))
        assert loaded["remote_llm_base_url"] == GATEWAY
        assert loaded["remote_llm_model"] == "azure/gpt-4o-mini"
        assert loaded["remote_llm_api_version"] == "2024-10-21"
        assert loaded["remote_llm_api_key"] == "sk-typed"
