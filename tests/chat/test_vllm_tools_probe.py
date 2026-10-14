"""The shared vLLM backend owns native function-call readiness."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from aorta.chat.config import reset_settings
from aorta.chat.inference.providers.local_vllm import LocalVLLMBackend


@pytest.fixture()
def backend(monkeypatch):
    monkeypatch.setenv("AORTA_CHAT_VLLM_BASE_URL", "http://vllm.example:8001/v1")
    monkeypatch.setenv("AORTA_CHAT_VLLM_API_KEY", "sk-local")
    monkeypatch.setenv("AORTA_CHAT_VLLM_MODEL", "Qwen/Qwen3.8-27B")
    reset_settings()
    yield LocalVLLMBackend()
    reset_settings()


def _client_returning(payload: dict, seen: dict):
    class FakeClient:
        def __init__(self, *_, headers=None, **__):
            seen["headers"] = headers

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def post(self, url, *, json, timeout):
            seen.update(url=url, request=json, timeout=timeout)

            class Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return payload

            return Response()

    return FakeClient


def _tool_response(*, source: str = "READY") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "aorta_tool_protocol_probe",
                                "arguments": json.dumps({"source": source}),
                            }
                        }
                    ]
                }
            }
        ]
    }


def test_probe_sends_and_validates_a_forced_native_call(backend, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        _client_returning(_tool_response(), seen),
    )

    asyncio.run(backend.probe_tools(timeout=3.5))

    assert seen["url"] == "http://vllm.example:8001/v1/chat/completions"
    assert seen["headers"] == {"Authorization": "Bearer sk-local"}
    assert seen["timeout"] == 3.5
    request = seen["request"]
    assert request["model"] == "Qwen/Qwen3.8-27B"
    assert request["tools"][0]["function"]["parameters"]["required"] == ["source"]
    assert request["tool_choice"]["function"]["name"] == "aorta_tool_protocol_probe"


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"choices": [{"message": {"content": "<tool_call>READY</tool_call>"}}]}, "one tool call"),
        (_tool_response(source="WRONG"), "source='READY'"),
    ],
)
def test_probe_rejects_text_or_wrong_arguments(backend, monkeypatch, payload, message):
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        _client_returning(payload, {}),
    )

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(backend.probe_tools())
