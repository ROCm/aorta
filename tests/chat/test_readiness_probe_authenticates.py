"""The readiness probe has to identify itself, like every other request does.

``_await_health`` polled anonymously. A bare vLLM does not mind, but a gateway
in front of one does: LiteLLM answers an unauthenticated ``/health`` with 500,
which is indistinguishable here from a server that is not up yet.

So against a proxy that was serving perfectly the poll failed every time, spent
the entire budget, and the session started late having learned nothing. The
welcome message waits on this, so the visible symptom was a blank chat for the
whole of the budget -- five minutes, before the budget was cut to one.

Measured against the live proxy while fixing it: 60s of failed polls before,
1.84s after.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from aorta.chat.config import reset_settings
from aorta.chat.inference.providers.local_vllm import LocalVLLMBackend


@pytest.fixture()
def backend(monkeypatch):
    monkeypatch.setenv("AORTA_CHAT_VLLM_BASE_URL", "http://proxy.example:4000/v1")
    monkeypatch.setenv("AORTA_CHAT_VLLM_API_KEY", "sk-the-key")
    reset_settings()
    yield LocalVLLMBackend()
    reset_settings()


class TestTheProbeCarriesTheKey:
    def test_the_header_is_built(self, backend):
        assert backend._health_headers() == {"Authorization": "Bearer sk-the-key"}

    def test_it_reaches_the_request(self, backend, monkeypatch):
        """Built but not passed to the client would look identical from here."""
        seen: dict[str, str] = {}

        class FakeClient:
            def __init__(self, *_, headers=None, **__):
                seen.update(headers or {})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def get(self, _url, **_kw):
                return httpx.Response(200, request=httpx.Request("GET", "http://x"))

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
        asyncio.run(backend._await_health(5, 1))

        assert seen.get("Authorization") == "Bearer sk-the-key"


class TestABareServerIsStillSupported:
    def test_no_key_means_no_header(self, monkeypatch):
        """A self-hosted vLLM with no auth must not be sent 'Bearer '."""
        monkeypatch.setenv("AORTA_CHAT_VLLM_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("AORTA_CHAT_VLLM_API_KEY", "")
        reset_settings()

        assert LocalVLLMBackend()._health_headers() == {}
        reset_settings()


class TestWhatTheBugLookedLike:
    """A 500 is not distinguishable from 'not up yet', which is why it cost the
    whole budget rather than failing loudly."""

    @staticmethod
    def _client_returning(code: int, counter: list[int]):
        class FakeClient:
            def __init__(self, *_, headers=None, **__):
                self.headers = headers or {}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def get(self, _url, **_kw):
                counter.append(1)
                # What LiteLLM does to an anonymous caller.
                status = 200 if self.headers.get("Authorization") else code
                return httpx.Response(status, request=httpx.Request("GET", "http://x"))

        return FakeClient

    def test_with_the_key_it_returns_on_the_first_poll(self, backend, monkeypatch):
        polls: list[int] = []
        monkeypatch.setattr(httpx, "AsyncClient", self._client_returning(500, polls))

        assert asyncio.run(backend._await_health(5, 1)) is True
        assert len(polls) == 1, f"polled {len(polls)} times for a ready server"

    def test_without_it_the_budget_is_spent(self, monkeypatch):
        """The old behaviour, kept as the thing that must not come back."""
        monkeypatch.setenv("AORTA_CHAT_VLLM_BASE_URL", "http://proxy.example:4000/v1")
        monkeypatch.setenv("AORTA_CHAT_VLLM_API_KEY", "")
        reset_settings()
        polls: list[int] = []
        monkeypatch.setattr(httpx, "AsyncClient", self._client_returning(500, polls))

        assert asyncio.run(LocalVLLMBackend()._await_health(3, 1)) is False
        assert len(polls) > 1, "an unauthenticated probe should have kept retrying"
        reset_settings()
