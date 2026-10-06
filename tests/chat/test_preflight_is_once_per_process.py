"""The backend is waited for once, not once per browser tab.

``on_chat_start`` awaited ``preflight`` before sending the welcome message, so
until it returned the user had an empty chat and nothing to type into. That is
the right shape when the wait is short. It was not short: the budget was 300s,
and with the backend down every new tab spent its own 300s arriving at the
same answer -- after which the session started anyway, exactly as it would
have without the wait.

Reachability is a property of the server, not of a conversation. So the wait
happens once per process now, and the budget is smaller: since the session
starts regardless of the outcome, the budget only decides how late the welcome
message is. A backend still warming up is found by the first request either
way.
"""

from __future__ import annotations

import asyncio

import pytest

cl = pytest.importorskip("chainlit", reason="requires the chat-ui extra")

from aorta.chat.inference.providers import local_vllm
from aorta.chat.ui import app


class _Backend:
    """A backend that records every wait and can be made slow."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.waits = 0

    async def preflight(self) -> None:
        self.waits += 1
        if self.delay:
            await asyncio.sleep(self.delay)

    def describe(self) -> str:
        return "fake backend"


@pytest.fixture()
def fresh(monkeypatch):
    """A process that has not waited yet."""
    monkeypatch.setattr(app, "_preflight_done", False)
    monkeypatch.setattr(app, "_preflight_lock", asyncio.Lock())


class TestItHappensOnce:
    async def test_a_second_session_does_not_wait_again(self, fresh):
        backend = _Backend()

        await app._preflight_once(backend)
        await app._preflight_once(backend)
        await app._preflight_once(backend)

        assert backend.waits == 1, f"waited {backend.waits} times"

    async def test_tabs_reconnecting_together_share_one_wait(self, fresh):
        """What a restarted server looks like: several browsers at once."""
        backend = _Backend(delay=0.05)

        await asyncio.gather(*(app._preflight_once(backend) for _ in range(8)))

        assert backend.waits == 1, (
            f"{backend.waits} tabs each started their own wait"
        )

    async def test_the_later_sessions_actually_waited_for_the_first(self, fresh):
        """Returning early without the backend being ready would be worse."""
        backend = _Backend(delay=0.05)
        done: list[float] = []

        async def session():
            await app._preflight_once(backend)
            done.append(asyncio.get_running_loop().time())

        started = asyncio.get_running_loop().time()
        await asyncio.gather(*(session() for _ in range(4)))

        assert all(t - started >= 0.05 for t in done), "a session skipped the wait"

    async def test_a_wait_that_gave_up_is_not_repeated(self, fresh):
        """preflight does not raise; it gives up and logs.

        Repeating it only delays the next session by the same budget to learn
        the same thing. A backend that recovers needs no wait -- the next
        request simply works.
        """
        backend = _Backend()

        await app._preflight_once(backend)
        await app._preflight_once(backend)

        assert backend.waits == 1


class TestOnStartItselfOnlyWaitsOnce:
    """Driven through the handler, because the helper alone proves nothing.

    Calling ``_preflight_once`` directly still passes if ``on_start`` never
    reaches for it, which is exactly the wiring that was wrong before.
    """

    @staticmethod
    def _stub_chainlit(monkeypatch, backend):
        store: dict = {}

        class FakeUserSession:
            def get(self, key, default=None):
                return store.get(key, default)

            def set(self, key, value):
                store[key] = value

        class FakeMessage:
            def __init__(self, content="", **_):
                self.content = content

            async def send(self):
                return self

        monkeypatch.setattr(app.cl, "user_session", FakeUserSession())
        monkeypatch.setattr(app.cl, "Message", FakeMessage)
        monkeypatch.setattr(app, "get_backend", lambda: backend)
        monkeypatch.setattr(app, "welcome_message", lambda _d: "hello")
        monkeypatch.setattr(app, "_SKIP_PREFLIGHT", False)

    async def test_three_sessions_wait_once_between_them(self, fresh, monkeypatch):
        backend = _Backend()
        self._stub_chainlit(monkeypatch, backend)

        for _ in range(3):
            await app.on_start()

        assert backend.waits == 1, f"on_start waited {backend.waits} times"

    async def test_every_session_still_gets_its_welcome(self, fresh, monkeypatch):
        """Waiting once must not mean the later tabs are greeted once."""
        backend = _Backend()
        sent: list[str] = []

        store: dict = {}

        class FakeUserSession:
            def get(self, key, default=None):
                return store.get(key, default)

            def set(self, key, value):
                store[key] = value

        class FakeMessage:
            def __init__(self, content="", **_):
                self.content = content

            async def send(self):
                sent.append(self.content)
                return self

        monkeypatch.setattr(app.cl, "user_session", FakeUserSession())
        monkeypatch.setattr(app.cl, "Message", FakeMessage)
        monkeypatch.setattr(app, "get_backend", lambda: backend)
        monkeypatch.setattr(app, "welcome_message", lambda _d: "hello")
        monkeypatch.setattr(app, "_SKIP_PREFLIGHT", False)

        for _ in range(3):
            await app.on_start()

        assert sent.count("hello") == 3

    async def test_the_skip_switch_still_skips(self, fresh, monkeypatch):
        backend = _Backend()
        self._stub_chainlit(monkeypatch, backend)
        monkeypatch.setattr(app, "_SKIP_PREFLIGHT", True)

        await app.on_start()

        assert backend.waits == 0


class TestTheBudgetIsNotMinutes:
    def test_it_is_shorter_than_it_was(self):
        """300s bought a later welcome and nothing else."""
        assert local_vllm.PREFLIGHT_TIMEOUT < 300

    def test_it_is_still_long_enough_for_a_restart(self):
        assert local_vllm.PREFLIGHT_TIMEOUT >= 30

    def test_the_poll_interval_still_divides_it(self):
        """Otherwise the last poll is cut short and the budget is a lie."""
        assert local_vllm.PREFLIGHT_TIMEOUT % local_vllm.PREFLIGHT_INTERVAL == 0

    def test_the_diagnostic_budget_is_separate(self):
        """probe answers an operator who is watching; it has its own budget."""
        assert local_vllm.PROBE_TIMEOUT != local_vllm.PREFLIGHT_TIMEOUT


class TestTheSkipStillWorks:
    def test_the_no_wait_switch_is_still_read(self):
        """The escape hatch for someone who knows their backend is up."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "src" / "aorta" / "chat" / "ui" / "app.py"
        ).read_text(encoding="utf-8")

        assert "_SKIP_PREFLIGHT" in source
        assert "if not _SKIP_PREFLIGHT:" in source

    def test_the_welcome_still_waits_on_the_backend(self):
        """Ordering is the point: a welcome before the check would mislead."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "src" / "aorta" / "chat" / "ui" / "app.py"
        ).read_text(encoding="utf-8")
        body = source[source.index("async def on_start()") :]

        assert body.index("_preflight_once") < body.index("welcome_message")
