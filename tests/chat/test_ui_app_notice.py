"""The web UI's per-request redaction disclosure, on both the error path and
across concurrent sessions.

Decision 16 promises the user is told what a request had removed before it
left. The UI drained that notice only after a successful answer, so a request
that was redacted and *then* failed in the graph or the provider disclosed
nothing -- and a user who stops after the failure is never told. The send had
already happened by then, so the disclosure is owed either way.

The other way the UI can fail the promise is by session, not by path: one
server process serves many browsers, so state that is per-process is a
disclosure one user can consume on another's behalf. ``TestTheNoticeIsDelivered``
seeds the notice to isolate the delivery half; ``TestTwoSessionsInOneProcess``
drives the real redaction through two overlapping sessions to pin the scoping
half, since a process-wide flag reset at session start satisfies every
sequential arrangement of it.

``aorta.chat.ui.app`` imports ``chainlit`` at module scope and the ``chat-ui``
extra is a separate install, so a fake is put in ``sys.modules`` first. The
alternative -- skipping when Chainlit is absent -- would leave a user-facing
security guarantee untested on the configuration these tests actually run in.
"""

from __future__ import annotations

import asyncio
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from aorta.chat import redaction
from aorta.chat.config import reset_settings

NOTICE = "aorta chat: redacted 3 filesystem paths from the outbound request."


class _FakeMessage:
    """Records what the session was shown, in order."""

    sent: list[str] = []

    def __init__(self, content: str = "") -> None:
        self.content = content

    async def send(self) -> None:
        type(self).sent.append(self.content)

    async def remove(self) -> None:
        pass


class _FakeUserSession:
    def __init__(self) -> None:
        self._values: dict = {}

    def get(self, key, default=None):
        return self._values.get(key, default)

    def set(self, key, value) -> None:
        self._values[key] = value


def _fake_chainlit() -> ModuleType:
    module = ModuleType("chainlit")
    module.Message = _FakeMessage
    module.user_session = _FakeUserSession()
    # Chainlit's decorators register and return the function; the tests call
    # the handlers directly, so identity is the whole contract needed here.
    module.on_chat_start = lambda fn: fn
    module.on_message = lambda fn: fn
    return module


@pytest.fixture()
def app(monkeypatch):
    """``aorta.chat.ui.app`` imported against the fake Chainlit."""
    monkeypatch.setitem(sys.modules, "chainlit", _fake_chainlit())
    monkeypatch.delitem(sys.modules, "aorta.chat.ui.app", raising=False)
    import aorta.chat.ui.app as app_module

    _FakeMessage.sent = []
    app_module.cl.user_session = _FakeUserSession()
    app_module.cl.user_session.set("history", [])
    app_module.cl.user_session.set("backend_error", None)
    yield app_module
    monkeypatch.delitem(sys.modules, "aorta.chat.ui.app", raising=False)


def _seed_pending_notice(app_module) -> None:
    app_module.cl.user_session.set(
        app_module._NOTICE_STATE_KEY, redaction.NoticeState(emitted=True, pending=NOTICE)
    )


class TestTheNoticeIsDelivered:
    async def test_after_a_successful_answer(self, app, monkeypatch):
        _seed_pending_notice(app)

        async def _answer(question, history, on_step=None):  # noqa: ARG001 - signature match
            return "the answer", [], {}

        monkeypatch.setattr(app, "invoke_agent", _answer)
        await app.on_message(SimpleNamespace(content="why did cell 3 fail?"))

        assert "the answer" in _FakeMessage.sent
        assert any(NOTICE in shown for shown in _FakeMessage.sent)

    async def test_after_a_failure_too(self, app, monkeypatch):
        """The request already left; the failure does not cancel the disclosure."""
        _seed_pending_notice(app)

        async def _explode(question, history, on_step=None):  # noqa: ARG001 - signature match
            raise RuntimeError("provider hung up")

        monkeypatch.setattr(app, "invoke_agent", _explode)
        await app.on_message(SimpleNamespace(content="why did cell 3 fail?"))

        assert any("An error occurred" in shown for shown in _FakeMessage.sent)
        assert any(NOTICE in shown for shown in _FakeMessage.sent)

    async def test_it_arrives_after_the_error_message_not_before(self, app, monkeypatch):
        """Order matters: the notice annotates the request that just happened."""
        _seed_pending_notice(app)

        async def _explode(question, history, on_step=None):  # noqa: ARG001 - signature match
            raise RuntimeError("provider hung up")

        monkeypatch.setattr(app, "invoke_agent", _explode)
        await app.on_message(SimpleNamespace(content="q"))

        error_at = next(i for i, s in enumerate(_FakeMessage.sent) if "An error occurred" in s)
        notice_at = next(i for i, s in enumerate(_FakeMessage.sent) if NOTICE in s)
        assert error_at < notice_at

    async def test_it_is_drained_so_the_session_sees_it_once(self, app, monkeypatch):
        _seed_pending_notice(app)

        async def _explode(question, history, on_step=None):  # noqa: ARG001 - signature match
            raise RuntimeError("provider hung up")

        monkeypatch.setattr(app, "invoke_agent", _explode)
        await app.on_message(SimpleNamespace(content="q"))
        await app.on_message(SimpleNamespace(content="q again"))

        assert sum(NOTICE in shown for shown in _FakeMessage.sent) == 1

    async def test_nothing_is_shown_when_nothing_was_redacted(self, app, monkeypatch):
        """An empty notice must not become a blank message in the transcript."""
        app.cl.user_session.set(app._NOTICE_STATE_KEY, redaction.NoticeState())

        async def _explode(question, history, on_step=None):  # noqa: ARG001 - signature match
            raise RuntimeError("provider hung up")

        monkeypatch.setattr(app, "invoke_agent", _explode)
        await app.on_message(SimpleNamespace(content="q"))

        assert not any(shown.startswith("_") for shown in _FakeMessage.sent)


class TestTheUiFlagsFromTheCli:
    """``aorta chat ui`` hands these over in the environment; nothing else can."""

    def test_no_wait_skips_the_startup_preflight(self, monkeypatch):
        from aorta.chat.config import UI_NO_WAIT_ENV

        monkeypatch.setenv(UI_NO_WAIT_ENV, "1")
        monkeypatch.setitem(sys.modules, "chainlit", _fake_chainlit())
        monkeypatch.delitem(sys.modules, "aorta.chat.ui.app", raising=False)
        import aorta.chat.ui.app as app_module

        assert app_module._SKIP_PREFLIGHT is True

    def test_it_preflights_by_default(self, app):
        assert app._SKIP_PREFLIGHT is False

    def test_verbose_raises_the_log_level(self, monkeypatch):
        from aorta.chat.config import UI_VERBOSE_ENV

        monkeypatch.setenv(UI_VERBOSE_ENV, "1")
        monkeypatch.setitem(sys.modules, "chainlit", _fake_chainlit())
        monkeypatch.delitem(sys.modules, "aorta.chat.ui.app", raising=False)
        import aorta.chat.ui.app as app_module

        assert app_module._VERBOSE is True


# ── two browser sessions in one process ───────────────────────────────────


@dataclass
class _Browser:
    """One browser session: its own ``user_session`` store and its own transcript."""

    values: dict = field(default_factory=dict)
    sent: list[str] = field(default_factory=list)


#: Chainlit resolves both ``cl.Message`` and ``cl.user_session`` from the session
#: in context, not from process state. The single-session fakes above have no
#: need to model that; a concurrency test does, so these do.
_current_browser: ContextVar[_Browser] = ContextVar("aorta_test_browser")


class _SessionScopedMessage:
    def __init__(self, content: str = "") -> None:
        self.content = content

    async def send(self) -> None:
        _current_browser.get().sent.append(self.content)

    async def remove(self) -> None:
        pass


class _SessionScopedUserSession:
    def get(self, key, default=None):
        return _current_browser.get().values.get(key, default)

    def set(self, key, value) -> None:
        _current_browser.get().values[key] = value


class _FakeBackend:
    def describe(self) -> str:
        return "fake backend"

    async def preflight(self) -> None:
        pass


#: One filesystem path, no addresses.
ALICE_ASKS = "the run under /home/alice/models/llama-70b failed"
#: One path and two IPv4 addresses, so Alice's notice and Bob's differ in wording
#: and each session can be shown to have been told about its *own* redaction.
BOB_ASKS = "hosts 10.42.7.9 and 10.42.7.10 under /home/bob/runs/latest are down"


#: Long enough that a loaded runner never trips it, short enough that a stuck
#: rendezvous fails the job rather than running it to the CI platform limit.
_RENDEZVOUS_TIMEOUT = 5.0


async def _meet(event: asyncio.Event, name: str) -> None:
    """Wait for the peer session, bounded.

    The ``finally`` blocks around each rendezvous cover a peer that fails
    *inside* the guarded region. They cannot cover one that fails before
    reaching the counter -- ``on_start`` raising, or ``_install`` leaving
    something unpatched -- and an unbounded ``Event.wait()`` turns that into a
    hung run rather than a red one. ``pytest-timeout`` is installed but nothing
    arms it, so this is the only thing standing between a starved rendezvous
    and a job that burns a runner reporting nothing.
    """
    try:
        await asyncio.wait_for(event.wait(), timeout=_RENDEZVOUS_TIMEOUT)
    except asyncio.TimeoutError:
        raise AssertionError(
            f"peer session never reached {name} within {_RENDEZVOUS_TIMEOUT}s"
        ) from None


def _notices(browser: _Browser) -> list[str]:
    return [shown for shown in browser.sent if "aorta chat: redacted" in shown]


def _the_notice(browser: _Browser) -> str:
    """The one notice this session was shown.

    Asserted rather than indexed: a session that was starved of its disclosure
    is the failure these tests exist to catch, and ``_notices(x)[0]`` reports it
    as a bare ``IndexError`` that names neither the session nor the promise.
    """
    shown = _notices(browser)
    assert len(shown) == 1, f"expected exactly one notice, got {shown}"
    return shown[0]


class TestTwoSessionsInOneProcess:
    """Why the notice cannot be keyed on process state.

    Unlike the tests above these drive the real ``redact_for_send`` rather than
    seeding ``pending`` by hand, so they cover the whole chain the CLI gets for
    free: a redaction inside a graph turn, this session's
    :class:`redaction.NoticeState`, and a ``cl.Message`` in this session's
    transcript. Overlapping the two turns is what makes it a proof -- the
    shortcut #437 rules out, resetting a module global in ``on_chat_start``,
    satisfies every sequential arrangement and only fails when two sessions are
    live at once.
    """

    @staticmethod
    def _install(app, monkeypatch, invoke) -> None:
        monkeypatch.setattr(app.cl, "Message", _SessionScopedMessage)
        monkeypatch.setattr(app.cl, "user_session", _SessionScopedUserSession())
        monkeypatch.setattr(app, "get_backend", lambda: _FakeBackend())
        monkeypatch.setattr(app, "invoke_agent", invoke)
        reset_settings()

    @staticmethod
    async def _session(app, browser: _Browser, *questions: str) -> None:
        """One browser session's whole life, in its own context."""
        token = _current_browser.set(browser)
        try:
            await app.on_start()
            for question in questions:
                await app.on_message(SimpleNamespace(content=question))
        finally:
            _current_browser.reset(token)

    @staticmethod
    def _overlapping_turn():
        """An ``invoke_agent`` that redacts, then blocks until its peer has too."""
        both_redacted = asyncio.Event()
        arrived = 0

        async def turn(question, history, on_step=None):  # noqa: ARG001 - signature match
            nonlocal arrived

            async def node() -> None:
                redaction.redact_for_send([HumanMessage(content=question)])

            try:
                # A child task is how LangGraph runs a node, and the shape the
                # binding in ``on_message`` has to survive.
                await asyncio.create_task(node())
            finally:
                # In ``finally`` because ``on_message`` swallows a graph
                # failure: without it, one session raising here would leave the
                # other waiting on a rendezvous nobody can reach. ``_meet``
                # bounds that wait, but only this releases the peer at once and
                # reports the real assertion rather than a timeout.
                arrived += 1
                if arrived == 2:
                    both_redacted.set()
            await _meet(both_redacted, "both_redacted")
            return f"answered: {question}", [], {}

        return turn

    async def test_neither_session_suppresses_the_other(self, app, monkeypatch):
        self._install(app, monkeypatch, self._overlapping_turn())
        alice, bob = _Browser(), _Browser()

        await asyncio.gather(
            self._session(app, alice, ALICE_ASKS),
            self._session(app, bob, BOB_ASKS),
        )

        assert len(_notices(alice)) == 1
        assert len(_notices(bob)) == 1

    async def test_each_session_is_told_about_its_own_redaction(self, app, monkeypatch):
        """Not merely *a* notice each: the one describing what that session sent."""
        self._install(app, monkeypatch, self._overlapping_turn())
        alice, bob = _Browser(), _Browser()

        await asyncio.gather(
            self._session(app, alice, ALICE_ASKS),
            self._session(app, bob, BOB_ASKS),
        )

        assert "IPv4" not in _the_notice(alice)
        assert "IPv4" in _the_notice(bob)

    async def test_both_are_told_how_to_turn_it_off(self, app, monkeypatch):
        """Decision 16's second half, in the same words as the CLI line."""
        self._install(app, monkeypatch, self._overlapping_turn())
        alice, bob = _Browser(), _Browser()

        await asyncio.gather(
            self._session(app, alice, ALICE_ASKS),
            self._session(app, bob, BOB_ASKS),
        )

        for browser in (alice, bob):
            notice = _the_notice(browser)
            assert "--no-redact" in notice
            assert "redact = false" in notice

    async def test_each_session_is_told_once_across_its_own_turns(self, app, monkeypatch):
        """Once per session, not once per redacting turn, with sessions overlapping."""
        self._install(app, monkeypatch, self._overlapping_turn())
        alice, bob = _Browser(), _Browser()

        await asyncio.gather(
            self._session(app, alice, ALICE_ASKS, ALICE_ASKS),
            self._session(app, bob, BOB_ASKS, BOB_ASKS),
        )

        assert len(_notices(alice)) == 1
        assert len(_notices(bob)) == 1

    async def test_no_session_state_leaks_to_the_process(self, app, monkeypatch):
        """The CLI fallback must come out of a UI process's turns untouched.

        ``aorta chat`` and ``aorta chat ask`` bind nothing and rely on that
        process-wide state being their session state.
        """
        self._install(app, monkeypatch, self._overlapping_turn())
        # Explicit, because an earlier test in the run may legitimately have
        # emitted onto the process-wide state; what is asserted is that these
        # two sessions do not.
        redaction.reset_session_notice()
        alice, bob = _Browser(), _Browser()

        await asyncio.gather(
            self._session(app, alice, ALICE_ASKS),
            self._session(app, bob, BOB_ASKS),
        )

        # The process-wide assertions below are satisfied by absence: two
        # sessions that redacted nothing at all leave the same state behind as
        # two that kept their disclosures to themselves. Pin the positive here
        # too rather than lean on the sibling test that happens to establish it
        # on the same fixtures.
        assert len(_notices(alice)) == 1
        assert len(_notices(bob)) == 1
        assert redaction.current_notice_state().emitted is False
        assert redaction.current_notice_state().pending is None
