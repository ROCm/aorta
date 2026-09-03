"""The web UI's per-request redaction disclosure, including on the error path.

Decision 16 promises the user is told what a request had removed before it
left. The UI drained that notice only after a successful answer, so a request
that was redacted and *then* failed in the graph or the provider disclosed
nothing -- and a user who stops after the failure is never told. The send had
already happened by then, so the disclosure is owed either way.

``aorta.chat.ui.app`` imports ``chainlit`` at module scope and the ``chat-ui``
extra is a separate install, so a fake is put in ``sys.modules`` first. The
alternative -- skipping when Chainlit is absent -- would leave a user-facing
security guarantee untested on the configuration these tests actually run in.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from aorta.chat import redaction

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

        async def _answer(question, history):  # noqa: ARG001 - signature match
            return "the answer", [], {}

        monkeypatch.setattr(app, "invoke_agent", _answer)
        await app.on_message(SimpleNamespace(content="why did cell 3 fail?"))

        assert "the answer" in _FakeMessage.sent
        assert any(NOTICE in shown for shown in _FakeMessage.sent)

    async def test_after_a_failure_too(self, app, monkeypatch):
        """The request already left; the failure does not cancel the disclosure."""
        _seed_pending_notice(app)

        async def _explode(question, history):  # noqa: ARG001 - signature match
            raise RuntimeError("provider hung up")

        monkeypatch.setattr(app, "invoke_agent", _explode)
        await app.on_message(SimpleNamespace(content="why did cell 3 fail?"))

        assert any("An error occurred" in shown for shown in _FakeMessage.sent)
        assert any(NOTICE in shown for shown in _FakeMessage.sent)

    async def test_it_arrives_after_the_error_message_not_before(self, app, monkeypatch):
        """Order matters: the notice annotates the request that just happened."""
        _seed_pending_notice(app)

        async def _explode(question, history):  # noqa: ARG001 - signature match
            raise RuntimeError("provider hung up")

        monkeypatch.setattr(app, "invoke_agent", _explode)
        await app.on_message(SimpleNamespace(content="q"))

        error_at = next(i for i, s in enumerate(_FakeMessage.sent) if "An error occurred" in s)
        notice_at = next(i for i, s in enumerate(_FakeMessage.sent) if NOTICE in s)
        assert error_at < notice_at

    async def test_it_is_drained_so_the_session_sees_it_once(self, app, monkeypatch):
        _seed_pending_notice(app)

        async def _explode(question, history):  # noqa: ARG001 - signature match
            raise RuntimeError("provider hung up")

        monkeypatch.setattr(app, "invoke_agent", _explode)
        await app.on_message(SimpleNamespace(content="q"))
        await app.on_message(SimpleNamespace(content="q again"))

        assert sum(NOTICE in shown for shown in _FakeMessage.sent) == 1

    async def test_nothing_is_shown_when_nothing_was_redacted(self, app, monkeypatch):
        """An empty notice must not become a blank message in the transcript."""
        app.cl.user_session.set(app._NOTICE_STATE_KEY, redaction.NoticeState())

        async def _explode(question, history):  # noqa: ARG001 - signature match
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
