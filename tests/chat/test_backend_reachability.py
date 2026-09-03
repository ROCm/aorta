"""Reachability of the LLM backend, reported honestly.

``preflight`` used to be the only readiness primitive, so every caller
inherited the interactive path's tolerance. Two consequences, and they pull in
opposite directions:

* ``aorta chat doctor`` printed ``[ ok ] llm backend`` with nothing listening,
  and spent preflight's whole budget doing it -- 302s measured against a
  closed port. A diagnostic that gives a confident all-clear on the most likely
  failure is worse than no diagnostic.
* ``preflight`` must nonetheless *stay* permissive. It waits five minutes for a
  server still loading weights and then starts anyway, and making that a hard
  failure would break the REPL during a warm-up the user can watch progressing.

So ``doctor`` calls ``probe``, which decides reachability itself on a
diagnostic's budget, and nothing about what a session tolerates changes. The
first group below would pass against a ``preflight`` that simply raised, which
is why the second group is here: it is the part that would be a regression.

The last group covers the other half of the same condition: a query with no
backend produced over 180 lines of httpcore/httpx/langchain frames ending in
"Connection error.", which named the address it failed to reach nowhere in any
of them.
"""

from __future__ import annotations

import logging
import socket
import time
from typing import Any

import httpx
import pytest

from aorta.chat import doctor
from aorta.chat.config import settings
from aorta.chat.inference.providers.local_vllm import LocalVLLMBackend
from aorta.chat.inference.unreachable import BackendUnreachableError, is_connection_failure

#: Generous ceiling for "answered in a diagnostic's time". The point is the
#: order of magnitude: the budget is five seconds, the bug took 302, and a
#: loaded CI runner is allowed to be slow without turning this into a flake.
RESPONSIVE_SECONDS = 60


@pytest.fixture()
def closed_port() -> int:
    """A loopback port that refuses connections, held so nothing can take it.

    Bound but never listened on, which is what makes a connect fail fast with
    ECONNREFUSED. Reserving it matters: an ephemeral port merely *observed* to
    be free can be claimed by something else mid-test, and this suite must never
    reach a live server.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        yield sock.getsockname()[1]


@pytest.fixture()
def nothing_listening(monkeypatch, closed_port: int) -> str:
    """Configure the vLLM backend against the closed port. Returns its base URL."""
    base_url = f"http://127.0.0.1:{closed_port}/v1"
    monkeypatch.setattr(settings, "llm_provider", "vllm")
    monkeypatch.setattr(settings, "vllm_base_url", base_url)
    return base_url


def _backend_check(report: doctor.Report) -> doctor.Check:
    found = [check for check in report.checks if check.name == "llm backend"]
    assert found, f"no llm backend check; got {[c.name for c in report.checks]}"
    return found[0]


@pytest.fixture()
def packets_dropped(monkeypatch) -> None:
    """A host that swallows the SYN instead of refusing it: a wrong host, a firewall."""

    async def _get(self, url: str, **kwargs: Any):
        raise httpx.ConnectTimeout("timed out", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)


# Sync tests that call ``doctor._check_backend`` are kept out of any class that
# also holds async ones. ``_check_backend`` uses ``asyncio.run``, and
# pytest-asyncio's function-scoped loop is still current when a sync test runs
# after an async sibling, which turns the check into a RuntimeError about nested
# loops -- a green test asserting nothing.


class TestDoctorReportsAnUnreachableBackend:
    """The reason this change exists. Every assertion here failed before it."""

    def test_a_closed_port_is_a_failure_not_an_ok(self, nothing_listening: str):
        report = doctor.Report()
        doctor._check_backend(report)
        assert _backend_check(report).status == doctor.FAIL

    def test_it_answers_in_seconds_rather_than_minutes(self, nothing_listening: str):
        """The old path waited out preflight's full five-minute budget."""
        report = doctor.Report()
        start = time.monotonic()
        doctor._check_backend(report)
        elapsed = time.monotonic() - start

        assert _backend_check(report).status == doctor.FAIL
        assert elapsed < RESPONSIVE_SECONDS, f"the backend check took {elapsed:.0f}s"

    def test_the_hint_names_the_address_and_both_ways_to_change_it(self, nothing_listening: str):
        report = doctor.Report()
        doctor._check_backend(report)
        hint = _backend_check(report).hint

        assert nothing_listening in hint
        assert "AORTA_CHAT_VLLM_BASE_URL" in hint
        assert "aorta chat config init" in hint

    def test_the_hint_is_not_prefixed_with_an_exception_class_name(
        self, nothing_listening: str
    ):
        """BackendUnreachableError's message is already operator-facing prose."""
        report = doctor.Report()
        doctor._check_backend(report)
        assert not _backend_check(report).hint.startswith("BackendUnreachableError")

    def test_run_checks_reaches_the_backend_check_at_all(self, nothing_listening: str):
        """Guards the wiring, not just the primitive: run_checks must call probe."""
        report = doctor.run_checks(backend=True)
        assert _backend_check(report).status == doctor.FAIL


class TestTheHealthUrlDropsOnlyTheApiSuffix:
    """``replace("/v1", "")`` removed every occurrence, not the suffix.

    A base URL carrying ``/v1`` in the authority or an earlier path segment was
    rewritten into a malformed address, so preflight and probe polled somewhere
    that could not answer and reported a healthy server unreachable.
    """

    @pytest.mark.parametrize(
        ("base_url", "expected"),
        [
            ("http://localhost:8000/v1", "http://localhost:8000/health"),
            ("http://localhost:8000/v1/", "http://localhost:8000/health"),
            # The authority contains "/v1" as part of "//v1.example".
            ("http://v1.example/v1", "http://v1.example/health"),
            ("https://host/v1-gateway/v1", "https://host/v1-gateway/health"),
            ("https://host/api/v1/v1", "https://host/api/v1/health"),
            # No suffix to drop: the whole base is a prefix of the health path.
            ("http://localhost:8000", "http://localhost:8000/health"),
        ],
    )
    def test_it_strips_a_trailing_v1_and_nothing_else(self, monkeypatch, base_url, expected):
        monkeypatch.setattr(settings, "vllm_base_url", base_url)

        assert LocalVLLMBackend().health_url() == expected


class TestPreflightStaysPermissive:
    """What the doctor fix must NOT change.

    Starting against a server that is still loading weights is deliberate. A
    fix that made ``preflight`` raise would satisfy the class above and regress
    the interactive path, so these pin the tolerance separately.
    """

    @pytest.mark.asyncio
    async def test_preflight_does_not_raise_when_nothing_answers(self, nothing_listening: str):
        backend = LocalVLLMBackend()
        assert await backend.preflight(timeout=1, interval=1) is None

    @pytest.mark.asyncio
    async def test_preflight_says_it_is_starting_anyway(
        self, nothing_listening: str, caplog
    ):
        backend = LocalVLLMBackend()
        with caplog.at_level(logging.WARNING):
            await backend.preflight(timeout=1, interval=1)
        assert "starting anyway" in caplog.text

    def test_preflights_default_budget_is_still_minutes(self):
        """A quietly shortened warm-up window is the regression to watch for."""
        from aorta.chat.inference.providers import local_vllm

        assert local_vllm.PREFLIGHT_TIMEOUT == 300

    @pytest.mark.asyncio
    async def test_a_server_that_is_still_warming_up_is_waited_for(self, monkeypatch):
        """503 then 200: preflight must keep polling rather than give up at the first."""
        codes = iter([503, 503, 200])
        seen: list[int] = []

        async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
            code = next(codes)
            seen.append(code)
            return httpx.Response(code, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", _get)
        await LocalVLLMBackend().preflight(timeout=30, interval=0)
        assert seen == [503, 503, 200]

    @pytest.mark.asyncio
    async def test_probe_accepts_a_server_that_answers(self, monkeypatch):
        """The permissive and strict paths agree whenever the server is up."""

        async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
            return httpx.Response(200, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", _get)
        assert await LocalVLLMBackend().probe(timeout=5) is None


class TestATimeoutIsNotAnEscapeHatch:
    """A host that drops packets rather than refusing them.

    ``httpx.ConnectTimeout`` is a ``TimeoutException``, not a ``ConnectError``,
    so a catch naming those two subclasses let it past -- and it is the ordinary
    symptom of a wrong host or a firewall, not an exotic case. Escaping broke
    both callers at once: ``preflight`` raised where it promises not to, and
    ``probe`` surfaced a bare ``ConnectTimeout`` whose message is empty.
    """

    @pytest.mark.asyncio
    async def test_preflight_still_starts_anyway(self, packets_dropped):
        assert await LocalVLLMBackend().preflight(timeout=1, interval=0) is None

    @pytest.mark.asyncio
    async def test_probe_reports_it_as_unreachable_with_the_hint(self, packets_dropped):
        with pytest.raises(BackendUnreachableError) as excinfo:
            await LocalVLLMBackend().probe(timeout=1)
        assert "AORTA_CHAT_VLLM_BASE_URL" in str(excinfo.value)


class TestATimeoutReachesTheReportAsAdvice:
    """Same condition, seen from the command. Sync-only class -- see the note above."""

    def test_the_doctor_names_the_address_not_a_bare_httpx_class(
        self, packets_dropped, nothing_listening: str
    ):
        report = doctor.Report()
        doctor._check_backend(report)
        check = _backend_check(report)

        assert check.status == doctor.FAIL
        assert "ConnectTimeout" not in check.hint
        assert nothing_listening in check.hint


class TestTheDoctorSurvivesAMissingExtra:
    """The install this check exists to diagnose is the one without the extra.

    ``unreachable`` reaches httpx and openai, so importing it at function scope
    -- outside the guard that turns a missing dependency into a finding -- made
    ``aorta chat doctor`` crash on exactly that install.
    """

    def test_a_missing_dependency_is_a_finding_not_a_traceback(self, monkeypatch):
        import builtins
        import sys

        real_import = builtins.__import__

        def _no_openai(name: str, *args: Any, **kwargs: Any):
            if name == "openai" or name.startswith("openai."):
                raise ModuleNotFoundError("No module named 'openai'", name="openai")
            return real_import(name, *args, **kwargs)

        # The eviction is load-bearing: the module is already in sys.modules
        # from this file's own imports, so without it `from ... import` never
        # reaches the patched __import__ and the test passes proving nothing.
        # Only this module is evicted -- dropping the provider factory too would
        # leave sys.modules and the parent package's attribute pointing at
        # different objects, and later tests patch it through the attribute.
        monkeypatch.delitem(sys.modules, "aorta.chat.inference.unreachable")
        monkeypatch.setattr(builtins, "__import__", _no_openai)

        report = doctor.Report()
        doctor._check_backend(report)
        check = _backend_check(report)
        assert check.status == doctor.FAIL
        assert "openai" in check.detail


class TestConnectionFailuresAreRecognisedThroughAWrapper:
    """LangGraph wraps a node's exception and the openai SDK chains to httpx."""

    def test_a_direct_httpx_connect_error_counts(self):
        assert is_connection_failure(httpx.ConnectError("refused"))

    def test_a_connect_timeout_counts(self):
        assert is_connection_failure(httpx.ConnectTimeout("timed out"))

    def test_a_cause_several_links_down_counts(self):
        inner = httpx.ConnectError("refused")
        middle = RuntimeError("openai wrapped it")
        middle.__cause__ = inner
        outer = RuntimeError("langgraph wrapped that")
        outer.__cause__ = middle
        assert is_connection_failure(outer)

    def test_an_ordinary_bug_does_not_count(self):
        assert not is_connection_failure(ValueError("off by one"))

    def test_a_self_referential_chain_terminates(self):
        """A __context__ cycle must not hang the failure renderer."""
        first = RuntimeError("a")
        second = RuntimeError("b")
        first.__context__ = second
        second.__context__ = first
        assert not is_connection_failure(first)


class TestAskExplainsRatherThanTracebacks:
    """What replaced the 180-odd-line trace, and what still keeps its own."""

    @pytest.fixture()
    def unreachable_backend(self, nothing_listening: str) -> LocalVLLMBackend:
        return LocalVLLMBackend()

    @pytest.fixture()
    def connection_failure(self) -> Exception:
        """A connect error as the graph delivers it: wrapped, cause chained."""
        wrapped = RuntimeError("Connection error.")
        wrapped.__cause__ = httpx.ConnectError("[Errno 111] Connection refused")
        return wrapped

    def test_the_message_names_the_url_instead_of_a_traceback(
        self, unreachable_backend, connection_failure, nothing_listening: str
    ):
        from aorta.cli import chat as cli

        message = cli._failure_message(connection_failure, unreachable_backend)
        assert "unreachable" in message.lower()
        assert nothing_listening in message
        assert "Traceback" not in message

    def test_the_message_is_short_enough_to_read(
        self, unreachable_backend, connection_failure
    ):
        """It replaced 180-odd lines. A dozen is a message; a hundred is a trace."""
        from aorta.cli import chat as cli

        message = cli._failure_message(connection_failure, unreachable_backend)
        assert len(message.splitlines()) < 15

    def test_the_message_says_what_to_do_next(
        self, unreachable_backend, connection_failure
    ):
        from aorta.cli import chat as cli

        message = cli._failure_message(connection_failure, unreachable_backend)
        assert "AORTA_CHAT_VLLM_BASE_URL" in message
        assert "aorta chat config init" in message

    def test_the_traceback_is_still_emitted_under_verbose(
        self, unreachable_backend, connection_failure, caplog
    ):
        """``-v`` sets DEBUG, which is the level this record is logged at."""
        from aorta.cli import chat as cli

        with caplog.at_level(logging.DEBUG, logger="aorta.cli.chat"):
            cli._failure_message(connection_failure, unreachable_backend)

        records = [r for r in caplog.records if r.exc_info]
        assert records, "no record carried the traceback"
        assert any("connection failure" in r.getMessage().lower() for r in records)

    def test_a_quiet_run_does_not_emit_the_traceback(
        self, unreachable_backend, connection_failure, caplog
    ):
        from aorta.cli import chat as cli

        with caplog.at_level(logging.INFO, logger="aorta.cli.chat"):
            cli._failure_message(connection_failure, unreachable_backend)
        assert not [r for r in caplog.records if r.exc_info]

    def test_an_unrelated_bug_keeps_its_traceback_and_the_generic_message(
        self, unreachable_backend, caplog
    ):
        """Anything that is not a connection failure is a bug until proven otherwise."""
        from aorta.cli import chat as cli

        with caplog.at_level(logging.ERROR, logger="aorta.cli.chat"):
            message = cli._failure_message(ValueError("off by one"), unreachable_backend)

        assert message == "An error occurred while processing your request."
        assert [r for r in caplog.records if r.exc_info]

    def test_no_backend_falls_back_to_the_generic_message(self, connection_failure):
        """``_ask_once`` is still callable without one, so this must not explode."""
        from aorta.cli import chat as cli

        message = cli._failure_message(connection_failure, None)
        assert message == "An error occurred while processing your request."

    @pytest.mark.asyncio
    async def test_ask_renders_the_hint_and_reports_failure(
        self, unreachable_backend, connection_failure, capsys, nothing_listening: str
    ):
        """End to end through the one-shot path: the text lands, the status is 1."""
        from aorta.cli import chat as cli

        async def _invoke(query: str, history: list):
            raise connection_failure

        _, ok = await cli._ask_once(
            _invoke, "q", [], "plain", quiet=False, backend=unreachable_backend
        )
        assert ok is False
        assert nothing_listening in capsys.readouterr().out
