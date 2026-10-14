"""Doctor verifies native tools for local vLLM without billing remote users."""

from __future__ import annotations

from aorta.chat import doctor
from aorta.chat.config import settings
from aorta.chat.inference.providers import factory


class _Backend:
    name = "vllm"

    def __init__(
        self,
        *,
        health_error: Exception | None = None,
        tools_error: Exception | None = None,
    ):
        self.health_error = health_error
        self.tools_error = tools_error
        self.tools_calls = 0

    async def probe(self, timeout=None):
        if self.health_error is not None:
            raise self.health_error

    async def probe_tools(self, timeout=None):
        self.tools_calls += 1
        if self.tools_error is not None:
            raise self.tools_error

    def describe(self):
        return "local vLLM test endpoint"


def _check(report: doctor.Report, name: str) -> doctor.Check:
    return next(check for check in report.checks if check.name == name)


def test_native_local_vllm_gets_a_real_tools_check(monkeypatch):
    backend = _Backend()
    monkeypatch.setattr(factory, "get_backend", lambda: backend)
    monkeypatch.setattr(settings, "llm_tool_mode", "native")

    report = doctor.Report()
    doctor._check_backend(report)

    assert _check(report, "llm backend").status == doctor.OK
    assert _check(report, "llm native tools").status == doctor.OK
    assert backend.tools_calls == 1


def test_a_failed_tools_probe_is_a_doctor_failure(monkeypatch):
    backend = _Backend(tools_error=RuntimeError("server rejected tools"))
    monkeypatch.setattr(factory, "get_backend", lambda: backend)
    monkeypatch.setattr(settings, "llm_tool_mode", "native")

    report = doctor.Report()
    doctor._check_backend(report)

    check = _check(report, "llm native tools")
    assert check.status == doctor.FAIL
    assert "server rejected tools" in check.hint


def test_text_mode_does_not_spend_a_tools_request(monkeypatch):
    backend = _Backend()
    monkeypatch.setattr(factory, "get_backend", lambda: backend)
    monkeypatch.setattr(settings, "llm_tool_mode", "text")

    report = doctor.Report()
    doctor._check_backend(report)

    assert all(check.name != "llm native tools" for check in report.checks)
    assert backend.tools_calls == 0


def test_tools_are_skipped_when_health_does_not_answer(monkeypatch):
    backend = _Backend(health_error=RuntimeError("offline"))
    monkeypatch.setattr(factory, "get_backend", lambda: backend)
    monkeypatch.setattr(settings, "llm_tool_mode", "native")

    report = doctor.Report()
    doctor._check_backend(report)

    assert _check(report, "llm backend").status == doctor.FAIL
    assert _check(report, "llm native tools").status == doctor.SKIP
    assert backend.tools_calls == 0
