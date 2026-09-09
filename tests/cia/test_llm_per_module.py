"""A module's own LM settings survive whichever agent reached DSPy first.

``ensure_configured`` is first-writer-wins, and Autopsy asked it for a larger
model and a larger budget from ``TriageRouter.__init__``. By then Watch's poll
loop had already configured the default, so Autopsy's request was dropped and
it reasoned at Watch's settings -- at 1024 tokens its reasoning consumed the
whole budget, the response was cut off before any output field arrived, and the
router scored every bundle ``tooling_gap`` at confidence 0.0. Nothing in the
verdict said the reasoning step had been truncated.

The fix is that a module binds its own LM. These tests hold that seam shut, and
hold ``ensure_configured`` to saying so when it discards arguments.
"""

from __future__ import annotations

import logging

import pytest

from aorta.cia import llm as llm_mod
from aorta.cia.llm import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    build_lm,
    configure_dspy,
    ensure_configured,
)


@pytest.fixture
def built(monkeypatch):
    """Capture what each LM is built with, without reaching a network."""
    calls: list[dict] = []

    class FakeLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.kwargs = kwargs

    monkeypatch.setattr(llm_mod.dspy, "LM", FakeLM)
    monkeypatch.setattr(llm_mod.dspy, "configure", lambda **_: None)
    monkeypatch.setattr(llm_mod, "_configured", False)
    for var in ("LITELLM_MODEL", "LITELLM_API_BASE", "LITELLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # These tests are about which argument wins, not about where the endpoint
    # comes from -- that is tests/cia/test_shared_provider.py. Pin the resolved
    # provider so the chat settings on the developer's machine cannot decide it.
    monkeypatch.setattr(llm_mod, "chat_provider", lambda **_: ("http://pinned:1/v1", "k", ""))
    return calls


class TestExplicitArgumentsWin:
    def test_an_explicit_model_is_not_overridden_by_the_environment(self, built, monkeypatch):
        """The environment used to win, so a module could not insist."""
        monkeypatch.setenv("LITELLM_MODEL", "claude-haiku-4-5")
        build_lm(model="claude-sonnet-4-6")
        assert built[0]["model"] == "openai/claude-sonnet-4-6"

    def test_the_configuration_supplies_a_model_when_none_is_named(self, built, monkeypatch):
        monkeypatch.setattr(
            llm_mod, "chat_provider", lambda **_: ("http://pinned:1/v1", "k", "qwen3-35b")
        )
        build_lm()
        assert built[0]["model"] == "openai/qwen3-35b"

    def test_a_default_model_applies_when_neither_is_given(self, built):
        """The configuration names no model and the caller names none."""
        build_lm()
        assert built[0]["model"] == f"openai/{DEFAULT_MODEL}"

    def test_an_explicit_endpoint_and_key_win_too(self, built, monkeypatch):
        monkeypatch.setenv("LITELLM_API_BASE", "http://elsewhere:4000")
        monkeypatch.setenv("LITELLM_API_KEY", "from-env")
        build_lm(api_base="http://mine:4000", api_key="mine")
        assert built[0]["api_base"] == "http://mine:4000"
        assert built[0]["api_key"] == "mine"


class TestBudget:
    def test_the_default_budget_leaves_room_to_reason_and_answer(self, built):
        build_lm()
        assert built[0]["max_tokens"] == DEFAULT_MAX_TOKENS

    def test_the_default_is_well_clear_of_the_budget_that_truncated_autopsy(self):
        assert DEFAULT_MAX_TOKENS >= 4096

    def test_a_module_can_ask_for_more(self, built):
        build_lm(max_tokens=8192)
        assert built[0]["max_tokens"] == 8192


class TestTheRouterKeepsItsOwnSettings:
    @staticmethod
    def _bind(monkeypatch):
        import aorta.cia.autopsy.router as router_mod

        bound: list = []

        class FakeReAct:
            def __init__(self, *a, **k):
                pass

            def set_lm(self, lm):
                bound.append(lm)

        monkeypatch.setattr(router_mod.dspy, "ReAct", FakeReAct)
        return router_mod, bound

    def test_the_router_binds_its_budget_whoever_configured_the_default(self, built, monkeypatch):
        """The regression itself: Watch configures first, Autopsy keeps its budget."""
        router_mod, bound = self._bind(monkeypatch)

        ensure_configured()  # Watch, arriving first with the cheap default.
        router_mod.TriageRouter()

        assert bound, "the router must bind an LM of its own"
        assert bound[0].kwargs["max_tokens"] == router_mod.TriageRouter.MAX_TOKENS

    def test_the_router_asks_for_more_than_the_budget_that_truncated_it(self):
        import aorta.cia.autopsy.router as router_mod

        assert router_mod.TriageRouter.MAX_TOKENS > 1024

    def test_the_router_follows_the_model_the_operator_configured(self, built, monkeypatch):
        """Pinning a model in code would override the deployment's choice."""
        monkeypatch.setattr(
            llm_mod, "chat_provider", lambda **_: ("http://pinned:1/v1", "k", "qwen3-35b")
        )
        router_mod, bound = self._bind(monkeypatch)

        router_mod.TriageRouter()

        assert bound[0].kwargs["model"] == "openai/qwen3-35b"

    def test_the_router_does_not_name_a_model_of_its_own(self):
        """A vendor model in the code is the site-specific default in disguise."""
        import inspect

        import aorta.cia.autopsy.router as router_mod

        source = inspect.getsource(router_mod.TriageRouter)
        assert "claude" not in source.lower()
        assert "gpt-" not in source.lower()


class TestEnsureConfiguredSaysWhenItDiscards:
    def test_the_first_caller_configures_the_default(self, built):
        ensure_configured()
        assert len(built) == 1

    def test_a_later_caller_passing_nothing_is_silent(self, built, caplog):
        ensure_configured()
        with caplog.at_level(logging.WARNING):
            ensure_configured()
        assert not caplog.records
        assert len(built) == 1

    def test_a_later_caller_passing_settings_is_warned_rather_than_ignored(self, built, caplog):
        """This is what happened to Autopsy, and it happened in silence."""
        ensure_configured()
        with caplog.at_level(logging.WARNING):
            ensure_configured(model="claude-sonnet-4-6", max_tokens=2048)

        assert caplog.records, "discarding a module's settings must not be silent"
        message = caplog.records[0].getMessage()
        assert "model" in message and "max_tokens" in message
        assert "build_lm" in message

    def test_a_discarded_request_still_does_not_rebuild_the_default(self, built, caplog):
        ensure_configured()
        with caplog.at_level(logging.WARNING):
            ensure_configured(model="claude-sonnet-4-6")
        assert len(built) == 1


def test_configure_dspy_still_serves_the_callers_that_pass_nothing(built):
    configure_dspy()
    assert built[0]["model"] == f"openai/{DEFAULT_MODEL}"
    assert built[0]["cache"] is False
