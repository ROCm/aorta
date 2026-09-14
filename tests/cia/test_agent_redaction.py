"""What the agents send is redacted, the way the graph's is.

``docs/chat/redaction.md`` and ``_send()`` exist because a remote provider
receives retrieved chunks and tool output, and those are rewritten -- paths and
addresses -- before they leave. The agents never go through ``_send``: Watch
ships log tails and job context, Launch discovery ships what it finds under a
home directory, and Autopsy ships bundle evidence, each building its own prompt.

The gate is therefore on the LM object every agent is given, for the reason the
comment on ``_send`` gives: a module added later cannot bypass what it does not
have to remember to call.
"""

from __future__ import annotations

import dspy
import pytest

from aorta.cia import llm as llm_mod
from aorta.cia.llm import RedactingLM, redact


class _Wire(dspy.LM):
    """Stands in for the network, below the gate."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen: list = []

    def forward(self, prompt=None, messages=None, **kwargs):
        self.seen.append(("forward", prompt, messages))
        return []

    async def aforward(self, prompt=None, messages=None, **kwargs):
        self.seen.append(("aforward", prompt, messages))
        return []


class _Probe(RedactingLM, _Wire):
    pass


@pytest.fixture(autouse=True)
def redaction_on(monkeypatch):
    """Pin the switch these tests are about.

    ``settings`` is a process-wide singleton and ``apply_cli_overrides`` mutates
    it, so an in-process ``aorta chat --no-redact`` anywhere earlier in the
    session leaves redaction off for everything after it. A test asserting that
    redaction happens should say which setting it is assuming rather than
    inherit whatever ran before it.
    """
    from aorta.chat.config import settings

    monkeypatch.setattr(settings, "redact", True)


@pytest.fixture
def lm() -> _Probe:
    return _Probe(model="openai/x", api_base="http://proxy/v1", api_key="k")


SECRETS = "tail of /home/avsharma/jobs/run.log on 149.28.124.225"


class TestNothingIdentifyingReachesTheWire:
    def test_a_prompt(self, lm):
        lm.forward(prompt=SECRETS)
        _, prompt, _ = lm.seen[0]
        assert "/home/avsharma" not in prompt
        assert "149.28.124.225" not in prompt

    def test_a_message(self, lm):
        lm.forward(messages=[{"role": "user", "content": SECRETS}])
        _, _, messages = lm.seen[0]
        assert "/home/avsharma" not in messages[0]["content"]

    async def test_the_async_path_too(self, lm):
        await lm.aforward(messages=[{"role": "user", "content": SECRETS}])
        _, _, messages = lm.seen[0]
        assert "/home/avsharma" not in messages[0]["content"]

    def test_a_positional_prompt(self, lm):
        """DSPy's __call__ takes items positionally, so those count too."""
        assert "/home/avsharma" not in redact(SECRETS)


class TestItLeavesTheCallerAlone:
    def test_the_caller_keeps_its_own_paths(self, lm):
        """A log Watch is reasoning about is still that log locally."""
        original = [{"role": "user", "content": SECRETS}]
        lm.forward(messages=original)
        assert original[0]["content"] == SECRETS

    def test_a_message_without_string_content_is_untouched(self, lm):
        blocks = [{"role": "user", "content": [{"type": "image"}]}]
        lm.forward(messages=blocks)
        _, _, messages = lm.seen[0]
        assert messages[0]["content"] == [{"type": "image"}]

    def test_a_message_that_needs_nothing_is_unchanged(self, lm):
        clean = [{"role": "user", "content": "step=5 loss=nan"}]
        lm.forward(messages=clean)
        _, _, messages = lm.seen[0]
        assert messages[0]["content"] == "step=5 loss=nan"


class TestTheGateIsOnEveryAgentLM:
    def test_build_lm_returns_one(self, monkeypatch):
        """Not a convention each module has to remember."""
        monkeypatch.setattr(llm_mod, "chat_provider", lambda **_: ("http://p/v1", "k", "m"))
        built = llm_mod.build_lm()
        assert isinstance(built, RedactingLM)

    def test_every_dspy_entry_point_is_covered(self):
        """Which one a module reaches is not this module's business to track."""
        for name in ("forward", "aforward", "__call__", "acall"):
            assert name in vars(RedactingLM), f"{name} is not gated"


class TestWhenTheChatPackageIsAbsent:
    def test_the_text_is_still_redacted(self, monkeypatch):
        """The agents run on a base install, and that is the point.

        This test used to assert the opposite -- that the text passed through
        unchanged -- which encoded the vulnerability as the intended behaviour:
        the [cia]-only install this package advertises was the one that sent
        evidence unredacted. The scrubber is core, so there is nothing to fall
        back from. See tests/cia/test_redaction_without_chat.py.
        """
        real_import = __import__

        def no_chat(name, *args, **kwargs):
            if name.startswith("aorta.chat"):
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", no_chat)
        assert redact(SECRETS) != SECRETS
        assert "/home/avsharma" not in redact(SECRETS)


def test_the_setting_that_turns_it_off_is_respected(monkeypatch):
    """redact = false in chat.toml is a decision the agents follow too."""
    from aorta.chat.config import settings

    monkeypatch.setattr(settings, "redact", False)
    assert redact(SECRETS) == SECRETS
