"""Redaction holds on the install the agents are advertised for.

The gate reached ``aorta.chat.redaction``, which imports ``aorta.chat.config``,
which needs pydantic-settings and — for the profile file — stdlib ``tomllib``,
so 3.11. Neither is in the ``[cia]`` extra. The import therefore failed on
exactly the install this package advertises as its headless one, and the
fallback returned the text unchanged: the base-install path was the one that
sent Watch and Autopsy evidence unredacted, and said nothing about it.

The scrubber itself was never the problem. ``aorta.probe.redaction`` is core —
stdlib and aorta's own bundle code — and ``aorta.chat.redaction`` is a thin
wrapper over the same function. The seam existed; this was reaching it through
the wrong door.
"""

from __future__ import annotations

import builtins
import subprocess
import sys

import pytest

from aorta.cia.llm import RedactionUnavailable, _redact_messages, redact

SECRETS = "tail of /home/avsharma/jobs/run.log on 149.28.124.225"


@pytest.fixture
def without_chat(monkeypatch):
    """A base install: [cia] present, the chat extras absent."""
    real_import = builtins.__import__

    def no_chat(name, *args, **kwargs):
        if name.startswith("aorta.chat"):
            raise ImportError("chat extras are not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_chat)


class TestOnABaseInstall:
    def test_a_path_is_still_scrubbed(self, without_chat):
        assert "/home/avsharma" not in redact(SECRETS)

    def test_an_address_is_still_scrubbed(self, without_chat):
        assert "149.28.124.225" not in redact(SECRETS)

    def test_the_placeholders_are_the_same_ones(self, without_chat):
        out = redact(SECRETS)
        assert "<PATH:" in out and "<IPV4:" in out

    def test_messages_too(self, without_chat):
        out = _redact_messages([{"role": "user", "content": SECRETS}])
        assert "/home/avsharma" not in out[0]["content"]

    def test_it_does_not_pass_the_text_through(self, without_chat):
        """The old behaviour, and the whole finding."""
        assert redact(SECRETS) != SECRETS


class TestTheSwitchIsStillHonoured:
    def test_redaction_off_in_chat_toml_is_respected(self, monkeypatch):
        from aorta.chat.config import settings

        monkeypatch.setattr(settings, "redact", False)
        assert redact(SECRETS) == SECRETS

    def test_an_unreadable_switch_means_redact(self, without_chat):
        """Not knowing whether it was turned off is not a reason to skip it."""
        assert redact(SECRETS) != SECRETS


class TestItRefusesRatherThanLeaking:
    def test_no_scrubber_is_an_error_not_a_passthrough(self, monkeypatch):
        real_import = builtins.__import__

        def no_scrubber(name, *args, **kwargs):
            if name == "aorta.probe.redaction":
                raise ImportError("gone")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_scrubber)
        with pytest.raises(RedactionUnavailable):
            redact(SECRETS)


def test_the_core_scrubber_needs_nothing_the_cia_extra_lacks():
    """Asserted in a fresh interpreter: the import graph is the claim."""
    probe = (
        "import sys, json, aorta.probe.redaction;"
        "bad=('pydantic','pydantic_settings','langchain','langgraph','chainlit',"
        "'chromadb','fastembed','torch');"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in bad)))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip().endswith("[]"), out.stdout


def test_the_gate_no_longer_reaches_through_the_chat_package():
    import inspect

    from aorta.cia import llm

    source = inspect.getsource(llm.redact)
    assert "aorta.probe.redaction" in source
    assert "aorta.chat.redaction" not in source.split('"""')[-1]
