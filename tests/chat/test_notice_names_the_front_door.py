"""The disclosure has to name a way out that exists where it is shown.

The notice told every user to "Disable with --no-redact". That flag is on
``aorta chat`` and ``aorta chat ask``; it is not on ``aorta chat ui``, which
has no command line to put it on. So the one line a browser user ever sees
about the gate recommended something they could not do, and the half of it
that worked -- the config file -- was easy to miss beside it.

The notice itself stays. A gate that rewrites what is sent and says nothing
teaches people to distrust the tool when an answer looks wrong, and the
server's log is not somewhere a browser user can read.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from aorta.chat import redaction
from aorta.chat.redaction import (
    CLI_OPT_OUT,
    UI_OPT_OUT,
    NoticeState,
    notice_line,
    redact_for_send,
    take_pending_notice,
    use_notice_state,
)

_WITH_A_PATH = "look at /apps/avsharma/jobs/cia-2026/bundle/report.json"


def shown(state: NoticeState) -> str:
    """The notice a front door holding *state* would render."""
    with use_notice_state(state):
        redact_for_send([HumanMessage(content=_WITH_A_PATH)])
        return take_pending_notice(state) or ""


class TestTheBrowserIsNotToldToPassAFlag:
    def test_it_does_not_mention_the_flag(self):
        assert "--no-redact" not in shown(NoticeState(opt_out=UI_OPT_OUT))

    def test_it_still_says_how_to_turn_it_off(self):
        """Naming the removal without naming the way out leaves people stuck."""
        assert "redact = false" in shown(NoticeState(opt_out=UI_OPT_OUT))

    def test_it_still_says_what_was_removed(self):
        assert "filesystem path" in shown(NoticeState(opt_out=UI_OPT_OUT))


class TestTheCliKeepsItsFlag:
    def test_the_flag_is_still_offered(self):
        """It exists there, and it is the quicker of the two."""
        assert "--no-redact" in shown(NoticeState(opt_out=CLI_OPT_OUT))

    def test_the_config_file_is_offered_too(self):
        assert "redact = false" in shown(NoticeState(opt_out=CLI_OPT_OUT))

    def test_that_is_the_default_for_a_state_nobody_configured(self):
        """aorta chat and aorta chat ask never construct one of these."""
        assert NoticeState().opt_out == CLI_OPT_OUT


class TestTheUiActuallyUsesIt:
    def test_every_state_the_ui_builds_carries_the_browser_wording(self):
        pytest.importorskip("chainlit", reason="requires the chat-ui extra")
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "src" / "aorta" / "chat" / "ui" / "app.py"
        ).read_text(encoding="utf-8")

        assert "redaction.NoticeState()" not in source, (
            "a NoticeState built without opt_out gets the CLI's flag"
        )
        assert source.count("opt_out=redaction.UI_OPT_OUT") == 2, (
            "both the startup state and the reconnect fallback need it"
        )


class TestItIsStillOncePerSession:
    def test_the_second_turn_is_silent(self):
        state = NoticeState(opt_out=UI_OPT_OUT)

        assert shown(state), "the first redaction should disclose"
        assert not shown(state), "the notice repeated on a later turn"

    def test_a_session_with_no_paths_is_never_told(self):
        """Otherwise it reports a redaction that did not happen."""
        state = NoticeState(opt_out=UI_OPT_OUT)
        with use_notice_state(state):
            redact_for_send([HumanMessage(content="what is aorta?")])

            assert take_pending_notice(state) is None


class TestTheLineItself:
    def test_an_explicit_hint_is_used_verbatim(self):
        line = notice_line(redaction.RedactionSummary(paths=1), UI_OPT_OUT)

        assert line.endswith(UI_OPT_OUT)

    def test_no_hint_falls_back_to_the_cli_form(self):
        """Callers that predate the argument keep what they had."""
        assert notice_line(redaction.RedactionSummary(paths=1)).endswith(CLI_OPT_OUT)
