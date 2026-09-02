"""What the web UI tells a user about its own boundaries.

This surface shipped three claims the code did not keep: the welcome message
advertised command execution as "sandboxed" and always available, the landing
page said commands run in a sandbox, and neither mentioned that outbound
redaction covers paths and IP addresses only. All three are user-facing
security claims, and the web user is the one person who cannot check them
against the source.

Deliberately importing ``aorta.chat.ui.welcome`` rather than
``aorta.chat.ui.app``: the latter imports ``chainlit``, which the ``chat-cli``
install these tests run under does not have. The text is what carries the
claim, so the text is what is pinned -- everywhere, not only where Chainlit is
installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aorta.chat.config import configure, reset_settings
from aorta.chat.ui import welcome

CHAINLIT_MD = Path(welcome.__file__).parent / "chainlit.md"

#: Words that assert a boundary. Any of them in the UI surface has to be
#: backed by something that enforces it, or reworded. "sandbox" is the one this
#: surface actually got wrong; the rest are here so the next overstatement is
#: caught by a failing test rather than by a reviewer.
GUARANTEE_WORDS = ("sandbox", "sandboxed", "guaranteed", "isolated", "secure")


@pytest.fixture(autouse=True)
def _default_settings():
    reset_settings()
    yield
    reset_settings()


class TestCommandExecutionIsDescribedHonestly:
    def test_the_default_says_execution_is_disabled(self):
        """``enable_shell_tool`` defaults to false (Decision 17a).

        The old message said the assistant could "run commands in a sandbox",
        which was wrong twice over: there is no sandbox, and on a default
        install there is no shell tool either.
        """
        text = welcome.capabilities()
        assert "disabled" in text
        assert "enable_shell_tool" in text

    def test_enabling_it_states_plainly_that_it_is_not_a_sandbox(self):
        configure(enable_shell_tool=True)
        text = welcome.capabilities()
        assert "not a sandbox" in text.lower()
        assert "anything the account running this server can" in text

    def test_enabling_it_names_the_allowlist(self):
        configure(enable_shell_tool=True, allowed_commands=["pytest", "ls"])
        text = welcome.capabilities()
        assert "pytest" in text
        assert "ls" in text

    def test_the_contained_tools_are_the_ones_described_as_contained(self):
        """read/list/search are confined by ``resolve_within``; say so, only so."""
        text = welcome.capabilities()
        assert "confined" in text


class TestRedactionScopeIsDisclosed:
    def test_it_names_paths_and_addresses(self):
        text = welcome.redaction_status()
        assert "filesystem paths" in text
        assert "IP addresses" in text

    def test_it_says_what_is_not_covered(self):
        """Decision 16's scope is paths and IPs; #420 tracks the rest.

        Naming only what *is* removed reads as a general-purpose scrubber,
        which is the misreading that gets a customer's key pasted in.
        """
        text = welcome.redaction_status()
        assert "credentials" in text
        assert "not" in text

    def test_turning_it_off_is_disclosed_too(self):
        configure(redact=False)
        text = welcome.redaction_status()
        assert "off" in text.lower()
        assert "as they are" in text


class TestWelcomeMessage:
    def test_it_carries_the_backend_description(self):
        assert "some-backend" in welcome.welcome_message("some-backend")

    def test_it_makes_no_unbacked_guarantee(self):
        configure(enable_shell_tool=True)
        text = welcome.welcome_message("backend").lower()
        for word in GUARANTEE_WORDS:
            if word not in text:
                continue
            # The only permitted use is the denial.
            assert "not a sandbox" in text, f"unbacked {word!r} claim in the welcome message"

    def test_the_default_message_claims_no_boundary_at_all(self):
        text = welcome.welcome_message("backend").lower()
        assert not any(word in text for word in GUARANTEE_WORDS), text


class TestChainlitLandingPage:
    """The landing page is read before anyone types, so it sets expectations."""

    def test_it_does_not_promise_a_sandbox(self):
        body = CHAINLIT_MD.read_text(encoding="utf-8").lower()
        for line in body.splitlines():
            if "sandbox" not in line:
                continue
            assert "not a sandbox" in line, f"unbacked sandbox claim: {line!r}"

    def test_it_says_command_execution_is_off_by_default(self):
        body = CHAINLIT_MD.read_text(encoding="utf-8")
        assert "off by default" in body.lower()
        assert "enable_shell_tool" in body

    def test_the_example_row_does_not_promise_a_sandbox_either(self):
        """The suppressed half of the same review comment (line 19).

        The capability table repeated the claim the bullet list made, which is
        how one wrong sentence becomes two.
        """
        body = CHAINLIT_MD.read_text(encoding="utf-8")
        rows = [line for line in body.splitlines() if "Run pytest" in line]
        assert rows, "the example table no longer has a command-execution row"
        for row in rows:
            assert "sandbox" not in row.lower()
            assert "only if the operator enabled" in row

    def test_it_is_shipped_in_the_wheel(self):
        """It is package data; a test that reads it must not be the only reader."""
        import tomllib

        pyproject = Path(welcome.__file__).parents[4] / "pyproject.toml"
        if not pyproject.is_file():  # installed wheel, not a source checkout
            pytest.skip("pyproject.toml is not present in an installed layout")
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        package_data = data["tool"]["setuptools"]["package-data"]
        assert "chat/ui/chainlit.md" in package_data.get("aorta", [])
