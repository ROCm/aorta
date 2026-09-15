"""The two Chainlit settings that decide who can reach this server.

Chainlit writes `.chainlit/config.toml` itself on first run, with defaults
chosen for a demo: `allow_origins = ["*"]` and `mask_user_env = false`. Those
arrived in the tree as generated output nobody had read.

They are the wrong defaults for this app in particular. The tools behind this
UI submit cluster jobs, compile pasted HIP and, where the shell tool is on, run
commands -- so a wildcard origin means any page a developer happens to have
open can talk to a local instance and start work on a GPU node. The keys it
holds reach a model provider and a Slurm cluster, and rendering them as plain
text puts them one screen-share away from being somebody else's.

Asserted here because the file regenerates. Delete it and Chainlit writes the
permissive pair straight back, so the committed values are the only thing
holding them.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

_CONFIG = Path(__file__).resolve().parents[2] / ".chainlit" / "config.toml"


@pytest.fixture(scope="module")
def project() -> dict:
    if not _CONFIG.is_file():
        pytest.skip("the Chainlit config is only present in a source checkout")
    return tomllib.loads(_CONFIG.read_text(encoding="utf-8"))["project"]


class TestWhoMayTalkToIt:
    def test_the_origins_are_not_a_wildcard(self, project):
        assert "*" not in project["allow_origins"]

    def test_they_are_the_origins_it_is_served_from(self, project):
        assert "http://localhost:8010" in project["allow_origins"]

    def test_both_spellings_of_local(self, project):
        """A browser sent to 127.0.0.1 does not send localhost as its origin."""
        assert "http://127.0.0.1:8010" in project["allow_origins"]

    def test_the_list_is_not_empty(self, project):
        """Empty would be tighter and would also stop the UI connecting."""
        assert project["allow_origins"]

    @pytest.mark.parametrize("origin", ["http://evil.example", "*", "null"])
    def test_nothing_else_is_admitted(self, project, origin):
        assert origin not in project["allow_origins"]


class TestWhatItShowsOnScreen:
    def test_keys_are_masked(self, project):
        assert project["mask_user_env"] is True

    def test_and_not_persisted(self, project):
        """Unchanged from Chainlit's default, and worth keeping."""
        assert project["persist_user_env"] is False


class TestTheFileIsCommittedOnPurpose:
    """Ignoring it would hand the next deployment the defaults again."""

    def test_it_is_tracked(self):
        import subprocess

        root = _CONFIG.resolve().parents[1]
        listed = subprocess.run(
            ["git", "ls-files", ".chainlit/config.toml"],
            cwd=root, capture_output=True, text=True,
        ).stdout.strip()

        assert listed == ".chainlit/config.toml"

    def test_the_settings_are_documented(self):
        doc = (_CONFIG.resolve().parents[1] / "docs" / "chat" / "configuration.md").read_text(
            encoding="utf-8"
        )

        assert "allow_origins" in doc
        assert "mask_user_env" in doc
