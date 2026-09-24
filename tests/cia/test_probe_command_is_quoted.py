"""Nothing from the job record reaches a remote shell unquoted.

The escalation builds a command, sends it to the head node, and that command
is itself an ``ssh`` to the compute node. Two shells parse it, and the outer
quoting -- correct, and applied to the whole string -- is what made the inner
gap easy to miss.

``--output`` was quoted. The redirection using the same value was not::

    --output '/out;id;#'  > /out;id;#/aorta_sweep.log

so a path carrying shell characters was safe as an argument and ran as a
command one token later. ``aorta_output`` comes from the job record, which is
written from a plan the model produced, so this is model-influenced text
reaching a shell on a GPU node.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

pytest.importorskip("dspy", reason="the probe needs the [cia] extra")

import aorta.cia.autopsy.probe as probe_mod

#: Benign, and unmistakable if it ever runs.
HOSTILE = "/tmp/out;touch /tmp/pwned;#"


@pytest.fixture()
def sent(tmp_path, monkeypatch):
    """Every command the probe would send, without sending any of them."""
    commands: list[str] = []

    def fake_ssh(node, cmd, background=False):
        commands.append(cmd)

        class Done:
            returncode = 0
            stdout = "WAITING"
            stderr = ""

        return Done()

    monkeypatch.setattr(probe_mod, "_ssh", fake_ssh)
    monkeypatch.setattr(probe_mod, "default_head_node", lambda: "head")
    monkeypatch.setattr(probe_mod, "ssh_user", lambda: "me")
    monkeypatch.setattr(probe_mod, "PROBE_TIMEOUT_SEC", 0)
    monkeypatch.setattr(
        probe_mod, "check_recipe_mode", lambda _r: {"mode": "matrix"}
    )
    monkeypatch.setattr(
        probe_mod, "resolve_recipe", lambda _b, _j: ("/recipes/x.yaml", "")
    )
    return commands


def _job(output: str):
    class Job:
        node = "cv350-node-01"
        head_node = "head"
        aorta_output = output
        recipe_path = "/recipes/x.yaml"
        sidecar_path = ""
        recipe = "a label"

    return Job()


def _inner(command: str) -> str:
    """The string the compute node's shell ends up parsing."""
    # The command is 'ssh ... node <quoted>'; the last token is that quoted part.
    return shlex.split(command)[-1]


def _tokens(inner: str) -> list[str]:
    """*inner* as the remote shell would tokenise it.

    Only the trailing backgrounding ``&`` is removed. Stripping every ``&``
    also breaks ``2>&1`` into two tokens, which is a property of the test and
    not of the command.
    """
    return shlex.split(inner.rstrip().removesuffix("&"))


class TestTheRedirectionTarget:
    def test_it_is_a_single_token(self, sent, tmp_path):
        probe_mod.run_aorta_probe(tmp_path, _job(HOSTILE), head_node="head")
        launch = next(c for c in sent if "sweep run" in c)
        tokens = _tokens(_inner(launch))

        assert tokens[tokens.index(">") + 1] == f"{HOSTILE}/aorta_sweep.log"

    def test_the_payload_stays_inside_one_token(self, sent, tmp_path):
        """Unquoted, the ';' ends the redirection and starts a command."""
        probe_mod.run_aorta_probe(tmp_path, _job(HOSTILE), head_node="head")
        tokens = _tokens(_inner(next(c for c in sent if "sweep run" in c)))

        assert "touch" not in [tok.split("/")[0] for tok in tokens], tokens
        assert not any(tok == "touch" for tok in tokens), tokens

    def test_nothing_runs_after_the_redirection(self, sent, tmp_path):
        """The shape of the bug: a second command one token along."""
        probe_mod.run_aorta_probe(tmp_path, _job(HOSTILE), head_node="head")
        launch = next(c for c in sent if "sweep run" in c)
        tokens = _tokens(_inner(launch))
        after = tokens[tokens.index(">") + 2 :]

        assert after == ["2>&1"], after


class TestTheOrdinaryCaseStillWorks:
    def test_a_normal_path_is_used_as_the_log(self, sent, tmp_path):
        probe_mod.run_aorta_probe(tmp_path, _job("/jobs/cia-1/bundle/aorta"), head_node="head")
        launch = next(c for c in sent if "sweep run" in c)

        assert "/jobs/cia-1/bundle/aorta/aorta_sweep.log" in _inner(launch)

    def test_the_output_argument_is_still_passed(self, sent, tmp_path):
        probe_mod.run_aorta_probe(tmp_path, _job("/jobs/cia-1/bundle/aorta"), head_node="head")
        launch = next(c for c in sent if "sweep run" in c)
        tokens = _tokens(_inner(launch))

        assert tokens[tokens.index("--output") + 1] == "/jobs/cia-1/bundle/aorta"

    def test_the_recipe_is_still_quoted(self, sent, tmp_path):
        probe_mod.run_aorta_probe(tmp_path, _job("/jobs/cia-1/bundle/aorta"), head_node="head")
        launch = next(c for c in sent if "sweep run" in c)
        tokens = _tokens(_inner(launch))

        assert tokens[tokens.index("--recipe") + 1] == "/recipes/x.yaml"


class TestTheSourceKeepsItQuoted:
    def test_the_redirection_is_not_interpolated_bare(self):
        source = Path(probe_mod.__file__).read_text(encoding="utf-8")

        assert "> {aorta_output}/aorta_sweep.log" not in source

    def test_the_log_path_is_quoted_separately(self):
        source = Path(probe_mod.__file__).read_text(encoding="utf-8")

        assert 'sweep_log = shlex.quote(' in source
