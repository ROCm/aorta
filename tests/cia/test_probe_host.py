"""The probe is told where to connect; it does not know a site's address.

One site's head node was the default argument here, which put that address in a
public repository and made every other site's installation quietly wrong -- the
probe would try an address belonging to somebody else and time out. It also
SSHed as root regardless of the CIA_SSH_USER the launcher had established.
"""

from __future__ import annotations

import inspect

import pytest

from aorta.cia.autopsy import probe
from aorta.cia.launch.cluster import ssh_user
from aorta.cia.launch.job import JobRecord


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for var in ("CIA_SSH_HOST", "CIA_SSH_USER"):
        monkeypatch.delenv(var, raising=False)


def _job(**overrides) -> JobRecord:
    fields = {
        "job_id": "cia-test",
        "node": "node1",
        "recipe": "a recipe",
        "launched_at": "2026-01-01T00:00:00Z",
        "log_path": "/tmp/cia-test/watch.log",
        "aorta_output": "/tmp/cia-test/aorta",
    }
    return JobRecord(**{**fields, **overrides})


class TestTheHeadNode:
    def test_no_address_is_shipped_as_a_default(self):
        default = inspect.signature(probe.run_aorta_probe).parameters["head_node"].default
        assert default == ""

    def test_nothing_configured_means_no_address(self):
        assert probe.default_head_node() == ""

    def test_the_environment_supplies_it(self, monkeypatch):
        monkeypatch.setenv("CIA_SSH_HOST", "head.example.invalid")
        assert probe.default_head_node() == "head.example.invalid"

    def test_it_declines_to_probe_rather_than_guessing(self, tmp_path, capsys):
        """With nowhere to connect, saying so beats dialling somebody else."""
        job = _job()
        assert probe.run_aorta_probe(tmp_path, job) is None
        assert "CIA_SSH_HOST" in capsys.readouterr().out

    def test_the_job_can_carry_its_own(self, tmp_path, monkeypatch):
        """A job launched through a head node remembers which one."""
        seen: list[str] = []
        monkeypatch.setattr(probe, "_ssh", lambda node, cmd, **k: seen.append(node))

        job = _job(head_node="from.the.job")
        probe.run_aorta_probe(tmp_path, job)

        assert seen and seen[0] == "from.the.job"

    def test_an_explicit_argument_beats_both(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CIA_SSH_HOST", "from.the.env")
        seen: list[str] = []
        monkeypatch.setattr(probe, "_ssh", lambda node, cmd, **k: seen.append(node))

        job = _job(head_node="from.the.job")
        probe.run_aorta_probe(tmp_path, job, head_node="from.the.caller")

        assert seen and seen[0] == "from.the.caller"


class TestTheLogin:
    def test_the_probe_does_not_hardcode_root(self):
        assert "root@" not in inspect.getsource(probe)

    def test_it_uses_the_account_the_launcher_established(self, monkeypatch):
        monkeypatch.setenv("CIA_SSH_USER", "someone")
        assert ssh_user() == "someone"

    def test_root_remains_only_the_last_resort(self, monkeypatch):
        monkeypatch.delenv("USER", raising=False)
        assert ssh_user() == "root"
