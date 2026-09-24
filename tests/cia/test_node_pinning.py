"""A node that was asked for and not used says so.

``build_sbatch_script`` dropped ``--nodelist`` whenever ``node_exists`` was
False, and ``node_exists`` was False for three unrelated reasons: the node is
unknown, Slurm is unreachable, or scontrol timed out. The directive vanished
either way, with nothing said, and the job ran wherever the scheduler put it.

For a repro pinned to one machine that means the verdict is about a different
machine -- and the report reads exactly the same as one that ran where it was
asked to.
"""

from __future__ import annotations

import logging
import subprocess

import pytest

from aorta.cia.launch import cluster
from aorta.cia.launch.cluster import build_sbatch_script, node_placement


def _script(node: str = "", **kw) -> str:
    return build_sbatch_script(
        command="echo hi", job_name="j", log_path="/tmp/l", node=node, **kw
    )


class TestWhyNotJustNo:
    def test_an_unknown_node_says_slurm_does_not_know_it(self, monkeypatch):
        monkeypatch.setattr(cluster, "slurm_available", lambda: True)
        monkeypatch.setattr(
            cluster.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1)
        )
        can_pin, why = node_placement("n1")
        assert not can_pin
        assert "does not know" in why

    def test_an_unreachable_slurm_is_not_blamed_on_the_node(self, monkeypatch):
        """This one is not about the node at all."""
        monkeypatch.setattr(cluster, "slurm_available", lambda: False)
        can_pin, why = node_placement("n1")
        assert not can_pin
        assert "not reachable" in why

    def test_a_timeout_says_it_could_not_be_asked(self, monkeypatch):
        monkeypatch.setattr(cluster, "slurm_available", lambda: True)

        def boom(*a, **k):
            raise subprocess.TimeoutExpired("scontrol", 15)

        monkeypatch.setattr(cluster.subprocess, "run", boom)
        can_pin, why = node_placement("n1")
        assert not can_pin
        assert "could not be asked" in why

    def test_a_known_node_pins_with_nothing_to_report(self, monkeypatch):
        monkeypatch.setattr(cluster, "slurm_available", lambda: True)
        monkeypatch.setattr(
            cluster.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0)
        )
        assert node_placement("n1") == (True, "")

    def test_no_node_requested_is_not_a_failure(self):
        assert node_placement("") == (False, "")


class TestTheSubstitutionIsVisible:
    @pytest.fixture(autouse=True)
    def _unknown_node(self, monkeypatch):
        monkeypatch.setattr(cluster, "slurm_available", lambda: True)
        monkeypatch.setattr(
            cluster.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1)
        )

    def test_the_operator_is_warned(self, caplog):
        with caplog.at_level(logging.WARNING, logger="aorta.cia.launch.cluster"):
            _script(node="n1")
        assert caplog.records, "dropping a requested node must not be silent"
        assert "n1" in caplog.records[0].getMessage()

    def test_the_job_log_records_it(self):
        """A warning is gone by the time the bundle is read; the log is not."""
        body = _script(node="n1")
        assert "not pinned" in body
        assert "n1" in body

    def test_the_reason_travels_with_it(self):
        assert "does not know" in _script(node="n1")

    def test_the_pin_really_is_dropped(self):
        assert "--nodelist" not in _script(node="n1")


class TestTheQuietPath:
    @pytest.fixture(autouse=True)
    def _known_node(self, monkeypatch):
        monkeypatch.setattr(cluster, "slurm_available", lambda: True)
        monkeypatch.setattr(
            cluster.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0)
        )

    def test_a_pinned_node_says_nothing(self, caplog):
        with caplog.at_level(logging.WARNING, logger="aorta.cia.launch.cluster"):
            script = _script(node="n1")
        assert not caplog.records
        assert "#SBATCH --nodelist=n1" in script
        assert "not pinned" not in script

    def test_asking_for_no_node_says_nothing(self, caplog):
        with caplog.at_level(logging.WARNING, logger="aorta.cia.launch.cluster"):
            script = _script()
        assert not caplog.records
        assert "not pinned" not in script


def test_a_node_name_with_a_quote_cannot_break_the_script(monkeypatch):
    """The message is interpolated into a shell line."""
    monkeypatch.setattr(cluster, "slurm_available", lambda: False)
    script = _script(node="n1'; rm -rf /; echo '")
    echoes = [line for line in script.splitlines() if "not pinned" in line]
    assert echoes and echoes[0].startswith("echo '")
    assert "rm -rf /; echo" not in script.replace(echoes[0], "")
