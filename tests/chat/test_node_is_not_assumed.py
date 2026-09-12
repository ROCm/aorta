"""The evidence summary should not name hardware nobody checked.

"submitted slurm job N to an MI355X node" was a constant. The node is often
not requested at all -- ``cia_demo_node`` is empty by default and the scheduler
chooses -- and where it is requested it can name anything the cluster has. So
a summary whose whole purpose is to be attributable asserted a GPU model on
every run, correct only on the one cluster it was written against.

It reports what Slurm says now, and where Slurm will not say, it says that
rather than picking something.
"""

from __future__ import annotations

import subprocess

import pytest

from aorta.cia.triage import sacct_nodelist


@pytest.fixture()
def cluster():
    pytest.importorskip("dspy", reason="cluster tools need the [cia] extra")
    import aorta.chat.tools.cluster as module

    return module


def _launch_line(cluster, result: dict) -> str:
    return next(line for line in cluster._fmt_tools_used(result) if "Launch" in line)


class TestTheSummaryNamesWhatRanIt:
    def test_the_resolved_node_is_reported(self, cluster):
        line = _launch_line(cluster, {"slurm_job_id": "12345", "node": "node07"})

        assert "node07" in line

    def test_no_gpu_model_is_claimed(self, cluster):
        """The constant was right on one cluster and wrong everywhere else."""
        line = _launch_line(cluster, {"slurm_job_id": "12345", "node": "node07"})

        assert "MI355X" not in line

    @pytest.mark.parametrize(
        "result",
        [
            {"slurm_job_id": "1"},
            {"slurm_job_id": "1", "node": ""},
            {"slurm_job_id": "1", "node": None},
        ],
        ids=["absent", "empty", "none"],
    )
    def test_an_unknown_node_is_said_to_be_unknown(self, cluster, result):
        line = _launch_line(cluster, result)

        assert "scheduler chose" in line
        assert "MI355X" not in line

    def test_the_job_id_is_still_there(self, cluster):
        """It is the handle for everything else about the run."""
        assert "12345" in _launch_line(cluster, {"slurm_job_id": "12345", "node": "n1"})

    def test_nothing_in_the_package_names_a_gpu_model(self, cluster):
        """Including the redirect text, which is a claim made to the model."""
        from pathlib import Path

        source = Path(cluster.__file__).read_text(encoding="utf-8")
        claims = [
            line
            for line in source.splitlines()
            if "MI355X" in line and not line.lstrip().startswith("#")
        ]

        assert claims == []


class TestAskingSlurmWhichNode:
    @staticmethod
    def _sacct(monkeypatch, stdout: str, *, fail: bool = False):
        def fake_run(argv, **kwargs):
            if fail:
                raise FileNotFoundError(2, "No such file or directory", "sacct")
            return subprocess.CompletedProcess(argv, 0, stdout, "")

        monkeypatch.setattr(subprocess, "run", fake_run)

    def test_the_node_is_read_from_the_accounting_record(self, monkeypatch):
        self._sacct(monkeypatch, "node07\n")

        assert sacct_nodelist("12345") == "node07"

    def test_a_queued_job_has_no_node_yet(self, monkeypatch):
        """Slurm writes "None assigned" before it places the job."""
        self._sacct(monkeypatch, "None assigned\n")

        assert sacct_nodelist("12345") == ""

    def test_no_slurm_is_not_an_error(self, monkeypatch):
        """The summary degrades to "the scheduler chose"; it does not fail."""
        self._sacct(monkeypatch, "", fail=True)

        assert sacct_nodelist("12345") == ""

    def test_empty_output_yields_nothing(self, monkeypatch):
        self._sacct(monkeypatch, "\n\n")

        assert sacct_nodelist("12345") == ""

    def test_a_node_range_is_passed_through_as_written(self, monkeypatch):
        """Slurm's own spelling for several nodes; inventing one would be worse."""
        self._sacct(monkeypatch, "node[07-09]\n")

        assert sacct_nodelist("12345") == "node[07-09]"

    def test_it_asks_for_the_node_list(self, monkeypatch):
        seen: dict = {}

        def fake_run(argv, **kwargs):
            seen["argv"] = argv
            return subprocess.CompletedProcess(argv, 0, "node07\n", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        sacct_nodelist("12345")

        assert "--format=NodeList" in seen["argv"]
        assert "12345" in seen["argv"]
