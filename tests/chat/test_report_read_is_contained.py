"""A report read stops at the jobs root, not at the directory it starts in.

``read_autopsy_report`` validated *job_id* and then built
``job_dir / "bundle" / "report.json"`` and opened it. The last two components
were never checked, and a bundle is written by the job itself -- so a workload
dropping a symlink at ``bundle/report.json`` sent the read anywhere, and
``read_text`` followed it.

What made it quiet is the reply. It names the path it asked for, which is
inside the jobs root, while showing the contents of a file that is not.
Nothing in the answer says the two differ, and the answer is what reaches the
model and then the user.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("dspy", reason="the cluster tools need the [cia] extra")

from aorta.chat.config import reset_settings
from aorta.chat.tools.cluster import list_cluster_jobs, read_autopsy_report

SECRET = '{"secret": "HOST FILE"}'
REAL = {"category": "gpu_race", "confidence": 0.95}


@pytest.fixture()
def jobs(tmp_path, monkeypatch):
    """A jobs root with one honest job and somewhere outside to point at."""
    root = tmp_path / "jobs"
    (root / "cia-good" / "bundle").mkdir(parents=True)
    (root / "cia-good" / "bundle" / "report.json").write_text(
        json.dumps(REAL), encoding="utf-8"
    )
    (root / "cia-good" / "job.json").write_text(
        json.dumps({"recipe": "r.yaml", "status": "done"}), encoding="utf-8"
    )
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "report.json").write_text(SECRET, encoding="utf-8")

    monkeypatch.setenv("AORTA_CHAT_JOBS_PATH", str(root))
    reset_settings()
    yield tmp_path
    reset_settings()


def _plant_symlink(tmp_path: Path, job_id: str = "cia-evil") -> None:
    """A job whose bundle points its report at a file outside the root."""
    bundle = tmp_path / "jobs" / job_id / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "report.json").symlink_to(tmp_path / "outside" / "report.json")
    (tmp_path / "jobs" / job_id / "job.json").write_text(
        json.dumps({"recipe": "r.yaml", "status": "done"}), encoding="utf-8"
    )


class TestTheSingleRead:
    def test_a_symlinked_report_does_not_leak(self, jobs):
        _plant_symlink(jobs)

        assert "HOST FILE" not in read_autopsy_report.func("cia-evil")

    def test_it_says_the_path_escaped(self, jobs):
        _plant_symlink(jobs)

        assert "escapes the jobs root" in read_autopsy_report.func("cia-evil")

    def test_a_symlinked_bundle_directory_is_refused_too(self, jobs):
        """The link can be one level up and the read still leaves."""
        job = jobs / "jobs" / "cia-linkdir"
        job.mkdir(parents=True)
        (job / "bundle").symlink_to(jobs / "outside")

        assert "HOST FILE" not in read_autopsy_report.func("cia-linkdir")

    def test_an_honest_report_still_reads(self, jobs):
        out = read_autopsy_report.func("cia-good")

        assert "gpu_race" in out

    @pytest.mark.parametrize("attempt", ["../outside", "/etc", "cia-good/../.."])
    def test_a_job_id_that_walks_out_is_still_refused(self, jobs, attempt):
        """The check that already existed must not have been lost."""
        assert "Error" in read_autopsy_report.func(attempt)


class TestTheListing:
    def test_it_does_not_leak_through_a_symlinked_report(self, jobs):
        _plant_symlink(jobs)

        assert "HOST FILE" not in list_cluster_jobs.func(10)

    def test_it_names_the_problem_rather_than_going_quiet(self, jobs):
        """A listing survives one bad job, but should not look like a clean one."""
        _plant_symlink(jobs)
        line = next(
            l for l in list_cluster_jobs.func(10).splitlines() if "cia-evil" in l
        )

        assert "outside the jobs root" in line

    def test_an_honest_job_still_shows_its_verdict(self, jobs):
        _plant_symlink(jobs)
        line = next(
            l for l in list_cluster_jobs.func(10).splitlines() if "cia-good" in l
        )

        assert "gpu_race" in line

    def test_one_bad_job_does_not_break_the_listing(self, jobs):
        _plant_symlink(jobs)
        out = list_cluster_jobs.func(10)

        assert "cia-good" in out and "cia-evil" in out


class TestTheCheckIsOnTheWholePath:
    def test_the_source_validates_the_report_not_the_directory(self):
        import aorta.chat.tools.cluster as cluster

        source = Path(cluster.__file__).read_text(encoding="utf-8")

        assert 'f"{job_id}/bundle/report.json"' in source
        assert 'report = job_dir / "bundle" / "report.json"' not in source
