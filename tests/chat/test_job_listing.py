"""The job list should contain jobs.

``jobs_root`` is not only job directories. The triage tools stage their own
work under it -- ``chat-kernels`` for a compiled kernel, ``chat-asm`` for an
assembled paste, ``staged`` for a pasted workload -- and every one of those is
touched the moment someone pastes something. Listing every directory by mtime
therefore put three staging directories at the top of the most-recent list, as
``recipe=? status=?`` entries, and pushed the jobs the user asked about off the
end of it.

What makes a directory a job is that Launch wrote a ``job.json`` in it.
"""

from __future__ import annotations

import json
import time

import pytest


@pytest.fixture()
def jobs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AORTA_CHAT_JOBS_PATH", str(tmp_path))
    from aorta.chat.config import reset_settings

    reset_settings()
    yield tmp_path
    reset_settings()


@pytest.fixture()
def listing(jobs_root):
    pytest.importorskip("dspy", reason="cluster tools need the [cia] extra")
    from aorta.chat.tools.cluster import list_cluster_jobs

    return list_cluster_jobs.func


def _job(root, name: str, **fields) -> None:
    directory = root / name
    directory.mkdir()
    record = {"job_id": name, "recipe": "consan", "status": "completed", **fields}
    (directory / "job.json").write_text(json.dumps(record), encoding="utf-8")
    time.sleep(0.01)


def _staging(root, name: str) -> None:
    (root / name).mkdir()
    time.sleep(0.01)


def _listed_names(output: str) -> list[str]:
    """The job names in a listing, ignoring the root path it prints above them.

    Matching the whole blob is what a temp directory named after the test
    defeats: pytest's own path contains "staged", so a substring check passes
    or fails on the fixture's name rather than on the listing.
    """
    return [line.strip().split()[0] for line in output.splitlines() if "recipe=" in line]


class TestStagingDirectoriesAreNotJobs:
    """They are newer than every job, so they sorted to the top of the list."""

    @pytest.mark.parametrize("name", ["chat-kernels", "chat-asm", "staged"])
    def test_one_is_not_listed(self, jobs_root, listing, name):
        _job(jobs_root, "cia-20260911-000000-aaaaaa")
        _staging(jobs_root, name)

        assert _listed_names(listing(limit=10)) == ["cia-20260911-000000-aaaaaa"]

    def test_they_do_not_take_the_places_of_real_jobs(self, jobs_root, listing):
        """The finding: a limit of 4 returned one job and three staging dirs."""
        for i in range(4):
            _job(jobs_root, f"cia-2026091{i}-000000-aaaaaa")
        for name in ("chat-kernels", "chat-asm", "staged"):
            _staging(jobs_root, name)

        names = _listed_names(listing(limit=4))

        assert len(names) == 4
        assert all(n.startswith("cia-") for n in names), names

    def test_a_root_with_only_staging_says_there_are_no_jobs(self, jobs_root, listing):
        """Rather than listing three directories as unreadable jobs."""
        for name in ("chat-kernels", "chat-asm", "staged"):
            _staging(jobs_root, name)

        assert "No jobs found" in listing()

    def test_a_stray_directory_is_ignored_too(self, jobs_root, listing):
        """The rule is what a job *is*, not a list of names to skip."""
        _job(jobs_root, "cia-20260911-000000-aaaaaa")
        _staging(jobs_root, "someone-elses-scratch")

        assert _listed_names(listing()) == ["cia-20260911-000000-aaaaaa"]

    def test_a_directory_whose_job_json_is_a_directory_is_ignored(self, jobs_root, listing):
        odd = jobs_root / "cia-odd"
        (odd / "job.json").mkdir(parents=True)

        assert _listed_names(listing()) == []


class TestRealJobsAreStillListed:
    def test_the_newest_comes_first(self, jobs_root, listing):
        _job(jobs_root, "cia-older")
        _job(jobs_root, "cia-newer")
        listed = listing()

        assert listed.index("cia-newer") < listed.index("cia-older")

    def test_the_limit_counts_jobs(self, jobs_root, listing):
        for i in range(6):
            _job(jobs_root, f"cia-{i}")
        _staging(jobs_root, "chat-asm")

        assert len(_listed_names(listing(limit=3))) == 3

    def test_the_record_is_still_read(self, jobs_root, listing):
        _job(jobs_root, "cia-20260911-000000-aaaaaa", recipe="waitcheck", status="failed")
        listed = listing()

        assert "waitcheck" in listed
        assert "failed" in listed

    def test_an_empty_root_says_so(self, jobs_root, listing):
        assert "No jobs found" in listing()
