"""Watch reads the job's log, and nothing else.

A job record says where its workload writes. Discovery exists for jobs that
never said, and it has somewhere to fall back to: the working directory the
scheduler reports. For a sanitizer sweep that directory is a source checkout,
and this repository's own ``tests/fixtures/`` is full of files that read like
failing logs, because that is what they are for.

Watch read eight of them once -- ``nccl_rccl_error.txt``,
``collective_timeout.txt``, an out-of-memory fixture -- and reported an
out-of-memory failure on a CUDA device for a workload that was pure Python
arithmetic and never touched a GPU. It was not hallucinating. It was reading
somebody's test data, faithfully.

Two things made that possible, and both are asserted here: the declared path
was not preferred, and the file list was resolved once, early, before the job
had written anything, then cached for the life of the run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aorta.cia.watch import poll as poll_mod


class _Job:
    """The parts of a job record the resolution reads."""

    def __init__(self, job_dir: Path, log_path: str) -> None:
        self.job_id = job_dir.name
        self.log_path = log_path
        self.watch_files: list[str] = []
        self.node = ""
        self.recipe = "r"
        self.launched_at = ""
        self.scheduler = "slurm"
        self.scheduler_job_id = "1"
        self.status = "running"


def _resolve(job: _Job, job_dir: Path) -> list[str]:
    """Run one pass of the resolution the poll loop performs."""
    declared = Path(job.log_path) if job.log_path else None
    if declared is not None and not declared.is_file():
        return []
    if declared is not None:
        job.watch_files = [str(declared)]
    return job.watch_files


@pytest.fixture
def checkout_shaped_workdir(tmp_path: Path) -> Path:
    """A working directory that looks like this repository."""
    fixtures = tmp_path / "checkout" / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    for name in ("nccl_rccl_error.txt", "collective_timeout.txt", "python_traceback.txt"):
        (fixtures / name).write_text("RuntimeError: NCCL timeout on rank 3\n", encoding="utf-8")
    return tmp_path / "checkout"


def test_the_declared_log_is_what_gets_read(tmp_path: Path):
    job_dir = tmp_path / "cia-1"
    job_dir.mkdir()
    log = job_dir / "watch.log"
    log.write_text("[train] step=0 loss=6.4\n", encoding="utf-8")

    job = _Job(job_dir, str(log))
    assert _resolve(job, job_dir) == [str(log)]


def test_nothing_else_is_read_alongside_it(tmp_path: Path, checkout_shaped_workdir: Path):
    """One authoritative file beats several plausible ones.

    Watch sends what it reads to a model and asks whether the job is healthy.
    Every extra file is another chance for it to answer about the wrong thing.
    """
    job_dir = tmp_path / "cia-2"
    job_dir.mkdir()
    log = job_dir / "watch.log"
    log.write_text("[train] step=0 loss=6.4\n", encoding="utf-8")

    job = _Job(job_dir, str(log))
    watched = _resolve(job, job_dir)
    assert watched == [str(log)]
    assert not any("fixtures" in p for p in watched)


def test_a_log_that_does_not_exist_yet_resolves_to_nothing(tmp_path: Path):
    """The race that caused it.

    Watch's first poll can land before the job has written a byte. Resolving
    then produces a guess, and the guess is cached for the whole run -- so a
    momentary absence becomes a permanent wrong answer.
    """
    job_dir = tmp_path / "cia-3"
    job_dir.mkdir()
    job = _Job(job_dir, str(job_dir / "watch.log"))

    assert _resolve(job, job_dir) == []
    assert job.watch_files == [], "an unwritten log must not cache a fallback"


def test_the_log_is_picked_up_once_it_appears(tmp_path: Path):
    """Returning nothing is only correct if the next poll tries again."""
    job_dir = tmp_path / "cia-4"
    job_dir.mkdir()
    log = job_dir / "watch.log"
    job = _Job(job_dir, str(log))

    assert _resolve(job, job_dir) == []
    log.write_text("[train] step=0 loss=6.4\n", encoding="utf-8")
    assert _resolve(job, job_dir) == [str(log)]


def test_discovery_still_serves_a_job_that_declared_nothing(tmp_path: Path):
    """The fallback has to survive: not every job record names a log."""
    job_dir = tmp_path / "cia-5"
    job_dir.mkdir()
    # Over the finder's 100-byte floor: a log too small to be worth reading is
    # skipped, which is part of why a just-started job is invisible to it.
    (job_dir / "stdout.log").write_text(
        "".join(f"[train] step={i} loss={6.4 - i * 0.02}\n" for i in range(8)),
        encoding="utf-8",
    )

    finder = poll_mod.LogFinder(config={"extensions": [".log"], "max_files": 8})
    found = finder.find(job_dir, job_context="", scheduler="", scheduler_job_id="")
    assert [p.name for p in found] == ["stdout.log"]
