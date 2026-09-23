"""Watch's file tools read this job's files, and nothing else on the host.

``read_file_tail`` and ``list_job_files`` are ReAct tools, so their ``path``
argument is chosen by the model. Unbound they read any file the agent can:
``/etc/passwd``, an SSH config, or a path the model noticed in a log.

Redaction does not cover this. It rewrites paths and addresses *in the text*,
not the contents of whatever was opened — the file comes back into the
trajectory and goes on to the provider with everything in it.

One LogWatcher serves every job in the poll loop, so the roots are bound per
assessment. Unbound means refuse: a caller that forgets gets nothing rather
than everything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aorta.cia.watch.watcher import (
    LogWatcher,
    list_job_files,
    read_file_tail,
    reading_within,
)


@pytest.fixture
def job(tmp_path):
    d = tmp_path / "cia-20260101-abc"
    (d / "sub").mkdir(parents=True)
    (d / "watch.log").write_text("step=5 loss=nan\n")
    (d / "sub" / "nested.log").write_text("nested\n")
    (tmp_path / "outside.txt").write_text("NOT_THIS_JOBS_BUSINESS\n")
    return d


class TestUnboundReadsNothing:
    def test_a_host_file(self):
        assert "refused" in read_file_tail("/etc/passwd")

    def test_even_a_real_job_file(self, job):
        """Fail closed: not knowing which job means not reading."""
        assert "refused" in read_file_tail(str(job / "watch.log"))

    def test_listing_too(self):
        assert "refused" in list_job_files("/etc")


class TestBoundToAJob:
    def test_reads_its_own_log(self, job):
        with reading_within(job):
            assert "loss=nan" in read_file_tail(str(job / "watch.log"))

    def test_reads_below_its_directory(self, job):
        with reading_within(job):
            assert "nested" in read_file_tail(str(job / "sub" / "nested.log"))

    def test_lists_its_own_directory(self, job):
        with reading_within(job):
            assert "watch.log" in list_job_files(str(job))

    @pytest.mark.parametrize(
        "target",
        ["/etc/passwd", "/etc/hostname", "~/.ssh/config"],
    )
    def test_refuses_a_host_file(self, target, job):
        with reading_within(job):
            out = read_file_tail(target)
        assert "refused" in out
        assert "root:x:0:0" not in out

    def test_refuses_a_traversal_out(self, job, tmp_path):
        with reading_within(job):
            out = read_file_tail(str(job / ".." / "outside.txt"))
        assert "refused" in out
        assert "NOT_THIS_JOBS_BUSINESS" not in out

    def test_refuses_a_symlink_out(self, job, tmp_path):
        (job / "escape.log").symlink_to(tmp_path / "outside.txt")
        with reading_within(job):
            out = read_file_tail(str(job / "escape.log"))
        assert "NOT_THIS_JOBS_BUSINESS" not in out

    def test_refuses_a_sibling_job(self, job, tmp_path):
        other = tmp_path / "cia-20260101-xyz"
        other.mkdir()
        (other / "watch.log").write_text("SOMEBODY_ELSES_RUN\n")
        with reading_within(job):
            out = read_file_tail(str(other / "watch.log"))
        assert "SOMEBODY_ELSES_RUN" not in out

    def test_the_scope_does_not_outlive_the_block(self, job):
        with reading_within(job):
            pass
        assert "refused" in read_file_tail(str(job / "watch.log"))


class TestMoreThanOneRoot:
    def test_a_log_outside_the_job_directory_can_be_allowed(self, job, tmp_path):
        """The launcher chooses where logs go; they are not always in job_dir."""
        elsewhere = tmp_path / "scratch"
        elsewhere.mkdir()
        (elsewhere / "train.log").write_text("step=1\n")

        with reading_within(job, elsewhere):
            assert "step=1" in read_file_tail(str(elsewhere / "train.log"))
            assert "loss=nan" in read_file_tail(str(job / "watch.log"))

    def test_and_still_nothing_else(self, job, tmp_path):
        elsewhere = tmp_path / "scratch"
        elsewhere.mkdir()
        with reading_within(job, elsewhere):
            assert "refused" in read_file_tail(str(tmp_path / "outside.txt"))


def test_forward_binds_what_it_is_given(monkeypatch, job):
    """The scope has to be live while the ReAct loop runs, not before it."""
    seen: list = []

    class FakeReAct:
        def forward(self, **kwargs):
            seen.append(read_file_tail(str(job / "watch.log")))
            return object()

    watcher = LogWatcher.__new__(LogWatcher)
    watcher.react = FakeReAct()

    watcher.forward("c", "ctx", "exp", allowed_roots=[job])
    assert "loss=nan" in seen[0]


def test_forward_without_roots_leaves_the_tools_shut(job):
    seen: list = []

    class FakeReAct:
        def forward(self, **kwargs):
            seen.append(read_file_tail(str(job / "watch.log")))
            return object()

    watcher = LogWatcher.__new__(LogWatcher)
    watcher.react = FakeReAct()

    watcher.forward("c", "ctx", "exp")
    assert "refused" in seen[0]
