"""A file the model names is only watched if it belongs to the job.

``LogFinder`` shows the model a directory listing and asks which files are
logs. The answer came back as strings, and the only things between a string
and being watched were ``is_file()`` and the exclude list -- so an absolute
path the model invented was accepted whenever that file happened to exist.

Reading the wrong file is not where it stops. Watch grants the parent of every
watched path as a root for its own file tools::

    allowed_roots=[job_dir, *(Path(p).parent for p in job.watch_files)]

so one hallucinated ``/etc/passwd`` would have handed it ``/etc``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("dspy", reason="the log finder needs the [cia] extra")

from aorta.cia.watch.log_finder import LogFinder


@pytest.fixture()
def job(tmp_path):
    """A job directory, a secret outside it, and a symlink between them."""
    job_dir = tmp_path / "job"
    (job_dir / "logs").mkdir(parents=True)
    # Over the scan's 100-byte floor, or it filters them out before the
    # containment check this file is about ever runs.
    (job_dir / "train.log").write_text("step=1 loss=0.5\n" * 20, encoding="utf-8")
    (job_dir / "logs" / "nested.log").write_text("nested\n" * 40, encoding="utf-8")
    (tmp_path / "secret.log").write_text("SECRET\n" * 40, encoding="utf-8")
    (job_dir / "sneaky.log").symlink_to(tmp_path / "secret.log")
    (job_dir / "out").symlink_to(tmp_path)
    return job_dir


class TestWhatIsAccepted:
    def test_a_log_in_the_job(self, job):
        assert LogFinder(config={})._within_job("train.log", job) is not None

    def test_a_log_deeper_in_the_job(self, job):
        assert LogFinder(config={})._within_job("logs/nested.log", job) is not None

    def test_an_absolute_path_that_is_genuinely_inside(self, job):
        """The model often answers with the full path it was shown."""
        inside = str(job / "train.log")

        assert LogFinder(config={})._within_job(inside, job) is not None


class TestWhatIsRefused:
    @pytest.mark.parametrize(
        "attempt", ["/etc/passwd", "/etc/shadow", "../secret.log", "../../etc/passwd"]
    )
    def test_a_path_outside_the_job(self, job, attempt):
        assert LogFinder(config={})._within_job(attempt, job) is None

    def test_an_absolute_path_to_a_real_file_elsewhere(self, job, tmp_path):
        """is_file() was true for this, which is why is_file() was not enough."""
        assert LogFinder(config={})._within_job(str(tmp_path / "secret.log"), job) is None

    def test_a_symlink_pointing_out_of_the_job(self, job):
        """Judged by where it points, not by where it sits."""
        assert LogFinder(config={})._within_job("sneaky.log", job) is None

    def test_a_symlinked_directory_mid_path(self, job):
        assert LogFinder(config={})._within_job("out/secret.log", job) is None

    def test_an_empty_answer(self, job):
        assert LogFinder(config={})._within_job("", job) is None


class TestTheScanDoesNotLeaveEither:
    def test_a_symlinked_log_is_not_returned(self, job):
        """rglob walks into symlinked directories, so the root is not enough."""
        found = LogFinder(config={})._scan_by_extension(job)
        names = {p.name for p in found}

        assert "train.log" in names
        assert "secret.log" not in names, f"the scan left the job: {found}"

    def test_nothing_returned_is_outside_the_job(self, job):
        resolved_job = job.resolve()

        for found in LogFinder(config={})._scan_by_extension(job):
            assert found.resolve().is_relative_to(resolved_job), found


class TestWhyItMatters:
    def test_watch_grants_the_parent_as_a_tool_root(self):
        """The reason an escaped path is worse than a wrong file.

        Pinned so that if this stops being true the containment above can be
        revisited, and so that anyone widening the roots sees what depends on
        them being narrow.
        """
        import aorta.cia.watch.poll as poll_mod

        source = Path(poll_mod.__file__).read_text(encoding="utf-8")

        assert "allowed_roots=[job_dir, *(Path(p).parent for p in job.watch_files)]" in source


class TestTheCheckIsNotASecondCopy:
    def test_it_uses_the_shared_resolver(self):
        """Two containment checks is how one loses the symlink case."""
        source = Path(LogFinder.__module__.replace(".", "/") + ".py")
        text = (Path(__file__).resolve().parents[2] / "src" / source).read_text(
            encoding="utf-8"
        )

        assert "from aorta.cia.autopsy.adapters.base import resolve_in_bundle" in text
        assert "return resolve_in_bundle(job_dir, candidate)" in text


class TestTheDiscoveryPathItself:
    """Through ``find``, because the helper alone proves nothing.

    A test that calls ``_within_job`` still passes when the call site stops
    using it, which is exactly the wiring that was wrong.
    """

    @staticmethod
    def _answering(monkeypatch, finder, files: list[str]):
        """Make the model's discovery return *files*."""

        class Pred:
            relevant_files = files

        monkeypatch.setattr(finder, "_discovery", lambda **_kwargs: Pred())

    def _busy_job(self, tmp_path) -> Path:
        """A job dir the extension scan will not shortcut on.

        The scan returns early when it finds three or fewer known logs, so the
        model is only consulted where there are more than that.
        """
        job_dir = tmp_path / "job"
        job_dir.mkdir(parents=True)
        for n in range(6):
            (job_dir / f"part{n}.log").write_text("line\n" * 40, encoding="utf-8")
        (tmp_path / "secret.log").write_text("SECRET\n" * 40, encoding="utf-8")
        return job_dir

    def test_an_invented_absolute_path_is_not_watched(self, tmp_path, monkeypatch):
        job_dir = self._busy_job(tmp_path)
        finder = LogFinder(config={})
        self._answering(monkeypatch, finder, ["/etc/passwd"])

        found = finder.find(job_dir)

        assert all("passwd" not in str(p) for p in found), found

    def test_a_real_file_outside_the_job_is_not_watched(self, tmp_path, monkeypatch):
        """is_file() is true here, which is why it was not enough."""
        job_dir = self._busy_job(tmp_path)
        finder = LogFinder(config={})
        self._answering(monkeypatch, finder, [str(tmp_path / "secret.log")])

        found = finder.find(job_dir)

        assert all("secret" not in str(p) for p in found), found

    def test_a_traversal_out_is_not_watched(self, tmp_path, monkeypatch):
        job_dir = self._busy_job(tmp_path)
        finder = LogFinder(config={})
        self._answering(monkeypatch, finder, ["../secret.log"])

        found = finder.find(job_dir)

        assert all("secret" not in str(p) for p in found), found

    def test_a_genuine_log_the_model_names_is_watched(self, tmp_path, monkeypatch):
        """Containment must not mean the discovery stops working."""
        job_dir = self._busy_job(tmp_path)
        finder = LogFinder(config={})
        self._answering(monkeypatch, finder, ["part3.log"])

        found = finder.find(job_dir)

        assert any(p.name == "part3.log" for p in found), found

    def test_nothing_returned_ever_leaves_the_job(self, tmp_path, monkeypatch):
        job_dir = self._busy_job(tmp_path)
        finder = LogFinder(config={})
        self._answering(
            monkeypatch, finder, ["/etc/passwd", "../secret.log", "part1.log"]
        )
        resolved_job = job_dir.resolve()

        for found in finder.find(job_dir):
            assert found.resolve().is_relative_to(resolved_job), found
