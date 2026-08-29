"""Unit tests for the no-follow fd-relative primitives in ``aorta.run._fsafe``.

These prove the building blocks the collector and retention guards stand on:
descent refuses a symlinked component, operations stay anchored to the fd (a
mid-walk ancestor swap cannot redirect them), and no file descriptor leaks.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from aorta.run import _fsafe
from tests.conftest import BlockedTooLong

pytestmark = pytest.mark.skipif(
    not _fsafe.HAVE_FD_TRAVERSAL, reason="fd-relative traversal unsupported here"
)


def _open_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


class TestRelativeComponents:
    def test_returns_components_below_the_anchor(self, tmp_path):
        anchor = tmp_path / "results"
        target = anchor / "wl" / "trial" / "rocprof"
        assert _fsafe.relative_components(anchor, target) == ["wl", "trial", "rocprof"]

    def test_none_when_target_is_outside_the_anchor(self, tmp_path):
        anchor = tmp_path / "results"
        assert _fsafe.relative_components(anchor, tmp_path / "other") is None

    def test_none_on_a_parent_traversal_component(self, tmp_path):
        anchor = tmp_path / "results"
        # A crafted path that escapes upward is refused even if it lands back
        # inside lexically.
        target = Path(os.path.join(str(anchor), "wl", os.pardir, os.pardir, "escape"))
        assert _fsafe.relative_components(anchor, target) is None

    def test_empty_for_the_anchor_itself(self, tmp_path):
        anchor = tmp_path / "results"
        assert _fsafe.relative_components(anchor, anchor) == []


class TestOpenDirNoFollow:
    def test_opens_a_clean_tree(self, tmp_path):
        leaf = tmp_path / "a" / "b" / "c"
        leaf.mkdir(parents=True)
        with _fsafe.open_dir_nofollow(tmp_path, ["a", "b", "c"]) as fd:
            assert "marker" not in os.listdir(fd)
            (leaf / "marker").write_text("x", encoding="utf-8")
            assert "marker" in os.listdir(fd)

    def test_refuses_a_symlinked_component(self, tmp_path):
        (tmp_path / "real").mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (tmp_path / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(_fsafe.UnsafePathError):
            with _fsafe.open_dir_nofollow(tmp_path, ["link"]):
                pass

    def test_refuses_a_missing_component(self, tmp_path):
        with pytest.raises(_fsafe.UnsafePathError):
            with _fsafe.open_dir_nofollow(tmp_path, ["nope"]):
                pass

    def test_does_not_leak_fds_on_success_or_failure(self, tmp_path):
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "link").symlink_to(tmp_path / "a", target_is_directory=True)
        before = _open_fd_count()
        for _ in range(50):
            with _fsafe.open_dir_nofollow(tmp_path, ["a", "b"]):
                pass
            with pytest.raises(_fsafe.UnsafePathError):
                with _fsafe.open_dir_nofollow(tmp_path, ["link", "b"]):
                    pass
        assert _open_fd_count() == before

    def test_reports_enoent_for_a_missing_component(self, tmp_path):
        """Absence is distinguishable from a hostile component.

        The collector guard reads this errno to tell "the collector never wrote
        anything" (nothing to prune) apart from "someone swapped the path"
        (keep everything), so the two must not both surface as a bare
        ``UnsafePathError``.
        """
        with pytest.raises(_fsafe.UnsafePathError) as excinfo:
            with _fsafe.open_dir_nofollow(tmp_path, ["nope"]):
                pass
        assert excinfo.value.errno == errno.ENOENT

    def test_a_dangling_symlink_is_eloop_not_enoent(self, tmp_path):
        """A broken link must not be able to impersonate an absent directory."""
        (tmp_path / "link").symlink_to(tmp_path / "never_existed")
        with pytest.raises(_fsafe.UnsafePathError) as excinfo:
            with _fsafe.open_dir_nofollow(tmp_path, ["link"]):
                pass
        assert excinfo.value.errno != errno.ENOENT


class TestTrustedAnchorIdentity:
    """The anchor is pinned to an inode, not just to a pathname.

    ``O_NOFOLLOW`` answers "is this a symlink", which is a different question
    from "is this still the operator's directory". These cover the two swaps
    that satisfy every no-follow check on the way down.
    """

    def test_refuses_a_real_directory_renamed_into_the_anchor_pathname(self, tmp_path):
        results = tmp_path / "results"
        (results / "trial").mkdir(parents=True)
        anchor = _fsafe.TrustedAnchor.freeze(results)
        assert anchor.identity is not None

        # The payload renames the results dir aside and moves a *real*
        # directory (with its own real "trial" child) into that pathname.
        # Nothing in the new path is a symlink.
        results.rename(tmp_path / "results.moved")
        planted = tmp_path / "planted"
        (planted / "trial").mkdir(parents=True)
        (planted / "trial" / "victim.txt").write_text("keep me", encoding="utf-8")
        planted.rename(results)

        with pytest.raises(_fsafe.UnsafePathError, match="frozen before the run"):
            with _fsafe.open_dir_nofollow(anchor, ["trial"]):
                pass
        # Unpinned, the same descent succeeds -- which is the gap being closed.
        with _fsafe.open_dir_nofollow(results, ["trial"]) as fd:
            assert "victim.txt" in os.listdir(fd)

    def test_refuses_when_the_anchors_parent_is_swapped_for_a_symlink(self, tmp_path):
        """The parent is opened by pathname, so the pin is what guards it.

        The parent sits above the operator's ``--results-dir`` and may hold the
        operator's own links, so it cannot be opened ``O_NOFOLLOW``. A payload
        that can write the grandparent renames it aside and leaves a link to a
        planted tree whose own ``results`` directory is perfectly ordinary.
        """
        base = tmp_path / "run"
        results = base / "results"
        (results / "trial").mkdir(parents=True)
        anchor = _fsafe.TrustedAnchor.freeze(results)

        base.rename(tmp_path / "run.moved")
        planted = tmp_path / "planted"
        (planted / "results" / "trial").mkdir(parents=True)
        (planted / "results" / "trial" / "victim.txt").write_text("keep", encoding="utf-8")
        base.symlink_to(planted, target_is_directory=True)

        with pytest.raises(_fsafe.UnsafePathError, match="frozen before the run"):
            with _fsafe.open_dir_nofollow(anchor, ["trial"]):
                pass

    def test_accepts_the_untouched_anchor(self, tmp_path):
        results = tmp_path / "results"
        (results / "trial").mkdir(parents=True)
        (results / "trial" / "artifact.txt").write_text("x", encoding="utf-8")
        anchor = _fsafe.TrustedAnchor.freeze(results)
        with _fsafe.open_dir_nofollow(anchor, ["trial"]) as fd:
            assert "artifact.txt" in os.listdir(fd)

    def test_freeze_of_a_missing_path_yields_an_unpinned_anchor(self, tmp_path):
        """A stat failure degrades to no-follow-only rather than raising."""
        anchor = _fsafe.TrustedAnchor.freeze(tmp_path / "not_yet")
        assert anchor.identity is None

    def test_as_anchor_passes_through_and_wraps(self, tmp_path):
        pinned = _fsafe.TrustedAnchor(tmp_path, (1, 2))
        assert _fsafe.as_anchor(pinned) is pinned
        assert _fsafe.as_anchor(str(tmp_path)) == _fsafe.TrustedAnchor(tmp_path, None)

    def test_no_fd_leak_when_the_identity_check_refuses(self, tmp_path):
        results = tmp_path / "results"
        (results / "trial").mkdir(parents=True)
        anchor = _fsafe.TrustedAnchor.freeze(results)
        results.rename(tmp_path / "results.moved")
        (tmp_path / "planted" / "trial").mkdir(parents=True)
        (tmp_path / "planted").rename(results)
        before = _open_fd_count()
        for _ in range(50):
            with pytest.raises(_fsafe.UnsafePathError):
                with _fsafe.open_dir_nofollow(anchor, ["trial"]):
                    pass
        assert _open_fd_count() == before


class TestChildNameEnforcement:
    """``dir_fd`` does not by itself confine an operation to that directory.

    ``os.open`` ignores ``dir_fd`` for an absolute path, ``..`` walks above it,
    and ``O_NOFOLLOW`` guards only the *final* component -- so a name carrying a
    separator has its intermediate components resolved with symlinks followed.
    No current caller can produce such a name (``relative_components`` decomposes
    a ``relative_to`` result; everything else forwards ``os.listdir`` entries),
    so these pin the helpers' contract for the next caller.
    """

    #: Stand-in resolved per test to an absolute path *inside* the sandbox.
    #: Deliberately not a real one like ``/etc``: mutation-testing this class
    #: means running it with the guard removed, and ``remove_entry_at`` would
    #: then genuinely try to empty whatever the name points at. A test whose
    #: failure mode is "recursively deletes a system directory" is not one to
    #: leave lying around, so every case aims inside the sandbox.
    ABSOLUTE = "<absolute-inside-sandbox>"

    #: Names that must never reach the kernel from one of these helpers.
    BAD_NAMES = ("", ".", "..", ABSOLUTE, "sub/child", "a/../../b")

    @staticmethod
    def _sandbox(tmp_path):
        """A held-fd directory whose parent is also disposable.

        ``..`` has to resolve to *something* for the unguarded case to be
        meaningful, so the fd is opened one level down and the level above it is
        sacrificial. A sibling file there is the canary.
        """
        sandbox = tmp_path / "sandbox"
        held = sandbox / "held"
        (held / "sub" / "child").mkdir(parents=True)
        canary = sandbox / "canary.txt"
        canary.write_text("keep me", encoding="utf-8")
        return held, canary

    @classmethod
    def _name(cls, bad, held):
        return str(held / "sub") if bad == cls.ABSOLUTE else bad

    @pytest.mark.parametrize("bad", BAD_NAMES)
    def test_open_dir_at_refuses(self, tmp_path, bad):
        held, canary = self._sandbox(tmp_path)
        base_fd = os.open(str(held), os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(_fsafe.UnsafePathError) as excinfo:
                with _fsafe.open_dir_at(base_fd, [self._name(bad, held)]):
                    pass
            assert excinfo.value.errno == errno.EINVAL
        finally:
            os.close(base_fd)
        assert canary.exists()

    @pytest.mark.parametrize("bad", BAD_NAMES)
    def test_open_dir_nofollow_refuses(self, tmp_path, bad):
        held, _canary = self._sandbox(tmp_path)
        with pytest.raises(_fsafe.UnsafePathError) as excinfo:
            with _fsafe.open_dir_nofollow(held, [self._name(bad, held)]):
                pass
        assert excinfo.value.errno == errno.EINVAL

    @pytest.mark.parametrize("bad", BAD_NAMES)
    def test_stat_at_refuses(self, tmp_path, bad):
        held, _canary = self._sandbox(tmp_path)
        base_fd = os.open(str(held), os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(_fsafe.UnsafePathError):
                _fsafe.stat_at(base_fd, self._name(bad, held))
        finally:
            os.close(base_fd)

    @pytest.mark.parametrize("bad", BAD_NAMES)
    def test_remove_entry_at_refuses(self, tmp_path, bad):
        """The worst one: ``remove_entry_at(fd, "..")`` would empty the parent."""
        held, canary = self._sandbox(tmp_path)
        base_fd = os.open(str(held), os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(_fsafe.UnsafePathError):
                _fsafe.remove_entry_at(base_fd, self._name(bad, held))
        finally:
            os.close(base_fd)
        # Nothing above the held fd was touched. Without the guard, the ".."
        # case empties this directory.
        assert canary.read_text(encoding="utf-8") == "keep me"

    @pytest.mark.parametrize("bad", BAD_NAMES)
    def test_secure_open_read_refuses(self, tmp_path, bad):
        held, _canary = self._sandbox(tmp_path)
        base_fd = os.open(str(held), os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(_fsafe.UnsafePathError):
                with _fsafe.secure_open_read(
                    base_fd, self._name(bad, held), encoding="utf-8"
                ):
                    pass
        finally:
            os.close(base_fd)

    @pytest.mark.parametrize("bad", BAD_NAMES)
    def test_open_write_nofollow_refuses(self, tmp_path, bad):
        held, _canary = self._sandbox(tmp_path)
        base_fd = os.open(str(held), os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(_fsafe.UnsafePathError):
                stream = _fsafe.open_write_nofollow(
                    base_fd, self._name(bad, held), encoding="utf-8"
                )
                stream.close()
        finally:
            os.close(base_fd)

    def test_the_raw_calls_being_guarded_really_do_escape(self, tmp_path):
        """Without the check, each bad name reaches outside the held fd.

        Asserted directly against ``os.open`` so the guard's necessity is
        recorded here rather than in a commit message.
        """
        inner = tmp_path / "inner"
        inner.mkdir()
        (tmp_path / "outside_marker").mkdir()
        target = tmp_path / "target"
        target.mkdir()
        (inner / "linkdir").symlink_to(target, target_is_directory=True)
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
        base_fd = os.open(str(inner), os.O_RDONLY | os.O_DIRECTORY)
        try:
            # ".." leaves the held directory.
            up = os.open("..", flags, dir_fd=base_fd)
            try:
                assert "outside_marker" in os.listdir(up)
            finally:
                os.close(up)
            # An absolute path makes dir_fd irrelevant -- it names a directory
            # that is not under the held fd at all.
            absolute = os.open(str(tmp_path / "outside_marker"), flags, dir_fd=base_fd)
            os.close(absolute)
            # A separator gets the INTERMEDIATE component resolved with symlinks
            # followed -- O_NOFOLLOW only ever guarded the last one.
            os.close(os.open("linkdir/../target", flags, dir_fd=base_fd))
        finally:
            os.close(base_fd)

    def test_legitimate_names_still_work(self, tmp_path):
        """The check must not reject an ordinary directory entry."""
        (tmp_path / "sub" / "child").mkdir(parents=True)
        (tmp_path / "sub" / "child" / "f.txt").write_text("x", encoding="utf-8")
        with _fsafe.open_dir_nofollow(tmp_path, ["sub", "child"]) as fd:
            assert "f.txt" in os.listdir(fd)
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            assert _fsafe.stat_at(base_fd, "sub") is not None
        finally:
            os.close(base_fd)

    def test_no_fd_leak_when_a_name_is_refused(self, tmp_path):
        (tmp_path / "sub").mkdir()
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            before = _open_fd_count()
            for _ in range(50):
                with pytest.raises(_fsafe.UnsafePathError):
                    with _fsafe.open_dir_at(base_fd, ["sub", ".."]):
                        pass
            assert _open_fd_count() == before
        finally:
            os.close(base_fd)


class TestSecureOpenReadFileType:
    """A regular file seen by the walk can be something else by the open.

    The walk's ``lstat`` and the read are separate syscalls, and this PR widened
    that window on purpose (artifacts are now listed first and reopened one at a
    time to bound descriptor use). A payload that swaps a matched CSV for a FIFO
    turns a blocking ``O_RDONLY`` into an indefinite wait, which hangs the sweep
    rather than degrading it -- strictly worse than any wrong metric.
    """

    def test_refuses_a_fifo_instead_of_blocking(self, tmp_path, no_hang):
        os.mkfifo(tmp_path / "stats.csv")
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            # The defect this replaces was a hang, so bound the test itself: a
            # regression must fail the alarm rather than stall the suite.
            with no_hang(5):
                with pytest.raises(_fsafe.UnsafePathError, match="not a regular file"):
                    with _fsafe.secure_open_read(base_fd, "stats.csv", encoding="utf-8"):
                        pass
        finally:
            os.close(base_fd)

    def test_a_blocking_open_on_that_fifo_really_does_hang(self, tmp_path, no_hang):
        """Pins the premise: without ``O_NONBLOCK`` the open never returns."""
        os.mkfifo(tmp_path / "stats.csv")
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(BlockedTooLong):
                with no_hang(2):
                    os.close(
                        os.open("stats.csv", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=base_fd)
                    )
        finally:
            os.close(base_fd)

    def test_refuses_a_directory_where_a_file_was_expected(self, tmp_path):
        (tmp_path / "stats.csv").mkdir()
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(OSError):
                with _fsafe.secure_open_read(base_fd, "stats.csv", encoding="utf-8"):
                    pass
        finally:
            os.close(base_fd)

    def test_reads_a_regular_file_normally(self, tmp_path):
        (tmp_path / "stats.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            with _fsafe.secure_open_read(
                base_fd, "stats.csv", encoding="utf-8", newline=""
            ) as stream:
                assert stream.read() == "a,b\n1,2\n"
        finally:
            os.close(base_fd)

    def test_the_handle_is_blocking_again(self, tmp_path):
        """``O_NONBLOCK`` is an open-time trick, not part of the parser's contract."""
        # Imported here rather than at module scope: this module is skipped
        # wholesale off POSIX, and a top-level import would turn that skip into
        # a collection error.
        import fcntl

        (tmp_path / "stats.csv").write_text("x", encoding="utf-8")
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            with _fsafe.secure_open_read(base_fd, "stats.csv", encoding="utf-8") as stream:
                flags = fcntl.fcntl(stream.fileno(), fcntl.F_GETFL)
                assert not flags & os.O_NONBLOCK
        finally:
            os.close(base_fd)

    def test_no_fd_leak_when_the_type_check_refuses(self, tmp_path, no_hang):
        os.mkfifo(tmp_path / "stats.csv")
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            before = _open_fd_count()
            for _ in range(50):
                with no_hang(5), pytest.raises(_fsafe.UnsafePathError):
                    with _fsafe.secure_open_read(base_fd, "stats.csv", encoding="utf-8"):
                        pass
            assert _open_fd_count() == before
        finally:
            os.close(base_fd)

    def test_no_fd_leak_when_wrapping_the_fd_fails(self, tmp_path, monkeypatch):
        """``fdopen`` only adopts the fd on success, so a failure must close it."""
        (tmp_path / "stats.csv").write_text("x", encoding="utf-8")

        def boom(*_args, **_kwargs):
            raise LookupError("unknown encoding")

        monkeypatch.setattr(os, "fdopen", boom)
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            before = _open_fd_count()
            for _ in range(50):
                with pytest.raises(LookupError):
                    with _fsafe.secure_open_read(base_fd, "stats.csv", encoding="bogus"):
                        pass
            assert _open_fd_count() == before
        finally:
            os.close(base_fd)


class TestOpenWriteNoFollow:
    """Writer validation must happen before any destructive truncation."""

    def test_refuses_hard_link_without_truncating_target(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("keep me", encoding="utf-8")
        os.link(outside, output_dir / "result.json")

        base_fd = os.open(str(output_dir), os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(_fsafe.UnsafePathError, match="hard links"):
                stream = _fsafe.open_write_nofollow(
                    base_fd, "result.json", encoding="utf-8"
                )
                # Closes the stream if the link-count guard is mutated away.
                stream.close()
        finally:
            os.close(base_fd)

        # Adding O_TRUNC back to os.open makes the refusal too late: the helper
        # still raises, but this canary has already been emptied.
        assert outside.read_text(encoding="utf-8") == "keep me"

    def test_overwrites_singly_linked_regular_file(self, tmp_path):
        target = tmp_path / "result.json"
        target.write_text("old content with a longer tail", encoding="utf-8")
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            with _fsafe.open_write_nofollow(
                base_fd, "result.json", encoding="utf-8"
            ) as stream:
                stream.write("new")
        finally:
            os.close(base_fd)
        assert target.read_text(encoding="utf-8") == "new"

    def test_no_fd_leak_when_hard_link_is_refused(self, tmp_path):
        outside = tmp_path / "outside.txt"
        outside.write_text("keep me", encoding="utf-8")
        os.link(outside, tmp_path / "result.json")
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            before = _open_fd_count()
            for _ in range(50):
                with pytest.raises(_fsafe.UnsafePathError):
                    _fsafe.open_write_nofollow(
                        base_fd, "result.json", encoding="utf-8"
                    )
            assert _open_fd_count() == before
        finally:
            os.close(base_fd)


class TestOpenDirAt:
    def test_descends_from_a_held_fd(self, tmp_path):
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "a" / "b" / "f.txt").write_text("x", encoding="utf-8")
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            with _fsafe.open_dir_at(base_fd, ["a", "b"]) as fd:
                assert "f.txt" in os.listdir(fd)
        finally:
            os.close(base_fd)

    def test_empty_components_yields_the_base_directory(self, tmp_path):
        (tmp_path / "f.txt").write_text("x", encoding="utf-8")
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            with _fsafe.open_dir_at(base_fd, []) as fd:
                assert "f.txt" in os.listdir(fd)
            # The base fd must survive: ``open_dir_at`` dups rather than
            # adopting it, so the caller's own descent stays usable.
            assert "f.txt" in os.listdir(base_fd)
        finally:
            os.close(base_fd)

    def test_refuses_a_symlinked_component(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (tmp_path / "link").symlink_to(outside, target_is_directory=True)
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(_fsafe.UnsafePathError):
                with _fsafe.open_dir_at(base_fd, ["link"]):
                    pass
        finally:
            os.close(base_fd)

    def test_no_fd_leak(self, tmp_path):
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "link").symlink_to(tmp_path / "a", target_is_directory=True)
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            before = _open_fd_count()
            for _ in range(50):
                with _fsafe.open_dir_at(base_fd, ["a", "b"]):
                    pass
                with pytest.raises(_fsafe.UnsafePathError):
                    with _fsafe.open_dir_at(base_fd, ["link", "b"]):
                        pass
            assert _open_fd_count() == before
        finally:
            os.close(base_fd)


class TestRemoveEntryAt:
    def test_removes_a_nested_tree(self, tmp_path):
        root = tmp_path / "tree"
        (root / "sub").mkdir(parents=True)
        (root / "top.txt").write_text("aaaa", encoding="utf-8")
        (root / "sub" / "deep.txt").write_text("bb", encoding="utf-8")
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            files, freed = _fsafe.remove_entry_at(base_fd, "tree")
        finally:
            os.close(base_fd)
        assert files == 2
        assert freed == 6
        assert not root.exists()

    def test_unlinks_a_symlink_without_following_it(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        victim = outside / "precious.txt"
        victim.write_text("keep me", encoding="utf-8")
        root = tmp_path / "tree"
        root.mkdir()
        (root / "escape").symlink_to(outside, target_is_directory=True)
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            _fsafe.remove_entry_at(base_fd, "tree")
        finally:
            os.close(base_fd)
        # The link was unlinked but its target tree survived untouched.
        assert not root.exists()
        assert victim.read_text(encoding="utf-8") == "keep me"

    def test_no_fd_leak(self, tmp_path):
        base_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            before = _open_fd_count()
            for i in range(30):
                d = tmp_path / f"tree{i}" / "sub"
                d.mkdir(parents=True)
                (d / "f.txt").write_text("x", encoding="utf-8")
                _fsafe.remove_entry_at(base_fd, f"tree{i}")
            assert _open_fd_count() == before
        finally:
            os.close(base_fd)


class TestIterRegularFiles:
    def test_yields_only_regular_files_and_skips_symlinks(self, tmp_path):
        root = tmp_path / "tree"
        (root / "sub").mkdir(parents=True)
        (root / "a.txt").write_text("a", encoding="utf-8")
        (root / "sub" / "b.txt").write_text("bb", encoding="utf-8")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.txt").write_text("leak", encoding="utf-8")
        (root / "sneak").symlink_to(outside / "leak.txt")
        (root / "sneakdir").symlink_to(outside, target_is_directory=True)

        base_fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)
        seen = {}
        try:
            for rel, _dir_fd, name, size in _fsafe.iter_regular_files(base_fd):
                seen[rel] = (name, size)
        finally:
            os.close(base_fd)
        assert set(seen) == {"a.txt", "sub/b.txt"}
        assert seen["sub/b.txt"] == ("b.txt", 2)


class TestPruneEmptyDirs:
    def test_removes_empty_subdirs_bottom_up_but_keeps_base(self, tmp_path):
        root = tmp_path / "tree"
        (root / "empty" / "deeper").mkdir(parents=True)
        (root / "kept").mkdir()
        (root / "kept" / "f.txt").write_text("x", encoding="utf-8")
        base_fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)
        try:
            _fsafe.prune_empty_dirs(base_fd)
        finally:
            os.close(base_fd)
        assert root.is_dir()
        assert not (root / "empty").exists()
        assert (root / "kept" / "f.txt").exists()
