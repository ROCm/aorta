"""Tests for ``aorta.ebpf.runner.BpftraceRunner``.

Covers the C1-C4 fixes added in this PR:
  - C1: ``startup_timeout_sec`` is enforced (previously documented but no-op).
  - C2: stderr is drained on its own thread so the OS pipe buffer cannot fill.
  - C3: reader-thread death is surfaced via ``is_running`` and logged from
        ``stop()`` instead of being silently swallowed.
  - C4: immediate-exit failures (binary missing on PATH, sudo prompt
        swallowed, "ERROR: Don't have permission to run BPF program")
        raise from ``start()`` with the captured stderr included.

Subprocesses are simulated via a ``FakePopen`` so the tests do not
require an actual ``bpftrace`` binary or any kernel privileges.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from aorta.ebpf import (
    SCRIPTS_DIR,
    BpftraceConfig,
    BpftraceRunner,
    BpftraceScriptVariant,
    BpftraceUnavailableError,
)

# ---------------------------------------------------------------------------
# FakePopen: a controllable Popen replacement.
# ---------------------------------------------------------------------------


class FakePopen:
    """In-memory stand-in for ``subprocess.Popen`` for runner tests.

    Behaviour knobs (set on the class via the factory below):
      - ``script_lines``: lines to feed on stdout, one per ``__iter__``
        step. Use ``None`` as a sentinel to block the reader thread
        until ``feed_stdout`` is called.
      - ``stderr_lines``: same shape, for stderr.
      - ``startup_rc``: if not None, ``poll()`` returns this immediately.
    """

    def __init__(
        self,
        cmd,
        *,
        stdout=None,
        stderr=None,
        text=None,
        bufsize=None,
        script_lines=(),
        stderr_lines=(),
        startup_rc=None,
        terminate_raises=None,
        kill_raises=None,
        wait_raises=None,
    ):
        self._stdout_q: list[str] = list(script_lines)
        self._stderr_q: list[str] = list(stderr_lines)
        self._stdout_event = threading.Event()
        self._stderr_event = threading.Event()
        # If the fake process is already "exited" (startup_rc is set), the
        # OS would have closed the pipes and the iterator would terminate
        # after consuming the queue. Mirror that here so drain threads
        # can exit during the start()-failure path.
        already_exited = startup_rc is not None
        self._stdout_done = already_exited
        self._stderr_done = already_exited
        self._returncode = startup_rc
        self.pid = 12345
        self.cmd = cmd
        # PR #162 round 2 (C3): exercise the ProcessLookupError race that
        # the real ``Popen`` exposes when the child has already been
        # reaped between ``poll()`` and ``terminate()`` / ``kill()``.
        self._terminate_raises = terminate_raises
        self._kill_raises = kill_raises
        self._wait_raises = wait_raises

        self.stdout = _BlockingLines(self._stdout_q, self._stdout_event, lambda: self._stdout_done)
        self.stderr = _BlockingLines(self._stderr_q, self._stderr_event, lambda: self._stderr_done)

    def poll(self):
        return self._returncode

    def terminate(self):
        if self._terminate_raises is not None:
            raise self._terminate_raises
        self._stdout_done = True
        self._stderr_done = True
        self._stdout_event.set()
        self._stderr_event.set()
        if self._returncode is None:
            self._returncode = 0

    def kill(self):
        if self._kill_raises is not None:
            raise self._kill_raises
        self._stdout_done = True
        self._stderr_done = True
        self._stdout_event.set()
        self._stderr_event.set()
        if self._returncode is None:
            self._returncode = 0

    def wait(self, timeout=None):
        if self._wait_raises is not None:
            raise self._wait_raises
        return self._returncode if self._returncode is not None else 0

    def feed_stdout(self, line: str) -> None:
        self._stdout_q.append(line)
        self._stdout_event.set()

    def finish_stdout(self) -> None:
        self._stdout_done = True
        self._stdout_event.set()


class _BlockingLines:
    """Iterable that yields queued lines, blocking when empty until signalled."""

    def __init__(self, queue, event, done_fn):
        self._q = queue
        self._event = event
        self._done = done_fn

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._q:
                return self._q.pop(0)
            if self._done():
                raise StopIteration
            self._event.wait(timeout=0.05)
            self._event.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(**cfg_kwargs) -> BpftraceRunner:
    defaults: dict = {
        "target_pid": 1234,
        "variant": BpftraceScriptVariant.TP_ONLY,
        "use_sudo": False,
        "bpftrace_path": "/usr/bin/bpftrace",
        "startup_timeout_sec": 0.5,
    }
    defaults.update(cfg_kwargs)
    return BpftraceRunner(BpftraceConfig(**defaults))


def _patch_popen(fake_factory):
    return patch("aorta.ebpf.runner.subprocess.Popen", side_effect=fake_factory)


# ---------------------------------------------------------------------------
# Build-command + availability
# ---------------------------------------------------------------------------


class TestCommandBuilding:
    def test_is_bpftrace_available_path_present(self, tmp_path: Path):
        binary = tmp_path / "fake_bpftrace"
        binary.touch()
        assert BpftraceRunner.is_bpftrace_available(str(binary)) is True

    def test_is_bpftrace_available_path_missing(self, tmp_path: Path):
        assert BpftraceRunner.is_bpftrace_available(str(tmp_path / "nope")) is False

    def test_unavailable_error_when_bpftrace_missing(self, tmp_path: Path):
        cfg = BpftraceConfig(
            target_pid=1,
            variant=BpftraceScriptVariant.TP_ONLY,
            use_sudo=False,
            bpftrace_path=None,
        )
        runner = BpftraceRunner(cfg)
        with patch("aorta.ebpf.runner.shutil.which", return_value=None):
            with pytest.raises(BpftraceUnavailableError):
                runner._build_command()

    def test_script_path_must_exist(self, tmp_path: Path):
        cfg = BpftraceConfig(
            target_pid=1,
            use_sudo=False,
            bpftrace_path="/usr/bin/bpftrace",
            script_path=tmp_path / "missing.bt",
        )
        runner = BpftraceRunner(cfg)
        with pytest.raises(FileNotFoundError):
            runner._build_command()

    def test_sudo_prefix_added_when_requested(self):
        cfg = BpftraceConfig(
            target_pid=42,
            variant=BpftraceScriptVariant.LIGHT,
            use_sudo=True,
            sudo_args=["-n"],
            bpftrace_path="/usr/bin/bpftrace",
        )
        runner = BpftraceRunner(cfg)
        cmd = runner._build_command()
        assert cmd[:2] == ["sudo", "-n"]
        assert cmd[2] == "/usr/bin/bpftrace"
        assert cmd[-1] == "42"
        assert cmd[-2].endswith("gpu_cont_light.bt")

    def test_extra_args_pass_through_before_script(self):
        cfg = BpftraceConfig(
            target_pid=7,
            use_sudo=False,
            bpftrace_path="/usr/bin/bpftrace",
            extra_args=["-q", "-B", "line"],
        )
        runner = BpftraceRunner(cfg)
        cmd = runner._build_command()
        # /usr/bin/bpftrace -q -B line <script> 7
        assert cmd[0] == "/usr/bin/bpftrace"
        assert cmd[1:4] == ["-q", "-B", "line"]
        assert cmd[-1] == "7"

    def test_resolve_script_path_uses_variant(self):
        cfg = BpftraceConfig(
            target_pid=1,
            use_sudo=False,
            variant=BpftraceScriptVariant.UNRELATED_KPROBE,
            bpftrace_path="/usr/bin/bpftrace",
        )
        assert cfg.resolve_script_path() == SCRIPTS_DIR / "gpu_cont_unrelated_kprobe.bt"

    def test_resolve_script_path_explicit_override_wins(self, tmp_path: Path):
        custom = tmp_path / "custom.bt"
        cfg = BpftraceConfig(
            target_pid=1,
            use_sudo=False,
            script_path=custom,
            bpftrace_path="/usr/bin/bpftrace",
        )
        assert cfg.resolve_script_path() == custom


# ---------------------------------------------------------------------------
# C1 + C4: startup detection
# ---------------------------------------------------------------------------


class TestStartupDetection:
    """C1+C4: ``start()`` must surface immediate-exit failures."""

    def test_immediate_exit_raises_with_stderr(self):
        def factory(*args, **kwargs):
            return FakePopen(
                args[0],
                **{k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "text", "bufsize")},
                startup_rc=1,
                stderr_lines=["ERROR: Don't have permission to run BPF program\n"],
            )

        runner = _make_runner()
        with _patch_popen(factory):
            with pytest.raises(RuntimeError) as excinfo:
                runner.start()
        # bpftrace's actual error must be reachable to the operator, not
        # buried in some private buffer.
        assert "permission" in str(excinfo.value).lower()
        assert "rc=1" in str(excinfo.value)

    def test_first_stdout_line_unblocks_start(self):
        # Healthy startup: bpftrace prints attach noise quickly. start()
        # must return as soon as it sees any line, not wait the whole
        # startup_timeout_sec.
        fake = {"obj": None}

        def factory(*args, **kwargs):
            fake["obj"] = FakePopen(
                args[0],
                **{k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "text", "bufsize")},
                script_lines=["10 HEARTBEAT alive\n"],
            )
            return fake["obj"]

        runner = _make_runner(startup_timeout_sec=5.0)
        t0 = time.monotonic()
        with _patch_popen(factory):
            runner.start()
            elapsed = time.monotonic() - t0
            try:
                # If start() honoured the early-return signal, this will be
                # well under the 5s timeout.
                assert elapsed < 1.0, f"start() blocked for {elapsed:.2f}s"
            finally:
                runner.stop()

    def test_timeout_without_first_line_proceeds(self):
        # If the script attaches many probes and is silent, start() must
        # still return after startup_timeout_sec rather than blocking
        # forever. A short timeout makes this fast.
        def factory(*args, **kwargs):
            return FakePopen(
                args[0],
                **{k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "text", "bufsize")},
                script_lines=[],
            )

        runner = _make_runner(startup_timeout_sec=0.2)
        with _patch_popen(factory):
            t0 = time.monotonic()
            runner.start()
            try:
                elapsed = time.monotonic() - t0
                # Should be ~ startup_timeout_sec, not infinite.
                assert 0.1 < elapsed < 1.0
                assert runner.is_running
            finally:
                runner.stop()


# ---------------------------------------------------------------------------
# C2: stderr is drained on its own thread.
# ---------------------------------------------------------------------------


class TestStderrDraining:
    def test_stderr_text_accessor_returns_drained_lines(self):
        # The drain thread accumulates bpftrace's stderr without ever
        # blocking the reader; verify the captured text is reachable.
        def factory(*args, **kwargs):
            return FakePopen(
                args[0],
                **{k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "text", "bufsize")},
                script_lines=["1 HEARTBEAT alive\n"],
                stderr_lines=[
                    "Attaching 12 probes...\n",
                    "WARNING: kernel symbol mismatch\n",
                ],
            )

        runner = _make_runner()
        with _patch_popen(factory):
            runner.start()
            try:
                # Give the stderr drain thread a moment to consume.
                time.sleep(0.2)
                stderr = runner.stderr_text()
            finally:
                runner.stop()
        assert "Attaching" in stderr
        assert "kernel symbol mismatch" in stderr

    def test_chatty_stderr_does_not_deadlock_stdout(self):
        # Pre-C2 failure mode: stderr=PIPE was never drained; a bpftrace
        # that writes >64 KiB to stderr would block on write() and
        # silently stop producing stdout events. Simulate by feeding a
        # large stderr volume and checking we still receive stdout
        # events afterwards.
        big_stderr = ["x" * 256 + "\n"] * 512  # ~128 KiB

        def factory(*args, **kwargs):
            return FakePopen(
                args[0],
                **{k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "text", "bufsize")},
                script_lines=["1 HEARTBEAT alive\n", "2 HEARTBEAT alive\n"],
                stderr_lines=big_stderr,
            )

        runner = _make_runner()
        with _patch_popen(factory):
            runner.start()
            try:
                time.sleep(0.3)
                events = runner.snapshot_events()
            finally:
                runner.stop()
        # Both heartbeats must have been parsed; pre-fix this would be 0
        # because the reader thread would be blocked behind a stuffed
        # stderr buffer.
        assert len(events) == 2


# ---------------------------------------------------------------------------
# C3: reader-thread death surfaces.
# ---------------------------------------------------------------------------


class TestReaderThreadDeathSurfaces:
    def test_is_running_false_when_reader_thread_dies(self):
        def factory(*args, **kwargs):
            # Empty stdout, drain thread will idle and stay alive.
            return FakePopen(
                args[0],
                **{k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "text", "bufsize")},
                script_lines=[],
            )

        runner = _make_runner()
        with _patch_popen(factory):
            runner.start()
            try:
                # Forcibly mark the reader thread as not-alive by replacing
                # the Thread object with a stub that reports dead. This
                # simulates an exception that already dropped the thread.
                class _DeadThread:
                    def is_alive(self):
                        return False

                    def join(self, timeout=None):
                        return None

                runner._reader_thread = _DeadThread()
                # is_running should now report False even though _proc
                # is still healthy.
                assert runner.is_running is False
            finally:
                runner.stop()

    def test_stop_logs_reader_exception(self, caplog):
        def factory(*args, **kwargs):
            return FakePopen(
                args[0],
                **{k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "text", "bufsize")},
                script_lines=[],
            )

        runner = _make_runner()
        import logging

        with _patch_popen(factory):
            runner.start()
            runner._reader_exception = RuntimeError("synthetic parser crash")
            with caplog.at_level(logging.WARNING, logger="aorta.ebpf.runner"):
                runner.stop()

        assert any("synthetic parser crash" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# General lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_double_start_raises(self):
        def factory(*args, **kwargs):
            return FakePopen(
                args[0],
                **{k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "text", "bufsize")},
                script_lines=["1 HEARTBEAT alive\n"],
            )

        runner = _make_runner()
        with _patch_popen(factory):
            runner.start()
            try:
                with pytest.raises(RuntimeError, match="already started"):
                    runner.start()
            finally:
                runner.stop()

    def test_stop_on_unstarted_returns_empty(self):
        runner = _make_runner()
        assert runner.stop() == []

    def test_drain_events_atomic_clear(self):
        def factory(*args, **kwargs):
            return FakePopen(
                args[0],
                **{k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "text", "bufsize")},
                script_lines=[
                    "1 HEARTBEAT alive\n",
                    "2 HEARTBEAT alive\n",
                ],
            )

        runner = _make_runner()
        with _patch_popen(factory):
            runner.start()
            try:
                # Wait briefly for both lines to be parsed.
                time.sleep(0.2)
                first = runner.drain_events()
                second = runner.drain_events()
            finally:
                runner.stop()
        assert len(first) == 2
        assert second == []

    def test_on_event_callback_exceptions_are_isolated(self):
        seen: list[str] = []

        def buggy(event):
            seen.append(event.event_type.name)
            raise RuntimeError("callback explodes")

        def factory(*args, **kwargs):
            return FakePopen(
                args[0],
                **{k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "text", "bufsize")},
                script_lines=["1 HEARTBEAT alive\n", "2 HEARTBEAT alive\n"],
            )

        cfg = BpftraceConfig(
            target_pid=1,
            use_sudo=False,
            bpftrace_path="/usr/bin/bpftrace",
            startup_timeout_sec=0.5,
        )
        runner = BpftraceRunner(cfg, on_event=buggy)
        with _patch_popen(factory):
            runner.start()
            try:
                time.sleep(0.2)
                events = runner.snapshot_events()
            finally:
                runner.stop()
        # Both events were parsed and stored even though the callback
        # raised; the runner must keep going on user-callback errors.
        assert len(events) == 2
        assert seen == ["HEARTBEAT", "HEARTBEAT"]


# ---------------------------------------------------------------------------
# PR #162 round 2 (C3): ``stop()`` honours its "never re-raises" docstring.
# ---------------------------------------------------------------------------


class TestStopNeverReRaises:
    """Pin the ``finally``-block-safe contract.

    Pre-fix, ``BpftraceRunner.stop()`` would propagate
    ``ProcessLookupError`` from ``Popen.terminate()`` / ``Popen.kill()``
    on a race where the bpftrace child had already been reaped (e.g.
    sudo's bpftrace exits the moment the traced PID dies). That broke
    distributed cleanup in ``fsdp_trainer.py`` because
    ``dist.destroy_process_group()`` would never run on the failing
    rank, leaking the rendezvous backend and hanging every other rank.
    """

    def _start_runner(self, fake_kwargs):
        captured: dict = {}

        def factory(*args, **kwargs):
            popen_only = {
                k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "text", "bufsize")
            }
            captured["popen"] = FakePopen(args[0], **popen_only, **fake_kwargs)
            return captured["popen"]

        runner = _make_runner()
        ctx = _patch_popen(factory)
        ctx.__enter__()
        runner.start()
        return runner, captured["popen"], ctx

    def test_stop_swallows_terminate_process_lookup_error(self, caplog):
        runner, _fake, ctx = self._start_runner(
            {
                "script_lines": [],
                "terminate_raises": ProcessLookupError(3, "No such process"),
            }
        )
        try:
            import logging

            with caplog.at_level(logging.DEBUG, logger="aorta.ebpf.runner"):
                events = runner.stop()
        finally:
            ctx.__exit__(None, None, None)
        # No exception propagated, return type is still the events list.
        assert isinstance(events, list)
        assert runner.is_running is False

    def test_stop_swallows_kill_process_lookup_error(self):
        # ``terminate()`` succeeds, but ``wait()`` times out so the code
        # path falls through to ``kill()``, which races. Pre-fix this
        # would re-raise ``ProcessLookupError`` from ``stop()``.
        import subprocess as _sp

        runner, _fake, ctx = self._start_runner(
            {
                "script_lines": [],
                "wait_raises": _sp.TimeoutExpired(cmd="bpftrace", timeout=0.1),
                "kill_raises": ProcessLookupError(3, "No such process"),
            }
        )
        try:
            events = runner.stop()
        finally:
            ctx.__exit__(None, None, None)
        assert isinstance(events, list)

    def test_stop_short_circuits_on_already_exited(self):
        # If the child already exited (poll() returns an rc), we must
        # not even attempt to terminate -- doing so was the original
        # source of the ProcessLookupError race.
        captured: dict = {}

        def factory(*args, **kwargs):
            popen_only = {
                k: v for k, v in kwargs.items() if k in ("stdout", "stderr", "text", "bufsize")
            }
            captured["popen"] = FakePopen(
                args[0],
                **popen_only,
                script_lines=["1 HEARTBEAT alive\n"],
            )
            return captured["popen"]

        runner = _make_runner()
        with _patch_popen(factory):
            runner.start()
            # Simulate child exit between start() and stop().
            captured["popen"]._returncode = 0
            captured["popen"]._stdout_done = True
            captured["popen"]._stderr_done = True
            captured["popen"]._stdout_event.set()
            captured["popen"]._stderr_event.set()
            # Make terminate() blow up to prove we never call it.
            captured["popen"]._terminate_raises = AssertionError(
                "terminate() must not be called after poll() reports exit"
            )
            captured["popen"]._kill_raises = AssertionError(
                "kill() must not be called after poll() reports exit"
            )
            events = runner.stop()
        assert isinstance(events, list)

    def test_stop_swallows_unexpected_exception(self, caplog):
        # Defensive: any unexpected exception type must still be
        # logged-and-swallowed, not propagated. The docstring promises
        # this without qualifying on exception class.
        runner, _fake, ctx = self._start_runner(
            {
                "script_lines": [],
                "terminate_raises": OSError(1, "Operation not permitted"),
            }
        )
        try:
            import logging

            with caplog.at_level(logging.ERROR, logger="aorta.ebpf.runner"):
                events = runner.stop()
        finally:
            ctx.__exit__(None, None, None)
        assert isinstance(events, list)


# ---------------------------------------------------------------------------
# PR #162 round 2 (C4): explicit ``bpftrace_path`` is validated up-front.
# ---------------------------------------------------------------------------


class TestExplicitBpftracePathValidation:
    """``BpftraceConfig.bpftrace_path`` must point at a real file.

    Pre-fix, a typo or stale absolute path would surface as a generic
    ``FileNotFoundError`` from ``subprocess.Popen`` deep inside
    ``start()`` -- callers (CLI, trainer) could not distinguish that
    from "binary missing entirely on PATH" without parsing the
    ``errno``. We now raise ``BpftraceUnavailableError`` synchronously
    from ``_build_command()`` with a self-explanatory message.
    """

    def test_explicit_path_to_missing_file_raises_unavailable(self, tmp_path: Path):
        cfg = BpftraceConfig(
            target_pid=1,
            use_sudo=False,
            bpftrace_path=str(tmp_path / "definitely-not-here"),
        )
        runner = BpftraceRunner(cfg)
        with pytest.raises(BpftraceUnavailableError) as excinfo:
            runner._build_command()
        msg = str(excinfo.value)
        assert "definitely-not-here" in msg
        assert "bpftrace_path" in msg

    def test_explicit_path_to_directory_raises_unavailable(self, tmp_path: Path):
        # Pointing at a directory was the same shape of bug; the
        # validation must use ``is_file()``, not just ``exists()``.
        cfg = BpftraceConfig(
            target_pid=1,
            use_sudo=False,
            bpftrace_path=str(tmp_path),
        )
        runner = BpftraceRunner(cfg)
        with pytest.raises(BpftraceUnavailableError):
            runner._build_command()

    def test_explicit_path_to_existing_file_is_accepted(self, tmp_path: Path):
        binary = tmp_path / "fake_bpftrace"
        binary.touch()
        cfg = BpftraceConfig(
            target_pid=1,
            use_sudo=False,
            bpftrace_path=str(binary),
        )
        runner = BpftraceRunner(cfg)
        cmd = runner._build_command()
        assert cmd[0] == str(binary)


# ---------------------------------------------------------------------------
# Vendored scripts ship as package data
# ---------------------------------------------------------------------------


class TestVendoredScriptsExist:
    @pytest.mark.parametrize("variant", list(BpftraceScriptVariant))
    def test_each_variant_resolves_to_an_existing_file(self, variant):
        # If pyproject.toml's package-data glob ever drops the .bt files
        # this would fail at the first installed-package use.
        path = SCRIPTS_DIR / variant.value
        assert path.exists(), f"vendored script missing: {path}"
