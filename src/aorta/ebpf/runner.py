"""Subprocess-based wrapper around the vendored bpftrace scripts.

The runner spawns ``sudo bpftrace <script> <pid>`` in a background thread,
streams stdout line-by-line into a parser, and exposes start/stop lifecycle
hooks suitable for being driven from a training loop.

The runner does NOT load eBPF bytecode in-process; it leverages the existing
bpftrace toolchain as a subprocess. This matches the operational model of
the upstream ebpfaultline scripts.
"""

from __future__ import annotations

import enum
import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .events import KernelEvent
from .parser import BpftraceLogParser

log = logging.getLogger(__name__)


SCRIPTS_DIR = Path(__file__).parent / "scripts"


class BpftraceScriptVariant(enum.Enum):
    """Vendored bpftrace script variants.

    Trade-offs (see ``scripts/PROVENANCE.md`` for the per-variant
    Heisenberg-risk table this list summarises):
      - ``FULL`` -- maximum visibility, but kprobes can serialize the kernel
        path enough to suppress non-deterministic GPU memory races.
      - ``LIGHT`` -- only the three KFD/SVM kprobes plus ioctl errors and
        signals; documented as the "smoking gun" path.
      - ``TP_ONLY`` -- tracepoints only, no kprobes; minimal Heisenberg
        effect; recommended default for production debugging.
      - ``ONE_KPROBE`` -- TP_ONLY + a single eviction kprobe (experiment).
      - ``UNRELATED_KPROBE`` -- TP_ONLY + an unrelated openat kprobe
        (control experiment).
    """

    FULL = "gpu_cont.bt"
    LIGHT = "gpu_cont_light.bt"
    TP_ONLY = "gpu_cont_tp_only.bt"
    ONE_KPROBE = "gpu_cont_1kprobe.bt"
    UNRELATED_KPROBE = "gpu_cont_unrelated_kprobe.bt"


@dataclass
class BpftraceConfig:
    """Configuration for a single ``BpftraceRunner`` invocation.

    Attributes:
        target_pid: PID of the process whose syscalls/signals are filtered.
        variant: Which vendored bpftrace script to run.
        use_sudo: Whether to prefix the command with ``sudo``. Defaults to
            True; bpftrace usually requires CAP_BPF/CAP_PERFMON or root.
        bpftrace_path: Optional explicit path to the ``bpftrace`` binary.
            If None, the runner uses the first ``bpftrace`` found on PATH.
        script_path: Optional explicit path to a ``.bt`` script (overrides
            ``variant``). Useful for custom or experimental scripts.
        extra_args: Extra arguments passed to bpftrace before the script.
        sudo_args: Extra arguments passed to ``sudo`` (e.g. ``["-n"]`` for
            non-interactive in CI).
        startup_timeout_sec: How long to wait for bpftrace to print its
            first attach line before considering startup failed.
    """

    target_pid: int
    variant: BpftraceScriptVariant = BpftraceScriptVariant.TP_ONLY
    use_sudo: bool = True
    bpftrace_path: str | None = None
    script_path: Path | None = None
    extra_args: list[str] = field(default_factory=list)
    sudo_args: list[str] = field(default_factory=lambda: ["-n"])
    startup_timeout_sec: float = 10.0

    def resolve_script_path(self) -> Path:
        if self.script_path is not None:
            return self.script_path
        return SCRIPTS_DIR / self.variant.value


class BpftraceUnavailableError(RuntimeError):
    """Raised when the bpftrace binary cannot be located on PATH."""


class BpftraceRunner:
    """Spawn and manage a single bpftrace process.

    Lifecycle:

        runner = BpftraceRunner(BpftraceConfig(target_pid=1234))
        runner.start()
        # ... workload runs ...
        events = runner.stop()
    """

    def __init__(
        self,
        config: BpftraceConfig,
        *,
        on_event: Callable[[KernelEvent], None] | None = None,
    ) -> None:
        self.config = config
        self._on_event = on_event
        self._parser = BpftraceLogParser()

        self._proc: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._events_lock = threading.Lock()
        self._events: list[KernelEvent] = []
        self._raw_lines: list[str] = []
        # Drained on its own thread; merging into stdout would either deadlock
        # (stderr=PIPE never drained) or pollute the event parser with bpftrace
        # status lines ("Attaching N probes..."). Holding them separately also
        # lets ``stderr_text()`` surface the actual bpftrace diagnostics on
        # startup / runtime failures.
        self._stderr_lines: list[str] = []
        self._reader_exception: BaseException | None = None
        self._running = False

    @staticmethod
    def is_bpftrace_available(bpftrace_path: str | None = None) -> bool:
        """Return True if a bpftrace binary is reachable on PATH or at the path."""
        if bpftrace_path:
            return Path(bpftrace_path).is_file()
        return shutil.which("bpftrace") is not None

    def _build_command(self) -> list[str]:
        if self.config.bpftrace_path:
            if not Path(self.config.bpftrace_path).is_file():
                raise BpftraceUnavailableError(
                    "BpftraceConfig.bpftrace_path was set to "
                    f"{self.config.bpftrace_path!r} but no regular file "
                    "exists at that path; install bpftrace there or "
                    "leave bpftrace_path unset to fall back to PATH lookup"
                )
            bpftrace_bin: str | None = self.config.bpftrace_path
        else:
            bpftrace_bin = shutil.which("bpftrace")

        if not bpftrace_bin:
            raise BpftraceUnavailableError(
                "bpftrace binary not found on PATH; install bpftrace or set "
                "BpftraceConfig.bpftrace_path"
            )

        script_path = self.config.resolve_script_path()
        if not script_path.exists():
            raise FileNotFoundError(f"bpftrace script not found: {script_path}")

        cmd: list[str] = []
        if self.config.use_sudo:
            cmd.append("sudo")
            cmd.extend(self.config.sudo_args)
        cmd.append(bpftrace_bin)
        cmd.extend(self.config.extra_args)
        cmd.append(str(script_path))
        cmd.append(str(self.config.target_pid))
        return cmd

    def start(self) -> None:
        """Spawn the bpftrace process and begin background log parsing.

        Detects the common immediate-exit failure modes (binary missing on
        PATH, ``sudo`` password prompt swallowed, "ERROR: Don't have
        permission to run BPF program") by polling the subprocess for up
        to ``BpftraceConfig.startup_timeout_sec``. If the process exits
        during that window the captured stderr is bundled into the
        ``RuntimeError`` so the operator sees bpftrace's own diagnostic
        rather than an opaque "no events arrived" surprise later.

        Returns as soon as either bpftrace produces its first stdout line
        (the parser's signal that probes have attached) or the timeout
        elapses without an early exit -- whichever comes first.
        """
        if self._running:
            raise RuntimeError("BpftraceRunner already started")

        cmd = self._build_command()
        log.info("Starting bpftrace: %s", " ".join(cmd))

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._running = True

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"bpftrace-reader-{self.config.target_pid}",
            daemon=True,
        )
        self._reader_thread.start()

        # bpftrace can write a lot to stderr (probe-attach status lines,
        # warnings about kernel-symbol mismatches). Without an active drain
        # the OS pipe buffer fills (~64 KiB on Linux) and bpftrace blocks on
        # write -- which looks to us like "events stopped arriving" with no
        # other signal. Spawn a dedicated drainer.
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop,
            name=f"bpftrace-stderr-{self.config.target_pid}",
            daemon=True,
        )
        self._stderr_thread.start()

        self._await_startup()

    def _await_startup(self) -> None:
        """Block until bpftrace either attaches or exits during startup.

        Three exit paths:
          - subprocess returns rc != None during the window -> raise with
            the captured stderr included so the operator can see why.
          - reader thread dies before any line arrives (parser bug, OS-
            level read failure) -> raise with the original exception
            chained.
          - first stdout line shows up OR the timeout elapses with the
            process still alive -> return; assume bpftrace has attached.
        """
        assert self._proc is not None
        deadline = time.monotonic() + self.config.startup_timeout_sec
        poll_interval_sec = 0.1
        while time.monotonic() < deadline:
            rc = self._proc.poll()
            if rc is not None:
                self._running = False
                # Give the stderr drainer a moment to flush so the
                # RuntimeError carries bpftrace's own error message
                # ("ERROR: Don't have permission...", "Could not open
                # tracepoint...") instead of an empty string.
                if self._stderr_thread is not None:
                    self._stderr_thread.join(timeout=0.5)
                stderr_text = self.stderr_text()
                raise RuntimeError(
                    f"bpftrace exited during startup with rc={rc}; "
                    f"stderr: {stderr_text.rstrip() or '<empty>'}"
                )
            reader = self._reader_thread
            if reader is not None and not reader.is_alive():
                self._running = False
                exc = self._reader_exception
                raise RuntimeError("bpftrace reader thread exited during startup") from exc
            if self._raw_lines:
                return
            time.sleep(poll_interval_sec)

        # Healthy bpftrace can take longer than ``startup_timeout_sec`` to
        # print its first event (e.g. when attaching a large probe set
        # under ``FULL``). The poll loop above already proved the process
        # didn't exit early, so accept this as "running but quiet".
        log.debug(
            "bpftrace produced no stdout within startup_timeout_sec=%.1fs; assuming attached.",
            self.config.startup_timeout_sec,
        )

    def _reader_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        try:
            for line in self._proc.stdout:
                self._raw_lines.append(line)
                event = self._parser.parse_line(line)
                if event is None:
                    continue
                with self._events_lock:
                    self._events.append(event)
                if self._on_event is not None:
                    try:
                        self._on_event(event)
                    except Exception:
                        log.exception("on_event callback raised; continuing")
        except BaseException as exc:
            # Surface to start()/is_running/stop() instead of letting the
            # thread silently die. ``log.exception`` would otherwise let a
            # caller-polling-``is_running`` see "still running" while no
            # events accumulate -- the original C3 failure mode.
            log.exception("bpftrace reader thread terminated unexpectedly")
            self._reader_exception = exc

    def _stderr_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stderr is not None
        try:
            for line in self._proc.stderr:
                self._stderr_lines.append(line)
                log.debug("bpftrace stderr: %s", line.rstrip())
        except Exception:
            log.exception("bpftrace stderr drain thread terminated unexpectedly")

    def stderr_text(self) -> str:
        """Return bpftrace's stderr captured so far, joined into one string."""
        return "".join(self._stderr_lines)

    def stop(self, timeout_sec: float = 5.0) -> list[KernelEvent]:
        """Terminate the bpftrace process and return all collected events.

        Contract: ``stop()`` never re-raises. Callers (notably the FSDP
        trainer's distributed cleanup ``finally`` block) rely on this so
        that an early-exited bpftrace, a sudo-killed subprocess, or a
        ``ProcessLookupError`` race between ``poll()`` and ``terminate()``
        cannot leak the rendezvous backend on every other rank.

        Errors hit during shutdown are logged and swallowed, including
        the previously surfaced ``_reader_exception`` from a crashed
        reader thread.
        """
        if not self._running or self._proc is None:
            return []

        log.info("Stopping bpftrace (pid=%s)", self._proc.pid)
        try:
            self._terminate_proc(timeout_sec)
        except BaseException:
            # Belt-and-braces: ``_terminate_proc`` already swallows the
            # documented races, but if anything else slips through we
            # still must not propagate -- the docstring promises this
            # method is finally-block-safe. Log with full traceback so
            # the failure is observable.
            log.exception("BpftraceRunner.stop() swallowed unexpected error")
        finally:
            self._running = False
            try:
                if self._reader_thread is not None:
                    self._reader_thread.join(timeout=timeout_sec)
                if self._stderr_thread is not None:
                    self._stderr_thread.join(timeout=timeout_sec)
            except BaseException:
                log.exception("BpftraceRunner.stop() swallowed thread-join error")

        if self._reader_exception is not None:
            log.warning(
                "bpftrace reader thread exited with %r; events collected up to "
                "the failure are returned but the run was incomplete.",
                self._reader_exception,
            )

        with self._events_lock:
            return list(self._events)

    def _terminate_proc(self, timeout_sec: float) -> None:
        """Best-effort ``terminate()`` -> ``wait()`` -> ``kill()`` sequence.

        Each ``Popen`` call is guarded against ``ProcessLookupError``
        because the kernel can reap the bpftrace child between our
        ``poll()`` short-circuit (below) and the actual signal -- e.g.
        sudo's bpftrace exits as soon as the traced PID dies.
        """
        assert self._proc is not None  # narrowed by caller

        # ``poll()`` short-circuit: if bpftrace already exited we have
        # nothing to terminate. This also avoids a guaranteed
        # ``ProcessLookupError`` from ``terminate()`` on the post-exit
        # PID-reuse-window OSes that send the signal eagerly.
        if self._proc.poll() is not None:
            return

        try:
            self._proc.terminate()
        except ProcessLookupError:
            return

        try:
            self._proc.wait(timeout=timeout_sec)
            return
        except subprocess.TimeoutExpired:
            log.warning(
                "bpftrace did not terminate in %.1fs; killing",
                timeout_sec,
            )

        try:
            self._proc.kill()
        except ProcessLookupError:
            return
        try:
            self._proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            log.error(
                "bpftrace did not exit even after SIGKILL within %.1fs; "
                "leaving the subprocess to the OS reaper",
                timeout_sec,
            )

    def snapshot_events(self) -> list[KernelEvent]:
        """Return a copy of events collected so far without stopping."""
        with self._events_lock:
            return list(self._events)

    def drain_events(self) -> list[KernelEvent]:
        """Atomically remove and return all events accumulated so far."""
        with self._events_lock:
            events = self._events
            self._events = []
            return events

    @property
    def is_running(self) -> bool:
        """True iff the subprocess AND the reader thread are both alive.

        The reader-thread check was added with C3: a reader that died
        with the subprocess still attached used to leave ``is_running``
        stuck on True while no further events arrived, which is exactly
        the silent-failure mode we want callers to be able to detect.
        """
        if not self._running or self._proc is None or self._proc.poll() is not None:
            return False
        reader = self._reader_thread
        if reader is not None and not reader.is_alive():
            return False
        return True

    def __enter__(self) -> BpftraceRunner:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


__all__ = [
    "BpftraceConfig",
    "BpftraceRunner",
    "BpftraceScriptVariant",
    "BpftraceUnavailableError",
    "SCRIPTS_DIR",
]
