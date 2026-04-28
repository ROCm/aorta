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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .events import KernelEvent
from .parser import BpftraceLogParser

log = logging.getLogger(__name__)


SCRIPTS_DIR = Path(__file__).parent / "scripts"


class BpftraceScriptVariant(enum.Enum):
    """Vendored bpftrace script variants.

    Trade-offs (from ebpfaultline README and PIPELINE_ANALYSIS):
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
    bpftrace_path: Optional[str] = None
    script_path: Optional[Path] = None
    extra_args: List[str] = field(default_factory=list)
    sudo_args: List[str] = field(default_factory=lambda: ["-n"])
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
        on_event: Optional[Callable[[KernelEvent], None]] = None,
    ) -> None:
        self.config = config
        self._on_event = on_event
        self._parser = BpftraceLogParser()

        self._proc: Optional[subprocess.Popen[str]] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._events_lock = threading.Lock()
        self._events: List[KernelEvent] = []
        self._raw_lines: List[str] = []
        self._running = False

    @staticmethod
    def is_bpftrace_available(bpftrace_path: Optional[str] = None) -> bool:
        """Return True if a bpftrace binary is reachable on PATH or at the path."""
        if bpftrace_path:
            return Path(bpftrace_path).is_file()
        return shutil.which("bpftrace") is not None

    def _build_command(self) -> List[str]:
        bpftrace_bin = self.config.bpftrace_path or shutil.which("bpftrace")
        if not bpftrace_bin:
            raise BpftraceUnavailableError(
                "bpftrace binary not found on PATH; install bpftrace or set "
                "BpftraceConfig.bpftrace_path"
            )

        script_path = self.config.resolve_script_path()
        if not script_path.exists():
            raise FileNotFoundError(f"bpftrace script not found: {script_path}")

        cmd: List[str] = []
        if self.config.use_sudo:
            cmd.append("sudo")
            cmd.extend(self.config.sudo_args)
        cmd.append(bpftrace_bin)
        cmd.extend(self.config.extra_args)
        cmd.append(str(script_path))
        cmd.append(str(self.config.target_pid))
        return cmd

    def start(self) -> None:
        """Spawn the bpftrace process and begin background log parsing."""
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
        except Exception:
            log.exception("bpftrace reader thread terminated unexpectedly")

    def stop(self, timeout_sec: float = 5.0) -> List[KernelEvent]:
        """Terminate the bpftrace process and return all collected events."""
        if not self._running or self._proc is None:
            return []

        log.info("Stopping bpftrace (pid=%s)", self._proc.pid)
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                log.warning("bpftrace did not terminate in %.1fs; killing", timeout_sec)
                self._proc.kill()
                self._proc.wait(timeout=timeout_sec)
        finally:
            self._running = False
            if self._reader_thread is not None:
                self._reader_thread.join(timeout=timeout_sec)

        with self._events_lock:
            return list(self._events)

    def snapshot_events(self) -> List[KernelEvent]:
        """Return a copy of events collected so far without stopping."""
        with self._events_lock:
            return list(self._events)

    def drain_events(self) -> List[KernelEvent]:
        """Atomically remove and return all events accumulated so far."""
        with self._events_lock:
            events = self._events
            self._events = []
            return events

    @property
    def is_running(self) -> bool:
        return self._running and self._proc is not None and self._proc.poll() is None

    def __enter__(self) -> "BpftraceRunner":
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
