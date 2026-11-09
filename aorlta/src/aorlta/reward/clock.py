"""The reward service's clock.

All service timing (leg timeouts, `stall_s`, heartbeats, ASan `max_wait_s`, control intervals, the
drain deadline) is in seconds of this clock. `scale` is real seconds per clock second: 1.0 in a run,
smaller in tests, so a test can exercise the configured defaults (a 300 s stall threshold, a 900 s
ASan request timeout) in a fraction of the wall time without changing any threshold.

`now()` is monotonic. With scale 1.0 it equals `time.monotonic()`, so a caller's
`time.monotonic() + scoring_s` is a valid drain deadline. `wall()` is UTC on the same timeline, for
records and spans.
"""

from __future__ import annotations

import math
import threading
import time


class Clock:
    def __init__(self, scale: float = 1.0):
        if not (isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0):
            raise ValueError(f"scale must be a finite number > 0, got {scale!r}")
        self.scale = float(scale)
        self._m0 = time.monotonic()
        self._w0 = time.time()

    def now(self) -> float:
        """Monotonic clock seconds (equal to time.monotonic() at scale 1)."""
        return self._m0 + (time.monotonic() - self._m0) / self.scale

    def wall(self) -> float:
        """UTC seconds since the epoch, on this clock's timeline."""
        return self._w0 + (self.now() - self._m0)

    def to_wall(self, t_now: float) -> float:
        """A `now()` reading as UTC."""
        return self._w0 + (t_now - self._m0)

    def real(self, seconds: float) -> float:
        """Real seconds for `seconds` of this clock (for blocking waits)."""
        return max(0.0, seconds) * self.scale

    def sleep(self, seconds: float, interrupt: threading.Event | None = None) -> bool:
        """Sleep `seconds` of this clock; True if `interrupt` was set first."""
        event = interrupt if interrupt is not None else threading.Event()
        return event.wait(self.real(seconds))
