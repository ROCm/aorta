"""ASan batching (design Section 6.2, ASan batching).

Candidates whose reward is still undecided after S2 queue for ASan. A request is sent when it can be
filled (`asan.max_packages` packages, counting in-band controls), or when the oldest queued package
has waited `asan.max_wait_s`, or when generation has ended and the queue is not empty. At most
`asan.max_inflight` requests are outstanding (None: no limit). ASan never runs speculatively: a
package is only queued once S2 has settled for its candidate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchPolicy:
    max_packages: int = 16
    max_wait_s: float = 20.0
    max_inflight: int | None = None
    inband_controls: int = 0  # control packages a request carries (0, or 2: first and last)

    @property
    def capacity(self) -> int:
        """Candidate packages per request."""
        return self.max_packages - self.inband_controls

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"max_packages {self.max_packages} leaves no room beside {self.inband_controls} controls")


def take_batch(ready_times: list[float], *, now: float, policy: BatchPolicy, generation_ended: bool,
               inflight: int) -> int:
    """How many of the queued packages (oldest first; `ready_times` are their queue times) to send in
    one request now: 0 to wait."""
    if not ready_times:
        return 0
    if policy.max_inflight is not None and inflight >= policy.max_inflight:
        return 0
    n = min(len(ready_times), policy.capacity)
    if len(ready_times) >= policy.capacity or generation_ended:
        return n
    if now - ready_times[0] >= policy.max_wait_s:
        return n
    return 0


def next_deadline(ready_times: list[float], policy: BatchPolicy) -> float | None:
    """When the oldest queued package's wait runs out."""
    return ready_times[0] + policy.max_wait_s if ready_times else None
