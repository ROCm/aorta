"""The RewardFn protocol (design Section 6.2, Reward function) and the decision types
of early exit: `decide(partial) -> Decided(reward) | NeedMore(legs) | Unscored(reason)`.

`partial` maps a leg to its result. A leg that is absent is still pending and may yet produce any
outcome; a leg that can no longer run (not in the profile, or its inputs never came) is present with
state `not_run`. `decide` may answer Decided or Unscored only when no outcome of the pending legs
could change the answer: then `compute` on any completion of `partial` gives the same result, which
the service checks again when it finalises.

A reward function whose scoring reads its own configuration names it in `settings` (standin.v11's
`clean_repeats`), which the reward cache's key holds beside the name, version and code hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ...schemas import Candidate, LegResult


@dataclass(frozen=True)
class Decided:
    reward: float
    parts: dict


@dataclass(frozen=True)
class NeedMore:
    legs: tuple[str, ...]


@dataclass(frozen=True)
class Unscored:
    reason: str


Decision = Decided | NeedMore | Unscored


@runtime_checkable
class RewardFn(Protocol):
    name: str
    version: str  # "11", "12"
    code_sha256: str  # hash of the module file(s) actually imported
    legs: tuple[str, ...]  # every leg its record names; compute needs all of them

    def decide(self, partial: dict[str, LegResult], cand: Candidate) -> Decision: ...

    def compute(self, legs: dict[str, LegResult], cand: Candidate) -> tuple[float | None, dict]: ...

    def claims_clean(self, reward: float | None, parts: dict | None) -> bool: ...


def code_sha256(*paths: str | Path) -> str:
    """sha256 over the module files, in the order given, each prefixed by its length."""
    h = hashlib.sha256()
    for p in paths:
        data = Path(p).read_bytes()
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()
