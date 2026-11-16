"""`WithSpeedup`: another reward function's clean reward, graded by a timing leg (reward/timing.py).

It wraps a reward function (`inner`) and adds one leg, by default `ref.speedup`, which the profile runs after the
inner function's correctness legs (its `requires`). Whatever the inner function scores as not clean stands as it is:
a wrong or failed candidate is never timed. A clean inner reward needs the timing leg, and its base is replaced by
`timing.speedup_base` of that leg (the broke-healthy and length parts stay the inner function's):

| the timing leg | base |
|---|---|
| clean (faster than the reference by the ratio, past the noise floor) | 0.75 to 1.0, by how far past the ratio |
| fail `not_faster` (correct, measured, not faster by the ratio) | below 0.5, by the measured speedup |
| anything else (unsettled, not run, a record that does not match its state) | unscored |

Only the first row is clean (`claims_clean`: a base of at least `timing.CLEAN_BASE`). So that this holds for every
reward the wrapper passes through, an inner reward that is not clean but has a base of `CLEAN_BASE` or more is
unscored (`inner_base_overlaps_clean`); hipk.v1 and standin.v11 have none.
"""

from __future__ import annotations

import hashlib

from ...schemas import Candidate, LegResult
from .. import timing
from .base import Decided, Decision, NeedMore, Unscored, code_sha256


class WithSpeedup:
    def __init__(self, inner, *, name: str, leg: str = "ref.speedup"):
        if leg in inner.legs:
            raise ValueError(f"{inner.name} already reads {leg}")
        self.inner, self.name, self.leg = inner, name, leg
        self.version = f"{inner.version}+speedup-1"
        self.legs = tuple(inner.legs) + (leg,)
        self.settings = {"inner": inner.name, "inner_settings": dict(getattr(inner, "settings", {}) or {}),
                         "speedup_leg": leg}
        own = code_sha256(__file__, timing.__file__)
        self.code_sha256 = hashlib.sha256(f"{inner.code_sha256}:{own}".encode()).hexdigest()

    def _grade(self, leg: LegResult, inner_parts: dict) -> tuple[float | None, dict]:
        settled = leg.state == "clean" or (leg.state == "fail" and leg.reason == "not_faster")
        if not settled:
            return None, {"reason": f"{self.leg}:{leg.reason or leg.state}"}
        rec = timing.measurement(leg)
        base = timing.speedup_base(leg.state, rec) if rec is not None else None
        if base is None:
            return None, {"reason": f"{self.leg}:no_measurement"}
        parts = {"base": float(base), "broke_healthy": float(inner_parts["broke_healthy"]),
                 "length": float(inner_parts["length"])}
        return parts["base"] + parts["broke_healthy"] + parts["length"], parts

    def _passed_through(self, reward: float | None, parts: dict) -> tuple[float | None, dict]:
        if reward is not None and parts.get("base", 0.0) >= timing.CLEAN_BASE:
            return None, {"reason": "inner_base_overlaps_clean"}
        return reward, parts

    def compute(self, legs: dict[str, LegResult], cand: Candidate) -> tuple[float | None, dict]:
        reward, parts = self.inner.compute(legs, cand)
        if reward is None or not self.inner.claims_clean(reward, parts):
            return self._passed_through(reward, parts)
        return self._grade(legs[self.leg], parts)

    def decide(self, partial: dict[str, LegResult], cand: Candidate) -> Decision:
        d = self.inner.decide(partial, cand)
        if not isinstance(d, Decided):
            return d
        if not self.inner.claims_clean(d.reward, d.parts):
            reward, parts = self._passed_through(d.reward, d.parts)
            return Unscored(parts["reason"]) if reward is None else d
        if self.leg not in partial:
            return NeedMore((self.leg,))
        reward, parts = self._grade(partial[self.leg], d.parts)
        return Unscored(parts["reason"]) if reward is None else Decided(reward, parts)

    @staticmethod
    def claims_clean(reward: float | None, parts: dict | None) -> bool:
        return reward is not None and isinstance(parts, dict) and parts.get("base", 0.0) >= timing.CLEAN_BASE

    def not_applicable_legs(self, legs: dict[str, LegResult], cand: Candidate):
        rule = getattr(self.inner, "not_applicable_legs", None)
        return frozenset(rule(legs, cand)) if rule is not None else frozenset()
