"""`standin.v11`: reward v11 (design Appendix B), the reward of the stand-in runs.

`compute` is Appendix B's code. Its open points are settled here as follows, each the fail-closed
reading:

- "in-round" sanitizers are those whose leg ran (state other than `not_run`), as Appendix B writes.
- "k settled-clean repeats": every reference and sanitizer leg that ran has `repeats >= k`
  (`clean_repeats`, default 1).
- "trusted compiled evidence": `gate.compile` is clean in this service's own run.
- "no structure drift": no leg reports a `structure_drift` finding or reason.
- "capped": not implemented; a clean that is not proven is unscored (`clean_not_proven`).
- "A compiled output channel is -1.0": not implemented (meaning unclear).

Three inputs Appendix B's code would let through are made unsettled before the ladder reads them,
so none can count as a pass: a `skipped_by_plan` leg (Section 6.3 option A: skipped legs reach the
reward as unsettled), a `finding` on a leg that is not a sanitizer, and a `fail` on a reference or
sanitizer leg whose reason the ladder does not know.

`decide` walks the same ladder over a partial record and answers only from legs it has; the first
pending leg it needs makes it answer NeedMore. So whenever it answers, `compute` on any completion
of the record gives that answer (tested in C2.T10).
"""

from __future__ import annotations

import dataclasses

from ...schemas import Candidate, LegResult
from . import base
from .base import Decided, Decision, NeedMore, Unscored, code_sha256

W = {"consan": 1.0, "waitcheck": 1.0, "asan": 1.0}
GATES = ["parse", "contract", "static", "reference", "compile",
         "no_artifact", "spec_invalid", "static_reject", "no_change_on_failing_parent"]
KNOWN_GATE_REASONS = frozenset([g for g in GATES if g != "compile"] + ["compile_error"])
LADDER = ("ref.native", "ref.edge", "san.consan", "san.waitcheck", "san.asan")
GATE_LEGS = tuple(f"gate.{g}" for g in GATES)
SAN_LEGS = tuple(f"san.{s}" for s in W)
LEGS = GATE_LEGS + LADDER
MISMATCH_REASONS = ("wrong_output", "edge_mismatch")
FAULT_REASONS = ("crash", "fault", "timeout")
# Every reason a None result can carry (C2.T10): a gate or reference leg that did not settle, its
# control bracket, an unknown gate reason, and the two sanitizer outcomes.
UNSCORED_REASONS = frozenset(
    [f"gate.{g}" for g in GATES] + ["ref.native", "ref.edge"] + [f"{leg}.control" for leg in LADDER]
    + ["unknown_gate_reason", "sanitizer_not_settled", "clean_not_proven"]
)
REWARD_MIN, REWARD_MAX = -1.55, 1.0


def _normalise(name: str, leg: LegResult) -> LegResult:
    """The three inputs that must not pass as clean become unsettled (see the module docstring)."""
    if leg.state == "skipped_by_plan":
        return dataclasses.replace(leg, state="unsettled", reason="skipped_by_plan")
    if leg.state == "finding" and not name.startswith("san."):
        return dataclasses.replace(leg, state="unsettled", reason="finding_on_non_sanitizer_leg")
    if (leg.state == "fail" and not name.startswith("gate.")
            and leg.reason not in MISMATCH_REASONS + FAULT_REASONS):
        return dataclasses.replace(leg, state="unsettled", reason=f"unknown_fail_reason:{leg.reason}")
    return leg


class StandinV11:
    name = "standin.v11"
    version = "11"
    legs = LEGS

    def __init__(self, clean_repeats: int = 1):
        if not (isinstance(clean_repeats, int) and not isinstance(clean_repeats, bool) and clean_repeats >= 1):
            raise ValueError(f"clean_repeats must be an int >= 1, got {clean_repeats!r}")
        self.k = clean_repeats
        self.code_sha256 = code_sha256(__file__, base.__file__)

    # --- Appendix B --------------------------------------------------------------------------------

    def compute(self, legs: dict[str, LegResult], cand: Candidate) -> tuple[float | None, dict]:
        legs = {name: _normalise(name, legs[name]) for name in LEGS}

        def confirmed():  # rule 5: a confirmed finding is always penalised
            return [s for s in W if legs[f"san.{s}"].state == "finding"]

        def sanitizer_base(found):
            in_round = [s for s in W if legs[f"san.{s}"].state != "not_run"]  # "in-round": Q8
            return -0.5 * sum(W[s] for s in found) / sum(W[s] for s in in_round)

        def unscored_unless_finding(reason):
            found = confirmed()
            return finish(sanitizer_base(found), clean=False) if found else (None, {"reason": reason})

        def finish(base_, clean):
            return self._finish(base_, clean, cand)

        for g in GATES:  # rule 1
            leg = legs[f"gate.{g}"]
            if leg.state == "unsettled":
                return unscored_unless_finding(f"gate.{g}")
            if leg.state == "fail":
                if leg.reason not in KNOWN_GATE_REASONS:
                    return None, {"reason": "unknown_gate_reason"}
                return finish(-0.9 if leg.reason == "compile_error" else -1.0, clean=False)
        for leg_name in LADDER:  # rules 2-4
            leg = legs[leg_name]
            if leg_name.startswith("ref.") and leg.state == "unsettled":
                return unscored_unless_finding(leg_name)
            if leg.state == "fail":  # "fail" = attributed to the candidate by the checker
                if leg.reason in MISMATCH_REASONS:
                    return finish(-0.7, clean=False)
                if leg.reason in FAULT_REASONS:
                    if leg_name == "san.consan" and leg.reason == "timeout":
                        continue  # rule 4: a ConSan-only timeout leaves its property unscored
                    if leg.bracket and leg.bracket["clean"]:
                        return finish(-0.8, clean=False)
                    return unscored_unless_finding(f"{leg_name}.control")
        found = confirmed()  # rules 5-6
        if found:
            return finish(sanitizer_base(found), clean=False)
        if any(legs[f"san.{s}"].state != "clean" for s in W):  # unsettled, skipped, not_run, ConSan timeout
            return None, {"reason": "sanitizer_not_settled"}
        if not (self._enough_clean_repeats(legs) and self._trusted_compiled_evidence(legs)
                and self._no_structure_drift(legs)):
            return None, {"reason": "clean_not_proven"}  # or capped: Q8
        return finish(1.0 if cand.task.reference_kind == "frozen" else 0.5, clean=True)

    @staticmethod
    def _finish(base_: float, clean: bool, cand: Candidate) -> tuple[float, dict]:
        broke = -0.5 if (cand.task.parent_healthy and not clean) else 0.0
        # Which tokens: Q8. The share is taken first: Appendix B's left-to-right -0.05 * tokens / max
        # gives -0.05000000000000001 at tokens == max, and a reward of -1.5500000000000003.
        length = -0.05 * (cand.response_tokens / cand.max_new_tokens)
        return base_ + broke + length, {"base": base_, "broke_healthy": broke, "length": length}

    def _enough_clean_repeats(self, legs: dict[str, LegResult]) -> bool:
        return all(legs[n].repeats >= self.k for n in LADDER if legs[n].state != "not_run")

    @staticmethod
    def _trusted_compiled_evidence(legs: dict[str, LegResult]) -> bool:
        return legs["gate.compile"].state == "clean"

    @staticmethod
    def _no_structure_drift(legs: dict[str, LegResult]) -> bool:
        for leg in legs.values():
            if leg.reason == "structure_drift" or any(f.get("kind") == "structure_drift" for f in leg.findings):
                return False
        return True

    # --- early exit --------------------------------------------------------------------------------

    def decide(self, partial: dict[str, LegResult], cand: Candidate) -> Decision:
        known = {name: _normalise(name, leg) for name, leg in partial.items() if name in LEGS}

        def pending_san() -> tuple[str, ...]:
            return tuple(n for n in SAN_LEGS if n not in known)

        def confirmed():
            return [s for s in W if known[f"san.{s}"].state == "finding"]

        def sanitizer_base(found):
            in_round = [s for s in W if known[f"san.{s}"].state != "not_run"]
            return -0.5 * sum(W[s] for s in found) / sum(W[s] for s in in_round)

        def decided(pair: tuple[float, dict]) -> Decided:
            return Decided(pair[0], pair[1])

        def unless_finding(reason: str) -> Decision:
            if pending_san():  # a sanitizer still to come could confirm a finding
                return NeedMore(pending_san())
            found = confirmed()
            return decided(self._finish(sanitizer_base(found), False, cand)) if found else Unscored(reason)

        for g in GATES:
            name = f"gate.{g}"
            if name not in known:
                return NeedMore((name,))
            leg = known[name]
            if leg.state == "unsettled":
                return unless_finding(name)
            if leg.state == "fail":
                if leg.reason not in KNOWN_GATE_REASONS:
                    return Unscored("unknown_gate_reason")
                return decided(self._finish(-0.9 if leg.reason == "compile_error" else -1.0, False, cand))
        for name in LADDER:
            if name not in known:
                return NeedMore((name,))
            leg = known[name]
            if name.startswith("ref.") and leg.state == "unsettled":
                return unless_finding(name)
            if leg.state == "fail":
                if leg.reason in MISMATCH_REASONS:
                    return decided(self._finish(-0.7, False, cand))
                if leg.reason in FAULT_REASONS:
                    if name == "san.consan" and leg.reason == "timeout":
                        continue
                    if leg.bracket and leg.bracket["clean"]:
                        return decided(self._finish(-0.8, False, cand))
                    return unless_finding(f"{name}.control")
        # Every leg is known from here on, so this is compute's tail.
        reward, parts = self.compute(known, cand)
        return Unscored(parts["reason"]) if reward is None else Decided(reward, parts)

    @staticmethod
    def claims_clean(reward: float | None, parts: dict | None) -> bool:
        """A v11 reward is clean exactly when its base is positive (+1.0 or +0.5); every other base
        is at most -0.17."""
        return reward is not None and isinstance(parts, dict) and parts.get("base", 0.0) > 0.0
