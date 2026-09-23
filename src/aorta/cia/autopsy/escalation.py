"""Which number decides whether Autopsy spends four hours on a production sweep.

``run_autopsy`` escalates when the router recommends ``aorta sweep run`` and the
confidence is under 0.85. That 0.85 was never measured. It was chosen against
the only two things that have ever produced a confidence here, and both of them
are the same hand-written curve: ``merge_watchdog_matrix`` returns 0.62 for a
bare watchdog NaN and the matrix classifier tops out at 0.95, while the router's
prompt asks a model in English for ``~0.62``, ``~0.55``, ``>= 0.9`` and ``0.0``.
The cutoff works because both ends of the comparison were authored together, by
the same person, on the same afternoon.

Phase 4 puts a calibrated encoder behind the same field, and a Laya probability
is a different distribution wearing the same name and the same range: trained
against a strictly proper scoring rule, then temperature-fitted per (question
type, option count). Carrying 0.85 across that change is the single edit in
``docs/plans/laya-system1-integration.md`` most likely to make Autopsy worse
than it is today, and it would do it in silence. Both numbers are floats in
[0, 1], both read as plausible in a report, and the only symptom is sweeps that
stop happening or sweeps that should not -- neither of which anybody notices
against a four-hour job that was always flaky.

There is no honest way to re-derive the cutoff in this change. Phase 1 has not
run, no Laya checkpoint has been scored against an Autopsy bundle, and no
temperature fit exists to make one of its probabilities mean what it says. So
this module does not guess a replacement and does not quietly keep the old one.
It makes the pairing the code has always relied on explicit -- a confidence
travels with the source that produced it, and a threshold is registered for the
sources it was chosen against -- and a source with no threshold of its own does
not get to borrow one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The rule-based adapters: ``classify_matrix``, ``classify_sanitizer``,
#: ``classify_rocgdb`` and the merge functions above them. Deterministic, and
#: every number they return is a literal somebody typed.
ADAPTER_RULES = "adapter_rules"

#: The router's ``confidence`` output field -- a float an LLM wrote because a
#: prompt asked it to. Nothing trained it to be a probability.
LLM_SELF_REPORT = "llm_self_report"

#: A calibrated head's probability for the category it chose. The one source
#: here that is a probability in the sense the word usually carries, and the one
#: source :data:`ESCALATION_THRESHOLD` was *not* chosen against.
LAYA = "laya"

#: The only ``next_probe`` that can escalate. The router may also answer
#: ``'none'``, and then nothing below is consulted.
SWEEP_PROBE = "aorta sweep run"

#: The escalation cutoff, unchanged, and now named where its provenance can be
#: stated beside it rather than inferred from a comment three modules away.
ESCALATION_THRESHOLD = 0.85

#: The sources :data:`ESCALATION_THRESHOLD` was chosen against, and therefore
#: the only ones it may be applied to.
#:
#: Both are on the same scale by construction: the prompt's ``~0.62`` is the
#: same 0.62 ``merge_watchdog_matrix`` hard-codes, because the prompt was
#: written by reading the adapters. That shared scale is what the cutoff was
#: fitted to, and it is the whole of its justification.
THRESHOLD_SOURCES = frozenset({ADAPTER_RULES, LLM_SELF_REPORT})


class UncalibratedThresholdError(ValueError):
    """A confidence was offered to a cutoff that was not derived against it.

    A programming error rather than a runtime condition: the caller is meant to
    say which source it is handing over, and every source either has a threshold
    registered for it or has one supplied explicitly. Raising is right because
    the alternative -- picking whichever cutoff is nearest and carrying on -- is
    exactly the silent failure this module exists to prevent.
    """


@dataclass(frozen=True)
class Confidence:
    """A confidence figure and the thing that produced it, travelling together.

    Autopsy has always had more than one source for this number -- the adapters
    answer when the router is unreachable, the router answers when it is not --
    and the report has never said which one it was. That was survivable while
    the two shared a scale. It stops being survivable the moment a third source
    arrives with a different one, so the pairing becomes a type rather than a
    convention.

    *detail* names the specific producer where there is one to name: the
    checkpoint identity for :data:`LAYA`, which is rule 2 of Decision 22 in
    ``docs/laya-packaging.md`` -- a report gets copied into a ticket and read on
    its own, and a verdict that cannot say which weights produced it cannot be
    compared against the next one.

    *caveat* is the rest of rule 2: a probability is a function of the
    checkpoint *and* the temperature fit applied to it, and the fit is where
    this one is weakest. Autopsy's category question offers eleven options, so
    it falls in the library's ``choice:11+`` bucket -- the one whose shipped
    temperature is low enough to sharpen logits rather than soften them, which
    is why 0.3.5 clamps it and warns. The clamp stops the sharpening; it does
    not make the bucket calibrated.

    **The caveat is disclosed here and deliberately does not gate.** Refusing to
    apply a *threshold* derived against another source is this module's job and
    is decided by :attr:`source`. Whether the source's own fit was trustworthy
    is a different fact, and an operator who has derived a cutoff against a
    clamped checkpoint derived it against the behaviour they will get -- so
    acting on the caveat here would override a measurement with a rule of thumb.
    It is carried so that nobody downstream reads the number as calibrated.
    """

    value: float
    source: str
    detail: str = ""
    caveat: str = ""

    def describe(self) -> str:
        """One line for an operator watching stderr, and for the report's reason.

        The caveat is a sentence fragment rather than a structured field at this
        end because it has to survive being read: this string lands in
        ``report["escalation"]["reason"]``, which is prose a person skims.
        :meth:`as_report_fields` carries the same fact where a program looks.
        """
        named = f"{self.source}: {self.detail}" if self.detail else self.source
        return f"confidence={self.value:.2f} ({named}){self.caveat}"

    def as_report_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {"source": self.source, "value": round(self.value, 3)}
        if self.detail:
            fields["detail"] = self.detail
        if self.caveat:
            fields["caveat"] = self.caveat
        return fields


@dataclass(frozen=True)
class Escalation:
    """The decision, the number it was made on, and why that was the number.

    *gated_on* is separate from the confidence the report carries because they
    are allowed to differ, and the case where they differ is the point of this
    module: Laya may supply the verdict's confidence while the cutoff is still
    applied to the adapters' figure, because that is the figure the cutoff was
    chosen against. A reader who is told only the outcome cannot audit that, and
    a reader who is told only the reported confidence would audit it wrongly.
    """

    escalate: bool
    gated_on: Confidence
    threshold: float | None
    reason: str

    def as_report_fields(self) -> dict[str, Any]:
        return {
            "escalated": self.escalate,
            "threshold": self.threshold,
            "gated_on": self.gated_on.as_report_fields(),
            "reason": self.reason,
        }


def decide(
    next_probe: str,
    reported: Confidence,
    *,
    rule_based: Confidence,
    laya_threshold: float | None = None,
) -> Escalation:
    """Whether to run the production sweep, and on whose number.

    *reported* is the confidence the report will carry. *rule_based* is the
    adapters' own figure, which exists for every bundle whether or not any model
    ran and is therefore always available as the thing the cutoff was fitted to.
    *laya_threshold* is a cutoff an operator has derived for a Laya probability;
    ``None`` means nobody has, which is the state of the world in this change.

    The three branches, and why they are in this order:

    1. A source :data:`ESCALATION_THRESHOLD` was chosen against is compared
       against it. This is what Autopsy has always done and it is unchanged.
    2. A Laya probability with a derived threshold is compared against *that*
       threshold. This is the branch Phase 1 is supposed to unlock, and it is
       here now so that the measurement has somewhere to land rather than
       arriving as a second patch to this function.
    3. A Laya probability with no derived threshold does not get compared at
       all. The escalation still happens, on the adapters' figure, which is a
       real number about this bundle on the scale the cutoff was fitted to --
       so the control flow is the one Autopsy had before Laya was switched on,
       and turning the tier on changes what the report *says* without changing
       what the pipeline *does*. That asymmetry is deliberate: the category is
       the thing Phase 4 set out to improve, and the sweep is the thing a wrong
       threshold would break.
    """
    if next_probe != SWEEP_PROBE:
        return Escalation(
            escalate=False,
            gated_on=reported,
            threshold=None,
            reason=f"the router recommended {next_probe or 'no probe'!r}, so there is nothing to escalate",
        )

    if reported.source in THRESHOLD_SOURCES:
        return _below(reported, ESCALATION_THRESHOLD, "the cutoff was chosen against this source")

    if reported.source != LAYA:
        raise UncalibratedThresholdError(
            f"no escalation threshold is registered for confidence source {reported.source!r}. "
            f"Register one in THRESHOLD_SOURCES if it shares a scale with "
            f"{sorted(THRESHOLD_SOURCES)}, or pass a threshold derived against it."
        )

    if laya_threshold is not None:
        return _below(
            reported,
            laya_threshold,
            "an operator supplied a cutoff derived against this checkpoint",
        )

    if rule_based.source not in THRESHOLD_SOURCES:
        raise UncalibratedThresholdError(
            f"the fallback confidence came from {rule_based.source!r}, which has no "
            "registered threshold either, so there is nothing left to gate on"
        )
    return _below(
        rule_based,
        ESCALATION_THRESHOLD,
        f"no cutoff has been derived for {reported.source}, so the escalation stays on the "
        "rule-based figure the 0.85 was chosen against",
    )


def _below(confidence: Confidence, threshold: float, why: str) -> Escalation:
    return Escalation(
        escalate=confidence.value < threshold,
        gated_on=confidence,
        threshold=threshold,
        reason=f"{confidence.describe()} against {threshold:.2f} -- {why}",
    )


__all__ = [
    "ADAPTER_RULES",
    "ESCALATION_THRESHOLD",
    "LAYA",
    "LLM_SELF_REPORT",
    "SWEEP_PROBE",
    "THRESHOLD_SOURCES",
    "Confidence",
    "Escalation",
    "UncalibratedThresholdError",
    "decide",
]
