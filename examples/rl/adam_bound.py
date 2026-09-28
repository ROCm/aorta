"""How far Adam can move one element, and the ceiling that bounds it.

The derivation is in ``verify_checkpoint_delta.py``, which checks a checkpoint
pair against :func:`optimiser_bound`. It lives here, with no dependency beyond
the standard library, so the trainer can refuse a run that ceiling cannot bound
without importing the verifier: the verifier is not part of what scores an
episode, and ``eval_episodes.scorer_files`` follows every import of the
trainer.
"""

from __future__ import annotations

import math

#: The multiple of ``lr * steps`` the ceiling allows.
SLACK = 4.0


def adam_step_ceiling(steps: int, beta1: float = 0.9, beta2: float = 0.999) -> float:
    """``sum_{t<=steps} c(t)``: Adam's worst-case displacement in units of ``lr``.

    See ``verify_checkpoint_delta.py`` for the derivation. Returned in units of
    ``lr`` so it can be compared directly with ``SLACK * steps``, and ``inf``
    once it passes the float range: ``b1^2 / b2 > 1`` grows it geometrically,
    and ``ratio ** t`` raises ``OverflowError`` there rather than returning
    ``inf``.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    ratio = beta1 * beta1 / beta2
    scale = (1.0 - beta1) / math.sqrt(1.0 - beta2)
    total = 0.0
    geometric = 0.0
    for t in range(1, steps + 1):
        try:
            geometric += ratio ** (t - 1)
        except OverflowError:
            return math.inf
        total += (
            scale * math.sqrt(geometric) * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
        )
    return total


def optimiser_bound(lr: float, steps: int, beta1: float = 0.9, beta2: float = 0.999) -> float:
    """``SLACK * lr * steps``, refused where it would not bound Adam.

    Raises ``ValueError`` when the worst-case Adam displacement over ``steps``
    exceeds the ceiling, because a ceiling below what the optimiser can
    legitimately do would report healthy tensors as damaged.
    """
    if not (math.isfinite(lr) and lr > 0):
        raise ValueError("lr must be a finite number > 0")
    if not (math.isfinite(beta1) and 0.0 <= beta1 < 1.0):
        raise ValueError("beta1 must be finite and in [0, 1)")
    if not (math.isfinite(beta2) and 0.0 < beta2 < 1.0):
        # The bound divides by beta2: it needs a positive second-moment decay.
        raise ValueError("beta2 must be finite and in (0, 1)")
    worst = adam_step_ceiling(steps, beta1, beta2)
    if worst > SLACK * steps:
        raise ValueError(
            f"{SLACK:g} x lr x steps is not a sound ceiling for {steps} Adam steps at "
            f"betas ({beta1}, {beta2}): the worst case is {worst / steps:.3f} x lr per "
            f"step. Verify shorter intervals, or chained links separately."
        )
    return SLACK * lr * steps
