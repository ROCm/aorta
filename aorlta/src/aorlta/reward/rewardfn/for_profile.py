"""The stand-in reward function a check profile names, built one way for a run (orchestrator/wiring.py) and for the
offline evaluators that re-score its records (selector/plan_eval.py)."""

from __future__ import annotations

from ..profiles import Profile


class UnsupportedRewardFn(ValueError):
    """The profile names a reward function that is not a stand-in one."""


def standin_rewardfn(profile: Profile):
    """standin.v11, the only stand-in reward function."""
    from .standin_v11 import StandinV11

    if profile.reward_fn == "standin.v11":
        return StandinV11(clean_repeats=profile.clean_repeats)
    raise UnsupportedRewardFn(f"profile {profile.name!r} is scored by {profile.reward_fn}, not a stand-in reward")
