"""The reward function a check profile names, built one way for a run (orchestrator/wiring.py) and for the
offline evaluators that re-score its records (selector/plan_eval.py): standin.v11, or one a plugin registered
(plugins.py)."""

from __future__ import annotations

from ..profiles import Profile


class UnsupportedRewardFn(ValueError):
    """The profile names a reward function that is neither standin.v11 nor registered by a plugin."""


def standin_rewardfn(profile: Profile):
    """standin.v11, or the registered reward function the profile names, built for it."""
    from ... import plugins

    if profile.reward_fn == "standin.v11":
        from .standin_v11 import StandinV11

        return StandinV11(clean_repeats=profile.clean_repeats)
    factory = plugins.REWARD_FNS.get(profile.reward_fn)
    if factory is not None:
        fn = factory(profile)
        if getattr(fn, "name", None) != profile.reward_fn:
            raise UnsupportedRewardFn(f"the reward function registered as {profile.reward_fn!r} is named "
                                      f"{getattr(fn, 'name', None)!r}")
        return fn
    raise UnsupportedRewardFn(f"profile {profile.name!r} is scored by {profile.reward_fn}, which is neither standin.v11 "
                              f"nor registered by a plugin")
