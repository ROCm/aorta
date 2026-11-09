"""Check profiles load strictly: unknown keys, legs in two stages, and requirements that do not come
first are refused; the shipped profiles load."""

import pytest

from aorlta.config import ConfigError
from aorlta.reward.profiles import load_profile, profile_from_dict


@pytest.mark.parametrize("name, reward_fn", [("fake", "standin.v11"), ("standin-static", "standin.v11"),
                                             ("standin-gpu", "standin.v11")])
def test_the_shipped_profiles_load(name, reward_fn):
    p = load_profile(name)
    assert p.name == name and p.reward_fn == reward_fn and p.legs


def base(**over):
    d = {"name": "fake", "reward_fn": "standin.v11",
         "pools": {"p": {"resource": "local_cpu", "adapter": "fake", "timeout": "gate"}},
         "stages": [{"name": "S0", "pool": "p", "legs": ["gate.parse"]},
                    {"name": "S1", "pool": "p", "legs": ["gate.compile"]}]}
    d.update(over)
    return d


@pytest.mark.parametrize("data, message", [
    (base(extra=1), "unknown key"),
    (base(stages=[{"name": "S0", "pool": "p", "legs": ["gate.parse", "gate.parse"]}]), "is in stages"),
    (base(requires={"gate.parse": ["gate.compile"]}), "does not come before it"),
    (base(requires={"gate.nothing": ["gate.parse"]}), "which no stage runs"),
    (base(stages=[{"name": "S0", "pool": "q", "legs": ["gate.parse"]}]), "unknown pool"),
    (base(name="something"), "name must be one of"),
    (base(pools={"p": {"resource": "local_cpu", "adapter": "fake", "timeout": "gate", "batched": True}}), "exactly one leg"),
])
def test_bad_profiles_are_refused(data, message):
    if message == "exactly one leg":
        data["stages"] = [{"name": "S0", "pool": "p", "legs": ["gate.parse", "gate.compile"]}]
    with pytest.raises(ConfigError, match=message):
        profile_from_dict(data)
