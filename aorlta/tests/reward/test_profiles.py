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


# --- where the check tools are ----------------------------------------------------------------------------------------

TOOL_VARS = ("AORLTA_ROCM_ROOT", "AORLTA_ROCJITSU_ROOT", "AORLTA_AORTA_ROOT", "AORLTA_LIBSTDCXX_SHIM")


@pytest.fixture
def no_tool_roots(monkeypatch):
    for name in TOOL_VARS:
        monkeypatch.delenv(name, raising=False)


def test_the_shipped_profiles_default_their_tools_under_opt(no_tool_roots):
    a = load_profile("standin-gpu").adapters
    assert a["standin.hipcc"] == {"hipcc": "/opt/rocm/bin/hipcc", "readelf": "/opt/rocm/llvm/bin/llvm-readelf"}
    assert a["standin.rj_waitcheck"] == {"binary": "/opt/rocjitsu/bin/rj_waitcheck",
                                         "ld_library_path": "/opt/libstdcxx-shim", "aorta_src": "/opt/aorta/src"}
    assert a["standin.aorta_consan"]["hook"] == "/opt/rocjitsu/lib/librocjitsu_dbi_hooks.so"
    assert a["standin.aorta_asan"]["control_source"] == "/opt/aorta/recipes/sanitizers/fixtures/repro/asan_heap_overflow.hip"
    assert load_profile("standin-static").adapters["standin.rj_waitcheck"] == a["standin.rj_waitcheck"]


def test_a_site_sets_where_its_tools_are(no_tool_roots, monkeypatch):
    monkeypatch.setenv("AORLTA_ROCM_ROOT", "/site/rocm-7.2/")
    monkeypatch.setenv("AORLTA_ROCJITSU_ROOT", "/site/rocjitsu")
    monkeypatch.setenv("AORLTA_AORTA_ROOT", "/site/aorta")
    monkeypatch.setenv("AORLTA_LIBSTDCXX_SHIM", "/site/shim")
    for name in ("standin-gpu", "standin-static"):
        settings = [v for s in load_profile(name).adapters.values() for v in s.values()]
        assert settings and all(v.startswith("/site/") and "${" not in v and "//" not in v for v in settings), name
    a = load_profile("standin-gpu").adapters
    assert a["standin.hipcc"]["hipcc"] == "/site/rocm-7.2/bin/hipcc"
    assert a["standin.aorta_consan"] == {"hook": "/site/rocjitsu/lib/librocjitsu_dbi_hooks.so",
                                         "aorta_src": "/site/aorta/src",
                                         "control_source": "/site/aorta/recipes/sanitizers/fixtures/repro/consan_lds_race.hip"}
    monkeypatch.setenv("AORLTA_ROCM_ROOT", "")  # empty is unset
    assert load_profile("standin-static").adapters["standin.hipcc"]["hipcc"] == "/opt/rocm/bin/hipcc"


def test_a_profile_names_only_the_tool_roots(no_tool_roots):
    with pytest.raises(ConfigError, match=r"\$\{HOME\} is not one of AORLTA_ROCM_ROOT"):
        profile_from_dict(base(adapters={"standin.hipcc": {"hipcc": "${HOME}/bin/hipcc"}}))
    p = profile_from_dict(base(adapters={"x": {"paths": ["${AORLTA_AORTA_ROOT}/a", "$plain"], "n": 3}}))
    assert p.adapters == {"x": {"paths": ["/opt/aorta/a", "$plain"], "n": 3}}
