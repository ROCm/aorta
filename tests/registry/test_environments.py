"""Tests for the environments registry."""

import pytest

from aorta.registry.errors import (
    RegistryCollisionError,
    RegistryError,
    UnknownEnvironmentError,
)
from aorta.registry.environments import get_environment, load_environments
from aorta.registry.types import Environment


def test_load_environments_includes_builtins(fake_env_eps):
    fake_env_eps([])
    result = load_environments()
    assert "local" in result
    assert "default" in result
    assert result["local"].source_package == "aorta"


def test_get_environment_returns_dataclass(fake_env_eps):
    fake_env_eps([])
    env = get_environment("local")
    assert isinstance(env, Environment)
    assert env.docker is None
    assert env.venv is None
    assert env.buck_target is None
    assert env.source_package == "aorta"


def test_get_environment_unknown_raises(fake_env_eps):
    fake_env_eps([])
    with pytest.raises(UnknownEnvironmentError) as exc:
        get_environment("not_a_real_env")
    msg = str(exc.value)
    assert "available:" in msg
    assert "plugin" in msg


def test_load_environments_discovers_plugin(fake_env_eps):
    fake_env_eps([
        ("nan-repro", {"docker": "rocm/private@sha256:abc"}, "fake_internal"),
    ])
    result = load_environments()
    assert result["nan-repro"].docker == "rocm/private@sha256:abc"
    assert result["nan-repro"].source_package == "fake_internal"


def test_collision_between_plugins_raises(fake_env_eps):
    fake_env_eps([
        ("shared", {"docker": "img:1"}, "plugin_a"),
        ("shared", {"docker": "img:2"}, "plugin_b"),
    ])
    with pytest.raises(RegistryCollisionError, match="plugin_a.*plugin_b"):
        load_environments()


def test_collision_plugin_vs_builtin_raises(fake_env_eps):
    fake_env_eps([("local", {}, "plugin_x")])
    with pytest.raises(RegistryCollisionError, match="aorta.*plugin_x"):
        load_environments()


def test_invalid_key_raises(fake_env_eps):
    fake_env_eps([("bad", {"rocm": "6.0"}, "plugin_x")])
    with pytest.raises(RegistryError, match="plugin_x.*rocm"):
        load_environments()


def test_non_string_key_raises_cleanly(fake_env_eps):
    fake_env_eps([("bad", {1: "x", "docker": "img"}, "plugin_x")])
    with pytest.raises(RegistryError, match="plugin_x.*non-string keys"):
        load_environments()


def test_non_string_value_raises(fake_env_eps):
    fake_env_eps([("bad", {"docker": 123}, "plugin_x")])
    with pytest.raises(RegistryError, match="plugin_x.*bad.*non-string"):
        load_environments()


def test_non_dict_payload_raises(fake_env_eps):
    fake_env_eps([("bad", "not-a-dict", "plugin_x")])
    with pytest.raises(RegistryError, match="plugin_x.*bad.*str"):
        load_environments()


# --- buck_target (#182) -----------------------------------------------------
#
# `buck_target` is the third baseline-recipe key, peering with `docker` and
# `venv`. The loader treats it identically to the existing keys: pass-through
# from the spec dict, no platform-side interpretation. These tests pin the
# field's round-trip through both the dataclass and the plugin loader, and
# confirm the field is also accepted alone (no docker, no venv) so a
# Buck-native workload can declare a pure-Buck environment.


def test_buck_target_field_round_trips_from_plugin(fake_env_eps):
    fake_env_eps([
        (
            "recom-repro-buck",
            {"buck_target": "//workloads/recom_repro:recom_repro"},
            "fake_internal",
        ),
    ])
    result = load_environments()
    env = result["recom-repro-buck"]
    assert env.buck_target == "//workloads/recom_repro:recom_repro"
    assert env.docker is None
    assert env.venv is None
    assert env.source_package == "fake_internal"


def test_buck_target_coexists_with_docker_for_buck_in_docker(fake_env_eps):
    # The Buck-inside-Docker tier from #37 (regression-gate buck-tier) sets
    # both keys -- platform must accept both without preferring one.
    fake_env_eps([
        (
            "buck-in-docker",
            {
                "docker": "rocm/private@sha256:abc",
                "buck_target": "//workloads/recom_repro:recom_repro",
            },
            "fake_internal",
        ),
    ])
    env = load_environments()["buck-in-docker"]
    assert env.docker == "rocm/private@sha256:abc"
    assert env.buck_target == "//workloads/recom_repro:recom_repro"


def test_environment_constructs_with_buck_target_only():
    # Direct dataclass construction -- pure-Buck environments don't need a
    # docker image or venv. Pins the field default and `asdict` shape so
    # downstream consumers (dispatcher's `_aorta_environment` round-trip,
    # `TrialResult.execution_env`) see the field.
    from dataclasses import asdict

    env = Environment(name="pure-buck", buck_target="//foo:bar")
    assert env.buck_target == "//foo:bar"
    assert env.docker is None
    assert env.venv is None
    assert asdict(env) == {
        "name": "pure-buck",
        "docker": None,
        "venv": None,
        "buck_target": "//foo:bar",
        "emulator": None,
        "mirage_profile": None,
        "source_package": "aorta",
    }


def test_buck_target_with_non_string_value_raises(fake_env_eps):
    # Mirrors `test_non_string_value_raises` for docker -- the validator is
    # a single allow-list and doesn't special-case keys, but pinning the
    # behaviour for the new key guards against future per-key validators
    # silently exempting it.
    fake_env_eps([("bad", {"buck_target": 123}, "plugin_x")])
    with pytest.raises(RegistryError, match="plugin_x.*bad.*non-string"):
        load_environments()
