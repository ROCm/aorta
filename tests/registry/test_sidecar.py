"""Tests for the JSON sidecar loader and its merge into the registry resolvers."""

import re

import pytest

from aorta.registry import (
    RegistryCollisionError,
    RegistryError,
    load_environments,
    load_mitigations,
    load_sidecar_environments,
    load_sidecar_mitigations,
)


# ---------- mitigations: merge into resolver ----------


def test_sidecar_mitigation_visible_in_merged_view(tmp_sidecar, fake_eps):
    fake_eps([])
    p = tmp_sidecar({"version": 1, "mitigations": {"my_flag": {"MY_ENV": "1"}}})
    mits = load_mitigations(extra_files=[p])
    assert "my_flag" in mits
    assert mits["my_flag"].env == {"MY_ENV": "1"}
    assert mits["my_flag"].source_package == f"sidecar:{p.name}"


def test_sidecar_keeps_builtins(tmp_sidecar, fake_eps):
    fake_eps([])
    p = tmp_sidecar({"version": 1, "mitigations": {"my_flag": {"X": "1"}}})
    mits = load_mitigations(extra_files=[p])
    assert "tf32_off" in mits
    assert mits["tf32_off"].source_package == "aorta"


def test_sidecar_collision_with_builtin(tmp_sidecar, fake_eps):
    fake_eps([])
    p = tmp_sidecar({"version": 1, "mitigations": {"tf32_off": {"X": "1"}}})
    with pytest.raises(
        RegistryCollisionError, match=f"aorta.*sidecar:{re.escape(p.name)}"
    ):
        load_mitigations(extra_files=[p])


def test_sidecar_collision_with_entry_point(tmp_sidecar, fake_eps):
    fake_eps([("foo", {"X": "1"}, "fake_plugin")])
    p = tmp_sidecar({"version": 1, "mitigations": {"foo": {"Y": "1"}}})
    with pytest.raises(
        RegistryCollisionError, match=f"fake_plugin.*sidecar:{re.escape(p.name)}"
    ):
        load_mitigations(extra_files=[p])


def test_sidecar_collision_between_two_files(tmp_sidecar, fake_eps):
    fake_eps([])
    a = tmp_sidecar({"version": 1, "mitigations": {"foo": {"X": "1"}}}, name="a.json")
    b = tmp_sidecar({"version": 1, "mitigations": {"foo": {"X": "2"}}}, name="b.json")
    with pytest.raises(
        RegistryCollisionError, match=r"sidecar:a\.json.*sidecar:b\.json"
    ):
        load_mitigations(extra_files=[a, b])


def test_two_sidecars_merge_when_disjoint(tmp_sidecar, fake_eps):
    fake_eps([])
    a = tmp_sidecar({"version": 1, "mitigations": {"foo": {"F": "1"}}}, name="a.json")
    b = tmp_sidecar({"version": 1, "mitigations": {"bar": {"B": "1"}}}, name="b.json")
    mits = load_mitigations(extra_files=[a, b])
    assert mits["foo"].source_package == "sidecar:a.json"
    assert mits["bar"].source_package == "sidecar:b.json"


def test_two_sidecars_same_basename_rejected(tmp_path, fake_eps):
    """Two paths with the same filename can't be disambiguated in `list`
    output or collision errors -- reject upfront with a clear message."""
    fake_eps([])
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    pa = dir_a / "shared.json"
    pb = dir_b / "shared.json"
    pa.write_text('{"version": 1, "mitigations": {"m1": {"X": "1"}}}', encoding="utf-8")
    pb.write_text('{"version": 1, "mitigations": {"m2": {"Y": "1"}}}', encoding="utf-8")
    with pytest.raises(RegistryError, match=r"share basename 'shared\.json'"):
        load_mitigations(extra_files=[pa, pb])


def test_same_sidecar_passed_twice_rejected(tmp_sidecar, fake_eps):
    """Same-file-twice falls out of the basename check -- same message
    works for both ambiguity cases."""
    fake_eps([])
    p = tmp_sidecar({"version": 1, "mitigations": {"m": {"X": "1"}}})
    with pytest.raises(RegistryError, match=f"share basename '{re.escape(p.name)}'"):
        load_mitigations(extra_files=[p, p])


def test_sidecar_collision_message_includes_full_path(tmp_sidecar, fake_eps):
    """Collision errors must surface the sidecar's full path so the
    operator knows which file to edit -- the basename tag alone is not
    enough when multiple sidecars are in play."""
    fake_eps([])
    p = tmp_sidecar({"version": 1, "mitigations": {"tf32_off": {"X": "1"}}})
    with pytest.raises(
        RegistryCollisionError, match=f"path: .*{re.escape(p.name)}"
    ):
        load_mitigations(extra_files=[p])


def test_extra_files_default_none_keeps_b3_behavior(fake_eps):
    fake_eps([])
    mits = load_mitigations()
    assert "tf32_off" in mits
    assert all(m.source_package == "aorta" for m in mits.values())


# ---------- environments: merge into resolver ----------


def test_sidecar_environment_visible(tmp_sidecar, fake_env_eps):
    fake_env_eps([])
    p = tmp_sidecar({
        "version": 1,
        "environments": {"my_local": {"docker": "myorg/x@sha256:abc"}},
    })
    envs = load_environments(extra_files=[p])
    assert envs["my_local"].docker == "myorg/x@sha256:abc"
    assert envs["my_local"].source_package == f"sidecar:{p.name}"


def test_sidecar_environment_collision_with_builtin(tmp_sidecar, fake_env_eps):
    fake_env_eps([])
    p = tmp_sidecar({"version": 1, "environments": {"local": {}}})
    with pytest.raises(
        RegistryCollisionError, match=f"aorta.*sidecar:{re.escape(p.name)}"
    ):
        load_environments(extra_files=[p])


def test_sidecar_environment_invalid_key(tmp_sidecar, fake_env_eps):
    fake_env_eps([])
    p = tmp_sidecar({"version": 1, "environments": {"bad": {"rocm": "6.0"}}})
    with pytest.raises(
        RegistryError,
        match=rf"{re.escape(p.name)}.*environments\.bad.*rocm",
    ):
        load_environments(extra_files=[p])


def test_sidecar_environment_with_buck_target_round_trips(
    tmp_sidecar, fake_env_eps
):
    # #182: `buck_target` is a peer of `docker` / `venv` in the sidecar
    # allow-list. Pin the round-trip so the sidecar payload and the
    # plugin payload accept the same schema (the two `_VALID_ENV_KEYS`
    # allow-lists are intentionally identical -- see sidecar.py).
    fake_env_eps([])
    p = tmp_sidecar({
        "version": 1,
        "environments": {
            "buck-env": {"buck_target": "//workloads/recom_repro:recom_repro"},
        },
    })
    envs = load_environments(extra_files=[p])
    assert envs["buck-env"].buck_target == "//workloads/recom_repro:recom_repro"
    assert envs["buck-env"].docker is None
    assert envs["buck-env"].source_package == f"sidecar:{p.name}"


# ---------- partial files (only one of mitigations / environments) ----------


def test_sidecar_with_only_environments(tmp_sidecar):
    p = tmp_sidecar({"version": 1, "environments": {"e1": {"venv": "/tmp/v"}}})
    assert load_sidecar_mitigations(p) == {}
    envs = load_sidecar_environments(p)
    assert envs["e1"].venv == "/tmp/v"


def test_sidecar_with_only_mitigations(tmp_sidecar):
    p = tmp_sidecar({"version": 1, "mitigations": {"m1": {"K": "v"}}})
    assert load_sidecar_environments(p) == {}
    mits = load_sidecar_mitigations(p)
    assert mits["m1"].env == {"K": "v"}


# ---------- schema validation ----------


def test_missing_version_rejected(tmp_sidecar):
    p = tmp_sidecar({"mitigations": {}})
    with pytest.raises(RegistryError, match="missing required key 'version'"):
        load_sidecar_mitigations(p)


def test_wrong_version_rejected(tmp_sidecar):
    p = tmp_sidecar({"version": 2, "mitigations": {}})
    with pytest.raises(RegistryError, match="unsupported version"):
        load_sidecar_mitigations(p)


def test_unknown_top_level_key_rejected(tmp_sidecar):
    p = tmp_sidecar({"version": 1, "stuff": {}})
    with pytest.raises(RegistryError, match="unknown top-level keys.*stuff"):
        load_sidecar_mitigations(p)


def test_non_string_env_value_rejected(tmp_sidecar):
    p = tmp_sidecar({"version": 1, "mitigations": {"foo": {"K": 5}}})
    with pytest.raises(
        RegistryError,
        match=rf"{re.escape(p.name)}.*mitigations\.foo\.K.*must be string.*int",
    ):
        load_sidecar_mitigations(p)


def test_mitigation_payload_must_be_object(tmp_sidecar):
    p = tmp_sidecar({"version": 1, "mitigations": {"foo": "not-a-dict"}})
    with pytest.raises(RegistryError, match=r"mitigations\.foo.*object of env vars"):
        load_sidecar_mitigations(p)


def test_environment_value_must_be_string_or_null(tmp_sidecar):
    p = tmp_sidecar({"version": 1, "environments": {"e": {"docker": 123}}})
    with pytest.raises(
        RegistryError, match=r"environments\.e\.docker.*string or null"
    ):
        load_sidecar_environments(p)


def test_invalid_json_reports_path(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(RegistryError, match=rf"{re.escape(p.name)}.*invalid JSON"):
        load_sidecar_mitigations(p)


def test_top_level_not_object_rejected(tmp_path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(RegistryError, match="top-level must be a JSON object"):
        load_sidecar_mitigations(p)


def test_missing_file_reports_path(tmp_path):
    p = tmp_path / "nonexistent.json"
    with pytest.raises(RegistryError, match="cannot read file"):
        load_sidecar_mitigations(p)
