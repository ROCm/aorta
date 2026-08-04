"""Unit tests for ``aorta.probe.redaction`` (issue #188 Phase 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from aorta.bundle.errors import RedactionError
from aorta.probe.redaction import (
    RedactingRedactor,
    RedactionCfg,
    _line_windows,
    parse_redaction,
    scrub_env_keys,
    scrub_text,
)
from aorta.probe.sandbox import MAX_LOG_BYTES
from aorta.triage.recipe import RecipeSchemaError, load_recipe

FIXTURES = Path(__file__).parent / "fixtures"


def _mark_run_root(run_dir: Path) -> None:
    run_dir.mkdir(exist_ok=True)
    (run_dir / "host_env.json").write_text("{}", encoding="utf-8")
    (run_dir / "recipe.resolved.yaml").write_text(
        "schema_version: 1\n",
        encoding="utf-8",
    )


def _result_json_paths(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    _mark_run_root(run_dir)
    src = run_dir / "none-none" / "trial_0" / "result.json"
    src.parent.mkdir(parents=True, exist_ok=True)
    dst = tmp_path / "out" / "none-none" / "trial_0" / "result.json"
    return src, dst


def test_env_key_glob():
    env = {
        "AWS_ACCESS_KEY_ID": "AKIA",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "PATH": "/usr/bin",
        "SAFE": "ok",
    }
    scrubbed, removed = scrub_env_keys(env, ("AWS_*",))
    assert removed == 2
    assert scrubbed == {"PATH": "/usr/bin", "SAFE": "ok"}


def test_env_key_glob_case_sensitive():
    env = {"aws_secret_key": "lower", "AWS_SECRET_KEY": "upper"}
    scrubbed, removed = scrub_env_keys(env, ("AWS_*",))
    assert removed == 1
    assert "aws_secret_key" in scrubbed
    assert "AWS_SECRET_KEY" not in scrubbed


def test_path_rewrite():
    text = (
        "a /home/user/a/data/file.txt\n"
        "b /home/user/a/data/file.txt\n"
        "c /home/user/b/other.log\n"
        "d /opt/third/path\n"
        "e /opt/third/path\n"
    )
    out, paths, v4, v6 = scrub_text(text, scrub_paths=True, scrub_ip_addresses=False)
    assert paths == 5
    assert "<PATH:0>" in out
    assert "<PATH:1>" in out
    assert "<PATH:2>" in out
    assert "/home/user" not in out
    assert v4 == 0 and v6 == 0


def test_path_rewrite_double_slash_and_file_url():
    """A path whose leading '/' follows another '/' must still scrub (Copilot).

    The negative lookbehind once included '/', which skipped the path inside
    `file:///home/user/...` and protocol-relative `//host/home/user/...`,
    leaking absolute paths the scrubber documents it removes.
    """
    text = "u file:///home/user/secret\np //host/home/user/secret\n"
    out, paths, v4, v6 = scrub_text(text, scrub_paths=True, scrub_ip_addresses=False)
    assert paths == 2
    assert "/home/user/secret" not in out
    assert "<PATH:0>" in out and "<PATH:1>" in out


def test_parse_redaction_non_string_key_fails_closed():
    """A non-string mapping key must raise RecipeSchemaError, not TypeError.

    YAML allows `1: x`; the unknown-key error sorts the offending keys, and a
    mixed str/int set would raise TypeError sorting -- escaping as an
    unhandled exception instead of the recipe schema error (Copilot).
    """
    with pytest.raises(RecipeSchemaError, match="unknown keys"):
        parse_redaction({1: "x", "scrub_paths": True})


def test_path_rewrite_no_reverse_mapping_persisted(tmp_path: Path):
    cfg = RedactionCfg(scrub_paths=True)
    redactor = RedactingRedactor(cfg)
    src = tmp_path / "stdout.log"
    dst = tmp_path / "out" / "stdout.log"
    src.write_text("loaded /secret/path/file\n", encoding="utf-8")
    counts = redactor.scrub_file(src, dst)
    assert counts.paths_rewritten == 1
    bundled = dst.read_text(encoding="utf-8")
    assert "/secret/path/file" not in bundled
    assert "<PATH:0>" in bundled
    assert "mapping" not in bundled.lower()


def test_ip_rewrite():
    text = "host 192.168.0.1 and 2001:db8::1 here"
    out, paths, v4, v6 = scrub_text(
        text,
        scrub_paths=False,
        scrub_ip_addresses=True,
    )
    assert paths == 0
    assert v4 == 1
    assert v6 == 1
    assert "<IPV4:0>" in out
    assert "<IPV6:0>" in out
    assert "192.168.0.1" not in out


@pytest.mark.parametrize(
    "snippet",
    [
        "loopback ::1 ok",
        "fe80 link fe80::1 ok",
        "bracket [::1] ok",
        "url http://[::1]:8080/path",
    ],
)
def test_ipv6_compressed_and_bracketed_forms(snippet: str):
    out, _, v4, v6 = scrub_text(
        snippet,
        scrub_paths=False,
        scrub_ip_addresses=True,
    )
    assert v4 == 0
    assert v6 >= 1
    assert "::1" not in out
    assert "[::1]" not in out


def test_redactor_kind_string():
    assert RedactingRedactor(RedactionCfg()).kind == "probe.v1"


@pytest.mark.timeout(15)
def test_redaction_dos_bound():
    """A 10 MiB slash run must not blow up regex CPU (DoS guard).

    Enforced with pytest-timeout (a declared dev dep) rather than a measured
    ``perf_counter() < 5.0`` assertion: a wall-clock assert is flaky under CI
    load, and -- more importantly -- if the regex DID catastrophically
    backtrack the call would never return, so the post-hoc assert would never
    run. The timeout marker bounds the whole test even on a true hang. The
    15 s budget is generous headroom over the sub-second normal runtime.
    """
    blob = "/" * MAX_LOG_BYTES
    result = scrub_text(blob, scrub_paths=True, scrub_ip_addresses=False)
    assert len(result) == 4


def test_redacting_redactor_scrubs_result_json_env(tmp_path: Path):
    cfg = RedactionCfg(
        scrub_env_keys=("AWS_*", "HOME", "USER"),
        scrub_paths=True,
    )
    redactor = RedactingRedactor(cfg)
    src, dst = _result_json_paths(tmp_path)
    src.write_text(
        json.dumps(
            {
                "verdict": "pass",
                "env": {"AWS_TOKEN": "x", "HIP_VISIBLE_DEVICES": "0"},
                "argv": ["/home/user/train.py"],
            }
        ),
        encoding="utf-8",
    )
    counts = redactor.scrub_file(src, dst)
    assert counts.env_keys_removed == 1
    doc = json.loads(dst.read_text(encoding="utf-8"))
    assert "AWS_TOKEN" not in doc["env"]
    assert "<PATH:" in doc["argv"][0]


def test_result_json_env_values_scrubbed(tmp_path: Path):
    """Retained env values are path/IP-scrubbed, not just key-filtered.

    Removing matching keys left a kept key's value (e.g. a *_PATH var)
    leaking an absolute path/IP into the bundle even with scrub_paths on
    (Copilot review). result.json env now matches host_env.json.
    """
    cfg = RedactionCfg(
        scrub_env_keys=("AWS_*",),
        scrub_paths=True,
        scrub_ip_addresses=True,
    )
    redactor = RedactingRedactor(cfg)
    src, dst = _result_json_paths(tmp_path)
    src.write_text(
        json.dumps(
            {
                "verdict": "pass",
                "env": {
                    "AWS_TOKEN": "drop-me",
                    "LD_LIBRARY_PATH": "/home/customer/lib",
                    "MASTER_ADDR": "192.168.1.42",
                },
            }
        ),
        encoding="utf-8",
    )
    counts = redactor.scrub_file(src, dst)
    doc = json.loads(dst.read_text(encoding="utf-8"))
    assert "AWS_TOKEN" not in doc["env"]
    assert counts.env_keys_removed == 1
    assert "/home/customer/lib" not in doc["env"]["LD_LIBRARY_PATH"]
    assert "192.168.1.42" not in doc["env"]["MASTER_ADDR"]
    assert counts.paths_rewritten >= 1
    assert counts.ips_rewritten >= 1


def test_result_json_nested_envsnapshot_preserves_types_and_scrubs_env_vars(
    tmp_path: Path,
):
    """TrialResult.env is a nested EnvSnapshot, not a flat process env.

    The old redactor treated every ``env`` dict as flat, stringifying ``rocm``
    and ``env_vars`` and then missing scrub patterns inside the string.
    """
    cfg = RedactionCfg(scrub_env_keys=("ROCBLAS_LOG_*",), scrub_paths=True)
    redactor = RedactingRedactor(cfg)
    src, dst = _result_json_paths(tmp_path)
    src.write_text(
        json.dumps(
            {
                "verdict": "pass",
                "env": {
                    "schema_version": "1.15",
                    "partial": False,
                    "rocm": {"version": "7.0.2"},
                    "env_vars": {
                        "ROCBLAS_LOG_PATH": "/home/customer/private.log",
                        "HIP_VISIBLE_DEVICES": "0",
                        "TENSILE_STREAMK5_FORCE_MODE": None,
                    },
                    "miopen_catalog": {
                        "env_overrides": {
                            "ROCBLAS_LOG_PATH": "/home/customer/duplicate.log",
                            "KEEP": "1",
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    counts = redactor.scrub_file(src, dst)
    doc = json.loads(dst.read_text(encoding="utf-8"))

    assert counts.env_keys_removed == 2
    assert doc["env"]["rocm"] == {"version": "7.0.2"}
    assert doc["env"]["partial"] is False
    assert "ROCBLAS_LOG_PATH" not in doc["env"]["env_vars"]
    assert doc["env"]["env_vars"]["HIP_VISIBLE_DEVICES"] == "0"
    assert doc["env"]["env_vars"]["TENSILE_STREAMK5_FORCE_MODE"] is None
    assert doc["env"]["miopen_catalog"]["env_overrides"] == {"KEEP": "1"}


@pytest.mark.parametrize("name", ["host_env.json", "env.json", "matrix.json"])
def test_json_env_var_mappings_obey_scrub_env_keys(name: str, tmp_path: Path):
    cfg = RedactionCfg(scrub_env_keys=("ROCBLAS_LOG_*",))
    redactor = RedactingRedactor(cfg)
    _mark_run_root(tmp_path)
    src = (
        tmp_path / "environments" / "rocm" / "env.json"
        if name == "env.json"
        else tmp_path / name
    )
    src.parent.mkdir(parents=True, exist_ok=True)
    dst = tmp_path / "out" / src.relative_to(tmp_path)
    payload = (
        {
            "cells": [
                {
                    "resolved_env_vars": {
                        "ROCBLAS_LOG_PATH": "/customer/private.log",
                        "KEEP": "1",
                    },
                    "extra_env": {
                        "ROCBLAS_LOG_PATH": "/customer/duplicate.log",
                        "KEEP": "1",
                    },
                }
            ]
        }
        if name == "matrix.json"
        else {
            "schema_version": "1.15",
            "env_vars": {
                "ROCBLAS_LOG_PATH": "/customer/private.log",
                "KEEP": "1",
            },
            "miopen_catalog": {
                "env_overrides": {
                    "ROCBLAS_LOG_PATH": "/customer/duplicate.log",
                    "KEEP": "1",
                }
            },
        }
    )
    src.write_text(json.dumps(payload), encoding="utf-8")

    counts = redactor.scrub_file(src, dst)
    doc = json.loads(dst.read_text(encoding="utf-8"))
    env = doc["cells"][0]["resolved_env_vars"] if name == "matrix.json" else doc["env_vars"]

    assert counts.env_keys_removed == 2
    assert env == {"KEEP": "1"}
    duplicate = (
        doc["cells"][0]["extra_env"]
        if name == "matrix.json"
        else doc["miopen_catalog"]["env_overrides"]
    )
    assert duplicate == {"KEEP": "1"}


def test_isolated_environment_placeholder_scrubs_descriptor_env(tmp_path: Path):
    _mark_run_root(tmp_path)
    src = tmp_path / "environments" / "isolated" / "env.json"
    src.parent.mkdir(parents=True)
    dst = tmp_path / "out" / "environments" / "isolated" / "env.json"
    src.write_text(
        json.dumps(
            {
                "name": "isolated",
                "snapshot_captured": False,
                "descriptor": {
                    "docker": "image",
                    "env": {
                        "AWS_SECRET_ACCESS_KEY": "secret",
                        "KEEP": "1",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    counts = RedactingRedactor(
        RedactionCfg(scrub_env_keys=("AWS_*",))
    ).scrub_file(src, dst)
    doc = json.loads(dst.read_text(encoding="utf-8"))

    assert counts.env_keys_removed == 1
    assert doc["descriptor"]["env"] == {"KEEP": "1"}


def test_result_json_mixed_type_flat_env_still_scrubs_keys(tmp_path: Path):
    src, dst = _result_json_paths(tmp_path)
    src.write_text(
        json.dumps(
            {
                "env": {
                    "AWS_SECRET_ACCESS_KEY": "secret",
                    "RETRY_COUNT": 3,
                }
            }
        ),
        encoding="utf-8",
    )

    counts = RedactingRedactor(
        RedactionCfg(scrub_env_keys=("AWS_*",))
    ).scrub_file(src, dst)
    doc = json.loads(dst.read_text(encoding="utf-8"))

    assert counts.env_keys_removed == 1
    assert doc["env"] == {"RETRY_COUNT": 3}


@pytest.mark.parametrize(
    "relative",
    [
        Path("inline_environments.sidecar.json"),
        Path("sidecars") / "customer.json",
    ],
)
def test_environment_sidecars_scrub_environment_and_mitigation_maps(
    relative: Path, tmp_path: Path
):
    _mark_run_root(tmp_path)
    src = tmp_path / relative
    src.parent.mkdir(parents=True, exist_ok=True)
    dst = tmp_path / "out" / relative
    src.write_text(
        json.dumps(
            {
                "version": 1,
                "environments": {
                    "customer": {
                        "docker": "image",
                        "env": {"AWS_SECRET_ACCESS_KEY": "secret", "KEEP": "1"},
                    }
                },
                "mitigations": {
                    "diagnostic": {
                        "AWS_SECRET_ACCESS_KEY": "secret",
                        "KEEP": "1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    counts = RedactingRedactor(
        RedactionCfg(scrub_env_keys=("AWS_*",))
    ).scrub_file(src, dst)
    doc = json.loads(dst.read_text(encoding="utf-8"))

    assert counts.env_keys_removed == 2
    assert doc["environments"]["customer"]["env"] == {"KEEP": "1"}
    assert doc["mitigations"]["diagnostic"] == {"KEEP": "1"}


def test_unrelated_collector_env_json_is_not_structurally_scrubbed(tmp_path: Path):
    _mark_run_root(tmp_path)
    src = (
        tmp_path
        / "none-none"
        / "trial_0"
        / "collector"
        / "environments"
        / "stats"
        / "env.json"
    )
    src.parent.mkdir(parents=True)
    dst = tmp_path / "out" / "collector" / "env.json"
    src.write_text(
        json.dumps({"metrics": {"env_vars": {"AWS_SCORE": 0.91, "KEEP": 1}}}),
        encoding="utf-8",
    )

    counts = RedactingRedactor(
        RedactionCfg(scrub_env_keys=("AWS_*",))
    ).scrub_file(src, dst)
    doc = json.loads(dst.read_text(encoding="utf-8"))

    assert counts.env_keys_removed == 0
    assert doc["metrics"]["env_vars"] == {"AWS_SCORE": 0.91, "KEEP": 1}


@pytest.mark.parametrize(
    "relative",
    [
        Path("none-none") / "_subprocess" / "trial_d0_m0_t0.json",
        Path("cells") / "none-none" / "_subprocess" / "trial_d0_m0_t0.json",
    ],
)
def test_dispatcher_trial_layouts_scrub_environment_copies(
    relative: Path, tmp_path: Path
):
    _mark_run_root(tmp_path)
    src = tmp_path / relative
    src.parent.mkdir(parents=True)
    dst = tmp_path / "out" / relative
    src.write_text(
        json.dumps(
            {
                "execution_env": {
                    "env": {"AWS_SECRET_ACCESS_KEY": "secret", "KEEP": "1"}
                },
                "env": {
                    "schema_version": "1.15",
                    "env_vars": {
                        "AWS_SECRET_ACCESS_KEY": "secret",
                        "KEEP": "1",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    counts = RedactingRedactor(
        RedactionCfg(scrub_env_keys=("AWS_*",))
    ).scrub_file(src, dst)
    doc = json.loads(dst.read_text(encoding="utf-8"))

    assert counts.env_keys_removed == 2
    assert doc["execution_env"]["env"] == {"KEEP": "1"}
    assert doc["env"]["env_vars"] == {"KEEP": "1"}


def test_unrelated_collector_result_json_keeps_non_environment_env_map(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    _mark_run_root(run_dir)
    src = run_dir / "none-none" / "trial_0" / "collector" / "result.json"
    src.parent.mkdir(parents=True)
    dst = tmp_path / "out" / "collector" / "result.json"
    src.write_text(
        json.dumps({"env": {"AWS_SCORE": 0.91, "KEEP": 1}}),
        encoding="utf-8",
    )

    counts = RedactingRedactor(
        RedactionCfg(scrub_env_keys=("AWS_*",))
    ).scrub_file(src, dst)
    doc = json.loads(dst.read_text(encoding="utf-8"))

    assert counts.env_keys_removed == 0
    assert doc["env"] == {"AWS_SCORE": 0.91, "KEEP": 1}


def test_unrelated_collector_matrix_json_is_not_structurally_scrubbed(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    _mark_run_root(run_dir)
    src = run_dir / "none-none" / "trial_0" / "collector" / "matrix.json"
    src.parent.mkdir(parents=True)
    dst = tmp_path / "out" / "collector" / "matrix.json"
    src.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "resolved_env_vars": {
                            "AWS_SCORE": 0.91,
                            "KEEP": 1,
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    counts = RedactingRedactor(
        RedactionCfg(scrub_env_keys=("AWS_*",))
    ).scrub_file(src, dst)
    doc = json.loads(dst.read_text(encoding="utf-8"))

    assert counts.env_keys_removed == 0
    assert doc["cells"][0]["resolved_env_vars"] == {
        "AWS_SCORE": 0.91,
        "KEEP": 1,
    }


@pytest.mark.parametrize(
    "raw",
    [
        r'{"path":"\u002fhome\u002fcustomer\u002fprivate"}',
        r'{"path":"\/home\/customer\/private"}',
    ],
)
def test_generic_collector_json_scrubs_semantic_strings_and_stays_valid(raw: str, tmp_path: Path):
    """Raw text replacement either missed or corrupted these valid encodings."""
    src = tmp_path / "collector.json"
    dst = tmp_path / "out" / "collector.json"
    src.write_text(raw, encoding="utf-8")

    counts = RedactingRedactor(RedactionCfg(scrub_paths=True)).scrub_file(src, dst)
    doc = json.loads(dst.read_text(encoding="utf-8"))

    assert counts.paths_rewritten == 1
    assert doc == {"path": "<PATH:0>"}


def test_truncated_generic_collector_json_is_still_scrubbed_and_bundled(tmp_path: Path):
    """A collector killed mid-write must not cost the operator the bundle.

    Nobody owns this artifact's schema, so there are no env keys to protect --
    only path/IP rewriting. Refusing the file denied a handout for the crashed
    run that needed it most. Schema-owned documents still fail closed
    (``test_corrupt_result_json_fails_closed``).
    """
    raw = '{"path":"/home/customer/private", "step": 4'
    src = tmp_path / "collector.json"
    dst = tmp_path / "out" / "collector.json"
    src.write_text(raw, encoding="utf-8")

    counts = RedactingRedactor(RedactionCfg(scrub_paths=True)).scrub_file(src, dst)

    assert counts.paths_rewritten == 1
    assert dst.read_text(encoding="utf-8") == raw.replace("/home/customer/private", "<PATH:0>")


@pytest.mark.parametrize(
    "raw",
    [
        '{"host": "/home/customer/logs/ru',  # truncated inside the string itself
        r'{"host": "/home/customer/logs\q"}',  # invalid escape
    ],
)
def test_unscannable_generic_collector_json_falls_back_to_text(raw: str, tmp_path: Path):
    """When a string token itself does not scan, the text scrubber takes over.

    This is the only case the token walk cannot serve, and it must degrade to
    the pre-existing streaming scrubber -- redaction still runs -- rather than
    fail the bundle.
    """
    src = tmp_path / "collector.json"
    dst = tmp_path / "out" / "collector.json"
    src.write_text(raw, encoding="utf-8")

    counts = RedactingRedactor(RedactionCfg(scrub_paths=True)).scrub_file(src, dst)
    staged = dst.read_text(encoding="utf-8")

    assert counts.paths_rewritten == 1
    assert "/home/customer/logs" not in staged
    assert "<PATH:0>" in staged


def test_generic_collector_json_keeps_numbers_and_duplicate_keys_verbatim(
    tmp_path: Path,
):
    """Only string tokens may be rewritten.

    Parsing and re-serializing silently rewrote the collector's own numbers:
    precision was dropped, ``1e400`` became the non-JSON token ``Infinity``,
    and duplicate keys collapsed -- an altered metric in an artifact the
    customer is asked to trust.
    """
    raw = (
        '{"precise": 1.2345678901234567890123456789, "exp": 1.0E+5, '
        '"overflow": 1e400, "dup": 1, "dup": 2, '
        '"big": 123456789012345678901234567890, "where": "/home/customer/x"}'
    )
    src = tmp_path / "collector.json"
    dst = tmp_path / "out" / "collector.json"
    src.write_text(raw, encoding="utf-8")

    counts = RedactingRedactor(RedactionCfg(scrub_paths=True)).scrub_file(src, dst)
    staged = dst.read_text(encoding="utf-8")

    assert counts.paths_rewritten == 1
    assert staged == raw.replace('"/home/customer/x"', '"<PATH:0>"')
    assert "Infinity" not in staged


def test_generic_collector_json_without_matches_is_byte_identical(tmp_path: Path):
    """No match means no rewrite: the staged copy is the source, byte for byte."""
    raw = '{\n  "a":   [1, 2.50, true, null],\n  "b": "plain \\u00e9 text"\n}\n'
    src = tmp_path / "collector.json"
    dst = tmp_path / "out" / "collector.json"
    src.write_text(raw, encoding="utf-8")

    cfg = RedactionCfg(scrub_paths=True, scrub_ip_addresses=True)
    counts = RedactingRedactor(cfg).scrub_file(src, dst)

    assert dst.read_bytes() == raw.encode("utf-8")
    assert (counts.paths_rewritten, counts.ips_rewritten) == (0, 0)


def _resolved_recipe_doc() -> dict:
    """The shape ``write_resolved_recipe`` actually emits.

    It is a triage-shaped document (``workload`` / ``steps`` / ``cells``, no
    ``mode``) regardless of the run's mode, so the env overlays live at
    top-level ``extra_env``, ``cells[*].extra_env``, and the inline-docker
    ``cells[*].environment.env``.
    """
    return {
        "schema_version": 1,
        "workload": "_subprocess",
        "trials": 1,
        "steps": 1,
        "ticket": "TKT-1",
        "extra_env": {"AWS_SECRET_ACCESS_KEY": "secret", "KEEP": "1"},
        "confound": {"threshold": 1.15, "baseline_cell": "none-none"},
        "cells": [
            {
                "name": "none-none",
                "mitigations": ["none"],
                "environment": {
                    "docker": "image",
                    "env": {"AWS_REGION": "us-east-1", "KEEP": "1"},
                },
                "extra_env": {"AWS_TOKEN": "secret", "KEEP": "1"},
            }
        ],
    }


def test_resolved_recipe_scrubs_every_env_overlay(tmp_path: Path):
    """The resolved recipe re-emits the overlays matrix.json also records.

    Treating it as plain text left ``extra_env`` unscrubbed while the
    identical values were removed from ``matrix.json``.
    """
    _mark_run_root(tmp_path)
    src = tmp_path / "recipe.resolved.yaml"
    dst = tmp_path / "out" / "recipe.resolved.yaml"
    src.write_text(yaml.safe_dump(_resolved_recipe_doc(), sort_keys=False), encoding="utf-8")

    counts = RedactingRedactor(
        RedactionCfg(scrub_env_keys=("AWS_*",))
    ).scrub_file(src, dst)
    doc = yaml.safe_load(dst.read_text(encoding="utf-8"))

    assert counts.env_keys_removed == 3
    assert doc["extra_env"] == {"KEEP": "1"}
    assert doc["cells"][0]["extra_env"] == {"KEEP": "1"}
    assert doc["cells"][0]["environment"] == {"docker": "image", "env": {"KEEP": "1"}}


def test_resolved_recipe_stays_loadable_after_scrubbing(tmp_path: Path):
    """Scrubbing must not break the resolved recipe's reload contract."""
    _mark_run_root(tmp_path)
    src = tmp_path / "recipe.resolved.yaml"
    dst = tmp_path / "out" / "recipe.resolved.yaml"
    src.write_text(yaml.safe_dump(_resolved_recipe_doc(), sort_keys=False), encoding="utf-8")

    RedactingRedactor(RedactionCfg(scrub_env_keys=("AWS_*",))).scrub_file(src, dst)

    recipe = load_recipe(dst)
    assert recipe.extra_env == {"KEEP": "1"}
    assert recipe.cells[0].extra_env == {"KEEP": "1"}
    assert recipe.inline_environments[0].env == {"KEEP": "1"}


def test_corrupt_resolved_recipe_fails_closed(tmp_path: Path):
    _mark_run_root(tmp_path)
    src = tmp_path / "recipe.resolved.yaml"
    dst = tmp_path / "out" / "recipe.resolved.yaml"
    src.write_text("cells: [\n  - name: 'unterminated\n", encoding="utf-8")

    with pytest.raises(RedactionError):
        RedactingRedactor(RedactionCfg(scrub_env_keys=("AWS_*",))).scrub_file(src, dst)

    assert not dst.exists()


@pytest.mark.parametrize(
    "document",
    [
        [{"extra_env": {"AWS_TOKEN": "secret"}}],
        None,
        "not-a-recipe",
    ],
)
def test_wrong_shaped_resolved_recipe_fails_closed(document: object, tmp_path: Path):
    _mark_run_root(tmp_path)
    src = tmp_path / "recipe.resolved.yaml"
    dst = tmp_path / "out" / "recipe.resolved.yaml"
    src.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(RedactionError):
        RedactingRedactor(RedactionCfg(scrub_env_keys=("AWS_*",))).scrub_file(
            src, dst
        )

    assert not dst.exists()


def test_probe_env_scrubs_paths_and_ips_in_retained_values(tmp_path: Path):
    """probe.env duplicates the cell env mapping result.json already scrubs."""
    src = tmp_path / "probe.env"
    dst = tmp_path / "out" / "probe.env"
    src.write_text(
        "AWS_TOKEN=secret\n"
        "LD_LIBRARY_PATH=/home/customer/private/lib\n"
        "MASTER_ADDR=192.168.1.42\n",
        encoding="utf-8",
    )

    counts = RedactingRedactor(
        RedactionCfg(
            scrub_env_keys=("AWS_*",),
            scrub_paths=True,
            scrub_ip_addresses=True,
        )
    ).scrub_file(src, dst)
    out = dst.read_text(encoding="utf-8")

    assert counts.env_keys_removed == 1
    assert counts.paths_rewritten == 1
    assert counts.ips_rewritten == 1
    assert "/home/customer/private/lib" not in out
    assert "192.168.1.42" not in out
    assert "LD_LIBRARY_PATH=<PATH:0>" in out
    assert "MASTER_ADDR=<IPV4:0>" in out


def test_ambiguous_run_root_inference_fails_closed(tmp_path: Path):
    outer = tmp_path / "outer"
    inner = outer / "inner"
    _mark_run_root(outer)
    _mark_run_root(inner)
    src = inner / "none-none" / "trial_0" / "result.json"
    src.parent.mkdir(parents=True)
    src.write_text(
        json.dumps({"env": {"AWS_SECRET_ACCESS_KEY": "secret"}}),
        encoding="utf-8",
    )
    dst = tmp_path / "out" / "result.json"

    with pytest.raises(RedactionError) as excinfo:
        RedactingRedactor(
            RedactionCfg(scrub_env_keys=("AWS_*",))
        ).scrub_file(src, dst)

    assert "multiple probe run roots" in str(excinfo.value.cause)
    assert not dst.exists()


def test_structured_json_scrubs_paths_and_ips_in_object_keys(tmp_path: Path):
    src, dst = _result_json_paths(tmp_path)
    src.write_text(
        json.dumps(
            {
                "/home/customer/private": {
                    "192.168.1.42": "value",
                },
                "env": {
                    "/home/customer/AWS_TOKEN": "secret",
                },
            }
        ),
        encoding="utf-8",
    )

    counts = RedactingRedactor(
        RedactionCfg(scrub_paths=True, scrub_ip_addresses=True)
    ).scrub_file(src, dst)
    doc = json.loads(dst.read_text(encoding="utf-8"))

    assert doc == {
        "<PATH:0>": {"<IPV4:0>": "value"},
        "env": {"<PATH:1>": "secret"},
    }
    assert counts.paths_rewritten == 2
    assert counts.ips_rewritten == 1


def test_structured_json_key_collision_fails_closed(tmp_path: Path):
    src, dst = _result_json_paths(tmp_path)
    src.write_text(
        json.dumps({"/home/customer/private": 1, "<PATH:0>": 2}),
        encoding="utf-8",
    )

    with pytest.raises(RedactionError):
        RedactingRedactor(RedactionCfg(scrub_paths=True)).scrub_file(src, dst)

    assert not dst.exists()


def test_result_json_placeholders_consistent_across_fields(tmp_path: Path):
    """Path/IP placeholders share one index across the whole result.json.

    The old per-leaf scrub_text() call allocated a fresh index per string,
    so two DIFFERENT paths in different fields both became <PATH:0> (a
    collision) and the SAME path in two fields could get different Ns
    (inconsistent). One shared per-file index fixes both (Copilot review).
    """
    cfg = RedactionCfg(scrub_paths=True)
    redactor = RedactingRedactor(cfg)
    src, dst = _result_json_paths(tmp_path)
    src.write_text(
        json.dumps(
            {
                "a": "load /home/user/alpha",
                "b": "load /home/user/beta",
                "c": "again /home/user/alpha",
            }
        ),
        encoding="utf-8",
    )
    counts = redactor.scrub_file(src, dst)
    doc = json.loads(dst.read_text(encoding="utf-8"))
    # alpha (fields a & c) shares one placeholder; beta (field b) gets a
    # distinct one -- no collision, and the same path is consistent.
    assert doc["a"].endswith("<PATH:0>")
    assert doc["c"].endswith("<PATH:0>")
    assert doc["b"].endswith("<PATH:1>")
    assert "<PATH:0>" not in doc["b"]  # distinct paths never collide on one N
    assert counts.paths_rewritten == 3


def test_text_stream_scrub_matches_whole_file(tmp_path: Path, monkeypatch):
    """Streaming a multi-window log scrubs identically + byte-faithfully.

    The text branch streams off disk instead of reading the whole file
    (memory spike on big stdout.log; Copilot review). Output must equal a
    whole-file scrub_text pass: placeholders consistent across windows and
    CRLF terminators preserved byte-for-byte. A tiny cap forces many windows.
    """
    monkeypatch.setattr("aorta.probe.redaction.MAX_LOG_BYTES", 64)
    cfg = RedactionCfg(scrub_paths=True, scrub_ip_addresses=True)
    redactor = RedactingRedactor(cfg)
    # Repeated path/IP across many windows so cross-window placeholder
    # consistency (one shared index) is exercised; CRLF terminators included.
    line = "loaded /home/user/alpha from 192.168.1.42\r\n"
    text = line * 200
    src = tmp_path / "stdout.log"
    dst = tmp_path / "out" / "stdout.log"
    src.write_bytes(text.encode("utf-8"))
    redactor.scrub_file(src, dst)
    expected, _, _, _ = scrub_text(text, scrub_paths=True, scrub_ip_addresses=True)
    out_bytes = dst.read_bytes()
    assert out_bytes == expected.encode("utf-8")
    assert b"\r\n" in out_bytes  # CRLF preserved (no universal-newline translation)
    out = out_bytes.decode("utf-8")
    assert "/home/user/alpha" not in out
    assert "192.168.1.42" not in out
    assert out.count("<PATH:0>") == 200  # one shared index, not per-window


def test_fixture_log_scrubs_paths_and_ips(tmp_path: Path):
    raw = (FIXTURES / "redaction_input.txt").read_text(encoding="utf-8")
    cfg = RedactionCfg(
        scrub_paths=True,
        scrub_ip_addresses=True,
    )
    redactor = RedactingRedactor(cfg)
    src = tmp_path / "stdout.log"
    dst = tmp_path / "out" / "stdout.log"
    src.write_text(raw, encoding="utf-8")
    counts = redactor.scrub_file(src, dst)
    out = dst.read_text(encoding="utf-8")
    assert counts.paths_rewritten >= 3
    assert counts.ips_rewritten >= 2
    assert "/home/customer" not in out
    assert "192.168.1.42" not in out


def test_invalid_utf8_log_still_scrubbed(tmp_path: Path):
    """A stray non-UTF-8 byte must not disable scrubbing (oyazdanb review).

    stdout.log / stderr.log are raw subprocess bytes by design; the
    redactor used to fail open (byte-copy the file) on the first
    UnicodeDecodeError, silently leaking every path/IP after it. It now
    decodes with errors="replace" and keeps scrubbing.
    """
    cfg = RedactionCfg(scrub_paths=True, scrub_ip_addresses=True)
    redactor = RedactingRedactor(cfg)
    src = tmp_path / "stdout.log"
    dst = tmp_path / "out" / "stdout.log"
    src.write_bytes(b"bad\xff  /home/user/secret 192.168.1.1\n")
    counts = redactor.scrub_file(src, dst)
    out = dst.read_text(encoding="utf-8")
    assert counts.paths_rewritten >= 1
    assert counts.ips_rewritten >= 1
    assert "/home/user/secret" not in out
    assert "192.168.1.1" not in out


def test_line_windows_reconstructs_input():
    text = "alpha\nbeta\r\ngamma\rdelta\nno-trailing-newline"
    assert "".join(_line_windows(text)) == text


def test_line_windows_counts_bytes_not_chars(monkeypatch):
    """Windowing budget is UTF-8 bytes, not code points (Copilot review).

    A char-count threshold would undercount multi-byte text and skip
    windowing even when the byte size exceeds the cap.
    """
    monkeypatch.setattr("aorta.probe.redaction.MAX_LOG_BYTES", 10)
    text = "\u00e9\u00e9\u00e9\n\u00e9\u00e9\u00e9\n"  # two 7-byte lines
    assert len(text) == 8  # code points (<= cap)
    assert len(text.encode("utf-8")) == 14  # bytes (> cap)
    windows = _line_windows(text)
    assert len(windows) == 2  # byte-aware splits; a char-aware cap would not
    assert "".join(windows) == text


def test_line_windows_hard_splits_oversized_line(monkeypatch):
    """A single line longer than the cap is split, not emitted whole (Copilot).

    Line-based windowing only flushes between lines, so a hostile newline-free
    log would otherwise be passed to the regex as one over-cap window and
    defeat the byte budget. Each emitted window must stay ``<= MAX_LOG_BYTES``.
    """
    monkeypatch.setattr("aorta.probe.redaction.MAX_LOG_BYTES", 20)
    text = "/" * 73  # one 73-byte line, no terminator, > cap
    windows = _line_windows(text)
    assert len(windows) > 1
    assert all(len(w.encode("utf-8")) <= 20 for w in windows)
    assert "".join(windows) == text


def test_line_windows_oversized_multibyte_line_stays_under_cap(monkeypatch):
    """Hard-split budget is bytes: a multi-byte over-cap line stays bounded."""
    monkeypatch.setattr("aorta.probe.redaction.MAX_LOG_BYTES", 20)
    text = "\u00e9" * 40  # 40 chars, 80 bytes, no terminator
    windows = _line_windows(text)
    assert all(len(w.encode("utf-8")) <= 20 for w in windows)
    assert "".join(windows) == text


@pytest.mark.parametrize(
    ("token", "scrub_paths", "scrub_ips", "expected_counts"),
    [
        ("/home/customer/private", True, False, (1, 0, 0)),
        ("192.168.1.42", False, True, (0, 1, 0)),
        ("[2001:db8::1]", False, True, (0, 0, 1)),
    ],
)
def test_oversized_line_hard_split_keeps_redaction_token_whole(
    monkeypatch,
    token,
    scrub_paths,
    scrub_ips,
    expected_counts,
):
    """A valid path/IP crossing a hard-split seam must not leak."""
    monkeypatch.setattr("aorta.probe.redaction.MAX_LOG_BYTES", 128)
    # max_chars is 32. The token starts at offset 31, so the naive split cut
    # it after one character. The trailing padding makes this one oversized,
    # newline-free line and exercises the production hard-split path.
    text = "x" * 30 + " " + token + " " + "y" * 128

    out, paths, ipv4, ipv6 = scrub_text(
        text,
        scrub_paths=scrub_paths,
        scrub_ip_addresses=scrub_ips,
    )

    assert token not in out
    assert (paths, ipv4, ipv6) == expected_counts
    assert all(len(w.encode("utf-8")) <= 128 for w in _line_windows(text))


def test_scrubbed_copy_preserves_restrictive_mode(tmp_path: Path):
    """Every scrub branch carries the source mode; a 0600 source stays 0600.

    RedactingRedactor rewrites files (write_text/write_bytes) which land at
    the umask default and would widen a restrictive source inside the
    shareable bundle (Copilot review; same class as IdentityRedactor #199).
    """
    cfg = RedactionCfg(scrub_env_keys=("AWS_*",), scrub_paths=True)
    redactor = RedactingRedactor(cfg)
    cases = {
        "result.json": json.dumps({"env": {"HOME": "/home/user"}}),
        "host_env.json": json.dumps({"env": {"HOME": "/home/user"}}),
        "stdout.log": "loaded /home/user/x\n",
        "probe.env": "HOME=/home/user\n",
    }
    for name, content in cases.items():
        src = tmp_path / name
        src.write_text(content, encoding="utf-8")
        src.chmod(0o600)
        dst = tmp_path / "out" / name
        redactor.scrub_file(src, dst)
        assert dst.stat().st_mode & 0o777 == 0o600, f"{name} mode widened"
    # binary / non-text artifact -> else branch
    binsrc = tmp_path / "core.bin"
    binsrc.write_bytes(b"\x00\x01\x02")
    binsrc.chmod(0o600)
    bindst = tmp_path / "out" / "core.bin"
    redactor.scrub_file(binsrc, bindst)
    assert bindst.stat().st_mode & 0o777 == 0o600, "binary copy mode widened"


def test_binary_artifact_copied_byte_identical(tmp_path: Path):
    """The binary no-scrub branch streams via copyfile, byte-identical + counts.

    The branch no longer reads the whole artifact into memory (memory spike on
    a multi-GB core dump); it must still copy contents exactly and report
    bytes_in == bytes_out == file size from stat() (Copilot).
    """
    payload = bytes(range(256)) * 8
    src = tmp_path / "core.bin"
    src.write_bytes(payload)
    dst = tmp_path / "out" / "core.bin"
    counts = RedactingRedactor(RedactionCfg(scrub_paths=True)).scrub_file(src, dst)
    assert dst.read_bytes() == payload
    assert counts.bytes_in == len(payload)
    assert counts.bytes_out == len(payload)


def test_scrub_text_spans_window_seam(monkeypatch):
    """A path/IP must not be missed when it lands on a window boundary.

    The old fixed-slice windowing cut tokens in half at ``i*MAX_LOG_BYTES``
    so neither regex pass matched them. Line-aware windows never split a
    line (when each line is within the cap), so an IP sitting just past the
    byte budget is still scrubbed. The cap here is set so the token-bearing
    line fits on its own but the two lines together force a between-lines
    break -- the seam falls on the newline, not mid-token.
    """
    monkeypatch.setattr("aorta.probe.redaction.MAX_LOG_BYTES", 35)
    text = "x" * 15 + "\n" + "192.168.1.1 /home/user/secret\n"
    out, paths, v4, v6 = scrub_text(text, scrub_paths=True, scrub_ip_addresses=True)
    assert v4 == 1
    assert paths == 1
    assert "192.168.1.1" not in out
    assert "/home/user/secret" not in out


def test_corrupt_result_json_fails_closed(tmp_path: Path):
    """Corrupt result.json fails closed with a typed BundleError (Issue E).

    A raw JSONDecodeError would escape staging as a traceback and is not
    an OSError, so the writer's OSError wrap would miss it -- and partial
    handling risks an unredacted bundle. RedactionError is a BundleError,
    so the CLI fails closed (no bundle written).
    """
    cfg = RedactionCfg(scrub_paths=True)
    redactor = RedactingRedactor(cfg)
    src, dst = _result_json_paths(tmp_path)
    src.write_text("{ this is not valid json", encoding="utf-8")
    with pytest.raises(RedactionError) as excinfo:
        redactor.scrub_file(src, dst)
    assert excinfo.value.path == src
    assert not dst.exists()


@pytest.mark.parametrize(
    "document",
    [
        [{"env": {"AWS_TOKEN": "secret"}}],
        None,
        "not-a-result",
    ],
)
def test_wrong_shaped_result_json_fails_closed(document: object, tmp_path: Path):
    src, dst = _result_json_paths(tmp_path)
    src.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RedactionError):
        RedactingRedactor(
            RedactionCfg(scrub_env_keys=("AWS_*",))
        ).scrub_file(src, dst)

    assert not dst.exists()


def test_corrupt_host_env_json_fails_closed(tmp_path: Path):
    cfg = RedactionCfg(scrub_env_keys=("AWS_*",))
    redactor = RedactingRedactor(cfg)
    _mark_run_root(tmp_path)
    src = tmp_path / "host_env.json"
    dst = tmp_path / "out" / "host_env.json"
    src.write_text("not json at all", encoding="utf-8")
    with pytest.raises(RedactionError):
        redactor.scrub_file(src, dst)


def test_probe_env_scrubs_env_keys(tmp_path: Path):
    cfg = RedactionCfg(scrub_env_keys=("AWS_*", "HOME", "USER"))
    redactor = RedactingRedactor(cfg)
    src = tmp_path / "probe.env"
    dst = tmp_path / "out" / "probe.env"
    src.write_text(
        "AWS_SECRET_ACCESS_KEY=supersecret\nHOME=/home/user\nSAFE=1\n",
        encoding="utf-8",
    )
    counts = redactor.scrub_file(src, dst)
    out = dst.read_text(encoding="utf-8")
    assert counts.env_keys_removed == 2
    assert "supersecret" not in out
    assert "SAFE=1" in out
