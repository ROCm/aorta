"""Tests for safe environment-bundle rendering in registry listings."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from aorta.cli._env_display import format_env_bundle
from aorta.cli.environments import environments
from aorta.cli.mitigations import mitigations
from aorta.cli.triage import triage

_VALUES = {
    "SAFE": "a=b",
    "SPACED": "two words",
    "EMPTY": "",
    "NEWLINE": "first\nsecond",
    "CONTROL": "\x1b[31m",
}


def test_format_env_bundle_keeps_simple_values_and_escapes_unsafe_values():
    assert format_env_bundle(_VALUES) == (
        r"CONTROL='\x1b[31m' EMPTY='' NEWLINE='first\nsecond' " r"SAFE=a=b SPACED='two words'"
    )
    assert format_env_bundle({}) == "(none)"


@pytest.mark.parametrize(
    ("command", "args"),
    [
        pytest.param(environments, ("list", "--file"), id="environments"),
        pytest.param(
            triage,
            ("list-environments", "--mitigations-file"),
            id="triage-environments",
        ),
        pytest.param(mitigations, ("list", "--file"), id="mitigations"),
        pytest.param(
            triage,
            ("list-mitigations", "--mitigations-file"),
            id="triage-mitigations",
        ),
    ],
)
def test_registry_listings_escape_ambiguous_and_control_values(tmp_path, command, args):
    sidecar = tmp_path / "registry.json"
    sidecar.write_text(
        json.dumps(
            {
                "version": 1,
                "environments": {"display-env": {"env": _VALUES}},
                "mitigations": {"display-mitigation": _VALUES},
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(command, [*args, str(sidecar)])

    assert result.exit_code == 0, result.output
    assert "SAFE=a=b" in result.output
    assert "SPACED='two words'" in result.output
    assert r"NEWLINE='first\nsecond'" in result.output
    assert r"CONTROL='\x1b[31m'" in result.output
    assert "\x1b" not in result.output
