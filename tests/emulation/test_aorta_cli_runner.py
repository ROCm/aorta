"""Tests for ``scripts/emulation/aorta_cli_runner.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "emulation" / "aorta_cli_runner.py"
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("aorta_cli_runner", _RUNNER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner():
    return _load_runner_module()


class TestParseAortaCliJson:
    def test_parses_string_list(self, runner):
        assert runner.parse_aorta_cli_json('["triage","run"]') == ["triage", "run"]

    def test_rejects_invalid_json(self, runner):
        with pytest.raises(SystemExit, match="invalid AORTA_CLI_JSON"):
            runner.parse_aorta_cli_json("{not json")

    def test_rejects_non_list(self, runner):
        with pytest.raises(SystemExit, match="must be a JSON list of strings"):
            runner.parse_aorta_cli_json('"triage"')

    def test_rejects_non_string_elements(self, runner):
        with pytest.raises(SystemExit, match="must be a JSON list of strings"):
            runner.parse_aorta_cli_json("[1, 2]")
