"""The corpus-v2 system prompt and its profiles, pinned to the exact text evaluations were computed under.

An evaluation column records the sha256 of its system prompt and two columns pair only
under the same one, so a silent edit to the prompt would make every recorded column
incomparable with a new one.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))

import rocjitsu_env as rj  # noqa: E402

#: The v2 prompt every corpus-v2 base column so far was computed under.
PLAIN_V2_SHA256 = "4dde1ec848cd2dfd97f8f269de43a50e937c0e082bba43ae7f7233a996147ace"
PROCEDURE_V2_SHA256 = "a0a084cbaef3bdff39fef234e6967f68269958c4a2e6c252cc680decb52d7aec"


def test_the_plain_v2_prompt_is_unchanged_by_the_profile_switch(monkeypatch):
    monkeypatch.delenv(rj.PROMPT_ENV, raising=False)
    assert hashlib.sha256(rj.system_prompt_v2("v2").encode()).hexdigest() == PLAIN_V2_SHA256
    assert rj.system_prompt_v2("v2", profile="plain") == rj.system_prompt_v2("v2")


def test_the_procedure_profile_appends_the_procedure_and_nothing_else(monkeypatch):
    monkeypatch.delenv(rj.PROMPT_ENV, raising=False)
    plain = rj.system_prompt_v2("v2", profile="plain")
    procedure = rj.system_prompt_v2("v2", profile="procedure")
    assert procedure == plain + rj.PROCEDURE_V2
    monkeypatch.setenv(rj.PROMPT_ENV, "procedure")
    assert rj.system_prompt_v2("v2") == procedure, "the environment variable selects it"
    with pytest.raises(SystemExit):
        rj.resolve_prompt("chain_of_thought")


def test_the_procedure_states_no_scenario_specific_fact():
    assert not re.search(r"0x[0-9a-f]+|\bk\d{3}\b", rj.PROCEDURE_V2), "no offset and no kernel id"


def test_the_quirks_profile_appends_the_quirk_and_leaves_the_procedure_prompt_alone(monkeypatch):
    monkeypatch.delenv(rj.PROMPT_ENV, raising=False)
    procedure = rj.system_prompt_v2("v2", profile="procedure")
    assert hashlib.sha256(procedure.encode()).hexdigest() == PROCEDURE_V2_SHA256, \
        "every procedure-prompt column on disk was computed under this exact text"
    assert rj.system_prompt_v2("v2", profile="procedure_quirks") == procedure + rj.QUIRKS_V2
    assert not re.search(r"0x[0-9a-f]+|\bk\d{3}\b", rj.QUIRKS_V2), "no offset and no kernel id"
