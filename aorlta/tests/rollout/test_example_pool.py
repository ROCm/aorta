"""The synthetic example pool (examples/standin-pool/) through the stand-in prompt source, as a run reads it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorlta.ids import sha256_file
from aorlta.rollout.templates import (
    STANDIN_MAX_INSNS, STANDIN_MIN_INSNS, STANDIN_SYSTEM, TemplateError, generation_prompt, pick_standin_kernels,
    standin_prompts, standin_training_kernels, standin_training_prompts, turn0_ids,
)

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "standin-pool"
ALL = ["example_block_sum", "example_transpose", "example_vector_add"]
TRAINING = ["example_block_sum", "example_vector_add"]  # example_transpose is held out


def _entries() -> dict:
    return json.loads((EXAMPLE / "kernels.json").read_text(encoding="utf-8"))


def _listing(kid: str) -> list:
    return json.loads((EXAMPLE / "listings" / f"{kid}.json").read_text(encoding="utf-8"))


def _listing_text(kid: str) -> str:
    """The listing as the prompt shows it: one `0x0000: <instruction>` line per pair."""
    return "\n".join(f"{off:#06x}: {txt}" for off, txt in _listing(kid))


def test_every_entry_is_in_the_format_the_loader_reads():
    entries = _entries()
    assert sorted(entries) == ALL
    for kid, entry in entries.items():
        assert entry["kid"] == kid
        assert all(isinstance(entry[key], str) and entry[key] for key in ("demangled", "family", "library"))
        assert type(entry["heldout"]) is bool
        listing = _listing(kid)
        assert all(isinstance(pair, list) and len(pair) == 2 and type(pair[0]) is int and isinstance(pair[1], str)
                   and pair[1] for pair in listing)
        offsets = [off for off, _ in listing]
        assert offsets[0] == 0 and offsets == sorted(set(offsets))
        assert listing[-1][1] == "s_endpgm"
        # the loader selects by n_insns alone: it must count the listing, and be in range or the kernel is dropped
        assert entry["n_insns"] == len(listing)
        assert STANDIN_MIN_INSNS <= entry["n_insns"] <= STANDIN_MAX_INSNS


def test_the_training_pool_is_every_kernel_not_held_out():
    kernels = standin_training_kernels(EXAMPLE)
    assert [k["kid"] for k in kernels] == TRAINING
    for k in kernels:
        assert k["listing"] == _listing_text(k["kid"])
        assert Path(k["listing_path"]) == EXAMPLE / "listings" / f"{k['kid']}.json"


@pytest.mark.parametrize("thinking", ["off", "on"])
def test_each_training_kernel_gives_a_prompt_record(rt, thinking):
    tok = rt.SpecialByteTokenizer()
    entries = _entries()
    records = standin_training_prompts(tok, thinking, EXAMPLE)
    assert [r.prompt_id for r in records] == [f"standin:{kid}" for kid in TRAINING]
    for r, kid in zip(records, TRAINING):
        entry = entries[kid]
        assert r.source == "standin" and r.turn0_ids is None
        system, user = r.messages
        assert system == {"role": "system", "content": STANDIN_SYSTEM} and user["role"] == "user"
        for text in (entry["demangled"], entry["family"], entry["library"], _listing_text(kid)):
            assert text in user["content"]
        assert r.task.contract_id == f"standin:{kid}"
        assert r.task.task_sha256 == sha256_file(EXAMPLE / "listings" / f"{kid}.json")
        ids = turn0_ids(r, tok, thinking)  # refuses ids that do not hash to prompt_ids_sha256
        tail = tok.encode(generation_prompt(thinking))
        assert ids[-len(tail):] == tail


def test_picking_includes_the_held_out_kernel_and_needs_enough_kernels(rt):
    assert [k["kid"] for k in pick_standin_kernels(3, EXAMPLE)] == ALL
    records = standin_prompts(rt.SpecialByteTokenizer(), 3, "off", EXAMPLE)
    assert [r.prompt_id for r in records] == [f"standin:{kid}" for kid in ALL]
    with pytest.raises(TemplateError, match="only 3"):
        pick_standin_kernels(4, EXAMPLE)


def test_the_prompts_render_with_the_models_chat_template(qwen):
    tail = qwen.encode(generation_prompt("off"))
    for r in standin_training_prompts(qwen, "off", EXAMPLE):
        assert turn0_ids(r, qwen, "off")[-len(tail):] == tail
