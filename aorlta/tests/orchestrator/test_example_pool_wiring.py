"""The synthetic example pool (examples/standin-pool/) through a run's prompt pool (orchestrator/wiring.py)."""

from __future__ import annotations

from pathlib import Path

from aorlta.orchestrator.wiring import StandinPromptPool, standin_dataset_dir
from aorlta.rollout.templates import generation_prompt, standin_training_prompts

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "standin-pool"


class _ByteTokenizer:
    """One id per UTF-8 byte, with a plain chat layout that ends in the generation prompt, as turn 0 needs."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    def render(self, messages, thinking, add_generation_prompt=True) -> str:
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + (generation_prompt(thinking) if add_generation_prompt else "")


def test_a_run_reads_the_example_through_its_prompt_pool():
    # a config names the pool by the dataset directory or its listings/ directory
    assert standin_dataset_dir(EXAMPLE) == EXAMPLE == standin_dataset_dir(EXAMPLE / "listings")
    tok = _ByteTokenizer()
    pool = StandinPromptPool(tok, "off", standin_dataset_dir(EXAMPLE / "listings"))
    assert pool.source == "standin"
    assert pool.records() == standin_training_prompts(tok, "off", EXAMPLE)
    assert [r.prompt_id for r in pool.records()] == ["standin:example_block_sum", "standin:example_vector_add"]
