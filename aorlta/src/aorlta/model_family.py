"""The model families AORLTA can train, told apart by a checkpoint's `config.json`.

What differs between them outside the tokenizer (whose control-token ids and generation prompts
`rollout.templates.ChatFormat` reads from the tokenizer itself):

- `qwen3_5`: Qwen3.5 (`Qwen3_5ForConditionalGeneration`). The checkpoints are vision-language, so transformers
  loads them with `AutoModelForImageTextToText`, and the text decoder's keys are `model.language_model.layers.N.*`.
  Its layers mix full attention with gated-DeltaNet linear attention, whose fast path the trainer requires
  (`trainer.kernels`).
- `qwen3`: dense Qwen3 (`Qwen3ForCausalLM`, Qwen3-0.6B to Qwen3-32B). Loaded with `AutoModelForCausalLM`; the
  decoder's keys are `model.layers.N.*`; full attention only, so the linear-attention fast path does not apply.
  Its engines run TokenSpeed's Triton attention kernels (`attention_backend: triton`): with the default ones,
  TokenSpeed 3b684074 on gfx950 served a dense Qwen3 wrong log-probs for whole groups of samples (a reply's first
  token one the model gives a log-probability of -31, every sample of a prompt sharing the cached prefill) and
  stopped with an illegal memory access within a few batches after a weight reload, in score and training runs
  alike; with the Triton kernels neither happened.

A config whose `model_type` is neither, or whose `architectures` is not exactly the family's, is refused: the LoRA
targets, the name map and the trainer's loader are only known for these two.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class ModelFamilyError(ValueError):
    """A checkpoint config that names no family AORLTA supports, or cannot be read."""


@dataclass(frozen=True)
class ModelFamily:
    model_type: str
    architecture: str  # the one entry of config.json's `architectures`
    auto_class: str  # the transformers auto class that loads the checkpoint
    decoder_prefix: str  # the checkpoint keys of decoder layer N start with decoder_prefix + "N."
    linear_attention: bool  # whether some layers are gated-DeltaNet linear attention
    attention_backend: str | None  # the engines' TokenSpeed `attention_backend`; None is TokenSpeed's own choice


FAMILIES = {
    "qwen3_5": ModelFamily("qwen3_5", "Qwen3_5ForConditionalGeneration", "AutoModelForImageTextToText",
                           "model.language_model.layers.", True, None),
    "qwen3": ModelFamily("qwen3", "Qwen3ForCausalLM", "AutoModelForCausalLM", "model.layers.", False, "triton"),
}


def family_of(config: Mapping) -> ModelFamily:
    """The family of a parsed `config.json`."""
    if not isinstance(config, Mapping):
        raise ModelFamilyError("the model config is not a JSON object")
    model_type = config.get("model_type")
    fam = FAMILIES.get(model_type) if isinstance(model_type, str) else None
    if fam is None:
        raise ModelFamilyError(f"model_type {model_type!r} is not a supported family ({', '.join(FAMILIES)})")
    archs = config.get("architectures")
    if archs != [fam.architecture]:
        raise ModelFamilyError(f"model_type {model_type!r} with architectures {archs!r}: expected [{fam.architecture!r}]")
    return fam


def read_family(model_dir: str | Path) -> ModelFamily:
    """The family of the checkpoint in `model_dir`, from its `config.json`."""
    path = Path(model_dir) / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelFamilyError(f"{path}: cannot be read: {type(exc).__name__}: {str(exc)[:120]}") from None
    return family_of(config)


def engine_attention_backend(model_dir: str | Path) -> str | None:
    """The engines' `attention_backend` for the checkpoint in `model_dir`: its family's, or None (TokenSpeed's own
    choice, as before families were told apart) when its config names no family AORLTA knows."""
    try:
        return read_family(model_dir).attention_backend
    except ModelFamilyError:
        return None
