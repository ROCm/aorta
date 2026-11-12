"""Turn token sequences and the stand-in prompts (design Section 6.1, Turn formats; Appendix C).

Turn 0 is the model's own chat template, rendered with `add_generation_prompt=True` and the thinking
flag, then tokenised with `add_special_tokens=False`, unless the prompt record carries exact turn-0
ids, which are sent as they are. Either way the ids must hash to the record's `prompt_ids_sha256`.

A repair turn is built one of two ways (`rollout.repair_turn_ids`). `append`, the stand-ins' default, is a
token-level append:

    ids_t = ids_{t-1} + reply_{t-1} + ([248046] if reply_{t-1} does not end in 248046 else []) + tok(SUFFIX_t)
    SUFFIX_t = "\\n<|im_start|>user\\n" + FEEDBACK_t + "<|im_end|>\\n<|im_start|>assistant\\n" + THINK_PREFIX

`rerender` renders the whole conversation again with the chat
template, each reply as an assistant turn (its decoded text) and each feedback as a user turn
(`rerendered_turn_ids`). The two differ: Qwen3.5's template drops the think block of every assistant turn before
the last user turn, trims every turn, and re-tokenises the reply's text. C1.T2 pins the first difference.

The stand-in prompts are the original listings of a kernel dataset (`kernels.json` and `listings/<kid>.json`;
`examples/standin-pool/` shows the format) plus a plain "fix this kernel" instruction (Section 6.10 item 7).
Nothing else in the dataset is read. The dataset directory is the prompt pool's path, or `AORLTA_STANDIN_DATASET`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from ..ids import sha256_file, sha256_ids, standin_prompt_id
from ..schemas import PromptRecord, TaskRef

ENDOFTEXT_ID = 248044
IM_START_ID = 248045
IM_END_ID = 248046  # <|im_end|>: the engine stops on it (Appendix C)
THINK_OPEN_ID = 248068
THINK_CLOSE_ID = 248069
CONTROL_IDS = frozenset({ENDOFTEXT_ID, IM_START_ID, IM_END_ID, THINK_OPEN_ID, THINK_CLOSE_ID})
CONTROL_TEXTS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>")
STOP_TOKEN_IDS = (IM_END_ID,)

THINK_PREFIX = {"off": "<think>\n\n</think>\n\n", "on": "<think>\n"}

# The stand-in texts (the earlier prototype's, unchanged).
STANDIN_SYSTEM = "You are an expert AMD GPU kernel engineer."
STANDIN_USER0 = (
    "Below is the gfx950 disassembly of the GPU kernel `{name}` ({family} kernel from {library}). "
    "It may contain a synchronization bug.\n\nFix this kernel: write a corrected, complete HIP C++ source file "
    "that implements it (include <hip/hip_runtime.h> and define the __global__ kernel; no main function). "
    "Reply with the source in a single ```cpp code block.\n\n```\n{listing}\n```"
)
STANDIN_DATASET = Path(os.environ.get("AORLTA_STANDIN_DATASET", "/data/aorlta/standin-pool"))
STANDIN_MIN_INSNS, STANDIN_MAX_INSNS = 60, 400


class TemplateError(ValueError):
    """A turn sequence cannot be built exactly as Appendix C specifies."""


class Tokenizer(Protocol):
    """What the rollout needs from the model's tokenizer."""

    def encode(self, text: str) -> list[int]:  # add_special_tokens=False
        ...

    def decode(self, ids: Sequence[int]) -> str:  # special tokens kept
        ...

    def render(self, messages: Sequence[Mapping[str, Any]], thinking: str, add_generation_prompt: bool = True) -> str:
        ...


class HFTokenizer:
    """The model's own Hugging Face tokenizer and chat template."""

    def __init__(self, tok: Any):
        self.tok = tok

    @classmethod
    def load(cls, model_dir: str | Path) -> HFTokenizer:
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True))

    def encode(self, text: str) -> list[int]:
        return [int(t) for t in self.tok.encode(text, add_special_tokens=False)]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)

    def render(self, messages: Sequence[Mapping[str, Any]], thinking: str, add_generation_prompt: bool = True) -> str:
        return self.tok.apply_chat_template(
            [dict(m) for m in messages],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=_thinking_on(thinking),
        )

    def token_text(self, token_id: int) -> str:
        """The vocabulary entry of one id, for dumps."""
        return self.tok.convert_ids_to_tokens(int(token_id))


def _thinking_on(thinking: str) -> bool:
    if thinking not in THINK_PREFIX:
        raise TemplateError(f"thinking must be 'off' or 'on', got {thinking!r}")
    return thinking == "on"


def generation_prompt(thinking: str) -> str:
    _thinking_on(thinking)
    return "<|im_start|>assistant\n" + THINK_PREFIX[thinking]


def _check_messages(messages: Any) -> list[dict]:
    """A chat ending in a user turn: system and user messages, or a conversation (an optional system message, then
    user and assistant turns alternating, user first and last)."""
    if not isinstance(messages, (list, tuple)) or not messages:
        raise TemplateError("messages must be a non-empty list")
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, Mapping) or set(m) != {"role", "content"}:
            raise TemplateError(f"message {i} must have exactly the keys role and content")
        if m["role"] not in ("system", "user", "assistant"):
            raise TemplateError(f"message {i}: a prompt holds system, user and assistant messages only, got {m['role']!r}")
        if not isinstance(m["content"], str):
            raise TemplateError(f"message {i}: content must be a str")
        out.append({"role": m["role"], "content": m["content"]})
    if out[-1]["role"] != "user":
        raise TemplateError("the last turn-0 message must be the user's")
    roles = [m["role"] for m in out]
    turns = roles[1:] if roles[0] == "system" else roles
    if "assistant" in roles and turns != ["user", "assistant"] * (len(turns) // 2) + ["user"]:
        raise TemplateError("a conversation is user and assistant turns alternating, after an optional system message")
    return out


def render_turn0(tokenizer: Tokenizer, messages: Sequence[Mapping[str, Any]], thinking: str) -> list[int]:
    """`apply_chat_template(messages, add_generation_prompt=True, enable_thinking=...)`, tokenised with
    add_special_tokens=False. The ids must end in the generation prompt for the thinking flag."""
    msgs = _check_messages(messages)
    tail = tokenizer.encode(generation_prompt(thinking))
    ids = tokenizer.encode(tokenizer.render(msgs, thinking))
    if len(ids) <= len(tail) or ids[-len(tail):] != tail:
        raise TemplateError(f"the rendered turn 0 does not end in the thinking-{thinking} generation prompt")
    return ids


def turn0_ids(prompt: PromptRecord, tokenizer: Tokenizer, thinking: str) -> list[int]:
    """The exact ids sent at turn 0: the record's own ids if it has them, else the rendered messages.
    They must hash to `prompt.prompt_ids_sha256`, so a changed tokenizer or template cannot change
    what is sent without being noticed."""
    if prompt.turn0_ids is not None:
        ids = [int(t) for t in prompt.turn0_ids]
    elif prompt.messages is not None:
        ids = render_turn0(tokenizer, prompt.messages, thinking)
    else:
        raise TemplateError(f"prompt {prompt.prompt_id!r} has neither turn-0 ids nor messages")
    if not ids:
        raise TemplateError(f"prompt {prompt.prompt_id!r} has no turn-0 ids")
    if sha256_ids(ids) != prompt.prompt_ids_sha256:
        raise TemplateError(f"prompt {prompt.prompt_id!r}: turn-0 ids do not match prompt_ids_sha256")
    return ids


def repair_suffix(feedback: str, thinking: str) -> str:
    return "\n<|im_start|>user\n" + feedback + "<|im_end|>\n" + generation_prompt(thinking)


def _expected_suffix_controls(thinking: str) -> list[int]:
    think = [THINK_OPEN_ID, THINK_CLOSE_ID] if thinking == "off" else [THINK_OPEN_ID]
    return [IM_START_ID, IM_END_ID, IM_START_ID, *think]


def next_turn_ids(
    prev_ids: Sequence[int], reply_ids: Sequence[int], feedback: str, tokenizer: Tokenizer, thinking: str
) -> list[int]:
    """The append rule of Appendix C. Refuses feedback whose text would tokenise into a control token,
    since that would change the turn structure the model sees."""
    _thinking_on(thinking)
    if not isinstance(feedback, str) or not feedback:
        raise TemplateError("feedback must be a non-empty str")
    if any(t in feedback for t in CONTROL_TEXTS):
        raise TemplateError("feedback contains the text of a control token")
    ids = [int(t) for t in prev_ids] + [int(t) for t in reply_ids]
    if not reply_ids or int(reply_ids[-1]) != IM_END_ID:
        ids.append(IM_END_ID)
    suffix = tokenizer.encode(repair_suffix(feedback, thinking))
    controls = [t for t in suffix if t in CONTROL_IDS]
    if controls != _expected_suffix_controls(thinking):
        raise TemplateError(f"the repair suffix tokenised to control ids {controls}")
    return ids + suffix


def rerender_ids(tokenizer: Tokenizer, messages: Sequence[Mapping[str, Any]], thinking: str) -> list[int]:
    """A full re-render of a conversation (for C1.T2 and the dumps)."""
    return tokenizer.encode(tokenizer.render([dict(m) for m in messages], thinking))


def rerendered_turn_ids(
    tokenizer: Tokenizer, messages: Sequence[Mapping[str, Any]], reply: str, feedback: str, thinking: str
) -> tuple[list[dict], list[int]]:
    """The rebuilt repair turn (`rollout.repair_turn_ids: rerender`): the whole conversation re-rendered
    with the model's chat template, the reply as an assistant turn and the feedback as a user turn after it, so the
    template drops the earlier turns' think blocks and trims every turn. -> (messages, ids). `reply` is the reply's
    text as decoded (special tokens kept, one trailing stop token dropped). Refuses feedback whose text would
    tokenise into a control token, as the append does."""
    _thinking_on(thinking)
    if not isinstance(feedback, str) or not feedback:
        raise TemplateError("feedback must be a non-empty str")
    if any(t in feedback for t in CONTROL_TEXTS):
        raise TemplateError("feedback contains the text of a control token")
    if not isinstance(reply, str):
        raise TemplateError("the reply must be a str")
    msgs = _check_messages([*messages, {"role": "assistant", "content": reply}, {"role": "user", "content": feedback}])
    tail = tokenizer.encode(generation_prompt(thinking))
    ids = tokenizer.encode(tokenizer.render(msgs, thinking))
    if len(ids) <= len(tail) or ids[-len(tail):] != tail:
        raise TemplateError(f"the rebuilt repair turn does not end in the thinking-{thinking} generation prompt")
    return msgs, ids


# --- stand-in prompts (Section 6.10 item 7) --------------------------------------------------------


def pick_standin_kernels(n: int, dataset: str | Path = STANDIN_DATASET) -> list[dict]:
    """Kernels with 60-400 instructions, sorted by kid, every
    (len // n)-th, first n; each with its original listing."""
    if not isinstance(n, int) or n < 1:
        raise TemplateError(f"n must be an int >= 1, got {n!r}")
    dataset = Path(dataset)
    kernels = json.loads((dataset / "kernels.json").read_text(encoding="utf-8"))
    cand = sorted(
        (v for v in kernels.values() if STANDIN_MIN_INSNS <= v["n_insns"] <= STANDIN_MAX_INSNS), key=lambda v: v["kid"]
    )
    step = max(1, len(cand) // n)
    out = [_with_listing(v, dataset) for v in cand[::step][:n]]
    if len(out) < n:
        raise TemplateError(f"the dataset has only {len(out)} stand-in kernels, {n} asked for")
    return out


def _with_listing(kernel: Mapping[str, Any], dataset: Path) -> dict:
    path = dataset / "listings" / f"{kernel['kid']}.json"
    listing = json.loads(path.read_text(encoding="utf-8"))
    return dict(kernel, listing="\n".join(f"{off:#06x}: {txt}" for off, txt in listing), listing_path=str(path))


def standin_training_kernels(dataset: str | Path = STANDIN_DATASET) -> list[dict]:
    """The stand-in training pool: every kernel with 60-400 instructions whose `heldout` flag is False, sorted by
    kid, each with its original listing. Held-out kernels stay out of training, and a
    kernel without a boolean `heldout` flag is left out too."""
    dataset = Path(dataset)
    kernels = json.loads((dataset / "kernels.json").read_text(encoding="utf-8"))
    pool = sorted(
        (v for v in kernels.values()
         if STANDIN_MIN_INSNS <= v["n_insns"] <= STANDIN_MAX_INSNS and v.get("heldout") is False),
        key=lambda v: v["kid"],
    )
    if not pool:
        raise TemplateError(f"{dataset} has no stand-in kernel outside the held-out set")
    return [_with_listing(v, dataset) for v in pool]


def standin_messages(kernel: Mapping[str, Any]) -> list[dict]:
    user = STANDIN_USER0.format(
        name=kernel["demangled"], family=kernel["family"], library=kernel["library"], listing=kernel["listing"]
    )
    return [{"role": "system", "content": STANDIN_SYSTEM}, {"role": "user", "content": user}]


def standin_prompts(
    tokenizer: Tokenizer, n: int, thinking: str, dataset: str | Path = STANDIN_DATASET
) -> list[PromptRecord]:
    """n stand-in prompt records, `prompt_id = standin:<kid>`, hashed as rendered for this thinking flag."""
    return standin_records(tokenizer, pick_standin_kernels(n, dataset), thinking)


def standin_training_prompts(
    tokenizer: Tokenizer, thinking: str, dataset: str | Path = STANDIN_DATASET
) -> list[PromptRecord]:
    """A prompt record for every kernel of `standin_training_kernels`."""
    return standin_records(tokenizer, standin_training_kernels(dataset), thinking)


def standin_records(tokenizer: Tokenizer, kernels: Sequence[Mapping[str, Any]], thinking: str) -> list[PromptRecord]:
    """One prompt record per kernel (with its listing), `prompt_id = standin:<kid>`."""
    out = []
    for kernel in kernels:
        messages = standin_messages(kernel)
        ids = render_turn0(tokenizer, messages, thinking)
        task = TaskRef(
            contract_id=f"standin:{kernel['kid']}",
            parent_id=None,
            parent_healthy=None,
            reference_kind=None,
            task_sha256=sha256_file(kernel["listing_path"]),
        )
        out.append(
            PromptRecord(
                prompt_id=standin_prompt_id(kernel["kid"]),
                turn0_ids=None,
                messages=messages,
                prompt_ids_sha256=sha256_ids(ids),
                task=task,
                source="standin",
            )
        )
    return out
