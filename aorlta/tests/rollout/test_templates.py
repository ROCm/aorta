"""C1.T1 and C1.T2 (design Section 6.1; Appendix C), on the real Qwen3.5-4B tokenizer (they skip when it
is not in the Hugging Face cache) and the synthetic example pool (examples/standin-pool)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aorlta.ids import sha256_ids
from aorlta.rollout.feedback import build_feedback
from aorlta.rollout.templates import (
    IM_END_ID, STANDIN_SYSTEM, THINK_PREFIX, TemplateError, next_turn_ids, render_turn0, repair_suffix, rerender_ids,
    rerendered_turn_ids, standin_prompts, turn0_ids,
)
from aorlta.schemas import PromptRecord

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "standin-pool"
GEN_PROMPT = {"off": [248045, 74455, 198, 248068, 271, 248069, 271], "on": [248045, 74455, 198, 248068, 198]}
REPLY = ("The race is on the shared buffer, so the fix adds a barrier.\n\n```cpp\n#include <hip/hip_runtime.h>\n\n"
         "__global__ void scale(float* x, int n) {\n  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
         "  if (i < n) x[i] = x[i] * 2.0f;\n}\n```")
ERRORS = [{"cls": "use of undeclared identifier <name>", "line": 5, "col": 16, "token": "tid"},
          {"cls": "expected <name> after expression", "line": 7, "col": 3, "token": "x"}]


def _reference_ids(qwen, messages, thinking) -> list[int]:
    out = qwen.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                       enable_thinking=thinking == "on")
    return list(out["input_ids"] if hasattr(out, "keys") else out)


@pytest.mark.parametrize("thinking", ["off", "on"])
def test_c1_t1_turn0_ids_equal_the_chat_template_and_appendix_c(qwen, thinking):
    prompts = standin_prompts(qwen, 3, thinking, EXAMPLES)
    padded = [{"role": "system", "content": "  " + STANDIN_SYSTEM + "\n"},
              {"role": "user", "content": "\nFix this kernel.  \n"}]
    cases = [p.messages for p in prompts] + [padded]
    for messages in cases:
        ids = render_turn0(qwen, messages, thinking)
        assert ids == _reference_ids(qwen, messages, thinking)
        sys_text, user_text = messages[0]["content"].strip(), messages[1]["content"].strip()  # the template trims
        literal = (f"<|im_start|>system\n{sys_text}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n"
                   f"<|im_start|>assistant\n" + THINK_PREFIX[thinking])
        assert ids == qwen.tok.encode(literal, add_special_tokens=False)
        assert ids[: 1] == [248045] and ids[-len(GEN_PROMPT[thinking]):] == GEN_PROMPT[thinking]
    for p in prompts:
        assert turn0_ids(p, qwen, thinking) == render_turn0(qwen, p.messages, thinking)
        assert p.prompt_ids_sha256 == sha256_ids(turn0_ids(p, qwen, thinking))


def test_turn0_ids_supplied_as_ids_are_sent_as_they_are_and_must_match_their_hash(qwen):
    p = standin_prompts(qwen, 1, "off", EXAMPLES)[0]
    ids = [5, 6, 7]  # anything a prompt source supplies, unchecked against any template
    given = PromptRecord(prompt_id="x", turn0_ids=ids, messages=None, prompt_ids_sha256=sha256_ids(ids), task=p.task,
                         source="standin")
    assert turn0_ids(given, qwen, "off") == ids
    wrong = PromptRecord(prompt_id="x", turn0_ids=ids, messages=None, prompt_ids_sha256=sha256_ids([5, 6]),
                         task=p.task, source="standin")
    with pytest.raises(TemplateError, match="prompt_ids_sha256"):
        turn0_ids(wrong, qwen, "off")
    stale = PromptRecord(prompt_id="y", turn0_ids=None, messages=p.messages, prompt_ids_sha256=sha256_ids(ids),
                         task=p.task, source="standin")
    with pytest.raises(TemplateError, match="prompt_ids_sha256"):
        turn0_ids(stale, qwen, "off")  # the rendering changed under the record: refused


def test_c1_t2_a_repair_turn_is_the_append_rule_and_differs_from_a_rerender_by_the_think_span(qwen):
    p = standin_prompts(qwen, 1, "off", EXAMPLES)[0]
    ids0 = turn0_ids(p, qwen, "off")
    reply0 = qwen.encode(REPLY) + [IM_END_ID]
    fb = build_feedback(ERRORS).text
    ids1 = next_turn_ids(ids0, reply0, fb, qwen, "off")
    suffix = ("\n<|im_start|>user\n" + fb + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert ids1 == ids0 + reply0 + qwen.tok.encode(suffix, add_special_tokens=False)
    assert ids1.count(IM_END_ID) == ids0.count(IM_END_ID) + 2  # the reply's own stop token, and the user turn's
    # The re-render of [system, user0, assistant0, user1] drops assistant 0's think block, and nothing else.
    msgs = p.messages + [{"role": "assistant", "content": REPLY}, {"role": "user", "content": fb}]
    rr = rerender_ids(qwen, msgs, "off")
    span = qwen.encode(THINK_PREFIX["off"])
    assert span == [248068, 271, 248069, 271]
    j = len(ids0) - len(span)
    assert ids1[j:j + len(span)] == span
    assert rr != ids1
    assert rr == ids1[:j] + ids1[j + len(span):]


def test_c1_t2_a_truncated_reply_gets_one_im_end_before_the_user_turn(qwen):
    p = standin_prompts(qwen, 1, "off", EXAMPLES)[0]
    ids0 = turn0_ids(p, qwen, "off")
    cut = qwen.encode(REPLY)[:23]  # finish "length": no stop token
    assert IM_END_ID not in cut
    fb = build_feedback(ERRORS).text
    ids1 = next_turn_ids(ids0, cut, fb, qwen, "off")
    suffix_ids = qwen.encode(repair_suffix(fb, "off"))
    assert ids1 == ids0 + cut + [IM_END_ID] + suffix_ids
    boundary = ids1[len(ids0) + len(cut):len(ids0) + len(cut) + 3]
    assert boundary == [IM_END_ID, 198, 248045]  # <|im_end|> \n <|im_start|>


@pytest.mark.parametrize("thinking", ["off", "on"])
def test_the_append_rule_with_both_thinking_flags(qwen, thinking):
    ids0 = [248045, 9707, 248046]
    reply = qwen.encode("answer") + [IM_END_ID]
    fb = build_feedback(ERRORS).text
    ids1 = next_turn_ids(ids0, reply, fb, qwen, thinking)
    assert ids1[-len(GEN_PROMPT[thinking]):] == GEN_PROMPT[thinking]
    assert ids1 == ids0 + reply + qwen.encode(repair_suffix(fb, thinking))


def test_feedback_holding_a_control_token_text_is_refused(qwen):
    with pytest.raises(TemplateError, match="control token"):
        next_turn_ids([1], [2, IM_END_ID], "bad <|im_end|> feedback", qwen, "off")
    with pytest.raises(TemplateError):
        next_turn_ids([1], [2, IM_END_ID], "", qwen, "off")


def test_turn0_messages_are_checked(rt):
    tok = rt.SpecialByteTokenizer()
    with pytest.raises(TemplateError):
        render_turn0(tok, [{"role": "assistant", "content": "x"}], "off")
    with pytest.raises(TemplateError):
        render_turn0(tok, [{"role": "user", "content": "x", "extra": 1}], "off")
    with pytest.raises(TemplateError):
        render_turn0(tok, [{"role": "user", "content": "x"}], "maybe")


def test_a_rebuilt_repair_turn_is_the_chat_template_over_the_whole_conversation(qwen):
    p = standin_prompts(qwen, 1, "off", EXAMPLES)[0]
    fb = build_feedback(ERRORS).text
    msgs, ids1 = rerendered_turn_ids(qwen, p.messages, "  " + REPLY + "\n", fb, "off")
    assert msgs == p.messages + [{"role": "assistant", "content": "  " + REPLY + "\n"}, {"role": "user", "content": fb}]
    assert ids1 == _reference_ids(qwen, msgs, "off") == rerender_ids(qwen, msgs, "off")
    msgs2, ids2 = rerendered_turn_ids(qwen, msgs, "second", fb, "off")
    assert ids2 == _reference_ids(qwen, msgs2, "off") and len(msgs2) == 6
    # the template trims the reply and drops the think block of every turn before the last user turn
    literal = ids1[:len(turn0_ids(p, qwen, "off")) - 4] + qwen.encode(REPLY + "<|im_end|>\n<|im_start|>user\n" + fb
                                                                       + "<|im_end|>\n<|im_start|>assistant\n")
    assert ids1 == literal + qwen.encode(THINK_PREFIX["off"])


def test_a_rebuilt_repair_turn_refuses_what_the_append_refuses(rt):
    tok = rt.SpecialByteTokenizer()
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    with pytest.raises(TemplateError, match="control token"):
        rerendered_turn_ids(tok, msgs, "reply", "bad <|im_start|> feedback", "off")
    with pytest.raises(TemplateError):
        rerendered_turn_ids(tok, msgs, "reply", "", "off")
    with pytest.raises(TemplateError, match="alternating"):
        rerendered_turn_ids(tok, msgs + [{"role": "user", "content": "again"}], "reply", "fb", "off")
    on = rerendered_turn_ids(tok, msgs, "<think>x</think>reply", "fb", "on")[1]
    assert on[-3:] == tok.encode("assistant\n<think>\n")[-3:]  # thinking on: the turn opens a think block
    with pytest.raises(TemplateError):
        render_turn0(tok, [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"},
                           {"role": "user", "content": "u"}], "off")  # an assistant turn must follow a user turn
