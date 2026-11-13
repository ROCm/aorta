"""A starter for a Q&A expert model (examples/verticals/README.md): questions in your domain with reference answers,
and a reward for matching the reference. Every part marked PLACEHOLDER is to be replaced with your own.

A pool is a directory holding `questions.jsonl`, one question per line: `{"id", "question", "answer", "aliases"?}`.
The model is asked the question and told to end its reply with a line `Answer: <answer>`. A reply without that line
gets one repair turn that says so (this module's parse and feedback); there is no compile step
(`rollout.compile_adapter: none`). The reward service runs one leg, `ref.answer`, on the CPU: the last `Answer:` line,
normalised, against the reference and its aliases.

Registered names: task `qa-expert`; parse `qa-expert.parse`; feedback `qa-expert.feedback`; profile `qa-expert`
(qa-expert.yaml); check adapter `qa-expert.match`; reward function `qa-expert.v1`.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
# PLACEHOLDER: the prompt. A domain expert usually wants a system message that names the domain.
SYSTEM = "You are a helpful assistant."
USER = "{question}\n\nEnd your reply with a line of the form 'Answer: <your answer>'."
ANSWER_RE = re.compile(r"^[ \t]*\**[ \t]*answer[ \t]*\**[ \t]*:[ \t]*(.*?)[ \t]*$", re.IGNORECASE)  # one line


def load_rows(pool_dir) -> list[dict]:
    rows = [json.loads(line) for line in (Path(pool_dir) / "questions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{pool_dir}/questions.jsonl lists an id twice")
    return rows


def row_sha256(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()


def normalise(text: str) -> str:
    """PLACEHOLDER normalisation: NFKC, lower case, surrounding quotes, backticks and asterisks, a trailing full stop
    and runs of space removed. A real expert task may want more (units, numeric tolerance, synonyms) or a judge."""
    t = unicodedata.normalize("NFKC", text).strip().lower()
    t = t.strip("*`'\" ").rstrip(".").strip()
    return re.sub(r"\s+", " ", t)


def final_answer(reply: str) -> str | None:
    """The last `Answer:` line's answer, or None."""
    found = [m.group(1) for line in reply.splitlines() if (m := ANSWER_RE.match(line))]
    return (found[-1] or None) if found else None  # the last marker decides, an empty one included


def messages(row: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(question=row["question"])}]


def records(pool_dir, tokenizer, thinking):
    from aorlta.ids import sha256_ids
    from aorlta.rollout.templates import render_turn0
    from aorlta.schemas import PromptRecord, TaskRef

    out = []
    for row in load_rows(pool_dir):
        msgs = messages(row)
        out.append(PromptRecord(
            prompt_id=f"qa-expert:{row['id']}", turn0_ids=None, messages=msgs,
            prompt_ids_sha256=sha256_ids(render_turn0(tokenizer, msgs, thinking)),
            task=TaskRef(contract_id=f"qa-expert:{row['id']}", parent_id=None, parent_healthy=None,
                         reference_kind=None, task_sha256=row_sha256(row)),
            source="standin"))
    return out


class ExpertParse:
    """`qa-expert.parse`: a reply with an `Answer:` line is an answer (nothing to compile); one without is rejected
    with a message, which `qa-expert.feedback` turns into a repair turn."""

    name = "qa-expert.parse"

    def __init__(self):
        from aorlta.rollout.templates import IM_END_ID

        self.reply_stops = (IM_END_ID,)

    def check_prompts(self, prompts, thinking, max_repair_turns=0):
        return None

    def check(self, text, thinking, prompt):
        from aorlta.rollout.compile_check import ParseOutcome

        reply = text.rsplit("</think>", 1)[-1] if thinking == "on" else text
        answer = final_answer(reply)
        if answer is None:
            return ParseOutcome("rejected", rejection=(
                "Your reply has no final line of the form 'Answer: <your answer>'. Give your answer on such a line.",
                "no_answer"))
        return ParseOutcome("edits", code=answer)


class ExpertFeedback:
    name = "qa-expert.feedback"

    def build(self, errors, max_chars):
        from aorlta.rollout.feedback import build_feedback

        return build_feedback(errors, max_chars)

    def build_apply(self, message, reason):
        from aorlta.rollout.feedback import Feedback

        return Feedback(message, 0, 0)


class ExpertMatch:
    """`qa-expert.match`, leg `ref.answer`: clean when the final answer matches the reference or an alias after
    `normalise`; `fail` `wrong_answer`, or `no_answer` when the reply has no `Answer:` line; unsettled when the task is
    unknown or the reply cannot be read."""

    version = "1"
    legs = ("ref.answer",)
    resource = "local_cpu"

    def __init__(self, ctx):
        self.name, self.ctx = ctx.name, ctx
        self.rows = {}
        for pool in ctx.pools:
            if (Path(pool) / "questions.jsonl").is_file():
                for row in load_rows(pool):
                    self.rows[row_sha256(row)] = row

    def run(self, job):
        from aorlta.reward.adapters.base import make_result, unsettled

        t0 = self.ctx.clock()
        try:
            row = self.rows.get(job.task_sha256)
            response = job.inputs.get("response")
            if job.leg != "ref.answer" or row is None or not response:
                return unsettled(job, "adapter_error", t_start=t0, t_end=self.ctx.clock())
            answer = final_answer(Path(response).read_text(encoding="utf-8").rsplit("</think>", 1)[-1])
            if answer is None:
                return make_result(job, "fail", "no_answer", t_start=t0, t_end=self.ctx.clock())
            # PLACEHOLDER: an exact match after normalising. A judge (a command or a service that reads the
            # question, the reference and the answer) goes here; a judge that gives no verdict is unsettled.
            wanted = {normalise(a) for a in [row["answer"], *row.get("aliases", [])]}
            if normalise(answer) in wanted:
                return make_result(job, "clean", t_start=t0, t_end=self.ctx.clock())
            return make_result(job, "fail", "wrong_answer", t_start=t0, t_end=self.ctx.clock())
        except Exception:  # noqa: BLE001 - an adapter never raises
            return unsettled(job, "adapter_error", t_start=t0, t_end=self.ctx.clock())

    def controls(self):
        return []

    def health(self):
        return {"name": self.name, "responsive": True, "questions": len(self.rows)}


class ExpertV1:
    """`qa-expert.v1`, a PLACEHOLDER reward: +1.0 for a matching answer, a value of its own for each failure the check
    names (`BASE`), then -0.05 x the share of the token budget used. An unsettled leg, or a reason `BASE` does not
    name, leaves the sample unscored."""

    name = "qa-expert.v1"
    version = "1"
    legs = ("ref.answer",)
    # TODO(your vertical): replace these values with your real reward, and the check's reasons with your own (a
    # judge's "partly right", say). Keep:
    #   - the total in [-1.55, 1.0], its parts exactly `base`, `broke_healthy` and `length`, adding up to it (the
    #     service recomputes it and leaves anything else unscored, with an alarm): so a base in [-1.5, 1.0];
    #   - an unsettled or missing leg, or a reason you did not name, unscored: never clean, never a score;
    #   - each failure its own value, so the answers to one prompt get different rewards (GRPO learns only from the
    #     differences);
    #   - `claims_clean` true only when the leg is clean.
    BASE = {"wrong_answer": -0.5, "no_answer": -1.0}

    def __init__(self, profile=None):
        from aorlta.reward.rewardfn.base import code_sha256

        self.settings = {}
        self.code_sha256 = code_sha256(__file__)

    def _base(self, legs):
        leg = legs.get("ref.answer")
        if leg is None:
            return "pending", "ref.answer"
        if leg.state == "clean":
            return "score", 1.0
        if leg.state == "fail" and leg.reason in self.BASE:
            return "score", self.BASE[leg.reason]
        return "unscored", f"ref.answer:{leg.reason or leg.state}"

    @staticmethod
    def _finish(base, cand):
        parts = {"base": float(base), "broke_healthy": 0.0,
                 "length": -0.05 * (cand.response_tokens / cand.max_new_tokens)}
        return parts["base"] + parts["broke_healthy"] + parts["length"], parts

    def compute(self, legs, cand):
        kind, value = self._base(legs)
        return (None, {"reason": str(value)}) if kind != "score" else self._finish(value, cand)

    def decide(self, partial, cand):
        from aorlta.reward.rewardfn.base import Decided, NeedMore, Unscored

        kind, value = self._base(partial)
        if kind == "pending":
            return NeedMore((value,))
        return Unscored(value) if kind == "unscored" else Decided(*self._finish(value, cand))

    @staticmethod
    def claims_clean(reward, parts):
        return reward is not None and isinstance(parts, dict) and parts.get("base", 0.0) > 0.0


def register() -> None:
    from aorlta import plugins

    plugins.register_task(plugins.Task("qa-expert", records))
    plugins.register_parse_adapter("qa-expert.parse", ExpertParse)
    plugins.register_feedback_template("qa-expert.feedback", ExpertFeedback)
    plugins.register_profile("qa-expert", HERE / "qa-expert.yaml")
    plugins.register_check_adapter("qa-expert.match", ExpertMatch)
    plugins.register_reward_fn("qa-expert.v1", ExpertV1)


register()
