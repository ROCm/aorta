"""A starter for a sanitizer- or NaN-triage task (examples/verticals/README.md): the model reads a tool's report and
answers with a structured verdict. Every part marked PLACEHOLDER is to be replaced with your own.

A pool is a directory holding `reports.jsonl`, one report per line: `{"id", "report", "label"}`. The reports that come
with it are toy text written for this starter, not the output of any tool. The model is shown the report and asked
for a JSON object `{"category": ..., "rationale": ...}`, with the category one of `CATEGORIES`. A reply with no JSON
object gets one repair turn that says so (this module's parse and feedback); there is no compile step
(`rollout.compile_adapter: none`). The reward service runs one leg, `ref.verdict`, on the CPU; as a placeholder it
checks only that the verdict has exactly the two keys, a category from the list and a short rationale. It does not
yet compare the category with the report's `label`: that, or running the real tool, is the check to write.

Registered names: task `sanitizer-triage`; parse `sanitizer-triage.parse`; feedback `sanitizer-triage.feedback`;
profile `sanitizer-triage` (sanitizer-triage.yaml); check adapter `sanitizer-triage.check`; reward function
`sanitizer-triage.v1`.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
# PLACEHOLDER: the closed list of verdicts, and the prompt. Use your own tool's categories.
CATEGORIES = ("no_issue", "memory", "race", "numeric", "inconclusive")
KEYS = ("category", "rationale")
MAX_RATIONALE_CHARS = 400
SYSTEM = "You triage reports from a checking tool."
USER = ("Report:\n{report}\n\nReply with one JSON object with exactly the keys \"category\" and \"rationale\": the "
        "category is one of " + ", ".join(CATEGORIES) + ", and the rationale is one or two sentences.")
OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def load_rows(pool_dir) -> list[dict]:
    rows = [json.loads(line) for line in (Path(pool_dir) / "reports.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{pool_dir}/reports.jsonl lists an id twice")
    return rows


def row_sha256(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()


def final_verdict(reply: str) -> dict | None:
    """The last JSON object in the reply that parses (no nesting: a verdict is flat), or None."""
    for text in reversed(OBJECT_RE.findall(reply)):
        try:
            obj = json.loads(text)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def verdict_problem(verdict: dict) -> str | None:
    """PLACEHOLDER check: None for a verdict with exactly `KEYS`, a category from `CATEGORIES` and a non-empty
    rationale of at most `MAX_RATIONALE_CHARS`; otherwise the failure's reason (`bad_keys`, `bad_category`,
    `bad_rationale`)."""
    if set(verdict) != set(KEYS):
        return "bad_keys"
    if verdict["category"] not in CATEGORIES:
        return "bad_category"
    r = verdict["rationale"]
    if not (isinstance(r, str) and r.strip() and len(r) <= MAX_RATIONALE_CHARS):
        return "bad_rationale"
    return None


def messages(row: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(report=row["report"])}]


def records(pool_dir, tokenizer, thinking):
    from aorlta.ids import sha256_ids
    from aorlta.rollout.templates import render_turn0
    from aorlta.schemas import PromptRecord, TaskRef

    out = []
    for row in load_rows(pool_dir):
        msgs = messages(row)
        out.append(PromptRecord(
            prompt_id=f"sanitizer-triage:{row['id']}", turn0_ids=None, messages=msgs,
            prompt_ids_sha256=sha256_ids(render_turn0(tokenizer, msgs, thinking)),
            task=TaskRef(contract_id=f"sanitizer-triage:{row['id']}", parent_id=None, parent_healthy=None,
                         reference_kind=None, task_sha256=row_sha256(row)),
            source="standin"))
    return out


class TriageParse:
    """`sanitizer-triage.parse`: a reply with a JSON object is a verdict (nothing to compile); one without is rejected
    with a message, which `sanitizer-triage.feedback` turns into a repair turn. Whether the verdict is well formed or
    right is the check's to say, not the parse's."""

    name = "sanitizer-triage.parse"

    def __init__(self):
        from aorlta.rollout.templates import IM_END_ID

        self.reply_stops = (IM_END_ID,)

    def check_prompts(self, prompts, thinking, max_repair_turns=0):
        return None

    def check(self, text, thinking, prompt):
        from aorlta.rollout.compile_check import ParseOutcome

        reply = text.rsplit("</think>", 1)[-1] if thinking == "on" else text
        verdict = final_verdict(reply)
        if verdict is None:
            return ParseOutcome("rejected", rejection=(
                "Your reply has no JSON object. Reply with one JSON object with the keys \"category\" and "
                "\"rationale\".", "no_verdict"))
        return ParseOutcome("edits", code=json.dumps(verdict, sort_keys=True))


class TriageFeedback:
    name = "sanitizer-triage.feedback"

    def build(self, errors, max_chars):
        from aorlta.rollout.feedback import build_feedback

        return build_feedback(errors, max_chars)

    def build_apply(self, message, reason):
        from aorlta.rollout.feedback import Feedback

        return Feedback(message, 0, 0)


class TriageCheck:
    """`sanitizer-triage.check`, leg `ref.verdict`: `fail` `no_verdict` for a reply with no JSON object, `fail` with
    `verdict_problem`'s reason for a verdict it rejects, clean otherwise; unsettled when the task is unknown or the
    reply cannot be read."""

    version = "1"
    legs = ("ref.verdict",)
    resource = "local_cpu"

    def __init__(self, ctx):
        self.name, self.ctx = ctx.name, ctx
        self.rows = {}
        for pool in ctx.pools:
            if (Path(pool) / "reports.jsonl").is_file():
                for row in load_rows(pool):
                    self.rows[row_sha256(row)] = row

    def run(self, job):
        from aorlta.reward.adapters.base import make_result, unsettled

        t0 = self.ctx.clock()
        try:
            row = self.rows.get(job.task_sha256)
            response = job.inputs.get("response")
            if job.leg != "ref.verdict" or row is None or not response:
                return unsettled(job, "adapter_error", t_start=t0, t_end=self.ctx.clock())
            verdict = final_verdict(Path(response).read_text(encoding="utf-8").rsplit("</think>", 1)[-1])
            if verdict is None:
                return make_result(job, "fail", "no_verdict", t_start=t0, t_end=self.ctx.clock())
            problem = verdict_problem(verdict)
            if problem is not None:
                return make_result(job, "fail", problem, t_start=t0, t_end=self.ctx.clock())
            # TODO(your vertical): the placeholder stops at the verdict's form. Compare `verdict["category"]` with
            # `row["label"]` (a `fail` `wrong_category` of its own), or run the real tool on the case and read its
            # result; a tool that gives no result is unsettled, never clean.
            return make_result(job, "clean", t_start=t0, t_end=self.ctx.clock())
        except Exception:  # noqa: BLE001 - an adapter never raises
            return unsettled(job, "adapter_error", t_start=t0, t_end=self.ctx.clock())

    def controls(self):
        return []

    def health(self):
        return {"name": self.name, "responsive": True, "reports": len(self.rows)}


class TriageV1:
    """`sanitizer-triage.v1`, a PLACEHOLDER reward: +1.0 for a verdict the check passes, a value of its own for each
    failure it names (`BASE`), then -0.05 x the share of the token budget used. An unsettled leg, or a reason `BASE`
    does not name, leaves the sample unscored."""

    name = "sanitizer-triage.v1"
    version = "1"
    legs = ("ref.verdict",)
    # TODO(your vertical): replace these values with your real reward (a wrong category from the check above, say,
    # nearer +1.0 than a malformed verdict). Keep:
    #   - the total in [-1.55, 1.0], its parts exactly `base`, `broke_healthy` and `length`, adding up to it (the
    #     service recomputes it and leaves anything else unscored, with an alarm): so a base in [-1.5, 1.0];
    #   - an unsettled or missing leg, or a reason you did not name, unscored: never clean, never a score;
    #   - each failure its own value, so the answers to one prompt get different rewards (GRPO learns only from the
    #     differences);
    #   - `claims_clean` true only when the leg is clean.
    BASE = {"bad_rationale": -0.4, "bad_category": -0.6, "bad_keys": -0.8, "no_verdict": -1.0}

    def __init__(self, profile=None):
        from aorlta.reward.rewardfn.base import code_sha256

        self.settings = {}
        self.code_sha256 = code_sha256(__file__)

    def _base(self, legs):
        leg = legs.get("ref.verdict")
        if leg is None:
            return "pending", "ref.verdict"
        if leg.state == "clean":
            return "score", 1.0
        if leg.state == "fail" and leg.reason in self.BASE:
            return "score", self.BASE[leg.reason]
        return "unscored", f"ref.verdict:{leg.reason or leg.state}"

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

    plugins.register_task(plugins.Task("sanitizer-triage", records))
    plugins.register_parse_adapter("sanitizer-triage.parse", TriageParse)
    plugins.register_feedback_template("sanitizer-triage.feedback", TriageFeedback)
    plugins.register_profile("sanitizer-triage", HERE / "sanitizer-triage.yaml")
    plugins.register_check_adapter("sanitizer-triage.check", TriageCheck)
    plugins.register_reward_fn("sanitizer-triage.v1", TriageV1)


register()
