"""A starter for an ISA-fuzzer-style task (examples/verticals/README.md): the model writes a short test program for a
stated goal, and a check validates it. Every part marked PLACEHOLDER is to be replaced with your own.

A pool is a directory holding `tasks.jsonl`, one task per line: `{"id", "goal", "min_lines", "max_lines",
"must_use"}`. The model is asked for a program in `toy-asm`, a made-up text format declared in `USER` (one
instruction per line: `NOP`, `MOV rD, rS|#imm`, or `ADD|SUB|MUL|AND|OR|XOR rD, rA, rB|#imm`, registers r0-r7), in one
fenced block. A reply with no fenced block gets one repair turn that says so (this module's parse and feedback); there
is no compile step (`rollout.compile_adapter: none`). The reward service runs one leg, `ref.program`, on the CPU: the
last fenced block against the toy grammar, the task's line range and its required instructions. With a command in the
profile's `isa-fuzzer.check.tool`, a program that passes those is then the tool's to decide: it reads `{"goal",
"program"}` as JSON on stdin and prints `{"verdict": "pass"}` or `{"verdict": "fail"}`; anything else leaves the
sample unscored.

Registered names: task `isa-fuzzer`; parse `isa-fuzzer.parse`; feedback `isa-fuzzer.feedback`; profile `isa-fuzzer`
(isa-fuzzer.yaml); check adapter `isa-fuzzer.check`; reward function `isa-fuzzer.v1`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
# PLACEHOLDER: the format and the prompt. Declare your real program format here, or point the model at its manual.
SYSTEM = "You write short test programs in the toy-asm format."
USER = ("Goal: {goal}\n\n"
        "Write a toy-asm program of {min_lines} to {max_lines} lines. toy-asm has one instruction per line: NOP; "
        "MOV rD, rS or MOV rD, #imm; or OP rD, rA, rB or OP rD, rA, #imm with OP one of ADD, SUB, MUL, AND, OR, XOR. "
        "Registers are r0 to r7; an immediate is # and a whole number of at most six digits, which may be negative. "
        "Give the program in one fenced code block.")
BLOCK_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
REG, IMM = r"r[0-7]", r"#-?\d{1,6}"
LINE_RE = re.compile(rf"NOP|MOV\s+{REG}\s*,\s*(?:{REG}|{IMM})"
                     rf"|(?:ADD|SUB|MUL|AND|OR|XOR)\s+{REG}\s*,\s*{REG}\s*,\s*(?:{REG}|{IMM})")


def load_rows(pool_dir) -> list[dict]:
    rows = [json.loads(line) for line in (Path(pool_dir) / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{pool_dir}/tasks.jsonl lists an id twice")
    return rows


def row_sha256(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()


def final_program(reply: str) -> str | None:
    """The last fenced block's text, or None."""
    found = BLOCK_RE.findall(reply)
    return found[-1] if found else None


def program_problem(program: str, row: dict) -> str | None:
    """PLACEHOLDER check: None for a program that keeps the toy grammar, the task's line range and its required
    instructions; otherwise the failure's reason (`malformed`, `wrong_length`, `missing_op`). A real check runs the
    program (an assembler, a simulator, your fuzzer's harness) instead, or after this, through the `tool` setting."""
    lines = [ln.strip() for ln in program.splitlines() if ln.strip()]
    if not all(LINE_RE.fullmatch(ln) for ln in lines):
        return "malformed"
    if not row["min_lines"] <= len(lines) <= row["max_lines"]:
        return "wrong_length"
    if not set(row.get("must_use", ())) <= {ln.split()[0] for ln in lines}:
        return "missing_op"
    return None


def messages(row: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(goal=row["goal"], min_lines=row["min_lines"],
                                                    max_lines=row["max_lines"])}]


def records(pool_dir, tokenizer, thinking):
    from aorlta.ids import sha256_ids
    from aorlta.rollout.templates import render_turn0
    from aorlta.schemas import PromptRecord, TaskRef

    out = []
    for row in load_rows(pool_dir):
        msgs = messages(row)
        out.append(PromptRecord(
            prompt_id=f"isa-fuzzer:{row['id']}", turn0_ids=None, messages=msgs,
            prompt_ids_sha256=sha256_ids(render_turn0(tokenizer, msgs, thinking)),
            task=TaskRef(contract_id=f"isa-fuzzer:{row['id']}", parent_id=None, parent_healthy=None,
                         reference_kind=None, task_sha256=row_sha256(row)),
            source="standin"))
    return out


class IsaParse:
    """`isa-fuzzer.parse`: a reply with a fenced block is a program (nothing to compile); one without is rejected with
    a message, which `isa-fuzzer.feedback` turns into a repair turn. Whether the program is any good is the check's to
    say, not the parse's."""

    name = "isa-fuzzer.parse"

    def __init__(self):
        from aorlta.rollout.templates import IM_END_ID

        self.reply_stops = (IM_END_ID,)

    def check_prompts(self, prompts, thinking, max_repair_turns=0):
        return None

    def check(self, text, thinking, prompt):
        from aorlta.rollout.compile_check import ParseOutcome

        reply = text.rsplit("</think>", 1)[-1] if thinking == "on" else text
        program = final_program(reply)
        if program is None:
            return ParseOutcome("rejected", rejection=(
                "Your reply has no fenced code block. Give the toy-asm program in one fenced code block.",
                "no_program"))
        return ParseOutcome("edits", code=program)


class IsaFeedback:
    name = "isa-fuzzer.feedback"

    def build(self, errors, max_chars):
        from aorlta.rollout.feedback import build_feedback

        return build_feedback(errors, max_chars)

    def build_apply(self, message, reason):
        from aorlta.rollout.feedback import Feedback

        return Feedback(message, 0, 0)


class IsaCheck:
    """`isa-fuzzer.check`, leg `ref.program`: `fail` `no_program` for a reply with no fenced block, `fail` with
    `program_problem`'s reason for a program it rejects, then clean, or the tool's (settings `tool`, a command as a
    list) `pass` clean and `fail` `tool_rejected`; unsettled when the task is unknown, the reply cannot be read, or the
    tool gives no verdict."""

    legs = ("ref.program",)
    resource = "local_cpu"

    def __init__(self, ctx):
        from aorlta.reward.adapters.base import ProcessRunner
        from aorlta.reward.adapters.toolchain import file_identity

        self.name, self.ctx = ctx.name, ctx
        self.tool = ctx.settings.get("tool")
        self.tool_timeout_s = float(ctx.settings.get("tool_timeout_s", 60.0))
        self.version = "1"  # cached legs are keyed by it: the tool, which decides what the format passes, is part of it
        if self.tool:
            files = {a: file_identity(a) for a in self.tool if isinstance(a, str) and Path(a).is_file()}
            exe = shutil.which(self.tool[0]) if isinstance(self.tool[0], str) else None  # as PATH finds it, if bare
            if exe:
                files[f"exe:{self.tool[0]}"] = f"{os.path.realpath(exe)}:{file_identity(exe)}"
            ident = json.dumps({"tool": list(self.tool), "timeout_s": self.tool_timeout_s, "files": files},
                               sort_keys=True)
            self.version = f"1+tool-{hashlib.sha256(ident.encode()).hexdigest()[:16]}"
        self.runner = ProcessRunner()  # kills the tool's whole process group on a timeout or at close()
        self.rows = {}
        for pool in ctx.pools:
            if (Path(pool) / "tasks.jsonl").is_file():
                for row in load_rows(pool):
                    self.rows[row_sha256(row)] = row

    def run(self, job):
        from aorlta.reward.adapters.base import make_result, unsettled

        t0 = self.ctx.clock()
        try:
            row = self.rows.get(job.task_sha256)
            response = job.inputs.get("response")
            if job.leg != "ref.program" or row is None or not response:
                return unsettled(job, "adapter_error", t_start=t0, t_end=self.ctx.clock())
            program = final_program(Path(response).read_text(encoding="utf-8").rsplit("</think>", 1)[-1])
            if program is None:
                return make_result(job, "fail", "no_program", t_start=t0, t_end=self.ctx.clock())
            problem = program_problem(program, row)
            if problem is not None:
                return make_result(job, "fail", problem, t_start=t0, t_end=self.ctx.clock())
            if not self.tool:
                return make_result(job, "clean", t_start=t0, t_end=self.ctx.clock())
            verdict = self._tool(row, program)
            if verdict == "pass":
                return make_result(job, "clean", t_start=t0, t_end=self.ctx.clock())
            if verdict == "fail":
                return make_result(job, "fail", "tool_rejected", t_start=t0, t_end=self.ctx.clock())
            return unsettled(job, "tool_no_verdict", t_start=t0, t_end=self.ctx.clock())
        except Exception:  # noqa: BLE001 - an adapter never raises
            return unsettled(job, "adapter_error", t_start=t0, t_end=self.ctx.clock())

    def _tool(self, row, program) -> str | None:
        proc = self.runner.run(list(self.tool), input=json.dumps({"goal": row["goal"], "program": program}),
                               timeout_s=min(self.tool_timeout_s, self.ctx.timeout_s))
        if proc.timed_out or proc.killed or proc.launch_error or proc.returncode != 0 or not proc.stdout.strip():
            return None
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
        except ValueError:
            return None
        v = out.get("verdict") if isinstance(out, dict) else None
        return v if v in ("pass", "fail") else None

    def controls(self):
        return []

    def health(self):
        return {"name": self.name, "responsive": True, "tasks": len(self.rows)}

    def close(self):
        self.runner.close()


class IsaV1:
    """`isa-fuzzer.v1`, a PLACEHOLDER reward: +1.0 for a program the check passes, a value of its own for each failure
    it names (`BASE`), then -0.05 x the share of the token budget used. An unsettled leg, or a reason `BASE` does not
    name, leaves the sample unscored."""

    name = "isa-fuzzer.v1"
    version = "1"
    legs = ("ref.program",)
    # TODO(your vertical): replace these values with your real reward, and the check's reasons with your tool's. Keep:
    #   - the total in [-1.55, 1.0], its parts exactly `base`, `broke_healthy` and `length`, adding up to it (the
    #     service recomputes it and leaves anything else unscored, with an alarm): so a base in [-1.5, 1.0];
    #   - an unsettled or missing leg, or a reason you did not name, unscored: never clean, never a score;
    #   - each failure its own value, ordered by how close it came, so the answers to one prompt get different
    #     rewards (GRPO learns only from the differences);
    #   - `claims_clean` true only when the leg is clean.
    BASE = {"tool_rejected": -0.3, "missing_op": -0.5, "wrong_length": -0.6, "malformed": -0.8, "no_program": -1.0}

    def __init__(self, profile=None):
        from aorlta.reward.rewardfn.base import code_sha256

        self.settings = {}
        self.code_sha256 = code_sha256(__file__)

    def _base(self, legs):
        leg = legs.get("ref.program")
        if leg is None:
            return "pending", "ref.program"
        if leg.state == "clean":
            return "score", 1.0
        if leg.state == "fail" and leg.reason in self.BASE:
            return "score", self.BASE[leg.reason]
        return "unscored", f"ref.program:{leg.reason or leg.state}"

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

    plugins.register_task(plugins.Task("isa-fuzzer", records))
    plugins.register_parse_adapter("isa-fuzzer.parse", IsaParse)
    plugins.register_feedback_template("isa-fuzzer.feedback", IsaFeedback)
    plugins.register_profile("isa-fuzzer", HERE / "isa-fuzzer.yaml")
    plugins.register_check_adapter("isa-fuzzer.check", IsaCheck)
    plugins.register_reward_fn("isa-fuzzer.v1", IsaV1)


register()
