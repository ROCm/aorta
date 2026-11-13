"""A worked example of a task of your own, as a plugin (docs/TUTORIAL.md): write a HIP kernel from its description.

Each task in `tasks.json` names a kernel, its exact signature, what it computes and how it is launched. The model
answers with a HIP source in one ```cpp block. The rollout compiles it with the stand-in compile check (hipcc -c) and
gives up to `rollout.max_repair_turns` repair turns with the compiler's errors; a reply that has no code block, or
whose code does not define the kernel, gets a repair turn that says so (this module's parse and feedback). The reward
service then runs three legs: the stand-in parse gate and compile gate, and `ref.native`, this module's check, which
builds the candidate with a generated host program, runs it on a GPU and compares every output element with a host
reference computed from the same inputs. The reward is `hipk.v1` (below).

An optional profile, `hipk-speed` (hipk-speed.yaml), adds a fourth leg after `ref.native`: `ref.speedup`, the
candidate timed against the task's reference kernel (`hipk.time`, judged by `aorlta.reward.timing`), scored by
`hipk.speed`, which grades hipk.v1's clean reward by the measured speedup (docs/REWARDS.md, "Timing").

Registered names: task `hipk`; parse `hipk.parse`; feedback `hipk.feedback`; profiles `hipk-gpu` (hipk-gpu.yaml) and
`hipk-speed` (hipk-speed.yaml); check adapters `hipk.run` and `hipk.time`; reward functions `hipk.v1` and
`hipk.speed`.

Run as a script, it checks the tasks themselves with the reference kernels in reference/:
    python examples/hip-kernels/hip_kernels.py selfcheck --hipcc /opt/rocm/bin/hipcc [--gpu 0] [--build-only]
and times one task's kernel against its reference with the timing check (a GPU is needed):
    python examples/hip-kernels/hip_kernels.py timecheck --task vector_add [--candidate my.hip] [--gpu 0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYSTEM = "You are an expert AMD GPU kernel engineer."
USER = (
    "Write a HIP kernel with exactly this signature:\n\n```cpp\n{signature}\n```\n\n"
    "It must compute {description}. It is launched {launch}.\n\n"
    "Write a complete HIP C++ source file: include <hip/hip_runtime.h>, define the kernel, and write no main "
    "function. Reply with the source in a single ```cpp code block."
)
RESULT_TAG = "AORLTA_RESULT "
SENTINEL = "-7777.0f"  # what an output element holds before the kernel runs: one it never writes is a mismatch
GUARD = 256  # elements past the end of every device array, which the kernel must leave alone
GUARD_VALUE = "31337.0f"


# --- the tasks -------------------------------------------------------------------------------------


def load_tasks(pool_dir: str | Path) -> list[dict]:
    """The pool's tasks, in file order: `<pool_dir>/tasks.json`, a list of task specs."""
    tasks = json.loads((Path(pool_dir) / "tasks.json").read_text(encoding="utf-8"))
    ids = [t["id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{pool_dir}/tasks.json lists a task id twice")
    return tasks


def task_sha256(task: dict) -> str:
    """What the check reads for a task, hashed: part of every check's cache key, and how the check finds the task."""
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode("utf-8")).hexdigest()


def messages(task: dict) -> list[dict]:
    user = USER.format(signature=task["signature"], description=task["description"], launch=task["launch"])
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def records(pool_dir, tokenizer, thinking):
    """The prompt records: one per task, prompt id `hipk:<id>`."""
    from aorlta.ids import sha256_ids
    from aorlta.rollout.templates import render_turn0
    from aorlta.schemas import PromptRecord, TaskRef

    out = []
    for task in load_tasks(pool_dir):
        msgs = messages(task)
        out.append(PromptRecord(
            prompt_id=f"hipk:{task['id']}", turn0_ids=None, messages=msgs,
            prompt_ids_sha256=sha256_ids(render_turn0(tokenizer, msgs, thinking)),
            task=TaskRef(contract_id=f"hipk:{task['id']}", parent_id=None, parent_healthy=None, reference_kind=None,
                         task_sha256=task_sha256(task)),
            source="standin"))
    return out


# --- the rollout: parse and feedback ------------------------------------------------------------


def defines_kernel(source: str, kernel: str) -> bool:
    return re.search(r"__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?void\s+" + re.escape(kernel) + r"\s*\(",
                     source) is not None


class HipkParse:
    """`hipk.parse`: the last ```cpp block (the stand-in parse's rule). No block, or a block that does not define the
    task's kernel, is rejected with a message, which `hipk.feedback` turns into a repair turn."""

    name = "hipk.parse"

    def __init__(self):
        from aorlta.rollout.templates import IM_END_ID

        self.reply_stops = (IM_END_ID,)

    def check_prompts(self, prompts, thinking, max_repair_turns=0):
        for p in prompts:
            if not p.prompt_id.startswith("hipk:"):
                raise ValueError(f"hipk.parse reads hipk prompts, got {p.prompt_id!r}")

    def check(self, text, thinking, prompt):
        from aorlta.rollout.compile_check import ParseOutcome, parse_code

        code = parse_code(text, thinking)
        if code is None:
            return ParseOutcome("rejected", rejection=(
                "Your reply has no complete ```cpp code block. Reply with the whole HIP source in one ```cpp block.",
                "no_code_block"))
        kernel = prompt.prompt_id.split(":", 1)[1]
        if not defines_kernel(code, kernel):
            return ParseOutcome("rejected", rejection=(
                f"Your source does not define the kernel `{kernel}`. Define it as a __global__ function with exactly "
                f"the signature asked for, and reply with the whole source in one ```cpp block.", "no_kernel"))
        return ParseOutcome("edits", code=code)


class HipkFeedback:
    """`hipk.feedback`: the stand-in compile feedback, and a rejected answer's message as its own repair turn."""

    name = "hipk.feedback"

    def build(self, errors, max_chars):
        from aorlta.rollout.feedback import build_feedback

        return build_feedback(errors, max_chars)

    def build_apply(self, message, reason):
        from aorlta.rollout.feedback import Feedback

        return Feedback(message, 0, 0)


# --- the check: build, run, compare ------------------------------------------------------------


def _f(x) -> str:
    """A float as a C++ float literal."""
    return f"{float(x)!r}f"


def harness_source(task: dict, kernel_source: str) -> str:
    """The candidate's source followed by a host program that fills the inputs, runs the kernel once, and prints one
    `AORLTA_RESULT {json}` line: `ok`, `mismatch` (an output element differs from the host reference beyond the
    tolerance), `out_of_bounds` (a write past the end of an array) or `hip_error`."""
    lines = [kernel_source.rstrip(), "", "// ---- AORLTA harness ----", "#include <cmath>", "#include <cstdint>",
             "#include <cstdio>", "#include <vector>", "",
             "static void fill(std::vector<float>& v, uint32_t seed, float lo, float hi) {",
             "  for (auto& x : v) { seed = seed * 1664525u + 1013904223u; "
             "x = lo + (hi - lo) * float(seed >> 8) / 16777216.0f; }",
             "}", "",
             "#define CHECK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) { "
             f"printf(\"{RESULT_TAG}{{\\\"status\\\": \\\"hip_error\\\", \\\"at\\\": \\\"%s\\\"}}\\n\", #call); "
             "return 0; } } while (0)", "",
             "int main() {"]
    for ctype, name, value in task["scalars"]:
        lines.append(f"  const {ctype} {name} = {value};")
    arrays = task["arrays"]
    for i, a in enumerate(arrays):
        n = a["size"]
        lines.append(f"  std::vector<float> h_{a['name']}(({n}) + {GUARD}, {GUARD_VALUE});")
        if a["role"] == "out":
            lines.append(f"  for (long i = 0; i < ({n}); ++i) h_{a['name']}[i] = {SENTINEL};")
        else:
            lines.append(f"  {{ std::vector<float> v({n}); fill(v, {1000 + 17 * i}u, {_f(a.get('lo', -1.0))}, "
                         f"{_f(a.get('hi', 1.0))}); for (long i = 0; i < ({n}); ++i) h_{a['name']}[i] = v[i]; }}")
        lines.append(f"  std::vector<float> r_{a['name']} = h_{a['name']};")
    lines.append("  {")
    for a in arrays:
        lines.append(f"    float* {a['name']} = r_{a['name']}.data();")
    lines.append("    " + task["reference"])
    lines.append("  }")
    for a in arrays:
        lines += [f"  float* d_{a['name']} = nullptr;",
                  f"  CHECK(hipMalloc(&d_{a['name']}, h_{a['name']}.size() * sizeof(float)));",
                  f"  CHECK(hipMemcpy(d_{a['name']}, h_{a['name']}.data(), h_{a['name']}.size() * sizeof(float), "
                  "hipMemcpyHostToDevice));"]
    names = {a["name"] for a in arrays}
    args = ", ".join(f"d_{x}" if x in names else x for x in task["args"])
    lines += [f"  {task['kernel']}<<<{task['grid']}, {task['block']}>>>({args});",
              "  CHECK(hipGetLastError());", "  CHECK(hipDeviceSynchronize());",
              "  long checked = 0, mismatches = 0, first = -1, oob = 0;"]
    for a in arrays:
        n = a["size"]
        lines += [f"  CHECK(hipMemcpy(h_{a['name']}.data(), d_{a['name']}, h_{a['name']}.size() * sizeof(float), "
                  "hipMemcpyDeviceToHost));",
                  f"  for (long i = ({n}); i < (long)h_{a['name']}.size(); ++i) if (h_{a['name']}[i] != {GUARD_VALUE}) "
                  "++oob;"]
        if a["role"] != "in":
            lines += [f"  for (long i = 0; i < ({n}); ++i) {{",
                      f"    float got = h_{a['name']}[i], want = r_{a['name']}[i];",
                      f"    bool same = (std::isnan(got) && std::isnan(want)) || std::fabs(got - want) <= "
                      f"{_f(task['atol'])} + {_f(task['rtol'])} * std::fabs(want);",
                      "    ++checked;",
                      "    if (!same) { if (first < 0) first = checked - 1; ++mismatches; }",
                      "  }"]
    lines += ["  const char* status = oob ? \"out_of_bounds\" : (mismatches ? \"mismatch\" : \"ok\");",
              f"  printf(\"{RESULT_TAG}{{\\\"status\\\": \\\"%s\\\", \\\"checked\\\": %ld, \\\"mismatches\\\": %ld, "
              "\\\"first\\\": %ld, \\\"out_of_bounds\\\": %ld}\\n\", status, checked, mismatches, first, oob);",
              "  return 0;", "}", ""]
    return "\n".join(lines)


def read_result(stdout: str) -> dict | None:
    """The harness's result line, or None."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_TAG):
            try:
                out = json.loads(line[len(RESULT_TAG):])
            except ValueError:
                return None
            return out if isinstance(out, dict) and isinstance(out.get("status"), str) else None
    return None


def verdict(build_rc, build_stderr, run, task) -> tuple[str, str | None, list[dict]]:
    """A leg's (state, reason, findings) from the harness's build and run. `run` is None when the build failed, else
    a ProcResult. Only what is the candidate's is blamed on it: a build failure is the candidate's when the compiler
    names its kernel or `main` (its compile alone passed in gate.compile, so the signature is what differs), a hang
    past the limit and a crash are its, and anything else is unsettled, never clean."""
    if build_rc != 0:
        if re.search(r"\b(" + re.escape(task["kernel"]) + r"|main)\b", build_stderr or ""):
            return "fail", "bad_signature", []
        return "unsettled", "harness_build_failed", []
    if run.launch_error or run.killed:
        return "unsettled", "run_not_started", []
    if run.timed_out:
        return "fail", "timeout", []
    result = read_result(run.stdout)
    if result is None:
        return ("fail", "crash", []) if run.returncode not in (0, None) else ("unsettled", "no_result", [])
    status = result["status"]
    if status == "ok" and run.returncode == 0 and result.get("checked", 0) > 0:
        return "clean", None, []
    if status in ("mismatch", "out_of_bounds"):
        return "fail", "wrong_output", []
    if status == "hip_error":
        return "fail", "crash", []
    return "unsettled", "no_result", []


class HipkRun:
    """`hipk.run`, leg `ref.native`: the candidate built with its task's harness (hipcc, settings `hipcc`, leg setting
    `target`), run on this worker's GPU (settings `gpus`: worker n runs on `gpus[(n - 1) % len(gpus)]`, by
    HIP_VISIBLE_DEVICES) for at most `run_timeout_s`, and its outputs compared with the host reference. Clean only
    when every output element matches and nothing was written past an array's end."""

    version = "1"
    legs = ("ref.native",)
    resource = "own_gpu"

    def __init__(self, ctx):
        from aorlta.reward.adapters.base import ProcessRunner

        self.name, self.ctx = ctx.name, ctx
        s = ctx.settings
        self.hipcc = s.get("hipcc") or "/opt/rocm/bin/hipcc"
        gpus = list(s.get("gpus") or [0])
        seq = int(ctx.worker_id.rsplit("/", 1)[-1]) if ctx.worker_id.rsplit("/", 1)[-1].isdigit() else 1
        self.gpu = int(gpus[(seq - 1) % len(gpus)])
        self.tasks = {}
        for pool in ctx.pools:
            if (Path(pool) / "tasks.json").is_file():
                for task in load_tasks(pool):
                    self.tasks[task_sha256(task)] = task
        self.runner = ProcessRunner()

    def run(self, job):
        from aorlta.reward.adapters.base import make_result, run_dir, unsettled

        t0 = self.ctx.clock()
        try:
            task = self.tasks.get(job.task_sha256)
            source = job.inputs.get("source")
            if job.leg != "ref.native" or task is None or not source or not job.output_dir:
                return unsettled(job, "adapter_error", t_start=t0, t_end=self.ctx.clock())
            d = run_dir(job)
            (d / "harness.hip").write_text(harness_source(task, Path(source).read_text(encoding="utf-8")),
                                           encoding="utf-8")
            target = job.leg_settings.get("target", "gfx950")
            budget = max(1.0, self.ctx.timeout_s)
            t_build = time.monotonic()
            build = self.runner.run([self.hipcc, f"--offload-arch={target}", "-O2", "-o", str(d / "harness"),
                                     str(d / "harness.hip")], timeout_s=budget, cwd=str(d))
            if build.timed_out or build.killed or build.launch_error:
                return unsettled(job, "harness_build_failed", t_start=t0, t_end=self.ctx.clock())
            (d / "build.log").write_text(build.stderr[-20000:], encoding="utf-8")
            run = None
            if build.returncode == 0:
                left = budget - (time.monotonic() - t_build)
                limit = min(float(job.leg_settings.get("run_timeout_s", 20.0)), left)
                if limit <= 0:
                    return unsettled(job, "no_time_left", t_start=t0, t_end=self.ctx.clock())
                env = dict(os.environ, HIP_VISIBLE_DEVICES=str(self.gpu))
                run = self.runner.run([str(d / "harness")], timeout_s=limit, env=env, cwd=str(d))
                (d / "run.log").write_text((run.stdout + "\n" + run.stderr)[-20000:], encoding="utf-8")
            state, reason, findings = verdict(build.returncode, build.stderr, run, task)
            return make_result(job, state, reason, findings=findings, t_start=t0, t_end=self.ctx.clock())
        except Exception:  # noqa: BLE001 - an adapter never raises
            return unsettled(job, "adapter_error", t_start=t0, t_end=self.ctx.clock())

    def controls(self):
        return []

    def health(self):
        return {"name": self.name, "responsive": True, "gpu": self.gpu, "tasks": len(self.tasks)}

    def close(self):
        self.runner.close()


# --- the reward --------------------------------------------------------------------------------


class HipkV1:
    """`hipk.v1`: -1.0 when the parse gate fails, -0.9 for a compile error, -0.8 when the kernel's signature is not
    the one asked for, -0.6 for a crash or a hang, -0.5 for a wrong output, +1.0 when every output matches; then
    -0.05 x the share of the token budget the answer used. A leg that did not settle leaves the sample unscored:
    an unscored sample is never counted clean. The parts are the service's (`base`, `broke_healthy` (always 0),
    `length`), so the reward stays within its range."""

    name = "hipk.v1"
    version = "1"
    legs = ("gate.parse", "gate.compile", "ref.native")
    RUN = {"wrong_output": -0.5, "crash": -0.6, "timeout": -0.6, "bad_signature": -0.8}

    def __init__(self, profile=None):
        from aorlta.reward.rewardfn.base import code_sha256

        self.settings = {}
        self.code_sha256 = code_sha256(__file__)

    def _base(self, legs) -> tuple[str, float | str]:
        """("score", base), ("unscored", reason) or ("pending", leg), reading the legs in order."""
        for name in self.legs:
            if name not in legs:
                return "pending", name
            leg = legs[name]
            if leg.state in ("unsettled", "skipped_by_plan", "finding", "not_run"):
                return "unscored", f"{name}:{leg.reason or leg.state}"
            if leg.state == "fail":
                if name == "gate.parse":
                    return "score", -1.0
                if name == "gate.compile":
                    return "score", -0.9 if leg.reason == "compile_error" else -1.0
                if leg.reason in self.RUN:
                    return "score", self.RUN[leg.reason]
                return "unscored", f"{name}:{leg.reason}"
        return "score", 1.0

    @staticmethod
    def _finish(base, cand):
        length = -0.05 * (cand.response_tokens / cand.max_new_tokens)
        parts = {"base": float(base), "broke_healthy": 0.0, "length": length}
        return parts["base"] + parts["broke_healthy"] + parts["length"], parts

    def compute(self, legs, cand):
        kind, value = self._base(legs)
        if kind != "score":
            return None, {"reason": str(value)}
        return self._finish(value, cand)

    def decide(self, partial, cand):
        from aorlta.reward.rewardfn.base import Decided, NeedMore, Unscored

        kind, value = self._base(partial)
        if kind == "pending":
            return NeedMore((value,))
        if kind == "unscored":
            return Unscored(value)
        return Decided(*self._finish(value, cand))

    @staticmethod
    def claims_clean(reward, parts):
        return reward is not None and isinstance(parts, dict) and parts.get("base", 0.0) > 0.0


# --- the timing check (optional: hipk-speed.yaml) ----------------------------------------------

TIMING_TAG = "AORLTA_TIMING "
LAUNCHER = "aorlta_launch_candidate"
# What the candidate's object may not define: a symbol of such a name in the executable takes the place of the HIP or
# HSA runtime's own (the timing code's hipEventElapsedTime, say) or the harness's. Nor may it define a function with C
# linkage other than the launcher and a kernel's host stub: a C++ name is mangled and cannot take a C function's place.
RESERVED_SYMBOL = re.compile(r"main|hip\w*|hsa_\w*|__hip[A-Za-z]\w*|aorlta_\w*")
SYMBOL_LINE = re.compile(r"^\s*\d+:\s+[0-9a-fA-F]+\s+\S+\s+(\w+)\s+(\w+)\s+\w+\s+(\S+)\s+(\S+)")
TIMING_STATUS = {"candidate_wrong_output": "timing_wrong_output", "reference_wrong_output": "reference_wrong_output",
                 "hip_error": "timing_hip_error", "too_slow": "too_slow_to_measure"}


def _launch_params(task: dict) -> str:
    return ", ".join([f"float* {a['name']}" for a in task["arrays"]]
                     + [f"{ctype} {name}" for ctype, name, _ in task["scalars"]])


def _launch_args(task: dict) -> str:
    return ", ".join([f"d_{a['name']}" for a in task["arrays"]] + [name for _, name, _ in task["scalars"]])


def launcher_source(task: dict, kernel_source: str) -> str:
    """The candidate's translation unit for the timing check: its source as it is, then one host function with C
    linkage that launches its kernel with the task's grid and block. It is compiled on its own, so nothing the source
    defines, a macro included, reaches the timing code."""
    return "\n".join([kernel_source.rstrip(), "", "// ---- AORLTA timing launcher ----",
                      f'extern "C" void {LAUNCHER}({_launch_params(task)}) {{',
                      f"  {task['kernel']}<<<{task['grid']}, {task['block']}>>>({', '.join(task['args'])});",
                      "}", ""])


def timing_source(task: dict, reference_source: str, scalars: dict | None = None) -> str:
    """The timing harness's own translation unit: the task's reference kernel (in namespace `aorlta_ref`, its
    `#include` lines hoisted out), and a host program `timing <warmup> <repeats> <launches> <budget_ms>` that prints
    one `AORLTA_TIMING {json}` line. It first runs each kernel once on fresh inputs and compares the outputs with the
    host reference (`reference_wrong_output`, `candidate_wrong_output`); times one batch of each to project the run's
    GPU time, and stops if that exceeds the budget (`too_slow`); then takes `warmup + repeats` rounds, each a fresh
    checked run of the candidate and then one timed batch of `launches` back-to-back launches of each kernel, in an
    order that alternates by round, on inputs reset before each batch. `ok` carries every batch's time per launch in
    microseconds, warm-up rounds included; `hip_error`, any HIP call that failed. `scalars` gives some of the task's
    scalars other values (C++ expressions) for the timing alone, so that the kernels run long enough to time."""
    scalars = scalars or {}
    ref_lines = reference_source.rstrip().splitlines()
    includes = [x for x in ref_lines if x.strip().startswith("#include")]
    body = [x for x in ref_lines if not x.strip().startswith("#include")]
    params, args = _launch_params(task), _launch_args(task)
    lines = [*includes, "#include <cmath>", "#include <cstdint>", "#include <cstdio>", "#include <cstdlib>",
             "#include <cstring>", "#include <vector>", "",
             "namespace aorlta_ref {", *body, "}  // namespace aorlta_ref",
             "", f'extern "C" void {LAUNCHER}({params});',
             f"static void aorlta_launch_reference({params}) {{",
             f"  aorlta_ref::{task['kernel']}<<<{task['grid']}, {task['block']}>>>({', '.join(task['args'])});", "}",
             "",
             "static void fill(std::vector<float>& v, uint32_t seed, float lo, float hi) {",
             "  for (auto& x : v) { seed = seed * 1664525u + 1013904223u; "
             "x = lo + (hi - lo) * float(seed >> 8) / 16777216.0f; }",
             "}",
             "static int say(const char* status) {",
             f"  printf(\"{TIMING_TAG}{{\\\"status\\\": \\\"%s\\\"}}\\n\", status);",
             "  return 0;",
             "}",
             "static void series(const char* key, const std::vector<float>& v) {",
             "  printf(\", \\\"%s\\\": [\", key);",
             "  for (size_t i = 0; i < v.size(); ++i) printf(\"%s%.6g\", i ? \", \" : \"\", v[i]);",
             "  printf(\"]\");",
             "}", "",
             "int main(int argc, char** argv) {",
             "  if (argc != 5) return 2;",
             "  const int warmup = atoi(argv[1]), repeats = atoi(argv[2]), launches = atoi(argv[3]);",
             "  const double budget_ms = atof(argv[4]);"]
    for ctype, name, value in task["scalars"]:
        lines.append(f"  const {ctype} {name} = {scalars.get(name, value)};")
    arrays = task["arrays"]
    for i, a in enumerate(arrays):
        n = a["size"]
        lines.append(f"  std::vector<float> h_{a['name']}(({n}) + {GUARD}, {GUARD_VALUE});")
        if a["role"] == "out":
            lines.append(f"  for (long i = 0; i < ({n}); ++i) h_{a['name']}[i] = {SENTINEL};")
        else:
            lines.append(f"  {{ std::vector<float> v({n}); fill(v, {1000 + 17 * i}u, {_f(a.get('lo', -1.0))}, "
                         f"{_f(a.get('hi', 1.0))}); for (long i = 0; i < ({n}); ++i) h_{a['name']}[i] = v[i]; }}")
        lines.append(f"  std::vector<float> r_{a['name']} = h_{a['name']};")
    lines.append("  {")
    for a in arrays:
        lines.append(f"    float* {a['name']} = r_{a['name']}.data();")
    lines.append("    " + task["reference"])
    lines.append("  }")
    for a in arrays:
        x = a["name"]
        lines += [f"  float* d_{x} = nullptr; float* p_{x} = nullptr;",
                  f"  const size_t bytes_{x} = h_{x}.size() * sizeof(float);",
                  f"  if (hipMalloc(&d_{x}, bytes_{x}) != hipSuccess || hipMalloc(&p_{x}, bytes_{x}) != hipSuccess",
                  f"      || hipMemcpy(p_{x}, h_{x}.data(), bytes_{x}, hipMemcpyHostToDevice) != hipSuccess)",
                  "    return say(\"hip_error\");"]
    resets = " && ".join(f"hipMemcpy(d_{x}, p_{x}, bytes_{x}, hipMemcpyDeviceToDevice) == hipSuccess"
                         for x in (a["name"] for a in arrays))
    lines += [f"  auto reset = [&]() -> bool {{ return {resets}; }};",
              "  std::vector<float> got;",
              "  auto verify = [&]() -> const char* {"]
    for a in arrays:
        x, n = a["name"], a["size"]
        lines += [f"    got.resize(h_{x}.size());",
                  f"    if (hipMemcpy(got.data(), d_{x}, bytes_{x}, hipMemcpyDeviceToHost) != hipSuccess)",
                  "      return \"hip_error\";",
                  f"    for (long i = ({n}); i < (long)got.size(); ++i) if (got[i] != {GUARD_VALUE}) return \"wrong\";"]
        if a["role"] != "in":
            lines += [f"    for (long i = 0; i < ({n}); ++i) {{",
                      f"      float g = got[i], w = r_{x}[i];",
                      f"      if (!((std::isnan(g) && std::isnan(w)) || std::fabs(g - w) <= {_f(task['atol'])} + "
                      f"{_f(task['rtol'])} * std::fabs(w))) return \"wrong\";",
                      "    }"]
    lines += ["    return nullptr;", "  };",
              f"  auto launch = [&](int who) {{ if (who) aorlta_launch_reference({args}); "
              f"else {LAUNCHER}({args}); }};",
              "  auto fresh = [&](int who) -> const char* {",
              "    if (!reset()) return \"hip_error\";",
              "    launch(who);",
              "    if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) return \"hip_error\";",
              "    return verify();",
              "  };",
              "  hipEvent_t t0, t1;",
              "  if (hipEventCreate(&t0) != hipSuccess || hipEventCreate(&t1) != hipSuccess)",
              "    return say(\"hip_error\");",
              "  auto batch = [&](int who) -> float {",
              "    if (!reset() || hipEventRecord(t0) != hipSuccess) return -1.0f;",
              "    for (int l = 0; l < launches; ++l) launch(who);",
              "    if (hipEventRecord(t1) != hipSuccess || hipEventSynchronize(t1) != hipSuccess || "
              "hipGetLastError() != hipSuccess) return -1.0f;",
              "    float ms = 0.0f;",
              "    if (hipEventElapsedTime(&ms, t0, t1) != hipSuccess) return -1.0f;",
              "    return 1000.0f * ms / launches;",
              "  };",
              "  for (int who = 1; who >= 0; --who) {",
              "    const char* bad = fresh(who);",
              "    if (bad) return say(!strcmp(bad, \"hip_error\") ? \"hip_error\" : "
              "(who ? \"reference_wrong_output\" : \"candidate_wrong_output\"));",
              "  }",
              "  const float pc = batch(0), pr = batch(1);",
              "  if (pc < 0.0f || pr < 0.0f) return say(\"hip_error\");",
              "  if ((pc + pr) / 1000.0 * launches * (warmup + repeats) > budget_ms) {",
              f"    printf(\"{TIMING_TAG}{{\\\"status\\\": \\\"too_slow\\\", \\\"candidate_us\\\": %.6g, "
              "\\\"reference_us\\\": %.6g}\\n\", pc, pr);",
              "    return 0;",
              "  }",
              "  std::vector<float> tc, tr;",
              "  for (int r = 0; r < warmup + repeats; ++r) {",
              "    const char* bad = fresh(0);",
              "    if (bad) return say(!strcmp(bad, \"hip_error\") ? \"hip_error\" : \"candidate_wrong_output\");",
              "    for (int k = 0; k < 2; ++k) {",
              "      const int who = (r + k) % 2;",
              "      const float us = batch(who);",
              "      if (us < 0.0f) return say(\"hip_error\");",
              "      (who ? tr : tc).push_back(us);",
              "    }",
              "  }",
              f"  printf(\"{TIMING_TAG}{{\\\"status\\\": \\\"ok\\\", \\\"warmup\\\": %d, \\\"launches\\\": %d\", "
              "warmup, launches);",
              "  series(\"candidate_us\", tc);",
              "  series(\"reference_us\", tr);",
              "  printf(\"}\\n\");",
              "  return 0;",
              "}", ""]
    return "\n".join(lines)


def defined_symbols(readelf_syms: str) -> list[tuple[str, str]]:
    """(name, type) of every global or weak symbol an object defines, from `llvm-readelf --syms --wide`."""
    out = []
    for line in readelf_syms.splitlines():
        m = SYMBOL_LINE.match(line)
        if m and m.group(2) in ("GLOBAL", "WEAK", "UNIQUE") and m.group(3) != "UND":
            out.append((m.group(4).split("@", 1)[0], m.group(1)))
    return out


def reserved_symbols(symbols: list[tuple[str, str]]) -> list[str]:
    """The symbols among `symbols` the candidate's object may not define (RESERVED_SYMBOL)."""
    bad = []
    for name, kind in symbols:
        if name == LAUNCHER:
            continue
        c_function = kind in ("FUNC", "IFUNC") and not name.startswith(("_Z", "__device_stub__"))
        if RESERVED_SYMBOL.fullmatch(name) or c_function:
            bad.append(name)
    return sorted(bad)


def read_timing(stdout: str) -> dict | None:
    """The timing harness's result line, or None."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(TIMING_TAG):
            try:
                out = json.loads(line[len(TIMING_TAG):])
            except ValueError:
                return None
            return out if isinstance(out, dict) and isinstance(out.get("status"), str) else None
    return None


def timing_verdict(run, policy) -> tuple[str, str | None, list[dict], dict | None]:
    """A timing leg's (state, reason, findings) from the harness's run (a ProcResult), and the judged measurement
    (`aorlta.reward.timing.Speedup`) when there was one. The candidate's correctness was settled by `ref.native`
    before this leg runs, so nothing that stops the measurement is blamed on it: a crash, a timeout, a wrong output
    under the timing harness, too slow to measure in the budget and noise are all unsettled, never clean."""
    from aorlta.reward import timing

    if run.launch_error or run.killed:
        return "unsettled", "run_not_started", [], None
    if run.timed_out:
        return "unsettled", "timing_timeout", [], None
    result = read_timing(run.stdout)
    if result is None:
        return "unsettled", "timing_crash" if run.returncode not in (0, None) else "no_result", [], None
    status = result["status"]
    if status != "ok":
        return "unsettled", TIMING_STATUS.get(status, "no_result"), [], None
    cand, ref = result.get("candidate_us"), result.get("reference_us")
    if run.returncode != 0 or not isinstance(cand, list) or not isinstance(ref, list):
        return "unsettled", "no_result", [], None
    sp = timing.judge(cand, ref, policy)
    state, reason, findings = timing.leg_outcome(sp)
    return state, reason, findings, sp


class HipkTime:
    """`hipk.time`, leg `ref.speedup`: the candidate timed against its task's reference kernel (`reference/<id>.hip`
    in the pool) on this worker's GPU (settings `gpus`, as `hipk.run`), and judged by `aorlta.reward.timing` with the
    leg's settings (`ratio`, `warmup`, `repeats`, `max_rel_mad`, `min_noise_floor`; also `launches` per timed batch,
    `gpu_budget_s` of projected GPU time, `run_timeout_s`, `target`, and `scalars`: values for the timing of any task
    scalar so named, `{n: "1 << 24"}` say, since at the tasks' own sizes a launch is mostly its overhead). The
    candidate is compiled alone (`launcher_source`); an object that defines a reserved symbol (`reserved_symbols`, read
    with llvm-readelf, settings `readelf`) is not timed. Clean only when the speedup clears `ratio` past the noise
    floor; `fail` `not_faster` when it falls short of it past the floor; everything else is unsettled. Run it after
    `ref.native` (the profile's `requires`)."""

    version = "1"
    legs = ("ref.speedup",)
    resource = "own_gpu"

    def __init__(self, ctx):
        from aorlta.reward.adapters.base import ProcessRunner

        self.name, self.ctx = ctx.name, ctx
        s = ctx.settings
        self.hipcc = s.get("hipcc") or "/opt/rocm/bin/hipcc"
        self.readelf = s.get("readelf") or str(Path(self.hipcc).parent.parent / "llvm" / "bin" / "llvm-readelf")
        gpus = list(s.get("gpus") or [0])
        seq = int(ctx.worker_id.rsplit("/", 1)[-1]) if ctx.worker_id.rsplit("/", 1)[-1].isdigit() else 1
        self.gpu = int(gpus[(seq - 1) % len(gpus)])
        self.tasks = {}
        for pool in ctx.pools:
            if (Path(pool) / "tasks.json").is_file():
                for task in load_tasks(pool):
                    self.tasks[task_sha256(task)] = (task, Path(pool) / "reference" / f"{task['id']}.hip")
        self.runner = ProcessRunner()

    def run(self, job):
        from aorlta.reward import timing
        from aorlta.reward.adapters.base import make_result, run_dir, unsettled

        t0 = self.ctx.clock()

        def out(state, reason=None, findings=()):
            return make_result(job, state, reason, findings=findings, t_start=t0, t_end=self.ctx.clock())

        try:
            entry = self.tasks.get(job.task_sha256)
            source = job.inputs.get("source")
            if job.leg != "ref.speedup" or entry is None or not source or not job.output_dir:
                return unsettled(job, "adapter_error", t_start=t0, t_end=self.ctx.clock())
            task, ref_path = entry
            settings = job.leg_settings
            policy = timing.TimingPolicy.from_settings(settings)
            launches = int(settings.get("launches", 20))
            gpu_budget_s = float(settings.get("gpu_budget_s", 10.0))
            scalars = settings.get("scalars") or {}
            if launches < 1 or not gpu_budget_s > 0 or not (
                    isinstance(scalars, dict) and all(isinstance(v, (str, int)) for v in scalars.values())):
                return unsettled(job, "adapter_error", t_start=t0, t_end=self.ctx.clock())
            d = run_dir(job)
            (d / "cand.hip").write_text(launcher_source(task, Path(source).read_text(encoding="utf-8")),
                                        encoding="utf-8")
            (d / "timing.hip").write_text(timing_source(task, ref_path.read_text(encoding="utf-8"),
                                                        {k: str(v) for k, v in scalars.items()}), encoding="utf-8")
            arch = f"--offload-arch={job.leg_settings.get('target', 'gfx950')}"
            deadline = time.monotonic() + max(1.0, self.ctx.timeout_s)

            def step(argv, limit=None, **kw):
                left = deadline - time.monotonic() if limit is None else min(limit, deadline - time.monotonic())
                return None if left <= 0 else self.runner.run(argv, timeout_s=left, cwd=str(d), **kw)

            build = step([self.hipcc, arch, "-O2", "-c", str(d / "cand.hip"), "-o", str(d / "cand.o")])
            if build is None or build.returncode != 0 or build.timed_out or build.killed or build.launch_error:
                if build is not None:
                    (d / "build.log").write_text(build.stderr[-20000:], encoding="utf-8")
                return out("unsettled", "harness_build_failed")
            syms = step([self.readelf, "--syms", "--wide", str(d / "cand.o")])
            if syms is None or syms.returncode != 0 or syms.timed_out or syms.killed or syms.launch_error:
                return out("unsettled", "symbol_check_failed")
            symbols = defined_symbols(syms.stdout)
            if LAUNCHER not in {name for name, _ in symbols}:
                return out("unsettled", "symbol_check_failed")
            if reserved_symbols(symbols):
                return out("unsettled", "defines_reserved_symbol")
            link = step([self.hipcc, arch, "-O2", str(d / "cand.o"), str(d / "timing.hip"), "-o", str(d / "timing")])
            if link is None or link.returncode != 0 or link.timed_out or link.killed or link.launch_error:
                if link is not None:
                    (d / "build.log").write_text(link.stderr[-20000:], encoding="utf-8")
                return out("unsettled", "harness_build_failed")
            if deadline - time.monotonic() <= 0:
                return out("unsettled", "no_time_left")
            env = dict(os.environ, HIP_VISIBLE_DEVICES=str(self.gpu))
            run = step([str(d / "timing"), str(policy.warmup), str(policy.repeats), str(launches),
                        repr(1000.0 * gpu_budget_s)], limit=float(settings.get("run_timeout_s", 60.0)), env=env)
            if run is None:
                return out("unsettled", "no_time_left")
            (d / "run.log").write_text((run.stdout + "\n" + run.stderr)[-20000:], encoding="utf-8")
            state, reason, findings, sp = timing_verdict(run, policy)
            if sp is not None:
                (d / "timing.json").write_text(json.dumps(sp.__dict__, indent=1), encoding="utf-8")
            return out(state, reason, findings)
        except Exception:  # noqa: BLE001 - an adapter never raises
            return unsettled(job, "adapter_error", t_start=t0, t_end=self.ctx.clock())

    def controls(self):
        return []

    def health(self):
        return {"name": self.name, "responsive": True, "gpu": self.gpu, "tasks": len(self.tasks)}

    def close(self):
        self.runner.close()


def hipk_speed(profile=None):
    """`hipk.speed`: hipk.v1, with a clean reward graded by `ref.speedup` (`aorlta.reward.rewardfn.speedup`): 0.75 to
    1.0 for a kernel faster than the reference by the leg's `ratio`, below 0.5 for a correct one that is not."""
    from aorlta.reward.rewardfn.speedup import WithSpeedup

    return WithSpeedup(HipkV1(profile), name="hipk.speed")


# --- registration ------------------------------------------------------------------------------


def register() -> None:
    from aorlta import plugins

    plugins.register_task(plugins.Task("hipk", records))
    plugins.register_parse_adapter("hipk.parse", HipkParse)
    plugins.register_feedback_template("hipk.feedback", HipkFeedback)
    plugins.register_profile("hipk-gpu", HERE / "hipk-gpu.yaml")
    plugins.register_profile("hipk-speed", HERE / "hipk-speed.yaml")
    plugins.register_check_adapter("hipk.run", HipkRun)
    plugins.register_check_adapter("hipk.time", HipkTime)
    plugins.register_reward_fn("hipk.v1", HipkV1)
    plugins.register_reward_fn("hipk.speed", hipk_speed)


if __name__ != "__main__":
    register()


# --- the self-check ----------------------------------------------------------------------------


def selfcheck(argv=None) -> int:
    """Each task's harness with its reference kernel (reference/<id>.hip): built, and unless --build-only run on the
    GPU, where every task must report `ok`. Prints one JSON line per task; exit 0 when all pass."""
    ap = argparse.ArgumentParser(prog="hip_kernels.py selfcheck")
    ap.add_argument("--hipcc", default="/opt/rocm/bin/hipcc")
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pool", default=str(HERE))
    ap.add_argument("--build-only", action="store_true", help="compile the harnesses, run nothing (no GPU)")
    args = ap.parse_args(argv)
    import subprocess

    failed = 0
    work = Path(tempfile.mkdtemp(prefix="hipk-selfcheck-"))
    try:
        for task in load_tasks(args.pool):
            ref = (HERE / "reference" / f"{task['id']}.hip").read_text(encoding="utf-8")
            src, exe = work / f"{task['id']}.hip", work / task["id"]
            src.write_text(harness_source(task, ref), encoding="utf-8")
            build = subprocess.run([args.hipcc, f"--offload-arch={args.arch}", "-O2", "-o", str(exe), str(src)],
                                   capture_output=True, text=True)
            row = {"task": task["id"], "build_rc": build.returncode}
            if build.returncode == 0 and not args.build_only:
                run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60,
                                     env=dict(os.environ, HIP_VISIBLE_DEVICES=str(args.gpu)))
                row["result"] = read_result(run.stdout)
                ok = run.returncode == 0 and row["result"] is not None and row["result"]["status"] == "ok"
            else:
                ok = build.returncode == 0
                if not ok:
                    row["stderr"] = build.stderr[-2000:]
            row["pass"] = ok
            failed += not ok
            print(json.dumps(row), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


def timecheck(argv=None) -> int:
    """One task's candidate (by default its own reference kernel, so a speedup near 1, which is `not_faster`) timed by
    `hipk.time` on one GPU, with hipk-speed.yaml's leg settings; `--set key=value` overrides one (a JSON value).
    Prints the leg's state, reason and findings and the judged measurement as one JSON line; exit 0."""
    ap = argparse.ArgumentParser(prog="hip_kernels.py timecheck")
    ap.add_argument("--task", required=True)
    ap.add_argument("--candidate", help="a HIP source defining the task's kernel (default: reference/<task>.hip)")
    ap.add_argument("--hipcc", default="/opt/rocm/bin/hipcc")
    ap.add_argument("--readelf", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pool", default=str(HERE))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from aorlta import plugins
    from aorlta.config import parse_yaml
    from aorlta.schemas import LegJob

    settings = dict(parse_yaml((HERE / "hipk-speed.yaml").read_bytes())["leg_settings"]["ref.speedup"])
    for kv in args.set:
        key, value = kv.split("=", 1)
        settings[key] = json.loads(value)
    task = next(t for t in load_tasks(args.pool) if t["id"] == args.task)
    candidate = args.candidate or str(Path(args.pool) / "reference" / f"{task['id']}.hip")
    work = Path(tempfile.mkdtemp(prefix="hipk-timecheck-"))
    try:
        ctx = plugins.CheckContext(name="hipk.time", worker_id="time/1",
                                   settings={"hipcc": args.hipcc, "readelf": args.readelf, "gpus": [args.gpu]},
                                   clock=time.time, timeout_s=args.timeout, pools=(args.pool,))
        adapter = HipkTime(ctx)
        job = LegJob(job_id="timecheck", leg="ref.speedup", candidate_id="timecheck", control_id=None,
                     source_sha256="0" * 64, task_sha256=task_sha256(task), leg_settings=settings,
                     inputs={"source": candidate}, output_dir=str(work))
        t0 = time.monotonic()
        res = adapter.run(job)
        measured = [json.loads(p.read_text()) for p in work.glob("*/timing.json")]
        print(json.dumps({"task": task["id"], "candidate": Path(candidate).name, "state": res.state,
                          "reason": res.reason, "findings": res.findings, "timing": measured[0] if measured else None,
                          "seconds": round(time.monotonic() - t0, 1)}), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selfcheck":
        sys.exit(selfcheck(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "timecheck":
        sys.exit(timecheck(sys.argv[2:]))
    print(__doc__)
    sys.exit(2)
