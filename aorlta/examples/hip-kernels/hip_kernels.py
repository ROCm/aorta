"""A worked example of a task of your own, as a plugin (docs/TUTORIAL.md): write a HIP kernel from its description.

Each task in `tasks.json` names a kernel, its exact signature, what it computes and how it is launched. The model
answers with a HIP source in one ```cpp block. The rollout compiles it with the stand-in compile check (hipcc -c) and
gives up to `rollout.max_repair_turns` repair turns with the compiler's errors; a reply that has no code block, or
whose code does not define the kernel, gets a repair turn that says so (this module's parse and feedback). The reward
service then runs three legs: the stand-in parse gate and compile gate, and `ref.native`, this module's check, which
builds the candidate with a generated host program, runs it on a GPU and compares every output element with a host
reference computed from the same inputs. The reward is `hipk.v1` (below).

Registered names: task `hipk`; parse `hipk.parse`; feedback `hipk.feedback`; profile `hipk-gpu` (hipk-gpu.yaml);
check adapter `hipk.run`; reward function `hipk.v1`.

Run as a script, it checks the tasks themselves with the reference kernels in reference/:
    python examples/hip-kernels/hip_kernels.py selfcheck --hipcc /opt/rocm/bin/hipcc [--gpu 0] [--build-only]
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


# --- registration ------------------------------------------------------------------------------


def register() -> None:
    from aorlta import plugins

    plugins.register_task(plugins.Task("hipk", records))
    plugins.register_parse_adapter("hipk.parse", HipkParse)
    plugins.register_feedback_template("hipk.feedback", HipkFeedback)
    plugins.register_profile("hipk-gpu", HERE / "hipk-gpu.yaml")
    plugins.register_check_adapter("hipk.run", HipkRun)
    plugins.register_reward_fn("hipk.v1", HipkV1)


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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selfcheck":
        sys.exit(selfcheck(sys.argv[2:]))
    print(__doc__)
    sys.exit(2)
