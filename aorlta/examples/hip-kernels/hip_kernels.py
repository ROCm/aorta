"""A worked example of a task of your own, as a plugin (docs/TUTORIAL.md): write a HIP kernel from its description.

Each task in `tasks.json` names a kernel, its exact signature, what it computes and how it is launched. The model
answers with a HIP source in one ```cpp block. The rollout compiles it with the stand-in compile check (hipcc -c) and
gives up to `rollout.max_repair_turns` repair turns with the compiler's errors; a reply that has no code block, or
whose code does not define the kernel, gets a repair turn that says so (this module's parse and feedback). The reward
service then runs three legs: this module's parse gate (the rollout's own parse rule) and the stand-in compile gate,
and `ref.native`, this module's check, which compiles the candidate alone into a device code object, runs its kernel
on a GPU from a host program built from the task alone, and compares every output element with a host reference
computed from the same inputs. The reward is `hipk.v1` (below).

An optional profile, `hipk-speed` (hipk-speed.yaml), adds a fourth leg after `ref.native`: `ref.speedup`, the
candidate timed against the task's reference kernel (`hipk.time`, judged by `aorlta.reward.timing`), scored by
`hipk.speed`, which grades hipk.v1's clean reward by the measured speedup (docs/REWARDS.md, "Timing").

Registered names: task `hipk`; parse `hipk.parse`; feedback `hipk.feedback`; profiles `hipk-gpu` (hipk-gpu.yaml) and
`hipk-speed` (hipk-speed.yaml); check adapters `hipk.gate`, `hipk.run` and `hipk.time`; reward functions `hipk.v1`
and `hipk.speed`.

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
import threading
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


def harness_source(task: dict) -> str:
    """The task's host program, which holds none of the candidate's code: it loads the candidate's device code object
    (argv[1]), fills the inputs, runs the task's kernel from it once and compares every output element with a host
    reference. It writes its record to the file argv[2], replacing it whole each time: `{"stage": "setup"}` at the
    start, `{"stage": "launched"}` just before the kernel runs, then one with a `status`: `ok`, `mismatch` (an output
    element differs from the host reference beyond the tolerance), `out_of_bounds` (a write past the end of an
    array), `no_kernel` (the code object has no kernel of the task's exact signature) or `hip_error` (with the
    `stage` it failed in)."""
    lines = ["#include <hip/hip_runtime.h>", "#include <cmath>", "#include <cstdint>", "#include <cstdio>",
             "#include <string>", "#include <vector>", "",
             "// The task's kernel, defined here only to name it (__builtin_get_device_side_mangled_name): the code",
             "// that runs is the candidate's, from the code object in argv[1].",
             f"{task['signature']} {{}}", "",
             "static const char* g_path = nullptr;",
             "static void put(const char* json) {  // written aside, then renamed over the record",
             "  std::string tmp = std::string(g_path) + \".partial\";",
             "  FILE* f = std::fopen(tmp.c_str(), \"w\");",
             "  if (f == nullptr) return;",
             "  std::fputs(json, f);",
             "  if (std::fclose(f) == 0) std::rename(tmp.c_str(), g_path);",
             "}", "",
             "static void fill(std::vector<float>& v, uint32_t seed, float lo, float hi) {",
             "  for (auto& x : v) { seed = seed * 1664525u + 1013904223u; "
             "x = lo + (hi - lo) * float(seed >> 8) / 16777216.0f; }",
             "}", "",
             "#define CHECK(stage, call) do { hipError_t e_ = (call); if (e_ != hipSuccess) { char b_[512]; "
             "std::snprintf(b_, sizeof b_, \"{\\\"status\\\": \\\"hip_error\\\", \\\"stage\\\": \\\"%s\\\", "
             "\\\"at\\\": \\\"%s\\\", \\\"error\\\": \\\"%s\\\"}\", stage, #call, hipGetErrorName(e_)); put(b_); "
             "return 0; } } while (0)", "",
             "int main(int argc, char** argv) {",
             "  if (argc != 3) return 2;",
             "  g_path = argv[2];",
             "  put(\"{\\\"stage\\\": \\\"setup\\\"}\");"]
    for ctype, name, value in task["scalars"]:
        lines.append(f"  {ctype} {name} = {value};")
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
                  f"  CHECK(\"setup\", hipMalloc(&d_{a['name']}, h_{a['name']}.size() * sizeof(float)));",
                  f"  CHECK(\"setup\", hipMemcpy(d_{a['name']}, h_{a['name']}.data(), h_{a['name']}.size() * "
                  "sizeof(float), hipMemcpyHostToDevice));"]
    names = {a["name"] for a in arrays}
    args = ", ".join(f"&d_{x}" if x in names else f"&{x}" for x in task["args"])
    lines += ["  hipModule_t mod_;", "  CHECK(\"setup\", hipModuleLoad(&mod_, argv[1]));", "  hipFunction_t fn_;",
              "  {",
              # gf_, not e_: CHECK declares its own e_, and `hipError_t e_ = (e_)` reads itself (undefined)
              f"    hipError_t gf_ = hipModuleGetFunction(&fn_, mod_, __builtin_get_device_side_mangled_name("
              f"{task['kernel']}));",
              "    if (gf_ == hipErrorNotFound || gf_ == hipErrorInvalidDeviceFunction) {",
              "      char b_[256];",
              "      std::snprintf(b_, sizeof b_, \"{\\\"status\\\": \\\"no_kernel\\\", \\\"stage\\\": \\\"setup\\\", "
              "\\\"error\\\": \\\"%s\\\"}\", hipGetErrorName(gf_));",
              "      put(b_);",
              "      return 0;", "    }", "    CHECK(\"setup\", gf_);", "  }",
              f"  void* args_[] = {{{args}}};",
              f"  dim3 grid_ = {task['grid']};", f"  dim3 block_ = {task['block']};",
              "  put(\"{\\\"stage\\\": \\\"launched\\\"}\");",
              "  CHECK(\"launch\", hipModuleLaunchKernel(fn_, grid_.x, grid_.y, grid_.z, block_.x, block_.y, block_.z, 0, "
              "nullptr, args_, nullptr));",
              "  CHECK(\"run\", hipDeviceSynchronize());",
              "  long checked = 0, mismatches = 0, first = -1, oob = 0;"]
    for a in arrays:
        n = a["size"]
        lines += [f"  CHECK(\"run\", hipMemcpy(h_{a['name']}.data(), d_{a['name']}, h_{a['name']}.size() * "
                  "sizeof(float), hipMemcpyDeviceToHost));",
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
              "  char b_[512];",
              "  std::snprintf(b_, sizeof b_, \"{\\\"status\\\": \\\"%s\\\", \\\"checked\\\": %ld, "
              "\\\"mismatches\\\": %ld, \\\"first\\\": %ld, \\\"out_of_bounds\\\": %ld}\", status, checked, "
              "mismatches, first, oob);",
              "  put(b_);", "  return 0;", "}", ""]
    return "\n".join(lines)


def read_record(path: str | Path) -> dict | None:
    """The harness's record file, or None (absent, unreadable, or not a record)."""
    try:
        out = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return out if isinstance(out, dict) and isinstance(out.get("status", out.get("stage")), str) else None


def verdict(run, record) -> tuple[str, str | None, list[dict]]:
    """A leg's (state, reason, findings) from the harness's run (a ProcResult) and its record (read_record). The
    verdict comes only from the record the harness writes, never from the candidate's output, and only what is the
    candidate's is blamed on it: a code object without the task's kernel (bad_signature), a wrong output, and a crash
    or a hang once the kernel was launched. A failure before the launch (an allocation, the first copy, loading the
    code object) is the node's, and leaves the leg unsettled, never clean."""
    if run.launch_error or run.killed:
        return "unsettled", "run_not_started", []
    record = record or {}
    status, stage = record.get("status"), record.get("stage")
    if status == "no_kernel":
        return "fail", "bad_signature", []
    if status == "hip_error":
        return ("unsettled", "setup_failed", []) if stage == "setup" else ("fail", "crash", [])
    if status in ("mismatch", "out_of_bounds"):
        return "fail", "wrong_output", []
    if status == "ok":
        if run.returncode == 0 and not run.timed_out and isinstance(record.get("checked"), int) \
                and record["checked"] > 0:
            return "clean", None, []
        return "unsettled", "no_result", []
    if stage == "launched":
        if run.timed_out:
            return "fail", "timeout", []
        if run.returncode not in (0, None):
            return "fail", "crash", []
    if stage == "setup" or not record:
        return "unsettled", ("setup_timeout" if run.timed_out else "setup_failed"), []
    return "unsettled", "no_result", []


_LEASE_LOCK = threading.Lock()
_LEASED: dict[int, int] = {}  # GPU -> live hipk.run adapters on it, in this process


def lease_gpu(gpus: list[int], preferred: int) -> int:
    """A GPU of `gpus` for a new worker: `preferred` if no live worker has it, else the first free one, else the
    least shared. The service retires and replaces workers under new ids, so an id says nothing about which GPUs
    the live workers hold."""
    with _LEASE_LOCK:
        free = [g for g in gpus if not _LEASED.get(g)]
        gpu = preferred if preferred in free else (free[0] if free else min(gpus, key=lambda g: _LEASED.get(g, 0)))
        _LEASED[gpu] = _LEASED.get(gpu, 0) + 1
        return gpu


def release_gpu(gpu: int) -> None:
    with _LEASE_LOCK:
        n = _LEASED.get(gpu, 0) - 1
        if n > 0:
            _LEASED[gpu] = n
        else:
            _LEASED.pop(gpu, None)


class HipkRun:
    """`hipk.run`, leg `ref.native`: the candidate compiled alone into a device code object (hipcc --genco, settings
    `hipcc`, leg setting `target`), so none of its host code is built or run; the task's harness, built from the task
    alone, loads it and runs its kernel on this worker's GPU (settings `gpus`, leased to live workers: see
    `lease_gpu`; by HIP_VISIBLE_DEVICES) for at most `run_timeout_s`, and compares its outputs with the host reference.
    Clean only when every output element matches and nothing was written past an array's end. `version` names the
    compiler hipcc runs and this file, so a leg cached under another compiler or harness is not reused."""

    legs = ("ref.native",)
    resource = "own_gpu"

    def __init__(self, ctx):
        from aorlta.reward.adapters.base import ProcessRunner
        from aorlta.reward.adapters.toolchain import toolchain_identity

        self.name, self.ctx = ctx.name, ctx
        s = ctx.settings
        self.hipcc = s.get("hipcc") or "/opt/rocm/bin/hipcc"
        self.version = f"2+{toolchain_identity(self.hipcc)}+{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}"
        gpus = [int(g) for g in (s.get("gpus") or [0])]
        tail = ctx.worker_id.rsplit("/", 1)[-1]
        seq = int(tail) if tail.isdigit() else 1
        self.gpu = lease_gpu(gpus, gpus[(seq - 1) % len(gpus)])
        self._leased = True
        self.tasks = {}
        for pool in ctx.pools:
            if (Path(pool) / "tasks.json").is_file():
                for task in load_tasks(pool):
                    self.tasks[task_sha256(task)] = task
        self.runner = ProcessRunner()
        self._harness_dir: Path | None = None
        self._harnesses: dict[tuple[str, str], Path] = {}

    def _harness(self, task: dict, target: str, timeout_s: float, evidence: Path) -> Path | None:
        """The task's harness program for `target`, built once per worker (its source is also written to
        `evidence`); None if it does not build."""
        src = harness_source(task)
        (evidence / "harness.hip").write_text(src, encoding="utf-8")
        key = (task_sha256(task), target)
        if key in self._harnesses:
            return self._harnesses[key]
        if self._harness_dir is None:
            self._harness_dir = Path(tempfile.mkdtemp(prefix="hipk-harness-"))
        d = self._harness_dir / f"{key[0][:16]}-{target}"
        d.mkdir(exist_ok=True)
        (d / "harness.hip").write_text(src, encoding="utf-8")
        build = self.runner.run([self.hipcc, f"--offload-arch={target}", "-O2", "-o", str(d / "harness"),
                                 str(d / "harness.hip")], timeout_s=timeout_s, cwd=str(d))
        (evidence / "harness-build.log").write_text(build.stderr[-20000:], encoding="utf-8")
        if build.returncode != 0 or build.timed_out or build.killed or build.launch_error:
            return None
        self._harnesses[key] = d / "harness"
        return d / "harness"

    def run(self, job):
        from aorlta.reward.adapters.base import make_result, run_dir, unsettled

        t0 = self.ctx.clock()
        try:
            task = self.tasks.get(job.task_sha256)
            source = job.inputs.get("source")
            if job.leg != "ref.native" or task is None or not source or not job.output_dir:
                return unsettled(job, "adapter_error", t_start=t0, t_end=self.ctx.clock())
            d = run_dir(job)
            target = job.leg_settings.get("target", "gfx950")
            budget = max(1.0, self.ctx.timeout_s)
            t_build = time.monotonic()
            co = self.runner.run([self.hipcc, "--genco", f"--offload-arch={target}", "-O2", "-o",
                                  str(d / "candidate.co"), str(source)], timeout_s=budget, cwd=str(d))
            (d / "build.log").write_text(co.stderr[-20000:], encoding="utf-8")
            if co.returncode != 0 or co.timed_out or co.killed or co.launch_error:  # it compiled in gate.compile
                return unsettled(job, "candidate_build_failed", t_start=t0, t_end=self.ctx.clock())
            harness = self._harness(task, target, max(1.0, budget - (time.monotonic() - t_build)), d)
            if harness is None:
                return unsettled(job, "harness_build_failed", t_start=t0, t_end=self.ctx.clock())
            left = budget - (time.monotonic() - t_build)
            limit = min(float(job.leg_settings.get("run_timeout_s", 20.0)), left)
            if limit <= 0:
                return unsettled(job, "no_time_left", t_start=t0, t_end=self.ctx.clock())
            env = dict(os.environ, HIP_VISIBLE_DEVICES=str(self.gpu))
            record = d / "record.json"
            run = self.runner.run([str(harness), str(d / "candidate.co"), str(record)], timeout_s=limit, env=env,
                                  cwd=str(d))
            (d / "run.log").write_text((run.stdout + "\n" + run.stderr)[-20000:], encoding="utf-8")
            state, reason, findings = verdict(run, read_record(record))
            return make_result(job, state, reason, findings=findings, t_start=t0, t_end=self.ctx.clock())
        except Exception:  # noqa: BLE001 - an adapter never raises
            return unsettled(job, "adapter_error", t_start=t0, t_end=self.ctx.clock())

    def controls(self):
        return []

    def health(self):
        return {"name": self.name, "responsive": True, "gpu": self.gpu, "tasks": len(self.tasks)}

    def close(self):
        self.runner.close()
        if self._leased:
            self._leased = False
            release_gpu(self.gpu)
        if self._harness_dir is not None:
            shutil.rmtree(self._harness_dir, ignore_errors=True)


class HipkGate:
    """`hipk.gate`, leg `gate.parse`: the reward's parse gate with the rollout's own rule (`hipk.parse`, which is
    the stand-in `parse_code`): the last ```cpp block, and none when a later block never closes. The stand-in gate
    takes the last complete block, so a final reply whose last block was cut off would be scored on an earlier one
    the rollout had rejected. Writes `source.hip` as the stand-in gate does."""

    name = "hipk.gate"
    version = "1"
    legs = ("gate.parse",)
    resource = "local_cpu"

    def __init__(self, ctx=None):
        self._clock = ctx.clock if ctx is not None else time.time

    def run(self, job):
        from aorlta.atomic import atomic_write_bytes
        from aorlta.reward.adapters.base import make_result, unsettled
        from aorlta.rollout.compile_check import parse_code

        t0 = self._clock()
        try:
            response = job.inputs.get("response") or (str(Path(job.package) / "response.txt") if job.package else None)
            if job.leg != "gate.parse" or not response or not job.output_dir:
                return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())
            source = parse_code(Path(response).read_text(encoding="utf-8"), "off")
            if source is None:
                return make_result(job, "fail", "parse", t_start=t0, t_end=self._clock())
            out = Path(job.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(out / "source.hip", source.encode("utf-8"))
            return make_result(job, "clean", t_start=t0, t_end=self._clock())
        except Exception:  # noqa: BLE001 - an adapter never raises
            return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())

    def controls(self):
        return []

    def health(self):
        return {"name": self.name, "responsive": True}


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

TIMING_STATUS = {"candidate_wrong_output": "timing_wrong_output", "reference_wrong_output": "reference_wrong_output",
                 "hip_error": "timing_hip_error", "too_slow": "too_slow_to_measure",
                 "candidate_no_kernel": "timing_no_kernel", "reference_no_kernel": "reference_no_kernel"}


def timing_source(task: dict, scalars: dict | None = None) -> str:
    """The timing harness, built from the task alone, like `harness_source`: `timing <candidate code object>
    <reference code object> <record> <warmup> <repeats> <launches> <budget_ms>`. It loads both code objects and finds
    the task's kernel in each by its exact mangled signature (`candidate_no_kernel`, `reference_no_kernel`); runs each
    once on fresh inputs and compares the outputs with the host reference (`reference_wrong_output`,
    `candidate_wrong_output`); times one batch of each to project the run's GPU time, and stops if that exceeds the
    budget (`too_slow`); then takes `warmup + repeats` rounds, each a fresh checked run of the candidate and then one
    timed batch of `launches` back-to-back launches of each kernel, in an order that alternates by round, on inputs
    reset before each batch. Its record (the file, written whole, never stdout): `ok` with every batch's time per
    launch in microseconds, warm-up rounds included; or `hip_error` and the call that failed. `scalars` gives some of
    the task's scalars other values (C++ expressions) for the timing alone, so that the kernels run long enough to
    time. No code of the candidate's runs on the host."""
    scalars = scalars or {}
    lines = ["#include <hip/hip_runtime.h>", "#include <cmath>", "#include <cstdint>", "#include <cstdio>",
             "#include <cstdlib>", "#include <cstring>", "#include <string>", "#include <vector>", "",
             "// The task's kernel, defined here only to name it: what runs is loaded from the two code objects.",
             f"{task['signature']} {{}}", "",
             "static const char* g_path = nullptr;",
             "static void put(const std::string& json) {  // written aside, then renamed over the record",
             '  std::string tmp = std::string(g_path) + ".partial";',
             '  FILE* f = std::fopen(tmp.c_str(), "w");',
             "  if (f == nullptr) return;",
             "  std::fputs(json.c_str(), f);",
             "  if (std::fclose(f) == 0) std::rename(tmp.c_str(), g_path);",
             "}",
             'static int say(const char* status) { put(std::string("{\\"status\\": \\"") + status + "\\"}"); '
             "return 0; }",
             "static std::string series(const std::vector<float>& v) {",
             '  std::string s = "[";',
             "  char b[32];",
             '  for (size_t i = 0; i < v.size(); ++i) { std::snprintf(b, sizeof b, "%s%.6g", i ? ", " : "", v[i]); '
             "s += b; }",
             '  return s + "]";',
             "}",
             "static void fill(std::vector<float>& v, uint32_t seed, float lo, float hi) {",
             "  for (auto& x : v) { seed = seed * 1664525u + 1013904223u; "
             "x = lo + (hi - lo) * float(seed >> 8) / 16777216.0f; }",
             "}", "",
             "int main(int argc, char** argv) {",
             "  if (argc != 8) return 2;",
             "  g_path = argv[3];",
             "  const int warmup = atoi(argv[4]), repeats = atoi(argv[5]), launches = atoi(argv[6]);",
             "  const double budget_ms = atof(argv[7]);"]
    for ctype, name, value in task["scalars"]:
        lines.append(f"  {ctype} {name} = {scalars.get(name, value)};")
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
                  '    return say("hip_error");']
    names = {a["name"] for a in arrays}
    args = ", ".join(f"&d_{x}" if x in names else f"&{x}" for x in task["args"])
    lines += ["  hipFunction_t fn_[2];",
              "  for (int who = 0; who < 2; ++who) {",
              "    hipModule_t mod;",
              '    if (hipModuleLoad(&mod, argv[1 + who]) != hipSuccess) return say("hip_error");',
              f"    hipError_t gf = hipModuleGetFunction(&fn_[who], mod, __builtin_get_device_side_mangled_name("
              f"{task['kernel']}));",
              "    if (gf == hipErrorNotFound || gf == hipErrorInvalidDeviceFunction)",
              '      return say(who ? "reference_no_kernel" : "candidate_no_kernel");',
              '    if (gf != hipSuccess) return say("hip_error");',
              "  }",
              f"  void* args_[] = {{{args}}};",
              f"  dim3 grid_ = {task['grid']};", f"  dim3 block_ = {task['block']};"]
    resets = " && ".join(f"hipMemcpy(d_{x}, p_{x}, bytes_{x}, hipMemcpyDeviceToDevice) == hipSuccess"
                         for x in (a["name"] for a in arrays))
    lines += [f"  auto reset = [&]() -> bool {{ return {resets}; }};",
              "  auto launch = [&](int who) -> bool {",
              "    return hipModuleLaunchKernel(fn_[who], grid_.x, grid_.y, grid_.z, block_.x, block_.y, block_.z, 0, "
              "nullptr, args_, nullptr) == hipSuccess;",
              "  };",
              "  std::vector<float> got;",
              "  auto verify = [&]() -> const char* {"]
    for a in arrays:
        x, n = a["name"], a["size"]
        lines += [f"    got.resize(h_{x}.size());",
                  f"    if (hipMemcpy(got.data(), d_{x}, bytes_{x}, hipMemcpyDeviceToHost) != hipSuccess)",
                  '      return "hip_error";',
                  f'    for (long i = ({n}); i < (long)got.size(); ++i) if (got[i] != {GUARD_VALUE}) return "wrong";']
        if a["role"] != "in":
            lines += [f"    for (long i = 0; i < ({n}); ++i) {{",
                      f"      float g = got[i], w = r_{x}[i];",
                      f"      if (!((std::isnan(g) && std::isnan(w)) || std::fabs(g - w) <= {_f(task['atol'])} + "
                      f'{_f(task["rtol"])} * std::fabs(w))) return "wrong";',
                      "    }"]
    lines += ["    return nullptr;", "  };",
              "  auto fresh = [&](int who) -> const char* {",
              '    if (!reset() || !launch(who) || hipDeviceSynchronize() != hipSuccess) return "hip_error";',
              "    return verify();",
              "  };",
              "  hipEvent_t t0, t1;",
              "  if (hipEventCreate(&t0) != hipSuccess || hipEventCreate(&t1) != hipSuccess)",
              '    return say("hip_error");',
              "  auto batch = [&](int who) -> float {",
              "    if (!reset() || hipEventRecord(t0) != hipSuccess) return -1.0f;",
              "    for (int l = 0; l < launches; ++l) if (!launch(who)) return -1.0f;",
              "    if (hipEventRecord(t1) != hipSuccess || hipEventSynchronize(t1) != hipSuccess) return -1.0f;",
              "    float ms = 0.0f;",
              "    if (hipEventElapsedTime(&ms, t0, t1) != hipSuccess) return -1.0f;",
              "    return 1000.0f * ms / launches;",
              "  };",
              "  for (int who = 1; who >= 0; --who) {",
              "    const char* bad = fresh(who);",
              '    if (bad) return say(!strcmp(bad, "hip_error") ? "hip_error" : '
              '(who ? "reference_wrong_output" : "candidate_wrong_output"));',
              "  }",
              "  const float pc = batch(0), pr = batch(1);",
              '  if (pc < 0.0f || pr < 0.0f) return say("hip_error");',
              "  if ((pc + pr) / 1000.0 * launches * (warmup + repeats) > budget_ms) {",
              "    char b[160];",
              '    std::snprintf(b, sizeof b, "{\\"status\\": \\"too_slow\\", \\"candidate_us\\": %.6g, '
              '\\"reference_us\\": %.6g}", pc, pr);',
              "    put(b);",
              "    return 0;",
              "  }",
              "  std::vector<float> tc, tr;",
              "  for (int r = 0; r < warmup + repeats; ++r) {",
              "    const char* bad = fresh(0);",
              '    if (bad) return say(!strcmp(bad, "hip_error") ? "hip_error" : "candidate_wrong_output");',
              "    for (int k = 0; k < 2; ++k) {",
              "      const int who = (r + k) % 2;",
              "      const float us = batch(who);",
              '      if (us < 0.0f) return say("hip_error");',
              "      (who ? tr : tc).push_back(us);",
              "    }",
              "  }",
              "  char head[96];",
              '  std::snprintf(head, sizeof head, "{\\"status\\": \\"ok\\", \\"warmup\\": %d, \\"launches\\": %d", '
              "warmup, launches);",
              '  put(std::string(head) + ", \\"candidate_us\\": " + series(tc) + ", \\"reference_us\\": " + '
              'series(tr) + "}");',
              "  return 0;",
              "}", ""]
    return "\n".join(lines)


def timing_verdict(run, record, policy) -> tuple[str, str | None, list[dict], dict | None]:
    """A timing leg's (state, reason, findings) from the harness's run (a ProcResult) and its record (read_record),
    and the judged measurement (`aorlta.reward.timing.Speedup`) when there was one. The candidate's correctness was
    settled by `ref.native` before this leg runs, so nothing that stops the measurement is blamed on it: a crash, a
    timeout, a wrong output under the timing harness, too slow to measure in the budget and noise are all unsettled,
    never clean."""
    from aorlta.reward import timing

    if run.launch_error or run.killed:
        return "unsettled", "run_not_started", [], None
    if run.timed_out:
        return "unsettled", "timing_timeout", [], None
    status = (record or {}).get("status")
    if status is None:
        return "unsettled", "timing_crash" if run.returncode not in (0, None) else "no_result", [], None
    if status != "ok":
        return "unsettled", TIMING_STATUS.get(status, "no_result"), [], None
    cand, ref = record.get("candidate_us"), record.get("reference_us")
    if run.returncode != 0 or not isinstance(cand, list) or not isinstance(ref, list):
        return "unsettled", "no_result", [], None
    sp = timing.judge(cand, ref, policy)
    state, reason, findings = timing.leg_outcome(sp)
    return state, reason, findings, sp


class HipkTime:
    """`hipk.time`, leg `ref.speedup`: the candidate timed against its task's reference kernel (`reference/<id>.hip`
    in the pool) on a GPU of settings `gpus` (leased like `hipk.run`'s), and judged by `aorlta.reward.timing` with the
    leg's settings (`ratio`, `warmup`, `repeats`, `max_rel_mad`, `min_noise_floor`; also `launches` per timed batch,
    `gpu_budget_s` of projected GPU time, `run_timeout_s`, `target`, and `scalars`: values for the timing of any task
    scalar so named, `{n: "1 << 24"}` say, since at the tasks' own sizes a launch is mostly its overhead). Both kernels
    are compiled alone into device code objects and run by `timing_source`'s harness, so none of the candidate's host
    code runs and its verdict comes only from the harness's record file. Clean only when the speedup clears `ratio`
    past the noise floor; `fail` `not_faster` when it falls short of it past the floor; everything else is unsettled.
    Run it after `ref.native` (the profile's `requires`). `version` names the compiler and this file."""

    legs = ("ref.speedup",)
    resource = "own_gpu"

    def __init__(self, ctx):
        from aorlta.reward.adapters.base import ProcessRunner
        from aorlta.reward.adapters.toolchain import toolchain_identity

        self.name, self.ctx = ctx.name, ctx
        s = ctx.settings
        self.hipcc = s.get("hipcc") or "/opt/rocm/bin/hipcc"
        self.version = f"2+{toolchain_identity(self.hipcc)}+{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}"
        gpus = [int(g) for g in (s.get("gpus") or [0])]
        tail = ctx.worker_id.rsplit("/", 1)[-1]
        seq = int(tail) if tail.isdigit() else 1
        self.gpu = lease_gpu(gpus, gpus[(seq - 1) % len(gpus)])
        self._leased = True
        self.tasks = {}
        for pool in ctx.pools:
            if (Path(pool) / "tasks.json").is_file():
                for task in load_tasks(pool):
                    self.tasks[task_sha256(task)] = (task, Path(pool) / "reference" / f"{task['id']}.hip")
        self.runner = ProcessRunner()
        self._work: Path | None = None
        self._built: dict[tuple, Path] = {}

    def _build(self, key: tuple, source: str, argv_for, deadline: float, log: Path) -> Path | None:
        """A build cached by `key` for this worker: `source` written to a file, then `argv_for(src, out)` builds
        `out` from it; None if it does not build in time."""
        if key in self._built:
            return self._built[key]
        if self._work is None:
            self._work = Path(tempfile.mkdtemp(prefix="hipk-time-"))
        out = self._work / hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]
        src = out.with_suffix(".hip")
        src.write_text(source, encoding="utf-8")
        left = deadline - time.monotonic()
        if left <= 0:
            return None
        res = self.runner.run(argv_for(src, out), timeout_s=left, cwd=str(self._work))
        log.write_text(res.stderr[-20000:], encoding="utf-8")
        if res.returncode != 0 or res.timed_out or res.killed or res.launch_error:
            return None
        self._built[key] = out
        return out

    def run(self, job):
        from aorlta.reward import timing
        from aorlta.reward.adapters.base import make_result, run_dir, unsettled

        t0 = self.ctx.clock()

        def out(state, reason=None, findings=()):
            return make_result(job, state, reason, findings=findings, t_start=t0, t_end=self.ctx.clock())

        try:
            entry = self.tasks.get(job.task_sha256)
            source = job.inputs.get("source")
            if job.leg != "ref.speedup" or entry is None or not source or not job.output_dir or not entry[1].is_file():
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
            scalars = {k: str(v) for k, v in scalars.items()}
            d = run_dir(job)
            src = timing_source(task, scalars)
            (d / "timing.hip").write_text(src, encoding="utf-8")
            target = settings.get("target", "gfx950")
            arch = f"--offload-arch={target}"
            deadline = time.monotonic() + max(1.0, self.ctx.timeout_s)
            cand = self.runner.run([self.hipcc, "--genco", arch, "-O2", "-o", str(d / "candidate.co"), str(source)],
                                   timeout_s=max(0.001, deadline - time.monotonic()), cwd=str(d))
            (d / "build.log").write_text(cand.stderr[-20000:], encoding="utf-8")
            ref_text = ref_path.read_text(encoding="utf-8")
            ref_co = self._build(("reference", ref_text, target), ref_text,
                                 lambda s_, o: [self.hipcc, "--genco", arch, "-O2", "-o", str(o), str(s_)], deadline,
                                 d / "reference-build.log")
            harness = self._build(("harness", src, target), src,
                                  lambda s_, o: [self.hipcc, arch, "-O2", "-o", str(o), str(s_)], deadline,
                                  d / "harness-build.log")
            if cand.returncode != 0 or cand.timed_out or cand.killed or cand.launch_error or ref_co is None \
                    or harness is None:
                return out("unsettled", "harness_build_failed")
            left = deadline - time.monotonic()
            if left <= 0:
                return out("unsettled", "no_time_left")
            env = dict(os.environ, HIP_VISIBLE_DEVICES=str(self.gpu))
            record = d / "record.json"
            run = self.runner.run([str(harness), str(d / "candidate.co"), str(ref_co), str(record), str(policy.warmup),
                                   str(policy.repeats), str(launches), repr(1000.0 * gpu_budget_s)],
                                  timeout_s=min(float(settings.get("run_timeout_s", 60.0)), left), env=env,
                                  cwd=str(d))
            (d / "run.log").write_text((run.stdout + "\n" + run.stderr)[-20000:], encoding="utf-8")
            state, reason, findings, sp = timing_verdict(run, read_record(record), policy)
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
        if self._leased:
            self._leased = False
            release_gpu(self.gpu)
        if self._work is not None:
            shutil.rmtree(self._work, ignore_errors=True)


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
    plugins.register_check_adapter("hipk.gate", HipkGate)
    plugins.register_check_adapter("hipk.time", HipkTime)
    plugins.register_reward_fn("hipk.v1", HipkV1)
    plugins.register_reward_fn("hipk.speed", hipk_speed)


if __name__ != "__main__":
    register()


# --- the self-check ----------------------------------------------------------------------------


def selfcheck(argv=None) -> int:
    """Each task's reference kernel (reference/<id>.hip), or with --sources DIR the kernel DIR/<id>.hip where there
    is one, through the check's own steps: compiled alone into a code object, the task's harness built, and unless
    --build-only run on the GPU and judged by `verdict`. Prints one JSON line per task; exit 0 when all are clean
    (built, with --build-only)."""
    import subprocess
    from types import SimpleNamespace

    def run(argv, timeout_s, env=None):  # what `verdict` reads of a ProcResult; standalone, so no aorlta import
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, env=env)
            return SimpleNamespace(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr, timed_out=False,
                                   killed=False, launch_error=None)
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            return SimpleNamespace(returncode=-9, stdout=out, stderr="", timed_out=True, killed=False,
                                   launch_error=None)

    ap = argparse.ArgumentParser(prog="hip_kernels.py selfcheck")
    ap.add_argument("--hipcc", default="/opt/rocm/bin/hipcc")
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pool", default=str(HERE))
    ap.add_argument("--sources", default=None, help="a directory of <task id>.hip kernels to check instead")
    ap.add_argument("--build-only", action="store_true", help="compile the harnesses, run nothing (no GPU)")
    args = ap.parse_args(argv)
    failed = 0
    work = Path(tempfile.mkdtemp(prefix="hipk-selfcheck-"))
    try:
        for task in load_tasks(args.pool):
            src = Path(args.sources or HERE / "reference") / f"{task['id']}.hip"
            if not src.is_file():
                continue
            d = work / task["id"]
            d.mkdir()
            (d / "harness.hip").write_text(harness_source(task), encoding="utf-8")
            co = run([args.hipcc, "--genco", f"--offload-arch={args.arch}", "-O2", "-o", str(d / "candidate.co"),
                      str(src)], 300)
            hb = run([args.hipcc, f"--offload-arch={args.arch}", "-O2", "-o", str(d / "harness"),
                      str(d / "harness.hip")], 300)
            row = {"task": task["id"], "source": str(src), "build_rc": [co.returncode, hb.returncode]}
            built = co.returncode == 0 and hb.returncode == 0
            if built and not args.build_only:
                res = run([str(d / "harness"), str(d / "candidate.co"), str(d / "record.json")], 60,
                          env=dict(os.environ, HIP_VISIBLE_DEVICES=str(args.gpu)))
                row["record"] = read_record(d / "record.json")
                row["stdout"] = res.stdout[-500:]
                row["verdict"] = verdict(res, row["record"])[:2]
                ok = row["verdict"][0] == "clean"
            else:
                ok = built
                if not ok:
                    row["stderr"] = (co.stderr + hb.stderr)[-2000:]
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
                                   settings={"hipcc": args.hipcc, "gpus": [args.gpu]},
                                   clock=time.time, timeout_s=args.timeout, pools=(args.pool,))
        adapter = HipkTime(ctx)
        job = LegJob(job_id="timecheck", leg="ref.speedup", candidate_id="timecheck", control_id=None,
                     source_sha256="0" * 64, task_sha256=task_sha256(task), leg_settings=settings,
                     inputs={"source": candidate}, output_dir=str(work))
        t0 = time.monotonic()
        try:
            res = adapter.run(job)
        finally:
            adapter.close()
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
